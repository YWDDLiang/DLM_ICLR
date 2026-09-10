"""Generate fresh H1 rich Plans with the configured Llama checkpoint."""

from __future__ import annotations
from pathlib import Path
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from crystal_dlm.h1_llm_planner import (
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    format_planner_prompt,
    load_llama3_compatible_config,
    ensure_peft_cache_compat,
    disable_peft_bnb_autodetect,
)
from .io import fingerprint, write_rows, write_json

STYLE = "h1_rich_plan_v1"


class PlanEnd(StoppingCriteria):
    def __init__(self, tokenizer, start_length):
        self.tokenizer, self.start_length = tokenizer, start_length

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return all(re.search(r"(?i)\bend\s*:\s*plan\b", value) is not None for value in text)


def generate_plans(assets, output, *, requests=1200, seed=17, device="cuda:0", usage_role="evaluation"):
    if not assets.planner_base or not assets.planner:
        raise ValueError("Fresh Plans require assets.planner_base and assets.planner")
    tokenizer = AutoTokenizer.from_pretrained(assets.planner, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        assets.planner_base,
        config=load_llama3_compatible_config(assets.planner_base),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, assets.planner).to(device).eval()
    torch.manual_seed(seed)
    rows = []
    for index in range(requests):
        prompt = format_planner_prompt(tokenizer, prompt_style=STYLE)
        encoded = tokenizer([prompt], padding=True, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([PlanEnd(tokenizer, encoded["input_ids"].shape[1])]),
            )
        raw = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        cleaned = clean_generated_plan_text(raw, prompt_style=STYLE, truncate_after_marker=True)
        row = {
            "schema": "crystal_plan_v1",
            "source_id": f"planner:{seed}:{index}",
            "original_ordinal": index,
            "raw_plan_text": cleaned,
            "raw_model_text": raw,
            "provenance": {
                "dataset_origin": "planner_generated",
                "original_split": "generated",
                "usage_role": usage_role,
            },
        }
        try:
            parsed = canonical_plan_record_for_style(cleaned, sample_idx=index, prompt_style=STYLE)
            row.update(
                plan_state=parsed["plan_state"],
                body_prompt=parsed["prompt"],
                body_eligible=True,
                ineligible_reason=None,
            )
        except (ValueError, KeyError, TypeError) as error:
            row.update(plan_state=None, body_prompt=None, body_eligible=False, ineligible_reason=str(error))
        for field in ("body_noise_seed", "refiner_noise_seed"):
            row[field] = int(fingerprint([seed, index, field])[:16], 16) & ((1 << 63) - 1)
        rows.append(row)
        if (index + 1) % 20 == 0:
            print({"planner_completed": index + 1, "requests": requests}, flush=True)
    write_rows(output, rows)
    write_json(
        Path(output).with_suffix(".generation.json"),
        {
            "requests": requests,
            "seed": seed,
            "batch_size": 1,
            "max_new_tokens": 96,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 50,
            "valid_plans": sum(row["body_eligible"] for row in rows),
            "failed_plans_preserved": True,
        },
    )
    return Path(output)
