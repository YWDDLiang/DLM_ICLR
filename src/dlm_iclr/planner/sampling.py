"""Generate fresh H1 rich Plans with the configured Llama checkpoint."""

from __future__ import annotations
from pathlib import Path
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from dlm_iclr._core.h1_llm_planner import (
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    format_planner_prompt,
    load_llama3_compatible_config,
    ensure_peft_cache_compat,
    disable_peft_bnb_autodetect,
)
from dlm_iclr.runtime.io import fingerprint, write_rows, write_json
from dlm_iclr.runtime.capacity import MAX_ATOMS

STYLE = "h1_rich_plan_v1"


def _call_in_fresh_process(function, *args, **kwargs):
    """Keep a preceding physical evaluator's global torch flags out of Planner.

    The spawned process uses the same default execution context as the first
    Planner stage. The caller's strict CHGNet settings are never disabled.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        return pool.submit(function, *args, **kwargs).result()


class PlanEnd(StoppingCriteria):
    def __init__(self, tokenizer, start_length):
        self.tokenizer, self.start_length = tokenizer, start_length

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return all(re.search(r"(?i)\bend\s*:\s*plan\b", value) is not None for value in text)


def generate_plans(
    assets,
    output,
    *,
    requests=1050,
    seed=17,
    device="cuda:0",
    usage_role="evaluation",
    batch_size=1,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    max_new_tokens=96,
    resample_over_capacity=True,
):
    if torch.are_deterministic_algorithms_enabled():
        # CHGNet single-point evaluation enables this global flag in its parent
        # process. CUDA top-p cumsum cannot run under that flag. Preserve the
        # original Planner implementation/seed and isolate the stage instead
        # of changing top-p or relaxing physical evaluation determinism.
        return _call_in_fresh_process(generate_plans, assets, output,
            requests=requests, seed=seed, device=device, usage_role=usage_role,
            batch_size=batch_size, temperature=temperature, top_p=top_p, top_k=top_k,
            max_new_tokens=max_new_tokens, resample_over_capacity=resample_over_capacity)
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
    def draw(width):
        prompt = format_planner_prompt(tokenizer, prompt_style=STYLE)
        encoded = tokenizer([prompt] * width, padding=True, add_special_tokens=False, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([PlanEnd(tokenizer, encoded["input_ids"].shape[1])]),
            )
        return tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    rows, rejected, attempted = _collect_plan_rows(
        draw, requests=requests, batch_size=batch_size, seed=seed,
        usage_role=usage_role, max_atoms=MAX_ATOMS,
        resample_over_capacity=resample_over_capacity,
    )
    write_rows(output, rows)
    rejected_path = Path(output).with_suffix(".rejected.jsonl")
    write_rows(rejected_path, rejected)
    write_json(
        Path(output).with_suffix(".generation.json"),
        {
            "requests": requests,
            "seed": seed,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "valid_plans": sum(row["body_eligible"] for row in rows),
            "resample_over_capacity": resample_over_capacity,
            "max_atoms": MAX_ATOMS,
            "attempted": attempted,
            "rejected_over_capacity": len(rejected),
            "rejected_records": str(rejected_path),
        },
    )
    return Path(output)


def _collect_plan_rows(draw, *, requests, batch_size, seed, usage_role,
                       max_atoms, resample_over_capacity):
    """Reject only over-capacity compositions; keep other failures observable."""
    from pymatgen.core import Composition

    rows, rejected, attempted = [], [], 0
    while len(rows) < requests:
        width = min(batch_size, requests - len(rows))
        for raw in draw(width):
            row = _plan_row(raw, len(rows), seed, usage_role)
            row['sampling_attempt'] = attempted
            attempted += 1
            formula = re.search(r'(?im)^\s*formula\s*:\s*([^\n]+)', row['raw_plan_text'])
            atom_count = None
            if formula:
                try:
                    atom_count = float(Composition(formula.group(1).strip()).num_atoms)
                except (ValueError, KeyError, TypeError):
                    pass
            if resample_over_capacity and atom_count is not None and atom_count > max_atoms:
                row.update(rejection_reason='atom_count_exceeds_capacity',
                           atom_count=atom_count, max_atoms=max_atoms)
                row['source_id'] = f'planner:{seed}:rejected:{attempted - 1}'
                rejected.append(row)
            else:
                rows.append(row)
        print({'planner_completed': len(rows), 'requests': requests,
               'attempted': attempted, 'rejected_over_capacity': len(rejected)}, flush=True)
    return rows, rejected, attempted


def _plan_row(raw, index, seed, usage_role):
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
    return row
