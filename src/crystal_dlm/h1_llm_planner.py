from __future__ import annotations
import re
from typing import Any, Dict
from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT
from crystal_dlm.r5_plan_body import (
    H1_RICH_PLAN_FORMAT,
    R5C_FORMULA_END_PLAN_FORMAT,
    format_composition_plan,
    has_plan_end_marker,
    has_plan_tail_after_end_marker,
    parse_composition_plan,
)
from crystal_dlm.r5_plan_state import build_body_prompt, validate_plan_state

H1_PLANNER_PROMPT_STYLE_CHAT = "chat_formula_end_v1"
H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL = "formula_prefill_v1"
H1_PLANNER_PROMPT_STYLE_RICH_PLAN = H1_RICH_PLAN_FORMAT
H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL = "h1_rich_plan_formula_prefill_v1"
H1_PLANNER_PROMPT_STYLES = (
    H1_PLANNER_PROMPT_STYLE_CHAT,
    H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
)
H1_PLANNER_SYSTEM_PROMPT = "You are a materials composition planner for de novo MP-20 bulk crystal generation. Generate only a composition formula plan. Do not generate lattice, coordinates, CIF, explanations, candidates, rankings, or database lookups."


def load_llama3_compatible_config(model_path: str, *, trust_remote_code: bool = True) -> Any:
    """Load Llama-3.1 config on older Transformers builds.

    The A800 environment currently pins a Transformers version whose LlamaConfig
    accepts only ``{"type", "factor"}`` rope scaling fields.  Llama-3.1 stores
    the richer ``rope_type=llama3`` dictionary.  For H1 prompts we stay far below
    the original context window, so falling back to dynamic rope scaling is a
    loader compatibility shim rather than a sampling prior.
    """
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.llama.configuration_llama import LlamaConfig

    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        if "rope_scaling" not in str(exc):
            raise
    config_dict, _ = PretrainedConfig.get_config_dict(model_path, trust_remote_code=trust_remote_code)
    rope_scaling = config_dict.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        config_dict["rope_scaling"] = {"type": "dynamic", "factor": float(rope_scaling.get("factor", 8.0))}
    return LlamaConfig(**config_dict)


def ensure_peft_cache_compat() -> None:
    """Expose cache symbols expected by newer PEFT on older Transformers."""
    import transformers

    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        Cache = object
        DynamicCache = object
    if not hasattr(transformers, "Cache"):
        transformers.Cache = Cache
    if not hasattr(transformers, "DynamicCache"):
        transformers.DynamicCache = DynamicCache
    if not hasattr(transformers, "EncoderDecoderCache"):

        class EncoderDecoderCache:
            """Compatibility placeholder for PEFT imports."""

        transformers.EncoderDecoderCache = EncoderDecoderCache


def disable_peft_bnb_autodetect() -> None:
    """Avoid optional bitsandbytes dispatch paths in older A800 environments."""
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model

        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass


def normalize_prompt_style(prompt_style: str | None = None) -> str:
    style = H1_PLANNER_PROMPT_STYLE_CHAT if prompt_style is None else str(prompt_style).strip()
    if style not in H1_PLANNER_PROMPT_STYLES:
        raise ValueError(
            f"unknown H1 planner prompt style {style!r}; expected one of {H1_PLANNER_PROMPT_STYLES}"
        )
    return style


def build_planner_user_prompt(*, sample_idx: int | None = None, prompt_style: str | None = None) -> str:
    style = normalize_prompt_style(prompt_style)
    sample_line = "" if sample_idx is None else f"\nsample_id: {int(sample_idx)}"
    if style in {H1_PLANNER_PROMPT_STYLE_RICH_PLAN, H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL}:
        return f"{CRYSLLMGEN_TEXT_PROMPT.rstrip()}\n\nReturn exactly seven lines in this format:\nformula: <flat integer-count formula with 1 to 20 atoms>\nanion: <oxide|sulfide|chalcogenide|halide|nitride|phosphide_or_phosphate|other>\ncharge: <neutral_plausible|single_element|all_metal|charge_fail|pauling_fail|oxidation_missing|validator_unavailable>\nlattice: <triclinic|monoclinic|orthorhombic|tetragonal|trigonal|hexagonal|cubic>\nspacegroup: <sg_001_002|sg_003_015|sg_016_074|sg_075_142|sg_143_167|sg_168_194|sg_195_230>\nvolume: <volpa_000_004 style volume-per-atom bin>\nend: plan\n\nRules:\n- Use valid element symbols only.\n- Use a chemically plausible MP-20-like bulk composition.\n- Do not include N, elements, counts, coordinates, lattice lengths, angles, CIF, candidates, or explanations.\n- Do not include any extra text before or after the seven lines.{sample_line}"
    prefill_hint = (
        "\nThe assistant prompt may already contain `formula:`; if so, continue with the formula value only."
        if style == H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL
        else ""
    )
    return f"{CRYSLLMGEN_TEXT_PROMPT.rstrip()}\n\nReturn exactly two lines in this format:\nformula: <flat integer-count formula with 1 to 20 atoms>\nend: plan\n\nRules:\n- Use valid element symbols only.\n- Use a chemically plausible MP-20-like bulk composition.\n- Do not include N, elements, counts, family, arity, size, lattice, or coordinates.\n- Do not include any extra text before or after the two lines.{prefill_hint}{sample_line}"


def build_planner_messages(
    *, sample_idx: int | None = None, prompt_style: str | None = None
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": H1_PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_planner_user_prompt(sample_idx=sample_idx, prompt_style=prompt_style),
        },
    ]


def format_chat_prompt(
    tokenizer: Any, *, sample_idx: int | None = None, prompt_style: str | None = None
) -> str:
    messages = build_planner_messages(sample_idx=sample_idx, prompt_style=prompt_style)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"


def format_planner_prompt(
    tokenizer: Any, *, sample_idx: int | None = None, prompt_style: str | None = None
) -> str:
    style = normalize_prompt_style(prompt_style)
    prompt = format_chat_prompt(tokenizer, sample_idx=sample_idx, prompt_style=style)
    if style == H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL:
        return prompt.rstrip() + " formula: "
    if style == H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL:
        return prompt.rstrip() + " formula:"
    return prompt


def clean_generated_plan_text(
    text: str, *, prompt_style: str | None = None, truncate_after_marker: bool = True
) -> str:
    """Normalize only the text boundary of a generated H1 formula plan.

    This helper deliberately does not repair chemistry or invent missing fields.
    It only removes prompt echo before the first ``formula:`` label, converts an
    inline ``end: plan`` marker to its own line, and optionally truncates after
    the generated marker so dialogue continuations cannot confuse strict parsing.
    """
    style = normalize_prompt_style(prompt_style)
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if style == H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL and "formula:" not in cleaned.lower():
        cleaned = "formula: " + cleaned.lstrip()
    if style == H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL and "formula:" not in cleaned.lower():
        cleaned = "formula: " + cleaned.lstrip()
    formula_idx = cleaned.lower().find("formula:")
    if formula_idx >= 0:
        cleaned = cleaned[formula_idx:]
    cleaned = re.sub("(?i)([^\\n])\\s+(end\\s*:\\s*plan\\b)", "\\1\\n\\2", cleaned, count=1)
    if truncate_after_marker:
        marker = re.search("(?im)^(\\s*end\\s*:\\s*plan\\b)", cleaned)
        if marker is None:
            marker = re.search("(?i)\\bend\\s*:\\s*plan\\b", cleaned)
        if marker is not None:
            cleaned = cleaned[: marker.end()]
    return cleaned.strip()


def canonical_plan_record_for_style(
    raw_plan_text: str, *, sample_idx: int | None = None, max_atoms: int = 20, prompt_style: str | None = None
) -> Dict[str, Any]:
    style = normalize_prompt_style(prompt_style)
    plan_style = (
        H1_RICH_PLAN_FORMAT
        if style in {H1_PLANNER_PROMPT_STYLE_RICH_PLAN, H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL}
        else R5C_FORMULA_END_PLAN_FORMAT
    )
    plan = parse_composition_plan(raw_plan_text, plan_style=plan_style, max_atoms=max_atoms)
    validation = validate_plan_state(plan)
    if not validation.valid:
        raise ValueError(f"invalid generated plan: {validation.to_dict()}")
    canonical_plan_text = format_composition_plan(plan, plan_style=plan_style)
    return {
        "sample_idx": None if sample_idx is None else int(sample_idx),
        "plan_text": canonical_plan_text,
        "parsed_plan": dict(plan),
        "plan_state": dict(plan),
        "prompt": build_body_prompt(plan).rstrip() + "\n",
        "plan_end_marker_present": has_plan_end_marker(raw_plan_text),
        "plan_tail_after_end_marker": has_plan_tail_after_end_marker(raw_plan_text),
    }
