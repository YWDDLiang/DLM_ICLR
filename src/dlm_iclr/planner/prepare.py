"""H1A2 seven-line planner supervision."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
from .._core.h1_llm_planner import (
    H1_PLANNER_PROMPT_VERSION,
    teacher_formula_answer,
    build_planner_messages,
    format_planner_prompt,
)
from .._core.r5_plan_state import PLAN_STATE_VERSION


def token_len(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def format_messages_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str | None:
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True))
    if len(messages) >= 2:
        return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    raise ValueError("planner messages must contain system and user turns")


def build_record(
    *,
    split: str,
    row_idx: int,
    plan_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tokenizer: Any = None,
    sample_weight: float = 1.0,
    prompt_style: str = "chat_formula_end_v1",
    include_sample_id: bool = True,
    task: str = "direct_plan",
    messages: list[dict[str, str]] | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    sample_idx = row_idx if include_sample_id else None
    answer = teacher_formula_answer(plan_state, prompt_style=prompt_style) if answer is None else str(answer)
    if messages is None:
        messages = build_planner_messages(sample_idx=sample_idx, prompt_style=prompt_style)
        prompt_text = (
            format_planner_prompt(tokenizer, sample_idx=sample_idx, prompt_style=prompt_style)
            if tokenizer is not None
            else None
        )
    else:
        prompt_text = format_messages_prompt(tokenizer, messages)
    return {
        "task": f"h1_llm_{task}",
        "h1a3_sample_type": task,
        "representation": f"h1_llm_plan_{prompt_style}",
        "prompt_version": H1_PLANNER_PROMPT_VERSION,
        "prompt_style": prompt_style,
        "plan_state_version": PLAN_STATE_VERSION,
        "split": split,
        "row_idx": int(row_idx),
        "include_sample_id": bool(include_sample_id),
        "messages": messages,
        "prompt": prompt_text,
        "answer": answer,
        "text": None if prompt_text is None else prompt_text + answer,
        "plan_text": answer,
        "plan_state": dict(plan_state),
        "metadata": dict(metadata),
        "num_atoms": int(plan_state["N"]),
        "num_elements": len(plan_state.get("elements") or []),
        "sample_weight": float(sample_weight),
        "prompt_model_length": token_len(tokenizer, prompt_text) if prompt_text is not None else None,
        "answer_model_length": token_len(tokenizer, answer),
    }


def build_records_for_plan(
    *, split, row_idx, plan_state, metadata, tokenizer, prompt_style, include_sample_id, sample_types, weights
):
    return [
        build_record(
            split=split,
            row_idx=row_idx,
            plan_state=plan_state,
            metadata=metadata,
            tokenizer=tokenizer,
            prompt_style=prompt_style,
            include_sample_id=include_sample_id,
            sample_weight=weights["direct_plan"],
        )
    ]
