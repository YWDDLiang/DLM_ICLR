"""R5 exact-length dynamic body helpers.

R5-C keeps the compact dynamic-v1 token language but removes the max-canvas
generation tail.  A plan with ``N`` atoms creates exactly ``7 + 4N`` answer
positions: atom count, lattice, then one element/XYZ block per atom.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from dlm_iclr._core.dynamic_crystal import (
    arrays_to_dynamic_answer,
    dynamic_answer_token_count,
    parse_dynamic_answer,
)
from dlm_iclr._core.fixed_slot import FixedSlotConfig, FixedSlotError, Z_TO_SYMBOL
from dlm_iclr._core.r5_plan_state import build_body_prompt, validate_plan_state


R5_EXACT_LENGTH_REPRESENTATION = "r5_exact_dynamic_v1"


def num_atoms_from_plan(plan_state: Mapping[str, Any], *, max_atoms: int = 20) -> int:
    try:
        num_atoms = int(plan_state["N"])
    except Exception as exc:
        raise FixedSlotError("plan_state is missing integer N") from exc
    if not 1 <= num_atoms <= int(max_atoms):
        raise FixedSlotError(f"plan_state N {num_atoms} outside 1..{max_atoms}")
    return num_atoms


def exact_body_token_count(plan_state_or_n: Mapping[str, Any] | int) -> int:
    if isinstance(plan_state_or_n, Mapping):
        num_atoms = num_atoms_from_plan(plan_state_or_n)
    else:
        num_atoms = int(plan_state_or_n)
    return dynamic_answer_token_count(num_atoms)


def validate_answer_matches_plan(plan_state: Mapping[str, Any], answer: str) -> Dict[str, Any]:
    validation = validate_plan_state(plan_state)
    if not validation.valid_N:
        raise FixedSlotError("plan_state has invalid N")
    expected_n = num_atoms_from_plan(plan_state)
    arrays = parse_dynamic_answer(answer, strict=True)
    if int(arrays["num_atoms"]) != expected_n:
        raise FixedSlotError(f"answer N {arrays['num_atoms']} does not match plan N {expected_n}")
    expected_elements = list(plan_state.get("elements") or [])
    expected_counts = [int(value) for value in plan_state.get("counts") or []]
    actual_counts: Dict[str, int] = {}
    for symbol in arrays["species"]:
        actual_counts[str(symbol)] = actual_counts.get(str(symbol), 0) + 1
    expected = dict(zip(expected_elements, expected_counts))
    if expected and actual_counts != expected:
        raise FixedSlotError(f"answer composition {actual_counts} does not match plan {expected}")
    return arrays


def build_exact_length_record(
    *,
    plan_state: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    answer_separator: str = "",
    sample_weight: float = 1.0,
) -> Dict[str, Any]:
    answer, diagnostics = arrays_to_dynamic_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        separator=answer_separator,
    )
    validate_answer_matches_plan(plan_state, answer)
    prompt = build_body_prompt(plan_state)
    return {
        "task": "r5_exact_dynamic_body",
        "representation": "dynamic_v1",
        "r5_representation": R5_EXACT_LENGTH_REPRESENTATION,
        "prompt": prompt,
        "answer": answer,
        "text": prompt.rstrip() + "\n" + answer,
        "answer_semantic_length": exact_body_token_count(plan_state),
        "answer_token_count": exact_body_token_count(plan_state),
        "num_atoms": int(plan_state["N"]),
        "plan_state": dict(plan_state),
        "r5_plan_state": dict(plan_state),
        "metadata": dict(metadata or {}),
        "sample_weight": float(sample_weight),
        "encode_diagnostics": diagnostics.to_dict(),
        "loss_profile": "fixed_slot",
    }


def required_token_ids(tokenizer: Any, tokens: Sequence[str]) -> List[int]:
    vocab = tokenizer.get_vocab()
    missing = [token for token in tokens if token not in vocab]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required crystal tokens: {missing[:10]}")
    return [int(vocab[token]) for token in tokens]


def exact_dynamic_schema_constraints(
    tokenizer: Any,
    num_atoms: int,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> List[List[int]]:
    num_atoms = int(num_atoms)
    if not 1 <= num_atoms <= config.max_atoms:
        raise ValueError(f"num_atoms {num_atoms} outside 1..{config.max_atoms}")
    allowed: List[List[int]] = []
    allowed.append(required_token_ids(tokenizer, [f"<N_{num_atoms:03d}>"]))
    for prefix in ("LA", "LB", "LC"):
        allowed.append(
            required_token_ids(
                tokenizer,
                [f"<{prefix}_{idx:03d}>" for idx in range(config.length_min_bin, config.length_max_bin + 1)],
            )
        )
    for prefix in ("AA", "AB", "AG"):
        allowed.append(
            required_token_ids(
                tokenizer,
                [f"<{prefix}_{idx:03d}>" for idx in range(config.angle_min_bin, config.angle_max_bin + 1)],
            )
        )
    element_tokens = [f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1)]
    coord_tokens = {
        axis: [f"<{axis}_{idx:03d}>" for idx in range(config.coord_min_bin, config.coord_max_bin + 1)]
        for axis in ("X", "Y", "Z")
    }
    for _ in range(num_atoms):
        allowed.append(required_token_ids(tokenizer, element_tokens))
        allowed.append(required_token_ids(tokenizer, coord_tokens["X"]))
        allowed.append(required_token_ids(tokenizer, coord_tokens["Y"]))
        allowed.append(required_token_ids(tokenizer, coord_tokens["Z"]))
    expected = exact_body_token_count(num_atoms)
    if len(allowed) != expected:
        raise RuntimeError(f"Built {len(allowed)} positions, expected {expected}")
    return allowed


def exact_dynamic_generation_schedule(num_atoms: int) -> List[List[int]]:
    num_atoms = int(num_atoms)
    element_positions = [7 + 4 * slot_index for slot_index in range(num_atoms)]
    x_positions = [8 + 4 * slot_index for slot_index in range(num_atoms)]
    y_positions = [9 + 4 * slot_index for slot_index in range(num_atoms)]
    z_positions = [10 + 4 * slot_index for slot_index in range(num_atoms)]
    return [[0], element_positions, [1, 2, 3, 4, 5, 6], x_positions, y_positions, z_positions]


def count_prefill_for_batch(tokenizer: Any, num_atoms: int, batch_size: int) -> Dict[int, List[int]]:
    token_ids = required_token_ids(tokenizer, [f"<N_{int(num_atoms):03d}>"])
    return {0: [int(token_ids[0])] * int(batch_size)}


__all__ = [
    "R5_EXACT_LENGTH_REPRESENTATION",
    "build_exact_length_record",
    "count_prefill_for_batch",
    "exact_body_token_count",
    "exact_dynamic_generation_schedule",
    "exact_dynamic_schema_constraints",
    "num_atoms_from_plan",
    "validate_answer_matches_plan",
]
