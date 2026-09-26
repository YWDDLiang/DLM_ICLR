"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

from __future__ import annotations

from dlm_iclr.runtime.capacity import MAX_ATOMS
from typing import Any, Mapping, Sequence
import torch
from dlm_iclr._core.fixed_slot import FixedSlotConfig, MASK_TOKEN_ID
from dlm_iclr._core.lattice_geometry import lattice_angle_rad
from dlm_iclr._core.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _lattice_matrix_from_token_ids,
)
from dlm_iclr._core.periodic_geometry_ops import minimum_image_distances


REPAIR_SUPPORT_PROTOCOL = {
    "representation": "dynamic_v1",
    "max_atoms": MAX_ATOMS,
    "coord_period": 100,
    "canonicalize_periodic_alias": True,
    "duplicate_coordinate_mask": True,
    "lattice_volume_mask": True,
    "min_lattice_rad": 1e-4,
    "pbc_min_distance_mask": True,
    "pbc_min_distance_A": 0.5,
    "pbc_image_radius": 2,
}


class TransferContractError(ValueError):
    """A source or repair contract is not safe to reinterpret as this dataset."""


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TransferContractError(f"{name} must be an exact integer")
    if isinstance(value, str) and not value.isdecimal():
        raise TransferContractError(f"{name} must be an exact nonnegative integer")
    return int(value)


def build_repair_constraints(tokenizer: Any) -> dict[str, Any]:
    """Token-ID maps for the fixed legacy schema/alias/125-image support."""
    config, vocab = FixedSlotConfig(), tokenizer.get_vocab()

    def ids(prefix: str, low: int, high: int) -> dict[int, int]:
        result = {}
        for value in range(low, high + 1):
            token = f"<{prefix}_{value:03d}>"
            if token not in vocab:
                raise TransferContractError(f"base constructor tokenizer lacks {token}; no vocabulary extension allowed")
            result[int(vocab[token])] = value
        if len(result) != high - low + 1:
            raise TransferContractError("crystal token IDs are not one-to-one")
        return result

    coords = {axis: ids(axis, 0, 100) for axis in "XYZ"}
    lengths = {axis: ids(axis, config.length_min_bin, config.length_max_bin) for axis in ("LA", "LB", "LC")}
    angles = {axis: ids(axis, config.angle_min_bin, config.angle_max_bin) for axis in ("AA", "AB", "AG")}
    return {
        **REPAIR_SUPPORT_PROTOCOL,
        "count_token_to_n": ids("N", 1, MAX_ATOMS),
        "coord_token_to_bin": coords,
        "coord_bin_to_token_id": {a: {v: k for k, v in m.items()} for a, m in coords.items()},
        "coordinate_alias_token_ids": {
            a: (int(vocab[f"<{a}_000>"]), int(vocab[f"<{a}_100>"])) for a in "XYZ"
        },
        "length_token_to_bin": lengths,
        "length_step": config.length_step,
        "angle_token_to_bin": angles,
        "z_bin_to_token_id": {v: k for k, v in coords["Z"].items()},
        "gamma_bin_to_token_id": {v: k for k, v in angles["AG"].items()},
        "zero_length_token_ids_by_position": {
            i: int(vocab[f"<{a}_000>"]) for i, a in enumerate(("LA", "LB", "LC"), 1)
        },
    }


def _checked_constraints(tokenizer: Any, constraints: Mapping[str, Any] | None) -> dict:
    result = build_repair_constraints(tokenizer) if constraints is None else dict(constraints)
    for key, value in REPAIR_SUPPORT_PROTOCOL.items():
        if result.get(key) != value:
            raise TransferContractError(f"repair support changed at {key}")
    if result.get("body_offset", 0) != 0 or result.get("length_step") != 0.1:
        raise TransferContractError("repair requires the unchanged base constructor body ABI")
    return result


def _fixed_cell(body: torch.Tensor, constraints: dict) -> tuple[torch.Tensor | None, str | None]:
    maps = constraints["angle_token_to_bin"]
    angles = [maps[a].get(int(body[p])) for a, p in (("AA", 4), ("AB", 5), ("AG", 6))]
    if any(v is None for v in angles) or lattice_angle_rad(*angles) <= 1e-4:
        return None, "invalid_fixed_lattice_angles"
    lattice = _lattice_matrix_from_token_ids(body, prompt_length=0, constraints=constraints)
    if lattice is None or not bool(torch.isfinite(lattice).all()):
        return None, "invalid_fixed_lattice"
    shell = torch.arange(-2, 3, dtype=lattice.dtype, device=lattice.device)
    shifts = torch.cartesian_prod(shell, shell, shell)
    shifts = shifts[(shifts != 0).any(-1)]
    if float(torch.linalg.vector_norm(shifts @ lattice, dim=-1).min()) < 0.5:
        return None, "periodic_self_image_below_0.5A"
    return lattice, None


def geometry_support_report(
    body_ids: Sequence[int] | torch.Tensor, *, tokenizer=None, constraints=None
) -> dict[str, Any]:
    """Check a complete native target without changing its physical structure."""
    constraints = _checked_constraints(tokenizer, constraints)
    body = torch.as_tensor(body_ids, dtype=torch.long)
    if body.ndim != 1 or len(body) < 11:
        return {"supported": False, "reason": "invalid_body_length"}
    n = constraints["count_token_to_n"].get(int(body[0]))
    if n is None or len(body) != 7 + 4 * n:
        return {"supported": False, "reason": "invalid_body_cardinality"}
    lattice, reason = _fixed_cell(body, constraints)
    if reason is not None:
        return {"supported": False, "reason": reason}
    coordinates = []
    for site in range(n):
        values = [
            constraints["coord_token_to_bin"][a].get(int(body[8 + 4 * site + k])) for k, a in enumerate("XYZ")
        ]
        if any(v is None for v in values):
            return {"supported": False, "reason": "incomplete_native_coordinates"}
        coordinates.append([float(v % 100) / 100 for v in values])
    minimum = None
    if n > 1:
        frac = torch.tensor(coordinates, dtype=lattice.dtype, device=lattice.device)
        distances = minimum_image_distances(frac[:, None] - frac[None, :], lattice, image_radius=2)
        distances.fill_diagonal_(torch.inf)
        minimum = float(distances.min())
        if minimum < 0.5:
            return {"supported": False, "reason": "native_pair_below_0.5A", "minimum_distance_A": minimum}
    return {"supported": True, "reason": None, "minimum_distance_A": minimum}


def supported_scalar_logits(
    raw_logits: torch.Tensor,
    input_body: Sequence[int] | torch.Tensor,
    prompt_length: int,
    N: int,
    position: int,
    tokenizer=None,
    constraints=None,
    mask_id: int = MASK_TOKEN_ID,
):
    """Return (active logits, report), preserving alias gradients and exact support.

    raw_logits may be V, L×V, or 1×L×V. position is body-relative. A false
    report['available'] means inference must roll back the whole staged XYZ;
    training must not restore an unsupported target to the action set.
    """
    constraints = _checked_constraints(tokenizer, constraints)
    n = _integer(N, "N")
    body = torch.as_tensor(input_body, dtype=torch.long, device=raw_logits.device)
    component = (position - 8) % 4
    if (
        body.ndim != 1
        or len(body) != 7 + 4 * n
        or position < 8
        or position >= len(body)
        or component > 2
        or int(body[position]) != int(mask_id)
        or constraints["count_token_to_n"].get(int(body[0])) != n
    ):
        raise TransferContractError("active scalar must be a masked XYZ position in exact 7+4N")
    if raw_logits.ndim == 1:
        vector = raw_logits
    elif raw_logits.ndim == 2:
        vector = raw_logits[prompt_length + position]
    elif raw_logits.ndim == 3 and raw_logits.shape[0] == 1:
        vector = raw_logits[0, prompt_length + position]
    else:
        raise TransferContractError("one logical scalar row is required")
    if not vector.is_floating_point():
        raise TransferContractError("LM logits must be floating point")
    minimum = torch.finfo(vector.dtype).min
    _, cell_reason = _fixed_cell(body, constraints)
    if cell_reason is not None:
        return vector.masked_fill(torch.ones_like(vector, dtype=torch.bool), minimum), {
            "available": False,
            "no_legal_completion": False,
            "reason": cell_reason,
            "legal_count": 0,
        }
    axis = "XYZ"[component]
    allowed = torch.zeros((len(body), vector.numel()), dtype=torch.bool, device=vector.device)
    allowed[position, list(constraints["coord_token_to_bin"][axis])] = True
    with torch.no_grad():
        # Only a detached support calculation uses the temporary body canvas.
        # The retained active-vector graph below is the differentiable path.
        probe = vector.new_zeros((1, len(body), vector.numel()))
        probe[0, position] = vector.detach()
        _apply_schema_masks(probe, body[None], 0, len(body), allowed, None)
        active = torch.zeros((1, len(body)), dtype=torch.bool, device=vector.device)
        active[0, position] = True
        reports = _apply_lightweight_decoding_masks(
            probe, body[None], 0, len(body), constraints, active, int(mask_id)
        )
        no_completion = (0, position) in reports["pbc_no_legal_completion"]
        selected = probe[0, position]
        legal = torch.isfinite(selected) & (selected > minimum)
        nonfinite = bool(torch.isnan(selected).any() or torch.isposinf(selected).any())
    transformed = vector.masked_fill(~allowed[position], minimum)
    canonical, alias = constraints["coordinate_alias_token_ids"][axis]
    merged = torch.logaddexp(transformed[canonical], transformed[alias])
    transformed = transformed.scatter(0, torch.tensor([canonical], device=vector.device), merged.reshape(1))
    available = bool(legal.any()) and not no_completion and not nonfinite
    reason = (
        "no_legal_Z_completion"
        if no_completion
        else "nonfinite_active_logits"
        if nonfinite
        else "empty_scalar_support"
        if not available
        else None
    )
    if not available:
        legal = torch.zeros_like(legal)
    return transformed.masked_fill(~legal, minimum), {
        "available": available,
        "no_legal_completion": no_completion,
        "reason": reason,
        "legal_count": int(legal.sum()),
    }
