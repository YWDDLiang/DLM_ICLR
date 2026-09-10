"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import torch
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r03_physics_transfer import supported_scalar_logits
from crystal_dlm.llada_generation import _apply_lightweight_decoding_masks


def numeric_order(n, branch):
    cell = list(range(1, 7))
    if branch == "G":
        return cell + [8 + 4 * site + axis for axis in range(3) for site in range(n)]
    return cell + [8 + 4 * site + axis for site in range(n) for axis in range(3)]


def legal_vector(vector, body, n, position, support):
    if position >= 8:
        return supported_scalar_logits(vector, body, 0, n, position, constraints=support)
    minimum = torch.finfo(vector.dtype).min
    family = "length_token_to_bin" if position < 4 else "angle_token_to_bin"
    axis = ("LA", "LB", "LC")[position - 1] if position < 4 else ("AA", "AB", "AG")[position - 4]
    ids = list(support[family][axis])
    # The detached probe computes only the denominator support. No physical
    # force/energy tensor enters the differentiable policy graph.
    with torch.no_grad():
        probe = vector.new_full((1, len(body), vector.numel()), minimum)
        probe[0, position, ids] = 0.0
        canvas = torch.tensor([body], device=vector.device)
        active = torch.zeros_like(canvas, dtype=torch.bool)
        active[0, position] = True
        _apply_lightweight_decoding_masks(probe, canvas, 0, len(body), support, active, MASK_TOKEN_ID)
        legal = torch.isfinite(probe[0, position]) & (probe[0, position] > minimum)
    return vector.masked_fill(~legal, minimum), {
        "available": bool(legal.any()),
        "legal_count": int(legal.sum()),
    }
