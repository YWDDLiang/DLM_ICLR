"""DLM-owned STOP / mode / count / ordered-site decisions.

Only representation and remaining-call feasibility restrict actions. There is
no confidence threshold, rank-based override, or one-atom full-cell forcing.
"""

from __future__ import annotations
import math
import torch

MODES = ("stop", "local_xyz", "all_xyz", "full_cell")
COUNTS = (1, 2, 4, 8)


def normalized_choices(logits: torch.Tensor, actions: list[int], temperature: float = 1.0):
    if logits.ndim != 1 or not actions or len(actions) != len(set(actions)) or temperature <= 0:
        raise ValueError("Invalid action vocabulary or temperature")
    selected = logits.index_select(0, torch.tensor(actions, device=logits.device))
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Nonfinite policy logits")
    return torch.log_softmax(selected.double() / temperature, 0)


def feasible_modes(n: int, available_calls: int):
    if not 1 <= n <= 20 or available_calls < 1:
        raise ValueError("Need one inspection call and a valid atom count")
    # available_calls includes this inspection and one future scoring call.
    modes = [0]
    if available_calls >= 1 + 3 + 1:
        modes.append(1)
    if available_calls >= 1 + 3 * n + 1:
        modes.append(2)
    if available_calls >= 1 + 6 + 3 * n + 1:
        modes.append(3)
    return modes


def feasible_counts(n: int, available_calls: int):
    return [i for i, count in enumerate(COUNTS) if count <= n and 2 + 3 * count <= available_calls]


def ordered_site_logprob(logits: torch.Tensor, order: list[int], temperature: float = 1.0):
    if len(order) != len(set(order)) or any(not 0 <= i < len(logits) for i in order):
        raise ValueError("Sites must be selected without replacement")
    remaining = list(range(len(logits)))
    result = logits.sum() * 0.0
    for site in order:
        result = result + normalized_choices(logits, remaining, temperature)[remaining.index(site)]
        remaining.remove(site)
    return result


def action_positions(n: int, mode: int, sites: list[int]):
    if mode == 0:
        return []
    return (list(range(1, 7)) if mode == 3 else []) + [8 + 4 * site + a for site in sites for a in range(3)]
