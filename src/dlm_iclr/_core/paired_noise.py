"""Deterministic per-request random streams for crystal construction."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


def derive_subseed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def paired_uniform(
    base_seed: int,
    *,
    stage: str,
    step: int,
    shape: Iterable[int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_subseed(base_seed, stage, int(step)))
    return torch.rand(
        tuple(int(value) for value in shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )


__all__ = ["derive_subseed", "paired_uniform"]
