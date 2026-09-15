"""Stateless matched-noise helpers for the exploratory P0/P1 screen."""

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


def paired_randn_bank(
    base_seeds: Iterable[int],
    *,
    role: str,
    diffusion_steps: int,
    trailing_shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build one full reverse-noise bank per registered pair.

    Every bank has the same max-atom shape. Arms with fewer atoms take a prefix,
    so shared pairs consume exactly the same coordinate and lattice noise where
    their tensor domains overlap.
    """

    if diffusion_steps < 2:
        raise ValueError("diffusion_steps must be at least two")
    rows = []
    for base_seed in base_seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(derive_subseed(int(base_seed), "parent", role))
        rows.append(
            torch.randn(
                (int(diffusion_steps) - 1, *(int(value) for value in trailing_shape)),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    if not rows:
        return torch.empty((0, diffusion_steps - 1, *trailing_shape), device=device, dtype=dtype)
    return torch.stack(rows)


__all__ = ["derive_subseed", "paired_randn_bank", "paired_uniform"]
