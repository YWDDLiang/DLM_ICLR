"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import torch


def minimum_image_distances(
    fractional_deltas: "torch.Tensor",
    lattice: "torch.Tensor",
    *,
    image_radius: int,
) -> "torch.Tensor":
    """Return the minimum norm over a centered bounded image shell.

    ``fractional_deltas`` may be ``[..., 3]`` with one lattice ``[3, 3]``, or
    ``[B, ..., 3]`` with batched lattices ``[B, 3, 3]``.
    """

    import torch

    vectors, _shifts = minimum_image_vectors(
        fractional_deltas,
        lattice,
        image_radius=image_radius,
    )
    squared = vectors.square().sum(dim=-1)
    # Preserve the historical non-zero floor used by every existing caller.
    return torch.sqrt(squared.clamp_min(1.0e-12))


def minimum_image_vectors(
    fractional_deltas: "torch.Tensor",
    lattice: "torch.Tensor",
    *,
    image_radius: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return bounded minimum-image Cartesian vectors and lattice shifts.

    The returned integer shift has the same leading shape as
    ``fractional_deltas`` and satisfies

    ``vector == (fractional_deltas + shift) @ lattice``.

    Selection uses exactly the same centered 27- or 125-image shell as
    :func:`minimum_image_distances`.  The selected vector remains
    differentiable with respect to the input delta and lattice away from image
    boundaries; the discrete argmin shift is intentionally non-differentiable.
    Batched lattices follow the existing convention: deltas have shape
    ``[B, ..., 3]`` and lattices have shape ``[B, 3, 3]``.
    """

    import torch

    if not isinstance(fractional_deltas, torch.Tensor) or not isinstance(lattice, torch.Tensor):
        raise TypeError("fractional_deltas and lattice must be torch tensors")
    if fractional_deltas.ndim < 1 or fractional_deltas.shape[-1] != 3:
        raise ValueError("fractional_deltas must end in three coordinates")
    if not (fractional_deltas.is_floating_point() and lattice.is_floating_point()):
        raise TypeError("fractional_deltas and lattice must be floating point")
    if fractional_deltas.device != lattice.device:
        raise ValueError("fractional_deltas and lattice must share a device")
    if fractional_deltas.dtype != lattice.dtype:
        raise ValueError("fractional_deltas and lattice must share a dtype")
    if not bool(torch.isfinite(fractional_deltas).all().item()):
        raise ValueError("fractional_deltas contain non-finite values")
    if not bool(torch.isfinite(lattice).all().item()):
        raise ValueError("lattice contains non-finite values")
    if int(image_radius) not in (1, 2):
        raise ValueError("image_radius must be one (27) or two (125)")

    rounded = torch.round(fractional_deltas)
    centered = fractional_deltas - rounded
    values = torch.arange(
        -int(image_radius),
        int(image_radius) + 1,
        dtype=centered.dtype,
        device=centered.device,
    )
    shell_shifts = torch.cartesian_prod(values, values, values).reshape(-1, 3)
    candidates = centered.unsqueeze(-2) + shell_shifts
    if lattice.ndim == 2:
        if tuple(lattice.shape) != (3, 3):
            raise ValueError("lattice must have shape [3, 3]")
        cartesian = torch.einsum("...nc,cd->...nd", candidates, lattice)
    elif lattice.ndim == 3:
        if (
            fractional_deltas.ndim < 2
            or lattice.shape[0] != fractional_deltas.shape[0]
            or tuple(lattice.shape[1:]) != (3, 3)
        ):
            raise ValueError("batched lattice requires deltas [B, ..., 3] and lattice [B, 3, 3]")
        cartesian = torch.einsum("b...nc,bcd->b...nd", candidates, lattice)
    else:
        raise ValueError("lattice must be rank two or three")

    selected = cartesian.square().sum(dim=-1).argmin(dim=-1)
    vector_index = selected.unsqueeze(-1).unsqueeze(-1).expand(*selected.shape, 1, 3)
    vectors = torch.gather(cartesian, dim=-2, index=vector_index).squeeze(-2)
    selected_shell_shifts = shell_shifts[selected]
    total_shifts = torch.round(selected_shell_shifts - rounded).to(torch.long)
    return vectors, total_shifts
