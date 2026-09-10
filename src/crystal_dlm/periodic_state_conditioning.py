"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PeriodicStateConfig:
    """Serializable with ``dataclasses.asdict(conditioner.config)``."""

    hidden_size: int
    width: int = 128
    max_sites: int = 20
    radial_basis_count: int = 16
    radial_cutoff_A: float = 6.0
    image_radius: int = 2

    def __post_init__(self) -> None:
        for name in ("hidden_size", "width", "max_sites", "radial_basis_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.radial_cutoff_A) or self.radial_cutoff_A <= 0:
            raise ValueError("radial_cutoff_A must be finite and positive")
        if self.image_radius not in (1, 2):
            raise ValueError("image_radius must be one (27) or two (125)")


def _mlp(input_size: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, width, dtype=torch.float32),
        nn.SiLU(),
        nn.Linear(width, width, dtype=torch.float32),
        nn.SiLU(),
    )


class PeriodicStateConditioner(nn.Module):
    """Encode real old geometry and program metadata, with zero initial output.

    ``species`` is atomic Z (1..118); zero means padding/unavailable species and
    produces no site increment or pooling contribution. ``site_known`` means
    the *old* fractional coordinates are available, independently of whether
    that site is currently active/remasked. ``active_sites`` selects updates,
    not occupancy: inactive known sites still supply context. ``program_rank``
    is a zero-based ordinal, normalized by ``max_sites - 1``.

    Unknown numerical entries may be arbitrary, including NaN: they are masked
    before arithmetic, never filled with guessed geometry. Known entries must
    be finite. A cell without known lattice has no metric or distance features;
    known fractional coordinates and composition/program flags remain usable.

    Pair RBFs sum over a centered finite 27/125-image shell, including self
    images except (i == j, shift == 0), with a cosine cutoff in Angstrom. The
    shell is bounded, not an exact neighbor enumeration for arbitrary skew
    bases. Absolute old-coordinate sin/cos features intentionally retain origin
    information; arbitrary fractional translation invariance is not promised.

    Parameters, buffers, arithmetic, and outputs remain FP32 even under parent
    dtype conversions or autocast. The caller casts only the returned increments
    to the base embedding dtype. Only the two final projections start at zero.
    """

    def __init__(self, config: PeriodicStateConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.species_embedding = nn.Embedding(119, width, padding_idx=0, dtype=torch.float32)
        # Six old-coordinate channels, rank, site/lattice-known, active flags.
        self.site_encoder = _mlp(width + 10, width)
        # RBF image sum, periodic relative coordinates, rank difference, endpoints.
        self.pair_mlp = _mlp(config.radial_basis_count + 7 + 2 * width, width)
        # Twelve cell scalars plus separate site and periodic-environment pools.
        self.cell_encoder = _mlp(12 + 2 * width, width)
        self.site_update = _mlp(3 * width, width)
        self.cell_projection = nn.Linear(width, config.hidden_size, bias=False, dtype=torch.float32)
        self.site_projection = nn.Linear(width, config.hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.cell_projection.weight)
        nn.init.zeros_(self.site_projection.weight)

        centers = torch.linspace(0, config.radial_cutoff_A, config.radial_basis_count, dtype=torch.float32)
        spacing = config.radial_cutoff_A / max(config.radial_basis_count - 1, 1)
        self.register_buffer("radial_centers", centers)
        self.register_buffer("radial_gamma", torch.tensor(spacing**-2, dtype=torch.float32))
        values = torch.arange(-config.image_radius, config.image_radius + 1, dtype=torch.float32)
        shifts = torch.cartesian_prod(values, values, values)
        self.register_buffer("image_shifts", shifts)
        self.register_buffer("zero_image", (shifts == 0).all(dim=-1))

    def _apply(self, fn, recurse: bool = True):
        # Preserve the original FP32 values, not a half -> float round trip.
        # This also covers gradients when a containing model calls .to(dtype=...).
        def keep_fp32(tensor: Tensor) -> Tensor:
            result = fn(tensor)
            if result.is_floating_point() and result.dtype != torch.float32:
                return tensor.to(device=result.device, dtype=torch.float32)
            return result

        return super()._apply(keep_fp32, recurse=recurse)

    def _radial_pairs(
        self, lattice: Tensor, fractional: Tensor, geometry_known: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return image-summed RBFs [B,N,N,R], centered deltas, and pair mask.

        Unlike the shared MIC operator, all in-cutoff shell images contribute.
        No width-sized hidden activation is allocated per periodic image.
        """
        delta = fractional.unsqueeze(1) - fractional.unsqueeze(2)
        delta = delta - torch.round(delta)
        images = delta.unsqueeze(-2) + self.image_shifts
        vectors = torch.einsum("bijsd,bdk->bijsk", images, lattice)
        distances = vectors.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
        n = fractional.shape[1]
        central_self = torch.eye(n, dtype=torch.bool, device=lattice.device).unsqueeze(-1) & self.zero_image
        pair_known = geometry_known.unsqueeze(1) & geometry_known.unsqueeze(2)
        image_mask = pair_known.unsqueeze(-1) & ~central_self & (distances < self.config.radial_cutoff_A)
        relative_distance = (distances / self.config.radial_cutoff_A).clamp(max=1)
        envelope = 0.5 * (1 + torch.cos(math.pi * relative_distance))
        envelope = torch.where(image_mask, envelope, 0.0)
        rbf = torch.exp(-self.radial_gamma * (distances.unsqueeze(-1) - self.radial_centers).square())
        radial_sum = (rbf * envelope.unsqueeze(-1)).sum(dim=-2)
        return radial_sum, delta, image_mask.any(dim=-1)

    @staticmethod
    def _cell_scalars(
        lattice: Tensor,
        lattice_known: Tensor,
        present: Tensor,
        site_known: Tensor,
        active_sites: Tensor,
        max_sites: int,
    ) -> Tensor:
        gram = lattice @ lattice.transpose(-1, -2)
        scale_squared = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
        six = torch.stack(
            [gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2], gram[:, 0, 1], gram[:, 0, 2], gram[:, 1, 2]], dim=-1
        ) / scale_squared.unsqueeze(-1)
        # Scalar triple product avoids eigendecomposition and singular det backward.
        volume = (lattice[:, 0] * torch.linalg.cross(lattice[:, 1], lattice[:, 2], dim=-1)).sum(dim=-1).abs()
        count = present.sum(dim=-1).float()
        denominator = count.clamp_min(1)
        physical = torch.cat(
            [
                six,
                0.5 * scale_squared.log().unsqueeze(-1),
                (volume / denominator).clamp_min(1e-12).log().unsqueeze(-1),
            ],
            dim=-1,
        )
        physical = torch.where(lattice_known.unsqueeze(-1), physical, 0.0)
        metadata = torch.stack(
            [
                lattice_known.float(),
                count / max_sites,
                site_known.sum(dim=-1).float() / denominator,
                active_sites.sum(dim=-1).float() / denominator,
            ],
            dim=-1,
        )
        return torch.cat([physical, metadata], dim=-1)

    def forward(
        self,
        *,
        lattice: Tensor,
        fractional: Tensor,
        species: Tensor,
        site_known: Tensor,
        lattice_known: Tensor,
        program_rank: Tensor,
        active_sites: Tensor,
    ) -> dict[str, Tensor]:
        """Return ``cell_embedding [B,H]`` and ``site_embeddings [B,N,H]``."""
        if fractional.ndim != 3 or fractional.shape[-1] != 3:
            raise ValueError("fractional must have shape [B, N, 3]")
        batch, n, _ = fractional.shape
        if not 1 <= n <= self.config.max_sites:
            raise ValueError("N must be in 1..max_sites")
        if lattice.shape != (batch, 3, 3) or lattice_known.shape != (batch,):
            raise ValueError("lattice/lattice_known must have shape [B,3,3]/[B]")
        for name, value in (
            ("species", species),
            ("site_known", site_known),
            ("program_rank", program_rank),
            ("active_sites", active_sites),
        ):
            if value.shape != (batch, n):
                raise ValueError(f"{name} must have shape [B, N]")
        if any(value.dtype != torch.bool for value in (site_known, lattice_known, active_sites)):
            raise TypeError("known/active flags must be bool tensors")
        if program_rank.dtype != torch.long or species.dtype not in (torch.int32, torch.int64):
            raise TypeError("program_rank must be long and species must be integer atomic Z")

        with torch.autocast(device_type=lattice.device.type, enabled=False):
            present = species != 0
            site_known = site_known & present
            active_sites = active_sites & present
            lattice = torch.where(lattice_known[:, None, None], lattice.float(), 0.0)
            fractional = torch.where(site_known.unsqueeze(-1), fractional.float(), 0.0)
            fractional = fractional.remainder(1.0)
            phase = 2 * math.pi * fractional
            periodic = torch.cat([phase.sin(), phase.cos()], dim=-1)
            periodic = torch.where(site_known.unsqueeze(-1), periodic, 0.0)
            rank = torch.where(present, program_rank, 0).float() / max(self.config.max_sites - 1, 1)
            flags = torch.stack(
                [
                    rank,
                    site_known.float(),
                    lattice_known[:, None].expand(-1, n).float(),
                    active_sites.float(),
                ],
                dim=-1,
            )
            sites = self.site_encoder(
                torch.cat(
                    [
                        self.species_embedding(species),
                        periodic,
                        flags,
                    ],
                    dim=-1,
                )
            )
            sites = torch.where(present.unsqueeze(-1), sites, 0.0)

            radial, delta, pair_mask = self._radial_pairs(
                lattice, fractional, site_known & lattice_known.unsqueeze(-1)
            )
            relative_phase = 2 * math.pi * delta
            pair_inputs = torch.cat(
                [
                    radial,
                    relative_phase.sin(),
                    relative_phase.cos(),
                    (rank.unsqueeze(1) - rank.unsqueeze(2)).unsqueeze(-1),
                    sites.unsqueeze(2).expand(-1, -1, n, -1),
                    sites.unsqueeze(1).expand(-1, n, -1, -1),
                ],
                dim=-1,
            )
            messages = self.pair_mlp(pair_inputs)
            messages = torch.where(pair_mask.unsqueeze(-1), messages, 0.0)
            neighbors = messages.sum(dim=2) / pair_mask.sum(dim=2, keepdim=True).clamp_min(1).float()
            count = present.sum(dim=1, keepdim=True).clamp_min(1).float()
            cell_scalars = self._cell_scalars(
                lattice, lattice_known, present, site_known, active_sites, self.config.max_sites
            )
            cell = self.cell_encoder(
                torch.cat(
                    [
                        cell_scalars,
                        sites.sum(dim=1) / count,
                        neighbors.sum(dim=1) / count,
                    ],
                    dim=-1,
                )
            )
            sites = self.site_update(
                torch.cat(
                    [
                        sites,
                        neighbors,
                        cell.unsqueeze(1).expand(-1, n, -1),
                    ],
                    dim=-1,
                )
            )
            return {
                "cell_embedding": torch.where(
                    present.any(dim=1, keepdim=True), self.cell_projection(cell), 0.0
                ),
                "site_embeddings": torch.where(present.unsqueeze(-1), self.site_projection(sites), 0.0),
            }
