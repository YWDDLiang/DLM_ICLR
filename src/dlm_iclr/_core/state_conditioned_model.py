"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any
import torch
from torch import nn
from dlm_iclr._core.fixed_slot import SYMBOL_TO_Z
from dlm_iclr._core.periodic_geometry_objective import build_geometry_token_support
from dlm_iclr._core.periodic_state_conditioning import PeriodicStateConditioner, PeriodicStateConfig


@dataclass(frozen=True)
class CrystalStateContext:
    old_token_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    num_sites: torch.Tensor
    program_rank: torch.Tensor
    active_token_mask: torch.Tensor
    task_ids: torch.Tensor | None = None
    numeric_noise_level: torch.Tensor | None = None
    # V2: declared log-volume, log-shape and Cartesian noise parameters.
    # None (legacy runtime) or an all-minus-one row means all three are unknown.
    numeric_noise_components: torch.Tensor | None = None


class StateConditionedDLM(nn.Module):
    """Add a numeric old-state residual before the existing Transformer."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        config: PeriodicStateConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.state_conditioner = PeriodicStateConditioner(config)
        self.state_config = config
        vocabulary = tokenizer.get_vocab()
        vocabulary_size = int(base_model.get_input_embeddings().weight.shape[0])
        support = build_geometry_token_support(tokenizer)
        values = torch.full((9, vocabulary_size), float("nan"), dtype=torch.float32)
        family_axes = [("length", a) for a in "ABC"]
        family_axes += [("angle", a) for a in "ABG"]
        family_axes += [("coord", a) for a in "XYZ"]
        for row, (family, axis) in enumerate(family_axes):
            table = support[family][axis]
            values[row, torch.tensor(table["ids"], dtype=torch.long)] = torch.tensor(
                table["values"], dtype=torch.float32
            )
        species = torch.zeros(vocabulary_size, dtype=torch.long)
        for token, token_id in vocabulary.items():
            match = re.fullmatch(r"<E_([A-Z][a-z]?)>", str(token))
            if match and match.group(1) in SYMBOL_TO_Z:
                species[int(token_id)] = int(SYMBOL_TO_Z[match.group(1)])
        self.register_buffer("geometry_values", values, persistent=False)
        self.register_buffer("species_by_token", species, persistent=False)

    def _apply(self, fn, recurse=True):
        # Integer decoding values are geometric constants, not model activations.
        # Preserve their original FP32 precision across parent dtype conversion.
        values = self.geometry_values
        result = super()._apply(fn, recurse=recurse)
        self.geometry_values = values.to(device=self.geometry_values.device, dtype=torch.float32)
        return result

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self):
        return self.get_input_embeddings().weight.device

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def geometry_inputs(self, context: CrystalStateContext) -> dict[str, torch.Tensor]:
        ids = context.old_token_ids
        if ids.ndim != 2 or context.active_token_mask.shape != ids.shape:
            raise ValueError("old state and active mask must have shape [batch, length]")
        batch, length = ids.shape
        prompt = context.prompt_lengths.to(device=ids.device, dtype=torch.long)
        counts = context.num_sites.to(device=ids.device, dtype=torch.long)
        if prompt.shape != (batch,) or counts.shape != (batch,):
            raise ValueError("one prompt length and atom count is required per row")
        if bool(((counts < 1) | (counts > self.state_config.max_sites)).any()):
            raise ValueError("atom count is outside the retained representation")
        if bool((prompt + 7 + 4 * counts > length).any()):
            raise ValueError("geometry body exceeds the token canvas")
        device = ids.device
        row_index = torch.arange(batch, device=device)[:, None]
        cell_positions = prompt[:, None] + torch.arange(1, 7, device=device)[None, :]
        cell_ids = ids[row_index, cell_positions]
        cell_values = self.geometry_values[torch.arange(6, device=device)[None, :], cell_ids]
        known = torch.isfinite(cell_values).all(-1) & (cell_values[:, :3] > 0).all(-1)
        safe_lengths = torch.where(known[:, None], cell_values[:, :3], 1.0)
        safe_angles = torch.where(known[:, None], cell_values[:, 3:], 90.0)
        alpha, beta, gamma = torch.deg2rad(safe_angles).unbind(-1)
        a, b, c = safe_lengths.unbind(-1)
        sin_gamma = gamma.sin()
        cx = c * beta.cos()
        cy = c * (alpha.cos() - beta.cos() * gamma.cos()) / sin_gamma.clamp_min(1e-8)
        cz_square = c.square() - cx.square() - cy.square()
        known = known & (sin_gamma.abs() > 1e-8) & (cz_square > 1e-10)
        zero = torch.zeros_like(a)
        lattice = torch.stack(
            (
                torch.stack((a, zero, zero), -1),
                torch.stack((b * gamma.cos(), b * sin_gamma, zero), -1),
                torch.stack((cx, cy, cz_square.clamp_min(1e-10).sqrt()), -1),
            ),
            1,
        )
        identity = torch.eye(3, device=device, dtype=torch.float32)[None]
        lattice = torch.where(known[:, None, None], lattice, identity)
        slots = torch.arange(self.state_config.max_sites, device=device)[None]
        valid_slot = slots < counts[:, None]
        element_positions = (prompt[:, None] + 7 + 4 * slots).clamp_max(length - 1)
        element_ids = ids[row_index, element_positions]
        species = self.species_by_token[element_ids] * valid_slot.to(torch.long)
        coordinates = []
        active_sites = torch.zeros_like(valid_slot)
        for axis in range(3):
            positions = (prompt[:, None] + 8 + 4 * slots + axis).clamp_max(length - 1)
            token_ids = ids[row_index, positions]
            coordinates.append(self.geometry_values[6 + axis, token_ids])
            active_sites |= context.active_token_mask[row_index, positions].bool()
        fractional = torch.stack(coordinates, -1)
        site_known = torch.isfinite(fractional).all(-1) & valid_slot
        fractional = torch.nan_to_num(fractional, nan=0.0).remainder(1.0)
        fractional = fractional * valid_slot[..., None]
        rank = context.program_rank.to(device=device, dtype=torch.long)
        if rank.shape != valid_slot.shape:
            raise ValueError("program_rank must cover the fixed padded site dimension")
        return {
            "lattice": lattice.float(),
            "fractional": fractional.float(),
            "species": species,
            "site_known": site_known,
            "lattice_known": known,
            "program_rank": rank,
            "active_sites": active_sites & valid_slot,
        }

    def state_embeddings(self, input_ids, geometry_context):
        if geometry_context is None:
            return self.get_input_embeddings()(input_ids)
        if geometry_context.old_token_ids.shape != input_ids.shape:
            raise ValueError("old context must align with the current canvas")
        encoded = self.state_conditioner(**self.geometry_inputs(geometry_context))
        embeddings = self.get_input_embeddings()(input_ids)
        batch, length, hidden = embeddings.shape
        residual = embeddings.new_zeros(batch, length, hidden)
        rows = torch.arange(batch, device=input_ids.device)[:, None]
        prompt = geometry_context.prompt_lengths.to(input_ids.device)
        cell_positions = prompt[:, None] + torch.arange(1, 7, device=input_ids.device)[None]
        cell_residual = encoded["cell_embedding"].to(embeddings.dtype)
        # Explicit expansion avoids PyTorch 2.4's deterministic CUDA
        # index_put broadcasting assertion; the assigned values are identical.
        residual[rows, cell_positions] = (
            cell_residual[:, None, :].expand(-1, cell_positions.shape[1], -1).contiguous()
        )
        slots = torch.arange(self.state_config.max_sites, device=input_ids.device)[None]
        valid = slots < geometry_context.num_sites.to(input_ids.device)[:, None]
        site_residual = encoded["site_embeddings"].to(embeddings.dtype) * valid[..., None]
        # E and XYZ receive the state of that native site, without moving body slots.
        for offset in range(4):
            positions = (prompt[:, None] + 7 + 4 * slots + offset).clamp_max(length - 1)
            expanded = positions[..., None].expand(-1, -1, hidden)
            residual = residual.scatter_add(1, expanded, site_residual)
        return embeddings + residual

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        geometry_context: CrystalStateContext | None = None,
        **kwargs,
    ):
        if geometry_context is None:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return self.base_model(
            input_ids=None,
            inputs_embeds=self.state_embeddings(input_ids, geometry_context),
            attention_mask=attention_mask,
            **kwargs,
        )

    def save_pretrained(self, output_dir: str | Path, **kwargs) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output, **kwargs)
        (output / "periodic_state_config.json").write_text(
            json.dumps(asdict(self.state_config), indent=2) + "\n", encoding="utf-8"
        )
        torch.save(self.state_conditioner.state_dict(), output / "periodic_state.pt")

    def load_state_conditioner(self, checkpoint_dir: str | Path) -> None:
        root = Path(checkpoint_dir)
        recorded = json.loads((root / "periodic_state_config.json").read_text(encoding="utf-8"))
        if recorded != asdict(self.state_config):
            raise ValueError("saved state conditioner configuration differs")
        state = torch.load(root / "periodic_state.pt", map_location="cpu", weights_only=True)
        self.state_conditioner.load_state_dict(state, strict=True)


def set_state_lora_trainable(model: StateConditionedDLM) -> dict[str, int]:
    """Preserve full saved tables, but optimize only existing LoRA and state input."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("state_conditioner.") or ".lora_A." in name or ".lora_B." in name
        )
    # Deployment likelihood has no dropout; keeping Module.training true leaves
    # the base model's training/checkpoint path enabled.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    counts = {"lora": 0, "conditioner": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        key = "frozen"
        if parameter.requires_grad:
            key = "conditioner" if name.startswith("state_conditioner.") else "lora"
        counts[key] += parameter.numel()
    if counts["lora"] == 0:
        raise ValueError("checkpoint contains no trainable existing LoRA parameters")
    return counts
