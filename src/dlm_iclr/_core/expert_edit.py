"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import NamedTuple
import torch
from torch import nn
from dlm_iclr._core.periodic_repair_model import FP32Module
from dlm_iclr._core.periodic_state_conditioning import PeriodicStateConditioner, PeriodicStateConfig
from dlm_iclr._core.state_conditioned_model import (
    CrystalStateContext,
    StateConditionedDLM,
    set_state_lora_trainable,
)
from dlm_iclr._core.periodic_geometry_objective import build_geometry_token_support


EDITOR_SCHEMA = "expert_crystal_editor_v1"


@dataclass(frozen=True)
class ExpertEditConfig:
    hidden_size: int
    width: int = 128
    max_sites: int = 20
    schema: str = EDITOR_SCHEMA

    def __post_init__(self):
        if (
            self.hidden_size < 1
            or self.width < 1
            or not 1 <= self.max_sites <= 20
            or self.schema != EDITOR_SCHEMA
        ):
            raise ValueError("invalid expert editor configuration")


@dataclass(frozen=True)
class EditContext:
    old_token_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    num_sites: torch.Tensor
    active_token_mask: torch.Tensor
    task_ids: torch.Tensor
    remaining_steps: torch.Tensor
    reveal_fraction: torch.Tensor


class EditOutput(NamedTuple):
    logits: torch.Tensor
    mode_logits: torch.Tensor
    site_logits: torch.Tensor
    count_logits: torch.Tensor
    quality_logits: torch.Tensor


class FloatMLP(FP32Module):
    def __init__(self, inputs, width, outputs, *, zero_output=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, width, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(width, outputs, dtype=torch.float32),
        )
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self.layers(value.float())


class ExpertEditDLM(StateConditionedDLM):
    def __init__(self, base_model, tokenizer, config: ExpertEditConfig):
        state = PeriodicStateConfig(
            hidden_size=config.hidden_size, width=config.width, max_sites=config.max_sites
        )
        super().__init__(base_model, tokenizer, state)
        self.editor_config = config
        self.current_state_conditioner = PeriodicStateConditioner(state)
        self.raw_cell_adapter = FloatMLP(27, config.width, config.hidden_size, zero_output=True)
        self.raw_site_adapter = FloatMLP(21, config.width, config.hidden_size, zero_output=True)
        self.task_adapter = FloatMLP(4, config.width, config.hidden_size, zero_output=True)
        self.mode_head = FloatMLP(2 * config.hidden_size, config.width, 4)
        self.site_head = FloatMLP(config.hidden_size, config.width, 1)
        self.count_head = FloatMLP(2 * config.hidden_size, config.width, 4)
        self.quality_head = FloatMLP(2 * config.hidden_size, config.width, 4)
        self.forward_calls = 0

    def extra_modules(self):
        return {
            name: getattr(self, name)
            for name in (
                "current_state_conditioner",
                "raw_cell_adapter",
                "raw_site_adapter",
                "task_adapter",
                "mode_head",
                "site_head",
                "count_head",
                "quality_head",
            )
        }

    def content_modules(self):
        return {
            "state_conditioner": self.state_conditioner,
            **{
                name: module
                for name, module in self.extra_modules().items()
                if name not in ("mode_head", "site_head", "count_head", "quality_head")
            },
        }

    def _context(self, value: EditContext, ids):
        rank = torch.arange(self.editor_config.max_sites, device=ids.device)[None].expand(ids.shape[0], -1)
        return CrystalStateContext(
            ids, value.prompt_lengths, value.num_sites, rank, value.active_token_mask, value.task_ids
        )

    def _raw_values(self, ids, context):
        batch, length = ids.shape
        rows = torch.arange(batch, device=ids.device)[:, None]
        cell_positions = context.prompt_lengths[:, None] + torch.arange(1, 7, device=ids.device)[None]
        cell = self.geometry_values[torch.arange(6, device=ids.device)[None], ids[rows, cell_positions]]
        sites = torch.arange(self.editor_config.max_sites, device=ids.device)[None]
        valid = sites < context.num_sites[:, None]
        coordinates = []
        for axis in range(3):
            positions = (context.prompt_lengths[:, None] + 8 + 4 * sites + axis).clamp_max(length - 1)
            coordinates.append(self.geometry_values[6 + axis, ids[rows, positions]])
        coords = torch.stack(coordinates, -1)
        cell_known = torch.isfinite(cell)
        coord_known = torch.isfinite(coords) & valid[..., None]
        cell = torch.where(cell_known, cell, 0.0)
        scale = cell.new_tensor([50.0, 50.0, 50.0, 180.0, 180.0, 180.0])
        cell = cell / scale
        coords = torch.where(coord_known, coords, 0.0).remainder(1.0)
        phase = 2 * torch.pi * coords
        periodic = torch.cat((phase.sin(), phase.cos()), -1)
        periodic = periodic * torch.cat((coord_known, coord_known), -1)
        return cell, cell_known.float(), periodic, coord_known.float(), valid

    def _validate(self, input_ids, context):
        if (
            context.old_token_ids.shape != input_ids.shape
            or context.active_token_mask.shape != input_ids.shape
        ):
            raise ValueError("old/current/active canvases must align")
        if any(
            value.shape != (input_ids.shape[0],)
            for value in (
                context.prompt_lengths,
                context.num_sites,
                context.task_ids,
                context.remaining_steps,
                context.reveal_fraction,
            )
        ):
            raise ValueError("one task, budget, length and site count is required per row")
        if bool(((context.task_ids < 0) | (context.task_ids > 1)).any()):
            raise ValueError("task must be G=0 or S=1")
        if (
            not bool(
                torch.isfinite(context.remaining_steps).all() & torch.isfinite(context.reveal_fraction).all()
            )
            or bool((context.remaining_steps < 0).any())
            or bool(((context.reveal_fraction < 0) | (context.reveal_fraction > 1)).any())
        ):
            raise ValueError("invalid edit budget or reveal fraction")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        *,
        edit_context: EditContext,
        detach_head_features=False,
        **kwargs,
    ):
        self._validate(input_ids, edit_context)
        self.forward_calls += 1
        old_context = self._context(edit_context, edit_context.old_token_ids)
        current_context = self._context(edit_context, input_ids)
        old_geometry = self.geometry_inputs(old_context)
        current_geometry = self.geometry_inputs(current_context)
        embeddings = self.state_embeddings(input_ids, old_context)
        current = self.current_state_conditioner(**current_geometry)
        old_cell, old_ck, old_xyz, old_xk, valid = self._raw_values(edit_context.old_token_ids, edit_context)
        new_cell, new_ck, new_xyz, new_xk, _ = self._raw_values(input_ids, edit_context)
        batch, length, hidden = embeddings.shape
        row = torch.arange(batch, device=input_ids.device)[:, None]
        cell_positions = (
            edit_context.prompt_lengths[:, None] + torch.arange(1, 7, device=input_ids.device)[None]
        )
        slots = torch.arange(self.editor_config.max_sites, device=input_ids.device)[None]
        cell_active = edit_context.active_token_mask[row, cell_positions].any(-1).float()
        cell_features = torch.cat(
            (
                old_cell,
                new_cell,
                old_ck,
                new_ck,
                old_geometry["lattice_known"][:, None].float(),
                current_geometry["lattice_known"][:, None].float(),
                cell_active[:, None],
            ),
            -1,
        )
        site_active = old_geometry["active_sites"].float()
        site_features = torch.cat(
            (
                old_xyz,
                new_xyz,
                old_xk,
                new_xk,
                site_active[..., None],
                old_geometry["lattice_known"][:, None, None].expand(-1, slots.shape[1], 1).float(),
                current_geometry["lattice_known"][:, None, None].expand(-1, slots.shape[1], 1).float(),
            ),
            -1,
        )
        task_features = torch.cat(
            (
                nn.functional.one_hot(edit_context.task_ids.long(), 2).float(),
                (edit_context.remaining_steps.float() / 32.0)[:, None],
                edit_context.reveal_fraction.float()[:, None],
            ),
            -1,
        )
        cell_delta = current["cell_embedding"] + self.raw_cell_adapter(cell_features)
        site_delta = (current["site_embeddings"] + self.raw_site_adapter(site_features)) * valid[..., None]
        residual = embeddings.new_zeros(batch, length, hidden)
        residual[row, cell_positions] = (
            cell_delta[:, None].to(embeddings.dtype).expand(-1, cell_positions.shape[1], -1).contiguous()
        )
        for offset in range(4):
            positions = (edit_context.prompt_lengths[:, None] + 7 + 4 * slots + offset).clamp_max(length - 1)
            residual = residual.scatter_add(
                1, positions[..., None].expand(-1, -1, hidden), site_delta.to(embeddings.dtype)
            )
        relative = torch.arange(length, device=input_ids.device)[None] - edit_context.prompt_lengths[:, None]
        body_mask = (relative >= 0) & (relative < 7 + 4 * edit_context.num_sites[:, None])
        residual += self.task_adapter(task_features)[:, None].to(embeddings.dtype) * body_mask[..., None]
        kwargs.pop("output_hidden_states", None)
        output = self.base_model(
            input_ids=None,
            inputs_embeds=embeddings + residual,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        if not output.hidden_states:
            raise RuntimeError("B0 must expose its final normalized hidden state")
        final = output.hidden_states[-1]
        if detach_head_features:
            final = final.detach()
        cell_hidden = final[row, cell_positions].float().mean(1)
        positions = (edit_context.prompt_lengths[:, None] + 7 + 4 * slots).clamp_max(length - 1)
        site_hidden = final[row, positions].float() * valid[..., None]
        pooled = site_hidden.sum(1) / edit_context.num_sites[:, None].clamp_min(1)
        global_hidden = torch.cat((cell_hidden, pooled), -1)
        sites = self.site_head(site_hidden).squeeze(-1).masked_fill(~valid, -1e4)
        return EditOutput(
            output.logits,
            self.mode_head(global_hidden),
            sites,
            self.count_head(global_hidden),
            self.quality_head(global_hidden),
        )

    def save_pretrained(self, output_dir, **kwargs):
        super().save_pretrained(output_dir, **kwargs)
        root = Path(output_dir)
        (root / "EXPERT_EDITOR.json").write_text(
            json.dumps({"schema": EDITOR_SCHEMA, "allowed_modes": getattr(self, "training_modes", None)})
            + "\n"
        )
        (root / "expert_edit_config.json").write_text(json.dumps(asdict(self.editor_config), indent=2) + "\n")
        torch.save(
            {name: module.state_dict() for name, module in self.extra_modules().items()},
            root / "expert_edit_modules.pt",
        )


def set_editor_trainable(model):
    counts = set_state_lora_trainable(model)
    for name, module in model.extra_modules().items():
        module.requires_grad_(True)
        counts[name] = sum(parameter.numel() for parameter in module.parameters())
    if (
        model.get_input_embeddings().weight.requires_grad
        or model.get_output_embeddings().weight.requires_grad
    ):
        raise ValueError("original B0 embedding/head tables must remain frozen")
    return counts


def materialize_edit_batch(examples, tokenizer, device, *, max_length=1024):
    lengths = [len(row["prefix"]) + len(row["old_body"]) for row in examples]
    width, batch = max(lengths), len(examples)
    if width > max_length:
        raise ValueError("editor view exceeds max length; truncation is forbidden")
    pad = int(tokenizer.pad_token_id)
    ids = torch.full((batch, width), pad, dtype=torch.long, device=device)
    old = ids.clone()
    targets = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    active = torch.zeros_like(ids, dtype=torch.bool)
    site_targets = torch.full((batch, 20), -1.0, device=device)
    quality_targets = torch.zeros(batch, 4, device=device)
    quality_mask = torch.zeros(batch, 4, dtype=torch.bool, device=device)
    for i, row in enumerate(examples):
        p, n = len(row["prefix"]), row["num_sites"]
        if len(row["old_body"]) != 7 + 4 * n or len(row["input_body"]) != len(row["old_body"]):
            raise ValueError("editor view changed its native body layout")
        ids[i, : lengths[i]] = torch.tensor(row["prefix"] + row["input_body"], device=device)
        old[i, : lengths[i]] = torch.tensor(row["prefix"] + row["old_body"], device=device)
        targets[i, p : lengths[i]] = torch.tensor(row["targets"], device=device)
        attention[i, : lengths[i]] = 1
        active[i, [p + j for j in row["active"]]] = True
        site_targets[i, :n] = torch.tensor(row["site_targets"], device=device)
        for j, value in enumerate(row["quality_targets"]):
            if value is not None:
                quality_targets[i, j], quality_mask[i, j] = float(value), True
    vector = lambda key, dtype=torch.long: torch.tensor(
        [row[key] for row in examples], dtype=dtype, device=device
    )
    context = EditContext(
        old,
        torch.tensor([len(row["prefix"]) for row in examples], device=device),
        vector("num_sites"),
        active,
        vector("task"),
        vector("remaining", torch.float32),
        vector("reveal", torch.float32),
    )
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "edit_context": context,
        "targets": targets,
        "mode_targets": vector("mode_target"),
        "count_targets": vector("count_target"),
        "site_targets": site_targets,
        "quality_targets": quality_targets,
        "quality_mask": quality_mask,
        "examples": examples,
    }


class ExpertEditObjective:
    def __init__(self, tokenizer, device, *, temperature=0.7):
        self.temperature, self.tables = temperature, {}
        for family, axes in build_geometry_token_support(tokenizer).items():
            for axis, table in axes.items():
                self.tables[(family, axis)] = (
                    torch.tensor(table["ids"], device=device),
                    torch.tensor(table["values"], device=device),
                )

    def typed_vector(self, logits, rows, positions, family, axis):
        ids, values = self.tables[(family, axis)]
        vector = logits[rows[:, None], positions[:, None], ids[None]]
        if family == "coord":
            zero, alias = int((values == 0).nonzero()[0]), int((values == 1).nonzero()[0])
            merged = torch.logaddexp(vector[:, zero], vector[:, alias])
            vector = vector.clone()
            vector[:, zero] = merged
            keep = torch.arange(len(ids), device=ids.device) != alias
            vector, ids = vector[:, keep], ids[keep]
        return vector.float() / self.temperature, ids

    def __call__(self, output, batch):
        targets, context = batch["targets"], batch["edit_context"]
        sums = output.logits.new_zeros(len(targets), dtype=torch.float32)
        counts = sums.clone()
        rows, positions = torch.nonzero(targets != -100, as_tuple=True)
        relative = positions - context.prompt_lengths[rows]
        field_sums = {
            name: output.logits.new_zeros((), dtype=torch.float32)
            for name in ("length", "angle", "coord", "first_lattice")
        }
        field_counts = dict.fromkeys(field_sums, 0)
        for kind in range(9):
            if kind < 6:
                select = relative == kind + 1
                family, axis = ("length", "ABC"[kind]) if kind < 3 else ("angle", "ABG"[kind - 3])
            else:
                select = (relative >= 8) & ((relative - 8).remainder(4) == kind - 6)
                family, axis = "coord", "XYZ"[kind - 6]
            rr, pp = rows[select], positions[select]
            if not len(rr):
                continue
            vector, ids = self.typed_vector(output.logits, rr, pp, family, axis)
            matches = targets[rr, pp, None] == ids[None]
            if not bool(matches.any(-1).all()):
                raise ValueError("corrected target is outside its typed B0 vocabulary")
            ce = nn.functional.cross_entropy(vector, matches.long().argmax(-1), reduction="none")
            field_sums[family] += ce.detach().sum()
            field_counts[family] += len(rr)
            if kind == 0:
                field_sums["first_lattice"] += ce.detach().sum()
                field_counts["first_lattice"] += len(rr)
            sums = sums.scatter_add(0, rr, ce)
            counts = counts.scatter_add(0, rr, torch.ones_like(ce))
        content = (sums / counts.clamp_min(1)).mean()

        def classification(logits, expected):
            selected = expected != -100
            if not bool(selected.any()):
                return logits.sum() * 0
            return nn.functional.cross_entropy(logits[selected], expected[selected], reduction="sum") / len(
                expected
            )

        mode = classification(output.mode_logits, batch["mode_targets"])
        count = classification(output.count_logits, batch["count_targets"])
        valid_sites = batch["site_targets"] >= 0
        site_loss = nn.functional.binary_cross_entropy_with_logits(
            output.site_logits, batch["site_targets"].clamp_min(0), reduction="none"
        )
        site = ((site_loss * valid_sites).sum(-1) / valid_sites.sum(-1).clamp_min(1)).mean()
        quality_loss = nn.functional.binary_cross_entropy_with_logits(
            output.quality_logits, batch["quality_targets"], reduction="none"
        )
        quality = (
            (quality_loss * batch["quality_mask"]).sum(-1) / batch["quality_mask"].sum(-1).clamp_min(1)
        ).mean()
        # All heads participate even when a particular rank has no labels for
        # one of them. Finite padding sentinels avoid inf*0 during this sum.
        connected_zero = sum(
            value.float().sum() * 0
            for value in (
                output.logits[..., :1],
                output.mode_logits,
                output.site_logits,
                output.count_logits,
                output.quality_logits,
            )
        )
        loss = content + 0.2 * mode + 0.1 * count + 0.2 * site + 0.3 * quality + connected_zero
        stats = {
            "content_ce": float(content.detach()),
            "mode_ce": float(mode.detach()),
            "site_bce": float(site.detach()),
            "quality_bce": float(quality.detach()),
            "supervised_tokens": int(counts.sum()),
            "loss": float(loss.detach()),
        }
        for field in field_sums:
            stats[field + "_content_ce_sum"] = float(field_sums[field])
            stats[field + "_content_tokens"] = field_counts[field]
        for task_id, name in ((0, "G"), (1, "S")):
            selected = (context.task_ids == task_id) & (counts > 0)
            stats[name + "_content_views"] = int(selected.sum())
            stats[name + "_content_ce_sum"] = float((sums[selected] / counts[selected]).sum().detach())
            auxiliary = torch.tensor(
                [
                    row.get("source_kind") == "mp20_geometry_auxiliary"
                    for row in batch.get("examples", [{}] * len(targets))
                ],
                device=counts.device,
                dtype=torch.bool,
            )
            for origin, choose in (("real", ~auxiliary), ("aux", auxiliary)):
                subset = selected & choose
                stats[name + "_" + origin + "_content_views"] = int(subset.sum())
                stats[name + "_" + origin + "_content_ce_sum"] = float(
                    (sums[subset] / counts[subset]).sum().detach()
                )
        return loss, stats


def inference_view(prefix, old, current, n, task, active=(), *, remaining=160, reveal=0.0):
    """An inference input constructed entirely from the current structure."""
    return {
        "prefix": list(prefix),
        "old_body": list(old),
        "input_body": list(current),
        "targets": [-100] * len(old),
        "active": list(active),
        "num_sites": n,
        "task": task,
        "remaining": remaining,
        "reveal": reveal,
        "mode_target": -100,
        "site_targets": [-1.0] * n,
        "count_target": -100,
        "quality_targets": [None] * 4,
        "kind": "inference",
        "record_id": "",
        "ancestor_id": "",
    }
