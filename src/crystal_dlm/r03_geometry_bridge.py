"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.lattice_geometry import lattice_angle_rad
from crystal_dlm.llada_generation import _apply_lightweight_decoding_masks, _lattice_matrix_from_token_ids
from crystal_dlm.r03_physics_transfer import build_repair_constraints


SCHEMA = "r03_construction_geometry_bridge_v1"


PROTOCOL = {
    "periodic_alias": "000_100_logaddexp_to_000",
    "minimum_inter_site_distance_A": 0.5,
    "image_radius": 2,
    "periodic_image_count": 125,
    "min_lattice_rad": 1e-4,
    "reveal_order": "unchanged_R03_lattice_X_Y_Z",
    "sampling_noise": "unchanged_frozen_paired_suffix_candidates",
    "self_image_hard_mask_added": False,
    "lattice_system_spacegroup_volume_bin_hard_rules_added": False,
    "replacement_or_retry": False,
}


class GeometryBridgeContractError(RuntimeError):
    """An ABI, schedule or missing-context defect; not a scientific failure."""


class GeometryNoLegalSupport(RuntimeError):
    """One batch-1 request has no admissible construction continuation."""

    def __init__(self, details: Mapping[str, Any], canvas: torch.Tensor, prompt_length: int):
        self.details = dict(details)
        self.partial_canvas = canvas.detach().cpu().clone()
        self.prompt_length = int(prompt_length)
        super().__init__(str(self.details["reason"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.details,
            "schema": SCHEMA,
            "failure_class": "construction_constraint_no_legal_support",
            "prompt_length": self.prompt_length,
            "partial_canvas_token_ids": self.partial_canvas.tolist(),
            "partial_body_token_ids": self.partial_canvas[:, self.prompt_length :].tolist(),
            "retry_or_replacement_used": False,
        }


def _stage(position: int) -> str:
    if position == 0 or (position >= 7 and (position - 7) % 4 == 0):
        return "prefilled_composition"
    if 1 <= position <= 6:
        return "lattice"
    return "XYZ"[(position - 8) % 4]


def _checked_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(int(position) for position in group) for group in groups)
    flattened = [position for group in normalized for position in group]
    length = len(flattened)
    if (
        length < 11
        or (length - 7) % 4
        or not 1 <= (length - 7) // 4 <= 20
        or sorted(flattened) != list(range(length))
        or any(not group for group in normalized)
    ):
        raise GeometryBridgeContractError("R03 groups must cover exact 7+4N positions once")
    stages = []
    for group in normalized:
        kinds = {_stage(position) for position in group}
        if len(kinds) != 1:
            raise GeometryBridgeContractError("geometry bridge cannot change or mix R03 axis groups")
        stages.append(next(iter(kinds)))
    order = {name: rank for rank, name in enumerate(("prefilled_composition", "lattice", "X", "Y", "Z"))}
    if stages[:2] != ["prefilled_composition", "lattice"] or [order[s] for s in stages] != sorted(
        order[s] for s in stages
    ):
        raise GeometryBridgeContractError(
            "the frozen R03 lattice-to-X-to-Y-to-Z dependency order is required"
        )
    return normalized


class ConstructionGeometryMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        tokenizer: Any,
        generation_position_groups: Sequence[Sequence[int]],
        native_constraints: Mapping[str, Any] | None,
        mask_id: int = MASK_TOKEN_ID,
        relax_final_z: bool = False,
    ):
        self.enabled = bool(enabled)
        self.relax_final_z = bool(relax_final_z)
        self.mask_id = int(mask_id)
        self.events: list[dict[str, Any]] = []
        self.candidate_calls = 0
        self.masked_tokens = 0
        self.alias_vectors = 0
        self.failed = False
        self.original_candidate = None
        self.groups: tuple[tuple[int, ...], ...] = ()
        self.constraints = None
        if not self.enabled:
            return
        self.groups = _checked_groups(generation_position_groups)
        required = {
            "representation": "dynamic_v1",
            "max_atoms": 20,
            "coord_period": 100,
            "duplicate_coordinate_mask": True,
            "lattice_volume_mask": True,
            "min_lattice_rad": 1e-4,
        }
        if not isinstance(native_constraints, Mapping) or any(
            native_constraints.get(k) != v for k, v in required.items()
        ):
            raise GeometryBridgeContractError(
                "original R03 schema, duplicate and nondegenerate-lattice guards must stay enabled"
            )
        # A pure tokenizer-map builder. No old physical checkpoint or dataset
        # is loaded by this shared helper.
        self.constraints = build_repair_constraints(tokenizer)
        for key in (
            "count_token_to_n",
            "coord_token_to_bin",
            "z_bin_to_token_id",
            "angle_token_to_bin",
            "gamma_bin_to_token_id",
        ):
            if native_constraints.get(key) != self.constraints.get(key):
                raise GeometryBridgeContractError(f"native R03 token map differs at {key}")
        # The frozen constructor has already applied these two original masks.
        # Do not touch logits outside the active group a second time.
        self.constraints = {
            **self.constraints,
            "lattice_volume_mask": False,
            "duplicate_coordinate_mask": False,
        }
        self.element_ids = {
            int(value)
            for token, value in tokenizer.get_vocab().items()
            if token.startswith("<E_") and token.endswith(">")
        }
        self.native_constraints_sha256 = hashlib.sha256(
            json.dumps(native_constraints, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def report(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "protocol": dict(PROTOCOL),
            "candidate_calls": self.candidate_calls,
            "active_alias_vectors": self.alias_vectors,
            "final_Z_distance_support_relaxed": self.relax_final_z,
            "newly_masked_legal_tokens": self.masked_tokens,
            "failed": self.failed,
            "events": list(self.events),
            "weights_changed": False,
            "prompt_changed": False,
            "composition_changed": False,
            "schedule_changed": False,
            "paired_noise_changed": False,
        }

    def _fail(self, reason, x, prompt_length, semantic_group, step, active, *, positions=None):
        self.failed = True
        details = {
            "reason": reason,
            "row_indices": [0],
            "semantic_group": int(semantic_group),
            "step_in_group": int(step),
            "group_positions": list(self.groups[semantic_group]),
            "active_positions": active[0].nonzero(as_tuple=False).flatten().tolist(),
            "failure_positions": list(positions or ()),
        }
        self.events.append({**details, "status": "no_legal_support"})
        raise GeometryNoLegalSupport(details, x, prompt_length)

    def apply_logits(
        self,
        logits: torch.Tensor,
        *,
        current_tokens: torch.Tensor,
        prompt_length: int,
        gen_length: int,
        semantic_group: int,
        step_in_group: int,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                current_tokens[:, prompt_length:].shape, dtype=torch.bool, device=current_tokens.device
            )
        x = current_tokens
        if (
            x.ndim != 2
            or x.shape[0] != 1
            or logits.ndim != 3
            or logits.shape[:2] != x.shape
            or prompt_length < 0
            or x.shape[1] != prompt_length + gen_length
            or gen_length != sum(map(len, self.groups))
            or not logits.is_floating_point()
            or not 0 <= semantic_group < len(self.groups)
            or step_in_group < 0
        ):
            raise GeometryBridgeContractError(
                "geometry ON requires one exact R03 request and the original candidate-logit ABI"
            )
        body = x[:, prompt_length:]
        n = self.constraints["count_token_to_n"].get(int(body[0, 0]))
        if n is None or gen_length != 7 + 4 * n:
            raise GeometryBridgeContractError("R03 N prefill is missing or its cardinality changed")
        if any(int(body[0, 7 + 4 * slot]) not in self.element_ids for slot in range(n)):
            raise GeometryBridgeContractError(
                "R03 element prefill is missing or outside the unchanged vocabulary"
            )
        active = torch.zeros_like(body, dtype=torch.bool)
        positions = list(self.groups[semantic_group])
        active[:, positions] = body[:, positions] == self.mask_id
        if not bool(active.any()):
            raise GeometryBridgeContractError(
                "frozen candidate hook was called without an active masked position"
            )
        stage = _stage(positions[0])
        if stage == "prefilled_composition":
            raise GeometryBridgeContractError(
                "composition must remain prefilled and cannot be sampled by this bridge"
            )
        if stage in "XYZ":
            for offset, name in ((1, "LA"), (2, "LB"), (3, "LC"), (4, "AA"), (5, "AB"), (6, "AG")):
                mapping = self.constraints["length_token_to_bin" if offset < 4 else "angle_token_to_bin"][
                    name
                ]
                if int(body[0, offset]) not in mapping:
                    raise GeometryBridgeContractError(
                        "active coordinates require the completed, valid-token lattice"
                    )
            angles = [
                self.constraints["angle_token_to_bin"][name][int(body[0, offset])]
                for offset, name in ((4, "AA"), (5, "AB"), (6, "AG"))
            ]
            lattice = _lattice_matrix_from_token_ids(
                x[0], prompt_length=prompt_length, constraints=self.constraints
            )
            if (
                lattice_angle_rad(*angles) <= PROTOCOL["min_lattice_rad"]
                or lattice is None
                or not bool(torch.isfinite(lattice).all())
            ):
                self._fail(
                    "nondegenerate_lattice_has_no_legal_continuation",
                    x,
                    prompt_length,
                    semantic_group,
                    step_in_group,
                    active,
                )
            for slot in range(n):
                for component, axis in enumerate("XYZ"):
                    token = int(body[0, 8 + 4 * slot + component])
                    if token != self.mask_id and token not in self.constraints["coord_token_to_bin"][axis]:
                        raise GeometryBridgeContractError(
                            "visible coordinate is outside the unchanged R03 coordinate vocabulary"
                        )
                    if stage == "Z" and axis in "XY" and token == self.mask_id:
                        raise GeometryBridgeContractError(
                            "active R03 Z requires every X and Y to be already visible"
                        )
        suffix = logits[:, prompt_length : prompt_length + gen_length]
        before = suffix[active].clone()
        if bool(torch.isnan(before).any() or torch.isposinf(before).any()):
            raise GeometryBridgeContractError("active model logits are nonfinite before geometry support")
        self.candidate_calls += 1
        if stage in "XYZ":
            self.alias_vectors += int(active.sum())
            support = dict(self.constraints)
            if stage == "Z" and self.relax_final_z:
                support["pbc_min_distance_mask"] = False
            report = _apply_lightweight_decoding_masks(
                logits, x, prompt_length, gen_length, support, active, self.mask_id
            )
            empty_pbc = sorted(position for row, position in report["pbc_no_legal_completion"] if row == 0)
            if empty_pbc:
                self._fail(
                    "pbc_no_legal_completion",
                    x,
                    prompt_length,
                    semantic_group,
                    step_in_group,
                    active,
                    positions=empty_pbc,
                )
        after = suffix[active]
        minimum = torch.finfo(logits.dtype).min
        legal_before = torch.isfinite(before) & (before > minimum)
        legal_after = torch.isfinite(after) & (after > minimum)
        if bool(torch.isnan(after).any() or torch.isposinf(after).any()):
            raise GeometryBridgeContractError("active geometry logits became nonfinite")
        empty = (legal_after.sum(dim=-1) == 0).nonzero(as_tuple=False).flatten().tolist()
        active_positions = active[0].nonzero(as_tuple=False).flatten().tolist()
        if empty:
            self._fail(
                "empty_active_legal_support",
                x,
                prompt_length,
                semantic_group,
                step_in_group,
                active,
                positions=[active_positions[index] for index in empty],
            )
        removed = int((legal_before & ~legal_after).sum())
        self.masked_tokens += removed
        self.events.append(
            {
                "status": "supported",
                "semantic_group": int(semantic_group),
                "step_in_group": int(step_in_group),
                "stage": stage,
                "active_positions": active_positions,
                "Z_distance_support_relaxed": stage == "Z" and self.relax_final_z,
                "legal_tokens_before": legal_before.sum(dim=-1).tolist(),
                "legal_tokens_after": legal_after.sum(dim=-1).tolist(),
                "newly_masked_legal_tokens": removed,
            }
        )
        return active
