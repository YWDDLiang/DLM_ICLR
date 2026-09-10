"""Learned KEEP/EDIT values. Physical labels are absent from this module's API."""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from crystal_dlm.continuous_keep_edit import structure_of
from crystal_dlm.expert_edit import inference_view, materialize_edit_batch

GEOMETRY_FEATURES = (
    "sites_scaled",
    "changed_site_fraction",
    "changed_token_fraction",
    "cell_changed",
    "fractional_displacement_rms",
    "fractional_displacement_max",
    "cartesian_displacement_rms",
    "cartesian_displacement_max",
    "geometry_delta_available",
)


def geometry_features(current, proposal, positions, num_sites):
    sites = {(position - 8) // 4 for position in positions if position >= 8}
    values = [
        num_sites / 20,
        len(sites) / num_sites,
        len(positions) / (6 + 3 * num_sites),
        float(any(position < 7 for position in positions)),
    ]
    try:
        a, b = structure_of(current), structure_of(proposal)
        if [str(site.specie) for site in a] != [str(site.specie) for site in b]:
            raise ValueError("site order changed")
        delta = np.asarray(b.frac_coords) - np.asarray(a.frac_coords)
        delta -= np.round(delta)
        frac = np.linalg.norm(delta, axis=1)
        cart = np.linalg.norm(delta @ np.asarray(a.lattice.matrix), axis=1)
        values += [
            float(np.sqrt(np.mean(frac**2))),
            float(frac.max()),
            float(np.sqrt(np.mean(cart**2))) / 10,
            float(cart.max()) / 10,
            1.0,
        ]
    except (ValueError, KeyError, TypeError):
        values += [0.0] * 5
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite geometry feature")
    return values


class ValueNetwork(nn.Module):
    def __init__(self, raw_features=8192, hidden_width=128, geometry_width=9):
        super().__init__()
        self.hidden = nn.Linear(raw_features, hidden_width)
        self.head = nn.Linear(hidden_width + geometry_width, 2)
        self.register_buffer("mean", torch.zeros(hidden_width + geometry_width))
        self.register_buffer("scale", torch.ones(hidden_width + geometry_width))

    def features(self, raw, geometry):
        return torch.cat((torch.nn.functional.silu(self.hidden(raw)), geometry), dim=1)

    def forward(self, raw, geometry):
        return self.head((self.features(raw, geometry) - self.mean) / self.scale)

    def save(self, path, *, metadata=None):
        torch.save(
            {
                "schema": "autonomous_crystal_value_v1",
                "state_dict": self.state_dict(),
                "raw_features": self.hidden.in_features,
                "hidden_width": self.hidden.out_features,
                "geometry_width": len(GEOMETRY_FEATURES),
                "metadata": metadata or {},
            },
            path,
        )


def load_value(path="bundled", device="cpu"):
    if str(path) == "bundled":
        path = Path(__file__).parent / "assets/autonomous_value.pt"
    saved = torch.load(path, map_location=device, weights_only=True)
    model = ValueNetwork(saved["raw_features"], saved["hidden_width"], saved["geometry_width"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval()


@torch.no_grad()
def extract_features(editor, tokenizer, rows, device, *, batch_size=16):
    """Match the trained task=1, remaining=80, reveal=1 feature view exactly."""
    captured, features = [], []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().float().cpu())

    hook = editor.quality_head.register_forward_pre_hook(capture)
    try:
        for start in range(0, len(rows), batch_size):
            views = [
                inference_view(
                    tokenizer(row["prompt"], add_special_tokens=False)["input_ids"],
                    row["current_tokens"],
                    row["proposal_tokens"],
                    row["num_sites"],
                    1,
                    row["action_positions"],
                    remaining=80,
                    reveal=1.0,
                )
                for row in rows[start : start + batch_size]
            ]
            batch = materialize_edit_batch(views, tokenizer, device)
            editor(
                batch["input_ids"], attention_mask=batch["attention_mask"], edit_context=batch["edit_context"]
            )
            if len(captured) != 1:
                raise RuntimeError("Expected one feature tensor from the editor forward")
            features.append(captured.pop())
    finally:
        hook.remove()
    return torch.cat(features) if features else torch.empty((0, 2 * editor.editor_config.hidden_size))


def choose(values, *, sun_weight=2.0, msun_weight=1.0):
    """KEEP has zero gain. Utility ties use SUN gain, then KEEP, then input order."""
    winner, key = None, (0.0, 0.0, True)
    for index, gain in enumerate(values):
        if gain is None:
            continue
        if not all(math.isfinite(float(value)) for value in gain):
            raise ValueError("nonfinite model value")
        candidate_key = (sun_weight * gain[0] + msun_weight * gain[1], gain[0], False)
        if candidate_key > key:
            key, winner = candidate_key, index
    return winner
