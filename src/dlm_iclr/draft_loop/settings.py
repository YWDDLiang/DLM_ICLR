"""Single fixed training recipe. These defaults are engineering starting points, not fitted results."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from .common import read_json

DEFAULTS = {
    "seed": 20260918,
    "max_rounds": 2,
    "plans_per_round": 256,
    "planner_oversample": 2,
    "max_planner_batches": 2,
    "planner_temperature": 0.9,
    "max_same_composition_per_round": 4,
    "development_sources": 48,
    "development_seed": 4317,
    "replay_fraction": 0.5,
    "history_fraction": 0.25,
    "teacher_steps": 800,
    "teacher_times": [20],
    "quantized_variants": 3,
    "teacher_shortlist": 2,
    "max_teachers_per_source": 1,
    "include_relaxed_teacher": True,
    "require_hull": True,
    "refresh_hull": False,
    "min_pairs": 16,
    "min_teachers": 24,
    "max_stalled_rounds": 1,
    "min_rounds_before_stopping": 1,
    "max_collection_stalls": 1,
    "runtime": {
        "devices": [], "generation_workers_per_device": 1,
        "physics_workers_per_device": 1, "teacher_batch_size": 1,
    },
    "evaluation": {"requests":1000,"seed":20260919,"refine_steps":[100,800],"direct_workers":32},
    "quality": {
        "energy_slack": 0.005, "energy_gain": 0.02,
        "force_slack": 0.05, "force_gain": 0.1,
        "stress_slack": 0.25, "stress_gain": 0.5,
        "terminal_hull_slack": 0.01, "max_teacher_terminal_hull": 0.1,
        "anchor_force": 0.5, "anchor_stress": 2.0,
        "max_unknown_fraction": 0.4,
    },
    "training": {
        "updates": 128, "effective_batch_size": 8, "learning_rate": 1e-6,
        "micro_batch_size": 4, "c1_batch_size": 16,
        "preference_weight": 0.1, "preference_beta": 0.5,
        "preference_warmup_updates": 24, "reference_kl": 0.1,
        "mask_min": 0.15, "mask_max": 0.85, "temperature": 0.7,
        "save_every": 16, "max_length": 1024,
        "c1_updates": 64, "c1_learning_rate": 5e-5,
    },
    "verifier": {
        "sources": 32, "candidate_samples": 3, "alternatives_per_source": 1,
        "max_remaining": 8, "axis": 2, "projection_width": 32,
        "ridge": 3.0, "min_train_sources": 16, "min_validation_sources": 6,
        "min_pair_accuracy": 0.55, "strength": 1.0, "utility_clip": 2.0,
    },
    "gate": {
        "max_invalid_increase": 0, "min_mean_gain": 0.01,
        "max_force_ratio": 1.02, "max_stress_ratio": 1.05,
        "max_hull_mean_increase": 0.005, "min_common_fraction": 0.8,
        "min_verified_hull_fraction": 0.25,
    },
    "budget": {
        "max_wall_seconds": 172800, "max_planner_draws": 4096,
        "max_drafts": 1800, "max_teacher_calls": 512,
        "max_physics_records": 12000,
    },
}


def merge(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in update.items():
        if key not in out:
            raise ValueError(f"Unknown loop setting: {key}")
        out[key] = merge(out[key], value) if isinstance(out[key], dict) else value
    return out


def load_loop(path: str | Path) -> dict:
    settings = merge(DEFAULTS, read_json(path))
    for key in ("max_rounds", "plans_per_round", "development_sources", "max_planner_batches"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be positive integer")
    for key in ("replay_fraction", "history_fraction"):
        if not 0 <= settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if settings["replay_fraction"] + settings["history_fraction"] >= 1:
        raise ValueError("The recipe must reserve probability for newly verified teachers")
    return settings
