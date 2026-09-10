"""Portable configuration. Relative asset paths resolve beside the config file."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
from .io import read_json


@dataclass
class Assets:
    base_model: str = "GSAI-ML/LLaDA-8B-Instruct"
    generator: str = ""
    editor: str = ""
    value: str = "bundled"
    refiner: str = ""
    planner_base: str = ""
    planner: str = ""
    chgnet: str = ""
    hull_cache: str = ""
    novelty_reference: str = ""


@dataclass
class Inference:
    plan_source: str = "H1A2_1200"
    requests: int | None = None
    legal_only: bool = False
    seed: int = 17
    planner_requests: int = 1200
    refiner_steps: int = 800
    editor_candidates: int = 8
    editor_max_calls: int = 80
    editor_batch_size: int = 64
    temperature: float = 0.7
    construction_recovery: bool = True
    adaptive_lattice_recovery: bool = True
    sun_weight: float = 2.0
    msun_weight: float = 1.0


@dataclass
class Training:
    plans: str = "CLEAN_TRAIN_1000"
    rounds: int = 3
    seed: int = 20260911
    epochs: int = 8
    batch_size: int = 16
    generator_lr: float = 5e-6
    generator_epochs: int = 4
    generator_batch_size: int = 32
    generator_beta: float = 0.1
    generator_anchor_weight: float = 0.2
    mask_cuts: int = 2
    editor_content_lr: float = 2e-6
    editor_heads_lr: float = 1e-4
    reference_kl_weight: float = 1.0
    value_epochs: int = 64
    value_sources_per_batch: int = 32
    value_head_lr: float = 1e-3
    value_hidden_lr: float = 1e-5
    teacher_mix: float = 0.5
    permute_atoms: bool = True


@dataclass
class Config:
    assets: Assets = field(default_factory=Assets)
    inference: Inference = field(default_factory=Inference)
    training: Training = field(default_factory=Training)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    path = Path(path).resolve()
    raw = read_json(path)
    unexpected = set(raw) - {"assets", "inference", "training"}
    if unexpected:
        raise ValueError(f"Unknown configuration sections: {sorted(unexpected)}")
    config = Config(
        Assets(**raw.get("assets", {})),
        Inference(**raw.get("inference", {})),
        Training(**raw.get("training", {})),
    )
    for name, value in vars(config.assets).items():
        if not value or value == "bundled":
            continue
        expanded = os.path.expandvars(os.path.expanduser(value))
        if expanded.startswith(("./", "../")):
            expanded = str((path.parent / expanded).resolve())
        setattr(config.assets, name, expanded)
    for obj, name in ((config.inference, "plan_source"), (config.training, "plans")):
        value = getattr(obj, name)
        if value.startswith(("./", "../")):
            setattr(obj, name, str((path.parent / value).resolve()))
    policy = config.inference
    if not 1 <= policy.editor_candidates <= 8 or policy.editor_max_calls < 1:
        raise ValueError("editor_candidates must be 1..8 and editor_max_calls must be positive")
    if not 1 <= policy.refiner_steps <= 1000:
        raise ValueError("refiner_steps must be within the trained 1000-step noise schedule")
    return config
