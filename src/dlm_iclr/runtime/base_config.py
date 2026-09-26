"""Portable configuration. Relative asset paths resolve beside the config file."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
from dlm_iclr.runtime.io import read_json


@dataclass
class Assets:
    base_model: str = "GSAI-ML/LLaDA-8B-Instruct"
    generator: str = ""
    editor: str = ""
    value: str = ""
    refiner: str = ""
    planner_base: str = ""
    planner: str = ""
    chgnet: str = ""
    hull_cache: str = ""
    novelty_reference: str = ""


@dataclass
class Inference:
    plan_source: str = "mp20_default"
    requests: int | None = 1000
    legal_only: bool = False
    seed: int = 17
    planner_requests: int = 1000
    refiner_steps: int = 800
    editor_candidates: int = 8
    editor_max_calls: int = 80
    editor_batch_size: int = 64
    temperature: float = 0.7
    geometry_monitor: bool = True
    construction_recovery: bool = True
    adaptive_lattice_recovery: bool = True
    sun_weight: float = 2.0
    msun_weight: float = 1.0


@dataclass
class Training:
    plans: str = "@run/data/plans/train.jsonl"
    seed: int = 20260911
    epochs: int = 8
    batch_size: int = 16
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
