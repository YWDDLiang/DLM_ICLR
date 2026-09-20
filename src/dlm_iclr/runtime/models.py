"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import os
import time
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from dlm_iclr._core.dynamic_crystal import Z_TO_SYMBOL
from dlm_iclr._core.fixed_slot import FixedSlotConfig
from dlm_iclr._core.llada_resize import ensure_llada_vocab_size
from dlm_iclr._core.transformers_compat import ensure_create_bidirectional_mask, ensure_llada2_rope_parameters


def model_class_for(config):
    return AutoModelForCausalLM if getattr(config, "model_type", None) == "llada2_moe" else AutoModel


def load_model_and_tokenizer(
    base_model_path: str, checkpoint_path: Optional[str], device: torch.device, *, mean_resizing=True
):
    """Optionally serialize GPU loading without changing model or request seeds."""
    device = torch.device(device)
    lock_root = os.environ.get('DLM_MODEL_LOAD_LOCK_DIR')
    if lock_root and device.type == 'cuda':
        from filelock import FileLock
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        index = device.index if device.index is not None else 0
        physical = visible[index] if len(visible) > index and visible[index] else str(device)
        key = hashlib.sha256(physical.encode()).hexdigest()[:16]
        folder = Path(lock_root); folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder/f'gpu_{key}.lock'), timeout=600):
            result = _load_model_and_tokenizer(base_model_path, checkpoint_path, device, mean_resizing=mean_resizing)
            # Let the existing reservation monitor account for this worker
            # before the next model on the same device begins a loading burst.
            time.sleep(float(os.environ.get('DLM_MODEL_LOAD_SETTLE_SECONDS', '2')))
            return result
    return _load_model_and_tokenizer(base_model_path, checkpoint_path, device, mean_resizing=mean_resizing)


def _load_model_and_tokenizer(
    base_model_path: str, checkpoint_path: Optional[str], device: torch.device, *, mean_resizing=True
):
    tokenizer_source = (
        checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if checkpoint_path and (Path(checkpoint_path) / "adapter_config.json").exists():
        from peft import PeftModel

        ensure_create_bidirectional_mask()
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            base_model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32,
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=mean_resizing)
        ensure_llada_vocab_size(model, len(tokenizer))
        # Optional resource-only staging: the full model is moved below as
        # before, avoiding simultaneous temporary adapter copies on the GPU.
        loading = {'torch_device': 'cpu'} if os.environ.get('DLM_MODEL_LOAD_CPU_ADAPTER') == '1' else {}
        model = PeftModel.from_pretrained(model, checkpoint_path, **loading)
    elif checkpoint_path:
        ensure_create_bidirectional_mask()
        model_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            checkpoint_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32,
        )
    else:
        ensure_create_bidirectional_mask()
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            base_model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32,
        )
    model.to(device).eval()
    return model, tokenizer


def _required_token_ids(tokenizer, tokens: List[str]) -> List[int]:
    vocab = tokenizer.get_vocab()
    missing = [token for token in tokens if token not in vocab]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required dynamic crystal tokens: {missing[:10]}")
    return [int(vocab[token]) for token in tokens]


def atom_count_token_id(tokenizer, atom_count: int) -> int:
    token = f"<N_{atom_count:03d}>"
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise RuntimeError(f"Tokenizer is missing atom-count token {token}")
    return int(vocab[token])


def build_dynamic_lightweight_constraints(
    tokenizer,
    *,
    duplicate_coordinate_mask: bool,
    lattice_volume_mask: bool,
    min_lattice_rad: float,
    canonicalize_periodic_alias: bool = False,
    pbc_min_distance_mask: bool = False,
    pbc_min_distance_A: float = 0.5,
    pbc_image_radius: int = 2,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Dict[str, Any] | None:
    if not (
        duplicate_coordinate_mask
        or lattice_volume_mask
        or canonicalize_periodic_alias
        or pbc_min_distance_mask
    ):
        return None
    if float(pbc_min_distance_A) <= 0.0:
        raise ValueError("pbc_min_distance_A must be positive")
    if int(pbc_image_radius) not in (1, 2):
        raise ValueError("pbc_image_radius must be one (27) or two (125)")
    vocab = tokenizer.get_vocab()
    coord_token_to_bin = {
        axis: {
            int(vocab[f"<{axis}_{i:03d}>"]): i for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        }
        for axis in ("X", "Y", "Z")
    }
    angle_token_to_bin = {
        prefix: {
            int(vocab[f"<{prefix}_{i:03d}>"]): i
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        }
        for prefix in ("AA", "AB", "AG")
    }
    length_token_to_bin = {
        prefix: {
            int(vocab[f"<{prefix}_{i:03d}>"]): i
            for i in range(config.length_min_bin, config.length_max_bin + 1)
        }
        for prefix in ("LA", "LB", "LC")
    }
    return {
        "representation": "dynamic_v1",
        "duplicate_coordinate_mask": bool(duplicate_coordinate_mask),
        "lattice_volume_mask": bool(lattice_volume_mask),
        "canonicalize_periodic_alias": bool(canonicalize_periodic_alias),
        "pbc_min_distance_mask": bool(pbc_min_distance_mask),
        "pbc_min_distance_A": float(pbc_min_distance_A),
        "pbc_image_radius": int(pbc_image_radius),
        "min_lattice_rad": float(min_lattice_rad),
        "max_atoms": config.max_atoms,
        "coord_period": config.coord_max_bin - config.coord_min_bin,
        "count_token_to_n": {
            atom_count_token_id(tokenizer, atom_count): atom_count
            for atom_count in range(1, config.max_atoms + 1)
        },
        "coord_token_to_bin": coord_token_to_bin,
        "coord_bin_to_token_id": {
            axis: {
                i: int(vocab[f"<{axis}_{i:03d}>"])
                for i in range(config.coord_min_bin, config.coord_max_bin + 1)
            }
            for axis in ("X", "Y", "Z")
        },
        "coordinate_alias_token_ids": {
            axis: (
                int(vocab[f"<{axis}_{config.coord_min_bin:03d}>"]),
                int(vocab[f"<{axis}_{config.coord_max_bin:03d}>"]),
            )
            for axis in ("X", "Y", "Z")
        },
        "length_token_to_bin": length_token_to_bin,
        "length_step": float(config.length_step),
        "z_bin_to_token_id": {
            i: int(vocab[f"<Z_{i:03d}>"]) for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        },
        "angle_token_to_bin": angle_token_to_bin,
        "gamma_bin_to_token_id": {
            i: int(vocab[f"<AG_{i:03d}>"]) for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        },
        "zero_length_token_ids_by_position": {
            1: int(vocab["<LA_000>"]),
            2: int(vocab["<LB_000>"]),
            3: int(vocab["<LC_000>"]),
        },
    }


def load_editor(base_model, checkpoint, device, *, trainable=False):
    """Load ordinary model files without operational receipts or machine paths."""
    from dlm_iclr._core.expert_edit import ExpertEditConfig, ExpertEditDLM, set_editor_trainable
    from dlm_iclr.runtime.io import read_json

    root = Path(checkpoint)
    base, tokenizer = load_model_and_tokenizer(str(base_model), str(checkpoint), device, mean_resizing=False)
    config_path = root / "expert_edit_config.json"
    config = (
        ExpertEditConfig(**read_json(config_path))
        if config_path.exists()
        else ExpertEditConfig(hidden_size=base.get_input_embeddings().weight.shape[1])
    )
    model = ExpertEditDLM(base, tokenizer, config).to(device)
    if config_path.exists():
        model.load_state_conditioner(root)
        saved = torch.load(root / "expert_edit_modules.pt", map_location="cpu", weights_only=True)
        for name, module in model.extra_modules().items():
            module.load_state_dict(saved[name], strict=True)
    if trainable:
        set_editor_trainable(model)
    else:
        model.requires_grad_(False)
    return model.eval(), tokenizer
