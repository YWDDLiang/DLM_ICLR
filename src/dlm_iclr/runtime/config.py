"""One configuration for data, training, sampling and evaluation."""

from copy import deepcopy
import json
import os
from pathlib import Path
from importlib.resources import files


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def validate_keys(config, defaults):
    """Reject unsupported sections and model roles instead of silently ignoring them."""
    unknown = set(config) - set(defaults) - {"_config_dir"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
    for section in ("models", "feedback"):
        if section in config:
            unknown = set(config[section]) - set(defaults[section])
            if unknown:
                raise ValueError(f"Unknown {section} settings: {', '.join(sorted(unknown))}")
    if "training" in config.get("feedback", {}):
        unknown = set(config["feedback"]["training"]) - set(defaults["feedback"]["training"])
        if unknown:
            raise ValueError(f"Unknown feedback training settings: {', '.join(sorted(unknown))}")


def load(path=None, overrides=(), *, dataset=None):
    defaults = json.loads(files("dlm_iclr").joinpath("defaults.json").read_text(encoding="utf-8"))
    from .datasets import profile, canonical_name, activate
    supplied = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    validate_keys(supplied, defaults)
    name = canonical_name(dataset or supplied.get("dataset", {}).get("name", "mp20"))
    preset = profile(name) if dataset or name in ("mp20", "perov-5", "mpts-52") else {}
    config = merge(merge(defaults, preset), supplied)
    config["dataset"]["name"] = name
    if dataset:
        old_name = canonical_name(supplied.get("dataset", {}).get("name", name))
        if old_name != name:
            raise ValueError("--dataset conflicts with --config; use the matching dataset config")
        config["dataset"].update({k: v for k, v in preset["dataset"].items() if k != "splits"})
    for assignment in overrides:
        key, raw = assignment.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = config
        *parents, leaf = key.split(".")
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = value
    validate_keys(config, defaults)
    # Paper runs set feedback.physical_rollback=False; learned KEEP remains enabled.
    for option in ("protect_sun", "physical_rollback"):
        if type(config["feedback"][option]) is not bool:
            raise ValueError(f"feedback.{option} must be a JSON boolean (true or false)")
    activate(config)
    config["_config_dir"] = str(Path(path).resolve().parent if path else Path.cwd())
    return config


def path(config, value):
    """Resolve a dataset/output path relative to the user configuration."""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    result = Path(expanded)
    return result if result.is_absolute() else Path(config["_config_dir"]) / result


def run_root(config):
    return path(config, config["output"])


def asset(config, name):
    value = config["models"][name]
    if value.startswith("@run/"):
        return str(run_root(config) / value[5:])
    if value.startswith(("hf:",)):
        return value[3:]
    return str(path(config, value)) if value else ""


def backend_config(config, stage="feedback"):
    from .base_config import Assets, Config, Inference, Training

    cfg = Config()
    cfg.assets = Assets(
        base_model=asset(config, "dlm"),
        generator=asset(config, "constructor"),
        editor=asset(config, "feedback"),
        value=asset(config, "verifier"),
        refiner=asset(config, "diffusion"),
        planner_base=asset(config, "planner_base"),
        planner=asset(config, "planner"),
        chgnet=asset(config, "chgnet"),
        hull_cache=str(run_root(config) / "hull"),
        novelty_reference=str(run_root(config) / "data/structures/train.jsonl"),
    )
    cfg.inference = Inference(
        temperature=config["periodic" if stage == "periodic" else "feedback"]["temperature"],
        refiner_steps=config["diffusion"]["steps"],
        editor_candidates=config["feedback"]["candidates"],
        editor_max_calls=config["feedback"]["max_calls"],
        editor_batch_size=config["feedback"]["batch_size"],
    )
    if stage == "constructor":
        cfg.inference.temperature = config["constructor"].get("temperature", 0.7)
    t = config["feedback"]["training"]
    cfg.training = Training(
        plans=t["plans"],
        seed=t["seed"],
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        editor_content_lr=t["content_lr"],
        editor_heads_lr=t["heads_lr"],
        reference_kl_weight=t["reference_kl_weight"],
        value_epochs=t["value_epochs"],
        value_sources_per_batch=t["value_sources_per_batch"],
        value_head_lr=t["value_head_lr"],
        value_hidden_lr=t["value_hidden_lr"],
        teacher_mix=t["teacher_mix"],
        permute_atoms=t["permute_atoms"],
    )
    return cfg
