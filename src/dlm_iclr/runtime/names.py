"""Canonical scientific names and readers for earlier configuration files."""
from copy import deepcopy

MODULE_NAMES = {"b0": "constructor", "c1": "periodic", "c2": "feedback"}
MODEL_NAMES = {**MODULE_NAMES, "value": "verifier"}
FEEDBACK_STAGES = {"editor": "reconstruction", "light": "refit", "value": "verifier"}
TRAINING_NAMES = {"light_epochs": "refit_epochs", "light_content_lr": "refit_content_lr",
                  "light_beta": "refit_beta", "light_fraction": "refit_fraction"}
PRESET_NAMES = {"H1A2_1000": "mp20_default"}


def module_key(name):
    return MODULE_NAMES.get(name, name)


def module_name(name):
    return module_key(name)


def model_key(name):
    return MODEL_NAMES.get(name, name)


def feedback_stage_key(name):
    return FEEDBACK_STAGES.get(name, name)


def preset_key(name):
    return PRESET_NAMES.get(name, name)


def _combine(left, right, key):
    if isinstance(left, dict) and isinstance(right, dict):
        result = deepcopy(left)
        for child, value in right.items():
            result[child] = _combine(result[child], value, key + "." + child) if child in result else value
        return result
    if left != right:
        raise ValueError(f"Conflicting configuration values for {key}")
    return left


def _rename(mapping, aliases):
    result = {}
    for key, value in mapping.items():
        name = aliases.get(key, key)
        result[name] = _combine(result[name], value, name) if name in result else value
    return result


def configuration_names(config):
    result = _rename(deepcopy(config), MODULE_NAMES)
    if "models" in result:
        result["models"] = _rename(result["models"], MODEL_NAMES)
    if "feedback" in result and "training" in result["feedback"]:
        result["feedback"]["training"] = _rename(result["feedback"]["training"], TRAINING_NAMES)
    sampling = result.get("sampling", {})
    if isinstance(sampling.get("plans"), str) and sampling["plans"].startswith("preset:"):
        sampling["plans"] = "preset:" + preset_key(sampling["plans"][7:])
    return result


def setting_key(key):
    parts = key.split(".")
    parts[0] = module_key(parts[0])
    if parts[0] == "models" and len(parts) > 1:
        parts[1] = model_key(parts[1])
    if parts[:2] == ["feedback", "training"] and len(parts) > 2:
        parts[2] = TRAINING_NAMES.get(parts[2], parts[2])
    return ".".join(parts)
