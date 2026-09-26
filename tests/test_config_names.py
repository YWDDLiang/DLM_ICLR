"""Configuration overrides, asset paths and shared construction controls."""
import json
from pathlib import Path

import pytest

from dlm_iclr.data.plans import load_plans
from dlm_iclr.runtime.config import load, asset, backend_config
from dlm_iclr.runtime.io import write_json
from dlm_iclr.runtime.pipeline import stage_settings


def test_config_and_overrides_preserve_explicit_fields(tmp_path):
    p = tmp_path / "settings.json"
    write_json(p, {
        "constructor": {"lr": 0.00003}, "periodic": {"temperature": 0.6},
        "feedback": {"training": {"refit_epochs": 3, "refit_beta": 0.2}},
        "models": {"constructor": "saved/constructor/final", "feedback": "saved/feedback/refit/checkpoint", "verifier": "saved/verifier.pt"},
        "sampling": {"plans": "preset:mp20_default"},
    })
    c = load(p, ["constructor.effective_batch_size=32", "feedback.training.refit_fraction=0.4", "models.periodic=periodic.pt"])
    assert c["constructor"]["lr"] == 0.00003
    assert c["constructor"]["effective_batch_size"] == 32
    assert c["periodic"]["temperature"] == 0.6
    assert c["feedback"]["training"]["refit_epochs"] == 3
    assert c["feedback"]["training"]["refit_beta"] == 0.2
    assert c["feedback"]["training"]["refit_fraction"] == 0.4
    assert c["sampling"]["plans"] == "preset:mp20_default"
    assert asset(c, "constructor") == str(tmp_path / "saved/constructor/final")
    assert asset(c, "feedback") == str(tmp_path / "saved/feedback/refit/checkpoint")
    assert asset(c, "verifier") == str(tmp_path / "saved/verifier.pt")
    assert asset(c, "periodic") == str(tmp_path / "periodic.pt")
    assert backend_config(c, stage="periodic").inference.temperature == 0.6


@pytest.mark.parametrize("settings", [
    {"unrecognized_module": {"lr": 1}},
    {"models": {"misspelled_model": "weights.pt"}},
    {"feedback": {"training": {"misspelled_epochs": 3}}},
])
def test_unsupported_configuration_is_rejected(tmp_path, settings):
    p = tmp_path / "invalid.json"
    write_json(p, settings)
    with pytest.raises(ValueError, match="Unknown"):
        load(p)


def test_default_config_export_has_descriptive_keys(tmp_path):
    from dlm_iclr.cli import main
    p = tmp_path / "config.json"
    main(["config", "--output", str(p)])
    c = json.loads(p.read_text())
    assert c["sampling"]["plans"] == "preset:mp20_default"
    assert c["models"]["constructor"] == "@run/constructor/final"
    assert c["models"]["periodic"] == "@run/periodic/best.pt"
    assert c["models"]["feedback"] == "@run/feedback/refit/checkpoint"
    assert c["models"]["verifier"] == "@run/feedback/verifier/relative_verifier.pt"


def test_named_preset_keeps_record_identity_and_noise():
    current, _ = load_plans("mp20_default", requests=1000)
    prefixed, _ = load_plans("preset:mp20_default", requests=1000)
    assert current == prefixed


def test_unsupported_cache_metadata_cannot_be_silently_reused(tmp_path):
    write_json(tmp_path / "unknown.settings.json", {"signature": "previous"})
    plans, _ = load_plans("mp20_default", requests=1)
    with pytest.raises(ValueError, match="Unsupported stage metadata"):
        stage_settings(load(), "periodic", plans, tmp_path)


def test_constructor_and_periodic_share_all_decoding_controls():
    from dataclasses import asdict
    config = load()
    base = asdict(backend_config(config, stage="constructor").inference)
    periodic = asdict(backend_config(config, stage="periodic").inference)
    assert base == periodic
    assert base["geometry_monitor"] and base["construction_recovery"]
    assert base["adaptive_lattice_recovery"]


def test_feedback_stage_receipts_match_default_asset_paths(tmp_path, monkeypatch):
    from dlm_iclr.feedback import workflow
    c = load(overrides=[f"output={tmp_path}"])
    names = []
    monkeypatch.setattr(workflow, "stage", lambda c, name, **kw: names.append(name) or {})
    result = workflow.train(c)
    assert names == ["warmup", "collect", "label", "compile", "reconstruction", "refit", "verifier", "risk"]
    assert Path(result["checkpoint"]) == tmp_path / "feedback/refit/checkpoint"
    assert all((tmp_path / "feedback/stages" / f"{n}.json").exists() for n in names)
