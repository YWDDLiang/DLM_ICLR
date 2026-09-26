"""Configuration migration preserves parameter values and explicit asset paths."""
import json
from pathlib import Path

import pytest

from dlm_iclr.data.plans import load_plans
from dlm_iclr.runtime.config import load, asset, backend_config
from dlm_iclr.runtime.io import write_json
from dlm_iclr.runtime.pipeline import stage_settings


def test_earlier_config_and_overrides_use_canonical_fields(tmp_path):
    p = tmp_path / "previous.json"
    write_json(p, {
        "b0": {"lr": 0.00003}, "c1": {"temperature": 0.6},
        "c2": {"training": {"light_epochs": 3, "light_beta": 0.2}},
        "models": {"b0": "saved/b0/final", "c2": "saved/c2/light/checkpoint", "value": "saved/value.pt"},
        "sampling": {"plans": "preset:H1A2_1000"},
    })
    c = load(p, ["b0.effective_batch_size=32", "c2.training.light_fraction=0.4", "models.c1=periodic.pt"])
    assert c["constructor"]["lr"] == 0.00003
    assert c["constructor"]["effective_batch_size"] == 32
    assert c["periodic"]["temperature"] == 0.6
    assert c["feedback"]["training"]["refit_epochs"] == 3
    assert c["feedback"]["training"]["refit_beta"] == 0.2
    assert c["feedback"]["training"]["refit_fraction"] == 0.4
    assert c["sampling"]["plans"] == "preset:mp20_default"
    assert not {"b0", "c1", "c2"} & c.keys()
    assert not {"b0", "c1", "c2", "value"} & c["models"].keys()
    assert asset(c, "constructor") == asset(c, "b0") == str(tmp_path / "saved/b0/final")
    assert asset(c, "feedback") == str(tmp_path / "saved/c2/light/checkpoint")
    assert asset(c, "verifier") == asset(c, "value") == str(tmp_path / "saved/value.pt")
    assert asset(c, "periodic") == str(tmp_path / "periodic.pt")
    assert backend_config(c, stage="periodic").inference.temperature == 0.6


def test_conflicting_alias_values_are_rejected(tmp_path):
    p = tmp_path / "ambiguous.json"
    write_json(p, {"b0": {"lr": 1}, "constructor": {"lr": 2}})
    with pytest.raises(ValueError, match="Conflicting configuration"):
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
    assert not {"b0", "c1", "c2"} & c.keys()


def test_preset_alias_keeps_record_identity_and_noise():
    current, _ = load_plans("mp20_default", requests=1000)
    previous, _ = load_plans("H1A2_1000", requests=1000)
    assert current == previous


def test_earlier_cache_metadata_cannot_be_silently_reused(tmp_path):
    write_json(tmp_path / "c1.settings.json", {"signature": "previous"})
    plans, _ = load_plans("mp20_default", requests=1)
    with pytest.raises(ValueError, match="earlier stage metadata"):
        stage_settings(load(), "periodic", plans, tmp_path)


def test_feedback_stage_receipts_match_default_asset_paths(tmp_path, monkeypatch):
    from dlm_iclr.feedback import workflow
    c = load(overrides=[f"output={tmp_path}"])
    names = []
    monkeypatch.setattr(workflow, "stage", lambda c, name, **kw: names.append(name) or {})
    result = workflow.train(c)
    assert names == ["warmup", "collect", "label", "compile", "reconstruction", "refit", "verifier", "risk"]
    assert Path(result["checkpoint"]) == tmp_path / "feedback/refit/checkpoint"
    assert all((tmp_path / "feedback/stages" / f"{n}.json").exists() for n in names)
