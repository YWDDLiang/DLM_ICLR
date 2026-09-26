"""Public commands route to the requested model and training stage."""
import importlib.util
from pathlib import Path

import pytest

from dlm_iclr.cli import main, parser
from dlm_iclr.runtime import pipeline


@pytest.mark.parametrize("public", ["constructor", "periodic", "feedback"])
def test_sample_name_routes_to_requested_stage(monkeypatch, public):
    calls = []
    monkeypatch.setattr(pipeline, "sample", lambda c, stage, **kw: calls.append(stage) or {})
    main(["sample", public])
    assert calls == [public]
    assert parser().parse_args(["train", public]).module == public


@pytest.mark.parametrize("public", ["reconstruction", "refit", "verifier"])
def test_feedback_training_names(public):
    args = parser().parse_args(["train", "feedback", "--stage", public])
    assert args.module == "feedback" and args.stage == public


def test_named_resume_boundary():
    args = parser().parse_args(["run", "--plans", "panel.jsonl", "--constructor", "periodic",
                                "--from-stage", "feedback"])
    assert args.constructor == "periodic" and args.from_stage == "feedback"


@pytest.mark.parametrize("public", ["constructor", "periodic", "feedback"])
def test_reproduction_names_preserve_command_plan(tmp_path, public):
    source = Path(__file__).resolve().parents[1] / "scripts/reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduce_names_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(["--stage", public, "--output", str(tmp_path)])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert commands[-1][3:5] == ["train", public]
