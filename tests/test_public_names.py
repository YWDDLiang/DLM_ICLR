"""Paper-facing commands retain the same implementation and checkpoints."""
import importlib.util
from pathlib import Path

import pytest

from dlm_iclr.cli import main, parser
from dlm_iclr.runtime import pipeline


@pytest.mark.parametrize("public,internal", [("constructor", "b0"), ("periodic", "c1"), ("feedback", "c2")])
def test_sample_name_routes_to_existing_stage(monkeypatch, public, internal):
    calls = []
    monkeypatch.setattr(pipeline, "sample", lambda c, stage, **kw: calls.append(stage) or {})
    main(["sample", public])
    main(["sample", internal])
    assert calls == [public, public]
    assert parser().parse_args(["train", public]).module == public


@pytest.mark.parametrize("public,internal", [("reconstruction", "editor"), ("refit", "light"), ("verifier", "value")])
def test_feedback_training_names(public, internal):
    args = parser().parse_args(["train", "feedback", "--stage", public])
    assert args.module == "feedback" and args.stage == public
    assert parser().parse_args(["train", "c2", "--stage", internal]).stage == public


def test_named_resume_boundary():
    args = parser().parse_args(["run", "--plans", "panel.jsonl", "--constructor", "periodic",
                                "--from-stage", "feedback"])
    assert args.constructor == "periodic" and args.from_stage == "feedback"


@pytest.mark.parametrize("public,internal", [("constructor", "b0"), ("periodic", "c1"), ("feedback", "c2")])
def test_reproduction_names_preserve_config_and_command_plan(tmp_path, public, internal):
    source = Path(__file__).resolve().parents[1] / "scripts/reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduce_names_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    for name in (public, internal):
        args = module.parser().parse_args(["--stage", name, "--output", str(tmp_path)])
        config, root = module.resolve(args)
        results.append((config, module.commands(args, config, root)))
    assert results[0] == results[1]
    assert results[0][1][-1][3:5] == ["train", public]
