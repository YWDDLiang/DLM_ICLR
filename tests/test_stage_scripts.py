"""Stage scripts prepare only the dependencies needed by the requested operation."""

import importlib.util
from pathlib import Path

import pytest


def launcher():
    source = Path(__file__).resolve().parents[1] / "scripts/reproduce.py"
    spec = importlib.util.spec_from_file_location("stage_entry_check", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("metric", ["direct", "sun"])
def test_evaluation_stage_only_evaluates_saved_final_records(metric, tmp_path):
    module = launcher()
    args = module.parser().parse_args(["--stage", "evaluate-" + metric, "--output", str(tmp_path)])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert len(commands) == 1
    command = commands[0]
    assert command[3:5] == ["evaluate", metric]
    assert Path(command[command.index("--structures") + 1]) == root / "samples/final.jsonl"
    assert "train" not in command and "sample" not in command and "prepare" not in command


def test_evaluation_allows_an_explicit_collection(tmp_path):
    module = launcher()
    source = tmp_path / "draft.jsonl"
    args = module.parser().parse_args(["--stage", "evaluate-direct", "--structures", str(source)])
    config, root = module.resolve(args)
    command = module.commands(args, config, root)[0]
    assert Path(command[command.index("--structures") + 1]) == source.resolve()


def test_optional_planner_training_does_not_sample():
    module = launcher()
    args = module.parser().parse_args(["--stage", "train-planner"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert "--with-planner" in commands[0]
    assert commands[1][3:5] == ["train", "planner"]
    assert len(commands) == 2


def test_planner_inference_only_samples():
    module = launcher()
    args = module.parser().parse_args(["--stage", "sample-planner"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert len(commands) == 1 and commands[0][3:5] == ["sample", "planner"]


@pytest.mark.parametrize("stage,expected,prepare", [
    ("feedback-collect", ["warmup", "collect", "label"], True),
    ("feedback-fit", ["compile", "reconstruction", "refit", "verifier", "risk"], False),
])
def test_feedback_collect_and_fit_have_separate_dependencies(stage, expected, prepare):
    module = launcher()
    args = module.parser().parse_args(["--stage", stage, "--resume"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    if prepare:
        assert commands.pop(0)[3] == "prepare"
    assert [c[c.index("--stage") + 1] for c in commands] == expected
    assert all(c[3:5] == ["train", "feedback"] and "--resume" in c for c in commands)


def test_inference_does_not_train_or_collect_feedback():
    module = launcher()
    args = module.parser().parse_args(["--stage", "inference"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert [c[3] for c in commands] == ["prepare", "plans", "run"]


def test_diffusion_stage_prepares_data_and_trains_refiner():
    module = launcher()
    args = module.parser().parse_args(["--stage", "diffusion"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert commands[0][3] == "prepare"
    assert commands[1][3:5] == ["train", "diffusion"]
    assert len(commands) == 2
