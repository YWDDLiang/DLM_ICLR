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


def test_optional_planner_trains_and_samples_without_exporting_default_plans():
    module = launcher()
    args = module.parser().parse_args(["--stage", "train-planner"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert "--with-planner" in commands[0]
    assert commands[1][3:5] == ["train", "planner"]
    assert commands[2][3:5] == ["sample", "planner"]
    assert len(commands) == 3


def test_diffusion_stage_prepares_data_and_trains_refiner():
    module = launcher()
    args = module.parser().parse_args(["--stage", "diffusion"])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert commands[0][3] == "prepare"
    assert commands[1][3:5] == ["train", "diffusion"]
    assert len(commands) == 2
