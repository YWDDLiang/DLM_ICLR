import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from dlm_iclr.runtime.config import load
from dlm_iclr.runtime.io import file_hash, read_rows, write_json, write_rows


def launcher():
    source = Path(__file__).resolve().parents[1] / "scripts/reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduction_priority_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_count_precedence(tmp_path):
    p = tmp_path / "config.json"
    write_json(p, {"sampling": {"requests": 7}, "planner": {"sampling": {"requests": 13}}})
    m = launcher()

    def resolve(*args):
        return m.resolve(m.parser().parse_args(["--config", str(p), *args]))[0]

    assert resolve()["sampling"]["requests"] == 7
    c = resolve("--set", "sampling.requests=9")
    assert c["sampling"]["requests"] == 9 and c["planner"]["sampling"]["requests"] == 13
    c = resolve("--set", "sampling.requests=9", "--num-samples", "11")
    assert c["sampling"]["requests"] == c["planner"]["sampling"]["requests"] == 11


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "true"])
def test_invalid_config_request_counts_fail(value):
    m = launcher()
    with pytest.raises(ValueError, match="positive integer"):
        m.resolve(m.parser().parse_args(["--set", "sampling.requests=" + value]))


def test_coverage_has_one_default_source_and_independent_overrides():
    from dlm_iclr.runtime.datasets import coverage_cutoffs, profile

    for dataset, expected in [("mp20", [0.4, 10]), ("perov-5", [0.2, 4]), ("mpts-52", [0.4, 10])]:
        assert coverage_cutoffs(dataset) == expected
        assert profile(dataset)["evaluation"]["coverage_cutoffs"] == expected
    modified = coverage_cutoffs("mp20")
    modified[0] = 123
    assert coverage_cutoffs("mp20") == [0.4, 10]
    assert load(overrides=["evaluation.coverage_cutoffs=[0.3,8]"])["evaluation"]["coverage_cutoffs"] == [
        0.3,
        8,
    ]


def test_main_sun_evaluation_qualifies_unconverged_labels(tmp_path, monkeypatch):
    from dlm_iclr.evaluation import workflow, sun

    crystal = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    records = [{"source_id": str(i), "success": True, "structure": crystal.as_dict()} for i in range(2)]
    base = dict(
        comp_valid=True,
        struct_valid=True,
        novel=True,
        unique_representative=True,
        reconstructed=True,
        terminal_energy_eV_atom=-1.0,
        hull_energy_eV_atom=-0.9,
        e_above_hull_eV_atom=-0.1,
        strict_sun=True,
        meta_sun=True,
    )
    raw_scores = [
        dict(base, terminal_status="verified", terminal_verified=True),
        dict(base, terminal_status="not_converged", terminal_verified=False),
    ]
    monkeypatch.setattr(sun, "score_records", lambda *a, **kw: (raw_scores, {}))
    c = load(overrides=["evaluation.composition=standard"])
    scores, result = workflow.evaluate(c, records, tmp_path, labels=[{}, {}])
    assert scores[1]["strict_sun"] is None
    assert result["metrics"]["SUN"]["passed"] == 1
    assert result["metrics"]["SUN"]["pending"] == 1
    assert result["requests"] == 2


def test_feedback_compiles_reconstruction_and_verifier_targets():
    from dlm_iclr.feedback.feedback import compile_sources

    plan = {
        "source_id": "train-case",
        "body_prompt": "prompt",
        "plan_state": {"N": 2},
        "provenance": {"usage_role": "train"},
    }
    current = {"token_ids": list(range(15)), "continuous_trace": {"editable": True}}
    proposal = list(range(15))
    proposal[8] = 99
    candidate = {
        "rank": 0,
        "commit": {"applied": True},
        "predicted_gain": [1, 1],
        "geometry_features": [0] * 9,
        "trace": {"proposal_tokens": proposal, "action": {"mode": 1, "positions": [8, 9, 10], "sites": [0]}},
    }
    before = dict(
        terminal_status="verified",
        terminal_verified=True,
        e_above_hull_eV_atom=0.2,
        strict_stable=False,
        meta_stable=False,
        novel=True,
        strict_sun=False,
    )
    after = dict(before, e_above_hull_eV_atom=-0.1, strict_stable=True, meta_stable=True, strict_sun=True)
    editor, verifier = compile_sources(
        [plan], [current], [{"candidates": [candidate]}], {"current": [before], "candidate_0": [after]}
    )
    assert len(editor) == len(verifier) == 1
    assert editor[0]["content_target_tokens"] == proposal
    assert editor[0]["site_targets"] == [1.0, 0.0]
    assert editor[0]["content_positions"] == [8, 9, 10]
    assert verifier[0]["before_targets"] == (0, 0)
    assert verifier[0]["after_targets"] == (1, 1)


class SmallTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = None
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(self.vocab, sort_keys=True))

    @property
    def special_tokens_map(self):
        return {"pad_token": self.pad_token, "eos_token": self.eos_token}

    def add_special_tokens(self, values):
        for token in values["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab))

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.vocab.get(t, 1) for t in re.findall(r"<[^>]+>|\S+", text)]}

    def save_pretrained(self, root):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "tokenizer.json", self.vocab)


def preparation_setup(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from dlm_iclr.data import adapters
    from dlm_iclr.planner import prepare as planner_prepare

    source = tmp_path / "structures.jsonl"
    crystal = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    write_rows(source, [{"id": "a", "structure": crystal.as_dict()}])
    config_path = tmp_path / "config.json"
    write_json(
        config_path,
        {"output": "run", "dataset": {"splits": {s: str(source) for s in ("train", "val", "test")}}},
    )
    c = load(config_path)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: SmallTokenizer())
    monkeypatch.setattr(
        planner_prepare, "build_records_for_plan", lambda **kw: [{"plan_state": kw["plan_state"]}]
    )
    calls = []
    original = adapters.iter_structures

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(adapters, "iter_structures", counted)
    return c, source, tmp_path / "run/data", calls


def test_prepare_reuses_data_and_preserves_planner_files(tmp_path, monkeypatch):
    from dlm_iclr.data.adapters import prepare

    c, source, root, calls = preparation_setup(tmp_path, monkeypatch)
    assert not prepare(c, with_planner=True)["reused"]
    assert len(calls) == 3
    planner_sha = file_hash(root / "planner/train.jsonl")
    assert prepare(c)["reused"]
    assert prepare(c, with_planner=True)["reused"]
    assert len(calls) == 3
    assert file_hash(root / "planner/train.jsonl") == planner_sha
    rows = read_rows(source)
    rows[0]["id"] = "b"
    write_rows(source, rows)
    assert not prepare(c)["reused"]
    assert len(calls) == 6
    assert file_hash(root / "planner/train.jsonl") == planner_sha
    assert not prepare(c, with_planner=True)["reused"]
    assert len(calls) == 9


def test_prepare_rebuilds_modified_outputs_and_honors_force(tmp_path, monkeypatch):
    from dlm_iclr.data.adapters import prepare

    c, _, root, calls = preparation_setup(tmp_path, monkeypatch)
    prepare(c)
    assert not (root / "planner/train.jsonl").exists()
    target = root / "constructor/train.jsonl"
    expected = file_hash(target)
    target.write_text("changed\n")
    assert not prepare(c)["reused"]
    assert file_hash(target) == expected and len(calls) == 6
    assert not prepare(c, force=True)["reused"]
    assert len(calls) == 9
