import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dlm_iclr.data.plans import load_plans, validate_plan
from dlm_iclr.data.selection import select
from dlm_iclr.runtime.config import load, backend_config
from dlm_iclr.runtime.io import read_rows, write_rows, write_json
from dlm_iclr.runtime.pipeline import stage_settings

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "src/dlm_iclr/data/presets/plans/H1A2_1000.jsonl"


def fresh(code, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    # Each test process should establish its capacity through the config loader.
    for name in ("DLM_MAX_ATOMS", "DLM_MIN_ATOMS", "DLM_DATASET_LABEL", "DLM_LENGTH_MAX_BIN"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code, *args],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result


def test_frozen_panel_selection_and_identity():
    manifest = json.loads(PANEL.with_suffix(".manifest.json").read_text())
    original = read_rows(PANEL)
    rows, _ = load_plans("preset:H1A2_1000", requests=1000)
    assert len(rows) == 1000
    assert all(validate_plan(r) is None for r in rows)
    assert rows[0]["source_id"] == "H1A2_1200:0000"
    assert rows[-1]["original_ordinal"] == 1011
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest() == manifest["selected_sha256"]
    expected = [i for i in range(1200) if i not in {r["original_ordinal"] for r in manifest["excluded_invalid_plans"]}][:1000]
    assert [r["original_ordinal"] for r in rows] == expected
    for a, b in zip(rows, original):
        assert a["body_prompt"] == b["body_prompt"]
        assert a["body_noise_seed"] == b["body_noise_seed"]
        assert a["refiner_noise_seed"] == b["refiner_noise_seed"]


def test_selection_logs_failures_and_refuses_overwriting(tmp_path):
    source = tmp_path / "source.jsonl"
    valid = read_rows(PANEL)[:2]
    write_rows(source, [dict(source_id="bad", body_eligible=False), *valid])
    c = load(overrides=[f"output={tmp_path}", "sampling.requests=2"])
    report = select(c, source=source)
    assert report["requests"] == 2 and len(report["excluded_invalid_plans"]) == 1
    assert [x["source_id"] for x in read_rows(report["output"])] == [x["source_id"] for x in valid]
    write_rows(source, list(reversed(valid)))
    with pytest.raises(ValueError, match="Selected Plans changed"):
        select(c, source=source)


@pytest.mark.parametrize("name,minimum,maximum", [("mp20", 1, 20), ("perov-5", 5, 5), ("MPTS-52", 1, 52)])
def test_dataset_capacity_roundtrip_in_fresh_worker(name, minimum, maximum):
    result = fresh('''
import json,sys
from dlm_iclr.runtime.config import load
c=load(dataset=sys.argv[1])
from dlm_iclr.runtime.capacity import MAX_ATOMS,MIN_ATOMS
from dlm_iclr._core.dynamic_crystal import arrays_to_dynamic_answer,parse_dynamic_answer
from dlm_iclr._core.h1_llm_planner import build_planner_user_prompt
from dlm_iclr.data.plans import validate_plan
from dlm_iclr._core.r5_plan_state import build_body_prompt
from dlm_iclr._core.expert_edit import ExpertEditConfig
from dlm_iclr.c2.scope import feasible_modes
n=MAX_ATOMS
answer,_=arrays_to_dynamic_answer([6,6,6],[90,90,90],['Si']*n,[[i/n,0,0] for i in range(n)])
assert len(parse_dynamic_answer(answer)['species']) == n
plan=dict(N=n,elements=['Si'],counts=[n],anion_framework='other',charge_bucket='single_element',lattice_system='cubic',spacegroup_bucket='sg_195_230',volume_per_atom_bin='volpa_010_014')
assert validate_plan({'plan_state':plan,'body_prompt':build_body_prompt(plan).rstrip()+'\\n'}) is None
assert ExpertEditConfig(hidden_size=8).max_sites == n
assert feasible_modes(n,80)
prompt=build_planner_user_prompt(prompt_style='h1_rich_plan_v1')
assert c['dataset']['label'] in prompt
print(json.dumps([MIN_ATOMS,MAX_ATOMS,7+4*n]))
''', name)
    assert json.loads(result.stdout) == [minimum, maximum, 7 + 4 * maximum]


def test_non_mp20_needs_external_panel(tmp_path):
    result = fresh('''
from dlm_iclr.runtime.config import load
c=load(dataset='perov-5')
from dlm_iclr.data.selection import select
try: select(c)
except ValueError as e: assert 'requires' in str(e)
else: raise AssertionError('missing panel accepted')
try: select(c, source='preset:H1A2_1000')
except ValueError as e: assert 'MP20 only' in str(e)
else: raise AssertionError('MP20 panel used for Perov')
''')
    assert result.returncode == 0


def test_constructor_outputs_and_resumption_are_isolated(tmp_path):
    c = load()
    plans, _ = load_plans("H1A2_1000", requests=1)
    stage_settings(c, "b0", plans, tmp_path)
    with pytest.raises(ValueError, match="separate output"):
        stage_settings(c, "c1", plans, tmp_path)
    c["b0"]["temperature"] = 0.9
    with pytest.raises(ValueError, match="changed"):
        stage_settings(c, "b0", plans, tmp_path)


def test_b0_policy_retains_native_constraints_without_c1():
    from dlm_iclr.c1.generation import construct
    import inspect
    c = load()
    policy = backend_config(c, stage="b0").inference
    assert not policy.geometry_monitor and not policy.construction_recovery
    assert not policy.adaptive_lattice_recovery
    assert inspect.signature(construct).parameters["geometry_monitor"].default is True
    from dlm_iclr.c1.generation import Constructor
    from dlm_iclr._core.dynamic_crystal import build_special_tokens
    class Tokenizer:
        def get_vocab(self):
            return {t: i for i, t in enumerate(build_special_tokens())}
    g = Constructor(None, Tokenizer(), policy)
    assert g.axis_head is None
    assert g.constraints["duplicate_coordinate_mask"]
    assert g.constraints["lattice_volume_mask"]
    assert g.constraints["min_lattice_rad"] == 1e-4


def test_one_command_uses_fresh_stages_and_never_trains_planner_by_default(tmp_path):
    spec = importlib.util.spec_from_file_location("reproduce", ROOT / "scripts/reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(["--output", str(tmp_path)])
    config, root = module.resolve(args)
    commands = module.commands(args, config, root)
    assert [x[3] for x in commands] == ["prepare", "plans", "train", "train", "train", "run"]
    assert [x[4] for x in commands if x[3] == "train"] == ["b0", "c1", "c2"]
    assert config["sampling"]["requests"] == config["planner"]["sampling"]["requests"] == 1000
    assert "--with-planner" not in commands[0]
    assert config["c2"]["training"]["plans"] == "@run/data/plans/train.jsonl"


def test_pipeline_evaluates_all_endpoints_without_relabeling(tmp_path, monkeypatch):
    from dlm_iclr.runtime import pipeline
    from dlm_iclr.evaluation import direct, workflow, hull
    c = load(overrides=[f"output={tmp_path}"])
    calls = []
    def fake_sample(config, stage, **kwargs):
        calls.append(stage)
        write_rows(tmp_path / f"{pipeline.STAGES[stage]}.jsonl", [{"source_id": "a", "success": False}])
        return {"stage": stage}
    def direct_metric(records, output, **kwargs):
        calls.append("direct:" + output.name)
        return [], {"requests": len(records)}
    def physical_metric(config, records, output):
        calls.append("sun:" + output.name)
        return [], {"requests": len(records)}
    monkeypatch.setattr(pipeline, "sample", fake_sample)
    monkeypatch.setattr(direct, "evaluate_direct", direct_metric)
    monkeypatch.setattr(workflow, "evaluate", physical_metric)
    monkeypatch.setattr(hull, "query", lambda *a, **k: calls.append("hull"))
    pipeline.run(c, "preset:H1A2_1000", output=tmp_path)
    assert calls == ["c1", "diffusion", "hull", "direct:raw", "sun:raw", "direct:refined", "sun:refined", "c2", "direct:edited", "sun:edited"]


def test_direct_unknown_fingerprint_is_not_counted_as_invalid(tmp_path, monkeypatch):
    from pymatgen.core import Structure, Lattice
    from dlm_iclr.evaluation import features
    from dlm_iclr.evaluation.direct import evaluate_direct
    crystal = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
    records = [{"source_id": "a", "success": True, "structure": crystal.as_dict()}]
    reference = tmp_path / "reference.jsonl"
    write_rows(reference, records)
    unknown = {"error": "resource_unknown:timeout", "comp_fp": None, "struct_fp": None}
    known = {"error": None, "comp_fp": [0.0], "struct_fp": [0.0]}
    monkeypatch.setattr(features, "compute_features", lambda *a, **k: ([unknown, known], {}))
    scores, report = evaluate_direct(records, tmp_path / "report", reference=reference)
    assert scores[0]["fingerprint_valid"] is None and scores[0]["valid"] is None
    assert report["unknown_counts"]["valid"] == 1
    assert report["metrics"]["cov_precision"] is None
    assert report["metrics"]["wdist_density"] is None
    assert report["coverage_bounds_percent"]["cov_precision"] == [0, 100]


def _blocked_worker(pipe):
    import time
    pipe.send({"ready": True})
    pipe.recv()
    time.sleep(30)


def test_fingerprint_watchdog_reaps_blocked_worker():
    import multiprocessing
    import time
    from dlm_iclr._core.isolated_workers import isolated_results
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    rows = list(isolated_results([("key", {})], worker_target=_blocked_worker,
                                worker_arguments=[()], task_timeout=.2, startup_timeout=30))
    assert rows[0][2]["status"] == "worker_error"
    assert time.monotonic() - started < 25
    assert {p.pid for p in multiprocessing.active_children()} <= before


@pytest.mark.parametrize("metrics", ["direct", "sun"])
def test_cli_limits_predictions_but_preserves_actual_denominator(tmp_path, monkeypatch, metrics):
    from dlm_iclr.cli import main
    from dlm_iclr.evaluation import direct, workflow
    source = tmp_path / "structures.jsonl"
    write_rows(source, [{"source_id": str(i), "success": False} for i in range(1002)])
    observed = []
    def score(rows, *args, **kwargs):
        observed.append(rows)
        return [], {"requests": len(rows)}
    if metrics == "direct":
        monkeypatch.setattr(direct, "evaluate_direct", score)
    else:
        monkeypatch.setattr(workflow, "evaluate", lambda c, rows, *a, **kw: score(rows))
    args = ["evaluate", metrics, "--structures", str(source), "--output", str(tmp_path / "out")]
    main(args)
    assert len(observed[-1]) == 1000
    assert observed[-1][-1]["source_id"] == "999"
    main([*args, "--num-samples", "2"])
    assert len(observed[-1]) == 2
    write_rows(source, [{"source_id": "only", "success": False}])
    main(args)
    assert len(observed[-1]) == 1
