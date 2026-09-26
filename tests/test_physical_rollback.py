from copy import deepcopy
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from dlm_iclr.evaluation.physics import record_key
from dlm_iclr.feedback.physical_rollback import select
from dlm_iclr.feedback.value import choose
from dlm_iclr.runtime.config import load
from dlm_iclr.runtime.finalize import finalize
from dlm_iclr.runtime.io import file_hash, read_json, read_rows, write_json, write_rows


def record(index, changed=False):
    structure = Structure(
        Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5 + 0.01 * (index + changed), 0.5, 0.5]]
    )
    return {"source_id": f"case:{index}", "ordinal": index, "success": True, "structure": structure.as_dict()}


def outcome(item, meta, sun=False):
    verified = meta is not None
    status = "verified" if verified else "not_converged"
    score = {
        "source_id": item["source_id"],
        "ordinal": item["ordinal"],
        "record_key": record_key(item),
        "terminal_status": status,
        "terminal_verified": verified,
        "strict_sun": sun if verified else None,
        "meta_sun": meta,
    }
    label = {
        "source_id": item["source_id"],
        "ordinal": item["ordinal"],
        "record_key": record_key(item),
        "status": status,
        "verified": verified,
        "terminal_energy": -0.1 if verified else None,
    }
    return score, label


@pytest.mark.parametrize(
    "before_meta,after_meta,after_sun,restore",
    [
        (True, False, False, True),
        (True, None, False, True),
        (True, True, False, False),
        (True, True, True, False),
        (False, False, False, False),
        (None, False, False, False),
    ],
)
def test_historical_physical_rollback_rule(before_meta, after_meta, after_sun, restore):
    old, new = record(0), record(0, True)
    before, old_label = outcome(old, before_meta)
    after, new_label = outcome(new, after_meta, after_sun)
    selected, labels, decisions = select([old], [new], [before], [after], [old_label], [new_label])
    assert selected == [old if restore else new]
    assert labels == [old_label if restore else new_label]
    assert decisions[0]["restored_reference"] is restore
    if after_meta is None:
        assert decisions[0]["reconstruction_meta_sun"] is None
        assert decisions[0]["reason"] == "reconstruction_unresolved"


@pytest.mark.parametrize("damage", ["order", "geometry", "label_geometry", "unqualified"])
def test_rollback_rejects_unbound_or_unqualified_results(damage):
    old, new = record(0), record(0, True)
    before, old_label = outcome(old, True)
    after, new_label = outcome(new, False)
    if damage == "order":
        after["ordinal"] = 99
    elif damage == "geometry":
        after["record_key"] = record_key(old)
    elif damage == "label_geometry":
        old_label["record_key"] = "wrong"
    else:
        before["terminal_verified"] = old_label["verified"] = False
        before["terminal_status"] = old_label["status"] = "not_converged"
    with pytest.raises(ValueError):
        select([old], [new], [before], [after], [old_label], [new_label])


def prepared_pair(root):
    old, new = [record(i) for i in range(2)], [record(i, True) for i in range(2)]
    write_rows(root / "refined.jsonl", old)
    write_rows(root / "edited.jsonl", new)
    for endpoint, records, flags in [("refined", old, [True, False]), ("edited", new, [False, True])]:
        values = [outcome(r, flag) for r, flag in zip(records, flags)]
        directory = root / "evaluation" / endpoint
        write_rows(directory / "scores.jsonl", [x[0] for x in values])
        write_rows(directory / "physics/labels.jsonl", [x[1] for x in values])
        write_json(directory / "physics/protocol.json", {"checkpoint_sha256": "fixture", "fmax": 0.1})
        write_json(
            directory / "scoring/summary.json",
            {
                "hull_reference_sha256": "hull-fixture",
                "novelty_evaluation": {"training_identity": {"source_sha256": "train-fixture"}},
            },
        )
    return read_rows(root / "refined.jsonl"), read_rows(root / "edited.jsonl")


def test_finalization_rescores_whole_selected_cohort_and_reuses_labels(tmp_path, monkeypatch):
    from dlm_iclr.evaluation import direct, workflow

    old, new = prepared_pair(tmp_path)
    input_sha = file_hash(tmp_path / "edited.jsonl")
    expected = [old[0], new[1]]
    calls = []

    def physical(config, records, out, *, labels):
        assert records == expected
        assert [l["record_key"] for l in labels] == [record_key(r) for r in expected]
        calls.append("physical")
        return [], {"recomputed_on_complete_cohort": True, "requests": len(records)}

    def direct_metrics(records, out, **kwargs):
        assert records == expected
        calls.append("direct")
        return [], {"recomputed_on_complete_cohort": True}

    monkeypatch.setattr(workflow, "evaluate", physical)
    monkeypatch.setattr(direct, "evaluate_direct", direct_metrics)
    c = load(overrides=[f"output={tmp_path}"])
    report = finalize(c, tmp_path, {"direct": {}, "sun": {}})
    assert calls == ["physical", "direct"]
    assert report["restored_references"] == 1 and report["endpoint"] == "rollback"
    assert report["sun"]["recomputed_on_complete_cohort"]
    assert read_rows(tmp_path / "final.jsonl") == expected
    assert file_hash(tmp_path / "edited.jsonl") == input_sha
    assert len(read_rows(tmp_path / "evaluation/rollback/decisions.jsonl")) == 2
    c["feedback"]["physical_rollback"] = False
    with pytest.raises(ValueError, match="policy changed"):
        finalize(c, tmp_path, {"direct": {}, "sun": {}})


def test_disabled_rollback_keeps_edited_endpoint_and_learned_keep(tmp_path, monkeypatch):
    from dlm_iclr.evaluation import direct, workflow

    items = [record(0, True)]
    write_rows(tmp_path / "edited.jsonl", items)

    def no_extra_evaluation(*a, **kw):
        raise AssertionError("Disabled rollback must not evaluate another collection")

    monkeypatch.setattr(workflow, "evaluate", no_extra_evaluation)
    monkeypatch.setattr(direct, "evaluate_direct", no_extra_evaluation)
    c = load(overrides=["feedback.physical_rollback=false"])
    report = finalize(c, tmp_path, {"direct": {"original": True}, "sun": {"original": True}})
    assert report["endpoint"] == "edited" and report["restored_references"] == 0
    assert file_hash(tmp_path / "final.jsonl") == file_hash(tmp_path / "edited.jsonl")
    assert choose([[-1.0, -1.0]]) is None  # Learned KEEP is unconditional.


def test_mixed_physical_protocols_are_rejected(tmp_path):
    prepared_pair(tmp_path)
    write_json(tmp_path / "evaluation/edited/physics/protocol.json", {"checkpoint_sha256": "different"})
    with pytest.raises(ValueError, match="matching physical"):
        finalize(load(), tmp_path, {"direct": {}, "sun": {}})


def test_finalization_can_resume_when_unknown_matching_resolves(tmp_path, monkeypatch):
    from dlm_iclr.evaluation import direct, workflow

    prepared_pair(tmp_path)
    score_path = tmp_path / "evaluation/edited/scores.jsonl"
    scores = read_rows(score_path)
    scores[0]["meta_sun"] = None
    write_rows(score_path, scores)
    monkeypatch.setattr(workflow, "evaluate", lambda c, rows, out, **kw: ([], {"requests": len(rows)}))
    monkeypatch.setattr(direct, "evaluate_direct", lambda rows, out, **kw: ([], {"requests": len(rows)}))
    c = load()
    assert finalize(c, tmp_path, {"direct": {}, "sun": {}})["restored_references"] == 1
    scores[0]["meta_sun"] = True
    write_rows(score_path, scores)
    assert finalize(c, tmp_path, {"direct": {}, "sun": {}})["restored_references"] == 0


def test_default_pipeline_reaches_physical_rollback(tmp_path, monkeypatch):
    from dlm_iclr.runtime import pipeline
    from dlm_iclr.evaluation import direct, workflow, hull

    old, new = prepared_pair(tmp_path)
    calls = []

    def sample(config, stage, **kwargs):
        calls.append(stage)
        records = new if stage == "feedback" else old
        write_rows(tmp_path / f"{pipeline.STAGES[stage]}.jsonl", records)
        return {"requests": len(records)}

    def physical(config, records, out, *, labels=None):
        calls.append((out.name, labels is not None))
        if out.name == "rollback":
            assert records == [old[0], new[1]]
            assert [l["record_key"] for l in labels] == [record_key(r) for r in records]
        return [], {"requests": len(records), "cohort": out.name}

    def direct_metrics(records, out, **kwargs):
        return [], {"requests": len(records), "cohort": out.name}

    monkeypatch.setattr(pipeline, "sample", sample)
    monkeypatch.setattr(workflow, "evaluate", physical)
    monkeypatch.setattr(direct, "evaluate_direct", direct_metrics)
    monkeypatch.setattr(hull, "query", lambda *a, **kw: {})
    result = pipeline.run(load(overrides=[f"output={tmp_path}"]), "preset:mp20_default", output=tmp_path)
    assert calls == [
        "periodic",
        "diffusion",
        ("raw", False),
        ("refined", False),
        "feedback",
        ("edited", False),
        ("rollback", True),
    ]
    assert result["final"]["sun"]["cohort"] == "rollback"
    assert read_json(tmp_path / "final.summary.json")["restored_references"] == 1


@pytest.mark.parametrize("protect", [True, False])
@pytest.mark.parametrize("rollback", [True, False])
def test_switches_are_independent(protect, rollback):
    c = load(
        overrides=[
            f"feedback.protect_sun={str(protect).lower()}",
            f"feedback.physical_rollback={str(rollback).lower()}",
        ]
    )
    assert c["feedback"]["protect_sun"] is protect
    assert c["feedback"]["physical_rollback"] is rollback
    assert choose([[-1.0, -1.0]]) is None


def test_switch_defaults_and_type_validation():
    c = load()
    assert c["feedback"]["protect_sun"] is c["feedback"]["physical_rollback"] is True
    for name in ("protect_sun", "physical_rollback"):
        with pytest.raises(ValueError, match="JSON boolean"):
            load(overrides=[f'feedback.{name}="false"'])


@pytest.mark.parametrize("enabled", [True, False])
def test_sun_switch_controls_protected_worker_inputs(tmp_path, monkeypatch, enabled):
    from dlm_iclr.runtime import pipeline

    item = record(0)
    write_rows(tmp_path / "plans.jsonl", [{"source_id": item["source_id"], "ordinal": 0}])
    write_rows(tmp_path / "refined.jsonl", [item])
    write_json(tmp_path / "diffusion.settings.json", {"signature": "fixture"})
    score, _ = outcome(item, True, True)
    write_rows(tmp_path / "evaluation/refined/scores.jsonl", [score])
    protected = []

    class Process:
        exitcode = 0

        def __init__(self, *, target, args):
            protected.append(args[-1])

        def start(self):
            write_json(tmp_path / "edited/000000.json", {"E": {"record": deepcopy(item)}})

        def join(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(pipeline.mp, "get_context", lambda name: SimpleNamespace(Process=Process))
    c = load(overrides=[f"feedback.protect_sun={str(enabled).lower()}", 'runtime.devices=["cpu"]'])
    pipeline.sample(c, "feedback", output=tmp_path)
    assert protected == [{item["source_id"]} if enabled else set()]
