"""Feedback fitting accepts only labels bound to the collected training sources."""

from copy import deepcopy
import pytest

from dlm_iclr.feedback.workflow import load_feedback_inputs, require_feedback_inputs
from dlm_iclr.evaluation.physics import record_key
from dlm_iclr.runtime.config import load
from dlm_iclr.runtime.io import write_rows


def inputs(tmp_path):
    config = load(overrides=[f"output={tmp_path}", "feedback.candidates=1"])
    root = tmp_path / "feedback"
    (root / "warmup/checkpoint").mkdir(parents=True)
    reference = {"source_id": "train:0", "success": False, "reason": "generation_failure"}
    candidate = dict(reference, reason="candidate_failure")
    bundle = {
        "plan": {"source_id": "train:0", "provenance": {"usage_role": "train"}},
        "F": {"record": reference},
        "E": {"candidates": [{"rank": 0, "record": candidate}]},
    }
    labels = {name: {"source_id": "train:0", "record_key": record_key(record)}
              for name, record in (("current", reference), ("candidate_0", candidate))}
    write_rows(root / "collection/bundles.jsonl", [bundle])
    for name, label in labels.items():
        write_rows(root / "labels" / f"{name}.jsonl", [label])
    return config, root, bundle, labels


def test_missing_collection_fails_before_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="05_collect_feedback.sh"):
        require_feedback_inputs(load(overrides=[f"output={tmp_path}"]))


def test_collected_labels_match_source_and_geometry(tmp_path):
    config, _, bundle, labels = inputs(tmp_path)
    bundles, scores = load_feedback_inputs(config)
    assert bundles == [bundle]
    assert scores == {name: [value] for name, value in labels.items()}


@pytest.mark.parametrize("damage", ["label_source", "geometry", "count", "split", "duplicate_rank"])
def test_invalid_feedback_pairing_is_rejected(tmp_path, damage):
    config, root, bundle, labels = inputs(tmp_path)
    if damage == "label_source":
        labels["candidate_0"]["source_id"] = "train:other"
    elif damage == "geometry":
        labels["candidate_0"]["record_key"] = labels["current"]["record_key"]
    elif damage == "split":
        bundle["plan"]["provenance"]["usage_role"] = "evaluation"
    elif damage == "duplicate_rank":
        bundle["E"]["candidates"].append(deepcopy(bundle["E"]["candidates"][0]))
    write_rows(root / "collection/bundles.jsonl", [bundle])
    write_rows(root / "labels/candidate_0.jsonl", [] if damage == "count" else [labels["candidate_0"]])
    with pytest.raises(ValueError):
        load_feedback_inputs(config)
