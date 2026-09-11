"""Audit published experiment artifacts without loading a model or physical evaluator."""

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/experiments"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode())


def main():
    manifest = read(ROOT / "manifest.json")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.resolve().is_relative_to(ROOT.resolve()), "Artifact path escapes data directory"
        assert path.stat().st_size == item["bytes"], item["path"]
        assert digest(path.read_bytes()) == item["sha256"], item["path"]
    current = ROOT / "completed_1050"
    plans = rows(current / "plans.jsonl.gz")
    assert len(plans) == 1050
    identity = [(row["source_id"], row["ordinal"]) for row in plans]
    assert len(set(identity)) == 1050 and [row["ordinal"] for row in plans] == list(range(1050))
    assert all(type(row[key]) is int and 0 <= row[key] < 2**63
               for row in plans for key in ("body_noise_seed", "refiner_noise_seed"))
    snapshots = {}
    for snapshot in manifest["main"]["snapshots"]:
        structures = rows(current / snapshot / "structures.jsonl.gz")
        scores = rows(current / snapshot / "scores.jsonl.gz")
        summary = read(current / snapshot / "summary.json")
        assert len(structures) == len(scores) == 1050
        assert [(row["source_id"], row["ordinal"]) for row in structures] == identity
        assert [(row["source_id"], row["ordinal"]) for row in scores] == identity
        for plan, record, score in zip(plans, structures, scores):
            if record["success"]:
                geometry = record["structure"] if record["structure"] is not None else record.get("body")
                key = fingerprint(geometry)
                if record["structure"]:
                    sites = record["structure"]["sites"]
                    assert len(sites) == plan["plan_state"]["N"]
                    assert dict(Counter(site["species"][0]["element"] for site in sites)) == record["declared_composition"]
                    assert record["declared_composition"] == dict(zip(plan["plan_state"]["elements"], plan["plan_state"]["counts"]))
            else:
                key = fingerprint({"source_id": record["source_id"], "generation_failure": record.get("reason")})
            assert key == score["record_key"], (snapshot, record["ordinal"])
        for field, expected in summary["known_counts"].items():
            known = sum(row[field] is True for row in scores)
            unknown = sum(row[field] is None for row in scores)
            assert known == expected and unknown == summary["unknown_counts"][field]
            assert summary["count_bounds"][field] == [known, known + unknown]
            if unknown:
                assert summary["counts"][field] is None and summary["percent"][field] is None
            else:
                assert summary["counts"][field] == known
                assert math.isclose(summary["percent"][field], 100 * known / 1050, abs_tol=1e-12)
        snapshots[snapshot] = scores
    paired = read(current / "paired_summary.json")
    for field, expected in paired["fields"].items():
        values = [(a[field], b[field]) for a, b in zip(snapshots["S0"], snapshots["S1"])
                  if a[field] is not None and b[field] is not None]
        assert len(values) == expected["both_known"]
        assert sum(a is False and b is True for a, b in values) == expected["gained"]
        assert sum(a is True and b is False for a, b in values) == expected["lost"]
        assert expected["net"] == expected["gained"] - expected["lost"]
    with (current / "runtime.csv").open(encoding="utf-8", newline="") as stream:
        runtime = list(csv.DictReader(stream))
    assert len(runtime) == 2100
    for row in runtime:
        assert 0 <= int(row["E_calls"]) <= 80 and 0 <= int(row["E_candidates"]) <= 8
    for snapshot in ("S0", "S1"):
        selected = [row for row in runtime if row["snapshot"] == snapshot]
        inference = read(current / snapshot / "summary.json")["inference"]
        assert sum(int(row["E_calls"]) for row in selected) == inference["E_calls_total"]
        assert sum(row["E_selected_rank"] != "unknown" for row in selected) == inference["E_edits"]
    training = read(current / "training/summary.json")
    for branch in ("G", "E", "value"):
        report = read(current / "training" / (branch + ".json"))
        assert report["optimizer_steps"] == training[branch]["optimizer_steps"]
        assert sum(value > 0 for value in report["parameter_delta_squared"].values()) == training[branch]["changed_parameter_tensors"]
    g_visits = read(current / "training/G_exposure.json")["source_visits"]
    e_visits = read(current / "training/E_exposure.json")["pair_visits"]
    assert len(g_visits) == 730 and set(g_visits.values()) == {4}
    assert len(e_visits) == 972 and set(e_visits.values()) == {8}
    content_visits = read(current / "training/E_exposure.json")["content_updated_pair_visits"]
    assert len(content_visits) == 445 and set(content_visits.values()) == {8}
    assert training["value"]["minimum_visits"] == training["value"]["maximum_visits"] == 64
    status = read(current / "experiment_status.json")
    assert status["status"] == "cancelled_by_user" and not status["winner_selected"]
    assert status["completed_snapshots"] == ["S0", "S1"] and status["completed_updates"] == 1
    history = read(ROOT / "historical/ledger.json")
    assert len(history["rows"]) == manifest["historical_rows"]
    assert all(row["source_id"] in history["sources"] for row in history["rows"])
    examples = read(current / "keep_edit_examples.json")
    for name, example in examples["inference_examples"].items():
        winner, best = None, (0.0, 0.0, True)
        for candidate in example["candidates"]:
            if not candidate["commit"]["applied"] or candidate["predicted_gain"] is None:
                continue
            ns, nms = candidate["predicted_gain"]
            key = (2 * ns + nms, ns, False)
            if key > best:
                winner, best = candidate["rank"], key
        assert winner == example["selection"]
        before, after = example["current"]["structure"], example["output"]["structure"]
        if name == "KEEP":
            assert before == after
        else:
            selected = next(row for row in example["candidates"] if row["rank"] == winner)
            if not selected["commit"]["lattice_changed"]:
                assert before["lattice"]["matrix"] == after["lattice"]["matrix"]
            for i, (a, b) in enumerate(zip(before["sites"], after["sites"])):
                for axis in range(3):
                    if 8 + 4 * i + axis not in selected["commit"]["changed"]:
                        assert a["abc"][axis] == b["abc"][axis]
    assert examples["training_examples"]["positive"]["accept_target"] == 1
    assert examples["training_examples"]["negative"]["accept_target"] == 0
    assert "content_target_tokens" not in examples["training_examples"]["negative"]
    assert examples["training_distribution"]["content_rows"] == len(content_visits)
    print(json.dumps({"status": "passed", "hashed_files": len(manifest["files"]),
                      "snapshots": 2, "structure_score_bindings": 2100, "paired_requests": 1050,
                      "historical_rows": len(history["rows"]), "keep_edit_examples_replayed": 2,
                      "models_or_physics_run": False}))


if __name__ == "__main__":
    main()
