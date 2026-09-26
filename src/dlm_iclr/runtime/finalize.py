"""Finalize the physically evaluated output collection without overwriting proposals."""

from pathlib import Path

from .config import run_root
from .io import file_hash, fingerprint, read_json, read_rows, write_json, write_rows


def _comparison_context(root):
    contexts = []
    for endpoint in ("refined", "edited"):
        directory = root / "evaluation" / endpoint
        physical = read_json(directory / "physics/protocol.json")
        scoring = read_json(directory / "scoring/summary.json")
        training = (scoring.get("novelty_evaluation") or {}).get("training_identity", {})
        contexts.append(
            {
                "physics": physical,
                "hull": scoring["hull_reference_sha256"],
                "training": training.get("source_sha256"),
            }
        )
    a, b = contexts
    if (
        a["physics"] != b["physics"]
        or a["hull"] != b["hull"]
        or (a["training"] is not None and b["training"] is not None and a["training"] != b["training"])
    ):
        raise ValueError("Physical rollback requires matching physical, hull and novelty reference protocols")
    return {"physics": a["physics"], "hull": a["hull"], "training": a["training"] or b["training"]}


def finalize(config, root, edited_report):
    from ..evaluation.direct import evaluate_direct
    from ..evaluation.workflow import evaluate
    from ..feedback.physical_rollback import RULE, select

    root = Path(root)
    edited = read_rows(root / "edited.jsonl")
    enabled = config["feedback"]["physical_rollback"]
    definition = {
        "physical_rollback": enabled,
        "protect_sun": config["feedback"]["protect_sun"],
        "edited_sha256": file_hash(root / "edited.jsonl"),
        "evaluation": config["evaluation"],
    }
    evidence = {}
    if enabled:
        paths = [
            "refined.jsonl",
            "evaluation/refined/scores.jsonl",
            "evaluation/edited/scores.jsonl",
            "evaluation/refined/physics/labels.jsonl",
            "evaluation/edited/physics/labels.jsonl",
            "evaluation/refined/physics/protocol.json",
            "evaluation/edited/physics/protocol.json",
            "evaluation/refined/scoring/summary.json",
            "evaluation/edited/scoring/summary.json",
        ]
        missing = [p for p in paths if not (root / p).is_file()]
        if missing:
            raise ValueError(
                "Evaluate references and reconstructions before physical rollback: " + ", ".join(missing)
            )
        evidence = {p: file_hash(root / p) for p in paths}
        definition["refined_sha256"] = evidence["refined.jsonl"]
        definition["comparison_context"] = _comparison_context(root)
        definition["rule"] = RULE
    signature = fingerprint(definition)
    receipt = root / "finalization.settings.json"
    if receipt.exists() and read_json(receipt)["signature"] != signature:
        raise ValueError("Final selection inputs or policy changed. Use a new output directory.")
    write_json(receipt, {"signature": signature, "definition": definition, "evaluation_artifacts": evidence})
    if enabled:
        records, labels, decisions = select(
            read_rows(root / "refined.jsonl"),
            edited,
            read_rows(root / "evaluation/refined/scores.jsonl"),
            read_rows(root / "evaluation/edited/scores.jsonl"),
            read_rows(root / "evaluation/refined/physics/labels.jsonl"),
            read_rows(root / "evaluation/edited/physics/labels.jsonl"),
        )
        write_rows(root / "rollback.jsonl", records)
        out = root / "evaluation/rollback"
        write_rows(out / "physics/labels.jsonl", labels)
        write_rows(out / "decisions.jsonl", decisions)
        # U/N and joint rates belong to the complete selected cohort, not a splice of old scores.
        _, physical = evaluate(config, records, out, labels=labels)
        _, direct = evaluate_direct(
            records,
            root / "direct/rollback",
            metrics=config["evaluation"]["direct"],
            reference=run_root(config) / "data/structures/test.jsonl",
            dataset=config["dataset"]["name"],
            workers=config["runtime"]["matching_workers"],
            composition=config["evaluation"]["composition"],
            coverage_cutoffs=config["evaluation"]["coverage_cutoffs"],
            cache=run_root(config) / "cache/direct",
        )
        endpoint = "rollback"
        restored = sum(d["restored_reference"] for d in decisions)
    else:
        records = edited
        direct, physical = edited_report["direct"], edited_report["sun"]
        endpoint, restored = "edited", 0
    write_rows(root / "final.jsonl", records)
    report = {
        "endpoint": endpoint,
        "structures": str(root / "final.jsonl"),
        "requests": len(records),
        "physical_rollback": enabled,
        "restored_references": restored,
        "protect_sun": config["feedback"]["protect_sun"],
        "direct": direct,
        "sun": physical,
    }
    write_json(root / "final.summary.json", report)
    return report
