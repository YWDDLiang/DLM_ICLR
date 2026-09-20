"""Exact cumulative Direct metrics from disjoint blocks; no repeated distances."""

from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from dlm_iclr.runtime.io import read_json, read_rows, write_json, write_rows


def merge_direct_blocks(blocks, output):
    """Merge blocks evaluated with save_aggregation=True and one fixed reference.

    Coverage recall uses elementwise minima over blocks, independently for the
    structure and composition distances, as in CrysLLMGen. Property distances
    use the union of valid samples, never the mean of block Wasserstein values.
    """
    blocks = [Path(path) for path in blocks]
    reports = [read_json(path / "summary.json") for path in blocks]
    if not reports:
        raise ValueError("At least one completed Direct block is required")
    first = reports[0]
    identity = ("reference_sha256", "reference_requests", "dataset", "coverage_cutoffs",
                "composition_policy", "packages", "valid_semantics")
    if any(any(report[key] != first[key] for key in identity) for report in reports):
        raise ValueError("Direct blocks use different references or evaluation protocols")
    rows = [row for block in blocks for row in read_rows(block / "scores.jsonl")]
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Direct aggregation blocks must contain disjoint requests")
    arrays = []
    for path in blocks:
        with np.load(path / "aggregation.npz", allow_pickle=False) as data:
            arrays.append({key: data[key] for key in data.files})
    structure_cutoff = first["coverage_cutoffs"]["structure"]
    composition_cutoff = first["coverage_cutoffs"]["composition"]
    recall_s = np.minimum.reduce([a["recall_structure"] for a in arrays])
    recall_c = np.minimum.reduce([a["recall_composition"] for a in arrays])
    precision_count = sum(int(np.sum((a["precision_structure"] <= structure_cutoff) &
                                    (a["precision_composition"] <= composition_cutoff))) for a in arrays)
    report = deepcopy(first)
    fields = ("comp_valid", "struct_valid", "valid", "V")
    for row in rows:
        predicates = row["comp_valid"], row["struct_valid"]
        row["V"] = False if False in predicates else None if None in predicates else True
    known = {key: sum(row[key] is True for row in rows) for key in fields}
    unknown = {key: sum(row[key] is None for row in rows) for key in fields}
    counts = {key: None if unknown[key] else known[key] for key in fields}
    percent = {key: None if counts[key] is None else 100 * counts[key] / len(rows) for key in fields}
    metrics = {key: None if value is None else round(value, 4) for key, value in percent.items()}
    metrics.update(
        cov_recall=round(100 * float(np.mean((recall_s <= structure_cutoff) &
                                            (recall_c <= composition_cutoff))), 4),
        cov_precision=round(100 * precision_count / len(rows), 4),
    )
    for name, field in (("wdist_density", "density"), ("wdist_num_elems", "num_elems")):
        values = np.concatenate([a[field] for a in arrays])
        metrics[name] = (round(float(wasserstein_distance(values, arrays[0]["reference_" + field])), 4)
                         if len(values) and not unknown["valid"] else None)
    report.pop("distribution_error", None)
    if metrics["wdist_density"] is None:
        report["distribution_error"] = "unresolved_validity_or_no_valid_fingerprintable_structures"
    report.update(
        blocks=[str(path) for path in blocks], aggregation="exact_union_of_disjoint_blocks",
        counts={"requests": len(rows), **counts}, known_counts=known, unknown_counts=unknown,
        count_bounds={key: [known[key], known[key] + unknown[key]] for key in fields},
        percent=percent, metrics=metrics, reported_metrics=list(metrics),
        complete=all(value is not None for value in metrics.values()),
        reconstructed=sum(report["reconstructed"] for report in reports),
        fingerprint_failures=sum(report["fingerprint_failures"] for report in reports),
        distribution_valid_samples=sum(report["distribution_valid_samples"] for report in reports),
        V_semantics="comp_valid_AND_struct_valid",
    )
    for key in ("structure_hits", "structure_misses", "composition_hits", "composition_misses"):
        report["validity_cache"][key] = sum(r["validity_cache"][key] for r in reports)
    for key in ("cache_hits", "computed"):
        report["fingerprints"][key] = sum(r["fingerprints"][key] for r in reports)
    write_rows(Path(output) / "scores.jsonl", rows)
    write_json(Path(output) / "summary.json", report)
    return rows, report
