"""Direct generation metrics, with a true composition/structure-only fast path."""

from __future__ import annotations
from collections import Counter
import csv
from functools import lru_cache
import importlib.metadata
import math
from pathlib import Path
from .evaluation_inputs import normalize_records, reconstruct
from .io import file_hash, read_rows, write_json, write_rows

BASIC_METRICS = ("comp_valid", "struct_valid")
FULL_METRICS = (*BASIC_METRICS, "valid", "wdist_density", "wdist_num_elems", "cov_recall", "cov_precision")
COVERAGE_CUTOFFS = {"mp20": (0.4, 10.0), "carbon": (0.2, 4.0), "perovskite": (0.2, 4.0)}


@lru_cache(maxsize=8192)
def _composition_valid(elements, amounts):
    from ._vendor.crysllmgen.validity import smact_validity

    return bool(smact_validity(elements, amounts))


def _reference_structures(path):
    from pymatgen.core import Structure

    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = read_rows(path)
    structures = []
    for index, row in enumerate(rows):
        try:
            payload = row.get("structure") or row.get("target_structure")
            structure = Structure.from_dict(payload) if payload else Structure.from_str(row["cif"], fmt="cif")
            structures.append(reconstruct({"success": True, "structure": structure.as_dict()}))
        except Exception as error:
            raise ValueError(f"Direct reference row {index} cannot be reconstructed: {error}") from error
    if not structures:
        raise ValueError("Direct reference contains no structures")
    return structures


def evaluate_direct(
    records, output, *, metrics="full", reference=None, dataset="mp20", workers=1, cache=None
):
    """Full frozen generation metrics, or only comp_valid/struct_valid on CPU.

    Full mode follows CrysLLMGen's fingerprint-gated ``valid`` and distribution
    subset. Its coverage uses every fingerprintable prediction, including
    chemically invalid predictions, and the original request denominator.
    """
    if metrics not in ("full", "comp_struct"):
        raise ValueError("Direct metrics must be 'full' or 'comp_struct'")
    if workers < 1:
        raise ValueError("Direct fingerprint workers must be positive")
    if metrics == "full" and (reference is None or not Path(reference).is_file()):
        raise ValueError("Full Direct evaluation requires an existing --reference CSV or JSONL")
    if metrics == "full" and dataset not in COVERAGE_CUTOFFS:
        raise ValueError(f"Unknown Direct coverage cutoff preset: {dataset}")
    from ._vendor.crysllmgen.validity import structure_validity

    records = normalize_records(records)
    rows, structures = [], []
    for record in records:
        row = {
            "source_id": record["source_id"],
            "ordinal": record["ordinal"],
            "reconstructed": False,
            "comp_valid": False,
            "struct_valid": False,
            "parser_error": None,
            "metric_errors": {},
        }
        try:
            structure = reconstruct(record)
        except Exception as error:
            structure = None
            row["parser_error"] = f"{type(error).__name__}: {error}"
        if structure is not None:
            row["reconstructed"] = True
            counts = Counter(int(value) for value in structure.atomic_numbers)
            elements = tuple(sorted(counts))
            amounts = tuple(counts[element] for element in elements)
            divisor = math.gcd(*amounts)
            for name, operation in (
                (
                    "comp_valid",
                    lambda: _composition_valid(elements, tuple(value // divisor for value in amounts)),
                ),
                ("struct_valid", lambda: bool(structure_validity(structure))),
            ):
                try:
                    row[name] = operation()
                except Exception as error:
                    row[name] = None
                    row["metric_errors"][name] = f"{type(error).__name__}: {error}"
        rows.append(row)
        structures.append(structure)
    output = Path(output)
    report = {
        "schema": "dlm_iclr_direct_v1",
        "mode": metrics,
        "reported_metrics": list(FULL_METRICS if metrics == "full" else BASIC_METRICS),
        "omitted_metrics": list(FULL_METRICS[2:] if metrics == "comp_struct" else ()),
        "denominator": "all_requested_structures_in_input_order",
        "geometry": "saved_output_before_physical_relaxation",
        "packages": {name: importlib.metadata.version(name) for name in ("pymatgen", "smact", "numpy")},
        "validity_implementation_sha256": file_hash(Path(__file__).parent / "_vendor/crysllmgen/validity.py"),
        "reconstructed": sum(row["reconstructed"] for row in rows),
    }
    extra = {}
    if metrics == "full":
        # This import and all expensive reference/fingerprint work are skipped
        # entirely by comp_struct, which also needs no config or model assets.
        from .direct_features import compute_features, coverage
        from scipy.stats import wasserstein_distance

        reference_structures = _reference_structures(reference)
        features, feature_report = compute_features(
            structures + reference_structures,
            cache=cache or output / "cache",
            workers=workers,
        )
        generated, ground_truth = features[: len(rows)], features[len(rows) :]
        failed_reference = [i for i, row in enumerate(ground_truth) if row["error"] is not None]
        if failed_reference:
            raise ValueError(
                "Direct reference fingerprints failed; no reference rows were dropped: "
                + str([(i, ground_truth[i]["error"]) for i in failed_reference[:10]])
            )
        selected = []
        for index, (row, feature) in enumerate(zip(rows, generated, strict=True)):
            row["fingerprint_valid"] = feature["error"] is None
            row["fingerprint_error"] = feature["error"]
            predicates = (row["comp_valid"], row["struct_valid"], row["fingerprint_valid"])
            row["valid"] = False if False in predicates else None if None in predicates else True
            if row["valid"] is True:
                selected.append(structures[index])
        extra = {"wdist_density": None, "wdist_num_elems": None}
        if selected:
            extra.update(
                wdist_density=round(
                    float(
                        wasserstein_distance(
                            [float(value.density) for value in selected],
                            [float(value.density) for value in reference_structures],
                        )
                    ),
                    4,
                ),
                wdist_num_elems=round(
                    float(
                        wasserstein_distance(
                            [len(set(value.species)) for value in selected],
                            [len(set(value.species)) for value in reference_structures],
                        )
                    ),
                    4,
                ),
            )
        struc_cutoff, comp_cutoff = COVERAGE_CUTOFFS[dataset]
        cov = coverage(
            generated, ground_truth, struc_cutoff=struc_cutoff, comp_cutoff=comp_cutoff, requested=len(rows)
        )
        extra.update({name: round(100 * value, 4) for name, value in cov.items()})
        report.update(
            reference_sha256=file_hash(reference),
            reference_requests=len(reference_structures),
            dataset=dataset,
            fingerprints=feature_report,
            distribution_valid_samples=len(selected),
            fingerprint_failures=sum(row["error"] is not None for row in generated),
            coverage_cutoffs={"structure": struc_cutoff, "composition": comp_cutoff},
            coverage_semantics="independent_structure_and_composition_nearest_distances",
            valid_semantics="comp_valid_AND_struct_valid_AND_fingerprint_available",
        )
        if not selected:
            report["distribution_error"] = "no_valid_fingerprintable_generated_structures"
        if any(row["valid"] is None for row in rows):
            extra["wdist_density"] = extra["wdist_num_elems"] = None
            report["distribution_error"] = "unresolved_generated_validity"
    fields = (*BASIC_METRICS, "valid") if metrics == "full" else BASIC_METRICS
    known = {field: sum(row[field] is True for row in rows) for field in fields}
    unknown = {field: sum(row[field] is None for row in rows) for field in fields}
    counts = {field: None if unknown[field] else known[field] for field in fields}
    percent = {field: None if counts[field] is None else 100 * counts[field] / len(rows) for field in fields}
    report.update(
        counts={"requests": len(rows), **counts},
        known_counts=known,
        unknown_counts=unknown,
        count_bounds={field: [known[field], known[field] + unknown[field]] for field in fields},
        percent=percent,
        metrics={
            **{key: None if value is None else round(value, 4) for key, value in percent.items()},
            **extra,
        },
    )
    report["complete"] = all(value is not None for value in report["metrics"].values())
    write_rows(output / "scores.jsonl", rows)
    write_json(output / "summary.json", report)
    return rows, report
