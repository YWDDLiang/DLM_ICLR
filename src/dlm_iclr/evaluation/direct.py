"""Direct generation metrics, with a true composition/structure-only fast path."""

from __future__ import annotations
from collections import Counter
import csv
from functools import lru_cache
import importlib.metadata
import math
from pathlib import Path
from dlm_iclr.evaluation.inputs import normalize_records, reconstruct
from dlm_iclr.runtime.io import file_hash, fingerprint, read_json, read_rows, write_json, write_rows

BASIC_METRICS = ("comp_valid", "struct_valid")
FULL_METRICS = (*BASIC_METRICS, "valid", "wdist_density", "wdist_num_elems", "cov_recall", "cov_precision")
COVERAGE_CUTOFFS = {"mp20": (0.4, 10.0), "carbon": (0.2, 4.0), "perovskite": (0.2, 4.0)}


def _persist_cache(path, value):
    try:
        write_json(path, value)
    except PermissionError:
        # Windows may deny replacement while a peer reads the same immutable entry.
        if read_json(path) != value:
            raise
        import os

        path.with_name(f".{path.name}.{os.getpid()}.tmp").unlink(missing_ok=True)


def _basic_shard(payload):
    records, output, cache, composition = payload
    return evaluate_direct(
        records, output, metrics="comp_struct", workers=1, cache=cache, composition=composition
    )


def _parallel_basic(records, output, workers, cache, composition):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import tempfile

    records = normalize_records(records)
    workers = min(workers, len(records))
    width = math.ceil(len(records) / workers)
    with tempfile.TemporaryDirectory(prefix="dlm_direct_") as temporary:
        tasks = [
            (
                records[i : i + width],
                str(Path(temporary) / str(i)),
                str(cache) if cache else None,
                composition,
            )
            for i in range(0, len(records), width)
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_basic_shard, tasks))
    rows = [row for values, _ in results for row in values]
    for row, original in zip(rows, records, strict=True):
        row["ordinal"] = original["ordinal"]
    report = results[0][1]
    known = {name: sum(row[name] is True for row in rows) for name in BASIC_METRICS}
    unknown = {name: sum(row[name] is None for row in rows) for name in BASIC_METRICS}
    counts = {name: None if unknown[name] else known[name] for name in BASIC_METRICS}
    percent = {
        name: None if counts[name] is None else 100 * counts[name] / len(rows) for name in BASIC_METRICS
    }
    cache_report = {"path": report["validity_cache"]["path"]}
    for key in ("structure_hits", "structure_misses", "composition_hits", "composition_misses"):
        cache_report[key] = sum(result["validity_cache"][key] for _, result in results)
    report.update(
        reconstructed=sum(row["reconstructed"] for row in rows),
        workers=workers,
        counts={"requests": len(rows), **counts},
        known_counts=known,
        unknown_counts=unknown,
        count_bounds={name: [known[name], known[name] + unknown[name]] for name in BASIC_METRICS},
        percent=percent,
        metrics={name: None if value is None else round(value, 4) for name, value in percent.items()},
        complete=all(value is not None for value in percent.values()),
        validity_cache=cache_report,
    )
    write_rows(Path(output) / "scores.jsonl", rows)
    write_json(Path(output) / "summary.json", report)
    return rows, report


@lru_cache(maxsize=8192)
def _composition_valid(elements, amounts, composition="smact3_mixed"):
    from dlm_iclr._vendor.crysllmgen.validity import smact_validity

    if composition == "smact3_mixed":
        from pymatgen.core import Element, Composition
        from .composition import composition_validity

        formula = Composition({Element.from_Z(z).symbol: n for z, n in zip(elements, amounts)}).formula
        return composition_validity(formula)["comp_valid"]
    return bool(smact_validity(elements, amounts))


@lru_cache(maxsize=2)
def _reference_structures_cached(path, modified_ns, size):
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


def _reference_structures(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _reference_structures_cached(str(path), stat.st_mtime_ns, stat.st_size)


def evaluate_direct(
    records,
    output,
    *,
    metrics="full",
    reference=None,
    dataset="mp20",
    workers=1,
    cache=None,
    composition="smact3_mixed",
    coverage_cutoffs=None,
    save_aggregation=False,
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
    if metrics == "full" and dataset not in COVERAGE_CUTOFFS and coverage_cutoffs is None:
        raise ValueError(f"Unknown Direct coverage cutoff preset: {dataset}")
    if metrics == "comp_struct" and workers > 1 and len(records) > 1:
        return _parallel_basic(records, output, workers, cache, composition)
    from dlm_iclr._vendor.crysllmgen.validity import structure_validity

    implementation = file_hash(Path(__file__).parents[1] / "_vendor/crysllmgen/validity.py")
    packages = {name: importlib.metadata.version(name) for name in ("pymatgen", "smact", "numpy")}
    validity_cache = None
    cache_counts = {
        "structure_hits": 0,
        "structure_misses": 0,
        "composition_hits": 0,
        "composition_misses": 0,
    }
    if cache is not None:
        definition = {
            "version": 1,
            "composition": composition,
            "implementation": implementation,
            "composition_implementation": file_hash(Path(__file__).with_name("composition.py")),
            "packages": packages,
        }
        validity_cache = Path(cache) / "validity" / fingerprint(definition)[:24]

    def composition_result(elements, amounts):
        target = None
        if validity_cache is not None:
            target = validity_cache / "composition" / (fingerprint([elements, amounts]) + ".json")
            if target.exists():
                cache_counts["composition_hits"] += 1
                return read_json(target)["comp_valid"]
        value = bool(_composition_valid(elements, amounts, composition))
        cache_counts["composition_misses"] += 1
        if target is not None:
            _persist_cache(target, {"comp_valid": value})
        return value

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
            target = None
            if validity_cache is not None:
                key = fingerprint(
                    {
                        "lattice": structure.lattice.matrix.tolist(),
                        "fractional_coordinates": structure.frac_coords.tolist(),
                        "atomic_numbers": [int(z) for z in structure.atomic_numbers],
                    }
                )
                target = validity_cache / "structure" / key[:2] / (key + ".json")
                if target.exists():
                    row.update(read_json(target))
                    cache_counts["structure_hits"] += 1
                    rows.append(row)
                    structures.append(structure)
                    continue
            cache_counts["structure_misses"] += 1
            counts = Counter(int(value) for value in structure.atomic_numbers)
            elements = tuple(sorted(counts))
            amounts = tuple(counts[element] for element in elements)
            divisor = math.gcd(*amounts)
            for name, operation in (
                (
                    "comp_valid",
                    lambda: composition_result(elements, tuple(value // divisor for value in amounts)),
                ),
                ("struct_valid", lambda: bool(structure_validity(structure))),
            ):
                try:
                    row[name] = operation()
                except Exception as error:
                    row[name] = None
                    row["metric_errors"][name] = f"{type(error).__name__}: {error}"
            if target is not None and not row["metric_errors"]:
                _persist_cache(target, {name: row[name] for name in BASIC_METRICS})
        rows.append(row)
        structures.append(structure)
    output = Path(output)
    report = {
        "schema": "dlm_iclr_direct_v1",
        "mode": metrics,
        "composition_policy": composition,
        "reported_metrics": list(FULL_METRICS if metrics == "full" else BASIC_METRICS),
        "omitted_metrics": list(FULL_METRICS[2:] if metrics == "comp_struct" else ()),
        "denominator": "all_requested_structures_in_input_order",
        "geometry": "saved_output_before_physical_relaxation",
        "packages": packages,
        "validity_implementation_sha256": implementation,
        "validity_cache": {
            "path": str(validity_cache) if validity_cache is not None else None,
            **cache_counts,
        },
        "reconstructed": sum(row["reconstructed"] for row in rows),
    }
    extra = {}
    if metrics == "full":
        # This import and all expensive reference/fingerprint work are skipped
        # entirely by comp_struct, which also needs no config or model assets.
        from dlm_iclr.evaluation.features import compute_features, coverage
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
        struc_cutoff, comp_cutoff = coverage_cutoffs or COVERAGE_CUTOFFS[dataset]
        cov, distances = coverage(
            generated, ground_truth, struc_cutoff=struc_cutoff, comp_cutoff=comp_cutoff,
            requested=len(rows), return_distances=True,
        )
        if save_aggregation:
            import numpy as np
            import os

            output.mkdir(parents=True, exist_ok=True)
            destination = output / "aggregation.npz"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, **distances,
                    density=np.asarray([float(s.density) for s in selected]),
                    num_elems=np.asarray([len(set(s.species)) for s in selected]),
                    reference_density=np.asarray([float(s.density) for s in reference_structures]),
                    reference_num_elems=np.asarray([len(set(s.species)) for s in reference_structures]),
                )
            temporary.replace(destination)
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
