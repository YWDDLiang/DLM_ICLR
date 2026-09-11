"""All-request stability, novelty and directed uniqueness after output selection."""

from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import gzip
import math
from pathlib import Path
from crystal_dlm.continuous_keep_edit import structure_of
from crystal_dlm.exact_sun_nu import conjunction, evaluate_sun_predicates
from .io import file_hash, fingerprint, read_json, read_rows, write_json, write_rows
from .physics import record_key


def load_training_index(path, *, cache=None):
    from pymatgen.core import Structure

    path = Path(path)
    digest = file_hash(path)
    cache_path = Path(cache) / (digest + ".json.gz") if cache else None
    if cache_path and cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            import json

            payload = json.load(stream)
        structures = [Structure.from_dict(value) for value in payload["structures"]]
        return (
            structures,
            payload["formula_index"],
            {"source_sha256": digest, "parse_failures": payload["parse_failures"]},
        )
    structures, index, failures = [], defaultdict(list), []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = read_rows(path)
    for ordinal, row in enumerate(rows):
        try:
            structure = (
                Structure.from_str(row["cif"], fmt="cif")
                if row.get("cif")
                else Structure.from_dict(row["structure"])
            )
            index[structure.composition.reduced_formula].append(len(structures))
            structures.append(structure)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"ordinal": ordinal, "error": str(error)})
    if cache_path:
        import json

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "structures": [value.as_dict() for value in structures],
                    "formula_index": dict(index),
                    "parse_failures": failures,
                },
                stream,
            )
    return structures, dict(index), {"source_sha256": digest, "parse_failures": failures}


class HullReference:
    def __init__(self, path):
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        from pymatgen.core import Composition

        path = Path(path)
        self.path = path / "official_slim_cache.jsonl" if path.is_dir() else path
        self.sha256 = file_hash(self.path)
        self.rows = {}
        for row in read_rows(self.path):
            system = row["chemsys"]
            if system in self.rows:
                raise ValueError(f"Duplicate hull reference chemistry: {system}")
            self.rows[system] = row
        self.diagrams = {}
        self.unavailable = {}
        self.PDEntry, self.PhaseDiagram, self.Composition = PDEntry, PhaseDiagram, Composition

    def energy(self, structure):
        system = "-".join(sorted(element.symbol for element in structure.composition.elements))
        if system not in self.rows or system in self.unavailable:
            return system, None
        if system not in self.diagrams:
            entries = [
                self.PDEntry(
                    self.Composition(row["composition"]),
                    float(row["energy"]),
                    name=str(row.get("entry_id", "")),
                )
                for row in self.rows[system]["entries"]
            ]
            covered = {element.symbol for entry in entries for element in entry.composition.elements}
            missing = set(system.split("-")) - covered
            if missing:
                self.unavailable[system] = "missing_reference_elements:" + ",".join(sorted(missing))
                return system, None
            try:
                self.diagrams[system] = self.PhaseDiagram(entries)
            except ValueError as error:
                self.unavailable[system] = str(error)
                return system, None
        return system, float(self.diagrams[system].get_hull_energy_per_atom(structure.composition))


def summarize(rows):
    fields = (
        "reconstructed",
        "comp_valid",
        "struct_valid",
        "strict_stable",
        "meta_stable",
        "strict_sun",
        "meta_sun",
        "terminal_verified",
        "verified_strict_sun",
        "verified_meta_sun",
    )
    counts = {
        field: None if any(row[field] is None for row in rows) else sum(row[field] is True for row in rows)
        for field in fields
    }
    counts["requests"] = len(rows)
    known_counts = {field: sum(row[field] is True for row in rows) for field in fields}
    unknown_counts = {field: sum(row[field] is None for row in rows) for field in fields}
    return {
        "counts": counts,
        "known_counts": known_counts,
        "unknown_counts": unknown_counts,
        "count_bounds": {
            field: [known_counts[field], known_counts[field] + unknown_counts[field]] for field in fields
        },
        "percent": {
            field: None if counts[field] is None else 100 * counts[field] / max(1, len(rows))
            for field in fields
        },
        "label_statuses": dict(Counter(row["terminal_status"] for row in rows)),
        "missing_hull_systems": sorted({row["chemsys"] for row in rows if row.get("hull_missing")}),
        "denominator": "all_requested_Plans_in_original_order",
        "main_stability": "terminal_energy_minus_reference_hull",
        "novelty_uniqueness_geometry": "output_structure_before_CHGNet_relaxation",
    }


@lru_cache(maxsize=2)
def _reference(path, sha256):
    return HullReference(path)


def score_records(records, labels, hull_cache, novelty_reference, output, *, cache=None, workers=4):
    if len(records) != len(labels):
        raise ValueError("Each requested structure requires one corresponding label result")
    output, cache = Path(output), Path(cache) if cache else Path(output) / "cache"
    reference_path = Path(hull_cache)
    if reference_path.is_dir():
        reference_path /= "official_slim_cache.jsonl"
    reference = _reference(str(reference_path.resolve()), file_hash(reference_path))
    structures, reconstructed, errors, hulls = [], {}, {}, {}
    for i, (record, label) in enumerate(zip(records, labels, strict=True)):
        if label.get("record_key") != record_key(record):
            raise ValueError("Physical label does not belong to this output structure")
        try:
            structure = structure_of(record)
            if not math.isfinite(structure.volume) or structure.volume <= 0:
                raise ValueError("nonpositive or nonfinite cell")
            reconstructed[i] = len(structures)
            structures.append(structure)
        except (ValueError, KeyError, TypeError) as error:
            errors[i] = str(error)
            continue
        hulls[i] = reference.energy(structure)
    required = [
        reconstructed[i]
        for i, label in enumerate(labels)
        if i in reconstructed
        and label.get("terminal_energy") is not None
        and (hulls[i][1] is None or float(label["terminal_energy"]) - hulls[i][1] <= 0.1)
    ]
    novel, unique, nu = [None] * len(structures), [None] * len(structures), None
    if required:
        train, index, identity = load_training_index(novelty_reference, cache=cache / "training_index")
        nu = evaluate_sun_predicates(
            structures,
            train,
            index,
            required,
            cache_dir=cache / "directed_matches",
            frozen_nu_sha256="dlm_iclr_directed_input_NU_v1",
            training_identity=identity,
            workers=workers,
            pair_timeout=30.0,
        )
        novel, unique = nu["novel"], nu["unique"]
    rows = []
    for i, (record, label) in enumerate(zip(records, labels, strict=True)):
        index = reconstructed.get(i)
        system, hull = hulls.get(i, (None, None))
        energy = label.get("terminal_energy")
        is_novel = novel[index] if index is not None else None
        is_unique = unique[index] if index is not None else None
        missing = index is not None and hull is None
        known = energy is not None and hull is not None and math.isfinite(float(energy))
        e_hull = float(energy) - hull if known else None
        # Missing references are unknown; an explicit failed generated endpoint
        # is a failure in the all-request denominator.
        unresolved = (missing and energy is not None) or label["status"] == "worker_error"
        stable = None if unresolved else bool(e_hull <= 0) if known else False
        meta = None if unresolved else bool(e_hull <= 0.1) if known else False
        verified = label.get("verified") is True
        comp_valid, struct_valid = False, False
        if index is not None:
            from ._vendor.crysllmgen.validity import smact_validity, structure_validity

            structure = structures[index]
            counts = Counter(int(value) for value in structure.atomic_numbers)
            elements = tuple(sorted(counts))
            amounts = [counts[element] for element in elements]
            divisor = math.gcd(*amounts)
            comp_valid = bool(smact_validity(elements, tuple(value // divisor for value in amounts)))
            struct_valid = bool(structure_validity(structure))
        row = {
            "source_id": record["source_id"],
            "ordinal": record["ordinal"],
            "record_key": record_key(record),
            "reconstructed": index is not None,
            "comp_valid": comp_valid,
            "struct_valid": struct_valid,
            "parser_error": errors.get(i),
            "terminal_status": label["status"],
            "terminal_verified": verified,
            "raw_energy_eV_atom": label.get("raw_energy"),
            "terminal_energy_eV_atom": energy,
            "e_above_hull_eV_atom": e_hull,
            "hull_energy_eV_atom": hull,
            "chemsys": system,
            "hull_missing": missing,
            "hull_error": reference.unavailable.get(system),
            "novel": is_novel,
            "unique_representative": is_unique,
            "strict_stable": stable,
            "meta_stable": meta,
            "strict_sun": conjunction(stable, is_novel, is_unique),
            "meta_sun": conjunction(meta, is_novel, is_unique),
            "verified_strict_sun": conjunction(verified, stable, is_novel, is_unique),
            "verified_meta_sun": conjunction(verified, meta, is_novel, is_unique),
            "raw": label.get("raw"),
            "terminal": label.get("terminal"),
        }
        rows.append(row)
    report = summarize(rows)
    report.update(
        hull_reference_sha256=reference.sha256,
        novelty_evaluation=nu,
        complete=not report["missing_hull_systems"]
        and not any(row["terminal_status"] == "worker_error" for row in rows)
        and (nu is None or nu["complete_sun_predicates"]),
    )
    write_rows(output / "scores.jsonl", rows)
    write_json(output / "summary.json", report)
    return rows, report


def evaluate_sun(
    config,
    structures,
    output,
    *,
    labels=None,
    devices=("cuda:0",),
    workers_per_device=4,
    nu_workers=4,
    cache=None,
):
    """Evaluate SUN and MSUN from a saved JSONL, optionally reusing bound labels.

    Reusing labels performs no CHGNet calls and requires no generator/editor
    checkpoints. The existing all-request SUN scorer and thresholds are retained.
    """
    from .evaluation_inputs import load_records

    if workers_per_device < 1 or not 1 <= nu_workers <= 64 or not devices:
        raise ValueError("Evaluation needs positive worker counts and at least one device")
    hull_path = Path(config.assets.hull_cache)
    if hull_path.is_dir():
        hull_path /= "official_slim_cache.jsonl"
    if not config.assets.hull_cache or not hull_path.is_file():
        raise ValueError("SUN/MSUN requires assets.hull_cache with the official reference entries")
    if not config.assets.novelty_reference or not Path(config.assets.novelty_reference).is_file():
        raise ValueError("SUN/MSUN requires assets.novelty_reference containing the training structures")
    output = Path(output)
    cache = Path(cache) if cache else output / "cache"
    records = load_records(structures)
    labels_path = Path(labels) if labels is not None else None
    if labels_path is not None:
        physical_labels = read_rows(labels_path)
        if len(physical_labels) != len(records):
            raise ValueError("Saved physical labels must cover every requested structure in input order")
        for record, label in zip(records, physical_labels, strict=True):
            if any(label.get(key) != record[key] for key in ("source_id", "ordinal")):
                raise ValueError("Saved physical label source/order does not match the evaluation input")
            if label.get("record_key") != record_key(record):
                raise ValueError("Physical label does not belong to this output structure")
    else:
        from .physics import label_records
        from .execution import setup_device

        if not config.assets.chgnet or not Path(config.assets.chgnet).is_file():
            raise ValueError("SUN/MSUN requires assets.chgnet unless --labels is supplied")
        setup_device(devices[0])
        physical_labels = label_records(
            records,
            config.assets.chgnet,
            output / "physics",
            devices=devices,
            workers_per_device=workers_per_device,
            cache=cache / "physics",
        )
    rows, report = score_records(
        records,
        physical_labels,
        config.assets.hull_cache,
        config.assets.novelty_reference,
        output,
        cache=cache / "scoring",
        workers=nu_workers,
    )
    report.update(
        input_sha256=file_hash(structures),
        reused_labels_sha256=file_hash(labels_path) if labels_path else None,
        metrics={"SUN": report["percent"]["strict_sun"], "MSUN": report["percent"]["meta_sun"]},
        metric_fields={"SUN": "strict_sun", "MSUN": "meta_sun"},
        stability_thresholds_eV_atom={"SUN": 0.0, "MSUN": 0.1},
    )
    write_json(output / "summary.json", report)
    return rows, report
