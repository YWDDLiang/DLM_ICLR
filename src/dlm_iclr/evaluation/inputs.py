"""Preserve every saved generation request when evaluating standalone files."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from dlm_iclr.runtime.io import read_rows


def normalize_records(records):
    result, identifiers, ordinals = [], set(), set()
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            raise ValueError(f"Evaluation row {index} must be a JSON object")
        row = dict(value)
        row.setdefault("source_id", str(row.get("attempt_id", index)))
        row.setdefault("ordinal", index)
        if not isinstance(row["source_id"], str) or not row["source_id"]:
            raise ValueError(f"Evaluation row {index} requires a nonempty source_id")
        if type(row["ordinal"]) is not int or row["ordinal"] < 0:
            raise ValueError(f"Evaluation row {index} requires a nonnegative integer ordinal")
        if row["source_id"] in identifiers or row["ordinal"] in ordinals:
            raise ValueError("Evaluation source IDs and ordinals must be unique; rows are never deduplicated")
        identifiers.add(row["source_id"])
        ordinals.add(row["ordinal"])
        if "success" not in row:
            row["success"] = (
                row["status"] == "succeeded"
                if "status" in row
                else row.get("structure") is not None or bool(row.get("body"))
            )
        if type(row["success"]) is not bool:
            raise ValueError(f"Evaluation row {index} requires a boolean success field")
        result.append(row)
    if not result:
        raise ValueError("Evaluation input contains no requested structures")
    return result


def load_records(path):
    return normalize_records(read_rows(Path(path)))


def reconstruct(record):
    from dlm_iclr._core.continuous_keep_edit import structure_of

    if not record["success"]:
        raise ValueError("generation_failure:" + str(record.get("reason", record.get("status", "failed"))))
    structure = structure_of(record)
    if not len(structure) or not structure.is_ordered:
        raise ValueError("empty_or_disordered_structure")
    if (
        not np.isfinite(structure.lattice.matrix).all()
        or not np.isfinite(structure.frac_coords).all()
        or not np.isfinite(structure.volume)
        or structure.volume <= 0
    ):
        raise ValueError("nonfinite_geometry_or_nonpositive_volume")
    return structure
