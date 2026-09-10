"""Adapters for MP-20 CSV, ordinary CIF CSV/JSONL and directories of CIF files."""

from __future__ import annotations
import csv
from pathlib import Path
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.expert_edit_data import arrays_from_structure
from crystal_dlm.r5_plan_state import plan_state_from_arrays, build_body_prompt
from .io import file_hash, fingerprint, read_rows, write_rows, write_json


def iter_structures(path):
    from pymatgen.core import Structure

    path = Path(path)
    if path.is_dir():
        for index, file in enumerate(sorted(path.glob("*.cif"))):
            yield index, file.stem, {"cif": file.read_text(encoding="utf-8")}
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                yield index, str(row.get("material_id") or row.get("id") or index), row
    else:
        for index, row in enumerate(read_rows(path)):
            yield index, str(row.get("material_id") or row.get("id") or index), row


def convert_dataset(source, output, *, dataset="mp20", split="train"):
    from pymatgen.core import Structure

    output, source = Path(output), Path(source)
    rows, failures = [], []
    for index, identifier, row in iter_structures(source):
        try:
            structure = (
                Structure.from_dict(row["structure"])
                if row.get("structure")
                else Structure.from_str(row["cif"], fmt="cif")
            )
            arrays = arrays_from_structure(structure.as_dict())
            arrays["num_atoms"] = len(structure)
            if not 1 <= len(structure) <= 20:
                raise ValueError("atom_count_outside_trained_representation")
            plan = plan_state_from_arrays(arrays, metadata=row)
            tokens, diagnostic = arrays_to_dynamic_tokens(
                arrays["lengths"], arrays["angles"], arrays["species"], arrays["frac_coords"]
            )
            if diagnostic.length_clips or diagnostic.angle_clips or diagnostic.coord_clips:
                raise ValueError("numeric_values_outside_trained_token_vocabulary")
            source_id = f"{dataset}:{split}:{identifier}:{index}"
            record = {
                "schema": "crystal_plan_v1",
                "source_id": source_id,
                "original_ordinal": index,
                "body_eligible": True,
                "plan_state": plan,
                "body_prompt": build_body_prompt(plan).rstrip() + "\n",
                "target_body": "".join(tokens),
                "target_structure": structure.as_dict(),
                "provenance": {
                    "dataset_origin": dataset,
                    "original_split": split,
                    "usage_role": "train" if split == "train" else "evaluation",
                    "source_row_index": index,
                    "material_id": identifier,
                },
            }
            for name in ("body_noise_seed", "refiner_noise_seed"):
                record[name] = int(fingerprint([source_id, name])[:16], 16) & ((1 << 63) - 1)
            rows.append(record)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"source_row_index": index, "material_id": identifier, "reason": str(error)})
    write_rows(output, rows)
    report = {
        "dataset_origin": dataset,
        "original_split": split,
        "converted": len(rows),
        "failures": failures,
        "source_sha256": file_hash(source) if source.is_file() else None,
        "representation": "dynamic_v1_7_plus_4N",
        "supported_sites": [1, 20],
    }
    write_json(output.with_suffix(".conversion.json"), report)
    return report
