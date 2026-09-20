"""Convert CSV, JSONL or CIF directories through a single crystal interface."""

import csv
import math
from pathlib import Path
from .plans import composition_key
from ..runtime.io import fingerprint, read_rows, write_rows, write_json
from ..runtime.config import path, run_root, asset


def iter_structures(source, fields=None):
    source = Path(source)
    fields = fields or {}
    if source.is_dir():
        rows = ({"id": p.stem, "cif": p.read_text(encoding="utf-8")} for p in sorted(source.glob("*.cif")))
    elif source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        rows = (row for batch in pq.ParquetFile(source).iter_batches(batch_size=1024) for row in batch.to_pylist())
    else:
        rows = read_rows(source)
    for index, original in enumerate(rows):
        row = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in original.items()}
        row.update({key: row[value] for key, value in fields.items() if value in row})
        yield index, str(row.get("id") or row.get("material_id") or index), row


def structure_from_row(row):
    """Dataset coordinates are explicitly Cartesian or fractional, never inferred by range."""
    from pymatgen.core import Structure

    if row.get("structure"):
        return Structure.from_dict(row["structure"])
    if "positions" in row and "cell" in row and "atomic_numbers" in row:
        return Structure(row["cell"], row["atomic_numbers"], row["positions"],
                         coords_are_cartesian=True, to_unit_cell=True)
    return Structure.from_str(row["cif"], fmt="cif")


def align_sites(arrays, plan):
    order = [
        i for element in plan["elements"] for i, species in enumerate(arrays["species"]) if species == element
    ]
    return dict(
        arrays,
        species=[arrays["species"][i] for i in order],
        frac_coords=[arrays["frac_coords"][i] for i in order],
    ), order


def prepare(config):
    from .parallel_prepare import prepare as implementation
    return implementation(config)
