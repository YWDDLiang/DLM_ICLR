"""Freeze a fixed Plan panel before generation; never select on generated quality."""
from pathlib import Path
from .plans import load_plans, PRESET_FILES
from ..runtime.config import path, run_root
from ..runtime.io import read_rows, write_rows, write_json, file_hash


def select(config, *, source=None, output=None):
    source = str(source or config["sampling"].get("plans") or "")
    if not source:
        raise ValueError("This dataset requires --source / --plans with its own saved Plan JSONL")
    if source.startswith("preset:"):
        if config["dataset"]["name"] != "mp20":
            raise ValueError("The packaged H1A2 panel belongs to MP20 only")
        location = Path(__file__).parent / "presets" / PRESET_FILES[source[7:]]
    elif source.startswith("@run/"):
        location = run_root(config) / source[5:]
    else:
        location = path(config, source)
    count = config["sampling"]["requests"]
    rows, report = load_plans(location, requests=count, legal_only=True, seed=config["b0"]["seed"])
    output = Path(output) if output else run_root(config) / "plans/evaluation.jsonl"
    if output.exists() and read_rows(output) != rows:
        raise ValueError("Selected Plans changed. Use another output directory.")
    write_rows(output, rows)
    report.update(source_sha256=file_hash(location), selected_sha256=file_hash(output),
                  dataset=config["dataset"]["name"], output=str(output),
                  validity="schema, rich fields, composition counts and dataset atom range; no SMACT or output filtering")
    write_json(output.with_suffix(".manifest.json"), report)
    return report
