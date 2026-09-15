"""Materials Project GGA/GGA+U reference energies with subsystem coverage."""

import itertools
import os
from pathlib import Path
from ..runtime.io import read_json, read_rows, write_json, write_rows


def chemical_system(record):
    if record.get("plan_state"):
        return "-".join(sorted(record["plan_state"]["elements"]))
    from .._core.continuous_keep_edit import structure_of

    return "-".join(sorted(e.symbol for e in structure_of(record).composition.elements))


def subsystems(system):
    elements = system.split("-")
    return [
        "-".join(group) for n in range(1, len(elements) + 1) for group in itertools.combinations(elements, n)
    ]


def query(records, output, *, api_key=None, batch_size=32, client=None):
    """Retrieve corrected competitor entries; candidate energy comes from CHGNet."""
    from contextlib import nullcontext

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parents = sorted(
        {chemical_system(row) for row in records if row.get("success", row.get("body_eligible", True))}
    )
    required = sorted({s for parent in parents for s in subsystems(parent)})
    if client is None:
        key = api_key or os.environ.get("MP_API_KEY")
        if not key:
            import getpass

            key = getpass.getpass("Materials Project API key: ")
        # MP entry payloads also occur under the newer pymatgen module names.
        import importlib
        import sys

        for new, old in {
            "pymatgen.core.entries": "pymatgen.entries.computed_entries",
            "pymatgen.analysis.compatibility": "pymatgen.entries.compatibility",
            "pymatgen.analysis.mixing_scheme": "pymatgen.entries.mixing_scheme",
        }.items():
            try:
                importlib.import_module(new)
            except ModuleNotFoundError as error:
                if error.name != new:
                    raise
                sys.modules[new] = importlib.import_module(old)
        from mp_api.client import MPRester

        connection = MPRester(key, mute_progress_bars=True)
    else:
        connection = nullcontext(client)
    with connection as mp:
        version = mp.get_database_version()
        directory = output / "entries" / str(version)
        pending = [system for system in required if not (directory / f"{system}.json").exists()]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            entries = mp.get_entries(
                batch, compatible_only=True, additional_criteria={"thermo_types": ["GGA_GGA+U"]}
            )
            grouped = {s: [] for s in batch}
            for entry in entries:
                system = "-".join(sorted(e.symbol for e in entry.composition.elements))
                if system in grouped:
                    grouped[system].append(
                        {
                            "composition": entry.composition.get_el_amt_dict(),
                            "energy": float(entry.energy),
                            "entry_id": str(entry.entry_id),
                        }
                    )
            for system, values in grouped.items():
                write_json(directory / f"{system}.json", values)
            print(
                {"hull_systems_complete": min(start + batch_size, len(pending)), "new_systems": len(pending)},
                flush=True,
            )
    reference = []
    for parent in parents:
        entries = {}
        for system in subsystems(parent):
            for entry in read_json(directory / f"{system}.json"):
                entries[(entry["entry_id"], tuple(sorted(entry["composition"].items())))] = entry
        reference.append({"chemsys": parent, "entries": list(entries.values()), "database_version": version})
    # Retain other parent systems from the same release when extending the cache.
    cache = output / "official_slim_cache.jsonl"
    if cache.exists():
        reference.extend(
            row
            for row in read_rows(cache)
            if row["chemsys"] not in parents and row.get("database_version") == version
        )
    write_rows(cache, reference)
    report = {
        "database_version": version,
        "thermo_types": ["GGA_GGA+U"],
        "parent_systems": len(parents),
        "exact_systems": len(required),
        "new_query_systems": len(pending),
        "reference": str(cache),
    }
    write_json(output / "query.json", report)
    return report
