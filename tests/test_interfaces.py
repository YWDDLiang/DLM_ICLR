import json
from pathlib import Path
import pytest
from pymatgen.core import Composition, Lattice, Structure
from dlm_iclr.runtime.config import load, path, run_root, asset
from dlm_iclr.runtime.io import read_rows, write_rows
from dlm_iclr.data.adapters import iter_structures
from dlm_iclr.evaluation.hull import query
from dlm_iclr.evaluation.direct import evaluate_direct
from dlm_iclr._core.continuous_keep_edit import commit_patch
from dlm_iclr._core.dynamic_crystal import arrays_to_dynamic_tokens, build_special_tokens


def test_config_merges_defaults_and_resolves_paths(tmp_path):
    config_file = tmp_path / "custom.json"
    config_file.write_text(json.dumps({"output": "runs", "dataset": {"name": "custom"}}))
    c = load(config_file, ["b0.effective_batch_size=32", 'runtime.devices=["cpu"]'])
    assert c["b0"]["effective_batch_size"] == 32 and c["b0"]["lora_r"] == 8
    assert run_root(c) == tmp_path / "runs"
    assert asset(c, "b0") == str(tmp_path / "runs/b0/final")


def test_dataset_column_mapping_preserves_polymorphs(tmp_path):
    source = tmp_path / "structures.jsonl"
    write_rows(source, [{"key": "a", "payload": "first"}, {"key": "b", "payload": "second"}])
    rows = list(iter_structures(source, {"id": "key", "cif": "payload"}))
    assert [r[1] for r in rows] == ["a", "b"]
    assert [r[2]["cif"] for r in rows] == ["first", "second"]


def test_hull_query_reuses_systems_and_closes_subsystems(tmp_path):
    class Entry:
        def __init__(self, formula):
            self.composition = Composition(formula)
            self.energy = -1.0
            self.entry_id = formula

    class Client:
        def __init__(self):
            self.calls = []

        def get_database_version(self):
            return "2026.09"

        def get_entries(self, systems, **kwargs):
            self.calls.append(list(systems))
            return [Entry(s.replace("-", "")) for s in systems]

    client = Client()
    records = [{"plan_state": {"elements": ["Na", "Cl"]}, "body_eligible": True}]
    first = query(records, tmp_path, client=client)
    assert first["exact_systems"] == 3
    assert {x for call in client.calls for x in call} == {"Cl", "Na", "Cl-Na"}
    assert len(read_rows(tmp_path / "official_slim_cache.jsonl")[0]["entries"]) == 3
    query(records, tmp_path, client=client)
    assert len(client.calls) == 1


def test_direct_retains_failed_requests(tmp_path):
    crystal = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    records = [
        {"source_id": "ok", "ordinal": 0, "success": True, "structure": crystal.as_dict()},
        {"source_id": "failed", "ordinal": 1, "success": False, "reason": "generation_failure"},
    ]
    rows, report = evaluate_direct(records, tmp_path, metrics="comp_struct")
    assert report["counts"]["requests"] == 2
    assert rows[0]["comp_valid"] and rows[0]["struct_valid"]
    assert rows[1]["comp_valid"] is False and rows[1]["struct_valid"] is False


def test_local_patch_preserves_unchanged_continuous_fields():
    structure = Structure(
        Lattice.cubic(6.0), ["Na", "Cl"], [[0.755202949, 0.682214320, 0.895296931], [0.2, 0.2, 0.2]]
    )
    tokens, _ = arrays_to_dynamic_tokens(
        structure.lattice.abc,
        structure.lattice.angles,
        [str(s.specie) for s in structure],
        structure.frac_coords,
    )
    vocab = {token: i for i, token in enumerate(build_special_tokens())}
    inverse = {i: t for t, i in vocab.items()}
    current = [vocab[t] for t in tokens]
    proposed = current.copy()
    proposed[8] = vocab["<X_074>"]
    proposed[9] = vocab["<Y_079>"]
    record = {"success": True, "source_id": "patch", "ordinal": 0, "structure": structure.as_dict()}
    updated, report = commit_patch(record, current, proposed, inverse)
    assert report["applied"]
    after = Structure.from_dict(updated["structure"])
    assert after.frac_coords[0, 2] == structure.frac_coords[0, 2]
    assert after.frac_coords[0, 0] == 0.74 and after.frac_coords[0, 1] == 0.79
