from pymatgen.core import Lattice, Structure
from dlm_iclr.evaluation import direct
from dlm_iclr._vendor.crysllmgen import validity


def test_validity_cache_reuses_endpoint_and_composition(tmp_path, monkeypatch):
    calls = {"composition": 0, "structure": 0}

    def composition(*_):
        calls["composition"] += 1
        return True

    def geometry(_):
        calls["structure"] += 1
        return True

    monkeypatch.setattr(direct, "_composition_valid", composition)
    monkeypatch.setattr(validity, "structure_validity", geometry)

    def row(name, position=0.5):
        structure = Structure(Lattice.cubic(5), ["Si", "Si"], [[0, 0, 0], [position] * 3])
        return {"source_id": name, "success": True, "structure": structure.as_dict()}

    cache = tmp_path / "cache"
    direct.evaluate_direct([row("raw")], tmp_path / "first", metrics="comp_struct", cache=cache)
    direct.evaluate_direct([row("kept")], tmp_path / "second", metrics="comp_struct", cache=cache)
    assert calls == {"composition": 1, "structure": 1}
    direct.evaluate_direct([row("edited", 0.4)], tmp_path / "third", metrics="comp_struct", cache=cache)
    assert calls == {"composition": 1, "structure": 2}
    direct.evaluate_direct(
        [row("other_policy")], tmp_path / "fourth", metrics="comp_struct", cache=cache, composition="smact3"
    )
    assert calls == {"composition": 2, "structure": 3}


def test_parallel_direct_preserves_order_and_failed_denominator(tmp_path):
    crystal = Structure(Lattice.cubic(5), ["Si", "Si"], [[0, 0, 0], [0.5] * 3]).as_dict()
    records = [
        {"source_id": "a", "success": True, "structure": crystal},
        {"source_id": "failure", "success": False},
        {"source_id": "b", "success": True, "structure": crystal},
    ]
    rows, report = direct.evaluate_direct(
        records, tmp_path / "parallel", metrics="comp_struct", workers=2, cache=tmp_path / "cache"
    )
    serial, expected = direct.evaluate_direct(
        records, tmp_path / "serial", metrics="comp_struct", cache=tmp_path / "cache"
    )
    assert [r["source_id"] for r in rows] == ["a", "failure", "b"]
    assert rows == serial
    assert report["counts"] == expected["counts"]
    assert report["counts"]["requests"] == 3
