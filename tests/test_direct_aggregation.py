import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from dlm_iclr.evaluation import direct, features
from dlm_iclr.evaluation.direct_aggregation import merge_direct_blocks
from dlm_iclr.runtime.io import write_rows


def test_block_metrics_equal_whole_panel_with_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "_composition_valid", lambda *_: True)

    def fingerprints(structures, **_):
        return ([{"comp_fp": [len(s)], "struct_fp": [s.lattice.a], "error": None}
                 if s is not None else {"comp_fp": None, "struct_fp": None, "error": "failure"}
                 for s in structures], {"cache_hits": 0, "computed": len(structures)})

    monkeypatch.setattr(features, "compute_features", fingerprints)

    def row(name, lattice, species):
        crystal = Structure(Lattice.cubic(lattice), species,
                            [[i / len(species)] * 3 for i in range(len(species))])
        return {"source_id": name, "success": True, "structure": crystal.as_dict()}

    reference = tmp_path / "reference.jsonl"
    write_rows(reference, [row("r1", 4, ["Si"]), row("r2", 6, ["Na", "Cl"])])
    records = [{"source_id": "failed", "success": False}, row("a", 4, ["Si"]),
               row("b", 5, ["Na", "Cl"]), row("c", 6, ["Si"])]
    options = dict(reference=reference, save_aggregation=True)
    blocks = []
    for index, group in enumerate((records[:1], records[1:2], records[2:])):
        output = tmp_path / f"block{index}"
        direct.evaluate_direct(group, output, **options)
        blocks.append(output)
    _, expected = direct.evaluate_direct(records, tmp_path / "whole", **options)
    rows, actual = merge_direct_blocks(blocks, tmp_path / "merged")
    assert [r["source_id"] for r in rows] == [r["source_id"] for r in records]
    for key, value in expected["metrics"].items():
        assert actual["metrics"][key] == value
    assert actual["counts"]["requests"] == 4
    assert actual["metrics"]["V"] == 75.0
    with pytest.raises(ValueError, match="disjoint"):
        merge_direct_blocks([blocks[1], blocks[1]], tmp_path / "overlap")


def test_coverage_matches_dense_independent_minima():
    from scipy.spatial.distance import cdist

    pred = [{"struct_fp": [0.0], "comp_fp": [10.0], "error": None},
            {"struct_fp": [10.0], "comp_fp": [0.0], "error": None}]
    ref = [{"struct_fp": [0.0], "comp_fp": [0.0], "error": None}]
    actual, distances = features.coverage(pred, ref, struc_cutoff=0.4, comp_cutoff=1,
                                         requested=3, block_size=1, return_distances=True)
    s = cdist([[0], [10]], [[0]])
    c = cdist([[10], [0]], [[0]])
    assert actual["cov_recall"] == np.mean((s.min(0) <= .4) & (c.min(0) <= 1)) == 1
    assert actual["cov_precision"] == 0
    np.testing.assert_equal(distances["recall_structure"], s.min(0))
