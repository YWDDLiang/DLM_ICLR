import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("pymatgen", "smact", "matminer", "ase"))


@unittest.skipUnless(AVAILABLE, "Direct evaluation extras are not installed")
class DirectEvaluationTests(unittest.TestCase):
    def setUp(self):
        from pymatgen.core import Lattice, Structure

        self.crystal = Structure(Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.record = {"source_id": "a", "ordinal": 0, "success": True, "structure": self.crystal.as_dict()}

    def test_fast_cli_never_imports_full_evaluation_or_models(self):
        from pymatgen.core import Lattice, Structure
        from dlm_iclr.io import read_json, read_rows, write_rows

        overlapping = Structure(Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.001, 0, 0]])
        records = [
            self.record,
            dict(self.record, source_id="overlap", ordinal=1, structure=overlapping.as_dict()),
            dict(self.record, source_id="failed", ordinal=2, success=False),
            dict(self.record, source_id="bad", ordinal=3, structure={}),
        ]
        script = """
import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matminer', 'torch', 'chgnet'} or fullname in {
            'dlm_iclr.direct_features', 'dlm_iclr.physics', 'dlm_iclr.evaluation'}:
            raise AssertionError('Fast Direct imported ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
from dlm_iclr.cli import main
main(sys.argv[1:])
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "structures.jsonl", root / "fast"
            write_rows(source, records)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    "evaluate-direct",
                    "--structures",
                    str(source),
                    "--output",
                    str(output),
                    "--metrics",
                    "comp_struct",
                    "--reference",
                    str(root / "intentionally_absent.csv"),
                ],
                text=True,
                capture_output=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"comp_valid": 50.0, "struct_valid": 25.0})
            report = read_json(output / "summary.json")
            self.assertEqual(report["counts"], {"requests": 4, "comp_valid": 2, "struct_valid": 1})
            self.assertEqual(report["reported_metrics"], ["comp_valid", "struct_valid"])
            self.assertNotIn("valid", read_rows(output / "scores.jsonl")[0])
            self.assertFalse((output / "cache").exists())

    def test_full_metrics_match_identical_reference_and_keep_failed_denominator(self):
        from dlm_iclr.direct import evaluate_direct
        from dlm_iclr.io import write_rows

        records = [
            self.record,
            dict(self.record, source_id="duplicate", ordinal=1),
            {"source_id": "failed", "ordinal": 2, "success": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "test.jsonl"
            write_rows(reference, [{"structure": self.crystal.as_dict()}])
            basic_rows, basic = evaluate_direct(records, root / "basic", metrics="comp_struct")
            full_rows, full = evaluate_direct(records, root / "full", reference=reference)
            self.assertEqual(
                full["metrics"],
                {
                    "comp_valid": 66.6667,
                    "struct_valid": 66.6667,
                    "valid": 66.6667,
                    "wdist_density": 0.0,
                    "wdist_num_elems": 0.0,
                    "cov_recall": 100.0,
                    "cov_precision": 66.6667,
                },
            )
            for key in ("comp_valid", "struct_valid"):
                self.assertEqual(basic["metrics"][key], full["metrics"][key])
                self.assertEqual([row[key] for row in basic_rows], [row[key] for row in full_rows])
            self.assertEqual(full["fingerprints"]["computed"], 1)
            with patch("dlm_iclr.direct_features._featurize", side_effect=AssertionError("cache miss")):
                _, repeated = evaluate_direct(records, root / "full", reference=reference)
            self.assertEqual(repeated["fingerprints"]["computed"], 0)
            self.assertEqual(repeated["metrics"], full["metrics"])

    def test_full_mode_reports_undefined_distributions_when_all_generations_fail(self):
        from dlm_iclr.direct import evaluate_direct
        from dlm_iclr.io import write_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "test.jsonl"
            write_rows(reference, [{"structure": self.crystal.as_dict()}])
            _, report = evaluate_direct(
                [{"source_id": "failed", "ordinal": 0, "success": False}],
                root / "full",
                reference=reference,
            )
            self.assertIsNone(report["metrics"]["wdist_density"])
            self.assertIsNone(report["metrics"]["wdist_num_elems"])
            self.assertEqual(report["metrics"]["cov_precision"], 0.0)
            self.assertEqual(report["metrics"]["cov_recall"], 0.0)
            self.assertFalse(report["complete"])

    def test_coverage_keeps_independent_nearest_neighbors_across_blocks(self):
        from dlm_iclr.direct_features import coverage

        predicted = [
            {"struct_fp": [0.0], "comp_fp": [10.0], "error": None},
            {"struct_fp": [10.0], "comp_fp": [0.0], "error": None},
        ]
        reference = [{"struct_fp": [0.0], "comp_fp": [0.0], "error": None}]
        # Different witnesses satisfy the two recall distances in the frozen
        # upstream definition; neither prediction satisfies both precision tests.
        self.assertEqual(
            coverage(predicted, reference, struc_cutoff=0.1, comp_cutoff=0.1, requested=3, block_size=1),
            {"cov_recall": 1.0, "cov_precision": 0.0},
        )

    def test_full_cli_supports_spawned_fingerprint_workers(self):
        from dlm_iclr.io import read_json, write_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, output = root / "structures.jsonl", root / "reference.jsonl", root / "full"
            write_rows(source, [self.record])
            write_rows(reference, [{"structure": self.crystal.as_dict()}])
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dlm_iclr",
                    "evaluate-direct",
                    "--structures",
                    str(source),
                    "--reference",
                    str(reference),
                    "--output",
                    str(output),
                    "--workers",
                    "2",
                ],
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(output / "summary.json")["metrics"]["cov_precision"], 100.0)


class EvaluationInputTests(unittest.TestCase):
    def test_legacy_attempts_preserve_failures_and_source_order(self):
        from dlm_iclr.evaluation_inputs import normalize_records

        records = [{"attempt_id": "z", "status": "failed"}, {"attempt_id": "a", "status": "succeeded"}]
        result = normalize_records(records)
        self.assertEqual([row["source_id"] for row in result], ["z", "a"])
        self.assertEqual([row["success"] for row in result], [False, True])
        self.assertNotIn("ordinal", records[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_records([records[0], records[0]])
        with self.assertRaisesRegex(ValueError, "no requested"):
            normalize_records([])


if __name__ == "__main__":
    unittest.main()
