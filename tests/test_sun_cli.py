import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


AVAILABLE = all(importlib.util.find_spec(name) for name in ("pymatgen", "smact", "ase"))


@unittest.skipUnless(AVAILABLE, "SUN scoring extras are not installed")
class SunCommandTests(unittest.TestCase):
    def make_inputs(self, root):
        from pymatgen.core import Lattice, Structure
        from dlm_iclr.io import write_json, write_rows
        from dlm_iclr.physics import record_key

        def crystal(species):
            return Structure(Lattice.cubic(5.6), species, [[0, 0, 0], [0.5, 0.5, 0.5]]).as_dict()

        records = [
            {"source_id": "strict", "ordinal": 0, "success": True, "structure": crystal(["Na", "Cl"])},
            {"source_id": "duplicate", "ordinal": 1, "success": True, "structure": crystal(["Na", "Cl"])},
            {"source_id": "meta", "ordinal": 2, "success": True, "structure": crystal(["Li", "Cl"])},
            {"source_id": "failure", "ordinal": 3, "success": False},
            {"source_id": "missing", "ordinal": 4, "success": True, "structure": crystal(["Yb", "Cl"])},
        ]
        labels = [
            {
                "source_id": row["source_id"],
                "ordinal": row["ordinal"],
                "record_key": record_key(row),
                "status": "verified" if row["success"] else "generation_failure",
                "verified": row["success"],
                "terminal_energy": energy,
            }
            for row, energy in zip(records, [-1.0, -1.0, 0.05, None, -1.0], strict=True)
        ]
        hull = [
            {
                "chemsys": "-".join(sorted([element, "Cl"])),
                "entries": [
                    {"composition": {element: 1}, "energy": 0.0},
                    {"composition": {"Cl": 1}, "energy": 0.0},
                ],
            }
            for element in ("Na", "Li")
        ]
        write_rows(root / "structures.jsonl", records)
        write_rows(root / "labels.jsonl", labels)
        write_rows(root / "hull.jsonl", hull)
        write_rows(root / "train.jsonl", [{"structure": crystal(["Si", "Si"])}])
        write_json(
            root / "config.json",
            {
                "assets": {
                    "hull_cache": "./hull.jsonl",
                    "novelty_reference": "./train.jsonl",
                }
            },
        )
        return records, labels

    def test_sun_and_msun_reuse_labels_without_torch_or_chgnet(self):
        from dlm_iclr.io import read_json, read_rows

        script = """
import importlib.abc, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'chgnet', 'transformers', 'matminer'}:
            raise AssertionError('Offline SUN imported ' + fullname)
sys.meta_path.insert(0, BlockModels())
from dlm_iclr.cli import main
main(sys.argv[1:])
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    "evaluate-sun",
                    "--config",
                    str(root / "config.json"),
                    "--structures",
                    str(root / "structures.jsonl"),
                    "--labels",
                    str(root / "labels.jsonl"),
                    "--output",
                    str(root / "sun"),
                    "--nu-workers",
                    "1",
                ],
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(root / "sun/summary.json")
            rows = read_rows(root / "sun/scores.jsonl")
            self.assertEqual(report["counts"]["requests"], 5)
            self.assertEqual(report["known_counts"]["strict_sun"], 1)
            self.assertEqual(report["known_counts"]["meta_sun"], 2)
            self.assertEqual(report["count_bounds"]["strict_sun"], [1, 2])
            self.assertEqual(report["count_bounds"]["meta_sun"], [2, 3])
            self.assertEqual(report["metrics"], {"SUN": None, "MSUN": None})
            self.assertIs(rows[1]["unique_representative"], False)
            self.assertIs(rows[3]["strict_sun"], False)
            self.assertIs(rows[4]["hull_missing"], True)
            self.assertFalse((root / "sun/physics").exists())

    def test_reused_labels_reject_wrong_geometry_or_order(self):
        from dlm_iclr.config import load_config
        from dlm_iclr.evaluation import evaluate_sun
        from dlm_iclr.io import write_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, labels = self.make_inputs(root)
            config = load_config(root / "config.json")
            labels[0]["record_key"] = "different_geometry"
            write_rows(root / "labels.jsonl", labels)
            with self.assertRaisesRegex(ValueError, "does not belong"):
                evaluate_sun(config, root / "structures.jsonl", root / "sun", labels=root / "labels.jsonl")
            self.make_inputs(root)
            write_rows(root / "labels.jsonl", list(reversed(labels)))
            with self.assertRaisesRegex(ValueError, "source/order"):
                evaluate_sun(config, root / "structures.jsonl", root / "sun", labels=root / "labels.jsonl")


if __name__ == "__main__":
    unittest.main()
