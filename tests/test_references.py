import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("pymatgen"), "pymatgen is not installed")
class ReferenceCoverageTests(unittest.TestCase):
    def test_unknown_counts_preserve_full_requested_denominator(self):
        from dlm_iclr.evaluation import summarize
        from dlm_iclr.feedback import endpoint_targets

        fields = (
            "reconstructed",
            "comp_valid",
            "struct_valid",
            "strict_stable",
            "meta_stable",
            "strict_sun",
            "meta_sun",
            "terminal_verified",
            "verified_strict_sun",
            "verified_meta_sun",
        )
        rows = [{field: True for field in fields}, {field: None for field in fields}]
        for row in rows:
            row.update(terminal_status="verified", chemsys="C-Ca-Yb", hull_missing=True)
        result = summarize(rows)
        self.assertEqual(result["counts"]["requests"], 2)
        self.assertIsNone(result["counts"]["strict_sun"])
        self.assertEqual(result["known_counts"]["strict_sun"], 1)
        self.assertEqual(result["unknown_counts"]["strict_sun"], 1)
        self.assertEqual(result["count_bounds"]["strict_sun"], [1, 2])
        self.assertIsNone(
            endpoint_targets(
                {"terminal_status": "verified", "terminal_verified": True, "e_above_hull_eV_atom": None}
            )
        )

    def test_constructible_subsystem_is_not_a_complete_reference(self):
        from pymatgen.core import Lattice, Structure
        from dlm_iclr.evaluation import HullReference
        from dlm_iclr.io import write_rows

        row = {
            "chemsys": "C-Ca-Yb",
            "entries": [
                {"composition": {"C": 1}, "energy": -6.0},
                {"composition": {"Ca": 1}, "energy": -2.0},
            ],
        }
        crystal = Structure(
            Lattice.cubic(6.0), ["C", "Ca", "Yb"], [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6]]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            write_rows(path, [row])
            reference = HullReference(path)
            self.assertEqual(reference.energy(crystal), ("C-Ca-Yb", None))
            self.assertEqual(reference.unavailable["C-Ca-Yb"], "missing_reference_elements:Yb")


if __name__ == "__main__":
    unittest.main()
