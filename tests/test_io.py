import tempfile
import unittest
from pathlib import Path

import numpy as np
from dlm_iclr.io import fingerprint, read_json, write_json


class NumericalSerializationTests(unittest.TestCase):
    def test_library_scalars_and_arrays_roundtrip_as_numbers(self):
        value = {
            "sites": [{"properties": {"magmom": np.float32(1.25)}}],
            "positions": np.array([[0.1, 0.2, 0.3]]),
            "verified": np.bool_(True),
        }
        expected = {
            "sites": [{"properties": {"magmom": 1.25}}],
            "positions": [[0.1, 0.2, 0.3]],
            "verified": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.json"
            write_json(path, value)
            self.assertEqual(read_json(path), expected)
        self.assertEqual(fingerprint(value), fingerprint(expected))

    def test_nonfinite_energy_is_not_silently_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "label.json", {"energy": np.float32("nan")})


if __name__ == "__main__":
    unittest.main()
