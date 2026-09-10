import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch extra is not installed")
class ValueTests(unittest.TestCase):
    def test_keep_does_not_need_measured_current_outcomes(self):
        from dlm_iclr.value import choose

        self.assertIsNone(choose([[0.0, 0.0], [-0.1, 0.1], None]))
        self.assertEqual(choose([[0.1, 0.0], [0.0, 0.3]]), 1)

    def test_identity_has_zero_gain_for_any_model_parameters(self):
        from dlm_iclr.value import ValueNetwork

        torch.manual_seed(9)
        model = ValueNetwork(raw_features=6, hidden_width=4)
        x, g = torch.randn(5, 6), torch.randn(5, 9)
        gain = model(x, g) - model(x.clone(), g.clone())
        self.assertTrue(torch.equal(gain, torch.zeros_like(gain)))

    def test_saved_value_is_portable_without_an_editor_directory(self):
        from dlm_iclr.value import ValueNetwork, load_value

        model = ValueNetwork(raw_features=6, hidden_width=4)
        x, g = torch.randn(5, 6), torch.randn(5, 9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pt"
            model.save(path)
            restored = load_value(path)
            self.assertTrue(torch.equal(model(x, g), restored(x, g)))

    def test_bundled_model_contains_no_physical_input_columns(self):
        from dlm_iclr.value import load_value, GEOMETRY_FEATURES

        model = load_value()
        self.assertEqual(model.hidden.in_features, 8192)
        self.assertEqual(model.head.in_features, 128 + 9)
        self.assertEqual(len(GEOMETRY_FEATURES), 9)
        self.assertTrue(torch.isfinite(model.head.weight).all())


if __name__ == "__main__":
    unittest.main()
