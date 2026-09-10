import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("pymatgen"), "pymatgen is not installed")
class ContinuousPatchTests(unittest.TestCase):
    def token_pair(self, before, after):
        from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
        from crystal_dlm.expert_edit_data import arrays_from_structure

        def tokens(structure):
            arrays = arrays_from_structure(structure.as_dict())
            return arrays_to_dynamic_tokens(
                arrays["lengths"], arrays["angles"], arrays["species"], arrays["frac_coords"]
            )[0]

        a, b = tokens(before), tokens(after)
        vocabulary = {token: index for index, token in enumerate(sorted(set(a + b)))}
        return (
            [vocabulary[token] for token in a],
            [vocabulary[token] for token in b],
            {value: key for key, value in vocabulary.items()},
        )

    def test_keep_retains_every_original_continuous_value(self):
        from pymatgen.core import Lattice, Structure
        from crystal_dlm.continuous_keep_edit import commit_patch

        structure = Structure(
            Lattice.cubic(7.123456789),
            ["Na", "Cl"],
            [[0.113456789, 0.223456789, 0.334567891], [0.6, 0.7, 0.8]],
        )
        before, after, inverse = self.token_pair(structure, structure)
        record = {"structure": structure.as_dict(), "success": True, "body": None}
        snapshot = copy.deepcopy(record)
        result, trace = commit_patch(record, before, after, inverse)
        self.assertEqual(result, snapshot)
        self.assertEqual(record, snapshot)
        self.assertFalse(trace["applied"])

    def test_local_edit_does_not_quantize_untouched_coordinates_or_cell(self):
        from pymatgen.core import Lattice, Structure
        from crystal_dlm.continuous_keep_edit import commit_patch

        structure = Structure(
            Lattice.cubic(7.123456789),
            ["Na", "Cl"],
            [[0.113456789, 0.223456789, 0.334567891], [0.6, 0.7, 0.8]],
        )
        candidate = Structure(
            structure.lattice, structure.species, [[0.19, 0.223456789, 0.334567891], [0.6, 0.7, 0.8]]
        )
        before, after, inverse = self.token_pair(structure, candidate)
        record = {"structure": structure.as_dict(), "success": True, "body": None}
        result, trace = commit_patch(record, before, after, inverse)
        self.assertTrue(trace["applied"])
        self.assertEqual(result["structure"]["lattice"], record["structure"]["lattice"])
        self.assertEqual(result["structure"]["sites"][1], record["structure"]["sites"][1])
        self.assertEqual(
            result["structure"]["sites"][0]["abc"][1:], record["structure"]["sites"][0]["abc"][1:]
        )
        self.assertAlmostEqual(result["structure"]["sites"][0]["abc"][0], 0.19)

    def test_single_atom_translation_is_keep(self):
        from pymatgen.core import Lattice, Structure
        from crystal_dlm.continuous_keep_edit import commit_patch

        structure = Structure(Lattice.cubic(4.0), ["Sn"], [[0.1, 0.2, 0.3]])
        candidate = Structure(structure.lattice, ["Sn"], [[0.2, 0.3, 0.4]])
        before, after, inverse = self.token_pair(structure, candidate)
        record = {"structure": structure.as_dict(), "success": True, "body": None}
        result, trace = commit_patch(record, before, after, inverse)
        self.assertEqual(result, record)
        self.assertEqual(trace["reason"], "rigid_translation_KEEP")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class PermutationTests(unittest.TestCase):
    def test_local_target_stays_on_the_same_atom_after_serialization_change(self):
        from crystal_dlm.editor_minibatch import permute_atoms

        current = list(range(19))
        target = current.copy()
        target[18] = 999
        row = {
            "num_sites": 3,
            "current_tokens": current,
            "content_target_tokens": target,
            "content_positions": [16, 17, 18],
            "action_positions": [16, 17, 18],
            "site_targets": [0.0, 0.0, 1.0],
        }
        changed = permute_atoms(row, [2, 0, 1])
        self.assertEqual(changed["current_tokens"][7:11], current[15:19])
        self.assertEqual(changed["content_positions"], [8, 9, 10])
        self.assertEqual(changed["content_target_tokens"][10], 999)
        self.assertEqual(changed["site_targets"], [1.0, 0.0, 0.0])
        self.assertEqual(row["current_tokens"], current)


if __name__ == "__main__":
    unittest.main()
