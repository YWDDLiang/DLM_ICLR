import copy
import tempfile
import unittest
from pathlib import Path

from dlm_iclr.io import read_rows, write_rows
from dlm_iclr.plans import axis_schedule, composition_key, load_plans, prepare_training, validate_plan


class PlanTests(unittest.TestCase):
    def test_bundled_training_conditions_are_separate_compositions(self):
        train, _ = load_plans("CLEAN_TRAIN_1000", legal_only=True)
        train_keys = {composition_key(row["plan_state"]) for row in train}
        self.assertEqual(len(train), 1000)
        self.assertEqual(len(train_keys), 1000)
        self.assertTrue(all(row["provenance"]["usage_role"] == "train" for row in train))
        for name in ("H1A2_1200", "R03_256"):
            evaluation, _ = load_plans(name, legal_only=True)
            self.assertFalse(train_keys & {composition_key(row["plan_state"]) for row in evaluation})

    def test_actual_h1a2_cohort_and_full_precision_seeds(self):
        rows, _ = load_plans("H1A2_1200")
        self.assertEqual(len(rows), 1200)
        self.assertEqual(sum(row["body_eligible"] for row in rows), 1186)
        self.assertEqual(rows[0]["body_noise_seed"], 8084656096254702263)
        self.assertEqual(rows[0]["refiner_noise_seed"], 7695583362835148981)
        self.assertIsNone(rows[10]["plan_state"])

    def test_first_1050_legal_in_original_order_include_one_atom(self):
        rows, _ = load_plans("H1A2_1200", requests=1050, legal_only=True)
        all_rows, _ = load_plans("H1A2_1200")
        self.assertEqual(
            [row["original_ordinal"] for row in rows],
            [row["original_ordinal"] for row in all_rows if row["body_eligible"]][:1050],
        )
        self.assertEqual(rows[-1]["original_ordinal"], 1061)
        self.assertEqual(rows[0]["plan_state"]["N"], 1)

    def test_r03_is_its_real_distinct_preset(self):
        rows, _ = load_plans("R03_256")
        self.assertEqual(len(rows), 256)
        self.assertEqual(sum(row["body_eligible"] for row in rows), 254)
        self.assertEqual(rows[0]["plan_state"]["elements"], ["Se", "Ta"])
        self.assertEqual(rows[0]["plan_state"]["counts"], [8, 4])

    def test_all_preset_schedules_cover_tokens_once_and_keep_xyz_order(self):
        for preset in ("H1A2_1200", "R03_256"):
            rows, _ = load_plans(preset, legal_only=True)
            for row in rows:
                plan = row["plan_state"]
                schedule = axis_schedule(plan)
                flat = [position for group in schedule for position in group]
                self.assertEqual(sorted(flat), list(range(7 + 4 * plan["N"])))
                groups = {position: index for index, group in enumerate(schedule) for position in group}
                self.assertLess(
                    max(groups[9 + 4 * i] for i in range(plan["N"])),
                    min(groups[10 + 4 * i] for i in range(plan["N"])),
                )

    def test_composition_exclusion_uses_reduced_ratios_and_not_element_order(self):
        self.assertEqual(
            composition_key({"elements": ["Na", "Cl"], "counts": [2, 2]}),
            composition_key({"elements": ["Cl", "Na"], "counts": [1, 1]}),
        )

    def test_invalid_count_is_accounted_before_generation(self):
        rows, _ = load_plans("H1A2_1200", requests=1)
        row = copy.deepcopy(rows[0])
        row["plan_state"]["counts"] = [2]
        self.assertEqual(validate_plan(row), "counts_do_not_match_N")

    def test_training_does_not_relabel_evaluation_data(self):
        rows, _ = load_plans("H1A2_1200", requests=1)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "train.jsonl"
            write_rows(source, rows)
            report = prepare_training(source, output, exclusions=[])
            self.assertEqual(read_rows(output), [])
            self.assertEqual(report["omitted"][0]["reason"], "source_reserved_for_evaluation")


if __name__ == "__main__":
    unittest.main()
