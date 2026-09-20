import unittest
import torch
from dlm_iclr.planner.sampling import _call_in_fresh_process


def read_child_flags():
    return torch.are_deterministic_algorithms_enabled()


class PlannerProcessContextTests(unittest.TestCase):
    def test_physics_determinism_does_not_leak_into_spawned_planner(self):
        before=torch.are_deterministic_algorithms_enabled()
        warning=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            self.assertFalse(_call_in_fresh_process(read_child_flags))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(before,warn_only=warning)


if __name__=='__main__':unittest.main()
