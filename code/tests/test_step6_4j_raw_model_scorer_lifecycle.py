import unittest

import numpy as np

from pi_jwm.step6_2a_planner_objective_side_state_v1 import planner_derived_return_birth_required


class Step64JRawModelScorerLifecycleTests(unittest.TestCase):
    def test_airfogsim_style_completion_boundary_is_exact(self):
        self.assertTrue(1.0 >= 1.0)
        self.assertFalse((1.0 - 1e-16) >= 1.0)

    def test_fp32_shadow_can_round_positive_remainder_to_zero(self):
        total = 0.6532350234152132
        computed = 0.6532350234152131
        self.assertGreater(total - computed, 0.0)
        self.assertEqual(float(np.float32(total)) - float(np.float32(computed)), 0.0)

    def test_return_birth_requires_finished_task_and_missing_flow(self):
        task = type("T", (), {"requires_return": True, "required_returned_size": 1.0, "return_destination_index": 2})()
        self.assertTrue(planner_derived_return_birth_required(task, True, False, 1))
        self.assertFalse(planner_derived_return_birth_required(task, False, False, 1))
        self.assertFalse(planner_derived_return_birth_required(task, True, True, 1))
        self.assertFalse(planner_derived_return_birth_required(task, True, False, 2))

    def test_no_return_requirement_is_not_birth(self):
        task = type("T", (), {"requires_return": False, "required_returned_size": 0.0, "return_destination_index": 2})()
        self.assertFalse(planner_derived_return_birth_required(task, True, False, 1))


if __name__ == "__main__":
    unittest.main()
