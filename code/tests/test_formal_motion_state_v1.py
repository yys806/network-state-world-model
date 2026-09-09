import unittest

import numpy as np

from pi_jwm.formal_motion_state_v1 import derive_causal_node_motion


class FormalMotionStateTests(unittest.TestCase):
    def test_derives_velocity_and_acceleration_without_future_values(self):
        state = np.zeros((4, 2, 7), dtype=np.float32)
        state[:, 0, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
        present = np.ones((4, 2), dtype=bool)

        result = derive_causal_node_motion(state, present, slot_seconds=0.5)

        np.testing.assert_allclose(result["node_motion_state"][1:, 0, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(result["node_motion_state"][2:, 0, 3], [4.0, 4.0])
        np.testing.assert_allclose(result["node_state"][1:, 0, 3], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(result["node_state"][2:, 0, 4], [4.0, 4.0])
        self.assertFalse(result["node_motion_mask"][0].any())
        self.assertTrue(result["node_motion_mask"][1, 0, :3].all())
        self.assertFalse(result["node_motion_mask"][1, 0, 3:].any())
        self.assertTrue(result["node_motion_mask"][2, 0].all())

        changed = state.copy()
        changed[3, 0, 0] = 1000.0
        changed_result = derive_causal_node_motion(changed, present, slot_seconds=0.5)
        np.testing.assert_array_equal(
            result["node_motion_state"][:3], changed_result["node_motion_state"][:3]
        )

    def test_presence_gaps_invalidate_motion_instead_of_bridging_them(self):
        state = np.zeros((4, 1, 7), dtype=np.float32)
        state[:, 0, 0] = [0.0, 1.0, 2.0, 3.0]
        present = np.array([[True], [False], [True], [True]])

        result = derive_causal_node_motion(state, present, slot_seconds=1.0)

        self.assertFalse(result["node_motion_mask"][2, 0].any())
        self.assertTrue(result["node_motion_mask"][3, 0, :3].all())
        self.assertFalse(result["node_motion_mask"][3, 0, 3:].any())

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "slot_seconds"):
            derive_causal_node_motion(
                np.zeros((2, 1, 7), dtype=np.float32),
                np.ones((2, 1), dtype=bool),
                slot_seconds=0.0,
            )


if __name__ == "__main__":
    unittest.main()
