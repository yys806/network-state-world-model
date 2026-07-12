import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class V7ActiveRateTeacherTest(unittest.TestCase):
    def test_ridge_active_rate_teacher_adds_teacher_arrays_with_active_mask(self):
        from pi_jwm.v7_active_rate_teacher import add_ridge_active_rate_teacher

        rng = np.random.default_rng(20260614)
        arrays = {
            "x_node": rng.normal(size=(4, 2, 3, 6)).astype("float32"),
            "x_link": rng.normal(size=(4, 2, 5, 4)).astype("float32"),
            "x_task": rng.normal(size=(4, 2, 7)).astype("float32"),
            "edge_a_hist": rng.normal(size=(4, 2, 5, 3)).astype("float32"),
            "edge_a_future": rng.normal(size=(4, 2, 5, 3)).astype("float32"),
            "y_link_rate": rng.uniform(0.0, 10.0, size=(4, 2, 5)).astype("float32"),
            "y_link_active": np.zeros((4, 2, 5), dtype="float32"),
            "edge_src_idx": np.array([0, 0, 1, 1, 2], dtype="int32"),
            "edge_dst_idx": np.array([1, 2, 0, 2, 0], dtype="int32"),
            "valid_edge_node": np.ones(5, dtype="int32"),
        }
        arrays["y_link_active"][:2, :, :2] = 1.0

        result = add_ridge_active_rate_teacher(arrays, train_idx=np.array([0, 1]), ridge_lambda=1.0)

        self.assertEqual(result.arrays["y_link_rate_teacher"].shape, arrays["y_link_rate"].shape)
        self.assertEqual(result.arrays["y_link_rate_teacher_mask"].shape, arrays["y_link_active"].shape)
        np.testing.assert_array_equal(result.arrays["y_link_rate_teacher_mask"], arrays["y_link_active"])
        self.assertEqual(result.summary["rate_teacher_mode"], "ridge_active")
        self.assertEqual(result.summary["rate_teacher_train_active_count"], 8)


if __name__ == "__main__":
    unittest.main()
