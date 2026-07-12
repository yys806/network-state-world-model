import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _state_arrays():
    return {
        "sample_seed": np.array([0, 0], dtype=np.int32),
        "sample_id": np.array([10, 11], dtype=np.int64),
        "x_node": np.zeros((2, 2, 3, 1), dtype=np.float32),
        "x_link": np.zeros((2, 2, 4, 1), dtype=np.float32),
        "x_task": np.zeros((2, 2, 1), dtype=np.float32),
        "y_node": np.zeros((2, 1, 3, 1), dtype=np.float32),
        "y_link": np.zeros((2, 1, 4, 1), dtype=np.float32),
        "y_task": np.zeros((2, 1, 1), dtype=np.float32),
    }


def _edge_action_arrays():
    future = np.zeros((2, 1, 4, 2), dtype=np.float32)
    future[0, 0, 1, 0] = 1.0
    return {
        "sample_seed": np.array([0, 0], dtype=np.int32),
        "sample_id": np.array([10, 11], dtype=np.int64),
        "edge_a_hist": np.zeros((2, 2, 4, 2), dtype=np.float32),
        "edge_a_future": future,
    }


class DatasetIntegrityGuardTest(unittest.TestCase):
    def test_core_action_match_rate_below_floor_is_rejected(self):
        from build_edge_action_v0 import validate_core_match_rates

        rows = [
            {
                "offload_total": 100,
                "offload_matched": 20,
                "rb_total": 100,
                "rb_matched": 0,
                "cpu_total": 100,
                "cpu_matched": 20,
            }
        ]
        with self.assertRaisesRegex(ValueError, "rb action match rate"):
            validate_core_match_rates(rows, minimum_rate=0.05)

    def test_missing_action_file_is_rejected(self):
        import build_edge_action_v0

        previous = build_edge_action_v0.STRICT_ACTION_DIR
        build_edge_action_v0.STRICT_ACTION_DIR = PROJECT_ROOT / "artifacts" / "__missing_strict_actions__"
        try:
            with self.assertRaisesRegex(FileNotFoundError, "offload_actions.csv"):
                build_edge_action_v0.read_action_file(0, "offload_actions.csv")
        finally:
            build_edge_action_v0.STRICT_ACTION_DIR = previous

    def test_all_zero_future_actions_are_rejected(self):
        from build_world_model_dataset_v0 import validate_world_model_inputs

        state = _state_arrays()
        edge_action = _edge_action_arrays()
        edge_action["edge_a_future"][:] = 0.0
        with self.assertRaisesRegex(ValueError, "edge_a_future is all zero"):
            validate_world_model_inputs(state, edge_action)

    def test_misaligned_sample_ids_are_rejected(self):
        from build_world_model_dataset_v0 import validate_world_model_inputs

        state = _state_arrays()
        edge_action = _edge_action_arrays()
        edge_action["sample_id"] = np.array([11, 10], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "sample_id"):
            validate_world_model_inputs(state, edge_action)

    def test_mismatched_edge_dimensions_are_rejected(self):
        from build_world_model_dataset_v0 import validate_world_model_inputs

        state = _state_arrays()
        edge_action = _edge_action_arrays()
        edge_action["edge_a_future"] = np.ones((2, 1, 5, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "edge dimension"):
            validate_world_model_inputs(state, edge_action)

    def test_non_finite_values_are_rejected(self):
        from build_world_model_dataset_v0 import validate_world_model_inputs

        state = _state_arrays()
        edge_action = _edge_action_arrays()
        state["x_link"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "x_link contains non-finite"):
            validate_world_model_inputs(state, edge_action)


if __name__ == "__main__":
    unittest.main()
