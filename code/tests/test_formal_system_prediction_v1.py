from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalSystemPredictionV1Tests(unittest.TestCase):
    def test_converts_normalized_rollout_to_one_step_real_system_predictions(self):
        from pi_jwm.airfogsim_tensor_v2 import FLOW_FEATURES, TASK_FEATURES
        from pi_jwm.formal_system_prediction_v1 import system_predictions_from_batch

        batch = {
            "history": {
                "task_lifecycle_index": torch.tensor([[[1, 3, -1]]]),
                "task_node_index": torch.tensor([[[[2, 0, -1, -1], [1, 0, -1, -1], [-1] * 4]]]),
            },
            "static": {
                "task_valid": torch.tensor([[True, True, False]]),
                "flow_valid": torch.tensor([[True, False]]),
            },
            "system_target": {
                "task_completion_event": torch.tensor([[[True, False, False]]]),
                "task_on_time_completion_event": torch.tensor([[[True, False, False]]]),
                "completed_task_delay": torch.tensor([[[4.0, 0.0, 0.0]]]),
                "completed_task_delay_valid": torch.tensor([[[True, False, False]]]),
                "delivered_data_total": torch.tensor([[3.0]]),
                "uav_energy_delta": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
                "uav_energy_valid": torch.tensor([[[True, False, False, False]]]),
                "source_service_delta": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
                "source_on_time_service_delta": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
            },
            "system_static": {
                "source_population_valid": torch.tensor([[False, True, True, False]]),
                "source_evaluable_task_count": torch.tensor([[0, 1, 2, 0]]),
            },
        }
        task_state = torch.zeros((1, 1, 3, len(TASK_FEATURES)))
        task_state[0, 0, 0, TASK_FEATURES.index("delay")] = 2.0
        flow_state = torch.zeros((1, 1, 2, len(FLOW_FEATURES)))
        flow_state[0, 0, 0, FLOW_FEATURES.index("delivered_this_slot")] = 1.0
        flow_state[0, 0, 1, FLOW_FEATURES.index("delivered_this_slot")] = 99.0
        lifecycle_logits = torch.zeros((1, 1, 3, 5))
        lifecycle_logits[0, 0, 0, 3] = 5.0
        lifecycle_logits[0, 0, 1, 3] = 5.0
        prediction = {
            "task_state_mean": task_state,
            "task_lifecycle_logits": lifecycle_logits,
            "flow_state_mean": flow_state,
            "flow_presence_logits": torch.zeros((1, 1, 2)),
            "uav_energy_delta_mean": torch.tensor([[[1.5, 0.0, 0.0, 0.0]]]),
        }
        stats = {
            "features": {
                "task_state": {
                    "mean": [0.0] * 7 + [10.0],
                    "scale": [1.0] * 7 + [2.0],
                },
                "flow_state": {
                    "mean": [0.0, 0.0, 0.0, 1.0, 0.0],
                    "scale": [1.0, 1.0, 1.0, 2.0, 1.0],
                },
            }
        }

        converted = system_predictions_from_batch(prediction, batch, stats, horizon_index=0)

        torch.testing.assert_close(
            converted["predicted_completion_event"],
            torch.tensor([[True, False, False]]),
        )
        self.assertEqual(14.0, float(converted["predicted_task_delay"][0, 0]))
        self.assertEqual(1.5, float(converted["predicted_delivered_data"][0]))
        torch.testing.assert_close(
            converted["predicted_source_service"],
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        )
        self.assertEqual(1.5, float(converted["predicted_uav_energy"][0, 0]))
        self.assertEqual(3.0, float(converted["true_delivered_data"][0]))

    def test_energy_prediction_is_none_when_head_is_absent(self):
        from pi_jwm.airfogsim_tensor_v2 import FLOW_FEATURES, TASK_FEATURES
        from pi_jwm.formal_system_prediction_v1 import system_predictions_from_batch

        batch = {
            "history": {
                "task_lifecycle_index": torch.tensor([[[1]]]),
                "task_node_index": torch.tensor([[[[0, -1, -1, -1]]]]),
            },
            "static": {
                "task_valid": torch.tensor([[True]]),
                "flow_valid": torch.tensor([[True]]),
            },
            "system_target": {
                "task_completion_event": torch.tensor([[[False]]]),
                "task_on_time_completion_event": torch.tensor([[[False]]]),
                "completed_task_delay": torch.tensor([[[0.0]]]),
                "completed_task_delay_valid": torch.tensor([[[False]]]),
                "delivered_data_total": torch.tensor([[0.0]]),
                "uav_energy_delta": torch.tensor([[[0.0]]]),
                "uav_energy_valid": torch.tensor([[[False]]]),
                "source_service_delta": torch.tensor([[[0.0]]]),
                "source_on_time_service_delta": torch.tensor([[[0.0]]]),
            },
            "system_static": {
                "source_population_valid": torch.tensor([[True]]),
                "source_evaluable_task_count": torch.tensor([[1]]),
            },
        }
        prediction = {
            "task_state_mean": torch.zeros((1, 1, 1, len(TASK_FEATURES))),
            "task_lifecycle_logits": torch.zeros((1, 1, 1, 5)),
            "flow_state_mean": torch.zeros((1, 1, 1, len(FLOW_FEATURES))),
            "flow_presence_logits": torch.zeros((1, 1, 1)),
        }
        stats = {
            "features": {
                "task_state": {"mean": [0.0] * len(TASK_FEATURES), "scale": [1.0] * len(TASK_FEATURES)},
                "flow_state": {"mean": [0.0] * len(FLOW_FEATURES), "scale": [1.0] * len(FLOW_FEATURES)},
            }
        }

        converted = system_predictions_from_batch(prediction, batch, stats, horizon_index=0)

        self.assertIsNone(converted["predicted_uav_energy"])


if __name__ == "__main__":
    unittest.main()
