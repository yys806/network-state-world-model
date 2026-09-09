from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _row(**overrides):
    row = {
        "time": 0.2,
        "physical_edge_id": "pe::v0::r0::V2I",
        "rb_index": 1,
        "rate_per_s": 12.5,
        "outage": False,
        "observed_mask": True,
        "missing_reason": None,
        "capture_phase": "after_fast_fading_before_transfer",
        "temporal_role": "outcome_only_not_same_frame_decision_input",
        "source_method": "AirFogSim_direct_per_rb_runtime_arrays",
    }
    row.update(overrides)
    return row


class FormalRbTargetTests(unittest.TestCase):
    def test_rebuilds_per_rb_outcomes_from_actions_and_transfer_events(self):
        from pi_jwm.airfogsim_tensor_v2 import infer_tensor_contract
        from pi_jwm.formal_rb_targets_v1 import (
            OUTCOME_TEMPORAL_ROLE,
            build_runtime_rb_outcome_observations,
        )

        graph = {
            "physical_edges": [
                {"id": "physical::vehicle_0::RSU_0::V2I", "src": "vehicle_0", "dst": "RSU_0"}
            ],
            "source_rb_actions": [
                {
                    "task_id": "Task_7",
                    "current_node_id": "vehicle_0",
                    "assigned_to": "RSU_0",
                    "time": 0.2,
                    "rb_indices": [3, 17],
                    "n_rb": 50,
                }
            ],
            "source_transfer_events": [
                {
                    "task_id": "Task_7",
                    "source": "vehicle_0",
                    "target": "RSU_0",
                    "time": 0.2,
                    "rb_indices": [3, 17],
                    "path": ["physical::vehicle_0::RSU_0::V2I"],
                    "planned_capacity": 2.5,
                }
            ],
        }

        observations, report = build_runtime_rb_outcome_observations(
            graph, slot_seconds=0.1
        )

        self.assertEqual(50, infer_tensor_contract([graph]).n_rb)
        self.assertEqual(2, len(observations))
        self.assertEqual([3, 17], [row["rb_index"] for row in observations])
        self.assertTrue(all(row["observed_mask"] for row in observations))
        self.assertTrue(
            all(row["physical_edge_id"] == "physical::vehicle_0::RSU_0::V2I" for row in observations)
        )
        self.assertTrue(all(row["temporal_role"] == OUTCOME_TEMPORAL_ROLE for row in observations))
        self.assertTrue(all(row["rate_per_s"] == 25.0 for row in observations))
        self.assertEqual("derived_from_action_and_runtime_event", report["source"])

    def test_preserves_rb_action_when_transfer_event_is_unobserved(self):
        from pi_jwm.formal_rb_targets_v1 import build_runtime_rb_outcome_observations

        graph = {
            "physical_edges": [
                {"id": "physical::vehicle_0::RSU_0::V2I", "src": "vehicle_0", "dst": "RSU_0"}
            ],
            "source_rb_actions": [
                {
                    "task_id": "Task_7",
                    "current_node_id": "vehicle_0",
                    "assigned_to": "RSU_0",
                    "time": 0.2,
                    "rb_indices": [3],
                    "n_rb": 50,
                }
            ],
            "source_transfer_events": [],
        }

        observations, report = build_runtime_rb_outcome_observations(graph, slot_seconds=0.1)
        self.assertEqual(1, len(observations))
        self.assertFalse(observations[0]["observed_mask"])
        self.assertIsNone(observations[0]["rate_per_s"])
        self.assertEqual("runtime_transfer_event_unavailable", observations[0]["missing_reason"])
        self.assertEqual(1, report["unobserved_count"])

    def test_builds_direct_per_rb_activity_and_rate_with_shared_mask(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        arrays, report = build_per_rb_target_arrays(
            time_values=np.asarray([0.1, 0.2, 0.3]),
            edge_vocab=["pe::v0::r0::V2I"],
            n_rb=3,
            observations=[
                _row(time=0.2, rb_index=1, rate_per_s=12.5),
                _row(time=0.2, rb_index=2, rate_per_s=0.0, outage=True),
            ],
        )

        self.assertEqual((3, 1, 3), arrays["link_rate_by_rb"].shape)
        self.assertTrue(arrays["link_activity"][1, 0, 1])
        self.assertFalse(arrays["link_activity"][1, 0, 2])
        self.assertEqual(12.5, arrays["link_rate_by_rb"][1, 0, 1])
        np.testing.assert_array_equal(
            arrays["link_activity_mask"], arrays["link_rate_by_rb_mask"]
        )
        self.assertEqual(2, report["observed_label_count"])
        self.assertEqual("AirFogSim data-unit/s", report["rate_unit"])

    def test_unobserved_rb_is_null_mask_not_zero_filled_label(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        arrays, _ = build_per_rb_target_arrays(
            time_values=np.asarray([0.1, 0.2]),
            edge_vocab=["pe::v0::r0::V2I"],
            n_rb=2,
            observations=[_row(time=0.2, rb_index=0, rate_per_s=0.0, outage=True)],
        )

        self.assertFalse(arrays["link_activity_mask"][1, 0, 1])
        self.assertFalse(arrays["link_rate_by_rb_mask"][1, 0, 1])
        self.assertEqual(0.0, arrays["link_rate_by_rb"][1, 0, 1])

    def test_runtime_outcome_missing_role_is_valid_unobserved_label(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        arrays, report = build_per_rb_target_arrays(
            time_values=np.asarray([0.1, 0.2]),
            edge_vocab=["pe::v0::r0::V2I"],
            n_rb=2,
            observations=[_row(
                time=0.2,
                rb_index=0,
                observed_mask=False,
                rate_per_s=None,
                missing_reason="runtime_channel_row_unavailable",
                temporal_role="runtime_outcome_missing",
            )],
        )
        self.assertFalse(arrays["link_rate_by_rb_mask"][1, 0, 0])
        self.assertEqual(0, report["observed_label_count"])

    def test_duplicate_time_edge_rb_is_rejected(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        with self.assertRaisesRegex(ValueError, "duplicate RB observation"):
            build_per_rb_target_arrays(
                time_values=np.asarray([0.1, 0.2]),
                edge_vocab=["pe::v0::r0::V2I"],
                n_rb=2,
                observations=[_row(time=0.2, rb_index=1), _row(time=0.2, rb_index=1)],
            )

    def test_cross_flow_collision_is_collapsed_only_when_direct_values_agree(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        first = _row(time=0.2, rb_index=1, flow_id="flow-a", rate_per_s=0.0)
        second = _row(time=0.2, rb_index=1, flow_id="flow-b", rate_per_s=0.0)
        arrays, report = build_per_rb_target_arrays(
            time_values=np.asarray([0.1, 0.2]),
            edge_vocab=["pe::v0::r0::V2I"],
            n_rb=2,
            observations=[first, second],
        )
        self.assertTrue(arrays["link_rate_by_rb_mask"][1, 0, 1])
        self.assertEqual(1, report["collision_count"])

    def test_future_or_decision_phase_rows_are_rejected(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        for bad in (
            _row(capture_phase="decision", temporal_role="decision_input"),
            _row(capture_phase="after_fast_fading_before_transfer", temporal_role="decision_input"),
        ):
            with self.assertRaisesRegex(ValueError, "outcome-only"):
                build_per_rb_target_arrays(
                    time_values=np.asarray([0.1, 0.2]),
                    edge_vocab=["pe::v0::r0::V2I"],
                    n_rb=2,
                    observations=[bad],
                )

    def test_time_alignment_requires_label_after_history(self):
        from pi_jwm.formal_rb_targets_v1 import validate_label_window_alignment

        validate_label_window_alignment(
            history_times=np.asarray([0.0, 0.1]),
            label_times=np.asarray([0.2, 0.3]),
        )
        with self.assertRaisesRegex(ValueError, "strictly after"):
            validate_label_window_alignment(
                history_times=np.asarray([0.0, 0.1]),
                label_times=np.asarray([0.1, 0.2]),
            )

    def test_float32_time_storage_preserves_decimal_slot_identity(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        arrays, _ = build_per_rb_target_arrays(
            time_values=np.asarray([16.1, 16.2], dtype=np.float32),
            edge_vocab=["pe::v0::r0::V2I"],
            n_rb=1,
            observations=[_row(time=16.2, rb_index=0, rate_per_s=3.0)],
        )
        self.assertTrue(arrays["link_rate_by_rb_mask"][1, 0, 0])

    def test_per_rb_labels_use_frozen_edge_capacity_when_requested(self):
        from pi_jwm.formal_rb_targets_v1 import build_per_rb_target_arrays

        arrays, report = build_per_rb_target_arrays(
            time_values=np.asarray([0.1, 0.2]),
            edge_vocab=["pe::v0::r0::V2I"],
            edge_capacity=3,
            n_rb=2,
            observations=[_row(time=0.2, rb_index=1)],
        )
        self.assertEqual((2, 3, 2), arrays["link_activity"].shape)
        self.assertEqual(3, report["edge_capacity"])
        self.assertFalse(arrays["link_activity_mask"][1, 1:, :].any())


if __name__ == "__main__":
    unittest.main()
