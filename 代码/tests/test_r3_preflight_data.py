from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class R3PreflightWindowTests(unittest.TestCase):
    def _index_rows(self) -> list[dict[str, str]]:
        return [
            {
                "trajectory_id": "train__r01",
                "seed": "1",
                "split": "train",
                "v3_status": "materialized",
                "v3_seed_dir": "seed_001",
                "observed_steps": "30",
            },
            {
                "trajectory_id": "train__r02",
                "seed": "2",
                "split": "train",
                "v3_status": "materialized",
                "v3_seed_dir": "seed_002",
                "observed_steps": "30",
            },
            {
                "trajectory_id": "locked__r09",
                "seed": "9",
                "split": "locked_test",
                "v3_status": "locked_integrity_only",
                "v3_seed_dir": "",
                "observed_steps": "30",
            },
        ]

    def test_r3_view_rejects_locked_test(self):
        from pi_jwm.r3_preflight_data import select_r3_windows

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "locked_test"):
                select_r3_windows(
                    Path(temporary),
                    self._index_rows(),
                    split="locked_test",
                    horizons=(1, 5, 20),
                )

    def test_r3_view_builds_eight_history_and_1_5_20_targets(self):
        from pi_jwm.r3_preflight_data import select_r3_windows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (1, 2):
                seed_dir = root / f"seed_{seed:03d}"
                seed_dir.mkdir()
                np.savez_compressed(
                    seed_dir / "trajectory_tensors.npz",
                    time=np.arange(30, dtype=np.float32),
                )
            rows = select_r3_windows(
                root,
                self._index_rows(),
                split="train",
                horizons=(1, 5, 20),
                history_steps=8,
                per_horizon=1,
                seed=20260804,
            )

        self.assertEqual({1, 5, 20}, {row.horizon_steps for row in rows})
        self.assertTrue(all(row.history_end - row.history_start == 8 for row in rows))
        self.assertTrue(
            all(row.target_end - row.target_start == row.horizon_steps for row in rows)
        )
        self.assertTrue(all(row.history_end == row.target_start for row in rows))
        self.assertTrue(all(row.split == "train" for row in rows))

    def test_load_r3_window_keeps_history_actions_and_targets_separate(self):
        from pi_jwm.r3_preflight_data import R3Window, load_r3_window

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_path = root / "trajectory_tensors.npz"
            np.savez_compressed(
                tensor_path,
                time=np.arange(12, dtype=np.float32),
                physical_node_state=np.arange(24, dtype=np.float32).reshape(12, 1, 2),
                task_action=np.arange(36, dtype=np.float32).reshape(12, 1, 3),
                physical_node_kind_index=np.asarray([2], dtype=np.int16),
            )
            window = R3Window(
                trajectory_id="train__r01",
                environment_seed=1,
                split="train",
                tensor_path=tensor_path,
                history_start=0,
                history_end=8,
                target_start=8,
                target_end=12,
                horizon_steps=4,
            )
            payload = load_r3_window(window)

        self.assertEqual((8, 1, 2), payload["history"]["physical_node_state"].shape)
        self.assertEqual((4, 1, 2), payload["target"]["physical_node_state"].shape)
        self.assertEqual((8, 1, 3), payload["history_action"]["task_action"].shape)
        self.assertEqual((4, 1, 3), payload["future_action"]["task_action"].shape)
        self.assertEqual([2], payload["static"]["physical_node_kind_index"].tolist())
        self.assertFalse(
            set(payload["history"]) & set(payload["target"]) == set()
        )
        np.testing.assert_array_equal(
            payload["target"]["physical_node_state"],
            np.arange(24, dtype=np.float32).reshape(12, 1, 2)[8:12],
        )

    def test_formal_r3_inputs_are_bound_and_locked_labels_remain_sealed(self):
        from pi_jwm.r3_preflight_data import verify_r3_inputs

        code_root = Path(__file__).resolve().parents[1]
        report = verify_r3_inputs(
            code_root / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3",
            code_root / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3",
        )

        self.assertTrue(report["ready"])
        self.assertFalse(report["locked_test_accessed"])
        self.assertEqual(3, report["dag_feature_count"])
        self.assertEqual(18, report["information_edge_feature_count"])
        self.assertTrue(all(report["bindings"].values()))


class R3ExplicitStateBatchTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        history_steps, horizon_steps = 2, 1
        return {
            "schema_version": "PIJWM-R3-Smoke-View-v1",
            "window": {"trajectory_id": "fixture", "split": "train"},
            "history": {
                "physical_node_state": np.ones((history_steps, 2, 9), dtype=np.float32),
                "physical_node_feature_mask": np.ones((history_steps, 2, 9), dtype=bool),
                "physical_node_present": np.ones((history_steps, 2), dtype=bool),
                "physical_edge_state": np.ones((history_steps, 2, 7), dtype=np.float32),
                "physical_edge_feature_mask": np.ones((history_steps, 2, 7), dtype=bool),
                "physical_edge_present": np.ones((history_steps, 2), dtype=bool),
                "information_node_state": np.ones((history_steps, 2, 7), dtype=np.float32),
                "information_node_feature_mask": np.ones((history_steps, 2, 7), dtype=bool),
                "information_node_present": np.ones((history_steps, 2), dtype=bool),
                "information_edge_state": np.ones((history_steps, 2, 18), dtype=np.float32),
                "information_edge_feature_mask": np.ones((history_steps, 2, 18), dtype=bool),
                "information_edge_present": np.ones((history_steps, 2), dtype=bool),
                "data_flow_state": np.ones((history_steps, 1, 5), dtype=np.float32),
                "data_flow_present": np.ones((history_steps, 1), dtype=bool),
                "task_state": np.ones((history_steps, 1, 8), dtype=np.float32),
                "task_present": np.ones((history_steps, 1), dtype=bool),
                "task_lifecycle_index": np.ones((history_steps, 1), dtype=np.int16),
                "task_dag_state": np.ones((history_steps, 1, 3), dtype=np.float32),
                "task_dag_state_present": np.ones((history_steps, 1), dtype=bool),
            },
            "target": {
                "physical_node_state": np.ones((horizon_steps, 2, 9), dtype=np.float32),
                "physical_node_feature_mask": np.ones((horizon_steps, 2, 9), dtype=bool),
                "physical_node_present": np.ones((horizon_steps, 2), dtype=bool),
                "physical_edge_state": np.ones((horizon_steps, 2, 7), dtype=np.float32),
                "physical_edge_feature_mask": np.ones((horizon_steps, 2, 7), dtype=bool),
                "physical_edge_present": np.ones((horizon_steps, 2), dtype=bool),
                "information_node_state": np.ones((horizon_steps, 2, 7), dtype=np.float32),
                "information_node_feature_mask": np.ones((horizon_steps, 2, 7), dtype=bool),
                "information_node_present": np.ones((horizon_steps, 2), dtype=bool),
                "information_edge_state": np.ones((horizon_steps, 2, 18), dtype=np.float32),
                "information_edge_feature_mask": np.ones((horizon_steps, 2, 18), dtype=bool),
                "information_edge_present": np.ones((horizon_steps, 2), dtype=bool),
                "data_flow_state": np.ones((horizon_steps, 1, 5), dtype=np.float32),
                "data_flow_present": np.ones((horizon_steps, 1), dtype=bool),
                "task_state": np.ones((horizon_steps, 1, 8), dtype=np.float32),
                "task_present": np.ones((horizon_steps, 1), dtype=bool),
                "task_lifecycle_index": np.ones((horizon_steps, 1), dtype=np.int16),
                "task_dag_state": np.ones((horizon_steps, 1, 3), dtype=np.float32),
                "task_dag_state_present": np.ones((horizon_steps, 1), dtype=bool),
            },
            "future_action": {
                "task_action": np.ones((horizon_steps, 1, 8), dtype=np.float32),
                "task_action_present": np.ones((horizon_steps, 1), dtype=bool),
                "task_action_information_node_index": np.zeros(
                    (horizon_steps, 1, 4), dtype=np.int32
                ),
            },
            "history_action": {
                "task_action": np.ones((history_steps, 1, 8), dtype=np.float32),
                "task_action_present": np.ones((history_steps, 1), dtype=bool),
                "task_action_information_node_index": np.zeros(
                    (history_steps, 1, 4), dtype=np.int32
                ),
            },
            "static": {
                "physical_node_kind_index": np.asarray([0, 1], dtype=np.int16),
                "physical_edge_endpoint_index": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
                "information_edge_endpoint_index": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
                "cip_agent_node_index": np.asarray([0, 1], dtype=np.int32),
                "cep_information_to_physical_edge_index": np.asarray([0, 1], dtype=np.int32),
                "cfl_information_edge_index": np.asarray([0], dtype=np.int32),
                "data_flow_valid": np.asarray([True]),
                "task_valid": np.asarray([True]),
            },
        }

    def _stats(self) -> dict[str, object]:
        dims = {
            "physical_node_state": 9,
            "physical_edge_state": 7,
            "information_node_state": 7,
            "information_edge_state": 18,
            "data_flow_state": 5,
            "task_state": 8,
            "task_dag_state": 3,
        }
        return {
            "source_split": "train",
            "features": {
                name: {"mean": [0.5] * dim, "scale": [0.5] * dim}
                for name, dim in dims.items()
            },
        }

    def test_explicit_batch_preserves_all_v3_groups_and_masks(self):
        from pi_jwm.r3_preflight_data import make_explicit_batch

        batch = make_explicit_batch(self._payload(), self._stats())
        self.assertEqual((1, 2, 2, 9), tuple(batch.history["physical_node_state"].shape))
        self.assertEqual(7, batch.history["physical_edge_state"].shape[-1])
        self.assertEqual(7, batch.history["information_node_state"].shape[-1])
        self.assertEqual(18, batch.history["information_edge_state"].shape[-1])
        self.assertEqual(3, batch.history["task_dag_state"].shape[-1])
        self.assertEqual((1, 2, 1, 8), tuple(batch.history_action["task_action"].shape))
        self.assertTrue(batch.history["information_edge_feature_mask"].all())
        self.assertTrue(batch.target["information_link_activity"].all())
        self.assertTrue(batch.target["information_link_activity_mask"].all())
        self.assertEqual("fixture", batch.metadata["trajectory_id"])

    def test_masked_missing_value_stays_zero_after_normalization(self):
        from pi_jwm.r3_preflight_data import make_explicit_batch

        payload = self._payload()
        payload["history"]["information_edge_feature_mask"][0, 0, 0] = False
        payload["history"]["information_edge_state"][0, 0, 0] = 0.0
        batch = make_explicit_batch(payload, self._stats())
        self.assertEqual(0.0, batch.history["information_edge_state"][0, 0, 0, 0].item())

    def test_nonzero_masked_missing_value_is_rejected(self):
        from pi_jwm.r3_preflight_data import make_explicit_batch

        payload = self._payload()
        payload["history"]["physical_node_feature_mask"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "physical_node_state"):
            make_explicit_batch(payload, self._stats())

    def test_normalization_must_be_train_only(self):
        from pi_jwm.r3_preflight_data import make_explicit_batch

        stats = self._stats()
        stats["source_split"] = "validation"
        with self.assertRaisesRegex(ValueError, "train-only"):
            make_explicit_batch(self._payload(), stats)

    def test_out_of_range_cross_graph_index_is_rejected(self):
        from pi_jwm.r3_preflight_data import make_explicit_batch

        payload = self._payload()
        payload["static"]["cip_agent_node_index"][1] = 2
        with self.assertRaisesRegex(ValueError, "cip_agent_node_index"):
            make_explicit_batch(payload, self._stats())


if __name__ == "__main__":
    unittest.main()
