from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
TEST_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, TEST_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from test_r3_world_model import make_batch


class R4GpuScreeningTests(unittest.TestCase):
    def setUp(self):
        self.dataset_root = (
            CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        )
        self.evaluation_root = (
            CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        )

    def test_frozen_screening_protocol_is_loaded_without_budget_drift(self):
        from pi_jwm.r4_gpu_screening import load_frozen_screening_protocol

        protocol = load_frozen_screening_protocol(self.evaluation_root)

        self.assertEqual(20260803, protocol.training_seed)
        self.assertEqual(30, protocol.max_epochs)
        self.assertEqual(5, protocol.patience)
        self.assertEqual(32, protocol.effective_batch_size)
        self.assertEqual(1.0e-4, protocol.minimum_improvement)

    def test_window_schedule_is_deterministic_balanced_and_nonlocked(self):
        from pi_jwm.r4_gpu_screening import build_training_window_schedule

        first = build_training_window_schedule(
            self.dataset_root,
            epochs=3,
            windows_per_epoch=32,
            seed=20260803,
        )
        second = build_training_window_schedule(
            self.dataset_root,
            epochs=3,
            windows_per_epoch=32,
            seed=20260803,
        )

        self.assertEqual(first, second)
        self.assertEqual(3, len(first))
        self.assertTrue(all(len(epoch) == 32 for epoch in first))
        for epoch in first:
            self.assertEqual(32, len({window.environment_seed for window in epoch}))
            self.assertTrue(all(window.split == "train" for window in epoch))
            self.assertTrue(all(window.horizon_steps == 3 for window in epoch))

    def test_locked_test_schedule_is_rejected(self):
        from pi_jwm.r4_gpu_screening import build_validation_windows

        with self.assertRaisesRegex(ValueError, "locked_test"):
            build_validation_windows(
                self.dataset_root,
                split="locked_test",
                horizons=(1,),
                seed=20260803,
            )

    def test_explicit_batches_collate_and_move_without_losing_namespaces(self):
        from pi_jwm.r4_gpu_screening import collate_explicit_batches, move_explicit_batch

        left = make_batch(horizon=1)
        right = make_batch(horizon=1)
        combined = collate_explicit_batches([left, right])
        moved = move_explicit_batch(combined, "cpu")

        self.assertEqual(2, moved.history["physical_node_state"].shape[0])
        self.assertEqual(2, moved.static["physical_edge_endpoint_index"].shape[0])
        self.assertEqual(2, len(moved.metadata["items"]))
        self.assertEqual(set(left.target), set(moved.target))

    def test_validation_metrics_use_information_edges_and_frozen_four_term_score(self):
        from pi_jwm.r4_gpu_screening import R4ValidationAccumulator

        batch = make_batch(horizon=1)
        batch.target["information_edge_state"].zero_()
        batch.target["information_edge_state"][0, 0, :2, 11] = torch.tensor([1.0, 0.0])
        batch.target["information_edge_state"][0, 0, :2, 12] = torch.tensor([2.0, 0.0])
        batch.target["information_link_activity"] = (
            batch.target["information_edge_state"][..., 11] > 0
        )
        batch.target["information_link_activity_mask"] = torch.ones_like(
            batch.target["information_link_activity"], dtype=torch.bool
        )
        batch.target["task_lifecycle_index"][0, 0] = torch.tensor([0, 1])

        predicted_explicit = {
            key: value.clone()
            for key, value in batch.target.items()
            if key.endswith("_state")
        }
        activity_logits = torch.where(
            batch.target["information_link_activity"],
            torch.tensor(12.0),
            torch.tensor(-12.0),
        )
        lifecycle_logits = torch.full((1, 1, 2, 5), -12.0)
        lifecycle_logits[0, 0, 0, 0] = 12.0
        lifecycle_logits[0, 0, 1, 1] = 12.0
        output = SimpleNamespace(
            predicted_explicit=predicted_explicit,
            predicted_logits={
                "information_link_activity": activity_logits,
                "task_lifecycle": lifecycle_logits,
            },
        )
        stats = {
            "source_split": "train",
            "features": {
                "physical_node_state": {"scale": [1.0] * 9},
                "physical_edge_state": {"scale": [1.0] * 7},
                "information_node_state": {"scale": [1.0] * 7},
                "information_edge_state": {"scale": [1.0] * 18},
                "data_flow_state": {"scale": [1.0] * 5},
                "task_state": {"scale": [1.0] * 8},
                "task_dag_state": {"scale": [1.0] * 3},
            },
        }
        accumulator = R4ValidationAccumulator(
            stats,
            selection_scales={
                "state.physical_node.position.rmse": 1.0,
                "state.physical_node.motion.rmse": 1.0,
                "state.physical_edge.distance.rmse": 1.0,
                "state.physical_edge.relative_speed.rmse": 1.0,
                "state.information_node.queue.mae": 1.0,
                "state.information_node.cpu_backlog.mae": 1.0,
                "state.information_edge.rate.rmse": 1.0,
                "state.flow.remaining_data.mae": 1.0,
                "state.task.deadline_remaining.mae": 1.0,
                "state.dag.unfinished_parent_count.mae": 1.0,
            },
        )
        accumulator.update(output, batch)
        report = accumulator.finalize()

        self.assertEqual(1.0, report["metrics"]["event.information_link_activity.auprc"]["value"])
        self.assertEqual(0.0, report["metrics"]["link.active_only_rate.mae"]["value"])
        self.assertEqual(0.4, report["metrics"]["task.lifecycle.macro_f1"]["value"])
        self.assertEqual(
            0.0,
            report["metrics"]["selection.required_continuous.normalized_error"]["value"],
        )
        self.assertAlmostEqual(0.15, report["validation_protocol_score"])


if __name__ == "__main__":
    unittest.main()
