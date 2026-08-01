from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
TEST_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_airfogsim_smoke_model_v2 import fake_batch


def fake_stats():
    features = {"node_state": 7, "physical_edge_state": 5, "flow_state": 5, "task_state": 8}
    return {
        "features": {
            name: {"mean": [1.0] * count, "scale": [2.0] * count}
            for name, count in features.items()
        }
    }


class SparseDiagnosticsV2Tests(unittest.TestCase):
    def test_zero_activity_and_persistence_baselines(self):
        from pi_jwm.airfogsim_sparse_diagnostics_v2 import (
            build_last_persistence_prediction,
            build_zero_activity_prediction,
        )

        history, target, static = fake_batch()
        history["link_activity"] = history["physical_edge_present"].clone()
        history["task_lifecycle_index"] = torch.full_like(history["task_present"], -1, dtype=torch.long)
        history["task_lifecycle_index"][history["task_present"]] = 3
        batch = {"history": history, "target": target, "static": static}

        zero = build_zero_activity_prediction(batch, fake_stats(), lifecycle_majority_index=1)
        self.assertTrue(torch.all(zero["link_activity_logits"] < 0))
        self.assertTrue(torch.all(zero["flow_presence_logits"] < 0))
        self.assertTrue(torch.all(zero["task_presence_logits"] < 0))
        self.assertTrue(torch.all(zero["task_lifecycle_logits"].argmax(dim=-1) == 1))

        persistent = build_last_persistence_prediction(batch, horizon_steps=2)
        torch.testing.assert_close(persistent["task_state"][:, 0], history["task_state"][:, -1])
        torch.testing.assert_close(persistent["task_state"][:, 1], history["task_state"][:, -1])
        self.assertTrue(torch.all(persistent["task_lifecycle_logits"][..., 3][target["task_present"]] > 0))

    def test_average_precision_handles_ties_and_no_positives(self):
        from pi_jwm.airfogsim_sparse_diagnostics_v2 import average_precision

        self.assertAlmostEqual(1.0, average_precision([0.9, 0.8, 0.1], [1, 1, 0]))
        self.assertIsNone(average_precision([0.1, 0.2], [0, 0]))
        self.assertAlmostEqual(2 / 3, average_precision([0.0, 0.0, 0.0], [1, 0, 1]))

    def test_evaluator_reports_required_metric_groups(self):
        from pi_jwm.airfogsim_sparse_diagnostics_v2 import (
            build_last_persistence_prediction,
            evaluate_prediction_batches,
        )

        history, target, static = fake_batch()
        history["link_activity"] = history["physical_edge_present"].clone()
        history["task_lifecycle_index"] = torch.full_like(history["task_present"], 2, dtype=torch.long)
        batch = {"history": history, "target": target, "static": static}
        prediction = build_last_persistence_prediction(batch, horizon_steps=2)
        report = evaluate_prediction_batches([prediction], [batch], fake_stats())

        self.assertIn("auprc", report["link_activity"])
        self.assertIn("rmse", report["active_only_rate"])
        self.assertIn("flow", report["presence"])
        self.assertIn("task", report["presence"])
        self.assertIn("macro_f1", report["task_lifecycle"])
        self.assertEqual(5, len(report["task_lifecycle"]["support"]))
        self.assertIn("physical_edge", report["state_mae_physical_units"])


if __name__ == "__main__":
    unittest.main()
