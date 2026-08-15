import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_v11_foundation_phase0_2 import (  # noqa: E402
    build_action_target_correlations,
    build_coupled_consistency,
    build_metric_contract,
    pearson,
    spearman,
)


def make_arrays():
    samples, horizon, history, edges, nodes = 4, 2, 2, 3, 2
    action_features = np.array(
        ["offload_count", "rb_task_count", "rb_total", "cpu_task_count", "cpu_total", "return_count"]
    )
    future = np.zeros((samples, horizon, edges, len(action_features)), dtype=np.float32)
    future[..., 1] = 1.0
    future[..., 2] = np.arange(samples * horizon * edges, dtype=np.float32).reshape(samples, horizon, edges)
    future[..., 3] = future[..., 2] + 1.0
    future[..., 4] = future[..., 2] * 0.5
    active = (future[..., 2] > 3).astype(np.float32)
    rate = future[..., 2] * 10.0
    return {
        "x_node": np.zeros((samples, history, nodes, 7), dtype=np.float32),
        "x_link": np.zeros((samples, history, edges, 5), dtype=np.float32),
        "x_task": np.zeros((samples, history, 9), dtype=np.float32),
        "edge_a_hist": np.zeros((samples, history, edges, len(action_features)), dtype=np.float32),
        "edge_a_future": future,
        "y_link_active": active,
        "y_link_rate": rate,
        "y_link": np.zeros((samples, horizon, edges, 5), dtype=np.float32),
        "y_node": np.zeros((samples, horizon, nodes, 7), dtype=np.float32),
        "y_task": np.zeros((samples, horizon, 9), dtype=np.float32),
        "edge_src_idx": np.array([0, 0, 1], dtype=np.int64),
        "edge_dst_idx": np.array([1, 1, 0], dtype=np.int64),
        "valid_edge_node": np.ones((edges,), dtype=np.int64),
        "node_features": np.array(["x", "y", "z", "speed", "acceleration", "cpu", "storage"]),
        "link_features": np.array(["distance", "rate_sum", "csi_mean", "active_task_count", "allocated_rb_count"]),
        "task_features": np.array(
            [
                "num_tasks",
                "total_task_size",
                "total_task_cpu",
                "mean_deadline",
                "mean_priority",
                "num_to_offload",
                "num_computing",
                "num_returning",
                "num_finished",
            ]
        ),
        "edge_action_features": action_features,
    }


class FoundationAuditTests(unittest.TestCase):
    def test_correlations_handle_monotonic_inputs(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(pearson(x, y), 1.0)
        self.assertAlmostEqual(spearman(x, y), 1.0)

    def test_metric_contract_marks_delay_as_partial_proxy(self):
        rows = build_metric_contract(make_arrays())
        status = {row["metric"]: row["status"] for row in rows}
        self.assertEqual(status["active_rate_rmse"], "available")
        self.assertEqual(status["delay"], "partial_proxy")
        self.assertEqual(status["ood_uncertainty"], "missing_or_diagnostic_only")

    def test_coupled_consistency_reports_rb_and_cpu_groups(self):
        arrays = make_arrays()
        split_indices = {"train": np.array([0, 1]), "test": np.array([2, 3])}
        rows, _ = build_coupled_consistency(arrays, split_indices)
        by_group = {(row["split"], row["group"]): row for row in rows}
        self.assertIn(("train", "rb_coupled"), by_group)
        self.assertIn(("test", "cpu_coupled"), by_group)
        self.assertGreater(by_group[("test", "cpu_coupled")]["pearson_count_total"], 0.0)

    def test_action_target_correlations_include_step_resource_total(self):
        arrays = make_arrays()
        rows = build_action_target_correlations(arrays, {"train": np.array([0, 1, 2, 3])})
        signals = {(row["action_signal"], row["target_signal"]) for row in rows}
        self.assertIn(("step_rb_cpu_total", "target_active_rate_sum"), signals)


if __name__ == "__main__":
    unittest.main()
