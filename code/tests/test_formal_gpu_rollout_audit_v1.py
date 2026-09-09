from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalGpuRolloutAuditV1Tests(unittest.TestCase):
    def _write_metrics(self, root: Path, method: str, values: list[float]) -> None:
        horizons = {}
        for index, value in enumerate(values, 1):
            horizons[f"k={index}"] = {
                "metrics": {
                    "state.node.x.mae": {"value": value},
                    "link.active_only_rate.mae": {"value": 2.0 + value},
                    "system.communication_throughput.mae": {"value": 3.0 + value},
                    "resource.rb_occupancy.mae": {"value": 4.0 + value},
                    "uncertainty.node.x.coverage_95": {"value": 0.99 - value / 100.0},
                }
            }
        (root / f"{method}__validation.json").write_text(
            json.dumps({"horizons": horizons}), encoding="utf-8"
        )

    def test_audit_reports_horizon_deltas_and_error_growth(self):
        from pi_jwm.formal_gpu_rollout_audit_v1 import audit_gpu_rollout_metrics

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_metrics(root, "coupled_dual_gnn_residual", [1.2, 2.0, 3.6])
            self._write_metrics(root, "last_persistence", [1.0, 1.8, 3.0])
            report = audit_gpu_rollout_metrics(root)

        self.assertEqual(["k=1", "k=2", "k=3"], report["horizons"])
        self.assertAlmostEqual(0.2, report["per_horizon"]["k=1"]["node_x_mae_delta"])
        self.assertAlmostEqual(0.6, report["per_horizon"]["k=3"]["node_x_mae_delta"])
        self.assertGreater(report["learned_node_x_error_growth"], 1.0)
        self.assertFalse(report["formal_performance_claim_ready"])

    def test_audit_rejects_missing_horizon_or_locked_path(self):
        from pi_jwm.formal_gpu_rollout_audit_v1 import audit_gpu_rollout_metrics

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_metrics(root, "coupled_dual_gnn_residual", [1.0, 2.0])
            self._write_metrics(root, "last_persistence", [1.0, 2.0])
            with self.assertRaisesRegex(ValueError, "horizon"):
                audit_gpu_rollout_metrics(root)


if __name__ == "__main__":
    unittest.main()
