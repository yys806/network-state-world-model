from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalGpuMultiseedAuditV1Tests(unittest.TestCase):
    def _write_run(self, root: Path, seed: int, *, locked: bool = False, hidden_dim: int = 32) -> Path:
        run = root / f"run_{seed}"
        (run / "metrics").mkdir(parents=True)
        config = {
            "schema_version": "PI-JWM-formal-training-config-v1",
            "seed": seed,
            "data_seed": 20260823,
            "device": "cuda",
            "splits": ["train", "validation", "calibration"],
            "train_limit": 256,
            "evaluation_limit": 128,
            "hidden_dim": hidden_dim,
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 3e-4,
            "zero_init_residual_state_heads": True,
            "residual_state_scale": 0.5,
            "learned_methods": ["coupled_dual_gnn_residual"],
            "tensor_root": "/tensor/nonlocked",
            "dataset_manifest_sha256": "dataset-hash",
            "locked_test_accessed": locked,
        }
        summary = {
            "schema_version": "PI-JWM-formal-training-summary-v1",
            "training_run_complete": True,
            "gpu_execution": True,
            "locked_test_accessed": locked,
            "formal_performance_claim_ready": False,
            "sample_counts": {"train": 256, "validation": 128, "calibration": 128},
        }
        comparison = [
            {
                "method": "last_persistence",
                "validation_link_f1": "0.68",
                "calibration_link_f1": "0.20",
                "validation_node_x_mae": "1.0",
                "validation_throughput_mae": "1.0",
                "validation_rb_occupancy_mae": "1.0",
                "validation_task_delay_mae": "1.0",
            },
            {
                "method": "coupled_dual_gnn_residual",
                "validation_link_f1": "0.74",
                "calibration_link_f1": "0.30",
                "validation_node_x_mae": "1.1",
                "validation_throughput_mae": "0.9",
                "validation_rb_occupancy_mae": "0.9",
                "validation_task_delay_mae": "0.9",
            },
        ]
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with (run / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison[0].keys())
            writer.writeheader()
            writer.writerows(comparison)
        (run / "metrics" / "coupled_dual_gnn_residual__threshold_selection.json").write_text(
            json.dumps({"selection_split": "calibration"}), encoding="utf-8"
        )
        files = {}
        for path in sorted(run.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                files[str(path.relative_to(run)).replace("\\", "/")] = {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        (run / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        return run

    def test_audit_reports_consistent_three_gpu_runs(self):
        from pi_jwm.formal_gpu_multiseed_audit_v1 import audit_gpu_multiseed_runs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = [self._write_run(root, seed) for seed in (20260824, 20260825, 20260826)]
            report = audit_gpu_multiseed_runs(runs)

        self.assertTrue(report["audit_passed"])
        self.assertEqual(3, report["seed_count"])
        self.assertEqual(0, report["total_manifest_mismatches"])
        self.assertAlmostEqual(0.06, report["metrics"]["validation_link_f1_delta"]["mean"])
        self.assertFalse(report["formal_performance_claim_ready"])

    def test_audit_rejects_locked_test_and_contract_drift(self):
        from pi_jwm.formal_gpu_multiseed_audit_v1 import audit_gpu_multiseed_runs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            locked_run = self._write_run(root, 20260824, locked=True)
            with self.assertRaisesRegex(ValueError, "locked_test"):
                audit_gpu_multiseed_runs([locked_run])

            first = self._write_run(root, 20260825)
            drifted = self._write_run(root, 20260826, hidden_dim=64)
            with self.assertRaisesRegex(ValueError, "contract drift"):
                audit_gpu_multiseed_runs([first, drifted])


if __name__ == "__main__":
    unittest.main()
