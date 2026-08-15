from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


REQUIRED_OUTPUTS = (
    "preflight_summary.json",
    "protocol.json",
    "combination_matrix.json",
    "selected_windows.json",
    "combination_seed_reports.json",
    "objective_reports.json",
    "validation_gate_reports.json",
    "checkpoint_reports.json",
    "input_provenance.json",
    "gpu_handoff.json",
    "failed_runs.json",
    "manifest.json",
)


class RunR5CpuPreflightTests(unittest.TestCase):
    def setUp(self):
        self.dataset_root = (
            CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        )
        self.evaluation_root = (
            CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        )
        self.r4_root = (
            CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r4_gpu_screening_v1"
        )

    def test_runner_writes_nonlocked_auditable_subset_without_gpu_claim(self):
        from run_r5_cpu_preflight import run_r5_cpu_preflight

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "r5"
            result = run_r5_cpu_preflight(
                self.dataset_root,
                self.evaluation_root,
                self.r4_root,
                output,
                splits=("train", "validation"),
                horizons=(1, 5, 20),
                per_horizon=1,
                hidden_dim=4,
                combination_ids=("A", "C"),
                execution_seeds=(20260803,),
            )
            self.assertTrue(result["r5_cpu_preflight_ready"])
            self.assertFalse(result["r5_gpu_ready"])
            self.assertFalse(result["full_matrix_and_seed_budget_run"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertEqual(2, result["combination_count"])
            self.assertEqual(2, result["completed_run_count"])
            for name in REQUIRED_OUTPUTS:
                self.assertTrue((output / name).is_file(), name)

            reports = json.loads(
                (output / "combination_seed_reports.json").read_text("utf-8")
            )
            c_report = next(row for row in reports if row["combination_id"] == "C")
            self.assertTrue(c_report["module_gradient_evidence"]["rssm"])
            self.assertTrue(c_report["module_gradient_evidence"]["heteroscedastic"])
            gates = json.loads(
                (output / "validation_gate_reports.json").read_text("utf-8")
            )
            self.assertEqual(2, len(gates))
            self.assertTrue(all(row["all_public_metrics_finite"] for row in gates))

            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertTrue(manifest["r5_cpu_preflight_ready"])
            self.assertNotIn("manifest.json", manifest["files"])

    def test_runner_rejects_locked_test_before_creating_output(self):
        from run_r5_cpu_preflight import run_r5_cpu_preflight

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "r5"
            with self.assertRaisesRegex(ValueError, "locked_test"):
                run_r5_cpu_preflight(
                    self.dataset_root,
                    self.evaluation_root,
                    self.r4_root,
                    output,
                    splits=("locked_test",),
                    horizons=(1,),
                    combination_ids=("A",),
                    execution_seeds=(20260803,),
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
