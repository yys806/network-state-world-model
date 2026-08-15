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
    "selected_windows.json",
    "objective_reports.json",
    "rollout_checks.json",
    "metric_interface_report.json",
    "input_provenance.json",
    "coupled_reference.pt",
    "no_coupling_control.pt",
    "manifest.json",
)


class RunR3CpuPreflightTests(unittest.TestCase):
    def test_r3_runner_writes_complete_nonlocked_artifact(self):
        from run_r3_cpu_preflight import run_r3_cpu_preflight

        dataset_root = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        evaluation_root = CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "r3"
            result = run_r3_cpu_preflight(
                dataset_root,
                evaluation_root,
                output,
                per_horizon=1,
                splits=("train",),
                horizons=(1,),
                hidden_dim=4,
            )
            self.assertTrue(result["r3_cpu_preflight_ready"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertEqual([1], result["rollout_horizons"])
            for name in REQUIRED_OUTPUTS:
                self.assertTrue((output / name).is_file(), name)
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertTrue(manifest["r3_cpu_preflight_ready"])
            self.assertNotIn("manifest.json", manifest["files"])


if __name__ == "__main__":
    unittest.main()
