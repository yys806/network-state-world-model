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
    "candidate_matrix.json",
    "selected_windows.json",
    "candidate_reports.json",
    "objective_reports.json",
    "checkpoint_reports.json",
    "input_provenance.json",
    "failed_candidates.json",
    "manifest.json",
)


class RunR4CpuPreflightTests(unittest.TestCase):
    def setUp(self):
        self.dataset_root = (
            CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        )
        self.evaluation_root = (
            CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        )

    def test_candidate_configs_cover_all_currently_executable_single_module_arms(self):
        from run_r4_cpu_preflight import executable_candidate_configs

        configs = executable_candidate_configs(
            hidden_dim=4,
            history_steps=8,
            information_rate_mean=1.0,
            information_rate_scale=2.0,
        )
        self.assertEqual(
            {
                "reference",
                "symlog_masked_mlp_v1",
                "simnorm_masked_mlp_v1",
                "rgcn_v1",
                "edge_conditioned_relation_mpnn_v1",
                "no_cross_graph_coupling_v1",
                "relation_constrained_cross_attention_v1",
                "graph_rssm_v1",
                "heteroscedastic_typed_v1",
                "hurdle_active_rate_v1",
                "explicit_dag_message_passing_v1",
                "soft_predicted_presence_v1",
            },
            set(configs),
        )
        reference = configs["reference"]
        for name, config in configs.items():
            changed = {
                family
                for family in reference.component_names()
                if getattr(reference, family) != getattr(config, family)
            }
            self.assertLessEqual(len(changed), 1, name)

    def test_runner_writes_auditable_nonlocked_cpu_artifact(self):
        from run_r4_cpu_preflight import run_r4_cpu_preflight

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "r4"
            result = run_r4_cpu_preflight(
                self.dataset_root,
                self.evaluation_root,
                output,
                per_horizon=1,
                splits=("train",),
                horizons=(1,),
                hidden_dim=4,
                candidate_names=("reference", "no_cross_graph_coupling_v1"),
            )
            self.assertTrue(result["r4_cpu_preflight_ready"])
            self.assertFalse(result["gpu_screening_ready"])
            self.assertFalse(result["full_executable_matrix_run"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertEqual(2, result["candidate_count"])
            for name in REQUIRED_OUTPUTS:
                self.assertTrue((output / name).is_file(), name)
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertTrue(manifest["r4_cpu_preflight_ready"])
            self.assertNotIn("manifest.json", manifest["files"])

    def test_runner_rejects_locked_test_request_before_output(self):
        from run_r4_cpu_preflight import run_r4_cpu_preflight

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "locked_test"):
                run_r4_cpu_preflight(
                    self.dataset_root,
                    self.evaluation_root,
                    Path(temporary) / "r4",
                    splits=("locked_test",),
                    horizons=(1,),
                    candidate_names=("reference",),
                )


if __name__ == "__main__":
    unittest.main()
