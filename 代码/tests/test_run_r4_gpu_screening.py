from __future__ import annotations

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


class RunR4GpuScreeningTests(unittest.TestCase):
    def setUp(self):
        self.evaluation_root = (
            CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        )

    def test_candidate_configs_cover_the_frozen_twelve_arm_matrix(self):
        from run_r4_gpu_screening import executable_candidate_configs

        configs = executable_candidate_configs(
            hidden_dim=16,
            history_steps=8,
            information_rate_mean=0.0,
            information_rate_scale=1.0,
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

    def test_formal_runner_rejects_non_cuda_device_before_output(self):
        from run_r4_gpu_screening import require_cuda_device

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                require_cuda_device("cpu", output)
            self.assertFalse(output.exists())

    def test_cuda_statistics_use_an_integer_device_index(self):
        from run_r4_gpu_screening import cuda_device_index

        self.assertEqual(0, cuda_device_index("cuda:0"))
        self.assertEqual(2, cuda_device_index("cuda:2"))

    def test_checkpoint_selection_scales_are_bound_and_complete(self):
        from run_r4_gpu_screening import load_selection_scales

        scales = load_selection_scales(self.evaluation_root)
        self.assertEqual(10, len(scales))
        self.assertTrue(all(value > 0.0 for value in scales.values()))

    def test_checkpoint_score_reproduction_uses_frozen_minimum_improvement(self):
        from run_r4_gpu_screening import checkpoint_score_reproduced

        self.assertTrue(checkpoint_score_reproduced(4.0, 4.0 + 9.0e-5, 1.0e-4))
        self.assertFalse(checkpoint_score_reproduced(4.0, 4.0 + 1.1e-4, 1.0e-4))


if __name__ == "__main__":
    unittest.main()
