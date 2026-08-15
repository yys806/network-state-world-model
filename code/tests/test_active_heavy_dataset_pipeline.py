import sys
import os
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class ActiveHeavyDatasetPipelineTest(unittest.TestCase):
    def test_build_subprocess_env_exposes_reference_airfogsim_package(self):
        from prepare_active_heavy_dataset_v1 import (
            AIRFOGSIM_EXAMPLE_ROOT,
            AIRFOGSIM_REFERENCE_ROOT,
            build_subprocess_env,
            build_subprocess_workdir,
        )

        env = build_subprocess_env({"PYTHONPATH": "existing-path", "KEEP": "yes"})
        python_paths = env["PYTHONPATH"].split(os.pathsep)

        self.assertEqual(Path(python_paths[0]), AIRFOGSIM_REFERENCE_ROOT)
        self.assertIn("existing-path", python_paths)
        self.assertEqual(env["KEEP"], "yes")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(Path(env["PI_JWM_AIRFOGSIM_EXAMPLE_DIR"]), AIRFOGSIM_EXAMPLE_ROOT)
        self.assertEqual(build_subprocess_workdir(), AIRFOGSIM_EXAMPLE_ROOT)

    def test_build_pipeline_commands_uses_distinct_v1_paths_and_seed_list(self):
        from prepare_active_heavy_dataset_v1 import build_pipeline_commands

        commands = build_pipeline_commands(
            seeds=[0, 1, 2],
            max_time=30.0,
            output_tag="active_heavy_v1",
        )

        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("export_multiseed_dataset_v0.py", joined)
        self.assertIn("export_strict_actions_v0.py", joined)
        self.assertIn("build_dataset_multiseed_v0.py", joined)
        self.assertIn("build_edge_action_v0.py", joined)
        self.assertIn("build_world_model_dataset_v0.py", joined)
        self.assertIn("world_model_dataset_active_heavy_v1", joined)
        self.assertIn("--seeds 0 1 2", joined)
        self.assertIn("--max-time 30.0", joined)

    def test_summarize_active_rate_distribution_counts_active_and_high_rate_items(self):
        from prepare_active_heavy_dataset_v1 import summarize_active_rate_distribution

        arrays = {
            "y_link_active": np.array([[[1, 0], [1, 1]], [[0, 0], [1, 0]]], dtype=np.float32),
            "y_link_rate": np.array([[[100, 0], [650, 250]], [[0, 0], [700, 0]]], dtype=np.float32),
            "sample_seed": np.array([0, 1], dtype=np.int32),
        }

        summary = summarize_active_rate_distribution(arrays, high_rate_threshold=600.0)

        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["active_link_steps"], 4)
        self.assertEqual(summary["high_rate_active_steps"], 2)
        self.assertAlmostEqual(summary["active_ratio"], 4 / 8)
        self.assertEqual(summary["per_seed"][0]["active_link_steps"], 3)
        self.assertEqual(summary["per_seed"][1]["high_rate_active_steps"], 1)


if __name__ == "__main__":
    unittest.main()
