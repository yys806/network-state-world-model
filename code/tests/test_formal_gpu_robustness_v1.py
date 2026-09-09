from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalGpuRobustnessV1Tests(unittest.TestCase):
    def test_keywise_protocol_lists_only_continuous_history_keys(self):
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from run_formal_gpu_robustness_keywise_v1 import KEYWISE_PERTURBATION_KEYS

        self.assertEqual(
            KEYWISE_PERTURBATION_KEYS,
            ("node_state", "physical_edge_state", "flow_state", "task_state"),
        )

    def test_execution_policy_records_evaluation_device(self):
        from pi_jwm.formal_gpu_robustness_v1 import execution_policy

        cpu_policy = execution_policy(torch.device("cpu"))
        self.assertFalse(cpu_policy["gpu_execution"])
        self.assertEqual(cpu_policy["evaluation_device"], "cpu")

        cuda_policy = execution_policy(torch.device("cuda"))
        self.assertTrue(cuda_policy["gpu_execution"])
        self.assertEqual(cuda_policy["evaluation_device"], "cuda")

    def test_perturb_history_changes_only_masked_continuous_observations(self):
        from pi_jwm.formal_gpu_robustness_v1 import perturb_history

        history = {
            "node_state": torch.zeros(1, 2, 3, 2),
            "node_present": torch.tensor([[[True, False, True], [True, True, False]]]),
            "physical_edge_state": torch.zeros(1, 2, 4, 2),
            "physical_edge_present": torch.tensor([[[True, False, True, False], [True, True, False, False]]]),
            "flow_state": torch.zeros(1, 2, 2, 2),
            "flow_present": torch.ones(1, 2, 2, dtype=torch.bool),
            "task_state": torch.zeros(1, 2, 5, 2),
            "task_present": torch.zeros(1, 2, 5, dtype=torch.bool),
            "task_action": torch.full((1, 2, 5, 3), 7.0),
        }
        perturbed = perturb_history(history, noise_scale=0.5, seed=17)

        self.assertTrue(torch.equal(history["task_action"], perturbed["task_action"]))
        self.assertTrue(torch.equal(history["node_present"], perturbed["node_present"]))
        self.assertTrue(torch.equal(history["task_state"], perturbed["task_state"]))
        invalid = ~history["node_present"].unsqueeze(-1).expand_as(history["node_state"])
        valid = history["node_present"].unsqueeze(-1).expand_as(history["node_state"])
        self.assertTrue(torch.equal(history["node_state"][invalid], perturbed["node_state"][invalid]))
        self.assertTrue(torch.any(history["node_state"][valid] != perturbed["node_state"][valid]))

    def test_zero_noise_is_exact_identity(self):
        from pi_jwm.formal_gpu_robustness_v1 import perturb_history

        value = {"node_state": torch.randn(1, 2, 3, 2), "node_present": torch.ones(1, 2, 3, dtype=torch.bool)}
        result = perturb_history(value, noise_scale=0.0, seed=3)
        self.assertTrue(torch.equal(value["node_state"], result["node_state"]))
        self.assertIsNot(value, result)

    def test_perturb_history_can_isolate_one_continuous_key(self):
        from pi_jwm.formal_gpu_robustness_v1 import perturb_history

        history = {
            "node_state": torch.zeros(1, 1, 2, 2),
            "node_present": torch.ones(1, 1, 2, dtype=torch.bool),
            "physical_edge_state": torch.zeros(1, 1, 2, 2),
            "physical_edge_present": torch.ones(1, 1, 2, dtype=torch.bool),
            "flow_state": torch.zeros(1, 1, 2, 2),
            "flow_present": torch.ones(1, 1, 2, dtype=torch.bool),
            "task_state": torch.zeros(1, 1, 2, 2),
            "task_present": torch.ones(1, 1, 2, dtype=torch.bool),
        }
        perturbed = perturb_history(
            history,
            noise_scale=0.5,
            seed=17,
            perturb_keys=("node_state",),
        )

        self.assertTrue(torch.any(history["node_state"] != perturbed["node_state"]))
        for key in ("physical_edge_state", "flow_state", "task_state"):
            self.assertTrue(torch.equal(history[key], perturbed[key]))


if __name__ == "__main__":
    unittest.main()
