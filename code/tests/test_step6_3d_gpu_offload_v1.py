"""CPU storage for GPU search results preserves causal result identity."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

from pi_jwm.step6_1_trained_candidate_rollout_v1 import OneStepRolloutResult
from run_step6_3d_one_cpu_solve_v1 import offload_transition_result


class GpuOffloadTests(unittest.TestCase):
    def test_result_tree_moves_to_cpu_without_changing_values_or_fingerprints(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        value = torch.tensor([[1.25, -2.5]], dtype=torch.float32, device=device)
        result = OneStepRolloutResult(
            {"h": value}, {"position": value}, {"edge": value}, {},
            {"mobility_values": value}, {"mapping": 3},
            {"rule": {"cpu_service": value}},
            {"latent": "parent", "state": "parent", "graph": "parent"},
            {"latent": "child", "state": "child", "graph": "child"},
        )
        offloaded = offload_transition_result(result)
        self.assertEqual(offloaded.latent["h"].device.type, "cpu")
        self.assertEqual(offloaded.state["position"].device.type, "cpu")
        self.assertEqual(offloaded.graph["edge"].device.type, "cpu")
        self.assertEqual(offloaded.action_tensor["mobility_values"].device.type, "cpu")
        self.assertEqual(offloaded.model_trace["rule"]["cpu_service"].device.type, "cpu")
        self.assertTrue(torch.equal(offloaded.latent["h"], value.cpu()))
        self.assertEqual(offloaded.input_fingerprints, result.input_fingerprints)
        self.assertEqual(offloaded.output_fingerprints, result.output_fingerprints)


if __name__ == "__main__":
    unittest.main()
