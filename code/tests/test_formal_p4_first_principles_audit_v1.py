import unittest

import numpy as np
import torch

from pi_jwm.formal_p4_first_principles_audit_v1 import (
    gate_aware_checkpoint_key,
    gradient_cosine_summary,
    masked_global_l1_oracle,
    shared_offset_preserves_ranking,
)


class FormalP4FirstPrinciplesAuditTests(unittest.TestCase):
    def test_global_l1_oracle_uses_per_sample_step_median(self):
        residual = np.array([[[[1.0], [3.0], [9.0]]]], dtype=np.float64)
        valid = np.array([[[True, True, True]]])

        result = masked_global_l1_oracle(residual, valid)

        np.testing.assert_allclose(result["correction"], [[[[3.0]]]])
        self.assertAlmostEqual(result["global_oracle_mae"], 8.0 / 3.0)
        self.assertEqual(result["entity_oracle_mae"], 0.0)

    def test_shared_edge_offset_cannot_change_valid_ranking(self):
        logits = np.array([[[0.2, -0.4, 0.9]]], dtype=np.float64)
        valid = np.ones_like(logits, dtype=bool)
        offsets = np.array([[[10.0]]], dtype=np.float64)

        self.assertTrue(shared_offset_preserves_ranking(logits, offsets, valid))

        per_edge = np.array([[[0.0, 20.0, 0.0]]], dtype=np.float64)
        self.assertFalse(shared_offset_preserves_ranking(logits, per_edge, valid))

    def test_gradient_summary_reports_conflict(self):
        result = gradient_cosine_summary(
            {
                "node": [torch.tensor([1.0, 0.0])],
                "link": [torch.tensor([-1.0, 0.0])],
                "task": [torch.tensor([0.0, 2.0])],
            }
        )

        self.assertAlmostEqual(result["pairs"]["link__node"]["cosine"], -1.0)
        self.assertTrue(result["pairs"]["link__node"]["conflict"])
        self.assertAlmostEqual(result["norms"]["task"], 2.0)

    def test_gate_aware_key_is_feasible_first(self):
        feasible = {
            "validation_link_f1_delta": -0.01,
            "node_x_mae_ratio": 1.20,
            "throughput_mae_delta": -0.1,
            "rb_occupancy_mae_delta": -0.1,
            "task_delay_mae_delta": -0.1,
            "validation_state_nll": 10.0,
        }
        lower_loss_but_failed = dict(feasible, node_x_mae_ratio=1.30, validation_state_nll=1.0)

        self.assertLess(
            gate_aware_checkpoint_key(feasible),
            gate_aware_checkpoint_key(lower_loss_but_failed),
        )


if __name__ == "__main__":
    unittest.main()
