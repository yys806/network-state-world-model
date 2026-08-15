from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_r3_world_model import make_batch


def dag_batch(*, reverse: bool = False, cycle: bool = False):
    batch = make_batch(horizon=1)
    if cycle:
        endpoints = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    elif reverse:
        endpoints = torch.tensor([[[1, 0]]], dtype=torch.long)
    else:
        endpoints = torch.tensor([[[0, 1]]], dtype=torch.long)
    edge_count = endpoints.shape[1]
    batch.static["dag_edge_index"] = endpoints
    batch.static["dag_edge_valid"] = torch.ones(1, edge_count, dtype=torch.bool)
    batch.history["dag_edge_present"] = torch.ones(
        1, 2, edge_count, dtype=torch.bool
    )
    batch.target["dag_edge_present"] = torch.ones(
        1, 1, edge_count, dtype=torch.bool
    )
    return batch


class R4DAGCandidateTests(unittest.TestCase):
    def _model(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        return build_r4_world_model(
            make_single_module_config(
                "dag",
                "explicit_dag_message_passing_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )

    def test_explicit_dag_messages_change_task_belief_with_edge_direction(self):
        torch.manual_seed(73)
        model = self._model()
        forward = model(dag_batch(), rollout_steps=1)
        reverse = model(dag_batch(reverse=True), rollout_steps=1)
        self.assertTrue(torch.isfinite(forward.predicted_belief.business_latent).all())
        self.assertGreater(
            torch.max(
                torch.abs(
                    forward.predicted_belief.business_latent
                    - reverse.predicted_belief.business_latent
                )
            ).item(),
            0.0,
        )

    def test_explicit_dag_rejects_active_cycle(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "acyclic"):
            model(dag_batch(cycle=True), rollout_steps=1)

    def test_explicit_dag_rejects_active_invalid_task_index(self):
        model = self._model()
        batch = dag_batch()
        batch.static["dag_edge_index"][0, 0, 1] = 99
        with self.assertRaisesRegex(ValueError, "dag_edge_index"):
            model(batch, rollout_steps=1)

    def test_explicit_dag_rollout_does_not_read_future_edge_targets(self):
        model = self._model()
        left = dag_batch()
        right = copy.deepcopy(left)
        right.target["dag_edge_present"].zero_()
        torch.testing.assert_close(
            model(left, rollout_steps=1).predicted_belief.joint_latent,
            model(right, rollout_steps=1).predicted_belief.joint_latent,
        )

    def test_source_contract_parent_child_rows_match_internal_endpoint_pairs(self):
        model = self._model()
        left = dag_batch()
        right = copy.deepcopy(left)
        right.static["dag_edge_index"] = torch.tensor(
            [[[0], [1]]], dtype=torch.long
        )
        torch.testing.assert_close(
            model(left, rollout_steps=1).predicted_belief.joint_latent,
            model(right, rollout_steps=1).predicted_belief.joint_latent,
        )


if __name__ == "__main__":
    unittest.main()
