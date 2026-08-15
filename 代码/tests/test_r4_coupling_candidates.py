from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_r3_world_model import make_batch


class R4CouplingCandidateTests(unittest.TestCase):
    def test_relation_constrained_attention_reads_only_mapped_source(self):
        from pi_jwm.r4_world_model import relation_constrained_attention

        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        source = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [100.0, 100.0]]])
        mapping = torch.tensor([[0, 1]])
        changed = source.clone()
        changed[:, 2] = -1000.0
        modules = (nn.Identity(), nn.Identity(), nn.Identity())
        left, _ = relation_constrained_attention(query, source, mapping, *modules)
        right, _ = relation_constrained_attention(query, changed, mapping, *modules)
        torch.testing.assert_close(left, right)

    def test_cross_attention_candidate_executes_and_uses_explicit_relations(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        torch.manual_seed(61)
        model = build_r4_world_model(
            make_single_module_config(
                "coupling",
                "relation_constrained_cross_attention_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )
        original = make_batch(horizon=1)
        changed = copy.deepcopy(original)
        changed.static["cip_agent_node_index"] = torch.tensor([[2, 0, 1]])
        changed.static["cep_information_to_physical_edge_index"] = torch.tensor(
            [[3, 2, 1, 0]]
        )
        changed.static["cfl_information_edge_index"] = torch.tensor([[3, 1]])

        left = model(original, rollout_steps=1).predicted_belief.joint_latent
        right = model(changed, rollout_steps=1).predicted_belief.joint_latent
        self.assertTrue(torch.isfinite(left).all())
        self.assertGreater(torch.max(torch.abs(left - right)).item(), 0.0)

    def test_cross_attention_candidate_rejects_active_invalid_mapping(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        model = build_r4_world_model(
            make_single_module_config(
                "coupling",
                "relation_constrained_cross_attention_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )
        batch = make_batch(horizon=1)
        batch.static["cip_agent_node_index"][0, 0] = 99
        with self.assertRaisesRegex(ValueError, "mapping"):
            model(batch, rollout_steps=1)


if __name__ == "__main__":
    unittest.main()
