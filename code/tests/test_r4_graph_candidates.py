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


def graph_batch(*, horizon: int = 1):
    batch = make_batch(horizon=horizon)
    batch.static["information_edge_kind_index"] = torch.tensor(
        [[0, 1, 2, 3]], dtype=torch.long
    )
    return batch


def permute_edges(batch, permutation: torch.Tensor):
    changed = copy.deepcopy(batch)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    for namespace in (changed.history, changed.target):
        for prefix in ("physical_edge", "information_edge"):
            for suffix in ("state", "feature_mask", "present"):
                key = f"{prefix}_{suffix}"
                namespace[key] = namespace[key].index_select(2, permutation)
    for key in ("physical_edge_endpoint_index", "information_edge_endpoint_index"):
        changed.static[key] = changed.static[key].index_select(1, permutation)
    changed.static["information_edge_kind_index"] = changed.static[
        "information_edge_kind_index"
    ].index_select(1, permutation)
    old_cep = batch.static["cep_information_to_physical_edge_index"]
    changed.static["cep_information_to_physical_edge_index"] = inverse[
        old_cep.index_select(1, permutation)
    ]
    changed.static["cfl_information_edge_index"] = inverse[
        batch.static["cfl_information_edge_index"]
    ]
    return changed


class R4GraphCandidateTests(unittest.TestCase):
    def test_graph_candidates_execute_finite_directed_rollout(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        for name in ("rgcn_v1", "edge_conditioned_relation_mpnn_v1"):
            with self.subTest(name=name):
                model = build_r4_world_model(
                    make_single_module_config(
                        "graph_encoder",
                        name,
                        hidden_dim=8,
                        history_steps=2,
                    )
                )
                output = model(graph_batch(horizon=5), rollout_steps=5)
                self.assertEqual(
                    (1, 5, 3, 8),
                    tuple(output.predicted_belief.information_latent.shape),
                )
                self.assertTrue(torch.isfinite(output.predicted_belief.joint_latent).all())

    def test_graph_candidates_are_invariant_to_consistent_edge_order(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        for name in ("rgcn_v1", "edge_conditioned_relation_mpnn_v1"):
            with self.subTest(name=name):
                torch.manual_seed(59)
                model = build_r4_world_model(
                    make_single_module_config(
                        "graph_encoder",
                        name,
                        hidden_dim=8,
                        history_steps=2,
                    )
                )
                original = graph_batch(horizon=1)
                changed = permute_edges(original, permutation)
                left = model(original, rollout_steps=1).predicted_belief
                right = model(changed, rollout_steps=1).predicted_belief
                torch.testing.assert_close(left.physical_latent, right.physical_latent)
                torch.testing.assert_close(left.information_latent, right.information_latent)
                torch.testing.assert_close(left.joint_latent, right.joint_latent)

    def test_rgcn_rejects_unknown_information_relation_type(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        model = build_r4_world_model(
            make_single_module_config(
                "graph_encoder",
                "rgcn_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )
        batch = graph_batch()
        batch.static["information_edge_kind_index"][0, 0] = 10
        with self.assertRaisesRegex(ValueError, "information_edge_kind_index"):
            model(batch, rollout_steps=1)

    def test_graph_candidates_ignore_padded_inactive_relations_but_reject_active_padding(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        for name in ("rgcn_v1", "edge_conditioned_relation_mpnn_v1"):
            with self.subTest(name=name):
                model = build_r4_world_model(
                    make_single_module_config(
                        "graph_encoder",
                        name,
                        hidden_dim=8,
                        history_steps=2,
                    )
                )
                batch = graph_batch()
                batch.static["physical_edge_endpoint_index"][0, -1] = -1
                batch.history["physical_edge_present"][0, :, -1] = False
                batch.static["information_edge_endpoint_index"][0, -1] = -1
                batch.static["information_edge_kind_index"][0, -1] = -1
                batch.history["information_edge_present"][0, :, -1] = False
                output = model(batch, rollout_steps=1)
                self.assertTrue(torch.isfinite(output.predicted_belief.joint_latent).all())

                batch.history["physical_edge_present"][0, -1, -1] = True
                with self.assertRaisesRegex(ValueError, "invalid endpoint"):
                    model(batch, rollout_steps=1)


if __name__ == "__main__":
    unittest.main()
