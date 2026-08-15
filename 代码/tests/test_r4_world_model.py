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


class R4WorldModelFactoryTests(unittest.TestCase):
    def test_symlog_and_simnorm_field_candidates_execute_same_public_contract(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        for name in ("symlog_masked_mlp_v1", "simnorm_masked_mlp_v1"):
            with self.subTest(name=name):
                model = build_r4_world_model(
                    make_single_module_config(
                        "field_encoder",
                        name,
                        hidden_dim=8,
                        history_steps=2,
                    )
                )
                output = model(make_batch(horizon=5), rollout_steps=5)
                self.assertEqual(
                    (1, 5, 8),
                    tuple(output.predicted_belief.joint_latent.shape),
                )
                self.assertTrue(torch.isfinite(output.predicted_belief.joint_latent).all())

    def test_field_candidates_ignore_values_hidden_by_feature_mask(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        for name in ("symlog_masked_mlp_v1", "simnorm_masked_mlp_v1"):
            with self.subTest(name=name):
                torch.manual_seed(37)
                model = build_r4_world_model(
                    make_single_module_config(
                        "field_encoder",
                        name,
                        hidden_dim=8,
                        history_steps=2,
                    )
                )
                left = make_batch(horizon=1)
                right = copy.deepcopy(left)
                left.history["physical_node_feature_mask"][0, 0, 0, 0] = False
                right.history["physical_node_feature_mask"][0, 0, 0, 0] = False
                left.history["physical_node_state"][0, 0, 0, 0] = -1.0e9
                right.history["physical_node_state"][0, 0, 0, 0] = 1.0e9
                torch.testing.assert_close(
                    model(left, rollout_steps=1).predicted_belief.joint_latent,
                    model(right, rollout_steps=1).predicted_belief.joint_latent,
                )

    def test_simnorm_normalizes_each_feature_group(self):
        from pi_jwm.r4_world_model import simnorm

        value = torch.tensor([[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0]])
        normalized = simnorm(value, group_size=4)
        torch.testing.assert_close(
            normalized.reshape(1, 2, 4).sum(dim=-1),
            torch.ones(1, 2),
        )

    def test_reference_factory_preserves_r3_public_rollout_contract(self):
        from pi_jwm.r4_module_registry import reference_r4_config
        from pi_jwm.r4_world_model import build_r4_world_model

        torch.manual_seed(41)
        model = build_r4_world_model(
            reference_r4_config(hidden_dim=8, history_steps=2)
        )
        output = model(make_batch(horizon=5), rollout_steps=5)

        self.assertEqual((1, 5, 3, 9), tuple(output.predicted_explicit["physical_node_state"].shape))
        self.assertEqual((1, 5, 4, 18), tuple(output.predicted_explicit["information_edge_state"].shape))
        self.assertEqual((1, 5, 8), tuple(output.predicted_belief.joint_latent.shape))
        self.assertEqual(model.config.component_names(), model.component_registry())

    def test_no_coupling_candidate_is_executable_and_ignores_cross_relations(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        torch.manual_seed(43)
        model = build_r4_world_model(
            make_single_module_config(
                "coupling",
                "no_cross_graph_coupling_v1",
                hidden_dim=8,
                history_steps=2,
            )
        )
        left = make_batch(horizon=1)
        right = copy.deepcopy(left)
        right.static["cip_agent_node_index"] = torch.tensor([[2, 0, 1]])
        right.static["cep_information_to_physical_edge_index"] = torch.tensor([[3, 2, 1, 0]])
        right.static["cfl_information_edge_index"] = torch.tensor([[3, 1]])

        left_output = model(left, rollout_steps=1)
        right_output = model(right, rollout_steps=1)
        torch.testing.assert_close(
            left_output.predicted_belief.joint_latent,
            right_output.predicted_belief.joint_latent,
        )

    def test_factory_rejects_known_but_unimplemented_candidate(self):
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        config = make_single_module_config("dynamics", "transformer_dynamics_v1")
        with self.assertRaisesRegex(ValueError, "not executable"):
            build_r4_world_model(config)

    def test_factory_rejects_multi_module_configuration(self):
        from dataclasses import replace

        from pi_jwm.r4_module_registry import reference_r4_config
        from pi_jwm.r4_world_model import build_r4_world_model

        config = replace(
            reference_r4_config(),
            coupling="no_cross_graph_coupling_v1",
            dynamics="graph_rssm_v1",
        )
        with self.assertRaisesRegex(ValueError, "one module family"):
            build_r4_world_model(config)


if __name__ == "__main__":
    unittest.main()
