from __future__ import annotations

import copy
import unittest

import torch

from test_formal_dual_graph_world_model_v1 import fake_formal_batch


class FormalDirectedDynamicWorldModelV2Tests(unittest.TestCase):
    def test_information_agent_history_does_not_read_physical_node_state(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(17)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=False,
            )
        ).eval()
        baseline = fake_formal_batch()
        changed_physical = copy.deepcopy(baseline)
        changed_physical["history"]["node_state"].add_(10000.0)

        with torch.no_grad():
            agent_baseline = model.encode_information_agent_history(
                baseline["history"], baseline["static"]
            )
            agent_changed = model.encode_information_agent_history(
                changed_physical["history"], changed_physical["static"]
            )

        torch.testing.assert_close(agent_baseline, agent_changed)

    def test_information_agent_history_changes_with_flow_state_and_direction(self):
        from pi_jwm.formal_dual_graph_world_model_v2 import (
            FormalDirectedDynamicWorldModelConfig,
            FormalDirectedDynamicWorldModelV2,
        )

        torch.manual_seed(19)
        model = FormalDirectedDynamicWorldModelV2(
            FormalDirectedDynamicWorldModelConfig(
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_cross_coupling=False,
            )
        ).eval()
        baseline = fake_formal_batch()
        changed_flow = copy.deepcopy(baseline)
        changed_flow["history"]["flow_state"][:, :, 0].add_(3.0)
        reversed_flow = copy.deepcopy(baseline)
        reversed_flow["static"]["flow_endpoint_index"][:, 0] = torch.tensor([1, 0])

        with torch.no_grad():
            agent_baseline = model.encode_information_agent_history(
                baseline["history"], baseline["static"]
            )
            agent_changed_flow = model.encode_information_agent_history(
                changed_flow["history"], changed_flow["static"]
            )
            agent_reversed_flow = model.encode_information_agent_history(
                reversed_flow["history"], reversed_flow["static"]
            )

        self.assertFalse(torch.allclose(agent_baseline, agent_changed_flow))
        self.assertFalse(torch.allclose(agent_baseline, agent_reversed_flow))


if __name__ == "__main__":
    unittest.main()
