from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FormalGraphOpsV1Tests(unittest.TestCase):
    def test_masked_index_mean_excludes_padding_and_missing_messages(self):
        from pi_jwm.formal_graph_ops_v1 import masked_index_mean

        messages = torch.tensor([[[2.0], [4.0], [100.0], [8.0]]])
        index = torch.tensor([0, 0, -1, 2])
        valid = torch.tensor([[True, True, True, False]])

        result = masked_index_mean(messages, index, output_size=3, valid_mask=valid)

        torch.testing.assert_close(result, torch.tensor([[[3.0], [0.0], [0.0]]]))

    def test_physical_message_pass_uses_real_endpoints(self):
        from pi_jwm.formal_graph_ops_v1 import physical_message_pass

        nodes = torch.tensor([[[1.0], [3.0], [9.0], [0.0]]])
        edges = torch.tensor([[[2.0], [6.0], [100.0]]])
        endpoints = torch.tensor([[0, 1], [1, 2], [-1, -1]])
        node_mask = torch.tensor([[True, True, True, False]])
        edge_mask = torch.tensor([[True, True, False]])

        node_messages, edge_messages = physical_message_pass(
            nodes, edges, endpoints, node_mask, edge_mask
        )

        torch.testing.assert_close(node_messages, torch.tensor([[[2.0], [4.0], [6.0], [0.0]]]))
        torch.testing.assert_close(edge_messages, torch.tensor([[[2.0], [6.0], [0.0]]]))

    def test_information_message_pass_has_same_endpoint_semantics(self):
        from pi_jwm.formal_graph_ops_v1 import information_message_pass

        agents = torch.tensor([[[2.0], [4.0], [8.0]]])
        flows = torch.tensor([[[10.0], [20.0]]])
        endpoints = torch.tensor([[0, 1], [1, 2]])
        agent_mask = torch.tensor([[True, True, True]])
        flow_mask = torch.tensor([[True, False]])

        agent_messages, flow_messages = information_message_pass(
            agents, flows, endpoints, agent_mask, flow_mask
        )

        torch.testing.assert_close(agent_messages, torch.tensor([[[10.0], [10.0], [0.0]]]))
        torch.testing.assert_close(flow_messages, torch.tensor([[[3.0], [0.0]]]))

    def test_dag_messages_only_travel_from_present_parent_to_child(self):
        from pi_jwm.formal_graph_ops_v1 import dag_message_pass

        tasks = torch.tensor([[[2.0], [4.0], [8.0]]])
        dag_edges = torch.tensor([[0, 1], [1, 2]])
        edge_mask = torch.tensor([[True, False]])
        task_mask = torch.tensor([[True, True, True]])

        result = dag_message_pass(tasks, dag_edges, edge_mask, task_mask)

        torch.testing.assert_close(result, torch.tensor([[[0.0], [2.0], [0.0]]]))

    def test_cip_exchanges_only_attached_agent_and_physical_node(self):
        from pi_jwm.formal_graph_ops_v1 import couple_agent_physical

        agents = torch.tensor([[[1.0], [2.0], [50.0]]])
        nodes = torch.tensor([[[10.0], [20.0], [30.0]]])
        attachment = torch.tensor([1, 2, -1])
        agent_mask = torch.tensor([[True, True, False]])
        node_mask = torch.tensor([[True, True, True]])

        agent_from_node, node_from_agent = couple_agent_physical(
            agents, nodes, attachment, agent_mask, node_mask
        )

        torch.testing.assert_close(agent_from_node, torch.tensor([[[20.0], [30.0], [0.0]]]))
        torch.testing.assert_close(node_from_agent, torch.tensor([[[0.0], [1.0], [2.0]]]))

    def test_cfe_supports_one_flow_carried_by_multiple_physical_edges(self):
        from pi_jwm.formal_graph_ops_v1 import couple_flow_bearer

        flows = torch.tensor([[[2.0], [6.0]]])
        edges = torch.tensor([[[10.0], [20.0], [30.0]]])
        bearer = torch.tensor([[[True, True, False], [False, True, True]]])
        flow_mask = torch.tensor([[True, True]])
        edge_mask = torch.tensor([[True, True, False]])

        flow_from_edge, edge_from_flow = couple_flow_bearer(
            flows, edges, bearer, flow_mask, edge_mask
        )

        torch.testing.assert_close(flow_from_edge, torch.tensor([[[15.0], [20.0]]]))
        torch.testing.assert_close(edge_from_flow, torch.tensor([[[2.0], [4.0], [0.0]]]))

    def test_all_empty_graph_outputs_are_finite_zeros(self):
        from pi_jwm.formal_graph_ops_v1 import (
            couple_flow_bearer,
            physical_message_pass,
        )

        nodes = torch.randn(2, 3, 4)
        edges = torch.randn(2, 2, 4)
        endpoints = torch.full((2, 2), -1, dtype=torch.long)
        node_mask = torch.zeros(2, 3, dtype=torch.bool)
        edge_mask = torch.zeros(2, 2, dtype=torch.bool)
        node_messages, edge_messages = physical_message_pass(
            nodes, edges, endpoints, node_mask, edge_mask
        )
        flow_messages, bearer_messages = couple_flow_bearer(
            torch.randn(2, 1, 4),
            edges,
            torch.zeros(2, 1, 2, dtype=torch.bool),
            torch.zeros(2, 1, dtype=torch.bool),
            edge_mask,
        )

        for value in (node_messages, edge_messages, flow_messages, bearer_messages):
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(0, torch.count_nonzero(value))


if __name__ == "__main__":
    unittest.main()
