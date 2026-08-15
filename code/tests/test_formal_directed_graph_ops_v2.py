from __future__ import annotations

import unittest

import torch


class FormalDirectedGraphOpsV2Tests(unittest.TestCase):
    def test_directed_messages_separate_incoming_and_outgoing_endpoints(self):
        from pi_jwm.formal_directed_graph_ops_v2 import directed_relation_messages

        entities = torch.tensor([[[1.0], [2.0], [3.0]]])
        relations = torch.tensor([[[10.0], [20.0]]])
        endpoints = torch.tensor([[0, 1], [0, 2]])
        entity_weight = torch.ones(1, 3)
        relation_weight = torch.ones(1, 2)

        incoming, outgoing, relation_context = directed_relation_messages(
            entities,
            relations,
            endpoints,
            entity_weight,
            relation_weight,
        )

        torch.testing.assert_close(incoming, torch.tensor([[[0.0], [10.0], [20.0]]]))
        torch.testing.assert_close(outgoing, torch.tensor([[[15.0], [0.0], [0.0]]]))
        torch.testing.assert_close(
            relation_context,
            torch.tensor([[[1.0, 2.0], [1.0, 3.0]]]),
        )

    def test_reversing_endpoint_direction_changes_messages(self):
        from pi_jwm.formal_directed_graph_ops_v2 import directed_relation_messages

        entities = torch.tensor([[[1.0], [2.0]]])
        relations = torch.tensor([[[5.0]]])
        weights = torch.ones(1, 2)
        relation_weight = torch.ones(1, 1)

        forward = directed_relation_messages(
            entities, relations, torch.tensor([[0, 1]]), weights, relation_weight
        )
        reverse = directed_relation_messages(
            entities, relations, torch.tensor([[1, 0]]), weights, relation_weight
        )

        self.assertFalse(torch.equal(forward[0], reverse[0]))
        self.assertFalse(torch.equal(forward[1], reverse[1]))
        self.assertFalse(torch.equal(forward[2], reverse[2]))

    def test_relation_order_permutation_preserves_entity_aggregation(self):
        from pi_jwm.formal_directed_graph_ops_v2 import directed_relation_messages

        entities = torch.tensor([[[1.0], [2.0], [3.0]]])
        relations = torch.tensor([[[10.0], [20.0]]])
        endpoints = torch.tensor([[0, 1], [2, 1]])
        entity_weight = torch.ones(1, 3)
        relation_weight = torch.tensor([[1.0, 0.5]])

        original = directed_relation_messages(
            entities, relations, endpoints, entity_weight, relation_weight
        )
        permuted = directed_relation_messages(
            entities,
            relations[:, [1, 0]],
            endpoints[[1, 0]],
            entity_weight,
            relation_weight[:, [1, 0]],
        )

        torch.testing.assert_close(original[0], permuted[0])
        torch.testing.assert_close(original[1], permuted[1])
        torch.testing.assert_close(original[2][:, [1, 0]], permuted[2])

    def test_soft_relation_weights_attenuate_messages_without_being_normalized_away(self):
        from pi_jwm.formal_directed_graph_ops_v2 import directed_relation_messages

        full, _, _ = directed_relation_messages(
            torch.ones(1, 2, 1),
            torch.tensor([[[8.0]]]),
            torch.tensor([[0, 1]]),
            torch.ones(1, 2),
            torch.tensor([[1.0]]),
        )
        soft, _, _ = directed_relation_messages(
            torch.ones(1, 2, 1),
            torch.tensor([[[8.0]]]),
            torch.tensor([[0, 1]]),
            torch.ones(1, 2),
            torch.tensor([[0.25]]),
        )

        self.assertAlmostEqual(8.0, float(full[0, 1, 0]))
        self.assertAlmostEqual(2.0, float(soft[0, 1, 0]))

    def test_multiple_soft_relations_use_structural_count_normalization(self):
        from pi_jwm.formal_directed_graph_ops_v2 import directed_relation_messages

        incoming, _, _ = directed_relation_messages(
            torch.ones(1, 3, 1),
            torch.tensor([[[2.0], [8.0]]]),
            torch.tensor([[0, 2], [1, 2]]),
            torch.ones(1, 3),
            torch.tensor([[1.0, 0.5]]),
        )

        self.assertAlmostEqual(3.0, float(incoming[0, 2, 0]))

    def test_empty_and_padded_relations_return_finite_zeros(self):
        from pi_jwm.formal_directed_graph_ops_v2 import directed_relation_messages

        incoming, outgoing, relation_context = directed_relation_messages(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            torch.tensor([[-1, -1], [0, 9]]),
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 2),
        )

        for value in (incoming, outgoing, relation_context):
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(0, int(torch.count_nonzero(value)))

    def test_direct_bearer_candidates_require_matching_direction(self):
        from pi_jwm.formal_directed_graph_ops_v2 import direct_bearer_candidates

        candidates = direct_bearer_candidates(
            torch.tensor([[[0, 1], [1, 0], [-1, -1]]]),
            torch.tensor([[[0, 1], [1, 0], [0, 2]]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )

        expected = torch.tensor(
            [[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]]
        )
        torch.testing.assert_close(candidates, expected)


if __name__ == "__main__":
    unittest.main()
