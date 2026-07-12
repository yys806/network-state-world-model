import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11GnnGreedySchedulerTest(unittest.TestCase):
    def test_group_rows_by_sample_step_orders_edges(self):
        from compare_v11_gnn_greedy_scheduler import group_rows_by_sample_step

        features = np.array([[10.0], [12.0], [20.0], [22.0]], dtype=np.float32)
        coords = np.array([[0, 1, 1], [0, 1, 0], [0, 2, 1], [0, 2, 0]], dtype=np.int64)
        target = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        rank = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        grouped = group_rows_by_sample_step(features, coords, target, rank)

        self.assertEqual(grouped.features.shape, (2, 2, 1))
        self.assertTrue(np.allclose(grouped.features[:, :, 0], np.array([[12.0, 10.0], [22.0, 20.0]], dtype=np.float32)))
        self.assertEqual(grouped.edge_count, 2)

    def test_edge_adjacency_falls_back_to_identity(self):
        from compare_v11_gnn_greedy_scheduler import build_edge_adjacency

        adjacency = build_edge_adjacency(3)

        self.assertTrue(torch.allclose(adjacency, torch.eye(3)))

    def test_edge_graph_scorer_forward_shape(self):
        from compare_v11_gnn_greedy_scheduler import EdgeGraphScorer

        model = EdgeGraphScorer(feature_dim=4, hidden_dim=8, layers=2, dropout=0.0)
        features = torch.randn(3, 5, 4)
        adjacency = torch.eye(5)

        logits = model(features, adjacency)

        self.assertEqual(tuple(logits.shape), (3, 5))


if __name__ == '__main__':
    unittest.main()
