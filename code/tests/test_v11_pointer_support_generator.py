import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11PointerSupportGeneratorTest(unittest.TestCase):
    def test_pointer_edge_scorer_forward_shape(self):
        from compare_v11_pointer_support_generator import PointerEdgeScorer

        model = PointerEdgeScorer(feature_dim=4, hidden_dim=8, dropout=0.0)
        features = torch.randn(3, 5, 4)

        logits = model(features)

        self.assertEqual(tuple(logits.shape), (3, 5))


if __name__ == '__main__':
    unittest.main()
