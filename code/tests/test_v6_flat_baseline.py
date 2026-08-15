import unittest
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.v6_flat_baseline import V6FlatBaseline, V6FlatBaselineConfig


class V6FlatBaselineTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        config = V6FlatBaselineConfig(
            input_dim=32,
            node_dim=7,
            task_dim=9,
            num_nodes=5,
            num_edges=11,
            horizon=3,
            hidden_dim=16,
        )
        model = V6FlatBaseline(config)
        outputs = model(torch.randn(4, config.input_dim))

        self.assertEqual(outputs["node"].shape, (4, 3, 5, 7))
        self.assertEqual(outputs["link_activity_logit"].shape, (4, 3, 11, 1))
        self.assertEqual(outputs["link_rate"].shape, (4, 3, 11, 1))
        self.assertEqual(outputs["task"].shape, (4, 3, 9))


if __name__ == "__main__":
    unittest.main()
