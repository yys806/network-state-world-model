import unittest

import numpy as np

from run_world_model_v4_dual_graph_rollout import build_physical_edge_history


class DualGraphFeatureTests(unittest.TestCase):
    def test_build_physical_edge_history_uses_endpoint_geometry(self):
        # Shape: samples=1, history=2, nodes=3, features=[x,y,z,speed,acc,cpu,storage].
        x_node = np.zeros((1, 2, 3, 7), dtype=np.float32)
        x_node[0, 0, 0, :4] = [0.0, 0.0, 0.0, 1.0]
        x_node[0, 0, 1, :4] = [3.0, 4.0, 0.0, 2.0]
        x_node[0, 1, 0, :4] = [1.0, 0.0, 0.0, 1.5]
        x_node[0, 1, 1, :4] = [1.0, 0.0, 12.0, 3.5]
        src_idx = np.array([0, 2], dtype=np.int32)
        dst_idx = np.array([1, -1], dtype=np.int32)
        valid = np.array([1, 0], dtype=np.int32)

        features = build_physical_edge_history(x_node, src_idx, dst_idx, valid)

        self.assertEqual(features.shape, (1, 2, 2, 8))
        np.testing.assert_allclose(features[0, 0, 0, :5], [3.0, 4.0, 0.0, 5.0, 1.0])
        np.testing.assert_allclose(features[0, 1, 0, :5], [0.0, 0.0, 12.0, 12.0, 2.0])
        np.testing.assert_allclose(features[0, :, 1, :], 0.0)


if __name__ == "__main__":
    unittest.main()
