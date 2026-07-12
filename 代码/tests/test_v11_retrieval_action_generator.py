import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RetrievalActionGeneratorTest(unittest.TestCase):
    def test_standardize_features_uses_train_statistics(self):
        from compare_v11_retrieval_action_generator import standardize_features

        train = np.array([[1.0, 10.0], [3.0, 10.0]], dtype=np.float32)
        query = np.array([[5.0, 10.0]], dtype=np.float32)

        train_z, query_z, stats = standardize_features(train, query)

        self.assertTrue(np.allclose(train_z[:, 0], np.array([-1.0, 1.0], dtype=np.float32)))
        self.assertAlmostEqual(float(query_z[0, 0]), 3.0, places=5)
        self.assertTrue(np.all(np.isfinite(query_z)))
        self.assertGreater(float(stats["std"][1]), 0.0)

    def test_retrieve_knn_indices_orders_by_distance(self):
        from compare_v11_retrieval_action_generator import retrieve_knn_indices

        prototypes = np.array([[0.0], [2.0], [10.0]], dtype=np.float32)
        query = np.array([[1.9]], dtype=np.float32)

        nearest, distance = retrieve_knn_indices(query, prototypes, k=2)

        self.assertEqual(nearest.tolist(), [[1, 0]])
        self.assertLess(float(distance[0, 0]), float(distance[0, 1]))

    def test_aggregate_retrieved_actions_supports_inverse_distance_weights(self):
        from compare_v11_retrieval_action_generator import aggregate_retrieved_actions

        actions = np.zeros((3, 2, 1, 6), dtype=np.float32)
        actions[:, 1, 0, 2] = np.array([0.0, 10.0, 30.0], dtype=np.float32)
        nearest = np.array([[0, 1]], dtype=np.int64)
        distance = np.array([[1.0, 3.0]], dtype=np.float32)

        aggregated = aggregate_retrieved_actions(actions, nearest, distance, mode="inverse_distance")

        self.assertAlmostEqual(float(aggregated[0, 1, 0, 2]), 2.5, places=5)

    def test_apply_retrieved_action_preserves_step0_and_replaces_selected_dims(self):
        from compare_v11_retrieval_action_generator import apply_retrieved_action

        baseline = np.zeros((1, 3, 2, 6), dtype=np.float32)
        baseline[0, 0, :, 2] = 3.0
        baseline[0, 1, :, 2] = 4.0
        retrieved = np.ones_like(baseline) * 9.0

        repaired = apply_retrieved_action(
            baseline,
            retrieved,
            alpha=1.0,
            replacement_mode="rb_cpu",
            preserve_step0=True,
            step_total_cap_scale=0.0,
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], 3.0))
        self.assertTrue(np.allclose(repaired[0, 1, :, 2], 9.0))
        self.assertTrue(np.allclose(repaired[0, 1, :, 4], 9.0))
        self.assertTrue(np.allclose(repaired[0, 1, :, 0], 0.0))

    def test_apply_retrieved_action_step_total_cap_limits_rb_total(self):
        from compare_v11_retrieval_action_generator import apply_retrieved_action

        baseline = np.zeros((1, 2, 2, 6), dtype=np.float32)
        baseline[0, 1, :, 2] = np.array([4.0, 6.0], dtype=np.float32)
        retrieved = baseline.copy()
        retrieved[0, 1, :, 2] = np.array([40.0, 60.0], dtype=np.float32)

        repaired = apply_retrieved_action(
            baseline,
            retrieved,
            alpha=1.0,
            replacement_mode="rb_only",
            preserve_step0=True,
            step_total_cap_scale=1.5,
        )

        self.assertAlmostEqual(float(np.sum(repaired[0, 1, :, 2])), 15.0, places=5)


if __name__ == "__main__":
    unittest.main()
