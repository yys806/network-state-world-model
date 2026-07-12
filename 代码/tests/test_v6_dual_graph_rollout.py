import unittest
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V6DualGraphRolloutTest(unittest.TestCase):
    def test_forward_returns_future_state_predictions_with_expected_shapes(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch, V6DualGraphConfig, V6DualGraphRollout

        batch_size = 2
        history = 3
        horizon = 4
        num_nodes = 5
        num_edges = 7
        node_dim = 6
        physical_edge_dim = 4
        info_edge_dim = 5
        action_dim = 3
        task_dim = 2

        config = V6DualGraphConfig(
            node_dim=node_dim,
            physical_edge_dim=physical_edge_dim,
            info_edge_dim=info_edge_dim,
            action_dim=action_dim,
            task_dim=task_dim,
            hidden_dim=16,
            horizon=horizon,
        )
        model = V6DualGraphRollout(config)

        batch = V6DualGraphBatch(
            node_history=torch.randn(batch_size, history, num_nodes, node_dim),
            physical_edge_history=torch.randn(batch_size, history, num_edges, physical_edge_dim),
            info_edge_history=torch.randn(batch_size, history, num_edges, info_edge_dim),
            action_history=torch.randn(batch_size, history, num_edges, action_dim),
            future_actions=torch.randn(batch_size, horizon, num_edges, action_dim),
            task_history=torch.randn(batch_size, history, task_dim),
        )

        outputs = model(batch)

        self.assertEqual(outputs["node"].shape, (batch_size, horizon, num_nodes, node_dim))
        self.assertEqual(outputs["link_activity_logit"].shape, (batch_size, horizon, num_edges, 1))
        self.assertEqual(outputs["link_rate"].shape, (batch_size, horizon, num_edges, 1))
        self.assertEqual(outputs["task"].shape, (batch_size, horizon, task_dim))

    def test_ablation_modes_share_output_interface(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch, V6DualGraphConfig, V6DualGraphRollout

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 6),
            physical_edge_history=torch.randn(2, 3, 5, 8),
            info_edge_history=torch.randn(2, 3, 5, 5),
            action_history=torch.randn(2, 3, 5, 3),
            future_actions=torch.randn(2, 2, 5, 3),
            task_history=torch.randn(2, 3, 7),
        )

        for mode in ("dual", "physical_only", "information_only"):
            with self.subTest(mode=mode):
                config = V6DualGraphConfig(
                    node_dim=6,
                    physical_edge_dim=8,
                    info_edge_dim=5,
                    action_dim=3,
                    task_dim=7,
                    hidden_dim=16,
                    horizon=2,
                    graph_mode=mode,
                )
                outputs = V6DualGraphRollout(config)(batch)
                self.assertEqual(outputs["node"].shape, (2, 2, 4, 6))
                self.assertEqual(outputs["link_activity_logit"].shape, (2, 2, 5, 1))

    def test_build_physical_edge_history_aligns_edges_with_endpoint_geometry(self):
        from pi_jwm.v6_data import build_physical_edge_history

        x_node = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0, 2.0], [3.0, 4.0, 1.0, 5.0]],
                    [[1.0, 0.0, 0.0, 3.0], [4.0, 4.0, 2.0, 7.0]],
                ]
            ]
        )
        edge_src_idx = torch.tensor([0])
        edge_dst_idx = torch.tensor([1])
        valid_edge_node = torch.tensor([1])

        physical = build_physical_edge_history(x_node, edge_src_idx, edge_dst_idx, valid_edge_node)

        self.assertEqual(physical.shape, (1, 2, 1, 8))
        self.assertAlmostEqual(float(physical[0, 0, 0, 0]), 3.0)
        self.assertAlmostEqual(float(physical[0, 0, 0, 1]), 4.0)
        self.assertAlmostEqual(float(physical[0, 0, 0, 2]), 1.0)
        self.assertAlmostEqual(float(physical[0, 0, 0, 3]), (3.0**2 + 4.0**2 + 1.0**2) ** 0.5)
        self.assertAlmostEqual(float(physical[0, 0, 0, 4]), 3.0)

    def test_v6_dataset_adapter_returns_model_batch_and_targets(self):
        from pi_jwm.v6_data import V6WorldModelDataset, make_normalization_stats

        arrays = {
            "x_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "x_link": torch.randn(3, 2, 5, 4).numpy().astype("float32"),
            "x_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_a_hist": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "edge_a_future": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "y_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "y_link_rate": torch.randn(3, 2, 5).numpy().astype("float32"),
            "y_link_active": torch.randint(0, 2, (3, 2, 5)).numpy().astype("float32"),
            "y_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_src_idx": torch.tensor([0, 0, 1, 2, 3]).numpy().astype("int32"),
            "edge_dst_idx": torch.tensor([1, 2, 2, 3, 0]).numpy().astype("int32"),
            "valid_edge_node": torch.ones(5).numpy().astype("int32"),
        }
        stats = make_normalization_stats(arrays, [0, 1])
        dataset = V6WorldModelDataset(arrays, [0, 1], stats)

        batch, target = dataset[0]

        self.assertEqual(batch.node_history.shape, (2, 4, 6))
        self.assertEqual(batch.physical_edge_history.shape, (2, 5, 8))
        self.assertEqual(batch.info_edge_history.shape, (2, 5, 4))
        self.assertEqual(batch.action_history.shape, (2, 5, 3))
        self.assertEqual(batch.future_actions.shape, (2, 5, 3))
        self.assertEqual(batch.task_history.shape, (2, 7))
        self.assertEqual(target["node"].shape, (2, 4, 6))
        self.assertEqual(target["link_activity"].shape, (2, 5, 1))
        self.assertEqual(target["link_rate"].shape, (2, 5, 1))
        self.assertEqual(target["task"].shape, (2, 7))

    def test_v6_dataset_adapter_returns_optional_rate_teacher_targets(self):
        from pi_jwm.v6_data import V6WorldModelDataset, make_normalization_stats, normalize

        arrays = {
            "x_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "x_link": torch.randn(3, 2, 5, 4).numpy().astype("float32"),
            "x_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_a_hist": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "edge_a_future": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "y_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "y_link_rate": torch.randn(3, 2, 5).numpy().astype("float32"),
            "y_link_rate_teacher": torch.randn(3, 2, 5).numpy().astype("float32"),
            "y_link_rate_teacher_mask": torch.randint(0, 2, (3, 2, 5)).numpy().astype("float32"),
            "y_link_active": torch.randint(0, 2, (3, 2, 5)).numpy().astype("float32"),
            "y_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_src_idx": torch.tensor([0, 0, 1, 2, 3]).numpy().astype("int32"),
            "edge_dst_idx": torch.tensor([1, 2, 2, 3, 0]).numpy().astype("int32"),
            "valid_edge_node": torch.ones(5).numpy().astype("int32"),
        }
        stats = make_normalization_stats(arrays, [0, 1])
        dataset = V6WorldModelDataset(arrays, [0, 1], stats)

        _, target = dataset[0]

        expected_teacher = normalize(arrays["y_link_rate_teacher"][0, ..., None], stats["y_link_rate"])[0]
        self.assertEqual(target["link_rate_teacher"].shape, (2, 5, 1))
        self.assertEqual(target["link_rate_teacher_mask"].shape, (2, 5, 1))
        torch.testing.assert_close(target["link_rate_teacher"], torch.from_numpy(expected_teacher))
        torch.testing.assert_close(
            target["link_rate_teacher_mask"],
            torch.from_numpy(arrays["y_link_rate_teacher_mask"][0, ..., None]),
        )

    def test_log1p_rate_target_transform_is_used_for_dataset_targets(self):
        from pi_jwm.v6_data import V6WorldModelDataset, make_normalization_stats, normalize

        arrays = {
            "x_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "x_link": torch.randn(3, 2, 5, 4).numpy().astype("float32"),
            "x_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_a_hist": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "edge_a_future": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "y_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "y_link_rate": np.array(
                [
                    [[0.0, 1.0, 3.0, 7.0, 15.0], [2.0, 4.0, 8.0, 16.0, 32.0]],
                    [[5.0, 10.0, 20.0, 40.0, 80.0], [6.0, 12.0, 24.0, 48.0, 96.0]],
                    [[1.0, 2.0, 4.0, 8.0, 16.0], [3.0, 6.0, 12.0, 24.0, 48.0]],
                ],
                dtype=np.float32,
            ),
            "y_link_active": torch.randint(0, 2, (3, 2, 5)).numpy().astype("float32"),
            "y_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_src_idx": torch.tensor([0, 0, 1, 2, 3]).numpy().astype("int32"),
            "edge_dst_idx": torch.tensor([1, 2, 2, 3, 0]).numpy().astype("int32"),
            "valid_edge_node": torch.ones(5).numpy().astype("int32"),
        }
        stats = make_normalization_stats(arrays, [0, 1], rate_target_transform="log1p_raw")
        dataset = V6WorldModelDataset(arrays, [0, 1], stats, rate_target_transform="log1p_raw")

        _, target = dataset[0]

        expected = normalize(np.log1p(arrays["y_link_rate"][0, ..., None]), stats["y_link_rate"])[0]
        torch.testing.assert_close(target["link_rate"], torch.from_numpy(expected))

    def test_log1p_rate_target_transform_round_trips_to_raw_rate(self):
        from pi_jwm.v6_data import inverse_transform_link_rate, transform_link_rate

        raw = np.array([0.0, 1.0, 9.0, 99.0], dtype=np.float32)

        transformed = transform_link_rate(raw, "log1p_raw")
        recovered = inverse_transform_link_rate(transformed, "log1p_raw")

        np.testing.assert_allclose(recovered, raw, rtol=1e-6, atol=1e-6)

    def test_log1p_inverse_transform_can_clip_before_expm1(self):
        from pi_jwm.v6_data import inverse_transform_link_rate

        log_values = np.array([0.0, np.log1p(9.0), 20.0], dtype=np.float32)

        recovered = inverse_transform_link_rate(log_values, "log1p_raw", clip_max=np.log1p(99.0))

        np.testing.assert_allclose(recovered, np.array([0.0, 9.0, 99.0], dtype=np.float32), rtol=1e-6, atol=1e-5)

    def test_rate_inverse_clip_quantile_uses_positive_rates(self):
        from run_world_model_v6_dual_graph_rollout import compute_rate_inverse_clip_max

        arrays = {
            "y_link_rate": np.array(
                [
                    [[0.0, 0.0, 10.0, 100.0]],
                    [[0.0, 0.0, 20.0, 200.0]],
                ],
                dtype=np.float32,
            )
        }
        train_idx = np.array([0, 1])

        clip_max = compute_rate_inverse_clip_max(arrays, train_idx, "log1p_raw", 0.5)

        self.assertGreater(clip_max, np.log1p(10.0))

    def test_residual_last_rate_target_subtracts_last_observed_rate(self):
        from pi_jwm.v6_data import V6WorldModelDataset, make_normalization_stats, normalize

        x_link = np.zeros((3, 2, 5, 4), dtype=np.float32)
        x_link[:, -1, :, 1] = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
            ],
            dtype=np.float32,
        )
        y_link_rate = np.array(
            [
                [[2.0, 4.0, 6.0, 8.0, 10.0], [3.0, 5.0, 7.0, 9.0, 11.0]],
                [[12.0, 23.0, 34.0, 45.0, 56.0], [13.0, 25.0, 37.0, 49.0, 61.0]],
                [[7.0, 9.0, 11.0, 13.0, 15.0], [8.0, 10.0, 12.0, 14.0, 16.0]],
            ],
            dtype=np.float32,
        )
        arrays = {
            "x_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "x_link": x_link,
            "x_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_a_hist": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "edge_a_future": torch.randn(3, 2, 5, 3).numpy().astype("float32"),
            "y_node": torch.randn(3, 2, 4, 6).numpy().astype("float32"),
            "y_link_rate": y_link_rate,
            "y_link_active": torch.randint(0, 2, (3, 2, 5)).numpy().astype("float32"),
            "y_task": torch.randn(3, 2, 7).numpy().astype("float32"),
            "edge_src_idx": torch.tensor([0, 0, 1, 2, 3]).numpy().astype("int32"),
            "edge_dst_idx": torch.tensor([1, 2, 2, 3, 0]).numpy().astype("int32"),
            "valid_edge_node": torch.ones(5).numpy().astype("int32"),
            "link_features": np.array(["distance", "rate_sum", "csi_mean", "active_task_count"]),
        }
        stats = make_normalization_stats(arrays, [0, 1], rate_target_transform="residual_last_rate")
        dataset = V6WorldModelDataset(arrays, [0, 1], stats, rate_target_transform="residual_last_rate")

        _, target = dataset[0]

        expected_residual = y_link_rate[0, ..., None] - x_link[0, -1, :, 1][None, :, None]
        expected = normalize(expected_residual, stats["y_link_rate"])[0]
        torch.testing.assert_close(target["link_rate"], torch.from_numpy(expected))

    def test_residual_last_rate_predictions_add_baseline_during_collection(self):
        from run_world_model_v6_dual_graph_rollout import denormalize_link_rate_prediction

        pred_residual_norm = np.array([[[[0.5], [1.0]]]], dtype=np.float32)
        stats = {
            "y_link_rate": (np.array([[[[1.0]]]], dtype=np.float32), np.array([[[[2.0]]]], dtype=np.float32)),
            "rate_target_transform": "residual_last_rate",
        }
        baseline = np.array([[[[10.0], [20.0]]]], dtype=np.float32)

        pred = denormalize_link_rate_prediction(pred_residual_norm, stats, baseline=baseline)

        np.testing.assert_allclose(pred, np.array([[[[12.0], [23.0]]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
