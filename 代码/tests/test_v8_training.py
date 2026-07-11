import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _make_arrays(num_samples=8):
    rng = np.random.default_rng(20260614)
    num_nodes = 4
    num_edges = 5
    history = 3
    horizon = 2
    node_dim = 6
    link_dim = 4
    action_dim = 3
    task_dim = 5
    arrays = {
        "x_node": rng.normal(size=(num_samples, history, num_nodes, node_dim)).astype("float32"),
        "x_link": rng.normal(size=(num_samples, history, num_edges, link_dim)).astype("float32"),
        "x_task": rng.normal(size=(num_samples, history, task_dim)).astype("float32"),
        "edge_a_hist": rng.normal(size=(num_samples, history, num_edges, action_dim)).astype("float32"),
        "edge_a_future": rng.normal(size=(num_samples, horizon, num_edges, action_dim)).astype("float32"),
        "y_node": rng.normal(size=(num_samples, horizon, num_nodes, node_dim)).astype("float32"),
        "y_link_rate": rng.uniform(0.0, 30.0, size=(num_samples, horizon, num_edges)).astype("float32"),
        "y_link_active": rng.integers(0, 2, size=(num_samples, horizon, num_edges)).astype("float32"),
        "y_task": rng.normal(size=(num_samples, horizon, task_dim)).astype("float32"),
        "edge_src_idx": np.array([0, 0, 1, 2, 3], dtype=np.int32),
        "edge_dst_idx": np.array([1, 2, 2, 3, 0], dtype=np.int32),
        "valid_edge_node": np.ones(num_edges, dtype=np.int32),
        "sample_seed": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
    }
    arrays["y_link_rate"] = arrays["y_link_rate"] * arrays["y_link_active"]
    return arrays


class V8TrainingTest(unittest.TestCase):
    def test_training_stats_ignore_validation_and_test_values(self):
        from pi_jwm.v6_data import make_normalization_stats

        arrays = _make_arrays()
        train_idx = np.array([0, 1, 2, 3], dtype=np.int64)
        original = make_normalization_stats(arrays, train_idx)
        arrays["x_node"][4:] = 1e6
        arrays["x_link"][4:] = -1e6
        arrays["y_link_rate"][4:] = 1e6
        changed = make_normalization_stats(arrays, train_idx)

        for key in ("x_node", "x_link", "y_link_rate"):
            np.testing.assert_array_equal(original[key][0], changed[key][0])
            np.testing.assert_array_equal(original[key][1], changed[key][1])

    def test_training_preflight_rejects_action_target_shape_mismatch(self):
        from run_world_model_v8_full_training import validate_training_arrays

        arrays = _make_arrays()
        arrays["edge_a_future"] = arrays["edge_a_future"][:, :, :-1]

        with self.assertRaisesRegex(ValueError, "edge_a_future.*horizon and edge dimensions"):
            validate_training_arrays(arrays, future_action_mode="full")

    def test_default_split_rejects_expanded_seed_sets(self):
        from run_world_model_v8_full_training import resolve_seed_splits

        sample_seed = np.repeat(np.arange(60), 2)

        with self.assertRaisesRegex(ValueError, "Explicit seed splits are required"):
            resolve_seed_splits(sample_seed)

    def test_training_preflight_rejects_zero_future_actions_for_conditioned_modes(self):
        from run_world_model_v8_full_training import validate_training_arrays

        arrays = _make_arrays()
        arrays["edge_a_future"][:] = 0.0
        with self.assertRaisesRegex(ValueError, "edge_a_future is all zero"):
            validate_training_arrays(arrays, future_action_mode="full")
        with self.assertRaisesRegex(ValueError, "first action step is all zero"):
            validate_training_arrays(arrays, future_action_mode="first_step_only")
        validate_training_arrays(arrays, future_action_mode="none")

    def test_dataset_can_hide_closed_loop_future_actions_after_first_step(self):
        from pi_jwm.v6_data import V6WorldModelDataset, make_normalization_stats

        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        dataset = V6WorldModelDataset(
            arrays,
            [0],
            stats,
            future_action_mode="first_step_only",
        )

        batch, _ = dataset[0]
        mean, std = stats["edge_a_future"]
        restored = batch.future_actions.numpy() * std[0] + mean[0]

        np.testing.assert_allclose(restored[0], arrays["edge_a_future"][0, 0], atol=1e-6)
        np.testing.assert_allclose(restored[1:], 0.0, atol=1e-6)

    def test_dataset_rejects_unknown_future_action_mode(self):
        from pi_jwm.v6_data import V6WorldModelDataset, make_normalization_stats

        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        with self.assertRaisesRegex(ValueError, "future_action_mode"):
            V6WorldModelDataset(arrays, [0], stats, future_action_mode="oracle_magic")

    def test_resolve_seed_splits_rejects_overlapping_seed_sets(self):
        from run_world_model_v8_full_training import resolve_seed_splits

        sample_seed = np.array([0, 1, 2, 3], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            resolve_seed_splits(
                sample_seed,
                train_seeds=[0, 1],
                val_seeds=[1, 2],
                test_seeds=[3],
            )

    def test_build_v8_model_from_arrays_uses_dataset_edge_indices(self):
        from pi_jwm.v8_training import build_v8_model_from_arrays

        arrays = _make_arrays()
        model = build_v8_model_from_arrays(arrays, hidden_dim=16)

        self.assertEqual(model.edge_src_idx.cpu().tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(model.edge_dst_idx.cpu().tolist(), [1, 2, 2, 3, 0])
        self.assertEqual(model.config.horizon, 2)

    def test_v8_training_helpers_do_not_import_script_entrypoints(self):
        import pi_jwm.v8_training as v8_training

        self.assertEqual(v8_training.compute_v8_loss.__module__, "pi_jwm.v8_training")
        self.assertEqual(v8_training.move_v8_batch_to_device.__module__, "pi_jwm.v8_training")
        self.assertEqual(v8_training.denormalize_v8_link_rate_prediction.__module__, "pi_jwm.v8_training")

    def test_train_one_epoch_and_evaluate_return_world_model_metrics(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model, train_v8_one_epoch

        torch.manual_seed(20260614)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        train_ds = V6WorldModelDataset(arrays, np.arange(6), stats)
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        train_loader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=collate_v6_world_model_batch)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(arrays, hidden_dim=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_v8_one_epoch(model, train_loader, optimizer, torch.device("cpu"))
        eval_metrics = evaluate_v8_model(model, eval_loader, torch.device("cpu"), stats)

        self.assertIn("total", train_metrics)
        self.assertGreaterEqual(train_metrics["total"], 0.0)
        self.assertIn("activity", eval_metrics)
        self.assertIn("active_rate", eval_metrics)
        self.assertIn("node", eval_metrics)
        self.assertIn("task", eval_metrics)
        self.assertTrue(np.isfinite(eval_metrics["link_rate"]["rmse"]))

    def test_v8_training_can_use_aux_soft_zero_rate_output(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model, train_v8_one_epoch

        torch.manual_seed(20260614)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        train_ds = V6WorldModelDataset(arrays, np.arange(6), stats)
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        train_loader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=collate_v6_world_model_batch)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(arrays, hidden_dim=16, active_rate_auxiliary=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_v8_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            rate_loss_mode="active_mixed",
            active_rate_auxiliary_weight=0.3,
            rate_output_mode="aux_soft_zero",
            inactive_rate_value=0.0,
        )
        eval_metrics = evaluate_v8_model(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            rate_output_mode="aux_soft_zero",
            inactive_rate_value=0.0,
        )

        self.assertIn("active_rate_auxiliary", train_metrics)
        self.assertIn("active_rate_auxiliary", eval_metrics)
        self.assertTrue(np.isfinite(eval_metrics["link_rate"]["rmse"]))

    def test_v8_evaluation_can_apply_hurdle_gate_power_in_normalized_space(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, collect_v8_predictions

        torch.manual_seed(20260617)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(arrays, hidden_dim=16, rate_output_mode="hurdle_soft")

        base = collect_v8_predictions(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            hurdle_gate_temperature=1.0,
            hurdle_gate_power=1.0,
        )
        softer_gate = collect_v8_predictions(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            hurdle_gate_temperature=1.0,
            hurdle_gate_power=0.5,
        )

        self.assertIn("link_positive_rate_pred", base)
        np.testing.assert_allclose(base["link_rate_pred"], collect_v8_predictions(model, eval_loader, torch.device("cpu"), stats)["link_rate_pred"], rtol=1e-6, atol=1e-6)
        self.assertFalse(np.allclose(base["link_rate_pred"], softer_gate["link_rate_pred"]))

    def test_v8_dual_hurdle_model_outputs_conservative_and_gated_rates(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, collect_v8_predictions

        torch.manual_seed(20260617)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(arrays, hidden_dim=16, rate_output_mode="hurdle_dual")
        batch, _ = next(iter(eval_loader))

        outputs = model(batch)
        main = collect_v8_predictions(model, eval_loader, torch.device("cpu"), stats, rate_output_mode="main")
        gated = collect_v8_predictions(model, eval_loader, torch.device("cpu"), stats, rate_output_mode="hurdle_gate")
        blend = collect_v8_predictions(model, eval_loader, torch.device("cpu"), stats, rate_output_mode="dual_soft_blend")

        self.assertIn("link_positive_rate", outputs)
        self.assertIn("link_hurdle_rate", outputs)
        self.assertEqual(outputs["link_rate"].shape, outputs["link_hurdle_rate"].shape)
        self.assertFalse(np.allclose(main["link_rate_pred"], gated["link_rate_pred"]))
        self.assertFalse(np.allclose(main["link_rate_pred"], blend["link_rate_pred"]))

    def test_v8_training_can_use_moe_active_rate_auxiliary_head(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model, train_v8_one_epoch

        torch.manual_seed(20260614)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        train_ds = V6WorldModelDataset(arrays, np.arange(6), stats)
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        train_loader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=collate_v6_world_model_batch)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(
            arrays,
            hidden_dim=16,
            active_rate_auxiliary=True,
            active_rate_head_mode="moe",
            num_rate_experts=3,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_v8_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            rate_loss_mode="active_mixed",
            active_rate_auxiliary_weight=0.3,
            rate_output_mode="aux_soft_zero",
            inactive_rate_value=0.0,
        )
        eval_metrics = evaluate_v8_model(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            rate_output_mode="aux_soft_zero",
            inactive_rate_value=0.0,
        )

        self.assertEqual(model.config.active_rate_head_mode, "moe")
        self.assertEqual(model.config.num_rate_experts, 3)
        self.assertIn("active_rate_auxiliary", train_metrics)
        self.assertIn("active_rate_auxiliary", eval_metrics)
        self.assertTrue(np.isfinite(eval_metrics["active_rate"]["active_rmse"]))

    def test_v8_training_can_use_recurrent_latent_transition(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model, train_v8_one_epoch

        torch.manual_seed(20260614)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        train_ds = V6WorldModelDataset(arrays, np.arange(6), stats)
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        train_loader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=collate_v6_world_model_batch)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(
            arrays,
            hidden_dim=16,
            latent_transition_mode="recurrent",
            active_rate_auxiliary=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_v8_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            rate_loss_mode="active_mixed",
            active_rate_auxiliary_weight=0.3,
            rate_output_mode="aux_soft_zero",
        )
        eval_metrics = evaluate_v8_model(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            rate_output_mode="aux_soft_zero",
        )

        self.assertEqual(model.config.latent_transition_mode, "recurrent")
        self.assertIn("total", train_metrics)
        self.assertTrue(np.isfinite(eval_metrics["task"]["rmse"]))

    def test_v8_training_can_use_stgcn_light_history_encoder(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model, train_v8_one_epoch

        torch.manual_seed(20260614)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        train_ds = V6WorldModelDataset(arrays, np.arange(6), stats)
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        train_loader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=collate_v6_world_model_batch)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(
            arrays,
            hidden_dim=16,
            history_encoder="stgcn_light",
            active_rate_auxiliary=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_v8_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            rate_loss_mode="active_mixed",
            active_rate_auxiliary_weight=0.3,
            rate_output_mode="aux_soft_zero",
        )
        eval_metrics = evaluate_v8_model(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            rate_output_mode="aux_soft_zero",
        )

        self.assertEqual(model.config.history_encoder, "stgcn_light")
        self.assertIn("total", train_metrics)
        self.assertTrue(np.isfinite(eval_metrics["link_rate"]["rmse"]))

    def test_v8_training_can_use_stgcn_full_history_encoder(self):
        from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, make_normalization_stats
        from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model, train_v8_one_epoch

        torch.manual_seed(20260614)
        arrays = _make_arrays()
        stats = make_normalization_stats(arrays, np.arange(6))
        train_ds = V6WorldModelDataset(arrays, np.arange(6), stats)
        eval_ds = V6WorldModelDataset(arrays, np.arange(6, 8), stats)
        train_loader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=collate_v6_world_model_batch)
        eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_v6_world_model_batch)
        model = build_v8_model_from_arrays(
            arrays,
            hidden_dim=16,
            history_encoder="stgcn_full",
            active_rate_auxiliary=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_v8_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            rate_loss_mode="active_mixed",
            active_rate_auxiliary_weight=0.3,
            rate_output_mode="aux_soft_zero",
        )
        eval_metrics = evaluate_v8_model(
            model,
            eval_loader,
            torch.device("cpu"),
            stats,
            rate_output_mode="aux_soft_zero",
        )

        self.assertEqual(model.config.history_encoder, "stgcn_full")
        self.assertIn("total", train_metrics)
        self.assertTrue(np.isfinite(eval_metrics["active_rate"]["active_rmse"]))

    def test_v8_loss_supports_explicit_task_weights(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.full((1, 1, 1, 1), 1.0),
            "link_activity_logit": torch.zeros(1, 1, 1),
            "link_rate": torch.full((1, 1, 1), 2.0),
            "task": torch.full((1, 1, 1), 3.0),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.ones(1, 1, 1),
            "link_rate": torch.zeros(1, 1, 1),
            "task": torch.zeros(1, 1, 1),
        }

        default_loss, default_parts = compute_v8_loss(outputs, target, rate_loss_mode="active_only")
        weighted_loss, weighted_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=2.0,
            activity_loss_weight=0.5,
            rate_loss_weight=0.1,
            task_loss_weight=3.0,
        )

        self.assertIn("node_weighted", weighted_parts)
        self.assertIn("activity_weighted", weighted_parts)
        self.assertIn("rate_weighted", weighted_parts)
        self.assertIn("task_weighted", weighted_parts)
        self.assertNotAlmostEqual(float(default_loss.detach()), float(weighted_loss.detach()))
        self.assertAlmostEqual(weighted_parts["node_weighted"], default_parts["node"] * 2.0)
        self.assertAlmostEqual(weighted_parts["rate_weighted"], default_parts["rate"] * 0.1)

    def test_v8_training_script_reports_validation_and_test_metrics(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=10)
        arrays["sample_seed"] = np.array([0, 1, 2, 3, 4, 5, 8, 8, 9, 9], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=4,
                    max_val_samples=2,
                    max_test_samples=2,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260614,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="gated",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=False,
                    active_rate_auxiliary_weight=0.0,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="main",
                    inactive_rate_value=0.0,
                    eval_hurdle_gate_temperature=1.0,
                    eval_hurdle_gate_power=0.5,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                )
            )

        self.assertIn("val_eval", summary)
        self.assertIn("test_eval", summary)
        self.assertIn("active_rate", summary["test_eval"])

    def test_v8_training_script_reports_best_checkpoint_metrics(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=10)
        arrays["sample_seed"] = np.array([0, 1, 2, 3, 4, 5, 8, 8, 9, 9], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=2,
                    max_train_samples=4,
                    max_val_samples=2,
                    max_test_samples=2,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260614,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    model_rate_output_mode="hurdle_soft",
                    rate_output_mode="main",
                    inactive_rate_value=0.0,
                    eval_hurdle_gate_temperature=1.0,
                    eval_hurdle_gate_power=0.5,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                )
            )

            self.assertIn("best_epoch", summary)
            self.assertIn("best_val_eval", summary)
            self.assertIn("best_test_eval", summary)
            self.assertIn("best_checkpoint_path", summary)
            self.assertEqual(summary["config"]["eval_hurdle_gate_power"], 0.5)
            self.assertEqual(summary["history"][0]["val"]["hurdle_gate"]["power"], 0.5)
            self.assertGreaterEqual(summary["best_epoch"], 1)
            self.assertLessEqual(summary["best_epoch"], 2)
            self.assertIn("active_rate", summary["best_test_eval"])
            self.assertTrue(Path(summary["best_checkpoint_path"]).exists())

    def test_v8_training_script_can_select_composite_best_metric(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=10)
        arrays["sample_seed"] = np.array([0, 1, 2, 3, 4, 5, 8, 8, 9, 9], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=2,
                    max_train_samples=4,
                    max_val_samples=2,
                    max_test_samples=2,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260614,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="recurrent",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_composite",
                    node_loss_weight=1.0,
                    activity_loss_weight=1.5,
                    rate_loss_weight=0.3,
                    task_loss_weight=1.0,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                )
            )

            self.assertEqual(summary["best_metric_name"], "val_composite")
            self.assertIn("best_metric_value", summary)
            self.assertIn("best_metric_values", summary["history"][0])
            self.assertIn("val_composite", summary["history"][0]["best_metric_values"])
            self.assertEqual(summary["config"]["node_loss_weight"], 1.0)
            self.assertEqual(summary["config"]["activity_loss_weight"], 1.5)
            self.assertTrue(Path(summary["best_checkpoint_path"]).exists())

    def test_compute_best_metric_values_supports_precision_constrained_active_rate(self):
        from run_world_model_v8_full_training import compute_best_metric_values

        val_metrics = {
            "active_rate": {"active_rmse": 300.0},
            "link_rate": {"rmse": 90.0},
            "node": {"rmse": 20.0},
            "task": {"rmse": 3.0},
            "activity": {"precision": 0.02, "recall": 0.20, "f1": 0.036},
        }

        values = compute_best_metric_values(
            val_metrics,
            min_precision=0.01,
            min_recall=0.05,
            precision_penalty_weight=10000.0,
            recall_penalty_weight=1000.0,
        )
        failing_values = compute_best_metric_values(
            {
                **val_metrics,
                "activity": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
            min_precision=0.01,
            min_recall=0.05,
            precision_penalty_weight=10000.0,
            recall_penalty_weight=1000.0,
        )

        self.assertIn("val_precision_constrained_active_rate", values)
        self.assertEqual(values["val_precision_constrained_active_rate"], 300.0)
        self.assertGreater(failing_values["val_precision_constrained_active_rate"], 300.0)
        self.assertAlmostEqual(failing_values["val_precision_constrained_active_rate"], 450.0)

    def test_resolve_seed_splits_accepts_custom_seed_groups(self):
        from run_world_model_v8_full_training import parse_seed_list, resolve_seed_splits

        sample_seed = np.array([0, 0, 1, 2, 2, 3, 4, 4], dtype=np.int32)
        train_idx, val_idx, test_idx, spec = resolve_seed_splits(
            sample_seed,
            train_seeds=parse_seed_list("0,1"),
            val_seeds=parse_seed_list("2 3"),
            test_seeds=parse_seed_list("4"),
        )

        self.assertEqual(train_idx.tolist(), [0, 1, 2])
        self.assertEqual(val_idx.tolist(), [3, 4, 5])
        self.assertEqual(test_idx.tolist(), [6, 7])
        self.assertEqual(spec["train_seeds"], [0, 1])
        self.assertEqual(spec["val_seeds"], [2, 3])
        self.assertEqual(spec["test_seeds"], [4])

    def test_run_training_records_custom_seed_split(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                )
            )

            self.assertEqual(summary["split_seed_spec"]["train_seeds"], [0, 1])
            self.assertEqual(summary["split_seed_spec"]["val_seeds"], [2])
            self.assertEqual(summary["split_seed_spec"]["test_seeds"], [3])
            self.assertEqual(summary["split_sizes"]["train"], 4)
            self.assertEqual(summary["split_sizes"]["val"], 2)
            self.assertEqual(summary["split_sizes"]["test"], 2)

    def test_build_active_rate_lds_config_uses_training_positive_rates_only(self):
        from run_world_model_v8_full_training import build_active_rate_lds_config

        arrays = {
            "y_link_rate": np.array(
                [
                    [[10.0, 0.0]],
                    [[20.0, 0.0]],
                    [[1000.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            "y_link_active": np.array(
                [
                    [[1.0, 0.0]],
                    [[1.0, 0.0]],
                    [[1.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        }
        args = Namespace(
            active_rate_reweight_mode="lds",
            lds_bin_width=50.0,
            lds_kernel_size=3,
            lds_sigma=1.0,
            lds_weight_min=0.5,
            lds_weight_max=3.0,
            lds_tail_quantile=1.0,
        )

        config = build_active_rate_lds_config(arrays, np.array([0, 1]), args)

        self.assertEqual(config["active_sample_count"], 2)
        self.assertEqual(sum(config["empirical_counts"]), 2)
        self.assertEqual(config["tail_cap"], 50.0)


    def test_v8_loss_can_sample_inactive_edges_for_training_only(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 4),
            "link_rate": torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "link_rate": torch.zeros(1, 1, 4),
            "task": torch.zeros(1, 1, 1),
        }

        _, full_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=1.0,
            inactive_loss_sample_ratio=1.0,
        )
        _, sampled_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=1.0,
            inactive_loss_sample_ratio=1.0 / 3.0,
            loss_sampling_seed=0,
        )

        self.assertAlmostEqual(full_parts["active_rate_loss"], 1.0)
        self.assertAlmostEqual(full_parts["inactive_rate_loss"], (4.0 + 9.0 + 16.0) / 3.0, places=5)
        self.assertAlmostEqual(sampled_parts["active_rate_loss"], 1.0)
        self.assertEqual(sampled_parts["inactive_loss_count"], 1.0)
        self.assertIn(round(sampled_parts["inactive_rate_loss"], 6), {4.0, 9.0, 16.0})

    def test_active_rate_loss_can_emphasize_high_rate_positive_edges(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 2),
            "link_rate": torch.tensor([[[1.0, 1.0]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.ones(1, 1, 2),
            "link_rate": torch.tensor([[[2.0, 5.0]]]),
            "link_rate_raw": torch.tensor([[[200.0, 700.0]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, base_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            high_rate_weight=1.0,
            high_rate_threshold=600.0,
        )
        _, high_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            high_rate_weight=3.0,
            high_rate_threshold=600.0,
        )

        self.assertAlmostEqual(base_parts["active_rate_loss"], (1.0 + 16.0) / 2.0)
        self.assertAlmostEqual(high_parts["active_rate_loss"], (1.0 + 3.0 * 16.0) / 4.0)
        self.assertEqual(high_parts["high_rate_count"], 1.0)

    def test_lds_rate_reweighting_upweights_rare_positive_rates(self):
        from pi_jwm.v8_training import fit_lds_rate_reweighting, lookup_lds_rate_weights

        common = np.full(20, 25.0, dtype=np.float32)
        rare = np.array([425.0, 450.0], dtype=np.float32)
        config = fit_lds_rate_reweighting(
            np.concatenate([common, rare]),
            bin_width=100.0,
            kernel_size=3,
            sigma=1.0,
            weight_min=0.5,
            weight_max=3.0,
            tail_quantile=1.0,
        )

        weights = lookup_lds_rate_weights(torch.tensor([25.0, 450.0]), config)

        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertGreaterEqual(float(weights.min()), 0.5)
        self.assertLessEqual(float(weights.max()), 3.0)
        self.assertEqual(sum(config["empirical_counts"]), 22)

    def test_lds_weighted_active_loss_changes_only_active_error_weighting(self):
        from pi_jwm.v8_training import compute_lds_weighted_active_loss

        rate_error = torch.tensor([1.0, 9.0, 100.0])
        raw_target = torch.tensor([25.0, 450.0, 0.0])
        active_mask = torch.tensor([True, True, False])
        config = {
            "bin_upper_bounds": [100.0, 300.0],
            "bin_weights": [0.5, 1.0, 3.0],
        }

        loss, mean_weight = compute_lds_weighted_active_loss(rate_error, raw_target, active_mask, config)

        self.assertAlmostEqual(float(loss), 27.5 / 3.5, places=6)
        self.assertAlmostEqual(mean_weight, 1.75)

    def test_v8_loss_can_use_lds_weights_for_active_rate_only(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 3),
            "link_rate": torch.tensor([[[1.0, 1.0, 10.0]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[1.0, 1.0, 0.0]]]),
            "link_rate": torch.tensor([[[2.0, 5.0, 0.0]]]),
            "link_rate_raw": torch.tensor([[[25.0, 450.0, 0.0]]]),
            "task": torch.zeros(1, 1, 1),
        }
        config = {
            "bin_upper_bounds": [100.0, 300.0],
            "bin_weights": [0.5, 1.0, 3.0],
        }

        _, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=0.25,
            active_rate_reweight_mode="lds",
            active_rate_lds_config=config,
        )

        self.assertAlmostEqual(parts["active_rate_loss"], (0.5 * 1.0 + 3.0 * 16.0) / 3.5, places=6)
        self.assertAlmostEqual(parts["inactive_rate_loss"], 100.0)
        self.assertEqual(parts["active_rate_reweight_mode"], "lds")
        self.assertAlmostEqual(parts["active_rate_lds_mean_weight"], 1.75)

    def test_ziln_nll_is_finite_prefers_matching_mu_and_backpropagates(self):
        from pi_jwm.v8_training import compute_ziln_nll, ziln_expected_rate

        activity_logit = torch.tensor([[8.0], [-8.0]], requires_grad=True)
        matching_mu = torch.tensor([[np.log(105.0)], [0.0]], dtype=torch.float32, requires_grad=True)
        bad_mu = torch.tensor([[np.log(10.0)], [0.0]], dtype=torch.float32)
        log_sigma = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
        target = torch.tensor([[100.0], [0.0]])

        matching_loss = compute_ziln_nll(activity_logit, matching_mu, log_sigma, target)
        bad_loss = compute_ziln_nll(activity_logit.detach(), bad_mu, log_sigma.detach(), target)
        expected = ziln_expected_rate(matching_mu[:1], log_sigma[:1])
        matching_loss.backward()

        self.assertTrue(torch.isfinite(matching_loss))
        self.assertLess(float(matching_loss.detach()), float(bad_loss.detach()))
        self.assertGreater(float(expected.detach()), 105.0)
        self.assertGreater(float(activity_logit.grad.abs().sum()), 0.0)
        self.assertGreater(float(matching_mu.grad.abs().sum()), 0.0)
        self.assertGreater(float(log_sigma.grad.abs().sum()), 0.0)

    def test_balanced_mse_prefers_matching_active_rates_and_backpropagates(self):
        from pi_jwm.v8_training import compute_balanced_mse_loss

        target = torch.tensor([1.0, 3.0, 5.0])
        matching = torch.tensor([1.1, 2.9, 5.1], requires_grad=True)
        shifted = torch.tensor([3.1, 4.9, 7.1])

        matching_loss, matching_count = compute_balanced_mse_loss(matching, target, noise_sigma=1.0)
        shifted_loss, shifted_count = compute_balanced_mse_loss(shifted, target, noise_sigma=1.0)
        matching_loss.backward()

        self.assertLess(float(matching_loss.detach()), float(shifted_loss.detach()))
        self.assertEqual(matching_count, 3)
        self.assertEqual(shifted_count, 3)
        self.assertGreater(float(matching.grad.abs().sum()), 0.0)

    def test_balanced_mse_falls_back_to_mse_for_sparse_positive_batch(self):
        from pi_jwm.v8_training import compute_balanced_mse_loss

        pred = torch.tensor([2.0, 5.0])
        target = torch.tensor([1.0, 3.0])

        loss, count = compute_balanced_mse_loss(pred, target, noise_sigma=1.0, minimum_count=3)

        self.assertAlmostEqual(float(loss), 2.5)
        self.assertEqual(count, 0)

    def test_balanced_mse_returns_zero_for_empty_positive_batch(self):
        from pi_jwm.v8_training import compute_balanced_mse_loss

        empty = torch.empty(0)

        loss, count = compute_balanced_mse_loss(empty, empty, noise_sigma=1.0)

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(count, 0)

    def test_v8_loss_can_use_balanced_mse_for_active_rate(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 3),
            "link_rate": torch.tensor([[[1.1, 2.9, 5.1]]], requires_grad=True),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.ones(1, 1, 3),
            "link_rate": torch.tensor([[[1.0, 3.0, 5.0]]]),
            "task": torch.zeros(1, 1, 1),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            active_rate_reweight_mode="bmc",
            active_rate_bmc_noise_sigma=1.0,
            active_rate_bmc_minimum_count=3,
        )
        loss.backward()

        self.assertEqual(parts["active_rate_reweight_mode"], "bmc")
        self.assertEqual(parts["active_rate_bmc_count"], 3.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(outputs["link_rate"].grad.abs().sum()), 0.0)

    def test_active_mass_loss_supervises_total_active_rate_mass(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 3, 1),
            "link_rate": torch.zeros(1, 1, 3, 1),
            "link_active_mass_total": torch.tensor([[[[10.0]]]], requires_grad=True),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0], [1.0]]]]),
            "link_rate": torch.tensor([[[[3.0], [0.0], [4.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            active_mass_loss_weight=0.5,
            rate_loss_weight=0.0,
            activity_loss_weight=0.0,
            node_loss_weight=0.0,
            task_loss_weight=0.0,
        )

        self.assertAlmostEqual(parts["active_mass_total_loss"], 9.0, places=6)
        self.assertEqual(parts["active_mass_loss_weight"], 0.5)
        loss.backward()
        self.assertGreater(float(outputs["link_active_mass_total"].grad.abs().sum()), 0.0)

    def test_active_mass_raw_loss_supervises_physical_rate_mass(self):
        from pi_jwm.v8_training import compute_v8_loss

        predicted_total = torch.tensor([[[[8.0]]]], requires_grad=True)
        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 2, 1),
            "link_rate": torch.zeros(1, 1, 2, 1),
            "link_active_mass_total": predicted_total,
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [1.0]]]]),
            "link_rate": torch.tensor([[[[3.0], [4.0]]]]),
            "link_rate_raw": torch.tensor([[[[40.0], [60.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            active_mass_loss_weight=1.0,
            active_mass_target_mode="raw",
            active_mass_raw_stats=(
                np.asarray([10.0], dtype=np.float32),
                np.asarray([5.0], dtype=np.float32),
            ),
            rate_loss_weight=0.0,
            activity_loss_weight=0.0,
            node_loss_weight=0.0,
            task_loss_weight=0.0,
        )

        self.assertAlmostEqual(parts["active_mass_total_loss"], 2500.0, places=6)
        self.assertEqual(parts["active_mass_target_mode"], "raw")
        loss.backward()
        self.assertLess(float(predicted_total.grad.item()), 0.0)

    def test_select_v8_link_rate_output_can_use_active_mass_allocation(self):
        from pi_jwm.v8_training import select_v8_link_rate_output

        outputs = {
            "link_rate": torch.zeros(1, 1, 2, 1),
            "link_active_mass_rate": torch.tensor([[[[3.0], [5.0]]]]),
        }

        selected = select_v8_link_rate_output(outputs, rate_output_mode="active_mass_alloc")

        torch.testing.assert_close(selected, outputs["link_active_mass_rate"])

    def test_v8_loss_supports_focal_activity_loss(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.tensor([[[5.0, -5.0]]]),
            "link_rate": torch.zeros(1, 1, 2),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[1.0, 0.0]]]),
            "link_rate": torch.zeros(1, 1, 2),
            "task": torch.zeros(1, 1, 1),
        }

        _, bce_parts = compute_v8_loss(outputs, target, activity_loss_mode="bce", activity_pos_weight=80.0)
        _, focal_parts = compute_v8_loss(
            outputs,
            target,
            activity_loss_mode="focal",
            activity_pos_weight=80.0,
            activity_focal_gamma=2.0,
        )

        self.assertLess(focal_parts["activity"], bce_parts["activity"])
        self.assertEqual(focal_parts["activity_loss_mode"], "focal")

    def test_run_training_records_active_heavy_loss_repair_config(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                )
            )

        self.assertEqual(summary["config"]["activity_loss_mode"], "focal")
        self.assertEqual(summary["config"]["activity_pos_weight"], 160.0)
        self.assertEqual(summary["config"]["inactive_loss_sample_ratio"], 0.25)
        self.assertIn("inactive_loss_count", summary["history"][0]["train"])


    def test_run_training_falls_back_when_best_metric_is_nan(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        arrays["y_link_active"][arrays["sample_seed"] == 2] = 0.0
        arrays["y_link_rate"][arrays["sample_seed"] == 2] = 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="bce",
                    activity_pos_weight=80.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=1.0,
                )
            )

            self.assertTrue(Path(summary["best_checkpoint_path"]).exists())

        self.assertEqual(summary["best_epoch"], 1)
        self.assertIsNotNone(summary["best_val_eval"])


    def test_v8_loss_penalizes_inactive_false_positive_probabilities(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.tensor([[[0.0, 4.0, -4.0]]]),
            "link_rate": torch.zeros(1, 1, 3),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[1.0, 0.0, 0.0]]]),
            "link_rate": torch.zeros(1, 1, 3),
            "task": torch.zeros(1, 1, 1),
        }

        no_penalty_loss, no_penalty_parts = compute_v8_loss(
            outputs,
            target,
            activity_loss_mode="focal",
            activity_pos_weight=160.0,
            false_positive_penalty_weight=0.0,
        )
        penalty_loss, penalty_parts = compute_v8_loss(
            outputs,
            target,
            activity_loss_mode="focal",
            activity_pos_weight=160.0,
            false_positive_penalty_weight=2.0,
        )

        self.assertGreater(no_penalty_parts["false_positive_penalty"], 0.4)
        self.assertAlmostEqual(no_penalty_parts["false_positive_penalty"], penalty_parts["false_positive_penalty"], places=6)
        self.assertGreater(float(penalty_loss.detach()), float(no_penalty_loss.detach()))

    def test_v8_loss_supports_dynamic_hard_negative_activity_loss(self):
        from pi_jwm.v8_training import compute_v8_loss

        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0], [0.0], [0.0]]]]),
            "link_rate": torch.zeros(1, 1, 4, 1),
            "task": torch.zeros(1, 1, 1),
        }
        logits = torch.tensor([[[[0.0], [4.0], [-2.0], [1.0]]]], requires_grad=True)
        outputs = {
            "node": torch.zeros_like(target["node"]),
            "link_activity_logit": logits,
            "link_rate": torch.zeros_like(target["link_rate"]),
            "task": torch.zeros_like(target["task"]),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=0.0,
            task_loss_weight=0.0,
            dynamic_hard_negative_weight=1.0,
            dynamic_hard_negative_ratio=0.5,
        )
        loss.backward()

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([4.0, 1.0]),
            torch.zeros(2),
            reduction="mean",
        )
        self.assertAlmostEqual(parts["dynamic_hard_negative_activity"], float(expected), places=6)
        self.assertEqual(parts["dynamic_hard_negative_count"], 2.0)
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=6)
        self.assertGreater(float(logits.grad[0, 0, 1, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 2, 0]), 0.0)

    def test_v8_loss_can_use_candidate_edge_mask_for_training(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 4),
            "link_rate": torch.tensor([[[1.0, 2.0, 10.0, 20.0]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "link_rate": torch.zeros(1, 1, 4),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=1.0,
            candidate_loss_mask=torch.tensor([True, True, False, False]),
        )

        self.assertEqual(parts["candidate_loss_edge_count"], 2.0)
        self.assertEqual(parts["active_loss_count"], 1.0)
        self.assertEqual(parts["inactive_loss_count"], 1.0)
        self.assertAlmostEqual(parts["active_rate_loss"], 1.0)
        self.assertAlmostEqual(parts["inactive_rate_loss"], 4.0)

    def test_v8_loss_can_mask_rate_edges_without_masking_activity(self):
        from pi_jwm.v8_training import compute_v8_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.tensor([[[0.0, -4.0, 4.0, -4.0]]]),
            "link_rate": torch.tensor([[[1.0, 2.0, 10.0, 20.0]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            "link_rate": torch.zeros(1, 1, 4),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=1.0,
            candidate_rate_loss_mask=torch.tensor([True, True, False, False]),
        )

        self.assertEqual(parts["candidate_loss_edge_count"], 4.0)
        self.assertEqual(parts["candidate_rate_loss_edge_count"], 2.0)
        self.assertEqual(parts["active_loss_count"], 1.0)
        self.assertEqual(parts["inactive_loss_count"], 1.0)
        self.assertAlmostEqual(parts["active_rate_loss"], 1.0)
        self.assertAlmostEqual(parts["inactive_rate_loss"], 4.0)
        self.assertGreater(parts["false_positive_penalty"], 0.25)

    def test_run_training_records_false_positive_penalty_config(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    false_positive_penalty_weight=2.0,
                    dynamic_hard_negative_weight=1.5,
                    dynamic_hard_negative_ratio=0.2,
                )
            )

        self.assertEqual(summary["config"]["false_positive_penalty_weight"], 2.0)
        self.assertEqual(summary["config"]["dynamic_hard_negative_weight"], 1.5)
        self.assertEqual(summary["config"]["dynamic_hard_negative_ratio"], 0.2)
        self.assertIn("false_positive_penalty", summary["history"][0]["train"])
        self.assertIn("dynamic_hard_negative_activity", summary["history"][0]["train"])

    def test_run_training_records_candidate_pruning_metrics(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        arrays["y_link_active"][:] = 0.0
        arrays["y_link_rate"][:] = 0.0
        arrays["y_link_active"][:, :, 0] = 1.0
        arrays["y_link_rate"][:, :, 0] = 10.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    false_positive_penalty_weight=0.0,
                    candidate_pruning_mode="train_active_plus_hard_negatives",
                    candidate_hard_negative_count=2,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertEqual(summary["config"]["candidate_pruning_mode"], "train_active_plus_hard_negatives")
        self.assertGreaterEqual(summary["candidate_loss_mask"]["edge_count"], 1)
        self.assertIn("best_test_eval_pruned", summary)
        self.assertIn("activity", summary["best_test_eval_pruned"])
        self.assertIn("candidate_loss_edge_count", summary["history"][0]["train"])

    def test_run_training_records_rate_only_candidate_pruning_scope(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        arrays["y_link_active"][:] = 0.0
        arrays["y_link_rate"][:] = 0.0
        arrays["y_link_active"][:, :, 0] = 1.0
        arrays["y_link_rate"][:, :, 0] = 10.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="aux_soft_zero",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    false_positive_penalty_weight=0.0,
                    candidate_pruning_mode="train_active_plus_hard_negatives",
                    candidate_hard_negative_count=2,
                    candidate_pruning_scope="rate_only",
                )
            )

        self.assertEqual(summary["config"]["candidate_pruning_scope"], "rate_only")
        train_metrics = summary["history"][0]["train"]
        self.assertGreater(train_metrics["candidate_loss_edge_count"], train_metrics["candidate_rate_loss_edge_count"])
        self.assertEqual(train_metrics["candidate_rate_loss_edge_count"], summary["candidate_loss_mask"]["edge_count"])

    def test_run_training_can_use_hurdle_model_rate_output(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    model_rate_output_mode="hurdle_soft",
                    rate_output_mode="main",
                    inactive_rate_value=0.0,
                    best_metric="val_precision_constrained_composite",
                    best_min_precision=0.01,
                    best_min_recall=0.05,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    hurdle_train_gate_mode="predicted",
                    hurdle_train_gate_power=0.5,
                    false_positive_penalty_weight=0.0,
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertEqual(summary["config"]["model_rate_output_mode"], "hurdle_soft")
        self.assertEqual(summary["config"]["hurdle_train_gate_mode"], "predicted")
        self.assertEqual(summary["config"]["hurdle_train_gate_power"], 0.5)
        self.assertEqual(summary["best_metric_name"], "val_precision_constrained_composite")
        self.assertIn("val_precision_constrained_composite", summary["history"][0]["best_metric_values"])
        self.assertIn("positive_rate_active", summary["best_test_eval"])

    def test_run_training_can_use_event_memory_link_features(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    model_rate_output_mode="hurdle_soft",
                    use_event_memory_features=True,
                    rate_output_mode="main",
                    inactive_rate_value=0.0,
                    best_metric="val_precision_constrained_composite",
                    best_min_precision=0.01,
                    best_min_recall=0.05,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    hurdle_train_gate_mode="predicted",
                    false_positive_penalty_weight=0.0,
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertTrue(summary["config"]["use_event_memory_features"])
        self.assertEqual(summary["config"]["info_edge_dim"], arrays["x_link"].shape[-1] + 3)

    def test_run_training_records_positive_rate_specialist_config(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        arrays["y_link_active"][:] = 0.0
        arrays["y_link_rate"][:] = 0.0
        arrays["y_link_active"][:, :, 0] = 1.0
        arrays["y_link_rate"][:, :, 0] = 10.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260615,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    model_rate_output_mode="hurdle_soft",
                    rate_output_mode="main",
                    inactive_rate_value=0.0,
                    best_metric="val_precision_constrained_composite",
                    best_min_precision=0.01,
                    best_min_recall=0.05,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    hurdle_train_gate_mode="predicted",
                    false_positive_penalty_weight=0.0,
                    positive_rate_specialist_weight=0.7,
                    positive_rate_target_mode="log1p",
                    positive_rate_loss_mode="huber",
                    positive_rate_tweedie_power=1.4,
                    high_rate_weight=2.5,
                    high_rate_threshold=600.0,
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertEqual(summary["config"]["positive_rate_specialist_weight"], 0.7)
        self.assertEqual(summary["config"]["positive_rate_target_mode"], "log1p")
        self.assertEqual(summary["config"]["positive_rate_loss_mode"], "huber")
        self.assertEqual(summary["config"]["positive_rate_tweedie_power"], 1.4)
        self.assertEqual(summary["config"]["high_rate_weight"], 2.5)
        self.assertEqual(summary["config"]["high_rate_threshold"], 600.0)
        self.assertIn("positive_rate_specialist", summary["history"][0]["train"])
        self.assertIn("high_rate_count", summary["history"][0]["train"])

    def test_run_training_records_active_mass_allocator_config(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        arrays["y_link_active"][:] = 0.0
        arrays["y_link_rate"][:] = 0.0
        arrays["y_link_active"][:, :, :2] = 1.0
        arrays["y_link_rate"][:, :, 0] = 10.0
        arrays["y_link_rate"][:, :, 1] = 5.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)

            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=99,
                    max_val_samples=99,
                    max_test_samples=99,
                    batch_size=2,
                    hidden_dim=8,
                    seed=20260617,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    adaptive_edge_context="none",
                    adaptive_edge_topk=8,
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    model_rate_output_mode="hurdle_mass",
                    rate_output_mode="active_mass_alloc",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    best_min_precision=0.0,
                    best_min_recall=0.0,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    best_min_f1=0.0,
                    best_max_link_rmse=0.0,
                    best_f1_penalty_weight=1000.0,
                    best_link_penalty_weight=10.0,
                    metric_checkpoints="",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    train_seeds="0,1",
                    val_seeds="2",
                    test_seeds="3",
                    activity_loss_mode="focal",
                    activity_pos_weight=160.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=0.25,
                    hurdle_train_gate_mode="predicted",
                    hurdle_train_gate_power=1.0,
                    use_event_memory_features=False,
                    event_memory_routing="shared",
                    false_positive_penalty_weight=0.0,
                    dynamic_hard_negative_weight=0.0,
                    dynamic_hard_negative_ratio=0.1,
                    eval_hurdle_gate_temperature=1.0,
                    eval_hurdle_gate_power=1.0,
                    positive_rate_specialist_weight=0.0,
                    positive_rate_target_mode="normalized",
                    positive_rate_loss_mode="mse",
                    positive_rate_tweedie_power=1.5,
                    high_rate_weight=1.0,
                    high_rate_threshold=0.0,
                    active_mass_loss_weight=0.4,
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertEqual(summary["config"]["model_rate_output_mode"], "hurdle_mass")
        self.assertEqual(summary["config"]["rate_output_mode"], "active_mass_alloc")
        self.assertEqual(summary["config"]["active_mass_loss_weight"], 0.4)
        self.assertEqual(summary["config"]["active_mass_target_mode"], "normalized")
        self.assertIn("active_mass_total_loss", summary["history"][0]["train"])
        self.assertIn("active_mass", summary["best_test_eval"])

    def test_run_training_can_route_event_memory_to_activity_only(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "out"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)
            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=4,
                    max_val_samples=2,
                    max_test_samples=2,
                    train_seeds="0",
                    val_seeds="1",
                    test_seeds="2",
                    batch_size=2,
                    hidden_dim=16,
                    seed=20260616,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="main",
                    model_rate_output_mode="hurdle_soft",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    best_min_precision=0.0,
                    best_min_recall=0.0,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    activity_loss_mode="focal",
                    activity_pos_weight=80.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=1.0,
                    hurdle_train_gate_mode="predicted",
                    use_event_memory_features=True,
                    event_memory_routing="activity_only",
                    false_positive_penalty_weight=0.0,
                    positive_rate_specialist_weight=0.0,
                    positive_rate_target_mode="normalized",
                    positive_rate_loss_mode="mse",
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertTrue(summary["config"]["use_event_memory_features"])
        self.assertEqual(summary["config"]["event_memory_routing"], "activity_only")
        self.assertEqual(summary["config"]["activity_memory_dim"], 3)
        self.assertEqual(summary["config"]["info_edge_dim"], arrays["x_link"].shape[-1] + 3)

    def test_run_training_records_sparse_adaptive_edge_context_config(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=8)
        arrays["sample_seed"] = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "out"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)
            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=1,
                    max_train_samples=4,
                    max_val_samples=2,
                    max_test_samples=2,
                    train_seeds="0",
                    val_seeds="1",
                    test_seeds="2",
                    batch_size=2,
                    hidden_dim=16,
                    seed=20260616,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    adaptive_edge_context="sparse_attention",
                    adaptive_edge_topk=2,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="main",
                    model_rate_output_mode="hurdle_soft",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    best_min_precision=0.0,
                    best_min_recall=0.0,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    activity_loss_mode="focal",
                    activity_pos_weight=80.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=1.0,
                    hurdle_train_gate_mode="predicted",
                    use_event_memory_features=False,
                    event_memory_routing="shared",
                    false_positive_penalty_weight=0.0,
                    positive_rate_specialist_weight=0.0,
                    positive_rate_target_mode="normalized",
                    positive_rate_loss_mode="mse",
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

        self.assertEqual(summary["config"]["adaptive_edge_context"], "sparse_attention")
        self.assertEqual(summary["config"]["adaptive_edge_topk"], 2)

    def test_run_training_can_save_metric_checkpoints(self):
        from run_world_model_v8_full_training import run_training

        arrays = _make_arrays(num_samples=10)
        arrays["sample_seed"] = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "out"
            dataset_dir.mkdir()
            np.savez(dataset_dir / "world_model_dataset_v0_samples.npz", **arrays)
            summary = run_training(
                Namespace(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    epochs=2,
                    max_train_samples=4,
                    max_val_samples=2,
                    max_test_samples=2,
                    train_seeds="0",
                    val_seeds="1",
                    test_seeds="2",
                    batch_size=2,
                    hidden_dim=16,
                    seed=20260617,
                    device="cpu",
                    graph_mode="dual",
                    fusion_mode="cross_attention",
                    fusion_num_heads=4,
                    history_encoder="mean",
                    latent_transition_mode="message_passing",
                    adaptive_edge_context="none",
                    adaptive_edge_topk=8,
                    rate_loss_mode="active_mixed",
                    inactive_rate_weight=0.05,
                    active_rate_auxiliary=True,
                    active_rate_auxiliary_weight=0.3,
                    active_rate_head_mode="mlp",
                    num_rate_experts=4,
                    rate_output_mode="main",
                    model_rate_output_mode="hurdle_soft",
                    inactive_rate_value=0.0,
                    best_metric="val_active_rate_rmse",
                    best_min_precision=0.0,
                    best_min_recall=0.0,
                    best_precision_penalty_weight=10000.0,
                    best_recall_penalty_weight=1000.0,
                    best_min_f1=0.0,
                    best_max_link_rmse=0.0,
                    best_f1_penalty_weight=1000.0,
                    best_link_penalty_weight=10.0,
                    metric_checkpoints="val_active_rate_rmse,val_link_rate_rmse,val_activity_f1",
                    node_loss_weight=0.5,
                    activity_loss_weight=1.0,
                    rate_loss_weight=0.3,
                    task_loss_weight=0.8,
                    activity_loss_mode="focal",
                    activity_pos_weight=80.0,
                    activity_focal_gamma=2.0,
                    inactive_loss_sample_ratio=1.0,
                    hurdle_train_gate_mode="predicted",
                    hurdle_train_gate_power=1.0,
                    use_event_memory_features=False,
                    event_memory_routing="shared",
                    false_positive_penalty_weight=0.0,
                    dynamic_hard_negative_weight=0.0,
                    dynamic_hard_negative_ratio=0.1,
                    eval_hurdle_gate_temperature=1.0,
                    eval_hurdle_gate_power=1.0,
                    positive_rate_specialist_weight=0.0,
                    positive_rate_target_mode="normalized",
                    positive_rate_loss_mode="mse",
                    positive_rate_tweedie_power=1.5,
                    high_rate_weight=1.0,
                    high_rate_threshold=0.0,
                    candidate_pruning_mode="none",
                    candidate_hard_negative_count=0,
                    candidate_pruning_scope="all_losses",
                )
            )

            self.assertIn("metric_checkpoints", summary)
            self.assertIn("val_active_rate_rmse", summary["metric_checkpoints"])
            self.assertTrue((output_dir / "checkpoints" / "v8_dual_best_val_active_rate_rmse.pt").exists())
            self.assertTrue((output_dir / "checkpoints" / "v8_dual_best_val_link_rate_rmse.pt").exists())
            self.assertTrue((output_dir / "checkpoints" / "v8_dual_best_val_activity_f1.pt").exists())

    def test_hurdle_teacher_forcing_rate_loss_does_not_backprop_to_activity_logit(self):
        from pi_jwm.v8_training import compute_v8_loss

        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[2.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        activity_logit = torch.tensor([[[[0.3], [0.3]]]], requires_grad=True)
        positive_rate = torch.tensor([[[[1.0], [1.0]]]], requires_grad=True)
        outputs = {
            "node": torch.zeros_like(target["node"]),
            "link_activity_logit": activity_logit,
            "link_positive_rate": positive_rate,
            "link_rate": torch.sigmoid(activity_logit) * positive_rate,
            "task": torch.zeros_like(target["task"]),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=1.0,
            task_loss_weight=0.0,
            hurdle_train_gate_mode="teacher_forcing",
        )
        loss.backward()

        self.assertEqual(parts["hurdle_train_gate_mode"], "teacher_forcing")
        torch.testing.assert_close(activity_logit.grad, torch.zeros_like(activity_logit))
        self.assertGreater(float(positive_rate.grad.abs().sum()), 0.0)

    def test_hurdle_detach_rate_loss_does_not_backprop_to_activity_logit(self):
        from pi_jwm.v8_training import compute_v8_loss

        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[2.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        activity_logit = torch.tensor([[[[0.3], [0.3]]]], requires_grad=True)
        positive_rate = torch.tensor([[[[1.0], [1.0]]]], requires_grad=True)
        outputs = {
            "node": torch.zeros_like(target["node"]),
            "link_activity_logit": activity_logit,
            "link_positive_rate": positive_rate,
            "link_rate": torch.sigmoid(activity_logit) * positive_rate,
            "task": torch.zeros_like(target["task"]),
        }

        loss, _ = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=1.0,
            task_loss_weight=0.0,
            hurdle_train_gate_mode="detach",
        )
        loss.backward()

        torch.testing.assert_close(activity_logit.grad, torch.zeros_like(activity_logit))
        self.assertGreater(float(positive_rate.grad.abs().sum()), 0.0)

    def test_hurdle_predicted_gate_power_changes_rate_loss_gate(self):
        from pi_jwm.v8_training import compute_v8_loss

        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0]]]]),
            "link_rate": torch.tensor([[[[2.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        activity_logit = torch.tensor([[[[0.0]]]])
        positive_rate = torch.tensor([[[[4.0]]]])
        outputs = {
            "node": torch.zeros_like(target["node"]),
            "link_activity_logit": activity_logit,
            "link_positive_rate": positive_rate,
            "link_rate": torch.sigmoid(activity_logit) * positive_rate,
            "task": torch.zeros_like(target["task"]),
        }

        default_loss, default_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=1.0,
            task_loss_weight=0.0,
            hurdle_train_gate_mode="predicted",
            hurdle_train_gate_power=1.0,
        )
        soft_loss, soft_parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=1.0,
            task_loss_weight=0.0,
            hurdle_train_gate_mode="predicted",
            hurdle_train_gate_power=0.5,
        )

        self.assertAlmostEqual(default_parts["rate"], 0.0, places=6)
        self.assertGreater(soft_parts["rate"], default_parts["rate"])
        self.assertAlmostEqual(float(default_loss.detach()), 0.0, places=6)
        self.assertGreater(float(soft_loss.detach()), float(default_loss.detach()))

    def test_hurdle_none_rate_loss_keeps_conservative_dual_rate(self):
        from pi_jwm.v8_training import compute_v8_loss

        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0]]]]),
            "link_rate": torch.tensor([[[[2.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        outputs = {
            "node": torch.zeros_like(target["node"]),
            "link_activity_logit": torch.zeros(1, 1, 1, 1),
            "link_positive_rate": torch.tensor([[[[10.0]]]]),
            "link_hurdle_rate": torch.tensor([[[[5.0]]]]),
            "link_rate": torch.tensor([[[[2.0]]]]),
            "task": torch.zeros_like(target["task"]),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=1.0,
            task_loss_weight=0.0,
            hurdle_train_gate_mode="none",
        )

        self.assertEqual(parts["hurdle_train_gate_mode"], "none")
        self.assertAlmostEqual(parts["rate"], 0.0, places=6)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

    def test_best_metric_values_support_link_f1_constraints(self):
        from run_world_model_v8_full_training import compute_best_metric_values

        val_metrics = {
            "active_rate": {"active_rmse": 280.0},
            "link_rate": {"rmse": 95.0},
            "node": {"rmse": 20.0},
            "task": {"rmse": 3.0},
            "activity": {"f1": 0.02, "precision": 0.02, "recall": 0.05},
        }

        values = compute_best_metric_values(
            val_metrics,
            min_f1=0.03,
            max_link_rmse=90.0,
            f1_penalty_weight=1000.0,
            link_penalty_weight=10.0,
        )

        self.assertIn("val_link_f1_constrained_active_rate", values)
        self.assertAlmostEqual(values["val_link_f1_constrained_active_rate"], 340.0)
        self.assertGreater(
            values["val_link_f1_constrained_composite"],
            values["val_composite"],
        )

    def test_positive_rate_specialist_log1p_loss_uses_normalized_rate_space(self):
        from pi_jwm.v8_training import compute_v8_loss

        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[9.0], [0.0]]]]),
            "link_rate_raw": torch.tensor([[[[99.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        positive_rate = torch.tensor([[[[8.0], [50.0]]]], requires_grad=True)
        outputs = {
            "node": torch.zeros_like(target["node"]),
            "link_activity_logit": torch.zeros(1, 1, 2, 1),
            "link_positive_rate": positive_rate,
            "link_rate": torch.zeros(1, 1, 2, 1),
            "task": torch.zeros_like(target["task"]),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=0.0,
            task_loss_weight=0.0,
            positive_rate_specialist_weight=1.0,
            positive_rate_target_mode="log1p",
            positive_rate_loss_mode="mse",
        )
        loss.backward()

        expected = (torch.log1p(torch.tensor(8.0)) - torch.log1p(torch.tensor(9.0))).pow(2)
        self.assertAlmostEqual(parts["positive_rate_specialist"], float(expected), places=6)
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=6)
        self.assertGreater(float(positive_rate.grad.abs().sum()), 0.0)

    def test_positive_rate_specialist_tweedie_loss_uses_active_raw_rates(self):
        from pi_jwm.v8_training import compute_v8_loss

        positive_rate = torch.tensor([[[[8.0], [20.0], [2.0]]]], requires_grad=True)
        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 3, 1),
            "link_positive_rate": positive_rate,
            "link_rate": torch.zeros(1, 1, 3, 1),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [1.0], [0.0]]]]),
            "link_rate": torch.zeros(1, 1, 3, 1),
            "link_rate_raw": torch.tensor([[[[10.0], [15.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        loss, parts = compute_v8_loss(
            outputs,
            target,
            node_loss_weight=0.0,
            activity_loss_weight=0.0,
            rate_loss_weight=0.0,
            task_loss_weight=0.0,
            positive_rate_specialist_weight=1.0,
            positive_rate_target_mode="raw",
            positive_rate_loss_mode="tweedie",
            positive_rate_tweedie_power=1.5,
            positive_rate_raw_stats=(
                np.zeros((1,), dtype=np.float32),
                np.ones((1,), dtype=np.float32),
            ),
        )
        loss.backward()

        y = torch.tensor([10.0, 15.0])
        mu = torch.tensor([8.0, 20.0]).clamp_min(1e-6)
        power = 1.5
        expected = (-y * mu.pow(1.0 - power) / (1.0 - power) + mu.pow(2.0 - power) / (2.0 - power)).mean()
        self.assertAlmostEqual(parts["positive_rate_specialist"], float(expected), places=6)
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=6)
        self.assertGreater(float(positive_rate.grad[..., :2, :].abs().sum()), 0.0)
        self.assertEqual(float(positive_rate.grad[..., 2, :].abs().sum()), 0.0)

if __name__ == "__main__":
    unittest.main()








