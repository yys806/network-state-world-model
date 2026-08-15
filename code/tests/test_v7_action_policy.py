import sys
import unittest
from pathlib import Path

import torch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class V7ActionPolicyTest(unittest.TestCase):
    def test_policy_seed_split_can_match_v10_explicit_split(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v7_action_policy import resolve_policy_seed_splits

        sample_seed = np.repeat(np.arange(60, dtype=np.int32), 2)
        train, val, test, spec = resolve_policy_seed_splits(
            sample_seed,
            train_seeds=[*range(16), *range(20, 60)],
            val_seeds=[16, 17],
            test_seeds=[18, 19],
        )

        self.assertEqual(len(train), 112)
        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual(spec["test_seeds"], [18, 19])

    def test_policy_predicts_future_edge_actions_with_expected_shapes(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=4,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            return_fusion_diagnostics=True,
        )
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 5, 6, 6),
            physical_edge_history=torch.randn(2, 5, 9, 8),
            info_edge_history=torch.randn(2, 5, 9, 5),
            action_history=torch.randn(2, 5, 9, 4),
            future_actions=torch.randn(2, 3, 9, 4),
            task_history=torch.randn(2, 5, 7),
        )

        outputs = V7ActionPolicy(config)(batch)

        self.assertEqual(outputs["action_logit"].shape, (2, 3, 9, 4))
        self.assertEqual(outputs["action_value"].shape, (2, 3, 9, 4))
        self.assertIn("fusion_attention", outputs)
        self.assertEqual(outputs["fusion_attention"].shape, (2, 9, 3, 3))

    def test_policy_can_emit_edge_activity_logits_when_enabled(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=4,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            use_edge_activity_head=True,
        )
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 5, 6, 6),
            physical_edge_history=torch.randn(2, 5, 9, 8),
            info_edge_history=torch.randn(2, 5, 9, 5),
            action_history=torch.randn(2, 5, 9, 4),
            future_actions=torch.randn(2, 3, 9, 4),
            task_history=torch.randn(2, 5, 7),
        )

        outputs = V7ActionPolicy(config)(batch)

        self.assertEqual(outputs["edge_logit"].shape, (2, 3, 9))
        self.assertEqual(outputs["action_logit"].shape, (2, 3, 9, 4))

    def test_policy_can_emit_step_total_log_when_enabled(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=4,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            use_step_total_head=True,
        )
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 5, 6, 6),
            physical_edge_history=torch.randn(2, 5, 9, 8),
            info_edge_history=torch.randn(2, 5, 9, 5),
            action_history=torch.randn(2, 5, 9, 4),
            future_actions=torch.randn(2, 3, 9, 4),
            task_history=torch.randn(2, 5, 7),
        )

        outputs = V7ActionPolicy(config)(batch)

        self.assertEqual(outputs["step_total_log"].shape, (2, 3, 3))

    def test_collate_action_policy_batch_includes_edge_active_target(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from run_v7_action_policy import collate_action_policy_batch

        batch = V6DualGraphBatch(
            node_history=torch.zeros(1, 2, 3),
            physical_edge_history=torch.zeros(1, 2, 4),
            info_edge_history=torch.zeros(1, 2, 5),
            action_history=torch.zeros(1, 2, 6),
            future_actions=torch.zeros(1, 2, 6),
            task_history=torch.zeros(1, 3),
        )
        target = {
            "action_active": torch.tensor([[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]),
            "action_value": torch.zeros(3, 2),
            "action_raw": torch.zeros(3, 2),
            "edge_active": torch.tensor([[1.0, 0.0, 1.0]]),
        }

        _, collated = collate_action_policy_batch([(batch, target), (batch, target)])

        self.assertEqual(tuple(collated["edge_active"].shape), (2, 1, 3))
        self.assertTrue(torch.equal(collated["edge_active"][0], target["edge_active"]))

    def test_continuous_policy_loss_can_use_edge_activity_head(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v7_action_policy import compute_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "action_value": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "edge_logit": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        }
        target = {
            "action_active": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
            "edge_active": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
            "action_value": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "action_raw": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        }

        loss, parts = compute_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(2),
            value_loss_weight=0.0,
            inactive_value_weight=0.0,
            edge_activity_loss_weight=2.0,
            edge_pos_weight=torch.ones(1),
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("edge_activity", parts)
        self.assertGreater(parts["edge_activity"], 0.0)

    def test_discrete_edge_loss_can_weight_active_sample_steps(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import compute_discrete_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 2, 2, 1), dtype=torch.float32),
            "edge_logit": torch.zeros((1, 2, 2), dtype=torch.float32),
            "action_value_bin_logit": torch.zeros((1, 2, 2, 1, 2), dtype=torch.float32),
        }
        target = {
            "action_active": torch.zeros((1, 2, 2, 1), dtype=torch.float32),
            "edge_active": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
            "action_raw": torch.zeros((1, 2, 2, 1), dtype=torch.float32),
            "action_value_bin": torch.full((1, 2, 2, 1), -100, dtype=torch.long),
        }

        _, unweighted = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            vocab={"values": np.zeros((1, 2), dtype=np.float32), "sizes": np.array([2]), "max_bins": 2},
            bin_loss_weight=0.0,
            edge_activity_loss_weight=1.0,
            edge_pos_weight=torch.ones(1),
        )
        _, weighted = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            vocab={"values": np.zeros((1, 2), dtype=np.float32), "sizes": np.array([2]), "max_bins": 2},
            bin_loss_weight=0.0,
            edge_activity_loss_weight=1.0,
            edge_pos_weight=torch.ones(1),
            active_edge_sample_weight=3.0,
        )

        self.assertGreater(weighted["edge_activity"], unweighted["edge_activity"])

    def test_discrete_edge_loss_can_weight_hard_negative_edges(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import compute_discrete_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "edge_logit": torch.tensor([[[4.0, -4.0]]], dtype=torch.float32),
            "action_value_bin_logit": torch.zeros((1, 1, 2, 1, 2), dtype=torch.float32),
        }
        target = {
            "action_active": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "edge_active": torch.zeros((1, 1, 2), dtype=torch.float32),
            "action_raw": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "action_value_bin": torch.full((1, 1, 2, 1), -100, dtype=torch.long),
        }

        _, unweighted = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            vocab={"values": np.zeros((1, 2), dtype=np.float32), "sizes": np.array([2]), "max_bins": 2},
            bin_loss_weight=0.0,
            edge_activity_loss_weight=1.0,
            edge_pos_weight=torch.ones(1),
        )
        _, weighted = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            vocab={"values": np.zeros((1, 2), dtype=np.float32), "sizes": np.array([2]), "max_bins": 2},
            bin_loss_weight=0.0,
            edge_activity_loss_weight=1.0,
            edge_pos_weight=torch.ones(1),
            hard_negative_edge_weight=5.0,
            hard_negative_edge_threshold=0.8,
        )

        self.assertGreater(weighted["edge_activity"], unweighted["edge_activity"])

    def test_edge_tversky_loss_can_penalize_false_positives_more(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import edge_tversky_loss

        edge_logit = torch.tensor([[[2.0, 2.0]]], dtype=torch.float32)
        edge_active = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)

        fp_heavy = edge_tversky_loss(edge_logit, edge_active, alpha=0.8, beta=0.2)
        fn_heavy = edge_tversky_loss(edge_logit, edge_active, alpha=0.2, beta=0.8)

        self.assertGreater(float(fp_heavy), float(fn_heavy))

    def test_discrete_policy_loss_can_add_edge_tversky_auxiliary_loss(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import compute_discrete_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "edge_logit": torch.tensor([[[2.0, 2.0]]], dtype=torch.float32),
            "action_value_bin_logit": torch.zeros((1, 1, 2, 1, 2), dtype=torch.float32),
        }
        target = {
            "action_active": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "edge_active": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
            "action_raw": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "action_value_bin": torch.full((1, 1, 2, 1), -100, dtype=torch.long),
        }

        _, no_tversky = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            vocab={"values": np.zeros((1, 2), dtype=np.float32), "sizes": np.array([2]), "max_bins": 2},
            bin_loss_weight=0.0,
            edge_activity_loss_weight=0.0,
        )
        loss, with_tversky = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            vocab={"values": np.zeros((1, 2), dtype=np.float32), "sizes": np.array([2]), "max_bins": 2},
            bin_loss_weight=0.0,
            edge_activity_loss_weight=0.0,
            edge_tversky_loss_weight=0.5,
            edge_tversky_alpha=0.8,
            edge_tversky_beta=0.2,
        )

        self.assertEqual(no_tversky["edge_tversky"], 0.0)
        self.assertGreater(with_tversky["edge_tversky"], 0.0)
        self.assertGreater(float(loss), with_tversky["activity"])

    def test_evaluate_predictions_uses_edge_prob_when_available(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v7_action_policy import evaluate_predictions

        predictions = {
            "prob": np.zeros((1, 1, 3, 2), dtype=np.float32),
            "active": np.array([[[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32),
            "edge_prob": np.array([[[0.9, 0.1, 0.8]]], dtype=np.float32),
            "value_pred": np.zeros((1, 1, 3, 2), dtype=np.float32),
            "value_true": np.ones((1, 1, 3, 2), dtype=np.float32),
        }

        metrics = evaluate_predictions(predictions, threshold=0.5)

        self.assertEqual(metrics["edge_activity_tp"], 2.0)
        self.assertEqual(metrics["edge_activity_fp"], 0.0)
        self.assertAlmostEqual(metrics["edge_activity_f1"], 1.0, places=6)

    def test_select_policy_score_prefers_edge_f1_when_edge_loss_enabled(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v7_action_policy import select_policy_score

        metrics = {"active_value_rmse": 10.0, "edge_activity_f1": 0.25}

        self.assertAlmostEqual(select_policy_score(metrics, edge_activity_loss_weight=1.0), -0.25)
        self.assertAlmostEqual(select_policy_score(metrics, edge_activity_loss_weight=0.0), 10.0)

    def test_policy_can_emit_discrete_value_bin_logits(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=4,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            value_mode="discrete_bins",
            max_value_bins=5,
        )
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 5, 6, 6),
            physical_edge_history=torch.randn(2, 5, 9, 8),
            info_edge_history=torch.randn(2, 5, 9, 5),
            action_history=torch.randn(2, 5, 9, 4),
            future_actions=torch.randn(2, 3, 9, 4),
            task_history=torch.randn(2, 5, 7),
        )

        outputs = V7ActionPolicy(config)(batch)

        self.assertEqual(outputs["action_logit"].shape, (2, 3, 9, 4))
        self.assertEqual(outputs["action_value_bin_logit"].shape, (2, 3, 9, 4, 5))
        self.assertNotIn("action_value", outputs)

    def test_policy_can_emit_coupled_value_token_logits(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=6,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            value_mode="coupled_tokens",
            value_token_group_count=4,
            max_value_tokens=7,
        )
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 5, 6, 6),
            physical_edge_history=torch.randn(2, 5, 9, 8),
            info_edge_history=torch.randn(2, 5, 9, 5),
            action_history=torch.randn(2, 5, 9, 6),
            future_actions=torch.randn(2, 3, 9, 6),
            task_history=torch.randn(2, 5, 7),
        )

        outputs = V7ActionPolicy(config)(batch)

        self.assertEqual(outputs["action_logit"].shape, (2, 3, 9, 6))
        self.assertEqual(outputs["action_value_token_logit"].shape, (2, 3, 9, 4, 7))
        self.assertNotIn("action_value", outputs)

    def test_policy_can_emit_hierarchical_value_token_logits(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=6,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            value_mode="hierarchical_tokens",
            value_token_group_count=4,
            max_value_count_tokens=3,
            max_value_total_tokens=5,
        )
        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 5, 6, 6),
            physical_edge_history=torch.randn(2, 5, 9, 8),
            info_edge_history=torch.randn(2, 5, 9, 5),
            action_history=torch.randn(2, 5, 9, 6),
            future_actions=torch.randn(2, 3, 9, 6),
            task_history=torch.randn(2, 5, 7),
        )

        outputs = V7ActionPolicy(config)(batch)

        self.assertEqual(outputs["action_logit"].shape, (2, 3, 9, 6))
        self.assertEqual(outputs["action_value_count_logit"].shape, (2, 3, 9, 4, 3))
        self.assertEqual(outputs["action_value_total_logit"].shape, (2, 3, 9, 4, 5))
        self.assertNotIn("action_value", outputs)

    def test_policy_rejects_invalid_fusion_mode(self):
        from pi_jwm.v7_action_policy import V7ActionPolicy, V7ActionPolicyConfig

        config = V7ActionPolicyConfig(
            node_dim=6,
            physical_edge_dim=8,
            info_edge_dim=5,
            action_dim=4,
            task_dim=7,
            hidden_dim=16,
            horizon=3,
            fusion_mode="bad",
        )

        with self.assertRaises(ValueError):
            V7ActionPolicy(config)

    def test_discrete_value_vocab_uses_positive_values_per_action_dim(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import build_value_vocab

        actions = np.zeros((2, 1, 3, 3), dtype=np.float32)
        actions[0, 0, :, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        actions[1, 0, :, 0] = np.array([2.0, 0.0, 1.0], dtype=np.float32)
        actions[:, 0, :, 1] = np.array([[0.0, 5.0, 5.0], [10.0, 0.0, 5.0]], dtype=np.float32)

        vocab = build_value_vocab(actions)

        self.assertEqual(vocab["max_bins"], 2)
        np.testing.assert_allclose(vocab["values"][0][: vocab["sizes"][0]], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(vocab["values"][1][: vocab["sizes"][1]], np.array([5.0, 10.0], dtype=np.float32))
        self.assertEqual(int(vocab["sizes"][2]), 1)
        self.assertEqual(float(vocab["values"][2, 0]), 0.0)

    def test_encode_value_bins_marks_inactive_as_ignore_index(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import build_value_vocab, encode_value_bins

        train_actions = np.zeros((1, 1, 4, 2), dtype=np.float32)
        train_actions[0, 0, :, 0] = np.array([1.0, 2.0, 0.0, 2.0], dtype=np.float32)
        train_actions[0, 0, :, 1] = np.array([5.0, 0.0, 10.0, 5.0], dtype=np.float32)
        vocab = build_value_vocab(train_actions)
        raw = torch.tensor(
            [
                [[1.0, 5.0], [0.0, 0.0], [2.0, 10.0], [2.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        bins = encode_value_bins(raw, vocab)

        expected = torch.tensor([[[0, 0], [-100, -100], [1, 1], [1, -100]]], dtype=torch.long)
        self.assertTrue(torch.equal(bins, expected))

    def test_decode_discrete_value_bins_maps_logits_to_vocab_values(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import decode_value_bins

        vocab = {
            "values": np.array([[1.0, 2.0, 0.0], [5.0, 10.0, 20.0]], dtype=np.float32),
            "sizes": np.array([2, 3], dtype=np.int64),
            "max_bins": 3,
        }
        logits = torch.tensor(
            [
                [[[0.1, 0.9, 9.0], [4.0, 1.0, 0.0]], [[0.8, 0.2, 9.0], [0.1, 0.2, 0.3]]],
            ],
            dtype=torch.float32,
        ).reshape(1, 2, 2, 1, 3)
        logits = torch.cat([logits, logits + 0.5], dim=3)

        decoded = decode_value_bins(logits, vocab)

        self.assertEqual(tuple(decoded.shape), (1, 2, 2, 2))
        self.assertAlmostEqual(float(decoded[0, 0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(decoded[0, 0, 0, 1]), 20.0)
        self.assertAlmostEqual(float(decoded[0, 1, 0, 0]), 1.0)

    def test_activity_value_weight_emphasizes_large_positive_actions_only(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import make_activity_value_weight

        active = torch.tensor([[[[1.0, 1.0, 0.0]]]], dtype=torch.float32)
        raw_action = torch.tensor([[[[1.0, 9.0, 100.0]]]], dtype=torch.float32)

        weight = make_activity_value_weight(
            active,
            raw_action,
            activity_value_weight=2.0,
            max_activity_value_weight=10.0,
        )

        self.assertGreater(float(weight[0, 0, 0, 1]), float(weight[0, 0, 0, 0]))
        self.assertAlmostEqual(float(weight[0, 0, 0, 2]), 1.0, places=6)
        self.assertLessEqual(float(weight.max()), 10.0)

    def test_activity_value_weight_zero_coeff_keeps_unit_weights(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import make_activity_value_weight

        active = torch.ones((1, 1, 2, 1), dtype=torch.float32)
        raw_action = torch.tensor([[[[1.0], [100.0]]]], dtype=torch.float32)

        weight = make_activity_value_weight(
            active,
            raw_action,
            activity_value_weight=0.0,
            max_activity_value_weight=10.0,
        )

        self.assertTrue(torch.equal(weight, torch.ones_like(weight)))

    def test_value_bin_class_weights_are_per_action_dim(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import build_value_vocab, make_value_bin_class_weights

        actions = np.zeros((1, 1, 8, 2), dtype=np.float32)
        actions[0, 0, :, 0] = np.array([1, 1, 1, 1, 2, 0, 0, 0], dtype=np.float32)
        actions[0, 0, :, 1] = np.array([5, 5, 10, 0, 0, 0, 0, 0], dtype=np.float32)
        vocab = build_value_vocab(actions)

        weights = make_value_bin_class_weights(actions, vocab, mode="inverse_sqrt")

        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertLess(float(weights[0, 0]), float(weights[0, 1]))
        self.assertLess(float(weights[1, 0]), float(weights[1, 1]))

    def test_coupled_value_vocab_uses_observed_group_tuples(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import build_coupled_value_vocab

        actions = np.zeros((2, 1, 3, 6), dtype=np.float32)
        actions[0, 0, 0, [1, 2]] = np.array([1.0, 25.0], dtype=np.float32)
        actions[0, 0, 1, [1, 2]] = np.array([2.0, 50.0], dtype=np.float32)
        actions[1, 0, 0, [1, 2]] = np.array([1.0, 25.0], dtype=np.float32)
        actions[0, 0, 2, [3, 4]] = np.array([1.0, 10.0], dtype=np.float32)
        actions[1, 0, 2, [3, 4]] = np.array([2.0, 20.0], dtype=np.float32)

        vocab = build_coupled_value_vocab(actions, groups=[[0], [1, 2], [3, 4], [5]])

        self.assertEqual(vocab["mode"], "coupled_tokens")
        self.assertEqual(vocab["groups"], [[0], [1, 2], [3, 4], [5]])
        self.assertEqual(vocab["max_tokens"], 3)
        self.assertEqual(int(vocab["sizes"][1]), 3)
        np.testing.assert_allclose(vocab["values"][1, 0], np.zeros(6, dtype=np.float32))
        np.testing.assert_allclose(
            vocab["values"][1, 1:3][:, [1, 2]],
            np.array([[1.0, 25.0], [2.0, 50.0]], dtype=np.float32),
        )

    def test_encode_coupled_value_tokens_ignores_inactive_groups(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import build_coupled_value_vocab, encode_coupled_value_tokens

        train_actions = np.zeros((1, 1, 3, 6), dtype=np.float32)
        train_actions[0, 0, 0, [1, 2]] = np.array([1.0, 25.0], dtype=np.float32)
        train_actions[0, 0, 1, [1, 2]] = np.array([2.0, 50.0], dtype=np.float32)
        train_actions[0, 0, 2, [3, 4]] = np.array([1.0, 10.0], dtype=np.float32)
        vocab = build_coupled_value_vocab(train_actions, groups=[[0], [1, 2], [3, 4], [5]])
        raw = torch.zeros((1, 3, 6), dtype=torch.float32)
        raw[0, 0, [1, 2]] = torch.tensor([2.0, 50.0])
        raw[0, 1, [3, 4]] = torch.tensor([1.0, 10.0])

        tokens = encode_coupled_value_tokens(raw, vocab)

        expected = torch.tensor([[[-100, 2, -100, -100], [-100, -100, 1, -100], [-100, -100, -100, -100]]])
        self.assertTrue(torch.equal(tokens, expected))

    def test_decode_coupled_value_tokens_reconstructs_group_dimensions(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import decode_coupled_value_tokens

        vocab = {
            "mode": "coupled_tokens",
            "groups": [[0], [1, 2], [3, 4], [5]],
            "values": np.array(
                [
                    [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0], [0, 1, 25, 0, 0, 0], [0, 2, 50, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 10, 0], [0, 0, 0, 2, 20, 0]],
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2]],
                ],
                dtype=np.float32,
            ),
            "sizes": np.array([3, 3, 3, 3], dtype=np.int64),
            "max_tokens": 3,
            "action_dim": 6,
        }
        logits = torch.zeros((1, 1, 2, 4, 3), dtype=torch.float32)
        logits[0, 0, 0, 1, 2] = 5.0
        logits[0, 0, 0, 2, 1] = 5.0
        logits[0, 0, 1, 1, 1] = 5.0
        logits[0, 0, 1, 3, 2] = 5.0

        decoded = decode_coupled_value_tokens(logits, vocab)

        expected = torch.tensor(
            [[[[0, 2, 50, 1, 10, 0], [0, 1, 25, 0, 0, 2]]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(decoded, expected))

    def test_masked_token_cross_entropy_uses_only_active_groups(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import masked_token_cross_entropy

        vocab = {"sizes": np.array([2, 2], dtype=np.int64)}
        logits = torch.tensor([[[[[0.0, 2.0], [2.0, 0.0]], [[5.0, 0.0], [0.0, 5.0]]]]], dtype=torch.float32)
        target = torch.tensor([[[[1, -100], [-100, 1]]]], dtype=torch.long)

        loss = masked_token_cross_entropy(logits, target, vocab)

        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[0.0, 2.0], [0.0, 5.0]], dtype=torch.float32),
            torch.tensor([1, 1], dtype=torch.long),
        )
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_discrete_policy_loss_can_use_edge_activity_head(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import compute_discrete_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "edge_logit": torch.zeros((1, 1, 2), dtype=torch.float32),
            "action_value_token_logit": torch.tensor(
                [[[[[0.0, 2.0], [2.0, 0.0]], [[5.0, 0.0], [0.0, 5.0]]]]],
                dtype=torch.float32,
            ),
        }
        target = {
            "action_active": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
            "edge_active": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
            "action_value_token": torch.tensor([[[[1, -100], [-100, 1]]]], dtype=torch.long),
            "action_raw": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        }
        vocab = {"sizes": np.array([2, 2], dtype=np.int64)}

        loss, parts = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(2),
            vocab=vocab,
            bin_loss_weight=1.0,
            edge_activity_loss_weight=2.0,
            edge_pos_weight=torch.ones(1),
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("edge_activity", parts)
        self.assertGreater(parts["edge_activity"], 0.0)

    def test_discrete_policy_loss_can_use_step_total_head(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import compute_discrete_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "action_value_token_logit": torch.tensor(
                [[[[[0.0, 2.0], [2.0, 0.0]], [[5.0, 0.0], [0.0, 5.0]]]]],
                dtype=torch.float32,
            ),
            "step_total_log": torch.zeros((1, 1, 3), dtype=torch.float32),
        }
        target = {
            "action_active": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "edge_active": torch.zeros((1, 1, 2), dtype=torch.float32),
            "action_value_token": torch.tensor([[[[1, -100], [-100, 1]]]], dtype=torch.long),
            "action_raw": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            "step_total_log": torch.log1p(torch.tensor([[[2.0, 50.0, 10.0]]], dtype=torch.float32)),
        }
        vocab = {"sizes": np.array([2, 2], dtype=np.int64)}

        loss, parts = compute_discrete_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(2),
            vocab=vocab,
            bin_loss_weight=0.0,
            step_total_loss_weight=2.0,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("step_total", parts)
        self.assertGreater(parts["step_total"], 0.0)

    def test_save_epoch_checkpoint_records_epoch_number(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import tempfile
        from pathlib import Path
        from run_v11_discrete_value_policy import save_epoch_checkpoint

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_epoch_checkpoint(
                checkpoint_dir=Path(tmp_dir),
                fusion_mode="cross_attention",
                epoch=3,
                model_state={"weight": torch.tensor([1.0])},
                checkpoint_payload={"config": {"hidden_dim": 4}, "activity_threshold": 0.9},
            )

            self.assertTrue(path.name.endswith("_epoch_003.pt"))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 3)
            self.assertTrue(torch.equal(payload["model_state"]["weight"], torch.tensor([1.0])))

    def test_discrete_policy_make_config_can_enable_edge_activity_head(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from argparse import Namespace
        from run_v11_discrete_value_policy import make_config

        arrays = {
            "x_node": np.zeros((1, 1, 2, 3), dtype=np.float32),
            "x_link": np.zeros((1, 1, 4, 5), dtype=np.float32),
            "edge_a_hist": np.zeros((1, 1, 4, 6), dtype=np.float32),
            "x_task": np.zeros((1, 1, 7), dtype=np.float32),
            "edge_a_future": np.zeros((1, 3, 4, 6), dtype=np.float32),
        }
        args = Namespace(
            hidden_dim=16,
            fusion_mode="cross_attention",
            fusion_num_heads=4,
            value_mode="coupled_tokens",
            use_edge_activity_head=True,
        )

        config = make_config(arrays, args, max_bins=3, value_token_group_count=4)

        self.assertTrue(config.use_edge_activity_head)
        self.assertEqual(config.value_mode, "coupled_tokens")

    def test_hierarchical_value_vocab_encodes_and_decodes_count_total(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import (
            build_hierarchical_value_vocab,
            decode_hierarchical_value_tokens,
            encode_hierarchical_value_tokens,
        )

        actions = np.zeros((2, 1, 3, 6), dtype=np.float32)
        actions[0, 0, 0, [1, 2]] = np.array([1.0, 25.0], dtype=np.float32)
        actions[0, 0, 1, [1, 2]] = np.array([2.0, 50.0], dtype=np.float32)
        actions[1, 0, 1, [1, 2]] = np.array([1.0, 50.0], dtype=np.float32)
        actions[0, 0, 2, [3, 4]] = np.array([1.0, 10.0], dtype=np.float32)
        vocab = build_hierarchical_value_vocab(actions, groups=[[0], [1, 2], [3, 4], [5]])
        raw = torch.zeros((1, 3, 6), dtype=torch.float32)
        raw[0, 0, [1, 2]] = torch.tensor([2.0, 50.0])
        raw[0, 1, [3, 4]] = torch.tensor([1.0, 10.0])

        encoded = encode_hierarchical_value_tokens(raw, vocab)

        self.assertEqual(vocab["mode"], "hierarchical_tokens")
        self.assertEqual(encoded["count"].tolist(), [[[-100, 2, -100, -100], [-100, -100, 1, -100], [-100, -100, -100, -100]]])
        self.assertEqual(encoded["total"].tolist(), [[[-100, 2, -100, -100], [-100, -100, 1, -100], [-100, -100, -100, -100]]])

        count_logits = torch.zeros((1, 1, 2, 4, int(vocab["max_count_tokens"])), dtype=torch.float32)
        total_logits = torch.zeros((1, 1, 2, 4, int(vocab["max_total_tokens"])), dtype=torch.float32)
        count_logits[0, 0, 0, 1, 2] = 5.0
        total_logits[0, 0, 0, 1, 2] = 5.0
        count_logits[0, 0, 0, 2, 1] = 5.0
        total_logits[0, 0, 0, 2, 1] = 5.0
        count_logits[0, 0, 1, 1, 1] = 5.0
        total_logits[0, 0, 1, 1, 2] = 5.0

        decoded = decode_hierarchical_value_tokens(count_logits, total_logits, vocab)

        expected = torch.tensor([[[[0, 2, 50, 1, 10, 0], [0, 1, 50, 0, 0, 0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(decoded, expected))

    def test_hierarchical_token_loss_supports_total_class_weights(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v11_discrete_value_policy import masked_hierarchical_token_cross_entropy

        vocab = {
            "count_sizes": np.array([2], dtype=np.int64),
            "total_sizes": np.array([3], dtype=np.int64),
        }
        count_logits = torch.tensor([[[[[0.0, 2.0]], [[4.0, 0.0]]]]], dtype=torch.float32)
        total_logits = torch.tensor([[[[[0.0, 1.0, 3.0]], [[3.0, 1.0, 0.0]]]]], dtype=torch.float32)
        target = {
            "count": torch.tensor([[[[1], [-100]]]], dtype=torch.long),
            "total": torch.tensor([[[[2], [-100]]]], dtype=torch.long),
        }
        total_weights = np.array([[1.0, 1.0, 2.0]], dtype=np.float32)

        loss = masked_hierarchical_token_cross_entropy(count_logits, total_logits, target, vocab, total_class_weights=total_weights)

        count_expected = torch.nn.functional.cross_entropy(torch.tensor([[0.0, 2.0]]), torch.tensor([1]))
        total_expected = torch.nn.functional.cross_entropy(torch.tensor([[0.0, 1.0, 3.0]]), torch.tensor([2]), reduction="none") * 2.0
        self.assertAlmostEqual(float(loss), float(count_expected + total_expected.mean()), places=6)

    def test_continuous_policy_loss_can_weight_large_active_values(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v7_action_policy import compute_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "action_value": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
        }
        target = {
            "action_active": torch.ones((1, 1, 2, 1), dtype=torch.float32),
            "action_value": torch.ones((1, 1, 2, 1), dtype=torch.float32),
            "action_raw": torch.tensor([[[[1.0], [100.0]]]], dtype=torch.float32),
        }

        unweighted_loss, unweighted_parts = compute_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            value_loss_weight=1.0,
            inactive_value_weight=0.0,
        )
        weighted_loss, weighted_parts = compute_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            value_loss_weight=1.0,
            inactive_value_weight=0.0,
            active_value_weight=2.0,
            max_active_value_weight=20.0,
        )

        self.assertAlmostEqual(unweighted_parts["active_value"], 1.0, places=6)
        self.assertGreater(weighted_parts["active_value"], unweighted_parts["active_value"])
        self.assertGreater(float(weighted_loss), float(unweighted_loss))

    def test_continuous_policy_activity_loss_can_weight_large_active_values(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_v7_action_policy import compute_policy_loss

        outputs = {
            "action_logit": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "action_value": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
        }
        target = {
            "action_active": torch.ones((1, 1, 2, 1), dtype=torch.float32),
            "action_value": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "action_raw": torch.tensor([[[[1.0], [100.0]]]], dtype=torch.float32),
        }

        _, unweighted_parts = compute_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            value_loss_weight=0.0,
            inactive_value_weight=0.0,
        )
        weighted_loss, weighted_parts = compute_policy_loss(
            outputs,
            target,
            pos_weight=torch.ones(1),
            value_loss_weight=0.0,
            inactive_value_weight=0.0,
            activity_value_weight=2.0,
            max_activity_value_weight=20.0,
        )

        self.assertGreater(weighted_parts["activity"], unweighted_parts["activity"])
        self.assertAlmostEqual(float(weighted_loss), weighted_parts["activity"], places=6)


if __name__ == "__main__":
    unittest.main()
