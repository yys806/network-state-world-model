import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11RolloutRewardTemplateSelectorTest(unittest.TestCase):
    def test_selector_protocol_marks_train_val_pairwise_fit_as_diagnostic(self):
        from compare_v11_rollout_reward_template_selector import make_selector_result_protocol

        train_only = make_selector_result_protocol(
            'rollout_reward_pairwise_gb_identity_gate_p0p3',
            'train',
        )
        train_val = make_selector_result_protocol(
            'rollout_reward_pairwise_gb_identity_gate_p0p3',
            'train_val',
        )

        self.assertEqual(train_only['result_kind'], 'deployable')
        self.assertEqual(train_val['result_kind'], 'test_best_diagnostic')
        self.assertEqual(train_val['fit_splits'], ['train', 'val'])

    def test_sample_active_sse_uses_only_true_active_links(self):
        from compare_v11_rollout_reward_template_selector import sample_active_sse

        predictions = {
            'link_rate_true': np.array([[[[10.0], [20.0]], [[30.0], [40.0]]]], dtype=np.float32),
            'link_rate_pred': np.array([[[[12.0], [999.0]], [[27.0], [41.0]]]], dtype=np.float32),
            'link_activity_true': np.array([[[[1.0], [0.0]], [[1.0], [1.0]]]], dtype=np.float32),
        }

        sse, count = sample_active_sse(predictions)

        self.assertEqual(count.tolist(), [3])
        self.assertAlmostEqual(float(sse[0]), 4.0 + 9.0 + 1.0)

    def test_mix_actions_by_sample_picks_candidate_rows(self):
        from compare_v11_rollout_reward_template_selector import mix_actions_by_sample

        a = np.zeros((3, 2, 1, 4), dtype=np.float32)
        b = np.ones((3, 2, 1, 4), dtype=np.float32)
        c = np.full((3, 2, 1, 4), 2.0, dtype=np.float32)

        mixed = mix_actions_by_sample([a, b, c], np.array([0, 2, 1]))

        self.assertTrue(np.allclose(mixed[0], 0.0))
        self.assertTrue(np.allclose(mixed[1], 2.0))
        self.assertTrue(np.allclose(mixed[2], 1.0))

    def test_make_sample_summary_features_has_expected_shape_and_values(self):
        from compare_v11_rollout_reward_template_selector import make_sample_summary_features

        actions = np.zeros((2, 3, 2, 4), dtype=np.float32)
        actions[0, :, :, 2] = [[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]]
        actions[1, :, :, 2] = [[0.0, 0.0], [5.0, 0.0], [6.0, 7.0]]

        features = make_sample_summary_features(actions)

        self.assertEqual(features.shape, (2, 10))
        self.assertTrue(np.allclose(features[0, :3], [1.0, 5.0, 4.0]))
        self.assertTrue(np.allclose(features[0, 3:6], [1.0, 2.0, 1.0]))
        self.assertAlmostEqual(float(features[1, 6]), (0.0 + 5.0 + 13.0) / 3.0)
        self.assertAlmostEqual(float(features[1, 9]), 2.0)

    def test_make_forecast_selector_features_concatenates_candidate_summaries(self):
        from compare_v11_rollout_reward_template_selector import make_forecast_selector_features

        actions_a = np.zeros((2, 2, 2, 4), dtype=np.float32)
        actions_b = np.zeros((2, 2, 2, 4), dtype=np.float32)
        actions_a[..., 2] = 1.0
        actions_b[..., 2] = 2.0
        pred_a = {
            'link_activity_prob': np.full((2, 2, 2, 1), 0.25, dtype=np.float32),
            'link_rate_pred': np.full((2, 2, 2, 1), 10.0, dtype=np.float32),
        }
        pred_b = {
            'link_activity_prob': np.full((2, 2, 2, 1), 0.5, dtype=np.float32),
            'link_rate_pred': np.full((2, 2, 2, 1), 12.0, dtype=np.float32),
        }

        features = make_forecast_selector_features([actions_a, actions_b], [pred_a, pred_b])

        self.assertEqual(features.shape, (2, 22))
        self.assertAlmostEqual(float(features[0, 0]), 4.0)
        self.assertAlmostEqual(float(features[0, 11]), 8.0)
        self.assertAlmostEqual(float(features[0, 20]), 2.0)
        self.assertAlmostEqual(float(features[0, 21]), 0.25)

    def test_make_candidate_error_features_uses_candidate_major_order(self):
        from compare_v11_rollout_reward_template_selector import make_candidate_error_features

        actions_a = np.zeros((2, 1, 2, 4), dtype=np.float32)
        actions_b = np.zeros((2, 1, 2, 4), dtype=np.float32)
        actions_a[:, 0, :, 2] = [[1.0, 2.0], [3.0, 4.0]]
        actions_b[:, 0, :, 2] = [[5.0, 6.0], [7.0, 8.0]]
        pred_a = {
            'link_activity_prob': np.full((2, 1, 2, 1), 0.25, dtype=np.float32),
            'link_rate_pred': np.full((2, 1, 2, 1), 10.0, dtype=np.float32),
        }
        pred_b = {
            'link_activity_prob': np.full((2, 1, 2, 1), 0.5, dtype=np.float32),
            'link_rate_pred': np.full((2, 1, 2, 1), 12.0, dtype=np.float32),
        }

        features, candidate_idx = make_candidate_error_features([actions_a, actions_b], [pred_a, pred_b])

        self.assertEqual(features.shape, (4, 12))
        self.assertEqual(candidate_idx.tolist(), [0, 0, 1, 1])
        self.assertAlmostEqual(float(features[0, 0]), 3.0)
        self.assertAlmostEqual(float(features[1, 0]), 7.0)
        self.assertAlmostEqual(float(features[2, 0]), 11.0)

    def test_choose_best_single_by_sample_sse_uses_active_weighted_total(self):
        from compare_v11_rollout_reward_template_selector import choose_best_single_by_sample_sse

        sample_sse = np.array(
            [
                [100.0, 200.0, 400.0],
                [900.0, 100.0, 100.0],
            ],
            dtype=np.float32,
        )
        active_count = np.array([1, 9], dtype=np.int64)

        best_idx, rmse = choose_best_single_by_sample_sse(sample_sse, active_count)

        self.assertEqual(best_idx, 1)
        self.assertTrue(np.allclose(rmse, [10.0, np.sqrt(30.0), np.sqrt(50.0)]))

    def test_conservative_improvement_choice_falls_back_to_default(self):
        from compare_v11_rollout_reward_template_selector import conservative_improvement_choice

        predicted_gain = np.array(
            [
                [-5.0, 1.0, 0.0],
                [0.0, 4.0, 9.0],
                [0.0, 6.0, 5.0],
            ],
            dtype=np.float32,
        )

        choice = conservative_improvement_choice(
            predicted_gain,
            default_idx=0,
            min_predicted_gain=5.0,
        )

        self.assertEqual(choice.tolist(), [0, 2, 1])

    def test_predict_improvement_regressor_choice_respects_candidate_major_order(self):
        from compare_v11_rollout_reward_template_selector import predict_improvement_regressor_choice

        class DummyModel:
            def predict(self, features):
                self.shape = features.shape
                return np.array([0.0, 0.0, 1.0, 10.0], dtype=np.float32)

        features = np.zeros((4, 3), dtype=np.float32)
        choice = predict_improvement_regressor_choice(
            DummyModel(),
            features,
            sample_count=2,
            candidate_count=2,
            default_idx=0,
            min_predicted_gain=5.0,
        )

        self.assertEqual(choice.tolist(), [0, 1])

    def test_predict_pairwise_gate_choice_uses_positive_class_probability(self):
        from compare_v11_rollout_reward_template_selector import predict_pairwise_gate_choice

        class DummyGate:
            classes_ = np.array([0, 1], dtype=np.int64)

            def predict_proba(self, features):
                return np.array(
                    [
                        [0.8, 0.2],
                        [0.45, 0.55],
                        [0.1, 0.9],
                    ],
                    dtype=np.float32,
                )

        choice = predict_pairwise_gate_choice(
            DummyGate(),
            np.zeros((3, 2), dtype=np.float32),
            default_idx=4,
            challenger_idx=0,
            min_probability=0.6,
        )

        self.assertEqual(choice.tolist(), [4, 4, 0])

        inverted_choice = predict_pairwise_gate_choice(
            DummyGate(),
            np.zeros((3, 2), dtype=np.float32),
            default_idx=4,
            challenger_idx=0,
            min_probability=0.3,
            invert=True,
        )

        self.assertEqual(inverted_choice.tolist(), [0, 4, 4])

    def test_predict_pairwise_quota_choice_limits_default_fraction(self):
        from compare_v11_rollout_reward_template_selector import predict_pairwise_quota_choice

        class DummyGate:
            classes_ = np.array([0, 1], dtype=np.int64)

            def predict_proba(self, features):
                return np.array(
                    [
                        [0.1, 0.9],
                        [0.3, 0.7],
                        [0.6, 0.4],
                        [0.9, 0.1],
                    ],
                    dtype=np.float32,
                )

        choice = predict_pairwise_quota_choice(
            DummyGate(),
            np.zeros((4, 2), dtype=np.float32),
            default_idx=4,
            challenger_idx=0,
            default_fraction=0.5,
        )

        self.assertEqual(choice.tolist(), [0, 0, 4, 4])

    def test_predict_pairwise_gain_choice_uses_gain_threshold(self):
        from compare_v11_rollout_reward_template_selector import predict_pairwise_gain_choice

        class DummyGain:
            def predict(self, features):
                return np.array([-1.0, 0.4, 2.0], dtype=np.float32)

        choice = predict_pairwise_gain_choice(
            DummyGain(),
            np.zeros((3, 2), dtype=np.float32),
            default_idx=4,
            challenger_idx=0,
            min_gain=0.5,
        )

        self.assertEqual(choice.tolist(), [4, 4, 0])

    def test_choice_rmse_from_sample_sse_uses_per_sample_choice(self):
        from compare_v11_rollout_reward_template_selector import choice_rmse_from_sample_sse

        sample_sse = np.array(
            [
                [100.0, 400.0],
                [900.0, 100.0],
                [25.0, 225.0],
            ],
            dtype=np.float32,
        )
        active_count = np.array([1, 4, 0], dtype=np.int64)

        rmse = choice_rmse_from_sample_sse(sample_sse, active_count, np.array([0, 1, 0]))

        self.assertAlmostEqual(float(rmse), np.sqrt((100.0 + 100.0) / 5.0))

    def test_select_pairwise_threshold_by_rmse_uses_calibration_sse(self):
        from compare_v11_rollout_reward_template_selector import select_pairwise_threshold_by_rmse

        score = np.array([0.1, 0.4, 0.8], dtype=np.float32)
        sample_sse = np.array(
            [
                [100.0, 10.0],
                [100.0, 40.0],
                [5.0, 80.0],
            ],
            dtype=np.float32,
        )
        active_count = np.ones((3,), dtype=np.int64)

        threshold, rmse, choice = select_pairwise_threshold_by_rmse(
            score,
            sample_sse,
            active_count,
            default_idx=1,
            challenger_idx=0,
            thresholds=[0.0, 0.5, 0.9],
        )

        self.assertEqual(float(threshold), 0.5)
        self.assertEqual(choice.tolist(), [1, 1, 0])
        self.assertAlmostEqual(float(rmse), np.sqrt((10.0 + 40.0 + 5.0) / 3.0))

    def test_stratified_fit_calibration_masks_keep_binary_classes_in_fit(self):
        from compare_v11_rollout_reward_template_selector import make_stratified_fit_calibration_masks

        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        active_count = np.ones((6,), dtype=np.int64)

        fit_mask, calibration_mask = make_stratified_fit_calibration_masks(
            labels,
            active_count,
            calibration_fraction=0.5,
            seed=7,
        )

        self.assertFalse(np.any(fit_mask & calibration_mask))
        self.assertEqual(set(labels[fit_mask].tolist()), {0, 1})
        self.assertGreater(int(np.sum(calibration_mask)), 0)

    def test_make_pairwise_gate_features_has_candidate_delta_block(self):
        from compare_v11_rollout_reward_template_selector import make_pairwise_gate_features

        actions_a = np.zeros((2, 1, 2, 4), dtype=np.float32)
        actions_b = np.zeros((2, 1, 2, 4), dtype=np.float32)
        actions_a[..., 2] = 1.0
        actions_b[..., 2] = 3.0
        pred_a = {
            'link_activity_prob': np.full((2, 1, 2, 1), 0.25, dtype=np.float32),
            'link_rate_pred': np.full((2, 1, 2, 1), 10.0, dtype=np.float32),
        }
        pred_b = {
            'link_activity_prob': np.full((2, 1, 2, 1), 0.5, dtype=np.float32),
            'link_rate_pred': np.full((2, 1, 2, 1), 14.0, dtype=np.float32),
        }

        features = make_pairwise_gate_features([actions_a, actions_b], [pred_a, pred_b], 0, 1, 'rich')

        self.assertGreater(features.shape[1], 27)
        self.assertAlmostEqual(float(features[0, 0]), 2.0)
        half = features.shape[1] // 3
        self.assertAlmostEqual(float(features[0, half]), 6.0)
        self.assertAlmostEqual(float(features[0, half * 2]), -4.0)

        global_features = make_pairwise_gate_features([actions_a, actions_b], [pred_a, pred_b], 0, 1, 'global')

        self.assertEqual(global_features.shape, (2, 27))

    def test_stack_pairwise_training_blocks_concatenates_rows(self):
        from compare_v11_rollout_reward_template_selector import stack_pairwise_training_blocks

        features, labels, gains, counts = stack_pairwise_training_blocks(
            [
                (np.ones((2, 3), dtype=np.float32), np.array([1, 0]), np.array([2.0, -1.0]), np.array([5, 6])),
                (np.full((1, 3), 2.0, dtype=np.float32), np.array([1]), np.array([4.0]), np.array([7])),
            ]
        )

        self.assertEqual(features.shape, (3, 3))
        self.assertEqual(labels.tolist(), [1, 0, 1])
        self.assertEqual(gains.tolist(), [2.0, -1.0, 4.0])
        self.assertEqual(counts.tolist(), [5, 6, 7])


if __name__ == '__main__':
    unittest.main()
