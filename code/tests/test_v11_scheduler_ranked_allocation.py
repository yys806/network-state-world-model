import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11SchedulerRankedAllocationTest(unittest.TestCase):
    def test_resolve_torch_device_cpu_and_auto(self):
        from diagnose_v11_scheduler_ranked_allocation import resolve_torch_device

        self.assertEqual(resolve_torch_device('cpu').type, 'cpu')
        self.assertIn(resolve_torch_device('auto').type, ('cpu', 'cuda'))

    def test_resolve_torch_device_rejects_unavailable_cuda(self):
        import torch
        from diagnose_v11_scheduler_ranked_allocation import resolve_torch_device

        if torch.cuda.is_available():
            self.skipTest('cuda is available in this environment')

        with self.assertRaises(RuntimeError):
            resolve_torch_device('cuda')

    def test_link_aware_selection_score_penalizes_large_positive_delta(self):
        from diagnose_v11_scheduler_ranked_allocation import link_aware_selection_score

        active_score = np.array([1.0, 1.0], dtype=np.float32)
        baseline_value = np.array([10.0, 10.0], dtype=np.float32)
        predicted_value = np.array([11.0, 30.0], dtype=np.float32)

        adjusted = link_aware_selection_score(
            active_score,
            baseline_value,
            predicted_value,
            mode='minus_delta',
            risk_weight=0.1,
        )

        self.assertGreater(float(adjusted[0]), float(adjusted[1]))

    def test_blend_policy_prior_selection_score_interpolates_rank_sources(self):
        from diagnose_v11_scheduler_ranked_allocation import blend_policy_prior_selection_score

        ranked_score = np.array([0.1, 0.9], dtype=np.float32)
        policy_prior = np.array([0.8, 0.2], dtype=np.float32)

        base = blend_policy_prior_selection_score(ranked_score, policy_prior, policy_weight=0.0)
        prior = blend_policy_prior_selection_score(ranked_score, policy_prior, policy_weight=1.0)
        mixed = blend_policy_prior_selection_score(ranked_score, policy_prior, policy_weight=0.25)

        self.assertTrue(np.allclose(base, ranked_score))
        self.assertTrue(np.allclose(prior, policy_prior))
        self.assertTrue(np.allclose(mixed, np.array([0.275, 0.725], dtype=np.float32)))

    def test_blend_policy_prior_selection_score_rejects_shape_mismatch(self):
        from diagnose_v11_scheduler_ranked_allocation import blend_policy_prior_selection_score

        with self.assertRaises(ValueError):
            blend_policy_prior_selection_score(
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([0.1], dtype=np.float32),
                policy_weight=0.5,
            )

    def test_edge_value_cap_limits_selected_target_before_step_cap(self):
        from diagnose_v11_scheduler_ranked_allocation import apply_selected_blend_repair_with_step_cap

        actions = np.zeros((1, 2, 1, 6), dtype=np.float32)
        actions[0, 1, 0, 2] = 4.0
        coordinates = np.array([[0, 1, 0]], dtype=np.int64)
        values = np.array([100.0], dtype=np.float32)

        repaired = apply_selected_blend_repair_with_step_cap(
            actions,
            coordinates,
            values,
            np.array([0], dtype=np.int64),
            alpha=1.0,
            step_total_cap_scale=0.0,
            edge_value_cap_scale=2.0,
        )

        self.assertAlmostEqual(float(repaired[0, 1, 0, 2]), 8.0, places=5)

    def test_step_cap_preserves_step0_and_limits_future_rb_total(self):
        from diagnose_v11_scheduler_ranked_allocation import apply_selected_blend_repair_with_step_cap

        actions = np.zeros((1, 3, 2, 6), dtype=np.float32)
        actions[0, 0, :, 2] = [3.0, 5.0]
        actions[0, 1, :, 2] = [2.0, 6.0]
        coordinates = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)
        values = np.array([20.0, 20.0], dtype=np.float32)
        selected = np.array([0, 1], dtype=np.int64)

        repaired = apply_selected_blend_repair_with_step_cap(
            actions,
            coordinates,
            values,
            selected,
            alpha=1.0,
            step_total_cap_scale=1.25,
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], actions[0, 0, :, 2]))
        self.assertAlmostEqual(float(np.sum(repaired[0, 1, :, 2])), 10.0, places=5)

    def test_step_cap_does_not_create_new_rb_support(self):
        from diagnose_v11_scheduler_ranked_allocation import apply_selected_blend_repair_with_step_cap

        actions = np.zeros((1, 2, 3, 6), dtype=np.float32)
        actions[0, 1, 0, 2] = 4.0
        coordinates = np.array([[0, 1, 0]], dtype=np.int64)
        values = np.array([12.0], dtype=np.float32)
        selected = np.array([0], dtype=np.int64)

        repaired = apply_selected_blend_repair_with_step_cap(
            actions,
            coordinates,
            values,
            selected,
            alpha=1.0,
            step_total_cap_scale=2.0,
        )

        self.assertGreater(float(repaired[0, 1, 0, 2]), 0.0)
        self.assertTrue(np.allclose(repaired[0, 1, 1:, 2], 0.0))

    def test_nonpositive_cap_scale_leaves_uncapped_repair(self):
        from diagnose_v11_scheduler_ranked_allocation import apply_selected_blend_repair_with_step_cap

        actions = np.zeros((1, 2, 1, 6), dtype=np.float32)
        actions[0, 1, 0, 2] = 4.0
        coordinates = np.array([[0, 1, 0]], dtype=np.int64)
        values = np.array([12.0], dtype=np.float32)

        repaired = apply_selected_blend_repair_with_step_cap(
            actions,
            coordinates,
            values,
            np.array([0], dtype=np.int64),
            alpha=1.0,
            step_total_cap_scale=0.0,
        )

        self.assertAlmostEqual(float(repaired[0, 1, 0, 2]), 12.0, places=5)

    def test_conservative_value_target_lcb_uses_tree_uncertainty(self):
        from diagnose_v11_scheduler_ranked_allocation import predict_conservative_value_target

        class Tree:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def predict(self, features):
                return self.values

        class Forest:
            estimators_ = [Tree([8.0, 20.0]), Tree([10.0, 30.0]), Tree([12.0, 40.0])]

            def predict(self, features):
                return np.array([10.0, 30.0], dtype=np.float32)

        features = np.zeros((2, 3), dtype=np.float32)
        predicted = predict_conservative_value_target(Forest(), features, mode='lcb', beta=1.0)

        self.assertLess(float(predicted[0]), 10.0)
        self.assertLess(float(predicted[1]), 30.0)
        self.assertGreater(float(predicted[0]), 0.0)

    def test_conservative_value_target_quantile_uses_tree_distribution(self):
        from diagnose_v11_scheduler_ranked_allocation import predict_conservative_value_target

        class Tree:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def predict(self, features):
                return self.values

        class Forest:
            estimators_ = [Tree([1.0]), Tree([5.0]), Tree([9.0])]

            def predict(self, features):
                return np.array([5.0], dtype=np.float32)

        features = np.zeros((1, 2), dtype=np.float32)
        predicted = predict_conservative_value_target(Forest(), features, mode='q25', beta=0.0)

        self.assertAlmostEqual(float(predicted[0]), 3.0, places=5)

    def test_conservative_value_target_non_ensemble_falls_back_to_mean(self):
        from diagnose_v11_scheduler_ranked_allocation import predict_conservative_value_target

        class Model:
            def predict(self, features):
                return np.array([2.0, 4.0], dtype=np.float32)

        features = np.zeros((2, 2), dtype=np.float32)
        predicted = predict_conservative_value_target(Model(), features, mode='lcb', beta=2.0)

        self.assertTrue(np.allclose(predicted, np.array([2.0, 4.0], dtype=np.float32)))

    def test_make_selector_target_supports_value_binary_and_log(self):
        from diagnose_v11_scheduler_ranked_allocation import make_selector_target

        gradient_score = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        true_value = np.array([0.0, 0.5, 3.0], dtype=np.float32)

        binary = make_selector_target('value_binary', gradient_score, true_value, min_effective_value=1.0)
        value_log = make_selector_target('value_log', gradient_score, true_value, min_effective_value=1.0)
        gradient = make_selector_target('gradient', gradient_score, true_value, min_effective_value=1.0)

        self.assertTrue(np.allclose(binary, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(value_log, np.log1p(true_value)))
        self.assertTrue(np.allclose(gradient, gradient_score))

    def test_ranked_step_redistribution_moves_mass_to_higher_value_edge(self):
        from diagnose_v11_scheduler_ranked_allocation import apply_ranked_step_redistribution

        actions = np.zeros((1, 2, 2, 6), dtype=np.float32)
        actions[0, 0, :, 2] = [3.0, 5.0]
        actions[0, 1, :, 2] = [4.0, 6.0]
        coordinates = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)
        values = np.array([1.0, 9.0], dtype=np.float32)

        repaired = apply_ranked_step_redistribution(
            actions,
            coordinates,
            values,
            np.array([0, 1], dtype=np.int64),
            alpha=1.0,
            step_total_scale=1.0,
            edge_value_cap_scale=0.0,
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], actions[0, 0, :, 2]))
        self.assertAlmostEqual(float(np.sum(repaired[0, 1, :, 2])), 10.0, places=5)
        self.assertGreater(float(repaired[0, 1, 1, 2]), float(actions[0, 1, 1, 2]))
        self.assertLess(float(repaired[0, 1, 0, 2]), float(actions[0, 1, 0, 2]))

    def test_ranked_step_redistribution_preserves_unselected_and_support(self):
        from diagnose_v11_scheduler_ranked_allocation import apply_ranked_step_redistribution

        actions = np.zeros((1, 2, 3, 6), dtype=np.float32)
        actions[0, 1, :, 2] = [4.0, 6.0, 0.0]
        coordinates = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)
        values = np.array([1.0, 9.0], dtype=np.float32)

        repaired = apply_ranked_step_redistribution(
            actions,
            coordinates,
            values,
            np.array([1], dtype=np.int64),
            alpha=1.0,
            step_total_scale=1.5,
            edge_value_cap_scale=0.0,
        )

        self.assertAlmostEqual(float(repaired[0, 1, 0, 2]), 4.0, places=5)
        self.assertGreater(float(repaired[0, 1, 1, 2]), 6.0)
        self.assertAlmostEqual(float(repaired[0, 1, 2, 2]), 0.0, places=5)


    def test_active_rate_row_includes_activity_f1_metrics(self):
        from run_v11_rb_total_repair import active_rate_row

        predictions = {
            'link_rate_true': np.array([[[[1.0], [0.0], [2.0], [0.0]]]], dtype=np.float32),
            'link_rate_pred': np.array([[[[1.0], [0.0], [2.5], [0.0]]]], dtype=np.float32),
            'link_activity_true': np.array([[[[1.0], [0.0], [1.0], [0.0]]]], dtype=np.float32),
            'link_activity_prob': np.array([[[[0.9], [0.8], [0.2], [0.1]]]], dtype=np.float32),
        }

        row = active_rate_row('candidate', 'val', predictions, float('nan'))

        self.assertAlmostEqual(float(row['activity_precision']), 0.5, places=6)
        self.assertAlmostEqual(float(row['activity_recall']), 0.5, places=6)
        self.assertAlmostEqual(float(row['activity_f1']), 0.5, places=6)
        self.assertEqual(int(row['activity_tp']), 1)
        self.assertEqual(int(row['activity_fp']), 1)
        self.assertEqual(int(row['activity_fn']), 1)
        self.assertEqual(int(row['activity_tn']), 1)

    def test_xgb_regressor_factory_fits_when_available(self):
        import importlib.util
        if importlib.util.find_spec('xgboost') is None:
            self.skipTest('xgboost is not installed')

        from diagnose_v11_rb_total_latent_identifiability import build_models

        factories = build_models(seed=11, rf_trees=4)
        self.assertIn('xgb', factories)
        model = factories['xgb']()
        features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        target = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        model.fit(features, target)
        pred = np.asarray(model.predict(features), dtype=np.float32)

        self.assertEqual(pred.shape, (4,))
        self.assertTrue(np.all(np.isfinite(pred)))

if __name__ == '__main__':
    unittest.main()
