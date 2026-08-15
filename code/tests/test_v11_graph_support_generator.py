import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11GraphSupportGeneratorTest(unittest.TestCase):
    def test_limit_indices_window_supports_offset_and_shuffle(self):
        from compare_v11_graph_support_generator import limit_indices_window

        indices = np.arange(10, dtype=np.int64)

        self.assertEqual(limit_indices_window(indices, limit=4, start=3).tolist(), [3, 4, 5, 6])
        self.assertEqual(limit_indices_window(indices, limit=4, start=8).tolist(), [8, 9])

        shuffled_a = limit_indices_window(indices, limit=4, start=0, shuffle_seed=7)
        shuffled_b = limit_indices_window(indices, limit=4, start=0, shuffle_seed=7)

        self.assertEqual(shuffled_a.tolist(), shuffled_b.tolist())
        self.assertNotEqual(shuffled_a.tolist(), [0, 1, 2, 3])

    def test_make_all_edge_coordinates_includes_inactive_edges(self):
        from compare_v11_graph_support_generator import make_all_edge_coordinates

        actions = np.zeros((2, 3, 4, 6), dtype=np.float32)

        coords = make_all_edge_coordinates(actions, steps=(1, 2))

        self.assertEqual(coords.shape, (16, 3))
        self.assertTrue(np.array_equal(coords[0], np.array([0, 1, 0])))
        self.assertTrue(np.array_equal(coords[-1], np.array([1, 2, 3])))

    def test_apply_support_repair_can_create_new_rb_support(self):
        from compare_v11_graph_support_generator import apply_support_generator_repair

        actions = np.zeros((1, 3, 3, 6), dtype=np.float32)
        actions[0, 0, :, 2] = [3.0, 4.0, 5.0]
        actions[0, 1, 0, 2] = 10.0
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]], dtype=np.int64)
        values = np.array([10.0, 6.0, 8.0], dtype=np.float32)
        selected = np.array([1], dtype=np.int64)

        repaired = apply_support_generator_repair(
            actions,
            coords,
            values,
            selected,
            alpha=1.0,
            step_total_cap_scale=1.5,
            edge_value_cap_scale=2.0,
            new_edge_value_cap=7.0,
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], actions[0, 0, :, 2]))
        self.assertGreater(float(repaired[0, 1, 1, 2]), 0.0)
        self.assertAlmostEqual(float(repaired[0, 1, 2, 2]), 0.0, places=5)
        self.assertLessEqual(float(np.sum(repaired[0, 1, :, 2])), 15.0 + 1e-5)

    def test_support_repair_caps_new_edge_value_before_step_scaling(self):
        from compare_v11_graph_support_generator import apply_support_generator_repair

        actions = np.zeros((1, 2, 2, 6), dtype=np.float32)
        actions[0, 1, 0, 2] = 10.0
        coords = np.array([[0, 1, 1]], dtype=np.int64)
        values = np.array([100.0], dtype=np.float32)

        repaired = apply_support_generator_repair(
            actions,
            coords,
            values,
            np.array([0], dtype=np.int64),
            alpha=1.0,
            step_total_cap_scale=0.0,
            edge_value_cap_scale=2.0,
            new_edge_value_cap=3.0,
        )

        self.assertAlmostEqual(float(repaired[0, 1, 1, 2]), 3.0, places=5)

    def test_support_reconstruction_clears_unselected_old_rb_support(self):
        from compare_v11_graph_support_generator import apply_support_generator_reconstruction

        actions = np.zeros((1, 2, 3, 6), dtype=np.float32)
        actions[0, 0, :, 2] = [1.0, 2.0, 3.0]
        actions[0, 1, 0, 2] = 10.0
        actions[0, 1, 1, 2] = 5.0
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]], dtype=np.int64)
        values = np.array([8.0, 5.0, 4.0], dtype=np.float32)

        repaired = apply_support_generator_reconstruction(
            actions,
            coords,
            values,
            np.array([0, 2], dtype=np.int64),
            alpha=1.0,
            step_total_cap_scale=1.0,
            edge_value_cap_scale=2.0,
            new_edge_value_cap=4.0,
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], actions[0, 0, :, 2]))
        self.assertGreater(float(repaired[0, 1, 0, 2]), 0.0)
        self.assertAlmostEqual(float(repaired[0, 1, 1, 2]), 0.0, places=5)
        self.assertGreater(float(repaired[0, 1, 2, 2]), 0.0)
        self.assertLessEqual(float(np.sum(repaired[0, 1, :, 2])), 15.0 + 1e-5)

    def test_hard_negative_rows_prioritize_high_scored_false_positives(self):
        from compare_v11_graph_support_generator import downsample_hard_negative_support_training_rows

        labels = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
        oracle_scores = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        warm_scores = np.array([0.9, 0.99, 0.8, 0.1, 0.05, 0.01], dtype=np.float32)

        rows = downsample_hard_negative_support_training_rows(
            labels,
            oracle_scores,
            warm_scores,
            max_rows=3,
            hard_fraction=1.0,
            seed=7,
        )

        self.assertIn(0, rows.tolist())
        self.assertIn(1, rows.tolist())
        self.assertIn(2, rows.tolist())

    def test_baseline_active_count_selection_limits_group_size(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
            ],
            dtype=np.int64,
        )
        score = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        baseline = np.array([0.0, 3.0, 0.0, 2.0, 1.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=3,
            group_mode='baseline_active_count',
            baseline_value=baseline,
        )

        self.assertEqual(selected.tolist(), [1, 3, 4])

    def test_baseline_active_plus_topk_adds_extra_high_ranked_candidates(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
            ],
            dtype=np.int64,
        )
        score = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.99], dtype=np.float32)
        baseline = np.array([0.0, 3.0, 0.0, 2.0, 1.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=1,
            group_mode='baseline_active_plus_topk',
            baseline_value=baseline,
        )

        self.assertEqual(selected.tolist(), [0, 1, 3, 4, 5])

    def test_baseline_active_plus_new_topk_only_adds_inactive_candidates(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=np.int64,
        )
        score = np.array([0.99, 0.1, 0.8, 0.7, 0.6], dtype=np.float32)
        baseline = np.array([5.0, 4.0, 0.0, 0.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=2,
            group_mode='baseline_active_plus_new_topk',
            baseline_value=baseline,
        )

        self.assertEqual(selected.tolist(), [0, 1, 2, 3])

    def test_baseline_active_plus_new_topk_threshold_prunes_low_score_inactive_candidates(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=np.int64,
        )
        score = np.array([0.99, 0.1, 0.8, 0.45, 0.2], dtype=np.float32)
        baseline = np.array([5.0, 4.0, 0.0, 0.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=3,
            group_mode='baseline_active_plus_new_topk_threshold',
            baseline_value=baseline,
            threshold=0.5,
        )

        self.assertEqual(selected.tolist(), [0, 1, 2])

    def test_baseline_active_topk_plus_new_topk_prunes_stale_active_candidates(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=np.int64,
        )
        score = np.array([0.2, 0.9, 0.8, 0.7, 0.1], dtype=np.float32)
        baseline = np.array([5.0, 4.0, 3.0, 0.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=2,
            group_mode='baseline_active_topk_plus_new_topk',
            baseline_value=baseline,
        )

        self.assertEqual(selected.tolist(), [1, 2, 3, 4])

    def test_baseline_active_value_topk_plus_new_topk_keeps_largest_baseline_values(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=np.int64,
        )
        score = np.array([0.2, 0.9, 0.8, 0.7, 0.1], dtype=np.float32)
        baseline = np.array([5.0, 4.0, 3.0, 0.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=2,
            group_mode='baseline_active_value_topk_plus_new_topk',
            baseline_value=baseline,
        )

        self.assertEqual(selected.tolist(), [0, 1, 3, 4])

    def test_baseline_active_value_topk_plus_new_fixed_parses_separate_counts(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 1, 5],
            ],
            dtype=np.int64,
        )
        score = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        baseline = np.array([9.0, 2.0, 7.0, 0.0, 0.0, 0.0], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=0,
            group_mode='baseline_active_value_top2_plus_new3',
            baseline_value=baseline,
        )

        self.assertEqual(selected.tolist(), [0, 2, 3, 4, 5])

    def test_support_threshold_selection_filters_low_scores_after_topk(self):
        from compare_v11_graph_support_generator import select_support_indices

        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]], dtype=np.int64)
        score = np.array([0.7, 0.4, 0.2], dtype=np.float32)

        selected = select_support_indices(
            coords,
            score,
            top_k=3,
            group_mode='support_threshold',
            threshold=0.5,
        )

        self.assertEqual(selected.tolist(), [0])

    def test_active_keep_score_mode_maps_to_base_mode(self):
        from compare_v11_graph_support_generator import active_keep_base_score_mode, is_active_keep_score_mode

        self.assertTrue(is_active_keep_score_mode('active_keep_support_value'))
        self.assertEqual(active_keep_base_score_mode('active_keep_support_value'), 'support_value')
        self.assertEqual(active_keep_base_score_mode('support_value'), 'support_value')

    def test_rate_context_selection_score_prefers_high_rate_edges(self):
        from compare_v11_graph_support_generator import _selection_score

        support = np.array([0.5, 0.5], dtype=np.float32)
        baseline = np.zeros((2,), dtype=np.float32)
        pred = np.array([10.0, 10.0], dtype=np.float32)
        oracle = np.zeros((2,), dtype=np.float32)
        edge_features = np.zeros((2, 6), dtype=np.float32)
        edge_features[0, -3:] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        edge_features[1, -3:] = np.array([2.0, 1.0, 0.5], dtype=np.float32)

        score = _selection_score(
            'support_rate_value',
            support,
            baseline,
            pred,
            oracle,
            risk_weight=0.0,
            edge_features=edge_features,
        )

        self.assertGreater(float(score[1]), float(score[0]))

    def test_group_rank_targets_normalize_within_sample_step(self):
        from compare_v11_graph_support_generator import make_group_rank_targets

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
        oracle = np.array([0.0, 2.0, 4.0, 0.0, 3.0], dtype=np.float32)
        values = np.ones((5,), dtype=np.float32)

        targets = make_group_rank_targets(coords, oracle, values, mode='gain_norm')

        self.assertTrue(np.allclose(targets[:3], np.array([0.0, 0.5, 1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(targets[3:], np.array([0.0, 1.0], dtype=np.float32)))

    def test_pairwise_preference_examples_add_symmetric_differences(self):
        from compare_v11_graph_support_generator import make_pairwise_preference_examples

        features = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]], dtype=np.int64)
        targets = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        diffs, labels = make_pairwise_preference_examples(
            features,
            coords,
            targets,
            max_pairs=2,
            negatives_per_positive=2,
            seed=3,
        )

        self.assertEqual(diffs.shape, (4, 2))
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])
        self.assertTrue(np.allclose(diffs[0], -diffs[1]))
        self.assertTrue(np.allclose(diffs[2], -diffs[3]))

    def test_order_rows_by_group_returns_contiguous_group_sizes(self):
        from compare_v11_graph_support_generator import order_rows_by_group

        coords = np.array(
            [
                [0, 2, 0],
                [0, 1, 0],
                [0, 2, 1],
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.int64,
        )
        rows = np.array([4, 0, 3, 1, 2], dtype=np.int64)

        ordered, group_sizes = order_rows_by_group(coords, rows)

        self.assertEqual(ordered.tolist(), [1, 4, 0, 2, 3])
        self.assertEqual(group_sizes.tolist(), [2, 2, 1])
        self.assertEqual(int(np.sum(group_sizes)), int(rows.size))

    def test_xgb_ranker_smoke_fit_predicts_finite_scores_when_available(self):
        import importlib.util

        if importlib.util.find_spec('xgboost') is None:
            self.skipTest('xgboost is not installed')

        from compare_v11_graph_support_generator import fit_xgb_ranker, predict_support_score

        features = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1]], dtype=np.int64)
        targets = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        rows = np.arange(4, dtype=np.int64)

        model = fit_xgb_ranker(features, coords, targets, rows, seed=3, trees=4)
        scores = predict_support_score(model, features)

        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_oracle_mass_at_selected_budget_uses_same_group_budget(self):
        from compare_v11_graph_support_generator import oracle_mass_at_selected_budget

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
        oracle = np.array([1.0, 5.0, 3.0, 2.0, 7.0], dtype=np.float32)
        selected = np.array([0, 2, 3], dtype=np.int64)

        budget_mass = oracle_mass_at_selected_budget(coords, oracle, selected)

        # Same selected budget means two edges for step 1 and one edge for step 2:
        # top2(5,3,1)=8 and top1(7,2)=7.
        self.assertAlmostEqual(float(budget_mass), 15.0, places=5)

    def test_support_selection_diagnostics_reports_mass_ratio_and_recall(self):
        from compare_v11_graph_support_generator import support_selection_diagnostics

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
            ],
            dtype=np.int64,
        )
        oracle = np.array([1.0, 5.0, 3.0, 4.0], dtype=np.float32)
        labels = np.array([0, 1, 1, 1], dtype=np.int64)
        selected = np.array([0, 2], dtype=np.int64)

        diagnostics = support_selection_diagnostics(coords, oracle, labels, selected)

        self.assertEqual(diagnostics['selected_count'], 2)
        self.assertEqual(diagnostics['selected_positive_count'], 1)
        self.assertAlmostEqual(diagnostics['selected_positive_precision'], 0.5, places=5)
        self.assertAlmostEqual(diagnostics['selected_positive_recall'], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(diagnostics['selected_oracle_mass'], 4.0, places=5)
        self.assertAlmostEqual(diagnostics['oracle_budget_mass'], 8.0, places=5)
        self.assertAlmostEqual(diagnostics['oracle_mass_ratio_at_budget'], 4.0 / 8.0, places=5)

    def test_repair_values_for_source_selects_predicted_or_true_values(self):
        from compare_v11_graph_support_generator import repair_values_for_source

        predicted = np.array([1.0, 2.0], dtype=np.float32)
        truth = np.array([3.0, 4.0], dtype=np.float32)

        self.assertTrue(np.allclose(repair_values_for_source(predicted, truth, 'pred_value'), predicted))
        self.assertTrue(np.allclose(repair_values_for_source(predicted, truth, 'true_value'), truth))
        self.assertTrue(np.allclose(
            repair_values_for_source(predicted, truth, 'train_positive_q75', {'train_positive_q75': 7.5}),
            np.array([7.5, 7.5], dtype=np.float32),
        ))
        self.assertTrue(np.allclose(
            repair_values_for_source(
                predicted,
                truth,
                'support_score_q75',
                {'train_positive_q75': 50.0},
                np.array([0.2, 0.8], dtype=np.float32),
            ),
            np.array([10.0, 40.0], dtype=np.float32),
        ))
        self.assertTrue(np.allclose(
            repair_values_for_source(
                predicted,
                truth,
                'pred_value_gate_0p5',
                {'train_positive_q75': 50.0},
                np.array([0.2, 0.8], dtype=np.float32),
            ),
            np.array([0.0, 2.0], dtype=np.float32),
        ))

        with self.assertRaises(ValueError):
            repair_values_for_source(predicted, truth, 'missing')

    def test_candidate_family_marks_true_value_source_as_diagnostic(self):
        from compare_v11_graph_support_generator import candidate_family_for_score_and_value_source

        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'pred_value'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'true_value'),
            'diagnostic_only',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'train_positive_q75'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'support_score_q75'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'pred_value_gate_0p5'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'candidate_value'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'positive_candidate_value'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'candidate_bin_expected'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'positive_candidate_bin_conservative'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'all_candidate_bin_expected'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'selected_gate_q75_t0p05'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'structured_stepwise_pred_reconstruct'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__support_value', 'support_value', 'structured_branch_value_pred_reconstruct'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('xgb_rank__oracle_support', 'oracle_support', 'pred_value'),
            'diagnostic_only',
        )
        self.assertEqual(
            candidate_family_for_score_and_value_source('diagnostic_only__oracle_support', 'oracle_support', 'pred_value'),
            'diagnostic_only',
        )

    def test_make_value_training_rows_and_weights_supports_all_weighted_mode(self):
        from compare_v11_graph_support_generator import make_value_training_rows_and_weights

        labels = np.array([0, 1, 0, 0], dtype=np.int64)
        values = np.array([0.0, 5.0, 2.0, 0.0], dtype=np.float32)

        positive_rows, positive_weights = make_value_training_rows_and_weights(
            labels,
            values,
            min_effective_value=1.0,
            mode='positive_only',
            positive_weight=4.0,
        )
        all_rows, all_weights = make_value_training_rows_and_weights(
            labels,
            values,
            min_effective_value=1.0,
            mode='all_weighted',
            positive_weight=4.0,
        )

        self.assertEqual(positive_rows.tolist(), [1, 2])
        self.assertTrue(np.allclose(positive_weights, np.array([1.0, 1.0], dtype=np.float32)))
        self.assertEqual(all_rows.tolist(), [0, 1, 2, 3])
        self.assertTrue(np.allclose(all_weights, np.array([1.0, 4.0, 4.0, 1.0], dtype=np.float32)))

    def test_make_candidate_value_features_adds_support_score_and_group_rank(self):
        from compare_v11_graph_support_generator import make_candidate_value_features

        edge_features = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0]], dtype=np.int64)
        support_score = np.array([0.2, 0.8, 0.5, 0.1], dtype=np.float32)
        selection_score = np.array([2.0, 9.0, 5.0, 7.0], dtype=np.float32)
        baseline = np.array([0.0, 10.0, 5.0, 3.0], dtype=np.float32)

        features = make_candidate_value_features(edge_features, coords, support_score, selection_score, baseline)

        self.assertEqual(features.shape, (4, 6))
        self.assertAlmostEqual(float(features[1, 1]), 0.8, places=5)
        self.assertAlmostEqual(float(features[1, 2]), 9.0, places=5)
        self.assertAlmostEqual(float(features[1, 3]), 10.0, places=5)
        self.assertAlmostEqual(float(features[1, 5]), 1.0, places=5)
        self.assertAlmostEqual(float(features[0, 5]), 0.0, places=5)

    def test_fit_candidate_value_regressor_handles_empty_selected_rows(self):
        from compare_v11_graph_support_generator import fit_candidate_value_regressor

        model = fit_candidate_value_regressor(
            np.zeros((3, 2), dtype=np.float32),
            np.array([0.0, 5.0, 0.0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            min_effective_value=1.0,
            positive_weight=4.0,
            seed=3,
        )

        pred = model.predict(np.zeros((2, 2), dtype=np.float32))

        self.assertTrue(np.allclose(pred, np.zeros((2,), dtype=np.float32)))

    def test_candidate_value_codebook_includes_zero_and_positive_bins(self):
        from compare_v11_graph_support_generator import build_candidate_value_codebook, encode_candidate_value_bins

        centers = build_candidate_value_codebook(
            np.array([0.0, 25.0, 25.0, 50.0, 75.0], dtype=np.float32),
            min_effective_value=1.0,
            positive_bin_count=3,
        )
        labels = encode_candidate_value_bins(
            np.array([0.0, 20.0, 60.0], dtype=np.float32),
            centers,
            min_effective_value=1.0,
        )

        self.assertAlmostEqual(float(centers[0]), 0.0, places=5)
        self.assertGreaterEqual(centers.shape[0], 3)
        self.assertEqual(int(labels[0]), 0)
        self.assertGreater(int(labels[1]), 0)
        self.assertGreater(int(labels[2]), 0)

    def test_candidate_value_bin_classifier_decodes_expected_and_conservative(self):
        from compare_v11_graph_support_generator import (
            decode_candidate_value_bins,
            fit_candidate_value_bin_classifier,
            predict_candidate_value_bin_probabilities,
        )

        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [2.0, 0.0],
                [2.1, 0.0],
                [3.0, 0.0],
                [3.1, 0.0],
            ],
            dtype=np.float32,
        )
        true_values = np.array([0.0, 0.0, 25.0, 25.0, 50.0, 50.0, 0.0, 0.0], dtype=np.float32)
        labels = (true_values > 0.0).astype(np.int64)
        rows = np.arange(features.shape[0], dtype=np.int64)

        model = fit_candidate_value_bin_classifier(
            features,
            true_values,
            labels,
            rows,
            min_effective_value=1.0,
            positive_weight=4.0,
            positive_bin_count=2,
            seed=5,
        )
        probabilities = predict_candidate_value_bin_probabilities(model, features[:3])
        expected = decode_candidate_value_bins(model, features[:3], 'expected')
        conservative = decode_candidate_value_bins(model, features[:3], 'conservative')

        self.assertEqual(probabilities.shape[0], 3)
        self.assertEqual(probabilities.shape[1], model.centers.shape[0])
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), np.ones((3,), dtype=np.float32)))
        self.assertTrue(np.all(expected >= 0.0))
        self.assertTrue(np.all(conservative >= 0.0))

    def test_selected_gate_source_parser_and_classifier(self):
        from compare_v11_graph_support_generator import (
            fit_selected_candidate_gate,
            parse_selected_gate_source,
            predict_support_score,
        )

        lookup_key, threshold = parse_selected_gate_source('selected_gate_q75_t0p05')
        self.assertEqual(lookup_key, 'train_positive_q75')
        self.assertAlmostEqual(threshold, 0.05, places=5)

        features = np.array([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1], [3.0], [3.1]], dtype=np.float32)
        values = np.array([0.0, 0.0, 25.0, 25.0, 0.0, 0.0, 50.0, 50.0], dtype=np.float32)
        labels = (values > 0.0).astype(np.int64)
        rows = np.arange(features.shape[0], dtype=np.int64)

        model = fit_selected_candidate_gate(
            features,
            values,
            labels,
            rows,
            min_effective_value=1.0,
            seed=7,
        )
        probability = predict_support_score(model, features)

        self.assertEqual(probability.shape, (8,))
        self.assertTrue(np.all(probability >= 0.0))
        self.assertTrue(np.all(probability <= 1.0))

    def test_step_budget_features_and_targets_aggregate_by_sample_step(self):
        from compare_v11_graph_support_generator import make_step_budget_features, make_step_budget_targets

        edge_features = np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
            ],
            dtype=np.float32,
        )
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1]], dtype=np.int64)
        support = np.array([0.2, 0.8, 0.1, 0.9], dtype=np.float32)
        selection = np.array([2.0, 8.0, 1.0, 9.0], dtype=np.float32)
        baseline = np.array([0.0, 4.0, 2.0, 0.0], dtype=np.float32)
        pred = np.array([1.0, 5.0, 3.0, 7.0], dtype=np.float32)
        true_values = np.array([0.0, 25.0, 50.0, 7.0], dtype=np.float32)

        budget = make_step_budget_features(edge_features, coords, support, selection, baseline, pred)
        targets = make_step_budget_targets(budget.keys, coords, true_values, min_effective_value=1.0, baseline_value=baseline)

        self.assertEqual(budget.keys.tolist(), [[0, 1], [0, 2]])
        self.assertEqual(budget.row_to_step.tolist(), [0, 0, 1, 1])
        self.assertEqual(budget.features.shape[0], 2)
        self.assertAlmostEqual(float(budget.features[0, 1]), 4.0, places=5)
        self.assertAlmostEqual(float(budget.features[0, 2]), 1.0, places=5)
        self.assertTrue(np.allclose(targets.total, np.array([25.0, 57.0], dtype=np.float32)))
        self.assertTrue(np.allclose(targets.count, np.array([1.0, 2.0], dtype=np.float32)))
        self.assertTrue(np.allclose(targets.new_count, np.array([0.0, 1.0], dtype=np.float32)))

    def test_group_rank_targets_can_use_true_value_norm(self):
        from compare_v11_graph_support_generator import make_group_rank_targets

        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0]], dtype=np.int64)
        oracle_scores = np.array([10.0, 5.0, 0.0, 7.0], dtype=np.float32)
        true_values = np.array([0.0, 25.0, 50.0, 10.0], dtype=np.float32)

        targets = make_group_rank_targets(coords, oracle_scores, true_values, mode='true_value_norm')

        self.assertTrue(np.allclose(targets[:3], np.array([0.0, 0.5, 1.0], dtype=np.float32)))
        self.assertAlmostEqual(float(targets[3]), 1.0, places=5)

    def test_step_budget_allocation_limits_count_and_preserves_step_total(self):
        from compare_v11_graph_support_generator import allocate_step_budget_values

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        score = np.array([0.1, 0.9, 0.5, 0.7, 0.2], dtype=np.float32)
        total_by_row = np.array([30.0, 30.0, 30.0, 10.0, 10.0], dtype=np.float32)
        count_by_row = np.array([2.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)

        allocated_selected, values = allocate_step_budget_values(
            coords,
            selected,
            score,
            total_by_row,
            count_by_row,
            allocation_mode='score',
        )

        self.assertEqual(allocated_selected.tolist(), [1, 2, 3])
        self.assertAlmostEqual(float(np.sum(values[:3])), 30.0, places=5)
        self.assertAlmostEqual(float(np.sum(values[3:])), 10.0, places=5)
        self.assertAlmostEqual(float(values[0]), 0.0, places=5)
        self.assertGreater(float(values[1]), float(values[2]))

    def test_step_budget_baseline_score_allocation_prefers_existing_active_edges(self):
        from compare_v11_graph_support_generator import allocate_step_budget_values

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 2], dtype=np.int64)
        score = np.array([0.99, 0.4, 0.3], dtype=np.float32)
        baseline = np.array([0.0, 5.0, 4.0], dtype=np.float32)
        total_by_row = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        count_by_row = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        allocated_selected, values = allocate_step_budget_values(
            coords,
            selected,
            score,
            total_by_row,
            count_by_row,
            allocation_mode='baseline_score',
            baseline_value=baseline,
        )

        self.assertEqual(allocated_selected.tolist(), [1, 2])
        self.assertAlmostEqual(float(values[0]), 0.0, places=5)
        self.assertAlmostEqual(float(np.sum(values)), 20.0, places=5)

    def test_step_budget_baseline_value_score_allocation_prefers_larger_existing_rb(self):
        from compare_v11_graph_support_generator import allocate_step_budget_values

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 2], dtype=np.int64)
        score = np.array([0.99, 0.2, 0.8], dtype=np.float32)
        baseline = np.array([1.0, 9.0, 4.0], dtype=np.float32)
        total_by_row = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        count_by_row = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        allocated_selected, values = allocate_step_budget_values(
            coords,
            selected,
            score,
            total_by_row,
            count_by_row,
            allocation_mode='baseline_value_score',
            baseline_value=baseline,
        )

        self.assertEqual(allocated_selected.tolist(), [1, 2])
        self.assertAlmostEqual(float(values[0]), 0.0, places=5)
        self.assertAlmostEqual(float(np.sum(values)), 20.0, places=5)

    def test_step_budget_new_quota_score_allocation_reserves_inactive_edges(self):
        from compare_v11_graph_support_generator import allocate_step_budget_values

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 2, 3], dtype=np.int64)
        score = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        baseline = np.array([5.0, 4.0, 0.0, 0.0], dtype=np.float32)
        total_by_row = np.full((4,), 20.0, dtype=np.float32)
        count_by_row = np.full((4,), 2.0, dtype=np.float32)
        new_count_by_row = np.full((4,), 1.0, dtype=np.float32)

        allocated_selected, values = allocate_step_budget_values(
            coords,
            selected,
            score,
            total_by_row,
            count_by_row,
            allocation_mode='new_quota_score',
            baseline_value=baseline,
            new_count_by_row=new_count_by_row,
        )

        self.assertEqual(allocated_selected.tolist(), [2, 0])
        self.assertGreater(float(values[2]), 0.0)
        self.assertGreater(float(values[0]), 0.0)
        self.assertAlmostEqual(float(np.sum(values)), 20.0, places=5)

    def test_step_budget_rate_score_allocation_prefers_high_rate_context(self):
        from compare_v11_graph_support_generator import allocate_step_budget_values

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 2], dtype=np.int64)
        score = np.ones((3,), dtype=np.float32)
        total_by_row = np.full((3,), 30.0, dtype=np.float32)
        count_by_row = np.full((3,), 3.0, dtype=np.float32)
        rate_score = np.array([0.1, 2.0, 0.5], dtype=np.float32)

        allocated_selected, values = allocate_step_budget_values(
            coords,
            selected,
            score,
            total_by_row,
            count_by_row,
            allocation_mode='rate_score',
            rate_score=rate_score,
        )

        self.assertEqual(allocated_selected.tolist(), [1, 2, 0])
        self.assertAlmostEqual(float(np.sum(values)), 30.0, places=5)
        self.assertGreater(float(values[1]), float(values[2]))
        self.assertGreater(float(values[2]), float(values[0]))

    def test_step_budget_filter_parser_and_family(self):
        from compare_v11_graph_support_generator import (
            candidate_family_for_repair_value_source,
            parse_step_budget_filter_source,
        )

        self.assertEqual(parse_step_budget_filter_source('step_budget_filter_pred_score'), ('pred', 'score'))
        self.assertEqual(parse_step_budget_filter_source('step_budget_filter_true_baseline_score'), ('true', 'baseline_score'))
        self.assertEqual(
            candidate_family_for_repair_value_source('step_budget_filter_pred_score'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_repair_value_source('step_budget_filter_true_score'),
            'diagnostic_only',
        )
        self.assertEqual(
            candidate_family_for_repair_value_source('step_budget_oracle_alloc_pred_score'),
            'diagnostic_only',
        )
        self.assertEqual(
            candidate_family_for_repair_value_source('step_budget_ranker_pred_score'),
            'graph_support_generator',
        )
        self.assertEqual(
            candidate_family_for_repair_value_source('step_budget_ranker_true_score'),
            'diagnostic_only',
        )

    def test_allocation_ranker_can_score_true_value_order(self):
        from compare_v11_graph_support_generator import fit_allocation_ranker

        coords = []
        values = []
        for step in range(1, 21):
            coords.extend([[0, step, 0], [0, step, 1], [0, step, 2], [0, step, 3]])
            values.extend([0.0, 20.0, 5.0, 0.0] if step % 2 else [0.0, 10.0, 30.0, 0.0])
        coords = np.asarray(coords, dtype=np.int64)
        true_values = np.asarray(values, dtype=np.float32)
        labels = (true_values > 0).astype(np.int64)
        features = np.stack(
            [
                true_values,
                np.log1p(true_values),
                labels.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        model = fit_allocation_ranker(
            features,
            coords,
            true_values,
            labels,
            np.arange(coords.shape[0], dtype=np.int64),
            min_effective_value=1.0,
            seed=7,
        )
        score = np.asarray(model.predict(features), dtype=np.float32)

        self.assertGreater(float(np.mean(score[true_values >= 20.0])), float(np.mean(score[true_values <= 0.0])))
        self.assertGreater(float(np.mean(score[true_values >= 20.0])), float(np.mean(score[(true_values > 0.0) & (true_values < 20.0)])))

    def test_allocation_filter_can_redirect_step_budget_to_true_edge(self):
        from compare_v11_graph_support_generator import (
            allocate_step_budget_values,
            fit_allocation_filter,
            make_allocation_filter_features,
            predict_support_score,
        )

        coords = np.asarray([[sample, 1, edge] for sample in range(16) for edge in (0, 1)], dtype=np.int64)
        edge_features = np.asarray(
            [[1.0 + 0.01 * sample, 0.0] if edge == 0 else [0.0, 1.0 + 0.01 * sample] for sample in range(16) for edge in (0, 1)],
            dtype=np.float32,
        )
        support = np.asarray([0.9 if edge == 0 else 0.1 for _sample in range(16) for edge in (0, 1)], dtype=np.float32)
        selection = support.copy()
        baseline = np.zeros((32,), dtype=np.float32)
        pred = np.ones((32,), dtype=np.float32)
        total_by_row = np.full((32,), 20.0, dtype=np.float32)
        count_by_row = np.ones((32,), dtype=np.float32)
        true_values = np.asarray([0.0 if edge == 0 else 20.0 for _sample in range(16) for edge in (0, 1)], dtype=np.float32)
        labels = (true_values > 0.0).astype(np.int64)
        selected = np.arange(32, dtype=np.int64)

        features = make_allocation_filter_features(
            edge_features,
            coords,
            support,
            selection,
            baseline,
            pred,
            total_by_row,
            count_by_row,
        )
        model = fit_allocation_filter(features, true_values, labels, selected, min_effective_value=1.0, seed=11)
        filter_score = predict_support_score(model, features)

        raw_selected, raw_values = allocate_step_budget_values(
            coords,
            selected,
            selection,
            total_by_row,
            count_by_row,
            allocation_mode='score',
        )
        filtered_selected, filtered_values = allocate_step_budget_values(
            coords,
            selected,
            filter_score,
            total_by_row,
            count_by_row,
            allocation_mode='score',
        )

        self.assertEqual(raw_selected.tolist(), list(range(0, 32, 2)))
        self.assertEqual(filtered_selected.tolist(), list(range(1, 32, 2)))
        self.assertAlmostEqual(float(np.sum(raw_values)), 320.0, places=5)
        self.assertAlmostEqual(float(np.sum(filtered_values)), 320.0, places=5)

    def test_step_budget_reconstruction_clears_affected_step_before_writing_budget(self):
        from compare_v11_graph_support_generator import apply_step_budget_reconstruction

        actions = np.zeros((1, 3, 3, 6), dtype=np.float32)
        actions[0, 1, :, 2] = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        actions[0, 2, :, 2] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        coords = np.array([[0, 1, 1]], dtype=np.int64)
        values = np.array([20.0], dtype=np.float32)
        selected = np.array([0], dtype=np.int64)

        repaired = apply_step_budget_reconstruction(actions, coords, values, selected)

        self.assertTrue(np.allclose(repaired[0, 1, :, 2], np.array([0.0, 20.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[0, 2, :, 2], actions[0, 2, :, 2]))

    def test_gated_step_budget_reconstruction_falls_back_on_low_confidence_steps(self):
        from compare_v11_graph_support_generator import apply_gated_step_budget_reconstruction

        fallback = np.zeros((1, 3, 3, 6), dtype=np.float32)
        fallback[0, 1, :, 2] = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        fallback[0, 2, :, 2] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        coords = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int64)
        values = np.array([20.0, 30.0], dtype=np.float32)
        selected = np.array([0, 1], dtype=np.int64)
        confidence = np.array([0.9, 0.2], dtype=np.float32)

        repaired = apply_gated_step_budget_reconstruction(
            fallback,
            coords,
            values,
            selected,
            confidence,
            threshold=0.5,
        )

        self.assertTrue(np.allclose(repaired[0, 1, :, 2], np.array([0.0, 20.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[0, 2, :, 2], fallback[0, 2, :, 2]))

    def test_reconstruct_gated_step_budget_keeps_reconstruct_fallback_for_low_confidence(self):
        from compare_v11_graph_support_generator import apply_gated_step_budget_reconstruction

        fallback = np.zeros((1, 3, 3, 6), dtype=np.float32)
        fallback[0, 1, :, 2] = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        fallback[0, 2, :, 2] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        coords = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int64)
        values = np.array([20.0, 30.0], dtype=np.float32)
        selected = np.array([0, 1], dtype=np.int64)
        confidence = np.array([0.9, 0.2], dtype=np.float32)

        repaired = apply_gated_step_budget_reconstruction(
            fallback,
            coords,
            values,
            selected,
            confidence,
            threshold=0.5,
        )

        self.assertTrue(np.allclose(repaired[0, 1, :, 2], np.array([0.0, 20.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[0, 2, :, 2], np.array([1.0, 2.0, 3.0], dtype=np.float32)))

    def test_soft_step_budget_repair_blends_selected_values_without_clearing_step(self):
        from compare_v11_graph_support_generator import apply_soft_step_budget_repair

        fallback = np.zeros((1, 3, 3, 6), dtype=np.float32)
        fallback[0, 1, :, 2] = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        fallback[0, 2, :, 2] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        coords = np.array([[0, 1, 1]], dtype=np.int64)
        values = np.array([21.0], dtype=np.float32)
        selected = np.array([0], dtype=np.int64)

        repaired = apply_soft_step_budget_repair(fallback, coords, values, selected, beta=0.5)

        self.assertTrue(np.allclose(repaired[0, 1, :, 2], np.array([5.0, 14.0, 9.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[0, 2, :, 2], fallback[0, 2, :, 2]))

    def test_action_budget_diagnostics_reports_totals_and_step_error(self):
        from compare_v11_graph_support_generator import action_budget_diagnostics

        actions = np.zeros((1, 3, 2, 6), dtype=np.float32)
        baseline = np.zeros_like(actions)
        truth = np.zeros_like(actions)
        actions[0, 1, :, 2] = np.array([4.0, 6.0], dtype=np.float32)
        actions[0, 2, :, 2] = np.array([2.0, 0.0], dtype=np.float32)
        baseline[0, 1, :, 2] = np.array([1.0, 1.0], dtype=np.float32)
        truth[0, 1, :, 2] = np.array([5.0, 5.0], dtype=np.float32)
        truth[0, 2, :, 2] = np.array([1.0, 1.0], dtype=np.float32)

        diagnostics = action_budget_diagnostics(actions, baseline, truth)

        self.assertAlmostEqual(diagnostics['action_rb_total'], 12.0, places=5)
        self.assertAlmostEqual(diagnostics['truth_rb_total'], 12.0, places=5)
        self.assertAlmostEqual(diagnostics['action_rb_total_ratio_to_truth'], 1.0, places=5)
        self.assertEqual(diagnostics['action_rb_nonzero_count'], 3)
        self.assertEqual(diagnostics['baseline_rb_nonzero_count'], 2)
        self.assertEqual(diagnostics['truth_rb_nonzero_count'], 4)
        self.assertAlmostEqual(diagnostics['action_step_total_mae_vs_truth'], 0.0, places=5)

    def test_mass_preserving_reallocation_preserves_step_total_and_prefers_high_score(self):
        from compare_v11_graph_support_generator import apply_mass_preserving_reallocation

        fallback = np.zeros((1, 3, 3, 6), dtype=np.float32)
        fallback[0, 1, :, 2] = np.array([5.0, 7.0, 8.0], dtype=np.float32)
        fallback[0, 2, :, 2] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]], dtype=np.int64)
        score = np.array([0.1, 0.2, 0.9], dtype=np.float32)
        selected = np.array([0, 1, 2], dtype=np.int64)

        repaired = apply_mass_preserving_reallocation(fallback, coords, score, selected, beta=1.0)

        self.assertAlmostEqual(float(np.sum(repaired[0, 1, :, 2])), 20.0, places=5)
        self.assertTrue(np.allclose(repaired[0, 2, :, 2], fallback[0, 2, :, 2]))
        self.assertGreater(float(repaired[0, 1, 2, 2]), float(repaired[0, 1, 1, 2]))
        self.assertGreater(float(repaired[0, 1, 1, 2]), float(repaired[0, 1, 0, 2]))

    def test_selected_step_confidence_maps_group_mean_to_selected_rows(self):
        from compare_v11_graph_support_generator import selected_step_confidence

        coords = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0]], dtype=np.int64)
        selected = np.array([0, 1], dtype=np.int64)
        support_score = np.array([0.2, 0.8, 0.9], dtype=np.float32)

        confidence = selected_step_confidence(coords, selected, support_score)

        self.assertTrue(np.allclose(confidence, np.array([0.5, 0.5, 0.0], dtype=np.float32)))

    def test_step_reconstruction_gate_falls_back_on_pred_value_for_inactive_steps(self):
        from compare_v11_graph_support_generator import apply_step_budget_step_gate_reconstruction

        baseline = np.zeros((2, 2, 2, 6), dtype=np.float32)
        fallback = baseline.copy()
        fallback[0, 1, :, 2] = np.array([2.0, 3.0], dtype=np.float32)
        fallback[1, 1, :, 2] = np.array([4.0, 5.0], dtype=np.float32)
        coords = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int64)
        values = np.array([20.0, 30.0], dtype=np.float32)
        selected = np.array([0, 1], dtype=np.int64)
        step_gate = np.array([0.9, 0.1], dtype=np.float32)

        repaired = apply_step_budget_step_gate_reconstruction(
            fallback,
            coords,
            values,
            selected,
            step_gate,
            threshold=0.5,
        )

        self.assertTrue(np.allclose(repaired[0, 1, :, 2], np.array([20.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[1, 1, :, 2], fallback[1, 1, :, 2]))

    def test_structured_step_score_rewards_selected_predicted_gain_with_new_penalty(self):
        from compare_v11_graph_support_generator import make_structured_step_score_matrix

        actions = np.zeros((1, 3, 4, 6), dtype=np.float32)
        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 2, 0],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 3], dtype=np.int64)
        selection_score = np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32)
        pred_value = np.array([10.0, 20.0, 30.0, 5.0], dtype=np.float32)
        baseline_value = np.array([7.0, 0.0, 0.0, 0.0], dtype=np.float32)

        no_penalty = make_structured_step_score_matrix(
            actions,
            coords,
            selected,
            selection_score,
            pred_value,
            baseline_value,
            new_edge_penalty=0.0,
        )
        with_penalty = make_structured_step_score_matrix(
            actions,
            coords,
            selected,
            selection_score,
            pred_value,
            baseline_value,
            new_edge_penalty=5.0,
        )

        self.assertEqual(no_penalty.shape, (1, 3))
        self.assertAlmostEqual(float(no_penalty[0, 0]), 0.0, places=5)
        self.assertGreater(float(no_penalty[0, 1]), float(no_penalty[0, 2]))
        self.assertLess(float(with_penalty[0, 1]), float(no_penalty[0, 1]))
        self.assertLess(float(with_penalty[0, 2]), float(no_penalty[0, 2]))

    def test_structured_step_score_complexity_penalty_can_prefer_narrow_candidate(self):
        from compare_v11_graph_support_generator import make_structured_step_score_matrix

        actions = np.zeros((1, 2, 4, 6), dtype=np.float32)
        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
            ],
            dtype=np.int64,
        )
        selection_score = np.ones((4,), dtype=np.float32)
        pred_value = np.array([20.0, 20.0, 1.0, 1.0], dtype=np.float32)
        baseline_value = np.zeros((4,), dtype=np.float32)
        narrow = np.array([0, 1], dtype=np.int64)
        wide = np.array([0, 1, 2, 3], dtype=np.int64)

        narrow_score = make_structured_step_score_matrix(
            actions,
            coords,
            narrow,
            selection_score,
            pred_value,
            baseline_value,
            new_edge_penalty=0.0,
            selected_count_penalty=0.0,
        )
        wide_score = make_structured_step_score_matrix(
            actions,
            coords,
            wide,
            selection_score,
            pred_value,
            baseline_value,
            new_edge_penalty=0.0,
            selected_count_penalty=2.0,
        )

        self.assertGreater(float(narrow_score[0, 1]), float(wide_score[0, 1]))

    def test_compose_structured_stepwise_actions_selects_best_candidate_per_step(self):
        from compare_v11_graph_support_generator import (
            StructuredActionCandidate,
            compose_structured_stepwise_actions,
        )

        baseline = np.zeros((2, 3, 2, 6), dtype=np.float32)
        baseline[:, 0, :, 2] = 9.0
        candidate_a = baseline.copy()
        candidate_b = baseline.copy()
        candidate_a[0, 1, :, 2] = np.array([5.0, 0.0], dtype=np.float32)
        candidate_b[0, 1, :, 2] = np.array([0.0, 7.0], dtype=np.float32)
        candidate_a[1, 2, :, 2] = np.array([3.0, 0.0], dtype=np.float32)
        candidate_b[1, 2, :, 2] = np.array([0.0, 4.0], dtype=np.float32)

        composed, counts = compose_structured_stepwise_actions(
            baseline,
            [
                StructuredActionCandidate(
                    name='a',
                    actions=candidate_a,
                    step_scores=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32),
                ),
                StructuredActionCandidate(
                    name='b',
                    actions=candidate_b,
                    step_scores=np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                ),
            ],
        )

        self.assertTrue(np.allclose(composed[:, 0, :, 2], baseline[:, 0, :, 2]))
        self.assertTrue(np.allclose(composed[0, 1, :, 2], np.array([0.0, 7.0], dtype=np.float32)))
        self.assertTrue(np.allclose(composed[1, 2, :, 2], np.array([3.0, 0.0], dtype=np.float32)))
        self.assertEqual(counts['a'], 1)
        self.assertEqual(counts['b'], 1)

    def test_branch_step_features_and_targets_capture_branch_quality(self):
        from compare_v11_graph_support_generator import (
            make_branch_step_features,
            make_branch_step_targets,
        )

        keys = np.array([[0, 1], [0, 2]], dtype=np.int64)
        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 2, 0],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
        selected = np.array([0, 1, 3], dtype=np.int64)
        selection_score = np.array([0.5, 1.5, 0.2, 2.0], dtype=np.float32)
        pred_value = np.array([10.0, 5.0, 1.0, 20.0], dtype=np.float32)
        baseline_value = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        true_value = np.array([8.0, 0.0, 0.0, 30.0], dtype=np.float32)
        labels = (true_value > 0.0).astype(np.int64)

        features = make_branch_step_features(keys, coords, selected, selection_score, pred_value, baseline_value)
        targets = make_branch_step_targets(keys, coords, selected, true_value, labels, baseline_value)

        self.assertEqual(features.shape, (2, 10))
        self.assertAlmostEqual(float(features[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(features[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(features[0, 3]), 2.0, places=5)
        self.assertAlmostEqual(float(features[1, 6]), 20.0, places=5)
        self.assertGreater(float(targets[1]), float(targets[0]))

    def test_learned_branch_selector_can_choose_higher_target_branch(self):
        from compare_v11_graph_support_generator import fit_branch_value_regressor

        features = []
        targets = []
        for step in range(40):
            features.append([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 0.0])
            targets.append(30.0)
            features.append([3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 8.0, 2.0, 8.0, 0.0])
            targets.append(5.0)
        model = fit_branch_value_regressor(np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), seed=9)
        predictions = np.asarray(
            model.predict(
                np.asarray(
                    [
                        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 0.0],
                        [3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 8.0, 2.0, 8.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            ),
            dtype=np.float32,
        )

        self.assertGreater(float(predictions[0]), float(predictions[1]))


if __name__ == '__main__':
    unittest.main()
