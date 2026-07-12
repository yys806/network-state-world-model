import unittest
import inspect

import numpy as np
import pandas as pd


class V4AblationAndActiveRateTest(unittest.TestCase):
    def test_v4_train_model_accepts_reproducibility_seed(self):
        from run_world_model_v4_dual_graph_rollout import train_model

        sig = inspect.signature(train_model)
        self.assertIn("torch_seed", sig.parameters)
        self.assertEqual(sig.parameters["torch_seed"].default, 42)

    def test_physical_variant_keeps_only_requested_feature_groups(self):
        from run_world_model_v4_dual_graph_ablation import make_physical_variant

        arrays = {"x_phy_edge": np.arange(2 * 1 * 3 * 8, dtype=np.float32).reshape(2, 1, 3, 8)}

        full = make_physical_variant(arrays, "dual_full")
        no_physical = make_physical_variant(arrays, "no_physical")
        distance_height_speed = make_physical_variant(arrays, "distance_height_speed")

        np.testing.assert_allclose(full["x_phy_edge"], arrays["x_phy_edge"])
        self.assertEqual(float(no_physical["x_phy_edge"].sum()), 0.0)

        kept = [3, 4, 7]
        dropped = [0, 1, 2, 5, 6]
        np.testing.assert_allclose(
            distance_height_speed["x_phy_edge"][..., kept],
            arrays["x_phy_edge"][..., kept],
        )
        self.assertEqual(float(distance_height_speed["x_phy_edge"][..., dropped].sum()), 0.0)

    def test_active_rate_features_append_physical_history(self):
        from run_world_model_v4_active_rate_calibration import build_rate_features_with_physical
        from run_world_model_v3_active_rate_calibration import EDGE_VOCAB_PATH

        num_edges = len(pd.read_csv(EDGE_VOCAB_PATH))
        arrays = {
            "x_node": np.zeros((1, 2, 2, 7), dtype=np.float32),
            "x_link": np.zeros((1, 2, num_edges, 5), dtype=np.float32),
            "x_task": np.zeros((1, 2, 4), dtype=np.float32),
            "edge_a_hist": np.zeros((1, 2, num_edges, 6), dtype=np.float32),
            "edge_a_future": np.zeros((1, 2, num_edges, 6), dtype=np.float32),
            "x_phy_edge": np.ones((1, 2, num_edges, 8), dtype=np.float32),
            "edge_src_idx": np.zeros(num_edges, dtype=np.int64),
            "edge_dst_idx": np.ones(num_edges, dtype=np.int64),
        }
        pred = {
            "active_prob": np.full((1, 2, num_edges), 0.25, dtype=np.float32),
            "rate_pred": np.full((1, 2, num_edges), 5.0, dtype=np.float32),
        }

        features_without_pred = build_rate_features_with_physical(arrays, np.array([0]))
        features_with_pred = build_rate_features_with_physical(arrays, np.array([0]), pred=pred)

        self.assertEqual(features_without_pred.shape[0], 1 * 2 * num_edges)
        self.assertGreaterEqual(features_without_pred.shape[1], 16)
        self.assertEqual(features_with_pred.shape[0], features_without_pred.shape[0])
        self.assertEqual(features_with_pred.shape[1], features_without_pred.shape[1] + 4)

    def test_stability_summary_reports_mean_std_and_count(self):
        from run_world_model_v4_seed_stability import summarize_stability

        df = pd.DataFrame(
            [
                {"physical_variant": "dual_full", "split": "test_seed_4", "activity_f1": 0.2, "task_rmse": 5.0},
                {"physical_variant": "dual_full", "split": "test_seed_4", "activity_f1": 0.4, "task_rmse": 7.0},
                {"physical_variant": "no_physical", "split": "test_seed_4", "activity_f1": 0.8, "task_rmse": 4.0},
            ]
        )
        out = summarize_stability(df)
        dual = out[out["physical_variant"] == "dual_full"].iloc[0]
        self.assertEqual(int(dual["runs"]), 2)
        self.assertAlmostEqual(float(dual["activity_f1_mean"]), 0.3)
        self.assertAlmostEqual(float(dual["task_rmse_mean"]), 6.0)
        self.assertGreater(float(dual["activity_f1_std"]), 0.0)

    def test_activity_ratio_threshold_matches_validation_active_ratio(self):
        from run_world_model_v4_activity_calibration import select_ratio_threshold, threshold_metrics

        y_val = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
        prob_val = np.array([0.91, 0.82, 0.64, 0.51, 0.30, 0.12], dtype=np.float32)

        selected = select_ratio_threshold(y_val, prob_val)
        metrics = threshold_metrics(y_val, prob_val, selected["threshold"])

        self.assertEqual(metrics["predicted_active_count"], int(y_val.sum()))
        self.assertAlmostEqual(metrics["predicted_active_ratio"], float(y_val.mean()))

    def test_activity_precision_constrained_threshold_respects_target_when_possible(self):
        from run_world_model_v4_activity_calibration import (
            select_precision_constrained_threshold,
            threshold_metrics,
        )

        y_val = np.array([1, 0, 1, 0, 0], dtype=np.float32)
        prob_val = np.array([0.95, 0.80, 0.70, 0.30, 0.10], dtype=np.float32)

        selected = select_precision_constrained_threshold(y_val, prob_val, min_precision=0.75)
        metrics = threshold_metrics(y_val, prob_val, selected["threshold"])

        self.assertGreaterEqual(metrics["precision"], 0.75)
        self.assertGreater(metrics["recall"], 0.0)

    def test_temperature_scaling_preserves_probability_ordering(self):
        from run_world_model_v4_activity_calibration import apply_temperature

        prob = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float32)
        cooled = apply_temperature(prob, temperature=0.5)
        warmed = apply_temperature(prob, temperature=2.0)

        np.testing.assert_array_equal(np.argsort(prob), np.argsort(cooled))
        np.testing.assert_array_equal(np.argsort(prob), np.argsort(warmed))
        self.assertGreater(abs(float(cooled[-1] - 0.5)), abs(float(warmed[-1] - 0.5)))

    def test_metric_suite_probability_calibration_reports_brier_and_ece(self):
        from run_world_model_metric_suite_v0 import probability_calibration_metrics

        y = np.array([0, 0, 1, 1], dtype=np.float32)
        prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        metrics = probability_calibration_metrics(y, prob, n_bins=2)

        self.assertLess(metrics["brier_score"], 0.05)
        self.assertAlmostEqual(metrics["ece"], 0.15, places=6)
        self.assertEqual(metrics["num_items"], 4)

    def test_metric_suite_summarizes_seed_variation(self):
        from run_world_model_metric_suite_v0 import summarize_metric_variation

        df = pd.DataFrame(
            [
                {"model": "m", "split": "test_seed_4", "activity_f1": 0.2, "task_rmse": 5.0},
                {"model": "m", "split": "test_seed_4", "activity_f1": 0.6, "task_rmse": 7.0},
                {"model": "n", "split": "test_seed_4", "activity_f1": 0.9, "task_rmse": 4.0},
            ]
        )

        out = summarize_metric_variation(df, group_cols=["model"], metric_cols=["activity_f1", "task_rmse"])
        row = out[out["model"] == "m"].iloc[0]

        self.assertEqual(int(row["runs"]), 2)
        self.assertAlmostEqual(float(row["activity_f1_mean"]), 0.4)
        self.assertAlmostEqual(float(row["activity_f1_min"]), 0.2)
        self.assertGreater(float(row["task_rmse_std"]), 0.0)

    def test_metric_suite_collects_offload_scaled_sweep_when_present(self):
        from run_world_model_metric_suite_v0 import collect_v5_resource_aware_rows

        out = collect_v5_resource_aware_rows()

        self.assertIn("v5_resource_aware_offload_scaled_sweep", set(out["category"]))
        part = out[out["category"].eq("v5_resource_aware_offload_scaled_sweep")]
        self.assertIn("group_top1_hit_mean", set(part["metric"]))
        self.assertIn("group_normalized_top1_regret_mean", set(part["metric"]))

    def test_runtime_comparison_computes_k_step_speedup(self):
        from run_world_model_runtime_comparison_v0 import compute_speedup

        row = compute_speedup(sim_step_ms=6.0, horizon=3, model_sample_ms=0.6)

        self.assertAlmostEqual(row["sim_k_step_ms"], 18.0)
        self.assertAlmostEqual(row["model_sample_ms"], 0.6)
        self.assertAlmostEqual(row["speedup"], 30.0)

    def test_runtime_summary_handles_empty_warmup_and_percentiles(self):
        from run_world_model_runtime_comparison_v0 import summarize_times

        out = summarize_times([1.0, 2.0, 3.0, 100.0], warmup=1)

        self.assertEqual(out["num_repeats"], 3)
        self.assertAlmostEqual(out["mean_ms"], 35.0)
        self.assertAlmostEqual(out["p50_ms"], 3.0)

    def test_physical_rollout_linear_extrapolation(self):
        from run_world_model_physical_rollout_baseline_v0 import linear_extrapolate_nodes

        x_node = np.zeros((1, 3, 1, 7), dtype=np.float32)
        x_node[0, :, 0, 0] = [0.0, 2.0, 4.0]
        x_node[0, :, 0, 1] = [1.0, 1.0, 1.0]
        pred = linear_extrapolate_nodes(x_node, horizon=2)

        self.assertEqual(pred.shape, (1, 2, 1, 7))
        self.assertAlmostEqual(float(pred[0, 0, 0, 0]), 6.0)
        self.assertAlmostEqual(float(pred[0, 1, 0, 0]), 8.0)
        self.assertAlmostEqual(float(pred[0, 1, 0, 1]), 1.0)

    def test_physical_rollout_edge_distance_uses_node_endpoints(self):
        from run_world_model_physical_rollout_baseline_v0 import edge_distance_from_nodes

        nodes = np.zeros((1, 1, 2, 7), dtype=np.float32)
        nodes[0, 0, 0, :3] = [0.0, 0.0, 0.0]
        nodes[0, 0, 1, :3] = [3.0, 4.0, 0.0]
        dist = edge_distance_from_nodes(nodes, np.array([0]), np.array([1]))

        self.assertEqual(dist.shape, (1, 1, 1))
        self.assertAlmostEqual(float(dist[0, 0, 0]), 5.0)

    def test_logged_action_spearman_rank_correlation(self):
        from run_world_model_logged_action_ranking_proxy_v0 import spearman_rank_correlation

        true = np.array([1.0, 2.0, 3.0, 4.0])

        self.assertAlmostEqual(spearman_rank_correlation(true, true), 1.0)
        self.assertAlmostEqual(spearman_rank_correlation(true, true[::-1]), -1.0)

    def test_logged_action_topk_hit_rate(self):
        from run_world_model_logged_action_ranking_proxy_v0 import topk_hit_rate

        true = np.array([10.0, 8.0, 6.0, 4.0])
        pred = np.array([9.0, 1.0, 7.0, 2.0])

        self.assertAlmostEqual(topk_hit_rate(true, pred, k=2), 0.5)
        self.assertAlmostEqual(topk_hit_rate(true, pred, k=10), 1.0)

    def test_logged_action_ranking_regret(self):
        from run_world_model_logged_action_ranking_proxy_v0 import ranking_regret

        true = np.array([10.0, 8.0, 6.0, 4.0])
        pred = np.array([1.0, 8.0, 9.0, 3.0])

        out = ranking_regret(true, pred, top_k=2)

        self.assertAlmostEqual(out["top1_regret"], 4.0)
        self.assertAlmostEqual(out["normalized_top1_regret"], 4.0 / 6.0)
        self.assertAlmostEqual(out["topk_best_regret"], 2.0)

    def test_counterfactual_utility_rewards_finished_and_penalizes_failures(self):
        from run_airfogsim_counterfactual_action_smoke_v0 import compute_counterfactual_utility

        out = compute_counterfactual_utility(
            start_done=2,
            end_done=7,
            start_failed=1,
            end_failed=3,
            throughput=100.0,
        )

        self.assertAlmostEqual(out["delta_done"], 5.0)
        self.assertAlmostEqual(out["delta_failed"], 2.0)
        self.assertAlmostEqual(out["utility"], 5.0 - 2.0 + 0.01 * np.log1p(100.0))

    def test_v5_resource_aware_utility_penalizes_rb_consumption(self):
        from run_world_model_v5_utility_ranking_smoke import add_resource_aware_utility

        df = pd.DataFrame(
            [
                {"airfogsim_utility": 1.05, "total_rb": 50},
                {"airfogsim_utility": 1.03, "total_rb": 1},
            ]
        )

        out = add_resource_aware_utility(df, rb_penalty=0.001)

        self.assertAlmostEqual(float(out.loc[0, "resource_aware_utility"]), 1.0)
        self.assertAlmostEqual(float(out.loc[1, "resource_aware_utility"]), 1.029)
        self.assertGreater(float(out.loc[1, "resource_aware_utility"]), float(out.loc[0, "resource_aware_utility"]))

    def test_v5_decision_baseline_evaluation_uses_selected_utility(self):
        from run_world_model_v5_decision_baselines_v0 import evaluate_baseline

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "resource_aware_utility": 1.0, "predict_total_rb": 0.0},
                {"decision_group_id": "g0", "resource_aware_utility": 2.0, "predict_total_rb": 1.0},
            ]
        )

        out = evaluate_baseline(df, "resource_aware_utility", "predict_total_rb", np.array([0, 1]), "test")

        self.assertEqual(out["utility"], "resource_aware_utility")
        self.assertEqual(out["baseline"], "predict_total_rb")
        self.assertAlmostEqual(out["top1_hit_mean"], 1.0)

    def test_v5_multifamily_baselines_add_family_heuristics(self):
        from run_world_model_v5_decision_baselines_v0 import add_decision_baseline_scores

        df = pd.DataFrame(
            [
                {
                    "action_family": "rb_count",
                    "candidate_id": "default",
                    "total_rb": 50,
                    "world_model_utility": 1.0,
                },
                {
                    "action_family": "mixed_offload_rb",
                    "candidate_id": "mixed_alt_rb_1",
                    "total_rb": 1,
                    "world_model_utility": 0.8,
                },
                {
                    "action_family": "offload_target",
                    "candidate_id": "offload_alt",
                    "total_rb": 50,
                    "world_model_utility": 1.1,
                },
            ]
        )

        out = add_decision_baseline_scores(df)

        self.assertIn("predict_mixed_family", out.columns)
        self.assertIn("predict_non_rb_family", out.columns)
        self.assertIn("predict_default", out.columns)
        self.assertGreater(float(out.loc[1, "predict_mixed_family"]), float(out.loc[0, "predict_mixed_family"]))
        self.assertGreater(float(out.loc[2, "predict_non_rb_family"]), float(out.loc[0, "predict_non_rb_family"]))
        self.assertGreater(float(out.loc[0, "predict_default"]), float(out.loc[1, "predict_default"]))
        self.assertGreater(float(out.loc[1, "predict_minus_total_rb"]), float(out.loc[0, "predict_minus_total_rb"]))

    def test_counterfactual_summarizes_ranking_metrics(self):
        from run_airfogsim_counterfactual_action_smoke_v0 import summarize_candidate_ranking

        df = pd.DataFrame(
            [
                {"candidate_id": "a", "airfogsim_utility": 10.0, "world_model_utility": 8.0},
                {"candidate_id": "b", "airfogsim_utility": 6.0, "world_model_utility": 9.0},
                {"candidate_id": "c", "airfogsim_utility": 2.0, "world_model_utility": 1.0},
            ]
        )
        out = summarize_candidate_ranking(df, top_k=2)

        self.assertAlmostEqual(out["top2_hit_rate"], 1.0)
        self.assertAlmostEqual(out["top1_regret"], 4.0)
        self.assertEqual(out["best_world_model_candidate"], "b")

    def test_counterfactual_rb_variants_keep_positive_blocks(self):
        from run_airfogsim_counterfactual_action_smoke_v0 import rb_count_variants

        variants = rb_count_variants(default_count=3, n_rb=10)

        self.assertIn(1, variants)
        self.assertIn(3, variants)
        self.assertIn(6, variants)
        self.assertEqual(len(variants), len(set(variants)))

    def test_multifamily_candidates_include_offload_rb_and_mixed_actions(self):
        from run_airfogsim_counterfactual_multifamily_v0 import build_multifamily_candidates

        offload_options = [
            {
                "task_id": "t0",
                "task_node_id": "vehicle_0",
                "default_target_id": "rsu_0",
                "alternative_target_id": "uav_0",
                "default_target_type": "I",
                "alternative_target_type": "U",
                "default_distance": 10.0,
                "alternative_distance": 30.0,
            }
        ]
        default_rb_plan = {"t0": [0, 1, 2, 3]}

        candidates = build_multifamily_candidates(
            offload_options,
            default_rb_plan,
            n_rb=10,
            max_candidates=8,
        )
        families = {candidate["action_family"] for candidate in candidates}

        self.assertIn("rb_count", families)
        self.assertIn("offload_target", families)
        self.assertIn("mixed_offload_rb", families)
        self.assertTrue(all(candidate["total_rb"] > 0 for candidate in candidates))
        offload_candidates = [candidate for candidate in candidates if candidate["action_family"] != "rb_count"]
        self.assertTrue(all(candidate["offload_distance_delta"] == 20.0 for candidate in offload_candidates))
        self.assertTrue(all(candidate["offload_alternative_distance"] == 30.0 for candidate in offload_candidates))
        self.assertTrue(all(candidate["offload_default_is_rsu"] == 1.0 for candidate in offload_candidates))
        self.assertTrue(all(candidate["offload_alternative_is_uav"] == 1.0 for candidate in offload_candidates))
        self.assertTrue(all(candidate["offload_target_type_changed"] == 1.0 for candidate in offload_candidates))

    def test_multifamily_offload_options_need_distinct_target(self):
        from run_airfogsim_counterfactual_multifamily_v0 import build_offload_options

        task_infos = [{"task_id": "t0", "task_node_id": "vehicle_0"}]
        neighbor_map = {
            "t0": [
                {"id": "rsu_0", "distance": 1.0},
                {"id": "rsu_0", "distance": 2.0},
            ]
        }

        options = build_offload_options(task_infos, neighbor_map)

        self.assertEqual(options, [])

    def test_multifamily_neighbor_distances_are_filled_from_env_when_missing(self):
        from run_airfogsim_counterfactual_multifamily_v0 import fill_neighbor_distances

        class FakeEnv:
            def getDistanceBetweenNodesById(self, source_id, target_id):
                distances = {
                    ("vehicle_0", "rsu_0"): 10.0,
                    ("vehicle_0", "uav_0"): 30.0,
                }
                return distances[(source_id, target_id)]

        neighbors = [{"id": "rsu_0"}, {"id": "uav_0", "distance": 0.0}]

        out = fill_neighbor_distances(FakeEnv(), "vehicle_0", neighbors)

        self.assertEqual(out[0]["distance"], 10.0)
        self.assertEqual(out[1]["distance"], 30.0)
        self.assertEqual(neighbors[0].get("distance"), None)

    def test_multifamily_group_spread_marks_nontrivial_labels(self):
        from run_airfogsim_counterfactual_multifamily_v0 import summarize_group_utility_spread

        labels = pd.DataFrame(
            [
                {"decision_group_id": "g0", "airfogsim_utility": 1.0},
                {"decision_group_id": "g0", "airfogsim_utility": 1.0},
                {"decision_group_id": "g1", "airfogsim_utility": 2.5},
                {"decision_group_id": "g1", "airfogsim_utility": 1.0},
            ]
        )

        out = summarize_group_utility_spread(labels, utility_col="airfogsim_utility", min_utility_spread=0.1)
        by_group = out.set_index("decision_group_id")

        self.assertAlmostEqual(float(by_group.loc["g0", "utility_spread"]), 0.0)
        self.assertFalse(bool(by_group.loc["g0", "is_nontrivial"]))
        self.assertAlmostEqual(float(by_group.loc["g1", "utility_spread"]), 1.5)
        self.assertTrue(bool(by_group.loc["g1", "is_nontrivial"]))

    def test_extended_candidates_include_cpu_scale_actions(self):
        from run_airfogsim_counterfactual_multifamily_v0 import build_cpu_candidates

        computing_tasks = [
            {"task_id": "t0", "assigned_to": "rsu_0", "base_cpu": 5.0},
            {"task_id": "t1", "assigned_to": "rsu_0", "base_cpu": 5.0},
        ]

        candidates = build_cpu_candidates(computing_tasks, max_candidates=4)
        families = {candidate["action_family"] for candidate in candidates}

        self.assertIn("cpu_scale", families)
        self.assertTrue(any(candidate["cpu_scale"] < 1.0 for candidate in candidates))
        self.assertTrue(any(candidate["cpu_scale"] > 1.0 for candidate in candidates))
        self.assertTrue(all(candidate["cpu_overrides"] for candidate in candidates))

    def test_extended_candidates_include_direct_and_relay_return_routes(self):
        from run_airfogsim_counterfactual_multifamily_v0 import build_return_route_candidates

        waiting_tasks = [
            {
                "task_id": "t0",
                "current_node_id": "vehicle_0",
                "direct_routes": [["rsu_near"], ["rsu_far"]],
                "relay_routes": [["uav_near", "rsu_near"]],
            }
        ]

        candidates = build_return_route_candidates(waiting_tasks, max_candidates=5)
        route_ids = {candidate["candidate_id"] for candidate in candidates}

        self.assertIn("return_t0_direct_rsu_near", route_ids)
        self.assertIn("return_t0_relay_uav_near_rsu_near", route_ids)
        self.assertTrue(all(candidate["return_route_overrides"] for candidate in candidates))

    def test_return_route_override_uses_waiting_return_queue(self):
        from run_airfogsim_counterfactual_multifamily_v0 import apply_return_route_overrides

        class FakeTask:
            def getTaskId(self):
                return "t0"

        class FakeTaskManager:
            def getTaskByTaskId(self, task_id):
                return None

            def getWaitingToReturnTaskInfos(self):
                return {"node_0": [FakeTask()]}

        class FakeTaskScheduler:
            def __init__(self):
                self.routes = {}

            def setTaskReturnRoute(self, env, task_id, route):
                self.routes[task_id] = route

        class FakeAlgorithm:
            def __init__(self):
                self.taskScheduler = FakeTaskScheduler()

        class FakeEnv:
            def __init__(self):
                self.task_manager = FakeTaskManager()

        env = FakeEnv()
        algorithm = FakeAlgorithm()
        candidate = {"return_route_overrides": {"t0": ["rsu_0"]}}

        applied = apply_return_route_overrides(env, algorithm, candidate)

        self.assertEqual(applied, 1)
        self.assertEqual(algorithm.taskScheduler.routes["t0"], ["rsu_0"])

    def test_v5_pairwise_ranking_pairs_respect_utility_order(self):
        from run_world_model_v5_utility_ranking_smoke import build_pairwise_pairs

        utility = np.array([1.0, 3.0, 2.0], dtype=np.float32)
        pairs = build_pairwise_pairs(utility)

        self.assertIn((1, 0), pairs)
        self.assertIn((1, 2), pairs)
        self.assertIn((2, 0), pairs)
        self.assertEqual(len(pairs), 3)

    def test_v5_group_pairwise_pairs_stay_within_decision_group(self):
        from run_world_model_v5_utility_ranking_smoke import build_group_pairwise_pairs

        utility = np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32)
        groups = np.array(["g0", "g0", "g1", "g1"])

        pairs = build_group_pairwise_pairs(utility, groups)

        self.assertIn((1, 0), pairs)
        self.assertIn((3, 2), pairs)
        self.assertNotIn((3, 0), pairs)
        self.assertEqual(len(pairs), 2)

    def test_v5_candidate_features_include_extended_action_fields(self):
        from run_world_model_v5_utility_ranking_smoke import build_candidate_features

        df = pd.DataFrame(
            [
                {
                    "rb_scale": 1.0,
                    "total_rb": 10,
                    "num_rb_tasks": 1,
                    "seed": 0,
                    "decision_time": 1.0,
                    "horizon": 3,
                    "action_family": "cpu_scale",
                    "cpu_scale": 1.5,
                    "total_cpu": 4.0,
                    "num_cpu_overrides": 2,
                    "num_return_route_overrides": 0,
                    "num_offload_overrides": 0,
                },
                {
                    "rb_scale": 1.0,
                    "total_rb": 0,
                    "num_rb_tasks": 0,
                    "seed": 0,
                    "decision_time": 1.0,
                    "horizon": 3,
                    "action_family": "return_route",
                    "cpu_scale": 1.0,
                    "total_cpu": 0.0,
                    "num_cpu_overrides": 0,
                    "num_return_route_overrides": 1,
                    "num_offload_overrides": 0,
                },
            ]
        )

        features = build_candidate_features(df)

        self.assertEqual(features.shape[0], 2)
        self.assertGreater(features.shape[1], 6)
        self.assertFalse(np.allclose(features[0], features[1]))

    def test_v5_candidate_features_include_offload_geometry_fields(self):
        from run_world_model_v5_utility_ranking_smoke import build_candidate_features

        base = {
            "rb_scale": 1.0,
            "total_rb": 4,
            "num_rb_tasks": 1,
            "seed": 0,
            "decision_time": 1.0,
            "horizon": 3,
            "action_family": "offload_target",
            "cpu_scale": 1.0,
            "total_cpu": 0.0,
            "num_cpu_overrides": 0,
            "num_return_route_overrides": 0,
            "num_offload_overrides": 1,
        }
        df = pd.DataFrame(
            [
                {
                    **base,
                    "offload_default_distance": 10.0,
                    "offload_alternative_distance": 30.0,
                    "offload_distance_delta": 20.0,
                    "offload_distance_ratio": 3.0,
                    "offload_default_is_rsu": 1.0,
                    "offload_alternative_is_uav": 1.0,
                    "offload_target_type_changed": 1.0,
                },
                {
                    **base,
                    "offload_default_distance": 10.0,
                    "offload_alternative_distance": 5.0,
                    "offload_distance_delta": -5.0,
                    "offload_distance_ratio": 0.5,
                    "offload_default_is_rsu": 1.0,
                    "offload_alternative_is_uav": 0.0,
                    "offload_target_type_changed": 0.0,
                },
            ]
        )

        with_geometry = build_candidate_features(df)
        without_geometry = build_candidate_features(
            df.drop(
                columns=[
                    "offload_default_distance",
                    "offload_alternative_distance",
                    "offload_distance_delta",
                    "offload_distance_ratio",
                    "offload_default_is_rsu",
                    "offload_alternative_is_uav",
                    "offload_target_type_changed",
                ]
            )
        )

        self.assertGreater(with_geometry.shape[1], without_geometry.shape[1])
        self.assertFalse(np.allclose(with_geometry[0], with_geometry[1]))

    def test_v5_rank_only_training_learns_simple_group_order(self):
        import torch

        from run_world_model_v5_utility_ranking_smoke import train_utility_head, predict_utility

        torch.manual_seed(42)
        features = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
        utility = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        groups = np.array(["g0", "g0", "g1", "g1"])

        model, _, _ = train_utility_head(
            features,
            utility,
            epochs=80,
            lr=0.05,
            hidden=4,
            device="cpu",
            groups=groups,
            pair_scope="group",
            reg_weight=0.0,
            rank_weight=1.0,
        )
        pred = predict_utility(model, features, "cpu")

        self.assertGreater(float(pred[1]), float(pred[0]))
        self.assertGreater(float(pred[3]), float(pred[2]))

    def test_v5_classical_scorer_uses_grouped_ranking_metrics(self):
        from run_world_model_v5_classical_ranker_v0 import evaluate_classical_model

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "target_utility": 1.0, "f0": 0.0},
                {"decision_group_id": "g0", "target_utility": 2.0, "f0": 1.0},
                {"decision_group_id": "g1", "target_utility": 1.0, "f0": 0.0},
                {"decision_group_id": "g1", "target_utility": 3.0, "f0": 2.0},
            ]
        )
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])

        out, pred = evaluate_classical_model(df, ["f0"], train_idx, test_idx, model_kind="ridge")

        self.assertEqual(len(pred), 4)
        self.assertAlmostEqual(out["test_top1_hit_mean"], 1.0)
        self.assertAlmostEqual(out["test_normalized_top1_regret_mean"], 0.0)

    def test_v5_dual_graph_decision_features_use_endpoint_physics(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import build_dual_graph_state_summary

        arrays = {
            "x_node": np.array(
                [
                    [
                        [[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 3.0]],
                        [[0.0, 0.0, 0.0, 2.0], [6.0, 8.0, 0.0, 5.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            "x_link": np.array(
                [
                    [
                        [[5.0, 10.0, 0.0], [8.0, 0.0, 0.0]],
                        [[10.0, 20.0, 0.0], [12.0, 0.0, 0.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            "edge_a_hist": np.array(
                [
                    [
                        [[0.0, 2.0], [0.0, 1.0]],
                        [[0.0, 4.0], [0.0, 0.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            "edge_src_idx": np.array([0, 0], dtype=np.int64),
            "edge_dst_idx": np.array([1, 1], dtype=np.int64),
            "valid_edge_node": np.array([1, 0], dtype=np.int64),
            "node_features": np.array(["x", "y", "z", "speed"]),
            "link_features": np.array(["distance", "rate_sum", "allocated_rb_count"]),
            "edge_action_features": np.array(["offload_count", "rb_total"]),
        }

        out = build_dual_graph_state_summary(arrays, sample_idx=0)

        self.assertAlmostEqual(out["dual_phy_distance_mean_last"], 10.0)
        self.assertAlmostEqual(out["dual_phy_speed_delta_mean_last"], 3.0)
        self.assertAlmostEqual(out["dual_comm_rate_sum_last"], 20.0)
        self.assertAlmostEqual(out["dual_action_rb_hist_sum"], 6.0)
        self.assertGreater(out["dual_comm_phy_rate_per_distance_last"], 0.0)

    def test_v5_dual_graph_decision_head_uses_grouped_metrics(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import evaluate_dual_graph_decision_head

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "target_utility": 1.0, "f0": 0.0, "dual_f0": 0.0},
                {"decision_group_id": "g0", "target_utility": 2.0, "f0": 0.0, "dual_f0": 1.0},
                {"decision_group_id": "g1", "target_utility": 1.0, "f0": 0.0, "dual_f0": 0.0},
                {"decision_group_id": "g1", "target_utility": 3.0, "f0": 0.0, "dual_f0": 2.0},
            ]
        )
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])

        out, pred = evaluate_dual_graph_decision_head(
            df,
            feature_cols=["f0", "dual_f0"],
            train_idx=train_idx,
            test_idx=test_idx,
            model_kind="ridge",
        )

        self.assertEqual(len(pred), 4)
        self.assertAlmostEqual(out["test_top1_hit_mean"], 1.0)
        self.assertAlmostEqual(out["test_normalized_top1_regret_mean"], 0.0)

    def test_v5_dual_graph_interactions_cross_action_and_state_features(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import build_interaction_features

        action = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        state = np.array([[11.0], [13.0]], dtype=np.float32)

        out, names = build_interaction_features(action, state, ["a0", "a1"], ["s0"])

        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(names, ["a0_x_s0", "a1_x_s0"])
        np.testing.assert_allclose(out[0], [22.0, 33.0])
        np.testing.assert_allclose(out[1], [65.0, 91.0])

    def test_v5_dual_graph_compact_features_keep_low_dimensional_state_action_terms(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import add_compact_dual_graph_features

        df = pd.DataFrame(
            [
                {
                    "total_rb": 10.0,
                    "num_offload_overrides": 1.0,
                    "dual_comm_rate_sum_last": 20.0,
                    "dual_phy_distance_mean_last": 5.0,
                    "dual_comm_active_ratio_last": 0.25,
                    "dual_comm_phy_rate_per_distance_last": 2.0,
                }
            ]
        )

        out, cols = add_compact_dual_graph_features(df)

        self.assertIn("compact_rb_x_rate", cols)
        self.assertIn("compact_offload_x_distance", cols)
        self.assertLessEqual(len(cols), 8)
        self.assertAlmostEqual(float(out.loc[0, "compact_rb_x_rate"]), np.log1p(10.0) * np.log1p(20.0), places=5)
        self.assertAlmostEqual(
            float(out.loc[0, "compact_offload_x_distance"]),
            np.log1p(1.0) * np.log1p(5.0),
            places=5,
        )

    def test_v5_family_ids_are_stable_for_known_action_families(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import build_action_family_ids

        df = pd.DataFrame(
            {
                "action_family": [
                    "rb_count",
                    "offload_target",
                    "mixed_offload_rb",
                    "unknown_family",
                ]
            }
        )

        ids, names = build_action_family_ids(df)

        self.assertEqual(names[:3], ["rb_count", "offload_target", "mixed_offload_rb"])
        self.assertEqual(ids.tolist()[:3], [0, 1, 2])
        self.assertEqual(int(ids[-1]), len(names) - 1)

    def test_v5_family_anchor_score_uses_resource_saving_prior(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import build_anchor_score

        df = pd.DataFrame(
            [
                {"total_rb": 2.0, "action_family": "rb_count"},
                {"total_rb": 8.0, "action_family": "mixed_offload_rb"},
            ]
        )

        score = build_anchor_score(df, mode="minus_total_rb")

        self.assertGreater(float(score[0]), float(score[1]))
        self.assertEqual(score.shape, (2,))

    def test_v5_family_anchor_score_can_use_throughput_rb_prior(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import build_anchor_score

        df = pd.DataFrame(
            [
                {"total_rb": 2.0, "action_family": "rb_count"},
                {"total_rb": 8.0, "action_family": "mixed_offload_rb"},
            ]
        )

        score = build_anchor_score(df, mode="plus_total_rb")

        self.assertLess(float(score[0]), float(score[1]))
        self.assertEqual(score.shape, (2,))

    def test_v5_family_specific_head_outputs_one_score_per_candidate(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import FamilySpecificUtilityHead
        import torch

        model = FamilySpecificUtilityHead(in_dim=3, num_families=4, hidden=5)
        x = torch.zeros((6, 3), dtype=torch.float32)
        family_ids = torch.tensor([0, 1, 2, 3, 1, 0], dtype=torch.long)
        anchor = torch.arange(6, dtype=torch.float32)

        out = model(x, family_ids, anchor)

        self.assertEqual(tuple(out.shape), (6,))
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))

    def test_v5_family_head_preserves_explicit_anchor_none(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import resolve_family_anchor_mode

        self.assertEqual(resolve_family_anchor_mode("family_mlp_rank", "none"), "none")
        self.assertEqual(resolve_family_anchor_mode("family_mlp_rank", "minus_total_rb"), "minus_total_rb")

    def test_v5_hybrid_selector_keeps_baseline_when_margin_small(self):
        from run_world_model_v5_hybrid_selector_v0 import apply_margin_hybrid_scores

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "total_rb": 1.0, "v5_predicted_utility": 0.10},
                {"decision_group_id": "g0", "total_rb": 5.0, "v5_predicted_utility": 0.12},
                {"decision_group_id": "g1", "total_rb": 5.0, "v5_predicted_utility": 0.10},
                {"decision_group_id": "g1", "total_rb": 1.0, "v5_predicted_utility": 0.80},
            ]
        )

        out = apply_margin_hybrid_scores(df, threshold=0.25)
        by_group = out.groupby("decision_group_id")["v5_predicted_utility"].idxmax()

        self.assertEqual(int(by_group.loc["g0"]), 0)
        self.assertEqual(int(by_group.loc["g1"]), 3)

    def test_v5_hybrid_selector_handles_filtered_dataframe_index(self):
        from run_world_model_v5_hybrid_selector_v0 import apply_margin_hybrid_scores

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "total_rb": 1.0, "v5_predicted_utility": 0.10},
                {"decision_group_id": "g0", "total_rb": 5.0, "v5_predicted_utility": 0.80},
            ],
            index=[10, 11],
        )

        out = apply_margin_hybrid_scores(df, threshold=0.25)

        self.assertEqual(len(out), 2)
        self.assertEqual(int(out["v5_predicted_utility"].idxmax()), 11)

    def test_v5_hybrid_selector_can_use_max_total_rb_baseline(self):
        from run_world_model_v5_hybrid_selector_v0 import apply_margin_hybrid_scores

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "total_rb": 1.0, "v5_predicted_utility": 0.80},
                {"decision_group_id": "g0", "total_rb": 5.0, "v5_predicted_utility": 0.70},
                {"decision_group_id": "g1", "total_rb": 5.0, "v5_predicted_utility": 0.10},
                {"decision_group_id": "g1", "total_rb": 1.0, "v5_predicted_utility": 0.80},
            ]
        )

        out = apply_margin_hybrid_scores(df, threshold=0.25, baseline_mode="max_total_rb")
        by_group = out.groupby("decision_group_id")["v5_predicted_utility"].idxmax()

        self.assertEqual(int(by_group.loc["g0"]), 1)
        self.assertEqual(int(by_group.loc["g1"]), 3)

    def test_v5_hybrid_selector_tiebreaks_baseline_with_original_learned_score(self):
        from run_world_model_v5_hybrid_selector_v0 import apply_margin_hybrid_scores

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "candidate_id": "a", "total_rb": 5.0, "v5_predicted_utility": 0.10},
                {"decision_group_id": "g0", "candidate_id": "b", "total_rb": 5.0, "v5_predicted_utility": 0.80},
            ]
        )

        out = apply_margin_hybrid_scores(df, threshold=1.0, baseline_mode="max_total_rb")
        chosen = int(out["v5_predicted_utility"].idxmax())

        self.assertEqual(chosen, 1)

    def test_v5_takeover_rows_use_only_observable_features_for_x(self):
        from run_world_model_v5_takeover_calibrator_v0 import build_takeover_rows

        df = pd.DataFrame(
            [
                {
                    "decision_group_id": "g0",
                    "candidate_id": "base",
                    "action_family": "rb_count",
                    "total_rb": 5.0,
                    "v5_predicted_utility": 0.1,
                    "target_utility": 2.0,
                },
                {
                    "decision_group_id": "g0",
                    "candidate_id": "learned",
                    "action_family": "offload_target",
                    "total_rb": 1.0,
                    "v5_predicted_utility": 0.9,
                    "target_utility": 3.0,
                },
            ]
        )

        rows = build_takeover_rows(df, baseline_mode="max_total_rb")

        self.assertEqual(int(rows.loc[0, "takeover_label"]), 1)
        self.assertGreater(float(rows.loc[0, "learned_margin"]), 0.0)
        self.assertNotIn("target_utility", rows.attrs["feature_columns"])
        self.assertNotIn("utility_spread", rows.attrs["feature_columns"])

    def test_v5_takeover_scores_apply_group_choices(self):
        from run_world_model_v5_takeover_calibrator_v0 import apply_takeover_scores

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "total_rb": 5.0, "v5_predicted_utility": 0.1},
                {"decision_group_id": "g0", "total_rb": 1.0, "v5_predicted_utility": 0.9},
                {"decision_group_id": "g1", "total_rb": 5.0, "v5_predicted_utility": 0.1},
                {"decision_group_id": "g1", "total_rb": 1.0, "v5_predicted_utility": 0.9},
            ]
        )
        takeover_rows = pd.DataFrame(
            [
                {"decision_group_id": "g0", "takeover_probability": 0.8},
                {"decision_group_id": "g1", "takeover_probability": 0.2},
            ]
        )

        out = apply_takeover_scores(df, takeover_rows, threshold=0.5, baseline_mode="max_total_rb")
        by_group = out.groupby("decision_group_id")["v5_predicted_utility"].idxmax()

        self.assertEqual(int(by_group.loc["g0"]), 1)
        self.assertEqual(int(by_group.loc["g1"]), 2)

    def test_v5_hybrid_selector_can_select_threshold_by_regret_first(self):
        from run_world_model_v5_hybrid_selector_v0 import select_threshold

        sweep = pd.DataFrame(
            [
                {"threshold": 0.1, "top1_hit_mean": 0.6, "normalized_top1_regret_mean": 0.2},
                {"threshold": 0.2, "top1_hit_mean": 0.5, "normalized_top1_regret_mean": 0.1},
            ]
        )

        threshold = select_threshold(sweep, selection_rule="regret_first")

        self.assertEqual(float(threshold), 0.2)

    def test_v5_hybrid_selector_keeps_top1_first_default(self):
        from run_world_model_v5_hybrid_selector_v0 import select_threshold

        sweep = pd.DataFrame(
            [
                {"threshold": 0.1, "top1_hit_mean": 0.6, "normalized_top1_regret_mean": 0.2},
                {"threshold": 0.2, "top1_hit_mean": 0.5, "normalized_top1_regret_mean": 0.1},
            ]
        )

        threshold = select_threshold(sweep, selection_rule="top1_first")

        self.assertEqual(float(threshold), 0.1)

    def test_v5_selector_probe_global_threshold_uses_train_rows(self):
        from run_world_model_v5_selector_probe_v0 import fit_global_threshold

        rows = pd.DataFrame(
            [
                {
                    "split": "train",
                    "learned_margin": 0.1,
                    "true_hit_learned": 0.0,
                    "true_hit_rb_tiebreak": 1.0,
                    "regret_learned": 1.0,
                    "regret_rb_tiebreak": 0.0,
                    "spearman_learned_choice": 0.0,
                    "spearman_rb_tiebreak_choice": 1.0,
                },
                {
                    "split": "train",
                    "learned_margin": 0.4,
                    "true_hit_learned": 0.0,
                    "true_hit_rb_tiebreak": 1.0,
                    "regret_learned": 1.0,
                    "regret_rb_tiebreak": 0.0,
                    "spearman_learned_choice": 0.0,
                    "spearman_rb_tiebreak_choice": 1.0,
                },
            ]
        )

        thresholds, sweep = fit_global_threshold(rows, [-1.0, 0.2, 0.5], "top1")

        self.assertEqual(thresholds, {"global": 0.5})
        self.assertIn("top1", sweep.columns)

    def test_v5_selector_probe_family_threshold_falls_back_to_global(self):
        from run_world_model_v5_selector_probe_v0 import apply_thresholds

        rows = pd.DataFrame(
            [
                {
                    "learned_family": "offload_target",
                    "family_pair": "offload_target|rb_count",
                    "learned_margin": 0.1,
                    "true_hit_learned": 1.0,
                    "true_hit_rb_tiebreak": 0.0,
                    "regret_learned": 0.0,
                    "regret_rb_tiebreak": 1.0,
                    "spearman_learned_choice": 1.0,
                    "spearman_rb_tiebreak_choice": 0.0,
                },
                {
                    "learned_family": "rare_family",
                    "family_pair": "rare_family|rb_count",
                    "learned_margin": 0.1,
                    "true_hit_learned": 0.0,
                    "true_hit_rb_tiebreak": 1.0,
                    "regret_learned": 1.0,
                    "regret_rb_tiebreak": 0.0,
                    "spearman_learned_choice": 0.0,
                    "spearman_rb_tiebreak_choice": 1.0,
                },
            ]
        )

        metrics = apply_thresholds(rows, {"global": 0.5, "offload_target": 0.0}, "learned_family")

        self.assertAlmostEqual(metrics["top1"], 1.0)
        self.assertAlmostEqual(metrics["take_rate"], 0.5)

    def test_v5_group_winner_loss_prefers_best_candidate(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import group_winner_cross_entropy
        import torch

        pred = torch.tensor([0.1, 1.0, 0.2, 0.4], dtype=torch.float32)
        utility = torch.tensor([0.0, 2.0, 0.0, 1.0], dtype=torch.float32)
        groups = np.array(["g0", "g0", "g1", "g1"])

        good_loss = group_winner_cross_entropy(pred, utility, groups)
        bad_loss = group_winner_cross_entropy(-pred, utility, groups)

        self.assertLess(float(good_loss), float(bad_loss))

    def test_v5_group_winner_loss_can_weight_by_utility_gap(self):
        from run_world_model_v5_dual_graph_decision_head_v0 import group_winner_cross_entropy
        import torch

        pred = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        utility = torch.tensor([10.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        groups = np.array(["large_gap", "large_gap", "small_gap", "small_gap"])
        weights = group_winner_cross_entropy(
            pred,
            utility,
            groups,
            gap_weight_power=1.0,
            return_group_weights=True,
        )[1]

        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_v5_family_winner_batch_uses_mainline_configuration(self):
        from run_world_model_v5_family_winner_seedheldout_batch_v0 import SEED_PAIRS, parse_args

        args = parse_args([])

        self.assertEqual(SEED_PAIRS, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.epochs, 120)
        self.assertEqual(args.hidden, 8)
        self.assertAlmostEqual(args.lr, 0.003)
        self.assertAlmostEqual(args.reg_weight, 0.2)
        self.assertAlmostEqual(args.winner_weight, 0.5)
        self.assertAlmostEqual(args.winner_gap_weight_power, 0.0)
        self.assertEqual(args.anchor_mode, "none")

        plus_args = parse_args(["--anchor-mode", "plus_total_rb"])
        self.assertEqual(plus_args.anchor_mode, "plus_total_rb")
        gap_args = parse_args(["--winner-gap-weight-power", "1.0"])
        self.assertAlmostEqual(gap_args.winner_gap_weight_power, 1.0)

    def test_v5_hard_pair_diagnostics_identify_choice_families(self):
        from run_world_model_v5_hard_pair_diagnostics_v0 import summarize_decision_groups

        df = pd.DataFrame(
            [
                {
                    "decision_group_id": "g0",
                    "candidate_id": "low_rb",
                    "action_family": "rb_count",
                    "total_rb": 1,
                    "target_utility": 2.0,
                    "v5_predicted_utility": 0.1,
                    "split": "test",
                },
                {
                    "decision_group_id": "g0",
                    "candidate_id": "mixed",
                    "action_family": "mixed_offload_rb",
                    "total_rb": 5,
                    "target_utility": 1.0,
                    "v5_predicted_utility": 1.0,
                    "split": "test",
                },
            ]
        )

        out = summarize_decision_groups(df)

        self.assertEqual(out.loc[0, "true_family"], "rb_count")
        self.assertEqual(out.loc[0, "learned_family"], "mixed_offload_rb")
        self.assertEqual(out.loc[0, "heuristic_family"], "rb_count")
        self.assertAlmostEqual(float(out.loc[0, "learned_regret"]), 1.0)

    def test_v5_score_blend_selects_weight_using_train_groups(self):
        from run_world_model_v5_score_blend_v0 import evaluate_blend_candidates

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "split": "train", "target_utility": 1.0, "v5_predicted_utility": 2.0, "total_rb": 1.0, "candidate_id": "low"},
                {"decision_group_id": "g0", "split": "train", "target_utility": 2.0, "v5_predicted_utility": 0.0, "total_rb": 9.0, "candidate_id": "high"},
                {"decision_group_id": "g1", "split": "test", "target_utility": 1.0, "v5_predicted_utility": 2.0, "total_rb": 1.0, "candidate_id": "low"},
                {"decision_group_id": "g1", "split": "test", "target_utility": 2.0, "v5_predicted_utility": 0.0, "total_rb": 9.0, "candidate_id": "high"},
            ]
        )

        selected, metrics = evaluate_blend_candidates(df, lambdas=[0.0, 1.0], prior_cols=["predict_total_rb"])

        self.assertEqual(selected["prior_col"], "predict_total_rb")
        self.assertAlmostEqual(float(selected["lambda"]), 1.0)
        test = metrics[(metrics["split"] == "test") & (metrics["is_selected"])]
        self.assertAlmostEqual(float(test.iloc[0]["top1_hit_mean"]), 1.0)

    def test_v5_gap_model_selection_uses_train_metrics_only(self):
        from run_world_model_v5_gap_model_selection_v0 import select_models_by_train_metrics

        candidates = pd.DataFrame(
            [
                {
                    "seed_pair": "seed01",
                    "model": "gap0",
                    "train_top1": 0.90,
                    "train_regret": 0.20,
                    "train_spearman": 0.3,
                    "test_top1": 0.1,
                    "test_regret": 0.9,
                },
                {
                    "seed_pair": "seed01",
                    "model": "gap1",
                    "train_top1": 0.80,
                    "train_regret": 0.05,
                    "train_spearman": 0.2,
                    "test_top1": 0.9,
                    "test_regret": 0.1,
                },
                {
                    "seed_pair": "seed23",
                    "model": "gap0",
                    "train_top1": 0.70,
                    "train_regret": 0.10,
                    "train_spearman": 0.5,
                    "test_top1": 0.8,
                    "test_regret": 0.2,
                },
                {
                    "seed_pair": "seed23",
                    "model": "gap1",
                    "train_top1": 0.70,
                    "train_regret": 0.20,
                    "train_spearman": 0.9,
                    "test_top1": 1.0,
                    "test_regret": 0.0,
                },
            ]
        )

        selected = select_models_by_train_metrics(candidates, rule="regret_first")

        self.assertEqual(selected.set_index("seed_pair").loc["seed01", "model"], "gap1")
        self.assertEqual(selected.set_index("seed_pair").loc["seed23", "model"], "gap0")

        top1_selected = select_models_by_train_metrics(candidates, rule="top1_first")

        self.assertEqual(top1_selected.set_index("seed_pair").loc["seed01", "model"], "gap0")
        self.assertEqual(top1_selected.set_index("seed_pair").loc["seed23", "model"], "gap0")

    def test_world_model_builders_accept_custom_output_dirs(self):
        from build_edge_action_v0 import build_edge_action_dataset
        from build_world_model_dataset_v0 import build_world_model_dataset

        edge_sig = inspect.signature(build_edge_action_dataset)
        world_sig = inspect.signature(build_world_model_dataset)

        self.assertIn("dataset_dir", edge_sig.parameters)
        self.assertIn("strict_action_dir", edge_sig.parameters)
        self.assertIn("output_dir", edge_sig.parameters)
        self.assertIn("state_dir", world_sig.parameters)
        self.assertIn("edge_action_dir", world_sig.parameters)
        self.assertIn("output_dir", world_sig.parameters)

    def test_v5_candidate_features_include_rb_and_throughput(self):
        from run_world_model_v5_utility_ranking_smoke import build_candidate_features

        df = pd.DataFrame(
            [
                {
                    "seed": 4,
                    "decision_time": 2.3,
                    "horizon": 3,
                    "rb_scale": 1.0,
                    "total_rb": 50,
                    "num_rb_tasks": 1,
                    "throughput": 100.0,
                    "delta_done": 1,
                    "delta_failed": 0,
                },
                {
                    "seed": 4,
                    "decision_time": 2.3,
                    "horizon": 3,
                    "rb_scale": 0.5,
                    "total_rb": 25,
                    "num_rb_tasks": 1,
                    "throughput": 40.0,
                    "delta_done": 1,
                    "delta_failed": 0,
                },
            ]
        )
        features = build_candidate_features(df)

        self.assertEqual(features.shape, (2, 6))
        self.assertGreater(float(features[0, 1]), float(features[1, 1]))
        self.assertAlmostEqual(float(features[0, 2]), float(features[1, 2]))

    def test_v5_decision_features_exclude_outcome_leakage(self):
        from run_world_model_v5_utility_ranking_smoke import build_candidate_features

        df = pd.DataFrame(
            [
                {
                    "seed": 4,
                    "decision_time": 2.3,
                    "horizon": 3,
                    "rb_scale": 1.0,
                    "total_rb": 50,
                    "num_rb_tasks": 1,
                    "throughput": 100.0,
                    "delta_done": 1.0,
                    "delta_failed": 0.0,
                },
                {
                    "seed": 4,
                    "decision_time": 2.3,
                    "horizon": 3,
                    "rb_scale": 1.0,
                    "total_rb": 50,
                    "num_rb_tasks": 1,
                    "throughput": 1.0,
                    "delta_done": 0.0,
                    "delta_failed": 3.0,
                },
            ]
        )

        no_leak = build_candidate_features(df, include_outcome_features=False)

        np.testing.assert_allclose(no_leak[0], no_leak[1])

    def test_v5_state_summary_features_use_only_history(self):
        from run_world_model_v5_utility_ranking_smoke import summarize_world_model_state

        arrays = {
            "x_task": np.array(
                [
                    [
                        [1.0, 10.0, 0.0],
                        [3.0, 20.0, 2.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "x_link": np.array(
                [
                    [
                        [[1.0, 0.0], [2.0, 5.0]],
                        [[3.0, 10.0], [4.0, 0.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            "x_node": np.array(
                [
                    [
                        [[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 2.0]],
                        [[0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 12.0, 7.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            "edge_a_hist": np.array(
                [
                    [
                        [[0.0, 1.0], [0.0, 2.0]],
                        [[0.0, 3.0], [0.0, 4.0]],
                    ]
                ],
                dtype=np.float32,
            ),
            "task_features": np.array(["num_tasks", "total_task_size", "num_returning"]),
            "link_features": np.array(["distance", "rate_sum"]),
            "node_features": np.array(["x", "y", "z", "speed"]),
            "edge_action_features": np.array(["offload_count", "rb_total"]),
        }

        out = summarize_world_model_state(arrays, sample_idx=0)

        self.assertAlmostEqual(out["state_num_tasks_last"], 3.0)
        self.assertAlmostEqual(out["state_total_task_size_last"], 20.0)
        self.assertAlmostEqual(out["state_active_link_ratio_last"], 0.5)
        self.assertAlmostEqual(out["state_rate_sum_last"], 10.0)
        self.assertAlmostEqual(out["state_mean_node_speed_last"], 6.0)
        self.assertAlmostEqual(out["state_rb_total_hist_sum"], 10.0)

    def test_v5_state_feature_enrichment_aligns_by_seed_and_time(self):
        from run_world_model_v5_utility_ranking_smoke import enrich_candidates_with_state_features

        df = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 0.8, "candidate_id": "a"},
                {"seed": 0, "decision_time": 0.3, "candidate_id": "b"},
            ]
        )
        sample_index = pd.DataFrame(
            [
                {"sample_id": 5, "seed": 0, "input_end_time": 0.8},
            ]
        )
        arrays = {
            "x_task": np.ones((6, 1, 1), dtype=np.float32),
            "x_link": np.zeros((6, 1, 1, 1), dtype=np.float32),
            "x_node": np.zeros((6, 1, 1, 4), dtype=np.float32),
            "edge_a_hist": np.zeros((6, 1, 1, 1), dtype=np.float32),
            "task_features": np.array(["num_tasks"]),
            "link_features": np.array(["rate_sum"]),
            "node_features": np.array(["x", "y", "z", "speed"]),
            "edge_action_features": np.array(["rb_total"]),
        }

        out = enrich_candidates_with_state_features(df, sample_index, arrays)

        self.assertEqual(int(out.loc[0, "state_available"]), 1)
        self.assertEqual(int(out.loc[1, "state_available"]), 0)
        self.assertAlmostEqual(float(out.loc[0, "state_num_tasks_last"]), 1.0)
        self.assertAlmostEqual(float(out.loc[1, "state_num_tasks_last"]), 0.0)

    def test_v5_filter_state_available_groups_removes_partial_groups(self):
        from run_world_model_v5_utility_ranking_smoke import filter_state_available_groups

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "state_available": 1.0, "candidate_id": "a"},
                {"decision_group_id": "g0", "state_available": 1.0, "candidate_id": "b"},
                {"decision_group_id": "g1", "state_available": 1.0, "candidate_id": "a"},
                {"decision_group_id": "g1", "state_available": 0.0, "candidate_id": "b"},
                {"decision_group_id": "g2", "state_available": 0.0, "candidate_id": "a"},
            ]
        )

        out = filter_state_available_groups(df)

        self.assertEqual(set(out["decision_group_id"]), {"g0"})
        self.assertEqual(len(out), 2)

    def test_v5_group_split_keeps_decision_groups_separate(self):
        from run_world_model_v5_utility_ranking_smoke import split_decision_groups

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "candidate_id": "a"},
                {"decision_group_id": "g0", "candidate_id": "b"},
                {"decision_group_id": "g1", "candidate_id": "a"},
                {"decision_group_id": "g1", "candidate_id": "b"},
                {"decision_group_id": "g2", "candidate_id": "a"},
                {"decision_group_id": "g2", "candidate_id": "b"},
            ]
        )

        train_idx, test_idx = split_decision_groups(df, test_fraction=1 / 3, seed=7)
        train_groups = set(df.iloc[train_idx]["decision_group_id"])
        test_groups = set(df.iloc[test_idx]["decision_group_id"])

        self.assertTrue(train_groups)
        self.assertTrue(test_groups)
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_v5_seed_split_holds_out_requested_seeds(self):
        from run_world_model_v5_utility_ranking_smoke import split_by_test_seeds

        df = pd.DataFrame(
            [
                {"seed": 0, "decision_group_id": "s0_g0"},
                {"seed": 0, "decision_group_id": "s0_g0"},
                {"seed": 1, "decision_group_id": "s1_g0"},
                {"seed": 2, "decision_group_id": "s2_g0"},
            ]
        )

        train_idx, test_idx = split_by_test_seeds(df, test_seeds=[1, 2])

        self.assertEqual(set(df.iloc[test_idx]["seed"]), {1, 2})
        self.assertEqual(set(df.iloc[train_idx]["seed"]), {0})

    def test_v5_group_ranking_metrics_average_per_decision_group(self):
        from run_world_model_v5_utility_ranking_smoke import grouped_ranking_metrics

        df = pd.DataFrame(
            [
                {"decision_group_id": "g0", "target_utility": 3.0, "v5_predicted_utility": 3.0},
                {"decision_group_id": "g0", "target_utility": 2.0, "v5_predicted_utility": 2.0},
                {"decision_group_id": "g1", "target_utility": 5.0, "v5_predicted_utility": 1.0},
                {"decision_group_id": "g1", "target_utility": 1.0, "v5_predicted_utility": 5.0},
            ]
        )

        out = grouped_ranking_metrics(df)

        self.assertEqual(out["num_groups"], 2)
        self.assertAlmostEqual(out["top1_hit_mean"], 0.5)
        self.assertAlmostEqual(out["normalized_top1_regret_mean"], 0.5)
        self.assertAlmostEqual(out["spearman_mean"], 0.0)

    def test_counterfactual_batch_summary_groups_by_decision(self):
        from run_airfogsim_counterfactual_label_dataset_v0 import summarize_label_dataset

        df = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "candidate_id": "a", "airfogsim_utility": 1.0},
                {"seed": 0, "decision_time": 1.0, "candidate_id": "b", "airfogsim_utility": 2.0},
                {"seed": 1, "decision_time": 2.0, "candidate_id": "a", "airfogsim_utility": 3.0},
            ]
        )
        summary = summarize_label_dataset(df)

        self.assertEqual(summary["num_rows"], 3)
        self.assertEqual(summary["num_decision_groups"], 2)
        self.assertAlmostEqual(summary["mean_candidates_per_group"], 1.5)

    def test_v5_stage_diagnostics_summarize_by_decision_stage(self):
        from run_world_model_v5_stage_diagnostics_v0 import summarize_stage_metrics

        predictions = pd.DataFrame(
            [
                {
                    "decision_group_id": "g0",
                    "target_utility": 3.0,
                    "v5_predicted_utility": 3.0,
                    "split": "test",
                },
                {
                    "decision_group_id": "g0",
                    "target_utility": 1.0,
                    "v5_predicted_utility": 1.0,
                    "split": "test",
                },
                {
                    "decision_group_id": "g1",
                    "target_utility": 4.0,
                    "v5_predicted_utility": 0.0,
                    "split": "test",
                },
                {
                    "decision_group_id": "g1",
                    "target_utility": 2.0,
                    "v5_predicted_utility": 5.0,
                    "split": "test",
                },
            ]
        )
        points = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "decision_stage": "offload_rb"},
                {"seed": 1, "decision_time": 2.0, "decision_stage": "compute"},
            ]
        )
        predictions["seed"] = [0, 0, 1, 1]
        predictions["decision_time"] = [1.0, 1.0, 2.0, 2.0]

        out = summarize_stage_metrics(predictions, points, model_name="toy")
        test = out[out["split"].eq("test")].set_index("decision_stage")

        self.assertEqual(int(test.loc["offload_rb", "num_groups"]), 1)
        self.assertAlmostEqual(float(test.loc["offload_rb", "top1_hit_mean"]), 1.0)
        self.assertEqual(int(test.loc["compute", "num_groups"]), 1)
        self.assertAlmostEqual(float(test.loc["compute", "top1_hit_mean"]), 0.0)
        self.assertAlmostEqual(float(test.loc["compute", "normalized_top1_regret_mean"]), 1.0)

    def test_v5_stage_baselines_include_grouped_train_test_split(self):
        from run_world_model_v5_stage_diagnostics_v0 import collect_baseline_stage_metrics

        labels = pd.DataFrame(
            [
                {
                    "seed": 0,
                    "decision_time": 1.0,
                    "decision_group_id": "g0",
                    "candidate_id": "default",
                    "action_family": "rb_count",
                    "airfogsim_utility": 2.0,
                    "total_rb": 50.0,
                    "rb_scale": 1.0,
                    "num_rb_tasks": 1,
                },
                {
                    "seed": 0,
                    "decision_time": 1.0,
                    "decision_group_id": "g0",
                    "candidate_id": "low",
                    "action_family": "rb_count",
                    "airfogsim_utility": 1.0,
                    "total_rb": 1.0,
                    "rb_scale": 0.02,
                    "num_rb_tasks": 1,
                },
                {
                    "seed": 1,
                    "decision_time": 2.0,
                    "decision_group_id": "g1",
                    "candidate_id": "default",
                    "action_family": "rb_count",
                    "airfogsim_utility": 3.0,
                    "total_rb": 50.0,
                    "rb_scale": 1.0,
                    "num_rb_tasks": 1,
                },
                {
                    "seed": 1,
                    "decision_time": 2.0,
                    "decision_group_id": "g1",
                    "candidate_id": "low",
                    "action_family": "mixed_offload_rb",
                    "airfogsim_utility": 4.0,
                    "total_rb": 1.0,
                    "rb_scale": 0.02,
                    "num_rb_tasks": 1,
                },
            ]
        )
        points = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "decision_stage": "offload_rb"},
                {"seed": 1, "decision_time": 2.0, "decision_stage": "offload_rb"},
            ]
        )

        out = collect_baseline_stage_metrics(labels, points, rb_penalty=0.001, test_fraction=0.5, split_seed=1)

        self.assertIn("train", set(out["split"]))
        self.assertIn("test", set(out["split"]))
        self.assertIn("all", set(out["split"]))
        for split_name in ["train", "test"]:
            groups = out[out["split"].eq(split_name)]["num_groups"].unique()
            self.assertEqual(groups.tolist(), [1])

    def test_v5_stage_difficulty_flags_tie_like_groups(self):
        from run_world_model_v5_stage_diagnostics_v0 import summarize_stage_difficulty

        labels = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "decision_group_id": "g0", "resource_aware_utility": 1.0},
                {"seed": 0, "decision_time": 1.0, "decision_group_id": "g0", "resource_aware_utility": 1.0},
                {"seed": 1, "decision_time": 2.0, "decision_group_id": "g1", "resource_aware_utility": 3.0},
                {"seed": 1, "decision_time": 2.0, "decision_group_id": "g1", "resource_aware_utility": 1.0},
            ]
        )
        points = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "decision_stage": "compute"},
                {"seed": 1, "decision_time": 2.0, "decision_stage": "offload_rb"},
            ]
        )

        out = summarize_stage_difficulty(labels, points, utility_col="resource_aware_utility", tie_epsilon=1e-6)
        by_stage = out.set_index("decision_stage")

        self.assertEqual(int(by_stage.loc["compute", "num_groups"]), 1)
        self.assertEqual(int(by_stage.loc["compute", "num_nontrivial_groups"]), 0)
        self.assertEqual(int(by_stage.loc["offload_rb", "num_nontrivial_groups"]), 1)
        self.assertAlmostEqual(float(by_stage.loc["offload_rb", "mean_utility_spread"]), 2.0)


if __name__ == "__main__":
    unittest.main()
