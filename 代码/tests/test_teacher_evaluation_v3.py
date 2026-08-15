from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def trajectory_arrays() -> dict[str, np.ndarray]:
    time_count = 3
    arrays = {
        "time": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        "physical_node_state": np.zeros((time_count, 1, 9), dtype=np.float32),
        "physical_node_feature_mask": np.ones((time_count, 1, 9), dtype=bool),
        "physical_node_present": np.ones((time_count, 1), dtype=bool),
        "physical_edge_state": np.zeros((time_count, 1, 7), dtype=np.float32),
        "physical_edge_feature_mask": np.ones((time_count, 1, 7), dtype=bool),
        "physical_edge_present": np.ones((time_count, 1), dtype=bool),
        "information_node_state": np.zeros((time_count, 1, 7), dtype=np.float32),
        "information_node_feature_mask": np.ones((time_count, 1, 7), dtype=bool),
        "information_node_present": np.ones((time_count, 1), dtype=bool),
        "information_edge_state": np.zeros((time_count, 1, 18), dtype=np.float32),
        "information_edge_feature_mask": np.ones((time_count, 1, 18), dtype=bool),
        "information_edge_present": np.ones((time_count, 1), dtype=bool),
        "data_flow_state": np.zeros((time_count, 1, 5), dtype=np.float32),
        "data_flow_present": np.asarray([[0], [1], [1]], dtype=bool),
        "data_flow_valid": np.asarray([1], dtype=bool),
        "task_state": np.zeros((time_count, 1, 8), dtype=np.float32),
        "task_present": np.asarray([[0], [1], [1]], dtype=bool),
        "task_valid": np.asarray([1], dtype=bool),
        "task_lifecycle_index": np.asarray([[-1], [1], [2]], dtype=np.int16),
    }
    arrays["physical_node_state"][:, 0, 0] = [1.0, 2.0, 4.0]
    arrays["physical_node_state"][:, 0, 3] = [1.0, 3.0, 6.0]
    arrays["physical_edge_state"][:, 0, 3] = [1.0, 2.0, 4.0]
    arrays["physical_edge_state"][:, 0, 4] = [0.0, 1.0, 3.0]
    arrays["information_node_state"][:, 0, 0] = [0.0, 1.0, 3.0]
    arrays["information_node_state"][:, 0, 2] = [0.0, 2.0, 5.0]
    arrays["information_edge_state"][:, 0, 11] = [0.0, 1.0, 0.0]
    arrays["information_edge_state"][:, 0, 12] = [0.0, 10.0, 0.0]
    return arrays


def train_normalization_stats() -> dict[str, object]:
    return {
        "source_split": "train",
        "features": {
            "physical_node_state": {"mean": [3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "physical_edge_state": {"mean": [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]},
            "information_node_state": {"mean": [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]},
            "information_edge_state": {"mean": [0.0] * 11 + [0.0, 5.0] + [0.0] * 5},
            "data_flow_state": {"mean": [0.0] * 5},
            "task_state": {"mean": [0.0] * 8},
            "task_dag_state": {"mean": [0.0] * 3},
        },
    }


class TeacherEvaluationV3Tests(unittest.TestCase):
    def test_last_persistence_uses_information_edges_and_one_step_history(self):
        from pi_jwm.teacher_evaluation_v3 import evaluate_teacher_trajectory

        report = evaluate_teacher_trajectory(trajectory_arrays(), method="last_persistence")
        metrics = report["metrics"]

        self.assertAlmostEqual(
            math.sqrt(2.5), metrics["state.physical_node.position.rmse"]["value"]
        )
        self.assertAlmostEqual(
            10.0, metrics["link.active_only_rate.mae"]["value"]
        )
        self.assertEqual(
            0.0, metrics["event.information_link_activity.f1"]["value"]
        )
        self.assertEqual(2, metrics["event.information_link_activity.f1"]["count"])
        self.assertEqual(
            "information_edge_state.outcome.active_task_count",
            metrics["event.information_link_activity.f1"]["source_fields"][0],
        )
        self.assertEqual(
            "not_computable", metrics["uncertainty.nll"]["status"]
        )

    def test_masks_exclude_unobserved_targets_and_zero_baseline_is_finite(self):
        from pi_jwm.teacher_evaluation_v3 import evaluate_teacher_trajectory

        arrays = trajectory_arrays()
        arrays["physical_node_state"][2, 0, 0] = 1e9
        arrays["physical_node_feature_mask"][2, 0, 0] = False
        report = evaluate_teacher_trajectory(arrays, method="zero_state")
        metrics = report["metrics"]

        self.assertEqual(2.0, metrics["state.physical_node.position.rmse"]["value"])
        self.assertEqual(0.0, metrics["event.information_link_activity.f1"]["value"])
        for metric in metrics.values():
            if metric["status"] == "computed":
                self.assertTrue(math.isfinite(float(metric["value"])))

    def test_zero_state_uses_train_mean_in_physical_units_when_stats_are_supplied(self):
        from pi_jwm.teacher_evaluation_v3 import evaluate_teacher_trajectory

        report = evaluate_teacher_trajectory(
            trajectory_arrays(),
            method="zero_state",
            normalization_stats=train_normalization_stats(),
        )
        self.assertEqual("train_feature_mean", report["continuous_baseline"])
        self.assertEqual(1.0, report["metrics"]["state.physical_node.position.rmse"]["value"])
        self.assertEqual(5.0, report["metrics"]["link.active_only_rate.mae"]["value"])

    def test_checkpoint_continuous_term_is_executable_with_frozen_train_scales(self):
        from pi_jwm.teacher_evaluation_v3 import (
            SELECTION_COMPONENTS,
            evaluate_teacher_trajectory,
            summarize_teacher_reports,
        )

        arrays = trajectory_arrays()
        arrays["task_dag_state"] = np.zeros((3, 1, 3), dtype=np.float32)
        arrays["task_dag_state_present"] = np.ones((3, 1), dtype=bool)
        report = evaluate_teacher_trajectory(
            arrays,
            method="zero_state",
            normalization_stats=train_normalization_stats(),
        )
        trajectory_selection = report["metrics"]["selection.required_continuous.normalized_error"]
        summary = summarize_teacher_reports(
            [report],
            selection_scales={metric_id: 1.0 for metric_id in SELECTION_COMPONENTS},
        )
        selection = summary["metrics"]["selection.required_continuous.normalized_error"]

        self.assertEqual("not_computable", trajectory_selection["status"])
        self.assertEqual("computed", selection["status"])
        self.assertEqual(10, selection["pooled_micro"]["count"])
        self.assertTrue(math.isfinite(selection["macro_mean"]))

    def test_dag_metric_uses_unfinished_parent_count_not_parent_count(self):
        from pi_jwm.teacher_evaluation_v3 import evaluate_teacher_trajectory

        arrays = trajectory_arrays()
        arrays["task_dag_state"] = np.asarray(
            [
                [[100.0, 1.0, 0.0]],
                [[200.0, 3.0, 1.0]],
                [[400.0, 8.0, 0.0]],
            ],
            dtype=np.float32,
        )
        arrays["task_dag_state_present"] = np.ones((3, 1), dtype=bool)
        stats = train_normalization_stats()
        stats["features"]["task_dag_state"]["mean"] = [1000.0, 2.0, 0.5]

        zero = evaluate_teacher_trajectory(
            arrays, method="zero_state", normalization_stats=stats
        )["metrics"]["state.dag.unfinished_parent_count.mae"]
        last = evaluate_teacher_trajectory(
            arrays, method="last_persistence", normalization_stats=stats
        )["metrics"]["state.dag.unfinished_parent_count.mae"]

        self.assertEqual(3.5, zero["value"])
        self.assertEqual(3.5, last["value"])
        self.assertEqual(2, zero["count"])

    def test_last_persistence_excludes_transitions_with_unobserved_history(self):
        from pi_jwm.teacher_evaluation_v3 import evaluate_teacher_trajectory

        arrays = trajectory_arrays()
        arrays["physical_node_state"][0, 0, 0] = 1e9
        arrays["physical_node_feature_mask"][0, 0, 0] = False
        report = evaluate_teacher_trajectory(arrays, method="last_persistence")
        position = report["metrics"]["state.physical_node.position.rmse"]

        self.assertEqual(1, position["count"])
        self.assertEqual(2.0, position["value"])

    def test_lifecycle_macro_f1_uses_the_frozen_five_class_label_space(self):
        from pi_jwm.teacher_evaluation_v3 import _macro_f1

        metric = _macro_f1(
            np.asarray([0, 1]),
            np.asarray([0, 0]),
            np.asarray([True, True]),
            previous_valid=np.asarray([True, True]),
            method="last_persistence",
        )

        self.assertAlmostEqual(2.0 / 15.0, metric["value"])
        self.assertEqual(5.0, metric["denominator"])

    def test_summary_macro_averages_computed_seed_values_and_preserves_na(self):
        from pi_jwm.teacher_evaluation_v3 import (
            evaluate_teacher_trajectory,
            summarize_teacher_reports,
        )

        first = evaluate_teacher_trajectory(trajectory_arrays(), method="last_persistence")
        second_arrays = trajectory_arrays()
        second_arrays["information_edge_state"][:, 0, 12] = [0.0, 20.0, 0.0]
        second = evaluate_teacher_trajectory(second_arrays, method="last_persistence")
        summary = summarize_teacher_reports([first, second])

        rate = summary["metrics"]["link.active_only_rate.mae"]
        self.assertEqual(2, rate["computed_seed_count"])
        self.assertEqual(2, rate["computed_trajectory_count"])
        self.assertEqual(15.0, rate["macro_mean"])
        self.assertEqual(15.0, rate["pooled_micro"]["value"])
        self.assertEqual([10.0, 20.0], rate["bootstrap_95_ci"])
        self.assertEqual(
            "computed",
            summary["metrics"]["event.information_link_activity.auprc"]["pooled_micro"]["status"],
        )
        self.assertEqual(
            0.5,
            summary["metrics"]["event.information_link_activity.auprc"]["pooled_micro"]["value"],
        )
        self.assertEqual(
            "not_computable", summary["metrics"]["uncertainty.nll"]["status"]
        )

        one_report = summarize_teacher_reports([first])
        self.assertIsNone(
            one_report["metrics"]["link.active_only_rate.mae"]["sample_std"]
        )
        self.assertEqual(
            [None, None],
            one_report["metrics"]["link.active_only_rate.mae"]["bootstrap_95_ci"],
        )

    def test_invalid_method_and_short_trajectory_are_rejected(self):
        from pi_jwm.teacher_evaluation_v3 import evaluate_teacher_trajectory

        with self.assertRaisesRegex(ValueError, "method"):
            evaluate_teacher_trajectory(trajectory_arrays(), method="oracle")
        short = trajectory_arrays()
        short = {name: value[:1] if value.ndim and value.shape[0] == 3 else value for name, value in short.items()}
        with self.assertRaisesRegex(ValueError, "two time steps"):
            evaluate_teacher_trajectory(short, method="last_persistence")

        invalid_lifecycle = trajectory_arrays()
        invalid_lifecycle["task_lifecycle_index"][2, 0] = 5
        with self.assertRaisesRegex(ValueError, r"lifecycle.*\[0, 4\]"):
            evaluate_teacher_trajectory(invalid_lifecycle, method="zero_state")


if __name__ == "__main__":
    unittest.main()
