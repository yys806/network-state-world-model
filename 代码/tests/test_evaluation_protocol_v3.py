from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class EvaluationProtocolV3Tests(unittest.TestCase):
    @staticmethod
    def _protocol():
        from pi_jwm.evaluation_protocol_v3 import build_fair_experiment_protocol

        return build_fair_experiment_protocol(
            environment_splits={
                "train": ["train-1"],
                "validation": ["validation-1"],
                "calibration": ["calibration-1"],
                "locked_test": ["locked-1"],
            },
            normalization_stats_sha256="a" * 64,
        )

    def test_registry_is_complete_auditable_and_teacher_aligned(self):
        from pi_jwm.evaluation_protocol_v3 import build_factual_metric_mapping, build_metric_registry

        registry = build_metric_registry()
        by_id = {row["metric_id"]: row for row in registry}
        self.assertEqual(len(registry), len(by_id))
        required_fields = {
            "metric_id",
            "layer",
            "direction",
            "formula",
            "unit",
            "numerator",
            "denominator",
            "source_fields",
            "mask_policy",
            "aggregation",
            "computable_when",
            "na_rule",
        }
        for row in registry:
            self.assertTrue(required_fields <= set(row), row.get("metric_id"))
            self.assertTrue(row["formula"])
            self.assertTrue(row["source_fields"])
            self.assertTrue(row["mask_policy"])
            self.assertTrue(row["na_rule"])

        for metric_id in (
            "state.physical_node.position.rmse",
            "state.physical_edge.distance.rmse",
            "state.information_node.queue.mae",
            "state.information_edge.rate.rmse",
            "event.information_link_activity.f1",
            "event.information_link_activity.auprc",
            "event.flow_present.f1",
            "task.lifecycle.macro_f1",
            "system.task_completion_rate",
            "system.latency.p95",
            "system.application_throughput",
            "resource.rb_utilization",
            "resource.cpu_utilization",
            "system.uav_energy_total",
            "system.completion_fairness_jain",
            "safety.task_flow_conservation_violation_rate",
            "uncertainty.nll",
            "uncertainty.coverage_95",
            "decision.action_regret",
            "deployment.inference_latency.p95",
        ):
            self.assertIn(metric_id, by_id)

        self.assertIn(
            "information_edge_state.outcome.rate_sum",
            by_id["state.information_edge.rate.rmse"]["source_fields"],
        )
        physical_sources = " ".join(
            field
            for row in registry
            if row["metric_id"].startswith("state.physical_edge")
            for field in row["source_fields"]
        ).lower()
        for forbidden in ("csi", "sinr", "rb", "rate", "throughput"):
            self.assertNotIn(forbidden, physical_sources)
        mapping = build_factual_metric_mapping()
        self.assertEqual(22, len(mapping))
        self.assertEqual(22, len({row["source_metric_name"] for row in mapping}))
        self.assertIn(
            "system.information_link_active_ratio",
            {row["canonical_metric_id"] for row in mapping},
        )
        self.assertNotIn(
            "physical",
            " ".join(by_id["system.information_link_active_ratio"]["source_fields"]).lower(),
        )
        self.assertEqual(
            "5 frozen lifecycle classes",
            by_id["task.lifecycle.macro_f1"]["denominator"],
        )
        self.assertIn(
            "average precision",
            by_id["event.information_link_activity.auprc"]["formula"].lower(),
        )

    def test_fair_protocol_freezes_splits_budgets_and_selection_score(self):
        from pi_jwm.evaluation_protocol_v3 import build_fair_experiment_protocol

        protocol = self._protocol()
        self.assertEqual(["train"], protocol["split_roles"]["fit"])
        self.assertEqual("validation", protocol["split_roles"]["checkpoint_selection"])
        self.assertEqual("calibration", protocol["split_roles"]["threshold_calibration"])
        self.assertEqual("locked_test", protocol["split_roles"]["final_evaluation"])
        self.assertEqual([20260803], protocol["budgets"]["module_screening"]["training_seeds"])
        self.assertEqual(
            [20260803, 20260804, 20260805],
            protocol["budgets"]["formal_comparison"]["training_seeds"],
        )
        self.assertEqual(32, protocol["common_training"]["batch_size"])
        self.assertEqual(1e-4, protocol["common_training"]["minimum_improvement"])
        self.assertEqual(4, len(protocol["checkpoint_selection"]["terms"]))
        self.assertEqual("minimize", protocol["checkpoint_selection"]["direction"])
        self.assertEqual("macro_mean_over_complete_environment_trajectories", protocol["reporting"]["primary_aggregation"])
        self.assertIn("training_seed", protocol["reporting"]["experimental_units"])
        self.assertIn("environment_trajectory_seed", protocol["reporting"]["experimental_units"])
        self.assertEqual(
            "selection.required_continuous.normalized_error",
            protocol["checkpoint_selection"]["terms"][-1]["metric_id"],
        )
        self.assertEqual(1.0, sum(row["weight"] for row in protocol["checkpoint_selection"]["terms"]))
        self.assertEqual("calibration", protocol["event_threshold_policy"]["split"])
        self.assertEqual("higher_threshold", protocol["event_threshold_policy"]["tie_break"])
        self.assertIn("10.1371/journal.pone.0118432", protocol["literature_basis"])

    def test_validator_rejects_leakage_and_unequal_method_budgets(self):
        from pi_jwm.evaluation_protocol_v3 import (
            build_fair_experiment_protocol,
            build_metric_registry,
            validate_evaluation_protocol,
        )

        registry = build_metric_registry()
        protocol = self._protocol()
        valid = validate_evaluation_protocol(registry, protocol)
        self.assertTrue(valid["evaluation_protocol_ready"])

        leaked = copy.deepcopy(protocol)
        leaked["split_roles"]["architecture_selection"] = "calibration"
        with self.assertRaisesRegex(ValueError, "calibration.*architecture"):
            validate_evaluation_protocol(registry, leaked)

        locked_fit = copy.deepcopy(protocol)
        locked_fit["split_roles"]["fit"] = ["train", "locked_test"]
        with self.assertRaisesRegex(ValueError, "locked_test.*fit"):
            validate_evaluation_protocol(registry, locked_fit)

        unfair = copy.deepcopy(protocol)
        unfair["method_budget_policy"] = "method_specific_optimizer_steps"
        with self.assertRaisesRegex(ValueError, "equal.*budget"):
            validate_evaluation_protocol(registry, unfair)

        bad_weight = copy.deepcopy(protocol)
        bad_weight["checkpoint_selection"]["terms"][0]["weight"] = 0.9
        with self.assertRaisesRegex(ValueError, "checkpoint.*weights"):
            validate_evaluation_protocol(registry, bad_weight)

        duplicate_term = copy.deepcopy(protocol)
        duplicate_term["checkpoint_selection"]["terms"][1]["metric_id"] = duplicate_term["checkpoint_selection"]["terms"][0]["metric_id"]
        with self.assertRaisesRegex(ValueError, "checkpoint.*unique"):
            validate_evaluation_protocol(registry, duplicate_term)

        bad_physical_registry = copy.deepcopy(registry)
        bad_physical_registry[0]["source_fields"].append(
            "physical_edge_state.outcome.rate_sum"
        )
        with self.assertRaisesRegex(ValueError, "geometry-only whitelist"):
            validate_evaluation_protocol(bad_physical_registry, protocol)


if __name__ == "__main__":
    unittest.main()
