from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def passing_exp03():
    return {
        "experiment_completed": True,
        "strict_dual_graph_ready": True,
        "reproducibility_passed": True,
        "corruption_detection_passed": True,
    }


def passing_exp04():
    return {
        "experiment_completed": True,
        "conservation_ready": True,
        "reproducibility_passed": True,
        "corruption_detection_passed": True,
        "gates": {
            "task_flow_conservation": True,
            "dependency_accounting_valid": True,
            "rb_valid": True,
            "cpu_valid": True,
            "energy_equation_valid": True,
            "channel_energy_input_valid": True,
            "same_seed_reproducible": True,
        },
    }


def passing_exp05():
    return {
        "experiment_completed": True,
        "action_sensitivity_ready": True,
        "total_pairs": 6,
        "accepted_pairs": 6,
        "failed_pair_ids": [],
        "corruption_detection_passed": True,
    }


class FrozenContractTests(unittest.TestCase):
    def test_first_control_slice_has_only_offload_and_rb_actions(self):
        from pi_jwm.main_experiment_contract import build_frozen_contract

        contract = build_frozen_contract()
        action_ids = [row["field_id"] for row in contract["actions"]]

        self.assertEqual(
            ["offload_target_path", "rb_allocation"],
            action_ids,
        )
        self.assertIn("cpu_allocation", contract["fixed_rules"])
        self.assertIn("uav_trajectory", contract["excluded_actions"])
        self.assertIn("transmit_power", contract["excluded_actions"])
        self.assertIn("mcs", contract["excluded_actions"])

    def test_contract_covers_strict_dual_graph_rollout_and_system_metrics(self):
        from pi_jwm.main_experiment_contract import build_frozen_contract

        contract = build_frozen_contract()
        target_ids = {row["target_id"] for row in contract["prediction_targets"]}
        metric_ids = {row["metric_id"] for row in contract["metrics"]}

        self.assertTrue(
            {
                "next_physical_node_state",
                "next_physical_edge_state",
                "next_information_agent_state",
                "next_information_flow_state",
                "next_task_state",
                "next_resource_state",
                "task_terminal_outcome",
            }.issubset(target_ids)
        )
        field_groups = {row["group"] for row in contract["state_fields"]}
        self.assertTrue({"information_agent", "information_flow", "task_auxiliary", "dag_auxiliary"}.issubset(field_groups))
        self.assertFalse(
            any(
                row["group"] == "information_node" and row["field_id"].startswith("task_")
                for row in contract["state_fields"]
            )
        )
        self.assertTrue(
            {
                "state_mae",
                "state_rmse",
                "link_activity_f1",
                "rollout_horizon_error",
                "nll",
                "conformal_coverage",
                "action_ranking_regret",
                "task_completion_rate",
                "latency_p95",
                "latency_p99",
                "energy_consumption",
                "resource_utilization",
                "fairness",
                "constraint_violation_rate",
            }.issubset(metric_ids)
        )

    def test_unmodelled_fields_have_null_value_and_zero_observation_mask(self):
        from pi_jwm.main_experiment_contract import build_frozen_contract

        contract = build_frozen_contract()
        missing = {row["field_id"]: row for row in contract["unmodelled_fields"]}

        for field_id in (
            "vehicle_energy",
            "rsu_energy",
            "cpu_compute_energy",
            "storage_occupancy",
        ):
            self.assertIsNone(missing[field_id]["value"])
            self.assertEqual(0, missing[field_id]["observed_mask"])
        self.assertEqual("never_zero_fill", contract["missing_value_policy"])


class ReadinessGateTests(unittest.TestCase):
    def test_passing_preflights_enable_simulation_but_not_unbuilt_datasets(self):
        from pi_jwm.main_experiment_contract import build_readiness_report

        result = build_readiness_report(passing_exp03(), passing_exp04(), passing_exp05())

        self.assertTrue(result["contract_ready"])
        self.assertTrue(result["simulator_preflight_ready"])
        self.assertTrue(result["simulation_training_ready"])
        self.assertFalse(result["formal_dataset_ready"])
        self.assertFalse(result["external_validation_ready"])

    def test_failed_conservation_gate_blocks_simulation_training(self):
        from pi_jwm.main_experiment_contract import build_readiness_report

        exp04 = passing_exp04()
        exp04["conservation_ready"] = False
        exp04["gates"]["cpu_valid"] = False

        result = build_readiness_report(passing_exp03(), exp04, passing_exp05())

        self.assertFalse(result["simulator_preflight_ready"])
        self.assertFalse(result["simulation_training_ready"])
        self.assertIn("exp04_conservation_ready", result["blocking_checks"])

    def test_formal_dataset_and_external_holdout_require_independent_manifests(self):
        from pi_jwm.main_experiment_contract import build_readiness_report

        formal_manifest = {
            "generation_completed": True,
            "field_masks_valid": True,
            "splits_frozen": True,
            "source_manifest_present": True,
        }
        external_manifest = {
            "local_data_verified": True,
            "license_verified": True,
            "field_semantics_verified": True,
            "holdout_split_frozen": True,
        }

        result = build_readiness_report(
            passing_exp03(),
            passing_exp04(),
            passing_exp05(),
            formal_dataset_manifest=formal_manifest,
            external_validation_manifest=external_manifest,
        )

        self.assertTrue(result["formal_dataset_ready"])
        self.assertTrue(result["external_validation_ready"])


if __name__ == "__main__":
    unittest.main()
