from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_cpu_preflight import (  # noqa: E402
    aggregate_metric_rows,
    audit_trajectory_record,
    validate_candidate_roles,
    write_r6_cpu_preflight_bundle,
)


def _roles() -> dict:
    return {
        "schema_version": "PIJWM-R6-Working-Candidate-Freeze-v1",
        "reference_control": "B",
        "primary_working_candidate": "B",
        "task_lifecycle_specialists": ["G"],
        "continuous_state_specialists": ["J"],
        "ablation_controls": ["F"],
        "r5_1_candidate_set_frozen": True,
        "r6_cpu_preflight_ready": True,
        "r6_gpu_strategy_training_ready": False,
        "final_method_frozen": False,
        "locked_test_accessed": False,
    }


def _trajectory(*, split: str = "validation", policy: str = "deadline_aware") -> dict:
    return {
        "trajectory_summary": {
            "trajectory_id": "scenario__r06",
            "seed": 6,
            "split": split,
            "cpu_policy": policy,
            "checks": {
                "action_ledgers_present": True,
                "cpu_policy_trace_valid": True,
                "dag_precedence_only": True,
            },
        },
        "graph_validation": {"dual_graph_v2_ready": True},
        "resource_validation": {
            "conservation_ready": True,
            "checks": {
                "cpu_capacity_violation_rate": 0.0,
                "energy_equation_violation_rate": 0.0,
                "task_flow_conservation_violation_rate": 0.0,
            },
        },
        "metric_results": {
            "metrics": [
                {"name": "task_completion_rate", "status": "available", "value": 0.8},
                {"name": "successful_task_delay_p95", "status": "available", "value": 1.2},
                {"name": "information_throughput", "status": "available", "value": 3.0},
                {"name": "uav_energy_total", "status": "available", "value": 10.0},
                {"name": "dependency_data_delivery_rate", "status": "not_applicable", "value": None},
            ]
        },
    }


class R6CpuPreflightTest(unittest.TestCase):
    def test_candidate_roles_are_frozen_and_final_method_is_not_claimed(self) -> None:
        roles = validate_candidate_roles(_roles())
        self.assertEqual(roles["primary_working_candidate"], "B")
        self.assertEqual(roles["task_lifecycle_specialists"], ["G"])
        self.assertFalse(roles["final_method_frozen"])

        bad = dict(_roles(), primary_working_candidate="J")
        with self.assertRaisesRegex(ValueError, "primary"):
            validate_candidate_roles(bad)

    def test_trajectory_audit_accepts_nonlocked_complete_evidence(self) -> None:
        result = audit_trajectory_record(_trajectory())
        self.assertEqual(result["status"], "audited")
        self.assertEqual(result["hard_constraint_violation_count"], 0)
        self.assertEqual(result["available_metric_count"], 4)
        self.assertEqual(result["not_applicable_metric_count"], 1)

    def test_trajectory_audit_rejects_locked_or_hard_constraint_violations(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked-test"):
            audit_trajectory_record(_trajectory(split="locked_test"))
        bad = _trajectory()
        bad["resource_validation"]["checks"]["cpu_capacity_violation_rate"] = 0.01
        with self.assertRaisesRegex(ValueError, "hard constraint"):
            audit_trajectory_record(bad)

    def test_formal_gate_shape_is_audited_and_false_gate_is_rejected(self) -> None:
        record = _trajectory()
        record["resource_validation"] = {
            "conservation_ready": True,
            "gates": {
                "cpu_valid": True,
                "rb_valid": True,
                "task_flow_conservation": True,
                "energy_equation_valid": True,
                "channel_energy_input_valid": True,
                "dependency_accounting_valid": True,
            },
        }
        self.assertEqual(audit_trajectory_record(record)["hard_constraint_violation_count"], 0)
        record["resource_validation"]["gates"]["rb_valid"] = False
        with self.assertRaisesRegex(ValueError, "hard constraint"):
            audit_trajectory_record(record)

    def test_aggregate_preserves_na_and_reports_grouped_mean(self) -> None:
        rows = [
            {"policy_id": "deadline_aware", "split": "validation", "metric_id": "task_completion_rate", "status": "available", "value": 0.8},
            {"policy_id": "deadline_aware", "split": "validation", "metric_id": "task_completion_rate", "status": "available", "value": 0.6},
            {"policy_id": "deadline_aware", "split": "validation", "metric_id": "dependency_data_delivery_rate", "status": "not_applicable", "value": None},
        ]
        result = aggregate_metric_rows(rows)
        key = ("deadline_aware", "validation", "task_completion_rate")
        self.assertAlmostEqual(result[key]["mean"], 0.7)
        self.assertEqual(result[key]["available_count"], 2)
        self.assertEqual(
            result[("deadline_aware", "validation", "dependency_data_delivery_rate")]["status"],
            "not_applicable",
        )

    def test_writer_emits_auditable_bundle(self) -> None:
        trajectory_rows = [audit_trajectory_record(_trajectory())]
        metric_rows = [
            {"policy_id": "deadline_aware", "split": "validation", "metric_id": "task_completion_rate", "status": "available", "value": 0.8}
        ]
        summary = {"r6_cpu_preflight_ready": True, "locked_test_accessed": False}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "r6"
            write_r6_cpu_preflight_bundle(
                output,
                candidate_roles=_roles(),
                trajectory_rows=trajectory_rows,
                metric_rows=metric_rows,
                metric_summary=aggregate_metric_rows(metric_rows),
                summary=summary,
                input_binding={"dataset_manifest_sha256": "abc"},
            )
            expected = {
                "candidate_roles.json",
                "trajectory_audit.csv",
                "metric_rows.csv",
                "metric_summary.csv",
                "summary.json",
                "README.md",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_entry_count"], 6)
            self.assertFalse(manifest["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
