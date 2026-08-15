from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "small_experiments"
    / "task_resource_conservation_audit.py"
)


def load_subject():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("exp04_subject", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_task_ledger():
    return [
        {
            "record_id": "comm-in",
            "kind": "communication",
            "phase": "offload",
            "planned_capacity": 3.0,
            "remaining_before": 2.0,
            "delivered_data": 2.0,
            "remaining_after": 0.0,
        },
        {
            "record_id": "compute",
            "kind": "compute",
            "allocated_cpu": 4.0,
            "dt": 0.1,
            "remaining_before": 0.3,
            "delivered_data": 0.3,
            "remaining_after": 0.0,
        },
        {
            "record_id": "comm-out",
            "kind": "communication",
            "phase": "return",
            "planned_capacity": 0.4,
            "remaining_before": 1.0,
            "delivered_data": 0.4,
            "remaining_after": 0.6,
        },
    ]


def valid_exp04_bundle():
    energy_row = {
        "record_id": "e0",
        "energy_before": 100.0,
        "energy_after": 98.0,
        "is_flying": 1,
        "is_hovering": 0,
        "using_sensor_num": 0,
        "sending_data_size": 2.0,
        "receiving_data_size": 0.0,
        "event_sending_data_size": 2.0,
        "event_receiving_data_size": 0.0,
        "fly_unit_cost": 1.0,
        "hover_unit_cost": 0.2,
        "sensing_unit_cost": 0.1,
        "send_unit_cost": 0.5,
        "receive_unit_cost": 0.25,
    }
    return {
        "task_ledger": valid_task_ledger(),
        "dependency_ledger": [
            {
                "kind": "dependency_flow",
                "dependency_flow_id": "flow::m0::return",
                "dependency_payload": 4.0,
                "physical_delivered_data": 4.0,
            },
            {
                "kind": "dependency_relation",
                "info_edge": "ie::m0::m1",
                "dependency_flow_id": "flow::m0::return",
                "dependency_payload": 4.0,
                "dependency_status": "arrived",
            },
        ],
        "rb_ledger": [
            {"record_id": "rb0", "task_id": "m0", "rb_indices": [0, 3], "n_rb": 4}
        ],
        "cpu_ledger": [
            {
                "record_id": "c0",
                "time": 1.0,
                "node_id": "p0",
                "task_id": "m0",
                "allocated_cpu": 1.0,
                "node_cpu_capacity": 1.0,
            }
        ],
        "uav_energy_ledger": [energy_row],
        "metric_computability": [],
    }


class TaskConservationLedgerTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_script_exists_before_ledger_can_be_used(self):
        self.assertTrue(SCRIPT_PATH.exists(), msg="exp04 conservation script has not been implemented")

    def test_valid_input_compute_and_return_rows_have_zero_residual(self):
        validator = getattr(self.subject, "validate_conservation_ledger", None)
        self.assertTrue(callable(validator), msg="task conservation validator is missing")
        if not callable(validator):
            return

        report = validator(valid_task_ledger(), tolerance=1e-9)

        self.assertTrue(report["task_flow_conservation"])
        self.assertEqual(0.0, report["max_abs_task_residual"])
        self.assertEqual([], report["failed_record_ids"])

    def test_wrong_delivered_amount_is_rejected(self):
        validator = getattr(self.subject, "validate_conservation_ledger", None)
        self.assertTrue(callable(validator), msg="task conservation validator is missing")
        if not callable(validator):
            return
        rows = copy.deepcopy(valid_task_ledger())
        rows[0]["delivered_data"] = 1.5

        report = validator(rows, tolerance=1e-9)

        self.assertFalse(report["task_flow_conservation"])
        self.assertIn("comm-in", report["failed_record_ids"])
        self.assertAlmostEqual(0.5, report["max_abs_task_residual"])

    def test_shared_dependency_relations_do_not_duplicate_one_physical_flow(self):
        validator = getattr(self.subject, "validate_shared_dependency_accounting", None)
        self.assertTrue(callable(validator), msg="dependency accounting validator is missing")
        if not callable(validator):
            return
        rows = [
            {
                "kind": "dependency_flow",
                "dependency_flow_id": "flow::m0::return",
                "dependency_payload": 4.0,
                "physical_delivered_data": 4.0,
            },
            {
                "kind": "dependency_relation",
                "info_edge": "ie::m0::m1",
                "dependency_flow_id": "flow::m0::return",
                "dependency_payload": 4.0,
                "dependency_status": "arrived",
            },
            {
                "kind": "dependency_relation",
                "info_edge": "ie::m0::m2",
                "dependency_flow_id": "flow::m0::return",
                "dependency_payload": 4.0,
                "dependency_status": "arrived",
            },
        ]

        report = validator(rows, tolerance=1e-9)

        self.assertTrue(report["dependency_accounting_valid"])
        self.assertEqual(4.0, report["unique_physical_delivered_data"])
        self.assertEqual(8.0, report["logical_dependency_payload"])

    def test_duplicate_dependency_flow_row_is_rejected(self):
        validator = getattr(self.subject, "validate_shared_dependency_accounting", None)
        self.assertTrue(callable(validator), msg="dependency accounting validator is missing")
        if not callable(validator):
            return
        flow = {
            "kind": "dependency_flow",
            "dependency_flow_id": "flow::m0::return",
            "dependency_payload": 4.0,
            "physical_delivered_data": 4.0,
        }

        report = validator([flow, dict(flow)], tolerance=1e-9)

        self.assertFalse(report["dependency_accounting_valid"])
        self.assertIn("duplicate_dependency_flow", report["errors"])


class ResourceGateTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_rb_indices_are_unique_and_inside_configured_capacity(self):
        validator = getattr(self.subject, "validate_rb_rows", None)
        self.assertTrue(callable(validator), msg="RB validator is missing")
        if not callable(validator):
            return
        valid = [{"record_id": "rb0", "task_id": "m0", "rb_indices": [0, 3], "n_rb": 4}]
        invalid = [{"record_id": "rb1", "task_id": "m1", "rb_indices": [0, 4], "n_rb": 4}]

        self.assertTrue(validator(valid)["rb_valid"])
        self.assertFalse(validator(invalid)["rb_valid"])
        self.assertEqual(["rb1"], validator(invalid)["failed_record_ids"])

    def test_cpu_allocations_do_not_exceed_node_capacity(self):
        validator = getattr(self.subject, "validate_cpu_rows", None)
        self.assertTrue(callable(validator), msg="CPU validator is missing")
        if not callable(validator):
            return
        rows = [
            {"record_id": "c0", "time": 1.0, "node_id": "p0", "task_id": "m0", "allocated_cpu": 0.6, "node_cpu_capacity": 1.0},
            {"record_id": "c1", "time": 1.0, "node_id": "p0", "task_id": "m1", "allocated_cpu": 0.4, "node_cpu_capacity": 1.0},
        ]

        valid = validator(rows, tolerance=1e-9)
        invalid_rows = copy.deepcopy(rows)
        invalid_rows[1]["allocated_cpu"] = 0.6
        invalid = validator(invalid_rows, tolerance=1e-9)

        self.assertTrue(valid["cpu_valid"])
        self.assertFalse(invalid["cpu_valid"])
        self.assertAlmostEqual(0.2, invalid["max_oversubscription"])

    def test_uav_energy_equation_and_channel_inputs_are_both_checked(self):
        validator = getattr(self.subject, "validate_uav_energy_rows", None)
        self.assertTrue(callable(validator), msg="UAV energy validator is missing")
        if not callable(validator):
            return
        row = {
            "record_id": "e0",
            "energy_before": 100.0,
            "energy_after": 98.0,
            "is_flying": 1,
            "is_hovering": 0,
            "using_sensor_num": 0,
            "sending_data_size": 2.0,
            "receiving_data_size": 0.0,
            "event_sending_data_size": 2.0,
            "event_receiving_data_size": 0.0,
            "fly_unit_cost": 1.0,
            "hover_unit_cost": 0.2,
            "sensing_unit_cost": 0.1,
            "send_unit_cost": 0.5,
            "receive_unit_cost": 0.25,
        }

        valid = validator([row], tolerance=1e-9)
        wrong_equation = validator([dict(row, energy_after=97.5)], tolerance=1e-9)
        wrong_input = validator([dict(row, event_sending_data_size=1.5)], tolerance=1e-9)

        self.assertTrue(valid["uav_energy_valid"])
        self.assertFalse(wrong_equation["energy_equation_valid"])
        self.assertFalse(wrong_input["channel_energy_input_valid"])

    def test_unmodeled_energy_fields_are_masked_not_zero_filled(self):
        builder = getattr(self.subject, "build_metric_computability", None)
        self.assertTrue(callable(builder), msg="metric computability builder is missing")
        if not callable(builder):
            return

        rows = builder([])
        by_field = {row["field_id"]: row for row in rows}

        self.assertEqual("direct", by_field["uav_energy"]["status"])
        self.assertEqual("not_modeled", by_field["vehicle_energy"]["status"])
        self.assertEqual("not_modeled", by_field["rsu_energy"]["status"])
        self.assertEqual("not_modeled", by_field["cpu_compute_energy"]["status"])
        self.assertIsNone(by_field["vehicle_energy"]["fill_value"])


class Exp04BundleAndExportTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_aggregate_validator_requires_every_modeled_gate(self):
        validator = getattr(self.subject, "validate_exp04_bundle", None)
        self.assertTrue(callable(validator), msg="exp04 bundle validator is missing")
        if not callable(validator):
            return

        report = validator(valid_exp04_bundle(), tolerance=1e-9)

        self.assertTrue(report["conservation_ready"])
        self.assertEqual([], report["failed_gates"])

    def test_four_independent_corruptions_are_detected(self):
        builder = getattr(self.subject, "build_exp04_corruption_report", None)
        self.assertTrue(callable(builder), msg="exp04 corruption builder is missing")
        if not callable(builder):
            return

        report = builder(valid_exp04_bundle(), tolerance=1e-9)

        self.assertTrue(report["all_corruptions_detected"])
        self.assertEqual(
            {"task_byte_residual", "illegal_rb", "cpu_oversubscription", "uav_energy_residual"},
            {row["corruption_id"] for row in report["cases"]},
        )

    def test_corruption_detection_compares_new_failures_against_a_failed_baseline(self):
        builder = getattr(self.subject, "build_exp04_corruption_report", None)
        self.assertTrue(callable(builder), msg="exp04 corruption builder is missing")
        if not callable(builder):
            return
        bundle = valid_exp04_bundle()
        bundle["uav_energy_ledger"][0]["event_sending_data_size"] = 1.5
        bundle["cpu_ledger"].append(
            {
                "record_id": "c1",
                "time": 1.0,
                "node_id": "p0",
                "task_id": "m1",
                "allocated_cpu": 0.5,
                "node_cpu_capacity": 1.0,
            }
        )

        report = builder(bundle, tolerance=1e-9)

        self.assertTrue(report["all_corruptions_detected"])
        self.assertIn("channel_energy_input_valid", report["baseline_failed_gates"])
        self.assertIn("cpu_valid", report["baseline_failed_gates"])

    def test_fake_runtime_is_exported_with_manifest_and_report(self):
        runner = getattr(self.subject, "run_exp04", None)
        self.assertTrue(callable(runner), msg="exp04 export runner is missing")
        if not callable(runner):
            return

        calls = []

        def fake_runtime(seed, max_time):
            calls.append((seed, max_time))
            return {
                "config": {"seed": seed, "max_time": max_time},
                "bundle": valid_exp04_bundle(),
                "runtime_summary": {"seed": seed, "steps": 1, "max_time": max_time},
            }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = runner(output_dir, seed=0, max_time=1.0, runtime_runner=fake_runtime)
            expected = {
                "bundle.json",
                "task_ledger.csv",
                "dependency_ledger.csv",
                "rb_ledger.csv",
                "cpu_ledger.csv",
                "uav_energy_ledger.csv",
                "metric_computability.csv",
                "validation_report.json",
                "corruption_report.json",
                "config_snapshot.json",
                "runtime_summary.json",
                "REPORT.md",
                "manifest.json",
            }
            self.assertTrue(result["conservation_ready"])
            self.assertTrue(result["experiment_completed"])
            self.assertTrue(result["reproducibility_passed"])
            self.assertEqual([(0, 1.0), (0, 1.0)], calls)
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("AirFogSim-PIJWM-exp04-v1", manifest["schema_version"])
            self.assertIn("bundle.json", manifest["files"])
            self.assertTrue(manifest["reproducibility"]["same_seed_bundle_hash_equal"])


class RuntimeLedgerBuilderTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_transfer_events_become_delivered_byte_and_rb_ledgers(self):
        builder = getattr(self.subject, "build_transfer_ledgers", None)
        self.assertTrue(callable(builder), msg="runtime transfer ledger builder is missing")
        if not callable(builder):
            return
        events = [
            {
                "event_id": "event::m0::offload::0::0.0",
                "task_id": "m0",
                "phase": "offload",
                "time": 0.0,
                "planned_capacity": 3.0,
                "remaining_before": 2.0,
                "delivered_data": 2.0,
                "rb_indices": [0, 3],
            }
        ]

        task_rows, rb_rows = builder(events, n_rb=4)

        self.assertEqual(0.0, task_rows[0]["remaining_after"])
        self.assertEqual([0, 3], rb_rows[0]["rb_indices"])
        self.assertEqual(4, rb_rows[0]["n_rb"])

    def test_cpu_runtime_row_uses_observed_progress_delta(self):
        builder = getattr(self.subject, "build_cpu_runtime_row", None)
        self.assertTrue(callable(builder), msg="runtime CPU ledger builder is missing")
        if not callable(builder):
            return

        row = builder(
            record_id="c0",
            time_value=1.0,
            node_id="p0",
            task_id="m0",
            allocated_cpu=4.0,
            node_cpu_capacity=4.0,
            dt=0.1,
            task_cpu=0.3,
            computed_before=0.0,
            computed_after=0.3,
        )

        self.assertEqual(0.3, row["delivered_data"])
        self.assertEqual(0.0, row["remaining_after"])

    def test_energy_runtime_row_keeps_manager_and_event_inputs_separate(self):
        builder = getattr(self.subject, "build_uav_energy_runtime_row", None)
        self.assertTrue(callable(builder), msg="runtime UAV energy ledger builder is missing")
        if not callable(builder):
            return

        row = builder(
            record_id="e0",
            time_value=1.0,
            uav_id="U_0",
            before={"energy": 100.0},
            after={"energy": 98.0, "is_flying": True, "is_hovering": False, "using_sensor_num": 0, "sending_data_size": 2.0, "receiving_data_size": 0.0},
            event_sending_data_size=1.5,
            event_receiving_data_size=0.0,
            costs={"fly_unit_cost": 1.0, "hover_unit_cost": 0.2, "sensing_unit_cost": 0.1, "send_unit_cost": 0.5, "receive_unit_cost": 0.25},
        )

        self.assertEqual(2.0, row["sending_data_size"])
        self.assertEqual(1.5, row["event_sending_data_size"])

    def test_exp03_runtime_evidence_is_assembled_without_inventing_fields(self):
        builder = getattr(self.subject, "assemble_runtime_conservation_bundle", None)
        self.assertTrue(callable(builder), msg="runtime conservation bundle assembler is missing")
        if not callable(builder):
            return
        evidence = {
            "bundle": {
                "transfer_events": [
                    {
                        "event_id": "event::m0::return::0::0.0",
                        "task_id": "m0",
                        "phase": "return",
                        "time": 0.0,
                        "planned_capacity": 1.0,
                        "remaining_before": 1.0,
                        "delivered_data": 1.0,
                        "rb_indices": [0],
                    }
                ],
                "dependency_flows": [
                    {"dependency_flow_id": "flow::m0::return", "dependency_payload": 1.0, "physical_delivered_data": 1.0}
                ],
                "ep_relations": [
                    {"dependency_flow_id": "flow::m0::return", "dependency_payload": 1.0, "dependency_status": "arrived"}
                ],
            }
        }

        bundle = builder(evidence, cpu_rows=[], energy_rows=[], n_rb=4)

        self.assertEqual("dependency_flow", bundle["dependency_ledger"][0]["kind"])
        self.assertEqual("dependency_relation", bundle["dependency_ledger"][1]["kind"])
        self.assertEqual("not_modeled", {row["field_id"]: row["status"] for row in bundle["metric_computability"]}["vehicle_energy"])


if __name__ == "__main__":
    unittest.main()
