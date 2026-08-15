from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "small_experiments"
sys.path.insert(0, str(SCRIPT_DIR))

from data_contract_field_audit import (
    build_field_contract,
    evaluate_readiness,
    run_audit,
    scan_action_runs,
    scan_candidate_data,
    scan_csv_runs,
    scan_npz_metadata,
)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def contract_by_id(contract: list[dict]) -> dict[str, dict]:
    return {row["field_id"]: row for row in contract}


class DataContractFieldAuditTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> dict[str, Path]:
        raw_root = root / "raw"
        raw_seed = raw_root / "seed_000"
        action_root = root / "actions"
        action_seed = action_root / "seed_000"
        candidate_root = root / "candidate_data"
        candidate_root.mkdir(parents=True)

        write_csv(
            raw_seed / "node_states.csv",
            ["seed", "time", "node_id", "node_type", "x", "y", "z", "speed", "acceleration", "cpu", "storage", "task_profile"],
            [
                {"seed": 0, "time": 0.1, "node_id": "vehicle_0", "node_type": "vehicle", "x": 0, "y": 0, "z": 0, "speed": 1, "acceleration": 0, "cpu": 2, "storage": 1, "task_profile": json.dumps({"lambda": 0.5, "dag_edge_prob": 0.0})},
                {"seed": 0, "time": 0.2, "node_id": "vehicle_0", "node_type": "vehicle", "x": 0.1, "y": 0, "z": 0, "speed": 1, "acceleration": 0, "cpu": 2, "storage": 1, "task_profile": json.dumps({"lambda": 0.5, "dag_edge_prob": 0.0})},
            ],
        )
        write_csv(
            raw_seed / "link_states.csv",
            ["seed", "time", "tx_id", "rx_id", "link_type", "distance", "rate_sum", "csi_mean", "active_task_count", "allocated_rb_count"],
            [{"seed": 0, "time": 0.1, "tx_id": "vehicle_0", "rx_id": "rsu_0", "link_type": "V2I", "distance": 10, "rate_sum": 5, "csi_mean": 90, "active_task_count": 1, "allocated_rb_count": 3}],
        )
        write_csv(
            raw_seed / "task_states.csv",
            ["seed", "time", "task_id", "task_node_id", "current_node_id", "assigned_to", "task_size", "task_cpu", "deadline", "priority", "arrival_time", "transmitted_size", "computed_size", "lifecycle_state", "failure_reason"],
            [{"seed": 0, "time": 0.1, "task_id": "Task_0", "task_node_id": "vehicle_0", "current_node_id": "vehicle_0", "assigned_to": "rsu_0", "task_size": 10, "task_cpu": 4, "deadline": 1, "priority": 0.5, "arrival_time": 0, "transmitted_size": 2, "computed_size": 1, "lifecycle_state": "computing", "failure_reason": "Unknown code."}],
        )

        action_specs = {
            "offload_actions.csv": (["seed", "time", "task_id", "task_node_id", "source_node_id", "target_node_id", "target_node_type", "candidate_count", "nearest_distance"], [{"seed": 0, "time": 0.1, "task_id": "Task_0", "task_node_id": "vehicle_0", "source_node_id": "vehicle_0", "target_node_id": "rsu_0", "target_node_type": "rsu", "candidate_count": 1, "nearest_distance": 10}]),
            "return_actions.csv": (["seed", "time", "task_id", "task_node_id", "current_node_id", "return_target_id", "return_distance"], []),
            "rb_actions.csv": (["seed", "time", "task_id", "task_node_id", "current_node_id", "assigned_to", "rb_count", "rb_indices"], [{"seed": 0, "time": 0.1, "task_id": "Task_0", "task_node_id": "vehicle_0", "current_node_id": "vehicle_0", "assigned_to": "rsu_0", "rb_count": 3, "rb_indices": "0 1 2"}]),
            "cpu_actions.csv": (["seed", "time", "task_id", "task_node_id", "assigned_to", "assigned_node_type", "allocated_cpu", "num_tasks_on_node"], [{"seed": 0, "time": 0.1, "task_id": "Task_0", "task_node_id": "vehicle_0", "assigned_to": "rsu_0", "assigned_node_type": "rsu", "allocated_cpu": 2, "num_tasks_on_node": 1}]),
            "uav_mobility_actions.csv": (["seed", "time", "uav_id", "current_x", "current_y", "current_z", "target_source", "target_x", "target_y", "target_z", "angle", "phi", "speed"], [{"seed": 0, "time": 0.1, "uav_id": "UAV_0", "current_x": 0, "current_y": 0, "current_z": 100, "target_source": "random", "target_x": 0, "target_y": 0, "target_z": 0, "angle": 0, "phi": 0, "speed": 5}]),
        }
        for filename, (columns, rows) in action_specs.items():
            write_csv(action_seed / filename, columns, rows)

        npz_path = root / "world_model.npz"
        np.savez_compressed(
            npz_path,
            node_features=np.asarray(["x", "y", "z", "speed", "acceleration", "cpu", "storage"]),
            link_features=np.asarray(["distance", "rate_sum", "csi_mean", "active_task_count", "allocated_rb_count"]),
            task_features=np.asarray(["num_tasks", "total_task_size", "num_finished"]),
            edge_action_features=np.asarray(["offload_count", "rb_total", "cpu_total", "return_count"]),
        )
        edge_summary = root / "edge_action_summary.json"
        edge_summary.write_text(
            json.dumps({"match_summary": [{"offload_total": 10, "offload_matched": 2, "rb_total": 10, "rb_matched": 2, "cpu_total": 20, "cpu_matched": 4, "return_total": 1, "return_matched": 1}]}),
            encoding="utf-8",
        )
        return {
            "raw_root": raw_root,
            "action_root": action_root,
            "npz_path": npz_path,
            "edge_summary": edge_summary,
            "candidate_root": candidate_root,
        }

    def test_source_scanners_record_actual_schema_counts_and_zero_dag(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))

            raw = scan_csv_runs(paths["raw_root"])
            actions = scan_action_runs(paths["action_root"])

        self.assertEqual(1, raw["seed_count"])
        self.assertTrue(raw["schemas_consistent"])
        self.assertEqual(0.0, raw["dag_edge_probability_max"])
        self.assertEqual(1, actions["files"]["offload_actions.csv"]["rows"])
        self.assertEqual(0, actions["files"]["return_actions.csv"]["rows"])
        self.assertIn("rb_indices", actions["files"]["rb_actions.csv"]["columns"])

    def test_npz_and_candidate_scanners_do_not_turn_papers_into_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))
            (paths["candidate_root"] / "DeepMIMO_paper.pdf").write_bytes(b"paper")

            npz = scan_npz_metadata(paths["npz_path"])
            candidates = scan_candidate_data(paths["candidate_root"])

        self.assertIn("csi_mean", npz["feature_names"]["link_features"])
        self.assertEqual([], candidates["data_files"])
        self.assertEqual(1, len(candidates["paper_files"]))

    def test_contract_separates_direct_derivable_missing_and_quality(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))
            inventory = {
                "raw": scan_csv_runs(paths["raw_root"]),
                "actions": scan_action_runs(paths["action_root"]),
                "processed": scan_npz_metadata(paths["npz_path"]),
                "candidates": scan_candidate_data(paths["candidate_root"]),
                "action_projection": {"minimum_core_match_rate": 0.2},
            }

            rows = contract_by_id(build_field_contract(inventory))

        self.assertEqual("direct", rows["timestamp"]["status"])
        self.assertEqual("derivable", rows["velocity_vector"]["status"])
        self.assertEqual("derivable", rows["task_remaining_input"]["status"])
        self.assertEqual("missing", rows["task_dag_edges"]["status"])
        self.assertEqual("zero_configured", rows["task_dag_edges"]["quality_flag"])
        self.assertEqual("missing", rows["node_energy"]["status"])
        self.assertEqual("direct", rows["offload_action_task_level"]["status"])
        self.assertEqual("partial", rows["aligned_core_action_sequence"]["quality_flag"])

    def test_derivable_and_mapping_fields_require_their_source_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))
            task_path = paths["raw_root"] / "seed_000" / "task_states.csv"
            columns = ["seed", "time", "task_id", "current_node_id", "assigned_to", "task_size", "task_cpu", "deadline", "priority", "arrival_time", "computed_size", "lifecycle_state", "failure_reason"]
            write_csv(
                task_path,
                columns,
                [{"seed": 0, "time": 0.1, "task_id": "Task_0", "current_node_id": "vehicle_0", "assigned_to": "rsu_0", "task_size": 10, "task_cpu": 4, "deadline": 1, "priority": 0.5, "arrival_time": 0, "computed_size": 1, "lifecycle_state": "computing", "failure_reason": "Unknown code."}],
            )
            inventory = {
                "raw": scan_csv_runs(paths["raw_root"]),
                "actions": scan_action_runs(paths["action_root"]),
                "processed": scan_npz_metadata(paths["npz_path"]),
                "candidates": scan_candidate_data(paths["candidate_root"]),
                "action_projection": {"minimum_core_match_rate": 0.2},
            }

            rows = contract_by_id(build_field_contract(inventory))

        self.assertEqual("missing", rows["task_remaining_input"]["status"])
        self.assertEqual("missing", rows["mn_source_mapping"]["status"])

    def test_missing_core_action_file_fails_audit_completeness_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))
            (paths["action_root"] / "seed_000" / "rb_actions.csv").unlink()
            inventory = {
                "raw": scan_csv_runs(paths["raw_root"]),
                "actions": scan_action_runs(paths["action_root"]),
                "processed": scan_npz_metadata(paths["npz_path"]),
                "candidates": scan_candidate_data(paths["candidate_root"]),
                "action_projection": {"minimum_core_match_rate": 0.2},
            }

            report = evaluate_readiness(build_field_contract(inventory), inventory)

        action_check = next(check for check in report["checks"] if check["name"] == "strict_action_files_present_and_core_nonempty")
        self.assertFalse(action_check["passed"])
        self.assertFalse(report["audit_completed"])

    def test_candidate_file_requires_semantic_verification_before_real_holdout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))
            write_csv(paths["candidate_root"] / "unrelated.csv", ["value"], [{"value": 1}])
            inventory = {
                "raw": scan_csv_runs(paths["raw_root"]),
                "actions": scan_action_runs(paths["action_root"]),
                "processed": scan_npz_metadata(paths["npz_path"]),
                "candidates": scan_candidate_data(paths["candidate_root"]),
                "action_projection": {"minimum_core_match_rate": 0.2},
            }

            row = contract_by_id(build_field_contract(inventory))["real_measurement_holdout"]

        self.assertEqual("missing", row["status"])
        self.assertEqual("unverified_local_files", row["quality_flag"])

    def test_readiness_gate_has_a_positive_path_for_complete_evidence(self):
        contract = [{"field_id": "required_field", "status": "direct", "quality_flag": "complete", "required_for_v1": True}]
        inventory = {
            "raw": {"schemas_consistent": True, "all_required_files_present": True, "all_required_files_nonempty": True},
            "actions": {"schemas_consistent": True, "all_required_files_present": True, "seed_count": 1, "files": {
                "offload_actions.csv": {"nonempty_seeds": 1},
                "rb_actions.csv": {"nonempty_seeds": 1},
                "cpu_actions.csv": {"nonempty_seeds": 1},
            }},
            "processed": {"exists": True},
            "candidates": {"data_files": []},
        }

        report = evaluate_readiness(contract, inventory)

        self.assertTrue(report["audit_completed"])
        self.assertTrue(report["formal_training_ready"])

    def test_readiness_is_false_without_dag_energy_and_paired_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.build_fixture(Path(temporary_directory))
            inventory = {
                "raw": scan_csv_runs(paths["raw_root"]),
                "actions": scan_action_runs(paths["action_root"]),
                "processed": scan_npz_metadata(paths["npz_path"]),
                "candidates": scan_candidate_data(paths["candidate_root"]),
                "action_projection": {"minimum_core_match_rate": 0.2},
            }
            contract = build_field_contract(inventory)

            report = evaluate_readiness(contract, inventory)

        self.assertTrue(report["audit_completed"])
        self.assertFalse(report["formal_training_ready"])
        self.assertIn("task_dag_edges", report["blocking_fields"])
        self.assertIn("aligned_core_action_sequence", report["blocking_fields"])

    def test_run_audit_exports_complete_traceable_bundle(self):
        expected = {"source_inventory.json", "field_contract.csv", "coverage_summary.csv", "validation_report.json", "REPORT.md"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self.build_fixture(root)
            output_dir = root / "out"

            result = run_audit(
                raw_root=paths["raw_root"],
                action_root=paths["action_root"],
                world_model_npz=paths["npz_path"],
                edge_action_summary=paths["edge_summary"],
                candidate_data_root=paths["candidate_root"],
                output_dir=output_dir,
            )

            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
            report = (output_dir / "REPORT.md").read_text(encoding="utf-8")

        self.assertTrue(result["audit_completed"])
        self.assertFalse(validation["formal_training_ready"])
        self.assertIn("审计完成不等于数据已满足正式训练条件", report)
        self.assertIn("候选真实数据当前只有文献证据，不计入本地字段覆盖", report)
        self.assertIn("AirFogSim是数据源，不是PI-JWM框架", report)

    def test_cli_executes_after_all_helpers_are_defined(self):
        script = SCRIPT_DIR / "data_contract_field_audit.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self.build_fixture(root)
            output_dir = root / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--raw-root", str(paths["raw_root"]),
                    "--action-root", str(paths["action_root"]),
                    "--world-model-npz", str(paths["npz_path"]),
                    "--edge-action-summary", str(paths["edge_summary"]),
                    "--candidate-data-root", str(paths["candidate_root"]),
                    "--output-dir", str(output_dir),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        self.assertEqual(0, completed.returncode, msg=completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["audit_completed"])
        self.assertFalse(payload["formal_training_ready"])


if __name__ == "__main__":
    unittest.main()
