from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "small_experiments"
sys.path.insert(0, str(SCRIPT_DIR))

from strict_dual_graph_validity import (
    _physical_plot_positions,
    build_controlled_fixture,
    compare_candidate_actions,
    run_experiment,
    validate_scenario,
)


def check_by_name(report: dict, name: str) -> dict:
    return next(check for check in report["checks"] if check["name"] == name)


class StrictDualGraphValidationTests(unittest.TestCase):
    def test_physical_plot_offsets_colocated_nodes_without_changing_data(self):
        scenario = build_controlled_fixture()

        positions = _physical_plot_positions(scenario)

        self.assertEqual(len(scenario["physical_nodes"]), len(set(positions.values())))
        self.assertEqual([50.0, 0.0], next(node for node in scenario["physical_nodes"] if node["id"] == "rsu0")["position"])
        self.assertEqual([50.0, 0.0], next(node for node in scenario["physical_nodes"] if node["id"] == "edge0")["position"])

    def test_controlled_fixture_is_nonempty_and_valid(self):
        scenario = build_controlled_fixture()

        report = validate_scenario(scenario)

        self.assertTrue(report["passed"])
        self.assertGreater(report["counts"]["information_edges"], 0)
        self.assertGreater(report["counts"]["mn_relations"], 0)
        self.assertGreater(report["counts"]["me_relations"], 0)
        self.assertGreater(report["counts"]["ep_relations"], 0)
        self.assertTrue(all(check["passed"] for check in report["checks"]))

    def test_cycle_is_rejected(self):
        scenario = build_controlled_fixture()
        scenario["information_edges"].append(
            {"id": "ie_m3_m0", "src": "m3", "dst": "m0", "data_mb": 1.0}
        )

        report = validate_scenario(scenario)

        self.assertFalse(check_by_name(report, "information_graph_is_dag")["passed"])
        self.assertFalse(report["passed"])

    def test_duplicate_source_mapping_is_rejected(self):
        scenario = build_controlled_fixture()
        scenario["mn_relations"].append(
            {"task": "m0", "relation": "source", "physical_node": "rsu0"}
        )

        report = validate_scenario(scenario)

        self.assertFalse(check_by_name(report, "mn_relation_cardinality")["passed"])
        self.assertFalse(report["passed"])

    def test_discontinuous_dependency_path_is_rejected(self):
        scenario = copy.deepcopy(build_controlled_fixture())
        scenario["ep_relations"][0]["path"] = ["e_edge_rsu", "e_veh_rsu"]

        report = validate_scenario(scenario)

        self.assertFalse(check_by_name(report, "ep_paths_valid")["passed"])
        self.assertFalse(report["passed"])

    def test_two_legal_actions_change_the_one_step_successor(self):
        scenario = build_controlled_fixture()

        result = compare_candidate_actions(scenario)

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rows"]), 2)
        first, second = result["rows"]
        self.assertNotEqual(first["execution_target"], second["execution_target"])
        self.assertNotEqual(first["rb_count"], second["rb_count"])
        self.assertNotEqual(first["remaining_input_mb_next"], second["remaining_input_mb_next"])
        self.assertNotEqual(first["input_path"], second["input_path"])

    def test_experiment_exports_complete_evidence_bundle(self):
        expected = {
            "scenario.json",
            "validation_report.json",
            "cross_relations.csv",
            "action_sensitivity.csv",
            "physical_graph.png",
            "information_graph.png",
            "joint_graph.png",
            "REPORT.md",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            result = run_experiment(output_dir)

            self.assertTrue(result["passed"])
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["passed"])
            self.assertEqual([], validation["failed_checks"])
            self.assertGreater(validation["counts"]["mn_relations"], 0)
            self.assertGreater(validation["counts"]["me_relations"], 0)
            self.assertGreater(validation["counts"]["ep_relations"], 0)
            report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("受控夹具验证：已完成", report)
            self.assertIn("AirFogSim真实非空DAG验证：待完成", report)
            for image_name in ("physical_graph.png", "information_graph.png", "joint_graph.png"):
                self.assertGreater((output_dir / image_name).stat().st_size, 1000)

    def test_cli_executes_after_all_helpers_are_defined(self):
        script = SCRIPT_DIR / "strict_dual_graph_validity.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, str(script), "--output-dir", temporary_directory],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        self.assertEqual(0, completed.returncode, msg=completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
