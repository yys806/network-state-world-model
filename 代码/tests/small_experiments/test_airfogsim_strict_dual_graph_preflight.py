from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "small_experiments"
    / "airfogsim_strict_dual_graph_preflight.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))

import airfogsim_strict_dual_graph_preflight as preflight


def minimal_config() -> dict:
    return {
        "simulation": {"max_simulation_time": 100, "simulation_interval": 0.1},
        "traffic": {"max_n_vehicles": 50, "max_n_UAVs": 10, "RSU_positions": []},
        "task": {
            "task_generation_kwargs": {"lambda": 0.5},
            "task_min_required_returned_size": 0,
            "task_max_required_returned_size": 0,
            "required_returned_size_kwargs": {"low": 0, "high": 0},
            "task_min_deadline": 1,
            "task_max_deadline": 2,
            "deadline_kwargs": {"low": 0.5, "high": 1},
        },
        "task_profile": {
            "task_node_gen_poss": 0.5,
            "task_node_profiles": [
                {"type": "UAV", "max_node_num": 30},
                {"type": "vehicle", "max_node_num": 20},
            ],
            "vehicle": {"lambda": 0.5, "dag_edge_prob": 0.0},
            "uav": {"lambda": 0.5, "dag_edge_prob": 0.0},
        },
    }


def valid_direct_bundle() -> dict:
    e01 = "pe::p0::p1"
    e12 = "pe::p1::p2"
    return {
        "physical_nodes": [
            {"id": "p0", "kind": "vehicle", "evidence": "direct"},
            {"id": "p1", "kind": "rsu", "evidence": "direct"},
            {"id": "p2", "kind": "uav", "evidence": "direct"},
        ],
        "physical_edges": [
            {"id": e01, "src": "p0", "dst": "p1", "evidence": "direct", "source_interface": "AirFogSim channel manager"},
            {"id": e12, "src": "p1", "dst": "p2", "evidence": "direct", "source_interface": "AirFogSim channel manager"},
        ],
        "information_nodes": [
            {"id": "m0", "evidence": "direct"},
            {"id": "m1", "evidence": "direct"},
        ],
        "information_edges": [
            {
                "id": "ie::m0::m1",
                "src": "m0",
                "dst": "m1",
                "data_mb": 0.25,
                "semantic": "dependency_payload",
                "evidence": "direct",
            }
        ],
        "mn_relations": [
            {"task": "m0", "relation": "source", "physical_node": "p0", "evidence": "direct"},
            {"task": "m0", "relation": "host", "physical_node": "p0", "evidence": "direct"},
            {"task": "m0", "relation": "exec", "physical_node": "p1", "evidence": "direct"},
            {"task": "m0", "relation": "ret", "physical_node": "p0", "evidence": "direct"},
            {"task": "m1", "relation": "source", "physical_node": "p0", "evidence": "direct"},
            {"task": "m1", "relation": "host", "physical_node": "p0", "evidence": "direct"},
            {"task": "m1", "relation": "exec", "physical_node": "p2", "evidence": "direct"},
            {"task": "m1", "relation": "ret", "physical_node": "p0", "evidence": "direct"},
        ],
        "me_relations": [
            {
                "task": "m0",
                "relation": "in",
                "source": "p0",
                "target": "p1",
                "path": [e01],
                "evidence": "direct",
            }
        ],
        "ep_relations": [
            {
                "info_edge": "ie::m0::m1",
                "source": "p1",
                "target": "p2",
                "path": [e12],
                "data_mb": 0.25,
                "evidence": "direct",
            }
        ],
    }


class AirFogSimStrictDualGraphPreflightTests(unittest.TestCase):
    def test_preflight_script_entrypoint_exists(self):
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_build_preflight_config_enables_real_dag_and_return_flow_without_mutating_source(self):
        builder = getattr(preflight, "build_preflight_config", None)
        self.assertTrue(callable(builder))
        if not callable(builder):
            return
        source = minimal_config()
        source_before = copy.deepcopy(source)

        configured = builder(source, seed=7, max_time=6.0)

        self.assertEqual(source_before, source)
        self.assertEqual(6.0, configured["simulation"]["max_simulation_time"])
        self.assertGreater(configured["task_profile"]["vehicle"]["dag_edge_prob"], 0)
        self.assertGreater(configured["task_profile"]["uav"]["dag_edge_prob"], 0)
        self.assertGreater(configured["task"]["required_returned_size_kwargs"]["low"], 0)
        self.assertGreater(configured["task"]["deadline_kwargs"]["low"], 1)
        self.assertEqual(7, configured["pi_jwm_preflight"]["seed"])
        self.assertIn("trajectory_id", configured["pi_jwm_preflight"])

    def test_canonical_json_hash_is_order_independent_and_can_ignore_volatile_keys(self):
        hasher = getattr(preflight, "canonical_json_hash", None)
        self.assertTrue(callable(hasher))
        if not callable(hasher):
            return
        left = {"b": 2, "a": {"x": 1}, "generated_at": "first"}
        right = {"generated_at": "second", "a": {"x": 1}, "b": 2}

        self.assertEqual(
            hasher(left, exclude_keys={"generated_at"}),
            hasher(right, exclude_keys={"generated_at"}),
        )

    def test_route_nodes_are_converted_to_stable_directed_physical_edge_ids(self):
        converter = getattr(preflight, "route_nodes_to_edges", None)
        self.assertTrue(callable(converter))
        if not callable(converter):
            return
        self.assertEqual(
            ["pe::p0::p1", "pe::p1::p2"],
            converter(["p0", "p1", "p2"]),
        )
        self.assertEqual([], converter(["p0"]))

    def test_direct_bundle_passes_all_strict_graph_gates(self):
        validator = getattr(preflight, "validate_export_bundle", None)
        self.assertTrue(callable(validator))
        if not callable(validator):
            return

        report = validator(valid_direct_bundle())

        self.assertTrue(report["experiment_completed"])
        self.assertTrue(report["strict_dual_graph_ready"])
        self.assertEqual([], report["failed_checks"])

    def test_airfogsim_precedence_only_edge_is_not_promoted_to_direct_ep_truth(self):
        classifier = getattr(preflight, "classify_ep_evidence", None)
        validator = getattr(preflight, "validate_export_bundle", None)
        self.assertTrue(callable(classifier))
        self.assertTrue(callable(validator))
        if not callable(classifier) or not callable(validator):
            return
        bundle = valid_direct_bundle()
        bundle["information_edges"][0].update(
            {"data_mb": None, "semantic": "precedence_only", "evidence": "direct"}
        )
        bundle["ep_relations"] = [
            classifier(
                bundle["information_edges"][0],
                source="p1",
                target="p2",
                transfer_event=None,
            )
        ]

        report = validator(bundle)

        self.assertEqual("not_modeled", bundle["ep_relations"][0]["evidence"])
        self.assertIsNone(bundle["ep_relations"][0]["data_mb"])
        self.assertEqual([], bundle["ep_relations"][0]["path"])
        self.assertFalse(report["strict_dual_graph_ready"])
        self.assertIn("ep_directly_observed", report["failed_checks"])

    def test_corrupted_cycle_duplicate_mn_and_broken_me_path_are_each_rejected(self):
        validator = getattr(preflight, "validate_export_bundle", None)
        self.assertTrue(callable(validator))
        if not callable(validator):
            return

        cycle = copy.deepcopy(valid_direct_bundle())
        cycle["information_edges"].append(
            {"id": "ie::m1::m0", "src": "m1", "dst": "m0", "data_mb": 0.1, "semantic": "dependency_payload", "evidence": "direct"}
        )
        cycle["ep_relations"].append(
            {"info_edge": "ie::m1::m0", "source": "p2", "target": "p1", "path": [], "data_mb": 0.1, "evidence": "direct"}
        )
        duplicate_mn = copy.deepcopy(valid_direct_bundle())
        duplicate_mn["mn_relations"].append(
            {"task": "m0", "relation": "source", "physical_node": "p2", "evidence": "direct"}
        )
        broken_me = copy.deepcopy(valid_direct_bundle())
        broken_me["me_relations"][0]["path"] = ["pe::p1::p2"]

        self.assertIn("information_graph_is_dag", validator(cycle)["failed_checks"])
        self.assertIn("mn_cardinality", validator(duplicate_mn)["failed_checks"])
        self.assertIn("me_paths_valid", validator(broken_me)["failed_checks"])

    def test_scheduler_selected_route_does_not_count_as_channel_observed_me_edge(self):
        bundle = valid_direct_bundle()
        bundle["physical_edges"][0]["source_interface"] = "AirFogSim scheduler route"

        report = preflight.validate_export_bundle(bundle)

        self.assertIn("me_channel_observed", report["failed_checks"])
        self.assertFalse(report["strict_dual_graph_ready"])

    def test_dag_normalization_preserves_precedence_semantics_without_inventing_payload(self):
        normalizer = getattr(preflight, "normalize_airfogsim_dags", None)
        self.assertTrue(callable(normalizer))
        if not callable(normalizer):
            return
        dag = nx.DiGraph()
        dag.add_edge("m0", "m1")

        rows = normalizer({"vehicle_0": dag}, trajectory_id="traj", step=4, time_value=0.4)

        self.assertEqual(1, len(rows))
        self.assertEqual("m0", rows[0]["src"])
        self.assertEqual("m1", rows[0]["dst"])
        self.assertEqual("precedence_only", rows[0]["semantic"])
        self.assertIsNone(rows[0]["data_mb"])
        self.assertEqual("direct", rows[0]["evidence"])

    def test_ready_offload_selection_explicitly_requests_dependency_filtering(self):
        selector = getattr(preflight, "select_ready_offload_decisions", None)
        self.assertTrue(callable(selector))
        if not callable(selector):
            return

        class FakeTaskScheduler:
            def __init__(self):
                self.check_dependency = None

            def getAllToOffloadTaskInfos(self, env, check_dependency=False):
                self.check_dependency = check_dependency
                return [{"task_node_id": "p0", "task_id": "m0"}]

        class FakeEntityScheduler:
            @staticmethod
            def getNeighborNodeInfosById(env, node_id, sorted_by, max_num):
                return [{"id": "p1", "distance": 3.5}]

        scheduler = FakeTaskScheduler()
        decisions = selector(object(), scheduler, FakeEntityScheduler())

        self.assertTrue(scheduler.check_dependency)
        self.assertEqual(
            [{"task_node_id": "p0", "task_id": "m0", "source_node_id": "p0", "target_node_id": "p1", "route_nodes": ["p0", "p1"], "distance": 3.5}],
            decisions,
        )

    def test_assembled_airfogsim_bundle_keeps_ep_missing_instead_of_using_shortest_path(self):
        assembler = getattr(preflight, "assemble_airfogsim_bundle", None)
        self.assertTrue(callable(assembler))
        if not callable(assembler):
            return
        trajectory_id = "traj"
        physical_nodes = [
            {"id": "p0", "kind": "vehicle", "evidence": "direct"},
            {"id": "p1", "kind": "rsu", "evidence": "direct"},
            {"id": "p2", "kind": "uav", "evidence": "direct"},
        ]
        physical_edges = [
            {"id": "pe::p0::p1", "src": "p0", "dst": "p1", "evidence": "direct", "source_interface": "AirFogSim channel manager"},
            {"id": "pe::p1::p2", "src": "p1", "dst": "p2", "evidence": "direct", "source_interface": "AirFogSim channel manager"},
        ]
        task_records = [
            {"id": "m0", "source": "p0", "host": "p1", "exec": "p1", "ret": "p0", "evidence": "direct"},
            {"id": "m1", "source": "p0", "host": "p0", "exec": "p2", "ret": "p0", "evidence": "direct"},
        ]
        dag_edges = [
            {"id": "ie::m0::m1", "src": "m0", "dst": "m1", "data_mb": None, "semantic": "precedence_only", "evidence": "direct"}
        ]
        offload_actions = [
            {"task_id": "m0", "source_node_id": "p0", "target_node_id": "p1", "route_nodes": ["p0", "p1"], "evidence": "direct"}
        ]

        bundle = assembler(
            trajectory_id=trajectory_id,
            physical_nodes=physical_nodes,
            physical_edges=physical_edges,
            task_records=task_records,
            dag_edges=dag_edges,
            offload_actions=offload_actions,
            return_actions=[],
        )
        report = preflight.validate_export_bundle(bundle)

        self.assertEqual("not_modeled", bundle["ep_relations"][0]["evidence"])
        self.assertEqual([], bundle["ep_relations"][0]["path"])
        self.assertFalse(report["strict_dual_graph_ready"])
        self.assertEqual(["ep_directly_observed"], report["failed_checks"])

    def test_experiment_writer_freezes_bundle_checks_reproducibility_and_detects_corruption(self):
        runner = getattr(preflight, "run_preflight_experiment", None)
        self.assertTrue(callable(runner))
        if not callable(runner):
            return

        def fake_runtime(seed: int, max_time: float) -> dict:
            bundle = preflight.assemble_airfogsim_bundle(
                trajectory_id="traj",
                physical_nodes=[
                    {"id": "p0", "kind": "vehicle", "evidence": "direct"},
                    {"id": "p1", "kind": "rsu", "evidence": "direct"},
                    {"id": "p2", "kind": "uav", "evidence": "direct"},
                ],
                physical_edges=[
                    {"id": "pe::p0::p1", "src": "p0", "dst": "p1", "evidence": "direct", "source_interface": "AirFogSim channel manager"},
                    {"id": "pe::p1::p2", "src": "p1", "dst": "p2", "evidence": "direct", "source_interface": "AirFogSim channel manager"},
                ],
                task_records=[
                    {"id": "m0", "source": "p0", "host": "p1", "exec": "p1", "ret": "p0", "evidence": "direct"},
                    {"id": "m1", "source": "p0", "host": "p0", "exec": "p2", "ret": "p0", "evidence": "direct"},
                ],
                dag_edges=[
                    {"id": "ie::m0::m1", "src": "m0", "dst": "m1", "data_mb": None, "semantic": "precedence_only", "evidence": "direct"}
                ],
                offload_actions=[
                    {"task_id": "m0", "source_node_id": "p0", "target_node_id": "p1", "route_nodes": ["p0", "p1"], "evidence": "direct"}
                ],
                return_actions=[],
            )
            return {
                "config": {"seed": seed, "max_time": max_time},
                "bundle": bundle,
                "runtime_summary": {"seed": seed, "steps": 4, "dag_edges": 1},
            }

        expected_files = {
            "REPORT.md",
            "bundle.json",
            "config_snapshot.json",
            "corruption_report.json",
            "ep_relations.csv",
            "information_edges.csv",
            "information_nodes.csv",
            "manifest.json",
            "me_relations.csv",
            "mn_relations.csv",
            "physical_edges.csv",
            "physical_nodes.csv",
            "runtime_summary.json",
            "validation_report.json",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            result = runner(output_dir, seed=3, max_time=4.0, runtime_runner=fake_runtime)

            self.assertEqual(expected_files, {path.name for path in output_dir.iterdir()})
            validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
            corruption = json.loads((output_dir / "corruption_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["experiment_completed"])
            self.assertFalse(validation["strict_dual_graph_ready"])
            self.assertEqual(["ep_directly_observed"], validation["failed_checks"])
            self.assertTrue(corruption["all_corruptions_detected"])
            self.assertTrue(manifest["reproducibility"]["same_seed_bundle_hash_equal"])
            self.assertTrue(manifest["reproducibility"].get("same_seed_config_hash_equal", False))
            self.assertEqual(
                hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
                manifest.get("source_code", {}).get("preflight_script_sha256"),
            )
            self.assertIn("airfogsim_source_sha256", manifest.get("source_code", {}))
            self.assertIn("python", manifest.get("environment", {}))
            self.assertIn("packages", manifest.get("environment", {}))
            self.assertFalse(result["strict_dual_graph_ready"])
            self.assertIn("当前已核验的AirFogSim任务、DAG和路由接口", (output_dir / "REPORT.md").read_text(encoding="utf-8"))

    def test_reproducibility_and_corruption_detection_are_part_of_overall_readiness_gate(self):
        runner = getattr(preflight, "run_preflight_experiment", None)
        self.assertTrue(callable(runner))
        if not callable(runner):
            return
        calls = {"count": 0}

        def changing_runtime(seed: int, max_time: float) -> dict:
            calls["count"] += 1
            bundle = copy.deepcopy(valid_direct_bundle())
            bundle["physical_nodes"][0]["run_nonce"] = calls["count"]
            return {"config": {"seed": seed}, "bundle": bundle, "runtime_summary": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = runner(Path(temporary_directory), 0, 1.0, changing_runtime)

        self.assertFalse(result["strict_dual_graph_ready"])
        self.assertIn("same_seed_reproducible", result["failed_checks"])

        def stable_runtime(seed: int, max_time: float) -> dict:
            return {"config": {"seed": seed}, "bundle": valid_direct_bundle(), "runtime_summary": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = runner(
                Path(temporary_directory),
                0,
                1.0,
                stable_runtime,
                corruption_builder=lambda bundle: {"all_corruptions_detected": False, "cases": []},
            )

        self.assertFalse(result["strict_dual_graph_ready"])
        self.assertIn("corruption_detection", result["failed_checks"])

    def test_cli_exposes_seed_time_and_output_arguments(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(0, completed.returncode, msg=completed.stderr)
        self.assertIn("--seed", completed.stdout)
        self.assertIn("--max-time", completed.stdout)
        self.assertIn("--output-dir", completed.stdout)

    def test_task_collection_includes_predictable_future_dag_nodes_omitted_by_get_all_tasks(self):
        collector = getattr(preflight, "iter_airfogsim_tasks", None)
        self.assertTrue(callable(collector))
        if not callable(collector):
            return

        class FakeTask:
            def __init__(self, task_id: str):
                self.task_id = task_id

            def getTaskId(self):
                return self.task_id

        current = FakeTask("m0")
        future = FakeTask("m1")

        class FakeManager:
            _to_generate_task_infos = {"p0": [future, current]}

            @staticmethod
            def getAllTasks():
                return [current]

        collected = collector(FakeManager())

        self.assertEqual(["m0", "m1"], [task.getTaskId() for task in collected])

    def test_task_record_exposes_real_outcome_and_progress_fields(self):
        serializer = getattr(preflight, "_task_record", None)
        self.assertTrue(callable(serializer))

        class FinishedTask:
            task_lifecycle_state = "finished"

            def getTaskId(self): return "m0"
            def getTaskNodeId(self): return "p0"
            def getCurrentNodeId(self): return "p1"
            def getAssignedTo(self): return "p1"
            def getToReturnNodeId(self): return "p0"
            def getTaskSize(self): return 0.4
            def getTaskCPU(self): return 0.8
            def getReturnedSize(self): return 0.1
            def getTaskArrivalTime(self): return 1.0
            def getTaskDeadline(self): return 5.0
            def getTaskPriority(self): return 0.75
            def getTransmittedSize(self): return 0.2
            def getComputedSize(self): return 0.8
            def getLastTransmissionTime(self): return 1.5
            def getLastComputeTime(self): return 2.2
            def getLastReturnTime(self): return 2.5
            def getLastOperationTime(self): return 2.5
            def getTaskFailureReason(self): return "Unknown code."
            def isFinished(self): return True

        row = serializer(FinishedTask(), "traj", 3.0)

        self.assertEqual("completed", row["terminal_status"])
        self.assertEqual(0.75, row["priority"])
        self.assertEqual(0.2, row["in_stage_transmitted_size"])
        self.assertEqual(0.8, row["computed_size"])
        self.assertEqual(2.5, row["completion_time"])
        self.assertEqual(1.5, row["task_delay"])
        self.assertIsNone(row["failure_reason"])


if __name__ == "__main__":
    unittest.main()
