from __future__ import annotations

import importlib.util
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "small_experiments"
    / "airfogsim_cross_graph_evidence_closure.py"
)


def load_subject():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("exp03_subject", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SharedParentOutputEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()
        self.edges = [
            {"id": "ie::m0::m1", "src": "m0", "dst": "m1"},
            {"id": "ie::m0::m2", "src": "m0", "dst": "m2"},
        ]
        self.tasks = [
            {"id": "m0", "source": "p0", "exec": "p1", "returned_size": 4.0},
            {"id": "m1", "source": "p0", "exec": "p2", "returned_size": 1.0},
            {"id": "m2", "source": "p0", "exec": "p3", "returned_size": 1.0},
        ]
        self.return_events = [
            {
                "event_id": "event::m0::return::0",
                "task_id": "m0",
                "phase": "return",
                "source": "p1",
                "target": "p0",
                "path": ["pe::p1::p0"],
                "delivered_data": 4.0,
                "flow_completed": True,
                "evidence": "direct_runtime_channel_event",
            }
        ]

    def test_script_exists_before_evidence_model_can_be_used(self):
        self.assertTrue(SCRIPT_PATH.exists(), msg="exp03 evidence script has not been implemented")

    def test_two_children_share_one_parent_output_flow_without_double_counting_bytes(self):
        if self.subject is None:
            self.fail("exp03 evidence script has not been implemented")
        builder = getattr(self.subject, "build_shared_parent_output_relations", None)
        self.assertTrue(callable(builder), msg="shared-parent-output builder is missing")
        if not callable(builder):
            return

        relations, flows = builder(self.edges, self.tasks, self.return_events)

        self.assertEqual({"flow::m0::return"}, {row["dependency_flow_id"] for row in relations})
        self.assertEqual(4.0, sum(row["physical_delivered_data"] for row in flows))
        self.assertTrue(all(row["dependency_payload"] == 4.0 for row in relations))
        self.assertTrue(all(row["dependency_status"] == "arrived" for row in relations))
        self.assertTrue(all(row["payload_semantics"] == "shared_parent_output" for row in relations))
        self.assertTrue(
            all(
                row["evidence_source"]
                == "pijwm_declared_semantics_airfogsim_observed_flow"
                for row in relations
            )
        )

    def test_offload_event_cannot_be_promoted_to_parent_output_evidence(self):
        if self.subject is None:
            self.fail("exp03 evidence script has not been implemented")
        builder = getattr(self.subject, "build_shared_parent_output_relations", None)
        self.assertTrue(callable(builder), msg="shared-parent-output builder is missing")
        if not callable(builder):
            return
        offload_event = dict(self.return_events[0], phase="offload")

        relations, flows = builder(self.edges, self.tasks, [offload_event])

        self.assertEqual([], flows)
        self.assertTrue(all(row["evidence"] == "not_observed" for row in relations))
        self.assertTrue(all(row["dependency_status"] == "pending" for row in relations))

    def test_shared_flow_accounting_rejects_duplicate_physical_bytes(self):
        if self.subject is None:
            self.fail("exp03 evidence script has not been implemented")
        builder = getattr(self.subject, "build_shared_parent_output_relations", None)
        validator = getattr(self.subject, "validate_shared_flow_accounting", None)
        self.assertTrue(callable(builder), msg="shared-parent-output builder is missing")
        self.assertTrue(callable(validator), msg="shared-flow accounting validator is missing")
        if not callable(builder) or not callable(validator):
            return
        relations, flows = builder(self.edges, self.tasks, self.return_events)
        duplicated = flows + [dict(flows[0], event_id="duplicate-event")]

        errors = validator(relations, duplicated)

        self.assertIn("duplicate_dependency_flow", errors)

    def test_airfogsim_return_size_field_is_used_as_dependency_payload(self):
        if self.subject is None:
            self.fail("exp03 evidence script has not been implemented")
        builder = getattr(self.subject, "build_shared_parent_output_relations", None)
        self.assertTrue(callable(builder), msg="shared-parent-output builder is missing")
        if not callable(builder):
            return
        tasks = [dict(row) for row in self.tasks]
        tasks[0].pop("returned_size")
        tasks[0]["return_size"] = 4.0

        relations, _ = builder(self.edges, tasks, self.return_events)

        self.assertTrue(all(row["dependency_payload"] == 4.0 for row in relations))


class WirelessTransferEventRecorderTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_recorder_clips_delivered_data_to_remaining_return_payload(self):
        recorder_type = getattr(self.subject, "WirelessTransferEventRecorder", None)
        self.assertTrue(callable(recorder_type), msg="wireless transfer recorder is missing")
        if not callable(recorder_type):
            return

        class FakeChannelManager:
            @staticmethod
            def getRateByChannelType(tx_idx, rx_idx, channel_type, rb_indices):
                self.assertEqual((2, 0, "V2I", [1, 3]), (tx_idx, rx_idx, channel_type, rb_indices))
                return [10.0, 20.0]

        class FakeEnv:
            simulation_time = 1.2
            simulation_interval = 0.1
            channel_manager = FakeChannelManager()

        recorder = recorder_type()
        event = recorder.make_event(
            FakeEnv(),
            {
                "task_id": "m0",
                "tx_idx": 2,
                "rx_idx": 0,
                "channel_type": "V2I",
                "RB_Nos": [1, 3],
            },
            {
                "phase": "return",
                "source": "p1",
                "target": "p0",
                "transmitted_before": 2.0,
                "required_size": 4.0,
                "sequence": 7,
            },
        )

        self.assertEqual([1, 3], event["rb_indices"])
        self.assertEqual([10.0, 20.0], event["rate_by_rb"])
        self.assertAlmostEqual(3.0, event["planned_capacity"])
        self.assertAlmostEqual(2.0, event["delivered_data"])
        self.assertTrue(event["flow_completed"])
        self.assertEqual(["pe::p1::p0"], event["path"])
        self.assertEqual("direct_runtime_channel_event", event["evidence"])

    def test_observed_me_path_is_assembled_only_from_runtime_channel_events(self):
        builder = getattr(self.subject, "build_observed_me_relations", None)
        self.assertTrue(callable(builder), msg="observed ME path builder is missing")
        if not callable(builder):
            return
        events = [
            {
                "event_id": "e0",
                "task_id": "m0",
                "phase": "offload",
                "source": "p0",
                "target": "p1",
                "path": ["pe::p0::p1"],
                "sequence": 0,
                "time": 0.1,
                "delivered_data": 1.0,
                "evidence": "direct_runtime_channel_event",
            },
            {
                "event_id": "e1",
                "task_id": "m0",
                "phase": "offload",
                "source": "p1",
                "target": "p2",
                "path": ["pe::p1::p2"],
                "sequence": 1,
                "time": 0.2,
                "delivered_data": 1.0,
                "evidence": "direct_runtime_channel_event",
            },
            {
                "event_id": "scheduler-only",
                "task_id": "m1",
                "phase": "offload",
                "source": "p0",
                "target": "p3",
                "path": ["pe::p0::p3"],
                "sequence": 0,
                "time": 0.1,
                "delivered_data": 1.0,
                "evidence": "direct_scheduler_decision",
            },
        ]

        relations = builder(events)

        self.assertEqual(1, len(relations))
        self.assertEqual("m0", relations[0]["task"])
        self.assertEqual(["pe::p0::p1", "pe::p1::p2"], relations[0]["path"])
        self.assertEqual("p0", relations[0]["source"])
        self.assertEqual("p2", relations[0]["target"])
        self.assertEqual("direct_runtime_channel_event", relations[0]["evidence"])

    def test_repeated_timeslots_on_one_link_are_one_topological_hop(self):
        builder = getattr(self.subject, "build_observed_me_relations", None)
        self.assertTrue(callable(builder), msg="observed ME path builder is missing")
        if not callable(builder):
            return
        events = [
            {
                "event_id": f"e{index}",
                "task_id": "m0",
                "phase": "offload",
                "source": "p0",
                "target": "p1",
                "path": ["pe::p0::p1"],
                "sequence": index,
                "time": 0.1 * (index + 1),
                "delivered_data": delivered,
                "evidence": "direct_runtime_channel_event",
            }
            for index, delivered in enumerate((0.4, 0.6))
        ]

        relations = builder(events)

        self.assertEqual(["pe::p0::p1"], relations[0]["path"])
        self.assertEqual(["e0", "e1"], relations[0]["event_ids"])
        self.assertAlmostEqual(1.0, relations[0]["physical_delivered_data"])

    def test_repeated_return_timeslots_are_one_dependency_path_hop(self):
        builder = getattr(self.subject, "build_shared_parent_output_relations", None)
        self.assertTrue(callable(builder), msg="shared-parent-output builder is missing")
        if not callable(builder):
            return
        tasks = [
            {"id": "m0", "source": "p0", "exec": "p1", "returned_size": 1.0},
            {"id": "m1", "source": "p0", "exec": "p2", "returned_size": 1.0},
        ]
        events = [
            {
                "event_id": f"r{index}",
                "task_id": "m0",
                "phase": "return",
                "source": "p1",
                "target": "p0",
                "path": ["pe::p1::p0"],
                "sequence": index,
                "time": 0.1 * (index + 1),
                "delivered_data": delivered,
                "flow_completed": index == 1,
                "evidence": "direct_runtime_channel_event",
            }
            for index, delivered in enumerate((0.4, 0.6))
        ]

        _, flows = builder([{"id": "ie::m0::m1", "src": "m0", "dst": "m1"}], tasks, events)

        self.assertEqual(["pe::p1::p0"], flows[0]["path"])
        self.assertEqual(["r0", "r1"], flows[0]["event_ids"])
        self.assertAlmostEqual(1.0, flows[0]["physical_delivered_data"])


def valid_exp03_bundle(subject):
    information_edges = [{"id": "ie::m0::m1", "src": "m0", "dst": "m1"}]
    tasks = [
        {"id": "m0", "source": "p0", "host": "p0", "exec": "p1", "ret": "p0", "returned_size": 4.0},
        {"id": "m1", "source": "p0", "host": "p0", "exec": "p2", "ret": "p0", "returned_size": 1.0},
    ]
    transfer_events = [
        {
            "event_id": "event::m0::return::0",
            "task_id": "m0",
            "phase": "return",
            "source": "p1",
            "target": "p0",
            "path": ["pe::p1::p0"],
            "delivered_data": 4.0,
            "flow_completed": True,
            "sequence": 0,
            "time": 0.5,
            "evidence": "direct_runtime_channel_event",
        },
        {
            "event_id": "event::m1::offload::0",
            "task_id": "m1",
            "phase": "offload",
            "source": "p0",
            "target": "p2",
            "path": ["pe::p0::p2"],
            "delivered_data": 2.0,
            "flow_completed": True,
            "sequence": 0,
            "time": 0.6,
            "evidence": "direct_runtime_channel_event",
        },
    ]
    ep_relations, dependency_flows = subject.build_shared_parent_output_relations(
        information_edges,
        tasks,
        transfer_events,
    )
    return {
        "physical_nodes": [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}],
        "physical_edges": [
            {"id": "pe::p1::p0", "src": "p1", "dst": "p0", "source_interface": "AirFogSim channel manager"},
            {"id": "pe::p0::p2", "src": "p0", "dst": "p2", "source_interface": "AirFogSim channel manager"},
        ],
        "information_nodes": tasks,
        "information_edges": information_edges,
        "mn_relations": [
            {"task": task["id"], "relation": relation, "physical_node": task[relation], "evidence": "direct"}
            for task in tasks
            for relation in ("source", "host", "exec", "ret")
        ],
        "transfer_events": transfer_events,
        "me_relations": subject.build_observed_me_relations(transfer_events),
        "dependency_flows": dependency_flows,
        "ep_relations": ep_relations,
    }


class Exp03StrictGateTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_controlled_direct_bundle_passes_every_exp03_gate(self):
        validator = getattr(self.subject, "validate_exp03_bundle", None)
        self.assertTrue(callable(validator), msg="exp03 strict validator is missing")
        if not callable(validator):
            return

        report = validator(valid_exp03_bundle(self.subject))

        self.assertTrue(report["strict_dual_graph_ready"])
        self.assertEqual([], report["failed_checks"])
        self.assertTrue(all(row["passed"] for row in report["checks"]))

    def test_me_channel_gate_rejects_relation_whose_event_was_removed(self):
        validator = getattr(self.subject, "validate_exp03_bundle", None)
        self.assertTrue(callable(validator), msg="exp03 strict validator is missing")
        if not callable(validator):
            return
        bundle = valid_exp03_bundle(self.subject)
        bundle["transfer_events"] = bundle["transfer_events"][1:]

        report = validator(bundle)

        self.assertIn("me_channel_observed", report["failed_checks"])

    def test_four_destructive_evidence_cases_are_detected(self):
        builder = getattr(self.subject, "build_exp03_corruption_report", None)
        self.assertTrue(callable(builder), msg="exp03 corruption builder is missing")
        if not callable(builder):
            return

        report = builder(valid_exp03_bundle(self.subject))

        self.assertTrue(report["all_corruptions_detected"])
        self.assertEqual(
            {"missing_channel_event", "offload_as_parent_output", "disconnected_me_path", "duplicate_shared_flow"},
            {row["case"] for row in report["cases"]},
        )
        self.assertTrue(all(row["detected"] for row in report["cases"]))


class Exp03WriterAndCliTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_combined_transmission_view_does_not_mutate_task_manager_lists(self):
        class Task:
            def __init__(self, task_id):
                self.task_id = task_id

            def getTaskId(self):
                return self.task_id

        class Manager:
            def __init__(self):
                self._offloading_tasks = {"node_0": [Task("input")]}
                self._returning_tasks = {"node_0": [Task("return")]}

        manager = Manager()
        first, first_count = self.subject.nonmutating_transmission_tasks(manager)
        second, second_count = self.subject.nonmutating_transmission_tasks(manager)

        self.assertEqual(["input"], [task.getTaskId() for task in manager._offloading_tasks["node_0"]])
        self.assertEqual(["input", "return"], [task.getTaskId() for task in first["node_0"]])
        self.assertEqual(["input", "return"], [task.getTaskId() for task in second["node_0"]])
        self.assertEqual(2, first_count)
        self.assertEqual(2, second_count)

    def test_writer_freezes_complete_reproducible_evidence_bundle(self):
        runner = getattr(self.subject, "run_exp03", None)
        self.assertTrue(callable(runner), msg="exp03 writer is missing")
        if not callable(runner):
            return

        def fake_runtime(seed: int, max_time: float):
            return {
                "config": {"seed": seed, "max_time": max_time},
                "bundle": valid_exp03_bundle(self.subject),
                "runtime_summary": {"seed": seed, "max_time": max_time},
            }

        expected_files = {
            "REPORT.md",
            "bundle.json",
            "config_snapshot.json",
            "corruption_report.json",
            "dependency_flows.csv",
            "ep_relations.csv",
            "information_edges.csv",
            "information_nodes.csv",
            "manifest.json",
            "me_relations.csv",
            "mn_relations.csv",
            "physical_edges.csv",
            "physical_edge_snapshots.csv",
            "physical_nodes.csv",
            "physical_node_snapshots.csv",
            "runtime_summary.json",
            "task_snapshots.csv",
            "transfer_events.csv",
            "offload_actions.csv",
            "return_actions.csv",
            "rb_actions.csv",
            "validation_report.json",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = runner(output_dir, seed=3, max_time=4.0, runtime_runner=fake_runtime)
            validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(expected_files, {path.name for path in output_dir.iterdir()})
            self.assertTrue(result["strict_dual_graph_ready"])
            self.assertTrue(validation["strict_dual_graph_ready"])
            self.assertTrue(validation["reproducibility_passed"])
            self.assertTrue(validation["corruption_detection_passed"])
            self.assertEqual(21, len(manifest["files"]))
            self.assertIn("airfogsim_source_sha256", manifest["source_code"])
            self.assertIn("exp02_preflight_script_sha256", manifest["source_code"])
            self.assertIn("packages", manifest["environment"])
            self.assertEqual(
                "local_environment_snapshot_not_a_complete_transitive_lockfile",
                manifest["environment"]["scope"],
            )

    def test_reproducibility_is_a_strict_readiness_gate(self):
        runner = getattr(self.subject, "run_exp03", None)
        self.assertTrue(callable(runner), msg="exp03 writer is missing")
        if not callable(runner):
            return
        calls = {"count": 0}

        def changing_runtime(seed: int, max_time: float):
            calls["count"] += 1
            bundle = valid_exp03_bundle(self.subject)
            bundle["physical_nodes"][0]["run_nonce"] = calls["count"]
            return {"config": {"seed": seed}, "bundle": bundle, "runtime_summary": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = runner(Path(temporary_directory), 0, 1.0, changing_runtime)

        self.assertFalse(result["strict_dual_graph_ready"])
        self.assertIn("same_seed_reproducible", result["failed_checks"])

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


class ContractAdapterIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_directed_link_export_covers_all_nine_airfogsim_directions(self):
        exporter = getattr(self.subject, "all_directed_link_rows", None)
        self.assertTrue(callable(exporter), msg="nine-direction link exporter is missing")
        if not callable(exporter):
            return

        class FakeEnv:
            simulation_time = 1.0
            vehicles = {"vehicle_0": object(), "vehicle_1": object()}
            UAVs = {"UAV_0": object(), "UAV_1": object()}
            RSUs = {"RSU_0": object(), "RSU_1": object()}

            @staticmethod
            def getDistanceBetweenNodesById(source, target):
                return 1.0

        rows = exporter(
            FakeEnv(),
            active={},
            rate_reader=lambda env, source, target, link_type: 2.0,
            csi_reader=lambda env, source, target: 3.0,
        )

        self.assertEqual(
            {"V2V", "V2U", "V2I", "U2V", "U2U", "U2I", "I2V", "I2U", "I2I"},
            {row["link_type"] for row in rows},
        )
        self.assertTrue(all(row["tx_id"] != row["rx_id"] for row in rows))
        self.assertEqual(30, len(rows))

    def test_exp03_installs_capacity_safe_cpu_callback(self):
        installer = getattr(self.subject, "install_capacity_safe_cpu_callback", None)
        self.assertTrue(callable(installer), msg="exp03 does not install the PI-JWM CPU adapter")
        if not callable(installer):
            return

        class Task:
            def __init__(self, task_id):
                self.task_id = task_id

            def getTaskId(self):
                return self.task_id

        class Node:
            def __init__(self, cpu):
                self.cpu = cpu

            def getFogProfile(self):
                return {"cpu": self.cpu}

        class Env:
            def __init__(self):
                self.nodes = {"a": Node(2.0), "b": Node(8.0)}

            def _getNodeById(self, node_id):
                return self.nodes[node_id]

        class Scheduler:
            def setComputingCallBack(self, env, callback):
                self.env = env
                self.callback = callback

        env = Env()
        scheduler = Scheduler()
        installer(env, scheduler)
        allocations = scheduler.callback(
            {"a": [Task("a0"), Task("a1")], "b": [Task("b0"), Task("b1")]}
        )

        self.assertEqual({"a0": 1.0, "a1": 1.0, "b0": 4.0, "b1": 4.0}, allocations)

    def test_exp03_repairs_channel_energy_inputs_from_direct_events(self):
        repair = getattr(self.subject, "repair_channel_energy_inputs", None)
        self.assertTrue(callable(repair), msg="exp03 does not repair the channel-energy boundary")
        if not callable(repair):
            return

        class Manager:
            def setThisTimeslotTransSize(self, sending, receiving):
                self.sending = dict(sending)
                self.receiving = dict(receiving)

        manager = Manager()
        repair(
            manager,
            [
                {"source": "UAV_0", "target": "RSU_0", "planned_capacity": 2.0},
                {"source": "UAV_0", "target": "vehicle_0", "planned_capacity": 3.0},
            ],
        )

        self.assertEqual({"UAV_0": 5.0}, manager.sending)
        self.assertEqual({"RSU_0": 2.0, "vehicle_0": 3.0}, manager.receiving)


if __name__ == "__main__":
    unittest.main()
