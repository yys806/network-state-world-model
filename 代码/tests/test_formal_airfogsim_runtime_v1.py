from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "formal_airfogsim_runtime_v1.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location("formal_airfogsim_runtime_v1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load formal AirFogSim runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTask:
    def getTaskId(self):
        return "Task_1"

    def getTaskArrivalTime(self):
        return 0.0

    def getTaskDeadline(self):
        return 2.0

    def getTaskCPU(self):
        return 5.0

    def getComputedSize(self):
        return 1.0

    def getTaskPriority(self):
        return 1.0


class FakeNode:
    def getFogProfile(self):
        return {"cpu": 10.0}


class FakeEnv:
    simulation_time = 1.0

    def _getNodeById(self, node_id):
        return FakeNode() if node_id == "RSU_0" else None


class FakeScheduler:
    def __init__(self):
        self.callback = None

    def setComputingCallBack(self, env, callback):
        self.callback = callback


class FormalAirFogSimRuntimeTests(unittest.TestCase):
    def test_wrapper_applies_scenario_cpu_policy_and_restores_dependencies(self):
        subject = load_subject()
        from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs

        spec = next(
            row
            for row in build_formal_trajectory_specs()
            if row.cpu_policy == "deadline_aware"
        )
        calls = {"original_install": 0}

        def original_build_config(config, seed, max_time):
            return config

        def original_install(env, scheduler):
            calls["original_install"] += 1

        preflight = SimpleNamespace(build_preflight_config=original_build_config)
        evidence = SimpleNamespace(install_capacity_safe_cpu_callback=original_install)

        def conservation_runner(seed, max_time):
            config = preflight.build_preflight_config(
                {
                    "traffic": {"max_n_vehicles": 5, "arrival_lambda": 0.1},
                    "task": {"task_generation_kwargs": {"lambda": 9.0}},
                    "task_profile": {
                        "vehicle": {"lambda": 9.0},
                        "uav": {"lambda": 9.0},
                    },
                },
                seed,
                max_time,
            )
            env = FakeEnv()
            scheduler = FakeScheduler()
            evidence.install_capacity_safe_cpu_callback(env, scheduler)
            allocations = scheduler.callback({"RSU_0": [FakeTask()]})
            return {
                "config": config,
                "bundle": {
                    "cpu_ledger": [
                        {
                            "record_id": "cpu::Task_1::1.000000",
                            "time": 1.0,
                            "task_id": "Task_1",
                            "node_id": "RSU_0",
                            "allocated_cpu": allocations["Task_1"],
                            "node_cpu_capacity": 10.0,
                        }
                    ],
                    "dependency_ledger": [{"kind": "dependency_flow"}],
                },
                "source_bundle": {
                    "physical_nodes": [
                        {"id": "RSU_0", "trajectory_id": "old"}
                    ],
                    "information_edges": [
                        {
                            "id": "dag::Task_0::Task_1",
                            "src": "Task_0",
                            "dst": "Task_1",
                            "data_mb": None,
                            "semantic": "precedence_only",
                            "trajectory_id": "old",
                        }
                    ],
                    "dependency_flows": [{"data_mb": 1.0}],
                    "ep_relations": [{"dependency_status": "arrived"}],
                },
                "runtime_summary": {"seed": seed},
            }

        result = subject.run_formal_airfogsim_trajectory(
            spec,
            max_time=30.0,
            evidence_module=evidence,
            preflight_module=preflight,
            conservation_runner=conservation_runner,
        )

        self.assertEqual(
            spec.scenario.max_vehicles,
            result["config"]["traffic"]["max_n_vehicles"],
        )
        self.assertEqual(
            spec.scenario.task_lambda,
            result["config"]["task_profile"]["vehicle"]["lambda"],
        )
        cpu_row = result["bundle"]["cpu_ledger"][0]
        self.assertEqual("deadline_aware", cpu_row["policy_id"])
        self.assertEqual(1.0, cpu_row["allocated_fraction"])
        self.assertEqual([], result["bundle"]["dependency_ledger"])
        self.assertEqual([], result["source_bundle"]["dependency_flows"])
        self.assertEqual([], result["source_bundle"]["ep_relations"])
        self.assertIsNone(result["source_bundle"]["information_edges"][0]["data_mb"])
        self.assertEqual(
            spec.trajectory_id,
            result["source_bundle"]["physical_nodes"][0]["trajectory_id"],
        )
        self.assertIs(original_build_config, preflight.build_preflight_config)
        self.assertIs(original_install, evidence.install_capacity_safe_cpu_callback)
        self.assertEqual(0, calls["original_install"])

    def test_wrapper_restores_patches_when_runner_fails(self):
        subject = load_subject()
        from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs

        original_build = lambda config, seed, max_time: config
        original_install = lambda env, scheduler: None
        preflight = SimpleNamespace(build_preflight_config=original_build)
        evidence = SimpleNamespace(install_capacity_safe_cpu_callback=original_install)

        with self.assertRaisesRegex(RuntimeError, "runner failed"):
            subject.run_formal_airfogsim_trajectory(
                build_formal_trajectory_specs()[0],
                max_time=1.0,
                evidence_module=evidence,
                preflight_module=preflight,
                conservation_runner=lambda seed, max_time: (_ for _ in ()).throw(
                    RuntimeError("runner failed")
                ),
            )

        self.assertIs(original_build, preflight.build_preflight_config)
        self.assertIs(original_install, evidence.install_capacity_safe_cpu_callback)


if __name__ == "__main__":
    unittest.main()
