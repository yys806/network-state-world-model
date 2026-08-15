from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_ROOT / "run_r6_joint_policy_gpu_readiness.py"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "run_r6_joint_policy_gpu_readiness",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R6 joint-policy readiness runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEnv:
    simulation_interval = 0.1

    def __init__(self) -> None:
        self.simulation_time = 0.0

    def isDone(self) -> bool:
        return False

    def step(self) -> None:
        self.simulation_time = round(self.simulation_time + self.simulation_interval, 10)


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.schedule_calls = 0

    def scheduleStep(self, env) -> None:
        self.schedule_calls += 1


class R6JointPolicyGpuReadinessRunnerTest(unittest.TestCase):
    def test_frozen_float32_time_is_quantized_to_airfogsim_interval(self) -> None:
        subject = load_subject()
        env = _FakeEnv()
        algorithm = _FakeAlgorithm()
        subject._step_default_until(env, algorithm, 8.100000381469727)
        self.assertAlmostEqual(8.1, env.simulation_time, places=9)
        self.assertEqual(81, algorithm.schedule_calls)

    def test_action_audit_fields_retain_exact_joint_bindings(self) -> None:
        subject = load_subject()
        from pi_jwm.r6_joint_action import (
            CPUAllocation,
            JointActionCandidate,
            OffloadBinding,
            RBAllocation,
        )

        candidate = JointActionCandidate.create(
            candidate_id="audit",
            template_id="deadline_first",
            offload=(OffloadBinding("task-1", "vehicle-1", "rsu-1"),),
            rb=(RBAllocation("task-1", (2, 3)),),
            cpu=(CPUAllocation("task-2", "rsu-1", 4.5),),
            descriptor=(0.0,) * JointActionCandidate.DESCRIPTOR_DIM,
        )
        record = SimpleNamespace(
            offload_applied_count=1,
            rb_task_count=1,
            cpu_task_count=1,
        )
        fields = subject._action_audit_fields(candidate, record, candidate_index=2)
        self.assertEqual(2, fields["candidate_index"])
        self.assertEqual("rsu-1", json.loads(fields["offload_plan_json"])[0]["target_node_id"])
        self.assertEqual([2, 3], json.loads(fields["rb_plan_json"])[0]["rb_ids"])
        self.assertEqual(4.5, json.loads(fields["cpu_plan_json"])[0]["amount"])

    def test_successful_regression_report_excludes_wall_clock_output(self) -> None:
        subject = load_subject()
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="",
            stderr="Ran 55 tests in 4.943s\nOK\n",
        )
        with mock.patch.object(subject.subprocess, "run", return_value=completed):
            report = subject._run_regression()
        self.assertTrue(report["passed"])
        self.assertEqual("", report["stdout_tail"])
        self.assertEqual("", report["stderr_tail"])
        self.assertGreater(report["test_file_count"], 0)


if __name__ == "__main__":
    unittest.main()
