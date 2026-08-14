from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.action_attempt_ledger_v1 import (  # noqa: E402
    ActionAttemptLedger,
    AttemptIdentity,
    validate_attempt_records,
)
from pi_jwm.airfogsim_full_dual_graph_collector_v2 import (  # noqa: E402
    CollectorAttemptRejected,
    execute_full_collector_step_v2,
)
from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    SnapshotPhase,
    TaskLifecycle,
)

from test_airfogsim_full_dual_graph_collector_v1 import (  # noqa: E402
    FakeEnv,
    FakeTask,
    SpySchedulers,
    built_for,
    phase_observer,
    physical_snapshot,
)


def prepared_attempt(built, *, run_role: str = "fixture"):
    ledger = ActionAttemptLedger()
    attempt = ledger.begin(
        AttemptIdentity(
            run_role=run_role,
            episode_id="episode-0",
            trajectory_id="traj0",
            frame_index=built.action.frame_index,
            candidate_ordinal=0,
        )
    )
    attempt.candidate_built(built.action)
    attempt.contract_validated()
    return ledger, attempt


def execute(env, built, schedulers, attempt, observer):
    return execute_full_collector_step_v2(
        env,
        built,
        attempt=attempt,
        trajectory_id="traj0",
        task_scheduler=schedulers,
        communication_scheduler=schedulers,
        computation_scheduler=schedulers,
        observer=observer,
    )


class RuntimeAdapterTests(unittest.TestCase):
    def test_pre_setter_validation_failure_has_no_runtime_call_or_step(self):
        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snapshot = physical_snapshot([task])
        built = built_for(snapshot)
        bad = replace(
            built,
            action=replace(
                built.action,
                rb_allocations=(
                    replace(built.action.rb_allocations[0], rb_index=99),
                ),
            ),
        )
        env = FakeEnv([task])
        schedulers = SpySchedulers()
        ledger, attempt = prepared_attempt(bad)

        with self.assertRaises(CollectorAttemptRejected) as caught:
            execute(env, bad, schedulers, attempt, phase_observer(snapshot, env))

        row = caught.exception.record
        self.assertEqual(row["terminal_stage"], "pre_setter_revalidation")
        self.assertEqual(row["environment_mutation_status"], "none")
        self.assertEqual(row["setter_calls"], [])
        self.assertFalse(row["env_step_called"])
        self.assertFalse(row["quarantined"])
        self.assertEqual(schedulers.calls, [])
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(validate_attempt_records(ledger.terminal_records()), ())

    def test_first_setter_exception_is_unknown_and_quarantined(self):
        class FailFirstScheduler(SpySchedulers):
            def setComputingCallBack(self, env, callback):
                del env, callback
                self.calls.append("cpu_started")
                raise RuntimeError("cpu callback install failed")

        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snapshot = physical_snapshot([task])
        built = built_for(snapshot)
        env = FakeEnv([task])
        schedulers = FailFirstScheduler()
        ledger, attempt = prepared_attempt(built)

        with self.assertRaises(CollectorAttemptRejected) as caught:
            execute(env, built, schedulers, attempt, phase_observer(snapshot, env))

        row = caught.exception.record
        self.assertEqual(row["terminal_stage"], "setter_application")
        self.assertEqual(row["environment_mutation_status"], "unknown_after_runtime_call")
        self.assertTrue(row["quarantined"])
        self.assertEqual(len(row["setter_calls"]), 1)
        self.assertEqual(row["setter_calls"][0]["setter_kind"], "cpu_callback")
        self.assertFalse(row["setter_calls"][0]["call_completed"])
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(validate_attempt_records(ledger.terminal_records()), ())

    def test_partial_setter_failure_is_confirmed_and_never_steps_or_retries(self):
        offload = FakeTask("off", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        returning = FakeTask(
            "ret", "u0", TaskLifecycle.WAITING_TO_RETURN, destination="v0"
        )
        snapshot = physical_snapshot([offload, returning])
        built = built_for(snapshot, frame_index=0)
        env = FakeEnv([offload, returning])
        schedulers = SpySchedulers(fail_return=True)
        ledger, attempt = prepared_attempt(built)

        with self.assertRaises(CollectorAttemptRejected) as caught:
            execute(env, built, schedulers, attempt, phase_observer(snapshot, env))

        row = caught.exception.record
        self.assertEqual(row["environment_mutation_status"], "confirmed")
        self.assertTrue(row["quarantined"])
        self.assertFalse(row["env_step_called"])
        self.assertEqual(
            [call["setter_kind"] for call in row["setter_calls"]],
            ["cpu_callback", "offload", "return_route"],
        )
        self.assertEqual(sum(call == ("return", "ret") for call in schedulers.calls), 1)
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(validate_attempt_records(ledger.terminal_records()), ())

    def test_env_step_exception_records_called_not_completed_and_restores_method(self):
        class StepFailureEnv(FakeEnv):
            def step(self):
                self.step_calls += 1
                raise RuntimeError("step failed")

        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snapshot = physical_snapshot([task])
        built = built_for(snapshot)
        env = StepFailureEnv([task])
        schedulers = SpySchedulers()
        ledger, attempt = prepared_attempt(built)
        self.assertNotIn("step", env.__dict__)

        with self.assertRaises(CollectorAttemptRejected) as caught:
            execute(env, built, schedulers, attempt, phase_observer(snapshot, env))

        row = caught.exception.record
        self.assertEqual(row["terminal_stage"], "env_step")
        self.assertTrue(row["env_step_called"])
        self.assertFalse(row["env_step_completed"])
        self.assertEqual(row["environment_mutation_status"], "confirmed")
        self.assertTrue(row["quarantined"])
        self.assertNotIn("step", env.__dict__)
        self.assertEqual(env.step_calls, 1)
        self.assertEqual(validate_attempt_records(ledger.terminal_records()), ())

    def test_outcome_failure_records_completed_step_and_quarantine(self):
        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snapshot = physical_snapshot([task])
        built = built_for(snapshot)
        env = FakeEnv([task])
        schedulers = SpySchedulers()
        base_observer = phase_observer(snapshot, env)

        def fail_outcome(observed_env, *, phase):
            if phase == SnapshotPhase.OUTCOME:
                raise RuntimeError("outcome capture failed")
            return base_observer(observed_env, phase=phase)

        ledger, attempt = prepared_attempt(built)
        with self.assertRaises(CollectorAttemptRejected) as caught:
            execute(env, built, schedulers, attempt, fail_outcome)

        row = caught.exception.record
        self.assertEqual(row["terminal_stage"], "outcome_capture")
        self.assertTrue(row["env_step_called"])
        self.assertTrue(row["env_step_completed"])
        self.assertEqual(row["environment_mutation_status"], "confirmed")
        self.assertTrue(row["quarantined"])
        self.assertEqual(env.step_calls, 1)
        self.assertEqual(validate_attempt_records(ledger.terminal_records()), ())

    def test_success_records_real_call_order_one_step_and_restores_wrappers(self):
        task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        snapshot = physical_snapshot([task])
        built = built_for(snapshot)
        env = FakeEnv([task])
        schedulers = SpySchedulers()
        ledger, attempt = prepared_attempt(built)
        original_methods = {
            name: getattr(schedulers, name)
            for name in (
                "setComputingCallBack",
                "setTaskOffloading",
                "setTaskReturnRoute",
                "setCommunicationWithRB",
            )
        }

        result = execute(env, built, schedulers, attempt, phase_observer(snapshot, env))
        row = ledger.terminal_records()[0]

        self.assertTrue(result.stepped)
        self.assertEqual(row["disposition"], "accepted")
        self.assertEqual(
            [call["setter_kind"] for call in row["setter_calls"]],
            ["cpu_callback", "offload", "rb"],
        )
        self.assertTrue(all(call["succeeded"] for call in row["setter_calls"]))
        self.assertEqual(env.step_calls, 1)
        self.assertNotIn("step", env.__dict__)
        for name, original in original_methods.items():
            self.assertEqual(getattr(schedulers, name), original)
        self.assertEqual(validate_attempt_records([row]), ())


if __name__ == "__main__":
    unittest.main()
