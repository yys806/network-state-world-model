"""Ledger-observed v2 adapter around the real P2-B v1 collector executor."""

from __future__ import annotations

from typing import Callable

from .action_attempt_ledger_v1 import AttemptHandle
from .airfogsim_full_dual_graph_collector_v1 import (
    FullCollectorStepResult,
    execute_full_collector_step,
)
from .airfogsim_full_dual_graph_frame_builder_v1 import BuiltFrameDecision
from .airfogsim_full_dual_graph_observer_v1 import (
    AirFogSimSnapshot,
    observe_airfogsim_snapshot,
)
from .full_dual_graph_collector_contract_v1 import SnapshotPhase


FULL_COLLECTOR_ADAPTER_VERSION = "PIJWM-AirFogSim-Full-Collector-v2"


class CollectorAttemptRejected(RuntimeError):
    """A terminal rejected attempt whose evidence has already been committed."""

    def __init__(
        self,
        record: dict[str, object],
        *,
        cause: BaseException | None = None,
        result: FullCollectorStepResult | None = None,
    ) -> None:
        self.record = record
        self.cause = cause
        self.result = result
        detail = str(record.get("rejection_detail", "collector attempt rejected"))
        super().__init__(detail)


class _RuntimeEvidence:
    def __init__(self, attempt: AttemptHandle) -> None:
        self.attempt = attempt
        self.pre_setter_revalidated = False
        self.setter_started_count = 0
        self.successful_setter_count = 0
        self.env_step_started = False
        self.env_step_completed = False
        self.outcome_captured = False

    def ensure_pre_setter_revalidated(self) -> None:
        if not self.pre_setter_revalidated:
            self.attempt.pre_setter_revalidated()
            self.pre_setter_revalidated = True

    def mutation_status(self) -> str:
        if self.env_step_started or self.successful_setter_count:
            return "confirmed"
        if self.setter_started_count:
            return "unknown_after_runtime_call"
        return "none"

    def terminal_stage(self) -> str:
        if self.env_step_completed:
            return "post_step_processing" if self.outcome_captured else "outcome_capture"
        if self.env_step_started:
            return "env_step"
        if self.setter_started_count:
            return "setter_application"
        return "pre_setter_revalidation"


class _SchedulerProxy:
    def __init__(self, target: object, evidence: _RuntimeEvidence) -> None:
        self._target = target
        self._evidence = evidence

    def __getattr__(self, name: str):
        return getattr(self._target, name)

    def _invoke(
        self,
        method_name: str,
        setter_kind: str,
        task_id: str | None,
        *args,
        false_is_failure: bool = False,
        **kwargs,
    ):
        self._evidence.ensure_pre_setter_revalidated()
        ordinal = self._evidence.attempt.setter_started(
            setter_kind=setter_kind, task_id=task_id
        )
        self._evidence.setter_started_count += 1
        method = getattr(self._target, method_name)
        try:
            result = method(*args, **kwargs)
        except Exception as exc:
            self._evidence.attempt.setter_finished(
                ordinal,
                call_completed=False,
                succeeded=False,
                error_type=type(exc).__name__,
                error_detail=str(exc) or type(exc).__name__,
            )
            raise
        succeeded = not (false_is_failure and result is not True)
        if succeeded:
            self._evidence.attempt.setter_finished(
                ordinal, call_completed=True, succeeded=True
            )
            self._evidence.successful_setter_count += 1
        else:
            self._evidence.attempt.setter_finished(
                ordinal,
                call_completed=True,
                succeeded=False,
                error_type="SetterRejected",
                error_detail=f"{method_name} returned {result!r}",
            )
        return result

    def setComputingCallBack(self, env, callback):
        return self._invoke(
            "setComputingCallBack",
            "cpu_callback",
            None,
            env,
            callback,
        )

    def setTaskOffloading(
        self,
        env,
        task_node_id,
        task_id,
        target_node_id,
        route=None,
    ):
        return self._invoke(
            "setTaskOffloading",
            "offload",
            str(task_id),
            env,
            task_node_id,
            task_id,
            target_node_id,
            route=route,
            false_is_failure=True,
        )

    def setTaskReturnRoute(self, env, task_id, route):
        return self._invoke(
            "setTaskReturnRoute",
            "return_route",
            str(task_id),
            env,
            task_id,
            route,
        )

    def setCommunicationWithRB(self, env, task_id, rb_nos):
        return self._invoke(
            "setCommunicationWithRB",
            "rb",
            str(task_id),
            env,
            task_id,
            rb_nos,
        )


def _rejection_code(stage: str) -> str:
    return {
        "pre_setter_revalidation": "pre_setter_revalidation_error",
        "setter_application": "setter_application_error",
        "env_step": "env_step_error",
        "outcome_capture": "outcome_capture_error",
        "post_step_processing": "post_step_processing_error",
    }[stage]


def _reject_exception(
    attempt: AttemptHandle,
    evidence: _RuntimeEvidence,
    exc: BaseException,
) -> CollectorAttemptRejected:
    stage = evidence.terminal_stage()
    detail = f"{type(exc).__name__}: {exc}"
    record = attempt.reject(
        terminal_stage=stage,
        rejection_code=_rejection_code(stage),
        rejection_detail=detail,
        environment_mutation_status=evidence.mutation_status(),
    )
    return CollectorAttemptRejected(record, cause=exc)


def execute_full_collector_step_v2(
    env,
    built: BuiltFrameDecision,
    *,
    attempt: AttemptHandle,
    trajectory_id: str,
    task_scheduler,
    communication_scheduler,
    computation_scheduler,
    observer: Callable[..., AirFogSimSnapshot] = observe_airfogsim_snapshot,
) -> FullCollectorStepResult:
    """Execute v1 once while recording real runtime boundaries into ``attempt``."""

    evidence = _RuntimeEvidence(attempt)
    task_proxy = _SchedulerProxy(task_scheduler, evidence)
    communication_proxy = _SchedulerProxy(communication_scheduler, evidence)
    computation_proxy = _SchedulerProxy(computation_scheduler, evidence)

    had_instance_step = hasattr(env, "__dict__") and "step" in env.__dict__
    prior_instance_step = env.__dict__.get("step") if hasattr(env, "__dict__") else None
    real_step = env.step

    def observed_step():
        evidence.ensure_pre_setter_revalidated()
        attempt.setters_applied()
        attempt.env_step_started()
        evidence.env_step_started = True
        result = real_step()
        attempt.env_step_completed()
        evidence.env_step_completed = True
        return result

    def observed_snapshot(observed_env, *, phase):
        snapshot = observer(observed_env, phase=phase)
        if phase == SnapshotPhase.OUTCOME:
            attempt.outcome_captured()
            evidence.outcome_captured = True
        return snapshot

    env.step = observed_step
    try:
        try:
            result = execute_full_collector_step(
                env,
                built,
                trajectory_id=trajectory_id,
                task_scheduler=task_proxy,
                communication_scheduler=communication_proxy,
                computation_scheduler=computation_proxy,
                observer=observed_snapshot,
            )
        except Exception as exc:
            raise _reject_exception(attempt, evidence, exc) from exc
    finally:
        if had_instance_step:
            env.step = prior_instance_step
        else:
            delattr(env, "step")

    if result.quarantined or not result.stepped:
        stage = evidence.terminal_stage()
        detail = result.quarantine_reason or "v1 collector returned an unstepped result"
        record = attempt.reject(
            terminal_stage=stage,
            rejection_code=_rejection_code(stage),
            rejection_detail=detail,
            environment_mutation_status=evidence.mutation_status(),
        )
        raise CollectorAttemptRejected(record, result=result)

    attempt.accept()
    return result
