"""Frozen, machine-readable GPU experiment protocol for PI-JWM R6."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence


R6_GPU_PROTOCOL_VERSION = "PIJWM-R6-online-joint-policy-GPU-protocol-v2"


@dataclass(frozen=True)
class GPUTrainingRun:
    run_id: str
    method_id: str
    state_mode: str
    seed: int
    max_environment_steps: int
    rollout_length: int
    minibatch_size: int
    ppo_epochs: int
    learning_rate: float
    clip_epsilon: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    evaluation_interval: int


@dataclass(frozen=True)
class GPUTrainingProtocol:
    schema_version: str
    methods: tuple[str, ...]
    state_modes: tuple[str, ...]
    seeds: tuple[int, ...]
    max_environment_steps: int
    rollout_length: int
    minibatch_size: int
    ppo_epochs: int
    learning_rate: float
    clip_epsilon: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    evaluation_interval: int
    early_stop_patience: int
    smoke_environment_steps: int
    checkpoint_split: str
    threshold_split: str
    locked_test_accessed: bool
    failure_retention: str
    state_source: str
    online_history_steps: int
    validation_step_limit: int
    atomic_resume_checkpoint: bool

    def formal_runs(self) -> tuple[GPUTrainingRun, ...]:
        rows = []
        for method, mode, seed in product(self.methods, self.state_modes, self.seeds):
            rows.append(
                GPUTrainingRun(
                    run_id=f"{method}__{mode}__seed_{seed}",
                    method_id=method,
                    state_mode=mode,
                    seed=seed,
                    max_environment_steps=self.max_environment_steps,
                    rollout_length=self.rollout_length,
                    minibatch_size=self.minibatch_size,
                    ppo_epochs=self.ppo_epochs,
                    learning_rate=self.learning_rate,
                    clip_epsilon=self.clip_epsilon,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef,
                    max_grad_norm=self.max_grad_norm,
                    evaluation_interval=self.evaluation_interval,
                )
            )
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["formal_runs"] = [asdict(run) for run in self.formal_runs()]
        payload["formal_run_count"] = len(payload["formal_runs"])
        payload["smoke_is_formal"] = False
        payload["checkpoint_tiebreak"] = [
            "validation_service_first_return_desc",
            "on_time_completion_rate_desc",
            "mean_latency_asc",
        ]
        return payload


def build_default_gpu_training_protocol() -> GPUTrainingProtocol:
    return GPUTrainingProtocol(
        schema_version=R6_GPU_PROTOCOL_VERSION,
        methods=("actor_critic", "ppo_clipped"),
        state_modes=("explicit_only", "latent_only", "explicit_latent"),
        seeds=(20260803, 20260804, 20260805),
        max_environment_steps=100000,
        rollout_length=128,
        minibatch_size=32,
        ppo_epochs=4,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        evaluation_interval=10000,
        early_stop_patience=5,
        smoke_environment_steps=2000,
        checkpoint_split="validation",
        threshold_split="calibration",
        locked_test_accessed=False,
        failure_retention="retain_failed_seed_with_logs_and_no_substitution",
        state_source="online_airfogsim_strict_dual_graph",
        online_history_steps=8,
        validation_step_limit=64,
        atomic_resume_checkpoint=True,
    )


@dataclass(frozen=True)
class FormalRunValidationSummary:
    expected_count: int
    complete_count: int
    failed_count: int


def validate_formal_run_records(
    protocol: GPUTrainingProtocol,
    records: Sequence[Mapping[str, Any]],
) -> FormalRunValidationSummary:
    expected = {run.run_id for run in protocol.formal_runs()}
    observed_rows = [dict(row) for row in records]
    observed_ids = [str(row.get("run_id", "")) for row in observed_rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("formal run records contain duplicate run_id")
    unknown = sorted(set(observed_ids).difference(expected))
    missing = sorted(expected.difference(observed_ids))
    if unknown:
        raise ValueError(f"unknown formal run IDs: {unknown}")
    if missing:
        raise ValueError(f"missing formal run IDs: {missing}")
    if any(row.get("formal") is not True for row in observed_rows):
        raise ValueError("smoke or nonformal record cannot enter the formal GPU matrix")
    statuses = [str(row.get("status", "")) for row in observed_rows]
    unsupported = sorted(set(statuses).difference({"complete", "failed"}))
    if unsupported:
        raise ValueError(f"unsupported formal run statuses: {unsupported}")
    return FormalRunValidationSummary(
        expected_count=len(expected),
        complete_count=statuses.count("complete"),
        failed_count=statuses.count("failed"),
    )


@dataclass(frozen=True)
class CheckpointMetric:
    environment_step: int
    validation_return: float
    on_time_completion_rate: float
    mean_latency: float
    hard_violation_count: int

    def __post_init__(self) -> None:
        if int(self.environment_step) <= 0:
            raise ValueError("checkpoint environment_step must be positive")
        if any(
            not math.isfinite(float(value))
            for value in (
                self.validation_return,
                self.on_time_completion_rate,
                self.mean_latency,
            )
        ):
            raise ValueError("checkpoint metrics must be finite")
        if not 0.0 <= float(self.on_time_completion_rate) <= 1.0:
            raise ValueError("on_time_completion_rate must lie in [0,1]")
        if float(self.mean_latency) < 0.0 or int(self.hard_violation_count) < 0:
            raise ValueError("latency and hard violations must be nonnegative")


@dataclass(frozen=True)
class CheckpointGateUpdate:
    eligible: bool
    improved: bool
    should_stop: bool
    no_improvement_count: int
    best_environment_step: int | None


class ValidationCheckpointGate:
    """Validation-only checkpoint gate with frozen deterministic tie breaking."""

    def __init__(self, *, patience: int) -> None:
        if int(patience) <= 0:
            raise ValueError("early-stop patience must be positive")
        self.patience = int(patience)
        self.best: CheckpointMetric | None = None
        self.no_improvement_count = 0
        self.last_environment_step = 0

    @staticmethod
    def _key(metric: CheckpointMetric) -> tuple[float, float, float]:
        return (
            float(metric.validation_return),
            float(metric.on_time_completion_rate),
            -float(metric.mean_latency),
        )

    def update(self, metric: CheckpointMetric) -> CheckpointGateUpdate:
        if metric.environment_step <= self.last_environment_step:
            raise ValueError("checkpoint environment steps must be strictly increasing")
        self.last_environment_step = metric.environment_step
        eligible = metric.hard_violation_count == 0
        improved = bool(
            eligible and (self.best is None or self._key(metric) > self._key(self.best))
        )
        if improved:
            self.best = metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        return CheckpointGateUpdate(
            eligible=eligible,
            improved=improved,
            should_stop=self.no_improvement_count >= self.patience,
            no_improvement_count=self.no_improvement_count,
            best_environment_step=None if self.best is None else self.best.environment_step,
        )
