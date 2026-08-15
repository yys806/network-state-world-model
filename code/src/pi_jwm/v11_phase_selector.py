"""Phase-conditioned benefit statistics for the PI-JWM v11 selector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .v11_selector import (
    CandidateBatch,
    CandidateOutcome,
    _pareto_dominated,
    aggregate_selected_metrics,
    observable_pareto_deltas,
)


@dataclass(frozen=True)
class PhaseCandidateStatistics:
    mean_benefit: np.ndarray
    benefit_std: np.ndarray
    positive_rate: np.ndarray
    count: np.ndarray
    default_index: int
    episode_length: int
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class PhaseSelectorConfig:
    z_value: float
    positive_rate_threshold: float
    minimum_mean_benefit: float
    min_count: int = 5


@dataclass(frozen=True)
class CalibratedPhaseSelector:
    status: str
    config: PhaseSelectorConfig
    calibration_rmse: float
    improvement_vs_default: float
    executed_count: int
    positive_precision: float
    negative_selection_rate: float


def _phase(sample_ids: np.ndarray, sample_count: int, episode_length: int) -> np.ndarray:
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if ids.shape != (sample_count,) or np.any(ids < 0):
        raise ValueError("sample IDs must be non-negative and match the sample count")
    length = int(episode_length)
    if length < 2:
        raise ValueError("episode length must be at least two")
    return np.mod(ids, length).astype(np.int64)


def fit_phase_candidate_statistics(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    sample_ids: np.ndarray,
    episode_length: int = 390,
) -> PhaseCandidateStatistics:
    """Fit phase-candidate benefit distributions from train outcomes only."""
    sample_count, candidate_count = batch.candidate_mask.shape
    if outcome.active_sse.shape != (sample_count, candidate_count):
        raise ValueError("candidate batch and outcome dimensions must match")
    phase = _phase(sample_ids, sample_count, episode_length)
    legal = batch.candidate_mask.copy()
    if outcome.action_applicable is not None:
        legal &= outcome.action_applicable
    if outcome.action_applied is not None:
        legal &= outcome.action_applied
    legal[:, outcome.default_index] = batch.candidate_mask[:, outcome.default_index]
    benefit = outcome.active_sse[:, outcome.default_index, None] - outcome.active_sse
    mean = np.full((int(episode_length), candidate_count), -np.inf, dtype=np.float64)
    std = np.full_like(mean, np.inf)
    positive = np.zeros_like(mean)
    count = np.zeros(mean.shape, dtype=np.int64)
    valid = outcome.active_count > 0
    for phase_index in range(int(episode_length)):
        phase_rows = phase == phase_index
        for candidate in range(candidate_count):
            keep = phase_rows & valid & legal[:, candidate]
            values = benefit[keep, candidate].astype(np.float64)
            if values.size:
                mean[phase_index, candidate] = float(np.mean(values))
                std[phase_index, candidate] = float(np.std(values, ddof=0))
                positive[phase_index, candidate] = float(np.mean(values > 0.0))
                count[phase_index, candidate] = int(values.size)
    default = int(outcome.default_index)
    mean[:, default] = 0.0
    std[:, default] = 0.0
    positive[:, default] = 1.0
    return PhaseCandidateStatistics(
        mean_benefit=mean,
        benefit_std=std,
        positive_rate=positive,
        count=count,
        default_index=default,
        episode_length=int(episode_length),
        candidate_names=batch.candidate_names,
    )


def build_observable_pareto_allowed(
    batch: CandidateBatch,
    default_index: int,
) -> np.ndarray:
    """Build the deployable task-energy proxy non-dominance mask."""
    task, energy = observable_pareto_deltas(batch, int(default_index))
    allowed = np.ones(batch.candidate_mask.shape, dtype=bool)
    for sample in range(allowed.shape[0]):
        allowed[sample] = ~_pareto_dominated(task[sample], energy[sample])
    allowed[:, int(default_index)] = True
    return allowed


def select_phase_candidates(
    statistics: PhaseCandidateStatistics,
    batch: CandidateBatch,
    sample_ids: np.ndarray,
    config: PhaseSelectorConfig,
    pareto_allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Choose the phase-LCB candidate using only deployable masks and features."""
    sample_count, candidate_count = batch.candidate_mask.shape
    if statistics.mean_benefit.shape[1] != candidate_count:
        raise ValueError("phase statistics and candidate batch dimensions must match")
    if batch.candidate_names and statistics.candidate_names:
        if tuple(batch.candidate_names) != tuple(statistics.candidate_names):
            raise ValueError("phase statistics candidate order mismatch")
    phase = _phase(sample_ids, sample_count, statistics.episode_length)
    allowed = batch.candidate_mask.copy()
    if pareto_allowed is not None:
        pareto = np.asarray(pareto_allowed, dtype=bool)
        if pareto.shape != allowed.shape:
            raise ValueError("Pareto mask must match candidate dimensions")
        allowed &= pareto
    default = int(statistics.default_index)
    allowed[:, default] = batch.candidate_mask[:, default]
    means = statistics.mean_benefit[phase]
    std = statistics.benefit_std[phase]
    count = statistics.count[phase]
    standard_error = np.full_like(std, np.inf, dtype=np.float64)
    finite = np.isfinite(std)
    standard_error[finite] = std[finite] / np.sqrt(np.maximum(count[finite], 1))
    z_value = float(config.z_value)
    score = means.copy() if z_value == 0.0 else means - z_value * standard_error
    allowed &= statistics.positive_rate[phase] >= float(config.positive_rate_threshold)
    allowed &= means >= float(config.minimum_mean_benefit)
    allowed &= count >= int(config.min_count)
    allowed[:, default] = batch.candidate_mask[:, default]
    return np.argmax(np.where(allowed, score, -np.inf), axis=1).astype(np.int64)


def _calibration_metrics(
    outcome: CandidateOutcome,
    choice: np.ndarray,
) -> tuple[float, float, int, float, float]:
    selected = aggregate_selected_metrics(outcome, choice)
    default_choice = np.full(choice.shape, outcome.default_index, dtype=np.int64)
    baseline = aggregate_selected_metrics(outcome, default_choice)
    rows = np.arange(choice.shape[0])
    benefit = outcome.active_sse[:, outcome.default_index] - outcome.active_sse[rows, choice]
    executed = (outcome.active_count > 0) & (choice != outcome.default_index)
    executed_count = int(np.sum(executed))
    precision = float(np.mean(benefit[executed] > 0.0)) if executed_count else 0.0
    negative = float(np.mean(benefit[executed] < 0.0)) if executed_count else 0.0
    rmse = float(selected["active_rate_rmse"])
    improvement = float(baseline["active_rate_rmse"] - rmse)
    return rmse, improvement, executed_count, precision, negative


def calibrate_phase_selector(
    statistics: PhaseCandidateStatistics,
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    sample_ids: np.ndarray,
    pareto_allowed: np.ndarray,
    z_values: tuple[float, ...],
    positive_rate_values: tuple[float, ...],
    minimum_mean_values: tuple[float, ...],
    min_count: int = 5,
    minimum_positive_precision: float = 0.65,
    maximum_negative_selection_rate: float = 0.20,
) -> CalibratedPhaseSelector:
    """Choose one defer configuration on calibration outcomes only."""
    safe: list[CalibratedPhaseSelector] = []
    for z_value in z_values:
        for positive_rate in positive_rate_values:
            for minimum_mean in minimum_mean_values:
                config = PhaseSelectorConfig(
                    float(z_value), float(positive_rate), float(minimum_mean), int(min_count)
                )
                choice = select_phase_candidates(
                    statistics, batch, sample_ids, config, pareto_allowed=pareto_allowed
                )
                rmse, improvement, executed, precision, negative = _calibration_metrics(
                    outcome, choice
                )
                if (
                    executed > 0
                    and precision >= float(minimum_positive_precision)
                    and negative <= float(maximum_negative_selection_rate)
                ):
                    safe.append(
                        CalibratedPhaseSelector(
                            status="safe_threshold",
                            config=config,
                            calibration_rmse=rmse,
                            improvement_vs_default=improvement,
                            executed_count=executed,
                            positive_precision=precision,
                            negative_selection_rate=negative,
                        )
                    )
    if safe:
        return min(
            safe,
            key=lambda result: (
                result.calibration_rmse,
                -result.executed_count,
                result.config.z_value,
                result.config.positive_rate_threshold,
                result.config.minimum_mean_benefit,
            ),
        )
    default = PhaseSelectorConfig(0.0, 1.1, np.inf, int(min_count))
    default_choice = np.full(batch.context.shape[0], statistics.default_index, dtype=np.int64)
    rmse, improvement, executed, precision, negative = _calibration_metrics(
        outcome, default_choice
    )
    return CalibratedPhaseSelector(
        status="no_safe_threshold",
        config=default,
        calibration_rmse=rmse,
        improvement_vs_default=improvement,
        executed_count=executed,
        positive_precision=precision,
        negative_selection_rate=negative,
    )
