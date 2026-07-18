"""Train and validation-select the PI-JWM objective-aligned v11 selector."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.evaluation.candidate_selection import choice_rmse_from_sample_sse
from pi_jwm.v11_labeling import load_candidate_label_cache
from pi_jwm.v11_objective_aligned_selector import (
    ObjectiveAlignedDecision,
    OpportunityCalibration,
    build_decision_aligned_targets,
    calibrate_opportunity_threshold,
    fit_objective_aligned_selector,
    predict_objective_aligned_selector,
    save_objective_aligned_checkpoint,
    select_objective_aligned,
)
from pi_jwm.v11_selector import (
    CandidateBatch,
    CandidateOutcome,
    ablate_candidate_batch,
    audit_candidate_library,
    canonical_sha256,
    file_sha256,
    observable_pareto_deltas,
)
from run_v11_selector_candidate_labels import limit_indices_seed_balanced
from train_v11_candidate_set_selector import (
    _choice_metrics,
    _metadata,
    _write_csv,
    choose_best_validation_config,
    masked_oracle_choice,
    validate_cache_protocol,
)


DEFAULT_OUTPUT_DIR = (
    CODE_ROOT
    / "artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/selector_training"
)
PARTIAL_IMPROVEMENT_RMSE = 230.85555814182524


@dataclass(frozen=True)
class FlattenedCandidateBenefitRows:
    features: np.ndarray
    target_benefit: np.ndarray
    sample_weight: np.ndarray
    sample_index: np.ndarray
    candidate_index: np.ndarray


def _stage_one_hot(stage: np.ndarray) -> np.ndarray:
    vocabulary = {"unknown": 0, "offload": 1, "compute": 2, "return": 3}
    indices = np.asarray(
        [vocabulary.get(str(value).lower(), 0) for value in stage], dtype=np.int64
    )
    return np.eye(4, dtype=np.float32)[indices]


def flatten_candidate_benefit_rows(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    weight_cap: float,
) -> FlattenedCandidateBenefitRows:
    """Flatten only legal candidates for the objective-aligned tree baseline."""

    targets = build_decision_aligned_targets(
        outcome, batch.candidate_mask, weight_cap=float(weight_cap)
    )
    sample_index, candidate_index = np.nonzero(batch.candidate_mask)
    stage = _stage_one_hot(batch.stage)[sample_index]
    features = np.concatenate(
        [
            batch.candidate_features[sample_index, candidate_index],
            batch.context[sample_index],
            stage,
        ],
        axis=1,
    ).astype(np.float32)
    return FlattenedCandidateBenefitRows(
        features=features,
        target_benefit=targets.candidate_benefit[
            sample_index, candidate_index
        ].astype(np.float32),
        sample_weight=targets.sample_weight[sample_index].astype(np.float32),
        sample_index=sample_index.astype(np.int64),
        candidate_index=candidate_index.astype(np.int64),
    )


def scatter_candidate_predictions(
    rows: FlattenedCandidateBenefitRows,
    predictions: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if values.shape[0] != rows.sample_index.shape[0]:
        raise ValueError("one prediction is required for every flattened candidate")
    result = np.full(shape, -np.inf, dtype=np.float32)
    result[rows.sample_index, rows.candidate_index] = values
    return result


def objective_ablation_specs() -> tuple[str, ...]:
    return (
        "without_opportunity",
        "without_uncertainty",
        "uniform_impact",
        "without_stage",
        "without_task",
        "without_resource",
        "without_energy",
    )


def validate_objective_cache_protocol(
    manifests: Mapping[str, Mapping[str, Any]],
) -> str:
    """Require the frozen schema-v5 train/calibration/validation contract."""

    try:
        return validate_cache_protocol(manifests, required_schema_version=5)
    except ValueError as error:
        if "schema" in str(error).lower():
            raise ValueError(f"objective-aligned selector requires schema 5: {error}") from error
        raise


def build_grid(
    hidden_dimensions: Sequence[int],
    weight_caps: Sequence[float],
) -> list[tuple[int, float]]:
    grid = [
        (int(hidden), float(weight_cap))
        for hidden in hidden_dimensions
        for weight_cap in weight_caps
    ]
    if not grid or any(hidden < 1 or cap <= 0.0 for hidden, cap in grid):
        raise ValueError("selector grid requires positive hidden dimensions and weight caps")
    if len(set(grid)) != len(grid):
        raise ValueError("selector grid contains duplicate configurations")
    return grid


def classify_validation_result(metrics: Mapping[str, Any]) -> str:
    """Apply the frozen success, partial-improvement, and failure gates."""

    required = {
        "rmse",
        "training_seed_std",
        "improved_seed_count",
        "activity_f1_drop",
        "link_rmse_relative_degradation",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"validation classification missing metrics: {missing}")
    success = bool(
        float(metrics["rmse"]) < 200.0
        and float(metrics["training_seed_std"]) <= 5.0
        and int(metrics["improved_seed_count"]) >= 7
        and float(metrics["activity_f1_drop"]) <= 0.002
        and float(metrics["link_rmse_relative_degradation"]) <= 0.02
    )
    if success:
        return "success"
    if float(metrics["rmse"]) < PARTIAL_IMPROVEMENT_RMSE:
        return "partial_improvement"
    return "failure"


def _subset_batch_outcome(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    indices: np.ndarray,
) -> tuple[CandidateBatch, CandidateOutcome]:
    selected = np.asarray(indices, dtype=np.int64)
    subset_batch = CandidateBatch(
        context=batch.context[selected],
        candidate_features=batch.candidate_features[selected],
        candidate_mask=batch.candidate_mask[selected],
        stage=batch.stage[selected],
        feature_names=batch.feature_names,
        candidate_names=batch.candidate_names,
        context_feature_names=batch.context_feature_names,
    )

    def optional(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else values[selected]

    subset_outcome = CandidateOutcome(
        active_sse=outcome.active_sse[selected],
        active_count=outcome.active_count[selected],
        link_sse=optional(outcome.link_sse),
        link_count=optional(outcome.link_count),
        activity_tp=optional(outcome.activity_tp),
        activity_fp=optional(outcome.activity_fp),
        activity_fn=optional(outcome.activity_fn),
        activity_tn=optional(outcome.activity_tn),
        action_applied=optional(outcome.action_applied),
        action_applicable=optional(outcome.action_applicable),
        default_index=outcome.default_index,
        task_utility=optional(outcome.task_utility),
        energy_total=optional(outcome.energy_total),
        result_kind=outcome.result_kind,
    )
    return subset_batch, subset_outcome


def _stack_predictions(fitted_models, batch: CandidateBatch) -> dict[str, np.ndarray]:
    predictions = [
        predict_objective_aligned_selector(fitted, batch) for fitted in fitted_models
    ]
    return {
        name: np.stack([prediction[name] for prediction in predictions], axis=0)
        for name in predictions[0]
    }


def _calibrate_and_select(
    fitted_models,
    calibration_batch: CandidateBatch,
    calibration_outcome: CandidateOutcome,
    validation_batch: CandidateBatch,
    validation_outcome: CandidateOutcome,
) -> tuple[
    OpportunityCalibration,
    ObjectiveAlignedDecision,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    calibration_predictions = _stack_predictions(fitted_models, calibration_batch)
    calibration_task, calibration_energy = observable_pareto_deltas(
        calibration_batch, calibration_outcome.default_index
    )
    unthresholded = select_objective_aligned(
        calibration_predictions["predicted_candidate_benefit"],
        calibration_predictions["candidate_uncertainty"],
        calibration_predictions["predicted_opportunity"],
        calibration_predictions["opportunity_uncertainty"],
        calibration_batch.candidate_mask,
        calibration_outcome.default_index,
        opportunity_threshold=-1e30,
        task_delta=calibration_task,
        energy_delta=calibration_energy,
    )
    calibration = calibrate_opportunity_threshold(
        unthresholded.opportunity_lcb,
        unthresholded.candidate_index,
        calibration_outcome,
    )
    validation_predictions = _stack_predictions(fitted_models, validation_batch)
    validation_task, validation_energy = observable_pareto_deltas(
        validation_batch, validation_outcome.default_index
    )
    decision = select_objective_aligned(
        validation_predictions["predicted_candidate_benefit"],
        validation_predictions["candidate_uncertainty"],
        validation_predictions["predicted_opportunity"],
        validation_predictions["opportunity_uncertainty"],
        validation_batch.candidate_mask,
        validation_outcome.default_index,
        opportunity_threshold=calibration.threshold,
        task_delta=validation_task,
        energy_delta=validation_energy,
    )
    return calibration, decision, calibration_predictions, validation_predictions


def _improved_seed_count(
    outcome: CandidateOutcome,
    choice: np.ndarray,
    sample_seed: np.ndarray,
) -> int:
    improved = 0
    for seed in np.unique(sample_seed):
        keep = sample_seed == seed
        selected_rmse = choice_rmse_from_sample_sse(
            outcome.active_sse[keep], outcome.active_count[keep], choice[keep]
        )
        default_choice = np.full(np.sum(keep), outcome.default_index, dtype=np.int64)
        default_rmse = choice_rmse_from_sample_sse(
            outcome.active_sse[keep], outcome.active_count[keep], default_choice
        )
        if selected_rmse is not None and default_rmse is not None and selected_rmse < default_rmse:
            improved += 1
    return int(improved)


def _classification_metrics(
    metrics: Mapping[str, Any],
    training_seed_std: float,
    improved_seed_count: int,
) -> dict[str, float | int]:
    default_f1 = metrics.get("default_activity_f1")
    selected_f1 = metrics.get("activity_f1")
    activity_drop = (
        0.0
        if default_f1 is None or selected_f1 is None
        else max(0.0, float(default_f1) - float(selected_f1))
    )
    default_link = metrics.get("default_link_rmse")
    selected_link = metrics.get("link_rmse")
    link_degradation = (
        0.0
        if default_link in (None, 0.0) or selected_link is None
        else max(0.0, (float(selected_link) - float(default_link)) / float(default_link))
    )
    return {
        "rmse": float(metrics["rmse"]),
        "training_seed_std": float(training_seed_std),
        "improved_seed_count": int(improved_seed_count),
        "activity_f1_drop": float(activity_drop),
        "link_rmse_relative_degradation": float(link_degradation),
    }


def _baseline_rows(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    sample_seed: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    default_choice = np.full(
        outcome.active_sse.shape[0], outcome.default_index, dtype=np.int64
    )
    oracle_choice = masked_oracle_choice(outcome.active_sse, batch.candidate_mask)
    for model, choice, kind in (
        ("ranked_allocation_default", default_choice, "deployable"),
        ("sample_oracle", oracle_choice, "sample_oracle"),
    ):
        metrics = _choice_metrics(outcome, choice, sample_seed, batch.candidate_mask)
        rows.append(
            {
                "model": model,
                "result_kind": kind,
                "validation_rmse": metrics["rmse"],
                "validation_link_rmse": metrics["link_rmse"],
                "validation_activity_f1": metrics["activity_f1"],
                "defer_ratio": 1.0 if model == "ranked_allocation_default" else 0.0,
            }
        )
    fixed_rows = []
    for candidate, name in enumerate(batch.candidate_names):
        choice = np.full(outcome.active_sse.shape[0], candidate, dtype=np.int64)
        choice[~batch.candidate_mask[:, candidate]] = outcome.default_index
        metrics = _choice_metrics(outcome, choice, sample_seed, batch.candidate_mask)
        fixed_rows.append((float(metrics["rmse"]), name, metrics))
    _, best_name, best_metrics = min(fixed_rows, key=lambda value: (value[0], value[1]))
    rows.append(
        {
            "model": f"best_fixed__{best_name}",
            "result_kind": "diagnostic_only",
            "validation_rmse": best_metrics["rmse"],
            "validation_link_rmse": best_metrics["link_rmse"],
            "validation_activity_f1": best_metrics["activity_f1"],
            "defer_ratio": 0.0,
        }
    )
    return rows


def _gain_concentration_rows(
    split_name: str,
    batch: CandidateBatch,
    outcome: CandidateOutcome,
) -> list[dict[str, Any]]:
    targets = build_decision_aligned_targets(outcome, batch.candidate_mask)
    positive = np.sort(targets.opportunity[targets.valid_sample])[::-1]
    total = float(np.sum(positive))
    rows = []
    for fraction in (0.01, 0.05, 0.10, 0.20, 1.0):
        count = max(1, int(np.ceil(positive.shape[0] * fraction)))
        rows.append(
            {
                "split": split_name,
                "top_fraction": fraction,
                "sample_count": count,
                "gain_share": 0.0 if total <= 0.0 else float(np.sum(positive[:count]) / total),
            }
        )
    return rows


def _select_from_benefit_predictions(
    calibration_benefit: np.ndarray,
    validation_benefit: np.ndarray,
    calibration_batch: CandidateBatch,
    calibration_outcome: CandidateOutcome,
    validation_batch: CandidateBatch,
    validation_outcome: CandidateOutcome,
) -> tuple[OpportunityCalibration, ObjectiveAlignedDecision]:
    calibration_values = np.asarray(calibration_benefit, dtype=np.float32).copy()
    validation_values = np.asarray(validation_benefit, dtype=np.float32).copy()
    if calibration_values.shape != calibration_batch.candidate_mask.shape:
        raise ValueError("calibration tree predictions must match candidate mask")
    if validation_values.shape != validation_batch.candidate_mask.shape:
        raise ValueError("validation tree predictions must match candidate mask")
    if not np.all(np.isfinite(calibration_values[calibration_batch.candidate_mask])):
        raise ValueError("legal calibration tree predictions must be finite")
    if not np.all(np.isfinite(validation_values[validation_batch.candidate_mask])):
        raise ValueError("legal validation tree predictions must be finite")
    calibration_values = np.where(
        calibration_batch.candidate_mask, calibration_values, 0.0
    )
    validation_values = np.where(
        validation_batch.candidate_mask, validation_values, 0.0
    )
    calibration_values[:, calibration_outcome.default_index] = 0.0
    validation_values[:, validation_outcome.default_index] = 0.0
    calibration_opportunity = np.maximum(
        0.0, np.max(calibration_values, axis=1)
    ).astype(np.float32)
    validation_opportunity = np.maximum(
        0.0, np.max(validation_values, axis=1)
    ).astype(np.float32)
    calibration_task, calibration_energy = observable_pareto_deltas(
        calibration_batch, calibration_outcome.default_index
    )
    calibration_decision = select_objective_aligned(
        calibration_values[None, ...],
        np.zeros_like(calibration_values)[None, ...],
        calibration_opportunity[None, ...],
        np.zeros_like(calibration_opportunity)[None, ...],
        calibration_batch.candidate_mask,
        calibration_outcome.default_index,
        opportunity_threshold=-1e30,
        task_delta=calibration_task,
        energy_delta=calibration_energy,
        z_value=0.0,
    )
    calibration = calibrate_opportunity_threshold(
        calibration_decision.opportunity_lcb,
        calibration_decision.candidate_index,
        calibration_outcome,
    )
    validation_task, validation_energy = observable_pareto_deltas(
        validation_batch, validation_outcome.default_index
    )
    decision = select_objective_aligned(
        validation_values[None, ...],
        np.zeros_like(validation_values)[None, ...],
        validation_opportunity[None, ...],
        np.zeros_like(validation_opportunity)[None, ...],
        validation_batch.candidate_mask,
        validation_outcome.default_index,
        opportunity_threshold=calibration.threshold,
        task_delta=validation_task,
        energy_delta=validation_energy,
        z_value=0.0,
    )
    return calibration, decision


def _run_xgboost_baseline(
    batches: Mapping[str, CandidateBatch],
    outcomes: Mapping[str, CandidateOutcome],
    validation_seed: np.ndarray,
    weight_cap: float,
    output_dir: Path,
    estimators: int,
) -> tuple[str, dict[str, Any]]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return "skipped_dependency_unavailable", {
            "model": "xgboost_candidate_benefit",
            "result_kind": "diagnostic_only",
            "status": "skipped_dependency_unavailable",
        }

    rows = {
        name: flatten_candidate_benefit_rows(
            batches[name], outcomes[name], weight_cap=float(weight_cap)
        )
        for name in ("train", "calibration", "validation")
    }
    train_targets = build_decision_aligned_targets(
        outcomes["train"], batches["train"].candidate_mask, weight_cap=float(weight_cap)
    )
    scale = float(train_targets.benefit_scale)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(estimators),
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=17,
        n_jobs=-1,
    )
    model.fit(
        rows["train"].features,
        rows["train"].target_benefit / scale,
        sample_weight=rows["train"].sample_weight,
        verbose=False,
    )
    predicted = {}
    for name in ("calibration", "validation"):
        row = rows[name]
        predicted[name] = scatter_candidate_predictions(
            row,
            model.predict(row.features).astype(np.float32) * scale,
            shape=outcomes[name].active_sse.shape,
        )
    calibration, decision = _select_from_benefit_predictions(
        predicted["calibration"],
        predicted["validation"],
        batches["calibration"],
        outcomes["calibration"],
        batches["validation"],
        outcomes["validation"],
    )
    metrics = _choice_metrics(
        outcomes["validation"],
        decision.candidate_index,
        validation_seed,
        batches["validation"].candidate_mask,
    )
    model_path = output_dir / "xgboost_candidate_benefit.json"
    model.save_model(model_path)
    row = {
        "model": "xgboost_candidate_benefit",
        "result_kind": "diagnostic_only",
        "status": "completed",
        "validation_rmse": metrics["rmse"],
        "validation_link_rmse": metrics["link_rmse"],
        "validation_activity_f1": metrics["activity_f1"],
        "improvement_vs_default": metrics["improvement_vs_default"],
        "defer_ratio": float(np.mean(decision.deferred)),
        "opportunity_quantile": calibration.quantile,
        "opportunity_threshold": calibration.threshold,
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
    }
    return "completed", row


def _ablation_row(
    name: str,
    decision: ObjectiveAlignedDecision,
    validation_batch: CandidateBatch,
    validation_outcome: CandidateOutcome,
    validation_seed: np.ndarray,
) -> dict[str, Any]:
    metrics = _choice_metrics(
        validation_outcome,
        decision.candidate_index,
        validation_seed,
        validation_batch.candidate_mask,
    )
    return {
        "ablation": name,
        "result_kind": "diagnostic_only",
        "validation_rmse": metrics["rmse"],
        "validation_link_rmse": metrics["link_rmse"],
        "validation_activity_f1": metrics["activity_f1"],
        "improvement_vs_default": metrics["improvement_vs_default"],
        "worst_seed_regret": metrics["worst_seed_regret"],
        "defer_ratio": float(np.mean(decision.deferred)),
    }


def _run_objective_ablations(
    batches: Mapping[str, CandidateBatch],
    outcomes: Mapping[str, CandidateOutcome],
    train_seed: np.ndarray,
    validation_seed: np.ndarray,
    selected_config: Mapping[str, Any],
    selected_calibration: OpportunityCalibration,
    selected_validation_predictions: Mapping[str, np.ndarray],
    epochs: int,
    learning_rate: float,
    device: str,
) -> list[dict[str, Any]]:
    validation_batch = batches["validation"]
    validation_outcome = outcomes["validation"]
    validation_task, validation_energy = observable_pareto_deltas(
        validation_batch, validation_outcome.default_index
    )
    rows: list[dict[str, Any]] = []
    decision_without_opportunity = select_objective_aligned(
        selected_validation_predictions["predicted_candidate_benefit"],
        selected_validation_predictions["candidate_uncertainty"],
        selected_validation_predictions["predicted_opportunity"],
        selected_validation_predictions["opportunity_uncertainty"],
        validation_batch.candidate_mask,
        validation_outcome.default_index,
        opportunity_threshold=-1e30,
        task_delta=validation_task,
        energy_delta=validation_energy,
    )
    rows.append(
        _ablation_row(
            "without_opportunity",
            decision_without_opportunity,
            validation_batch,
            validation_outcome,
            validation_seed,
        )
    )
    decision_without_uncertainty = select_objective_aligned(
        selected_validation_predictions["predicted_candidate_benefit"],
        np.zeros_like(selected_validation_predictions["candidate_uncertainty"]),
        selected_validation_predictions["predicted_opportunity"],
        np.zeros_like(selected_validation_predictions["opportunity_uncertainty"]),
        validation_batch.candidate_mask,
        validation_outcome.default_index,
        opportunity_threshold=selected_calibration.threshold,
        task_delta=validation_task,
        energy_delta=validation_energy,
    )
    rows.append(
        _ablation_row(
            "without_uncertainty",
            decision_without_uncertainty,
            validation_batch,
            validation_outcome,
            validation_seed,
        )
    )

    retrain_epochs = min(100, int(epochs))
    for name in objective_ablation_specs()[2:]:
        if name == "uniform_impact":
            ablated_batches = dict(batches)
            impact_weighting = False
        else:
            group = name.removeprefix("without_")
            ablated_batches = {
                split: ablate_candidate_batch(batch, group)
                for split, batch in batches.items()
            }
            impact_weighting = True
        fitted = fit_objective_aligned_selector(
            ablated_batches["train"],
            outcomes["train"],
            hidden_dim=int(selected_config["hidden_dim"]),
            weight_cap=float(selected_config["weight_cap"]),
            epochs=retrain_epochs,
            learning_rate=float(learning_rate),
            seed=17,
            device=device,
            group_ids=train_seed,
            impact_weighting=impact_weighting,
        )
        _, decision, _, _ = _calibrate_and_select(
            [fitted],
            ablated_batches["calibration"],
            outcomes["calibration"],
            ablated_batches["validation"],
            outcomes["validation"],
        )
        row = _ablation_row(
            name,
            decision,
            ablated_batches["validation"],
            outcomes["validation"],
            validation_seed,
        )
        row.update(
            {
                "training_seed": 17,
                "epochs": retrain_epochs,
                "impact_weighting": impact_weighting,
            }
        )
        rows.append(row)
    if tuple(row["ablation"] for row in rows) != objective_ablation_specs():
        raise RuntimeError("objective ablation output does not match the frozen protocol")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--weight-cap", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--sample-limit-per-split", type=int, default=0)
    parser.add_argument("--allow-smoke-gate-failure", action="store_true")
    parser.add_argument("--run-xgboost", action="store_true")
    parser.add_argument("--xgboost-estimators", type=int, default=300)
    parser.add_argument("--run-ablations", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "train": args.train_cache.resolve(),
        "calibration": args.calibration_cache.resolve(),
        "validation": args.validation_cache.resolve(),
    }
    loaded = {name: load_candidate_label_cache(path) for name, path in cache_paths.items()}
    manifests = {name: values[2] for name, values in loaded.items()}
    configuration_digest = validate_objective_cache_protocol(manifests)
    batches = {name: values[0] for name, values in loaded.items()}
    outcomes = {name: values[1] for name, values in loaded.items()}
    metadata = {name: _metadata(path) for name, path in cache_paths.items()}
    if int(args.sample_limit_per_split) > 0:
        for name in ("train", "calibration", "validation"):
            all_indices = np.arange(metadata[name]["sample_seed"].shape[0])
            selected = limit_indices_seed_balanced(
                all_indices,
                metadata[name]["sample_seed"],
                int(args.sample_limit_per_split),
            )
            batches[name], outcomes[name] = _subset_batch_outcome(
                batches[name], outcomes[name], selected
            )
            metadata[name] = {
                key: values[selected] for key, values in metadata[name].items()
            }

    validation_gate = audit_candidate_library(
        outcomes["validation"].active_sse,
        outcomes["validation"].active_count,
        action_applied=(
            outcomes["validation"].action_applied
            if outcomes["validation"].action_applied is not None
            else batches["validation"].candidate_mask
        ),
        candidate_mask=batches["validation"].candidate_mask,
        applicability_mask=(
            outcomes["validation"].action_applicable
            if outcomes["validation"].action_applicable is not None
            else batches["validation"].candidate_mask
        ),
        identity_index=0,
    )
    if not validation_gate["passed"] and not bool(args.allow_smoke_gate_failure):
        raise RuntimeError(f"candidate library gate failed: {validation_gate}")

    grid_rows = []
    comparison_rows = _baseline_rows(
        batches["validation"],
        outcomes["validation"],
        metadata["validation"]["sample_seed"],
    )
    calibration_rows = []
    fitted_by_config = {}
    checkpoints_by_config = {}
    results_by_config = {}
    for hidden_dim, weight_cap in build_grid(args.hidden_dim, args.weight_cap):
        config_id = f"h{hidden_dim}_w{weight_cap:g}"
        fitted_models = []
        checkpoint_records = []
        per_training_seed_rmse = []
        for training_seed in args.training_seeds:
            fitted = fit_objective_aligned_selector(
                batches["train"],
                outcomes["train"],
                hidden_dim=hidden_dim,
                weight_cap=weight_cap,
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                seed=int(training_seed),
                device=args.device,
                group_ids=metadata["train"]["sample_seed"],
            )
            checkpoint = args.output_dir / f"objective_selector_{config_id}_seed{training_seed}.pt"
            save_objective_aligned_checkpoint(
                checkpoint,
                fitted,
                configuration_digest=configuration_digest,
                training_seed=int(training_seed),
            )
            checkpoint_records.append(
                {
                    "file": checkpoint.name,
                    "sha256": file_sha256(checkpoint),
                    "training_seed": int(training_seed),
                    "benefit_scale": float(fitted.benefit_scale),
                }
            )
            single_calibration, single_decision, _, _ = _calibrate_and_select(
                [fitted],
                batches["calibration"],
                outcomes["calibration"],
                batches["validation"],
                outcomes["validation"],
            )
            single_metrics = _choice_metrics(
                outcomes["validation"],
                single_decision.candidate_index,
                metadata["validation"]["sample_seed"],
                batches["validation"].candidate_mask,
            )
            per_training_seed_rmse.append(float(single_metrics["rmse"]))
            calibration_rows.extend(
                {
                    "config_id": config_id,
                    "training_seed": int(training_seed),
                    **row,
                }
                for row in single_calibration.curve
            )
            fitted_models.append(fitted)

        calibration, decision, calibration_prediction, validation_prediction = (
            _calibrate_and_select(
                fitted_models,
                batches["calibration"],
                outcomes["calibration"],
                batches["validation"],
                outcomes["validation"],
            )
        )
        metrics = _choice_metrics(
            outcomes["validation"],
            decision.candidate_index,
            metadata["validation"]["sample_seed"],
            batches["validation"].candidate_mask,
        )
        improved_count = _improved_seed_count(
            outcomes["validation"],
            decision.candidate_index,
            metadata["validation"]["sample_seed"],
        )
        seed_std = float(np.std(per_training_seed_rmse, ddof=0))
        classification_metrics = _classification_metrics(
            metrics, seed_std, improved_count
        )
        row = {
            "config_id": config_id,
            "hidden_dim": hidden_dim,
            "weight_cap": weight_cap,
            "validation_rmse": metrics["rmse"],
            "validation_link_rmse": metrics["link_rmse"],
            "validation_activity_f1": metrics["activity_f1"],
            "improvement_vs_default": metrics["improvement_vs_default"],
            "worst_seed_regret": metrics["worst_seed_regret"],
            "seed_std": seed_std,
            "improved_seed_count": improved_count,
            "defer_ratio": float(np.mean(decision.deferred)),
            "opportunity_quantile": calibration.quantile,
            "opportunity_threshold": calibration.threshold,
            "classification": classify_validation_result(classification_metrics),
            "training_seed_rmse": json.dumps(per_training_seed_rmse),
            **{
                key: value
                for key, value in classification_metrics.items()
                if key != "rmse"
            },
        }
        grid_rows.append(row)
        comparison_rows.append(
            {
                "model": f"objective_aligned__{config_id}",
                "result_kind": (
                    "deployable" if row["classification"] == "success" else "diagnostic_only"
                ),
                "validation_rmse": metrics["rmse"],
                "validation_link_rmse": metrics["link_rmse"],
                "validation_activity_f1": metrics["activity_f1"],
                "improvement_vs_default": metrics["improvement_vs_default"],
                "defer_ratio": row["defer_ratio"],
            }
        )
        fitted_by_config[config_id] = fitted_models
        checkpoints_by_config[config_id] = checkpoint_records
        results_by_config[config_id] = {
            "calibration": calibration,
            "decision": decision,
            "metrics": metrics,
            "calibration_prediction": calibration_prediction,
            "validation_prediction": validation_prediction,
        }

    best = choose_best_validation_config(grid_rows)
    best_result = results_by_config[best["config_id"]]
    best_decision = best_result["decision"]
    xgboost_status = "not_requested"
    if bool(args.run_xgboost):
        xgboost_status, xgboost_row = _run_xgboost_baseline(
            batches,
            outcomes,
            metadata["validation"]["sample_seed"],
            weight_cap=float(best["weight_cap"]),
            output_dir=args.output_dir,
            estimators=int(args.xgboost_estimators),
        )
        comparison_rows.append(xgboost_row)
    ablation_status = "not_requested"
    ablation_rows: list[dict[str, Any]] = []
    if bool(args.run_ablations):
        ablation_rows = _run_objective_ablations(
            batches,
            outcomes,
            metadata["train"]["sample_seed"],
            metadata["validation"]["sample_seed"],
            best,
            best_result["calibration"],
            best_result["validation_prediction"],
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            device=args.device,
        )
        _write_csv(
            args.output_dir / "feature_and_decision_ablation.csv", ablation_rows
        )
        ablation_status = "completed"
    _write_csv(args.output_dir / "selector_grid_results.csv", grid_rows)
    _write_csv(args.output_dir / "selector_comparison.csv", comparison_rows)
    _write_csv(args.output_dir / "opportunity_calibration.csv", calibration_rows)
    concentration_rows = []
    for name in ("train", "calibration", "validation"):
        concentration_rows.extend(
            _gain_concentration_rows(name, batches[name], outcomes[name])
        )
    _write_csv(args.output_dir / "gain_concentration.csv", concentration_rows)

    trace_rows = best_decision.to_records(metadata["validation"]["sample_ids"])
    actual_benefit = (
        outcomes["validation"].active_sse[:, outcomes["validation"].default_index]
        - outcomes["validation"].active_sse[
            np.arange(outcomes["validation"].active_sse.shape[0]),
            best_decision.candidate_index,
        ]
    )
    for index, row in enumerate(trace_rows):
        selected = int(best_decision.candidate_index[index])
        proposed = int(best_decision.proposed_candidate_index[index])
        row.update(
            {
                "seed": int(metadata["validation"]["sample_seed"][index]),
                "stage": str(batches["validation"].stage[index]),
                "candidate_name": batches["validation"].candidate_names[selected],
                "proposed_candidate_name": batches["validation"].candidate_names[proposed],
                "actual_selected_benefit": float(actual_benefit[index]),
            }
        )
    _write_csv(args.output_dir / "decision_trace_validation.csv", trace_rows)

    protocol_metadata = manifests["train"].get("protocol_metadata", {})
    try:
        source_git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_git_sha = "unavailable"
    selected_records = checkpoints_by_config[best["config_id"]]
    freeze_payload = {
        "configuration_digest": configuration_digest,
        "selected_config": {
            "config_id": best["config_id"],
            "hidden_dim": int(best["hidden_dim"]),
            "weight_cap": float(best["weight_cap"]),
        },
        "checkpoint_records": selected_records,
        "candidate_names": list(batches["train"].candidate_names),
        "feature_names": list(batches["train"].feature_names),
        "context_feature_names": list(batches["train"].context_feature_names),
        "cache_sha256": {
            name: str(manifests[name].get("cache_sha256", ""))
            for name in ("train", "calibration", "validation")
        },
        "calibration": {
            "quantile": best_result["calibration"].quantile,
            "threshold": best_result["calibration"].threshold,
        },
        "defer_rule": {
            "ranking": "ensemble_mean_predicted_sse_benefit",
            "candidate_lcb": "mean_minus_1.64_total_std_gt_zero",
            "opportunity_lcb": "calibration_quantile_threshold",
            "pareto": "observable_task_energy_proxy_non_dominated",
        },
        "source_git_sha": source_git_sha,
        "label_source_git_sha": protocol_metadata.get("source_git_sha"),
        "world_checkpoint_sha256": protocol_metadata.get("world_checkpoint_sha256"),
        "policy_checkpoint_sha256": protocol_metadata.get("policy_checkpoint_sha256"),
    }
    result_class = str(best["classification"])
    frozen_manifest = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "method": "objective_aligned_opportunity_benefit_selector",
        "configuration_frozen": result_class == "success",
        "configuration_digest": configuration_digest,
        "candidate_gate": validation_gate,
        "selected_config": best,
        "selected_checkpoint_records": selected_records,
        "selector_freeze_payload": freeze_payload,
        "selector_freeze_digest": canonical_sha256(freeze_payload),
        "selection_split": "validation",
        "defer_calibration_split": "calibration",
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "result_kind": "deployable" if result_class == "success" else "diagnostic_only",
        "classification": result_class,
    }
    frozen_path = args.output_dir / "frozen_selector_manifest.json"
    frozen_path.write_text(
        json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        **frozen_manifest,
        "runtime_seconds": float(time.time() - started),
        "xgboost_status": xgboost_status,
        "ablation_status": ablation_status,
        "outputs": {
            "grid": str(args.output_dir / "selector_grid_results.csv"),
            "comparison": str(args.output_dir / "selector_comparison.csv"),
            "calibration": str(args.output_dir / "opportunity_calibration.csv"),
            "trace": str(args.output_dir / "decision_trace_validation.csv"),
            "gain_concentration": str(args.output_dir / "gain_concentration.csv"),
            "ablations": (
                str(args.output_dir / "feature_and_decision_ablation.csv")
                if ablation_status == "completed"
                else None
            ),
            "frozen_manifest": str(frozen_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
