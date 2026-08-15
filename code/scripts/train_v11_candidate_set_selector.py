"""Train and validation-select the PI-JWM v11 benefit/risk selector."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.evaluation.candidate_selection import choice_rmse_from_sample_sse
from pi_jwm.v11_labeling import load_candidate_label_cache
from pi_jwm.v11_selector import (
    CandidateBatch,
    CandidateOutcome,
    aggregate_selected_metrics,
    audit_candidate_library,
    canonical_sha256,
    file_sha256,
    fit_listwise_selector,
    observable_pareto_deltas,
    predict_fitted_selector,
    select_with_defer,
)


DEFAULT_REPORT_DIR = CODE_ROOT / "artifacts/reports/pi_jwm_v11_selector_finalization_20260719"


def validate_cache_protocol(
    manifests: Mapping[str, Mapping[str, Any]],
    required_schema_version: int | None = None,
) -> str:
    required = ("train", "calibration", "validation")
    missing = [name for name in required if name not in manifests]
    if missing:
        raise ValueError(f"missing candidate label caches: {missing}")
    for name in required:
        if str(manifests[name].get("split_name")) != name:
            raise ValueError(f"cache split mismatch for {name}")
    digests = {str(manifests[name].get("configuration_digest")) for name in required}
    if len(digests) != 1:
        raise ValueError("candidate label cache configuration digest mismatch")
    candidate_orders = {tuple(manifests[name].get("candidate_names", ())) for name in required}
    if len(candidate_orders) != 1:
        raise ValueError("candidate order mismatch across caches")
    feature_orders = {tuple(manifests[name].get("feature_names", ())) for name in required}
    if len(feature_orders) != 1:
        raise ValueError("feature order mismatch across caches")
    context_orders = {
        tuple(manifests[name].get("context_feature_names", ())) for name in required
    }
    if len(context_orders) != 1:
        raise ValueError("context feature order mismatch across caches")
    if required_schema_version is not None:
        versions = {int(manifests[name].get("schema_version", -1)) for name in required}
        if versions != {int(required_schema_version)}:
            raise ValueError(
                f"formal selector training requires cache schema {required_schema_version}; found {sorted(versions)}"
            )
    return next(iter(digests))


def validate_physical_bridge_manifest(
    manifest_path: Path,
    cache_paths: Mapping[str, Path],
) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = str(manifest.get("bridge_manifest_digest", ""))
    payload = {key: value for key, value in manifest.items() if key != "bridge_manifest_digest"}
    actual = canonical_sha256(payload)
    if len(expected) != 64 or actual != expected:
        raise ValueError("physical bridge manifest digest mismatch")
    if not bool(manifest.get("bridge_gate_passed", False)):
        raise ValueError("physical bridge gate did not pass")
    if bool(manifest.get("matched_test_accessed", False)) or bool(
        manifest.get("external_holdout_accessed", False)
    ):
        raise ValueError("physical bridge manifest reports locked split access")
    if int(manifest.get("actual_outcome_feature_count", -1)) != 0:
        raise ValueError("physical bridge manifest contains actual outcome features")
    records = manifest.get("augmented_caches")
    if not isinstance(records, Mapping):
        raise ValueError("physical bridge manifest has no augmented cache records")
    for split in ("train", "calibration", "validation"):
        record = records.get(split)
        if not isinstance(record, Mapping) or split not in cache_paths:
            raise ValueError(f"physical bridge manifest is missing {split} cache")
        if file_sha256(cache_paths[split]) != str(record.get("sha256", "")):
            raise ValueError(f"physical bridge cache hash mismatch for {split}")
    return actual


def validate_physical_cache_presence(
    manifests: Mapping[str, Mapping[str, Any]],
    manifest_supplied: bool,
) -> bool:
    flags = {
        split: any(
            str(name).startswith("physical_")
            for name in manifests[split].get("feature_names", ())
        )
        for split in ("train", "calibration", "validation")
    }
    if len(set(flags.values())) != 1:
        raise ValueError("physical feature presence differs across caches")
    has_physical = next(iter(flags.values()))
    if has_physical and not bool(manifest_supplied):
        raise ValueError("physical cache requires --physical-bridge-manifest")
    if bool(manifest_supplied) and not has_physical:
        raise ValueError("selector cache does not contain physical benefit features")
    return has_physical


def calibrate_improvement_bias(
    predicted: np.ndarray,
    actual: np.ndarray,
    candidate_mask: np.ndarray,
    allow_empty: bool = False,
) -> float:
    prediction = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(actual, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("predicted, actual, and candidate_mask must share shape")
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        if bool(allow_empty):
            observable = prediction[mask & np.isfinite(prediction)]
            if observable.size:
                return float(np.max(observable) + 1.0)
        raise ValueError("calibration requires at least one valid candidate outcome")
    return float(np.median(prediction[valid] - target[valid]))


def choose_best_validation_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("validation config rows must not be empty")
    return min(
        rows,
        key=lambda row: (
            float(row["validation_rmse"]),
            float(row["worst_seed_regret"]),
            float(row["seed_std"]),
            str(row["config_id"]),
        ),
    )


def classify_selector_validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered deployable selector validation gates."""

    def finite(name: str) -> float:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    rmse_field = "validation_rmse" if "validation_rmse" in metrics else "rmse"
    checks = {
        "rmse": finite(rmse_field) < 230.8556,
        "improved_seed_count": int(metrics["improved_seed_count"]) >= 7,
        "positive_precision": finite("executed_positive_precision") >= 0.65,
        "negative_selection_rate": finite("negative_selection_rate") <= 0.20,
        "activity_f1_drop": finite("activity_f1_drop") <= 0.002,
        "link_rmse_relative_degradation": (
            finite("link_rmse_relative_degradation") <= 0.02
        ),
        "training_seed_std": finite("training_seed_std") <= 5.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": dict(metrics),
    }


def enforce_result_kinds(
    rows: list[dict[str, Any]], gate_passed: bool
) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    if not bool(gate_passed):
        for row in result:
            if str(row.get("result_kind")) != "sample_oracle":
                row["result_kind"] = "diagnostic_only"
    return result


def masked_oracle_choice(active_sse: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    sse = np.asarray(active_sse, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if sse.ndim != 2 or sse.shape != mask.shape or np.any(mask.sum(axis=1) == 0):
        raise ValueError("active_sse and candidate_mask must be compatible non-empty rows")
    return np.argmin(np.where(mask, sse, np.inf), axis=1).astype(np.int64)


def _metadata(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as arrays:
        return {
            "sample_ids": arrays["sample_ids"].copy(),
            "sample_seed": arrays["sample_seed"].copy(),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _choice_metrics(
    outcome: CandidateOutcome,
    choice: np.ndarray,
    sample_seed: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_metrics = aggregate_selected_metrics(outcome, choice)
    selected_rmse = selected_metrics["active_rate_rmse"]
    default_choice = np.full((outcome.active_sse.shape[0],), outcome.default_index, dtype=np.int64)
    default_rmse = choice_rmse_from_sample_sse(
        outcome.active_sse, outcome.active_count, default_choice
    )
    default_metrics = aggregate_selected_metrics(outcome, default_choice)
    available = (
        np.ones_like(outcome.active_sse, dtype=bool)
        if candidate_mask is None
        else np.asarray(candidate_mask, dtype=bool)
    )
    oracle_choice = masked_oracle_choice(outcome.active_sse, available)
    oracle_rmse = choice_rmse_from_sample_sse(
        outcome.active_sse, outcome.active_count, oracle_choice
    )
    oracle_metrics = aggregate_selected_metrics(outcome, oracle_choice)
    per_seed = []
    for seed in sorted(int(value) for value in np.unique(sample_seed)):
        keep = np.asarray(sample_seed) == seed
        if not np.any(outcome.active_count[keep] > 0):
            continue
        seed_choice = choice[keep]
        seed_sse = outcome.active_sse[keep]
        seed_count = outcome.active_count[keep]
        seed_rmse = choice_rmse_from_sample_sse(seed_sse, seed_count, seed_choice)
        seed_oracle = choice_rmse_from_sample_sse(
            seed_sse, seed_count, masked_oracle_choice(seed_sse, available[keep])
        )
        seed_default = choice_rmse_from_sample_sse(
            seed_sse,
            seed_count,
            np.full(seed_choice.shape, outcome.default_index, dtype=np.int64),
        )
        per_seed.append(
            {
                "seed": seed,
                "rmse": seed_rmse,
                "default_rmse": seed_default,
                "oracle_rmse": seed_oracle,
                "regret": seed_rmse - seed_oracle,
            }
        )
    row_index = np.arange(outcome.active_sse.shape[0])
    sample_benefit = (
        outcome.active_sse[:, outcome.default_index]
        - outcome.active_sse[row_index, np.asarray(choice, dtype=np.int64)]
    )
    executed = (
        np.asarray(choice, dtype=np.int64) != outcome.default_index
    ) & (outcome.active_count > 0)
    executed_positive_precision = (
        float(np.mean(sample_benefit[executed] > 0.0)) if np.any(executed) else 1.0
    )
    negative_selection_rate = (
        float(np.mean(sample_benefit[executed] < 0.0)) if np.any(executed) else 0.0
    )
    activity_f1_drop = (
        None
        if default_metrics["activity_f1"] is None or selected_metrics["activity_f1"] is None
        else float(default_metrics["activity_f1"] - selected_metrics["activity_f1"])
    )
    link_rmse_relative_degradation = (
        None
        if default_metrics["link_rmse"] is None or selected_metrics["link_rmse"] is None
        else float(
            (selected_metrics["link_rmse"] - default_metrics["link_rmse"])
            / max(float(default_metrics["link_rmse"]), 1e-12)
        )
    )
    return {
        "rmse": None if selected_rmse is None else float(selected_rmse),
        "active_rate_rmse": None if selected_rmse is None else float(selected_rmse),
        "link_rmse": selected_metrics["link_rmse"],
        "activity_f1": selected_metrics["activity_f1"],
        "default_rmse": None if default_rmse is None else float(default_rmse),
        "default_link_rmse": default_metrics["link_rmse"],
        "default_activity_f1": default_metrics["activity_f1"],
        "oracle_rmse": None if oracle_rmse is None else float(oracle_rmse),
        "oracle_link_rmse": oracle_metrics["link_rmse"],
        "oracle_activity_f1": oracle_metrics["activity_f1"],
        "improvement_vs_default": (
            None
            if default_rmse is None or selected_rmse is None
            else float(default_rmse - selected_rmse)
        ),
        "worst_seed_regret": max((float(row["regret"]) for row in per_seed), default=0.0),
        "improved_seed_count": int(
            sum(float(row["rmse"]) < float(row["default_rmse"]) for row in per_seed)
        ),
        "executed_positive_precision": executed_positive_precision,
        "negative_selection_rate": negative_selection_rate,
        "activity_f1_drop": activity_f1_drop,
        "link_rmse_relative_degradation": link_rmse_relative_degradation,
        "per_seed": per_seed,
    }


def _pairwise_features(batch: CandidateBatch, default_index: int) -> np.ndarray:
    default = batch.candidate_features[:, default_index : default_index + 1]
    context = np.broadcast_to(batch.context[:, None, :], (*batch.candidate_features.shape[:2], batch.context.shape[1]))
    return np.concatenate(
        [batch.candidate_features, np.broadcast_to(default, batch.candidate_features.shape), batch.candidate_features - default, context],
        axis=2,
    ).astype(np.float32)


def _fit_pairwise_baseline(
    train_batch: CandidateBatch,
    train_outcome: CandidateOutcome,
    model_name: str,
    seed: int,
):
    features = _pairwise_features(train_batch, train_outcome.default_index)
    labels = train_outcome.improvement > 0.0
    valid = train_batch.candidate_mask & (train_outcome.active_count[:, None] > 0)
    valid[:, train_outcome.default_index] = False
    x = features[valid]
    y = labels[valid].astype(np.int64)
    if x.shape[0] == 0 or np.unique(y).size < 2:
        return None
    if model_name == "rf_pairwise":
        model = RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif model_name == "gb_pairwise":
        model = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=3,
            min_samples_leaf=4,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown pairwise baseline: {model_name}")
    model.fit(x, y)
    return model


def _predict_pairwise(model, batch: CandidateBatch, default_index: int) -> np.ndarray:
    score = np.full(batch.candidate_mask.shape, -np.inf, dtype=np.float32)
    score[:, default_index] = 0.5
    if model is None:
        return score
    features = _pairwise_features(batch, default_index)
    probabilities = model.predict_proba(features.reshape(-1, features.shape[2]))
    positive_column = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    score[:] = probabilities[:, positive_column].reshape(score.shape)
    score[~batch.candidate_mask] = -np.inf
    score[:, default_index] = 0.5
    return score


def _fit_pointwise(train_batch: CandidateBatch, train_outcome: CandidateOutcome, seed: int):
    context = np.broadcast_to(
        train_batch.context[:, None, :],
        (*train_batch.candidate_features.shape[:2], train_batch.context.shape[1]),
    )
    features = np.concatenate([train_batch.candidate_features, context], axis=2)
    valid = train_batch.candidate_mask & (train_outcome.active_count[:, None] > 0)
    model = GradientBoostingRegressor(
        n_estimators=160,
        max_depth=3,
        min_samples_leaf=4,
        loss="huber",
        random_state=int(seed),
    )
    model.fit(features[valid], train_outcome.improvement[valid])
    return model


def _predict_pointwise(model, batch: CandidateBatch) -> np.ndarray:
    context = np.broadcast_to(
        batch.context[:, None, :],
        (*batch.candidate_features.shape[:2], batch.context.shape[1]),
    )
    features = np.concatenate([batch.candidate_features, context], axis=2)
    prediction = model.predict(features.reshape(-1, features.shape[2])).reshape(batch.candidate_mask.shape)
    return np.where(batch.candidate_mask, prediction, -np.inf).astype(np.float32)


def _classical_rows(
    train_batch: CandidateBatch,
    train_outcome: CandidateOutcome,
    calibration_batch: CandidateBatch,
    calibration_outcome: CandidateOutcome,
    validation_batch: CandidateBatch,
    validation_outcome: CandidateOutcome,
    validation_seed: np.ndarray,
    seed: int,
    allow_empty_calibration: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    default_choice = np.full(
        (validation_outcome.active_sse.shape[0],), validation_outcome.default_index, dtype=np.int64
    )
    for name, choice in (
        ("ranked_allocation_default", default_choice),
        ("sample_oracle", masked_oracle_choice(validation_outcome.active_sse, validation_batch.candidate_mask)),
    ):
        metrics = _choice_metrics(validation_outcome, choice, validation_seed, validation_batch.candidate_mask)
        rows.append(
            {
                "model": name,
                "result_kind": "sample_oracle" if name == "sample_oracle" else "deployable",
                "validation_rmse": metrics["rmse"],
                "validation_link_rmse": metrics["link_rmse"],
                "validation_activity_f1": metrics["activity_f1"],
                "improvement_vs_default": metrics["improvement_vs_default"],
                "defer_ratio": 1.0 if name == "ranked_allocation_default" else 0.0,
            }
        )
    for model_name in ("rf_pairwise", "gb_pairwise"):
        model = _fit_pairwise_baseline(train_batch, train_outcome, model_name, seed)
        calibration_score = _predict_pairwise(model, calibration_batch, train_outcome.default_index)
        thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
        best = None
        for threshold in thresholds:
            score = calibration_score.copy()
            choice = np.argmax(score, axis=1)
            maximum = np.max(score, axis=1)
            choice[maximum < threshold] = train_outcome.default_index
            metrics = _choice_metrics(
                calibration_outcome, choice, np.zeros_like(choice), calibration_batch.candidate_mask
            )
            key = (metrics["rmse"], -float(threshold))
            if best is None or key < best[0]:
                best = (key, float(threshold))
        validation_score = _predict_pairwise(model, validation_batch, train_outcome.default_index)
        choice = np.argmax(validation_score, axis=1)
        maximum = np.max(validation_score, axis=1)
        choice[maximum < best[1]] = train_outcome.default_index
        metrics = _choice_metrics(validation_outcome, choice, validation_seed, validation_batch.candidate_mask)
        rows.append(
            {
                "model": model_name,
                "result_kind": "deployable",
                "validation_rmse": metrics["rmse"],
                "validation_link_rmse": metrics["link_rmse"],
                "validation_activity_f1": metrics["activity_f1"],
                "improvement_vs_default": metrics["improvement_vs_default"],
                "defer_ratio": float(np.mean(choice == train_outcome.default_index)),
                "calibration_threshold": best[1],
            }
        )
    pointwise = _fit_pointwise(train_batch, train_outcome, seed)
    calibration_prediction = _predict_pointwise(pointwise, calibration_batch)
    bias = calibrate_improvement_bias(
        calibration_prediction,
        calibration_outcome.improvement,
        calibration_batch.candidate_mask,
        allow_empty=allow_empty_calibration,
    )
    validation_prediction = _predict_pointwise(pointwise, validation_batch) - bias
    choice = np.argmax(validation_prediction, axis=1)
    maximum = np.max(validation_prediction, axis=1)
    choice[maximum <= 0.0] = train_outcome.default_index
    metrics = _choice_metrics(validation_outcome, choice, validation_seed, validation_batch.candidate_mask)
    rows.append(
        {
            "model": "pointwise_benefit_gb",
            "result_kind": "deployable",
            "validation_rmse": metrics["rmse"],
            "validation_link_rmse": metrics["link_rmse"],
            "validation_activity_f1": metrics["activity_f1"],
            "improvement_vs_default": metrics["improvement_vs_default"],
            "defer_ratio": float(np.mean(choice == train_outcome.default_index)),
            "calibration_bias": bias,
        }
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR / "selector_training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--allow-smoke-gate-failure", action="store_true")
    parser.add_argument("--physical-bridge-manifest", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "train": args.train_cache,
        "calibration": args.calibration_cache,
        "validation": args.validation_cache,
    }
    physical_manifest_path = getattr(args, "physical_bridge_manifest", None)
    if physical_manifest_path is not None:
        validate_physical_bridge_manifest(physical_manifest_path, cache_paths)
    loaded = {name: load_candidate_label_cache(path) for name, path in cache_paths.items()}
    manifests = {name: value[2] for name, value in loaded.items()}
    validate_physical_cache_presence(
        manifests, manifest_supplied=physical_manifest_path is not None
    )
    configuration_digest = validate_cache_protocol(
        manifests,
        required_schema_version=None if bool(args.allow_smoke_gate_failure) else 6,
    )
    batches = {name: value[0] for name, value in loaded.items()}
    outcomes = {name: value[1] for name, value in loaded.items()}
    metadata = {name: _metadata(path) for name, path in cache_paths.items()}
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
        raise RuntimeError(f"candidate library gate failed; selector training is forbidden: {validation_gate}")

    comparison_rows = _classical_rows(
        batches["train"],
        outcomes["train"],
        batches["calibration"],
        outcomes["calibration"],
        batches["validation"],
        outcomes["validation"],
        metadata["validation"]["sample_seed"],
        seed=17,
        allow_empty_calibration=bool(args.allow_smoke_gate_failure),
    )
    config_rows = []
    checkpoints = {}
    checkpoint_records = {}
    grid = itertools.product(args.hidden_dim, args.temperature, args.dropout)
    for hidden_dim, temperature, dropout in grid:
        config_id = f"h{hidden_dim}_t{temperature:g}_d{dropout:g}"
        ensemble_rank_scores = []
        ensemble_improvements = []
        ensemble_uncertainties = []
        per_training_seed_rmse = []
        checkpoint_paths = []
        config_checkpoint_records = []
        for training_seed in args.training_seeds:
            fitted = fit_listwise_selector(
                batches["train"],
                outcomes["train"],
                hidden_dim=int(hidden_dim),
                temperature=float(temperature),
                dropout=float(dropout),
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                seed=int(training_seed),
                device=args.device,
                group_ids=metadata["train"]["sample_seed"],
            )
            calibration_prediction = predict_fitted_selector(fitted, batches["calibration"])["predicted_improvement"]
            bias = calibrate_improvement_bias(
                calibration_prediction,
                outcomes["calibration"].improvement / fitted.target_scale,
                batches["calibration"].candidate_mask,
                allow_empty=bool(args.allow_smoke_gate_failure),
            )
            validation_heads = predict_fitted_selector(fitted, batches["validation"])
            validation_improvement = validation_heads["predicted_improvement"] - bias
            ensemble_rank_scores.append(validation_heads["score"])
            ensemble_improvements.append(validation_improvement)
            ensemble_uncertainties.append(validation_heads["uncertainty"])
            seed_decision = select_with_defer(
                validation_heads["score"][None, ...],
                default_index=outcomes["validation"].default_index,
                ensemble_improvement=validation_improvement[None, ...],
                ensemble_uncertainty=validation_heads["uncertainty"][None, ...],
                candidate_mask=batches["validation"].candidate_mask,
                task_delta=observable_pareto_deltas(
                    batches["validation"], outcomes["validation"].default_index
                )[0],
                energy_delta=observable_pareto_deltas(
                    batches["validation"], outcomes["validation"].default_index
                )[1],
            )
            seed_metrics = _choice_metrics(
                outcomes["validation"],
                seed_decision.candidate_index,
                metadata["validation"]["sample_seed"],
                batches["validation"].candidate_mask,
            )
            per_training_seed_rmse.append(seed_metrics["rmse"])
            checkpoint_path = args.output_dir / f"candidate_set_ranker_{config_id}_seed{training_seed}.pt"
            torch.save(
                {
                    "state_dict": {name: value.detach().cpu() for name, value in fitted.model.state_dict().items()},
                    "candidate_dim": batches["train"].candidate_features.shape[2],
                    "context_dim": batches["train"].context.shape[1],
                    "hidden_dim": int(hidden_dim),
                    "dropout": float(dropout),
                    "temperature": float(temperature),
                    "training_seed": int(training_seed),
                    "calibration_bias": float(bias),
                    "target_scale": float(fitted.target_scale),
                    "candidate_mean": torch.from_numpy(fitted.candidate_mean),
                    "candidate_scale": torch.from_numpy(fitted.candidate_scale),
                    "context_mean": torch.from_numpy(fitted.context_mean),
                    "context_scale": torch.from_numpy(fitted.context_scale),
                    "configuration_digest": configuration_digest,
                    "history": fitted.history,
                },
                checkpoint_path,
            )
            checkpoint_paths.append(str(checkpoint_path))
            config_checkpoint_records.append(
                {
                    "file": checkpoint_path.name,
                    "sha256": file_sha256(checkpoint_path),
                    "training_seed": int(training_seed),
                    "calibration_bias": float(bias),
                    "target_scale": float(fitted.target_scale),
                }
            )
        ensemble = np.stack(ensemble_rank_scores, axis=0)
        decision = select_with_defer(
            ensemble,
            default_index=outcomes["validation"].default_index,
            ensemble_improvement=np.stack(ensemble_improvements, axis=0),
            ensemble_uncertainty=np.stack(ensemble_uncertainties, axis=0),
            candidate_mask=batches["validation"].candidate_mask,
            task_delta=observable_pareto_deltas(
                batches["validation"], outcomes["validation"].default_index
            )[0],
            energy_delta=observable_pareto_deltas(
                batches["validation"], outcomes["validation"].default_index
            )[1],
        )
        metrics = _choice_metrics(
            outcomes["validation"],
            decision.candidate_index,
            metadata["validation"]["sample_seed"],
            batches["validation"].candidate_mask,
        )
        row = {
            "config_id": config_id,
            "hidden_dim": int(hidden_dim),
            "temperature": float(temperature),
            "dropout": float(dropout),
            "validation_rmse": metrics["rmse"],
            "validation_link_rmse": metrics["link_rmse"],
            "validation_activity_f1": metrics["activity_f1"],
            "improvement_vs_default": metrics["improvement_vs_default"],
            "worst_seed_regret": metrics["worst_seed_regret"],
            "seed_std": float(np.std(per_training_seed_rmse, ddof=0)),
            "training_seed_std": float(np.std(per_training_seed_rmse, ddof=0)),
            "defer_ratio": float(np.mean(decision.deferred)),
            "training_seed_rmse": json.dumps(per_training_seed_rmse),
            "improved_seed_count": metrics["improved_seed_count"],
            "executed_positive_precision": metrics["executed_positive_precision"],
            "negative_selection_rate": metrics["negative_selection_rate"],
            "activity_f1_drop": metrics["activity_f1_drop"],
            "link_rmse_relative_degradation": metrics[
                "link_rmse_relative_degradation"
            ],
        }
        config_rows.append(row)
        checkpoints[config_id] = checkpoint_paths
        checkpoint_records[config_id] = config_checkpoint_records
    best = choose_best_validation_config(config_rows)
    selector_validation_gate = classify_selector_validation_gate(best)
    freeze_passed = bool(validation_gate["passed"]) and bool(
        selector_validation_gate["passed"]
    )
    comparison_rows.extend(
        {
            "model": f"candidate_set_ranker__{row['config_id']}",
            "result_kind": "deployable" if freeze_passed else "diagnostic_only",
            "validation_rmse": row["validation_rmse"],
            "validation_link_rmse": row["validation_link_rmse"],
            "validation_activity_f1": row["validation_activity_f1"],
            "improvement_vs_default": row["improvement_vs_default"],
            "defer_ratio": row["defer_ratio"],
        }
        for row in config_rows
    )
    comparison_rows = enforce_result_kinds(comparison_rows, freeze_passed)
    _write_csv(args.output_dir / "selector_grid_results.csv", config_rows)
    _write_csv(args.output_dir / "selector_comparison.csv", comparison_rows)
    selected_records = checkpoint_records[best["config_id"]]
    protocol_metadata = manifests["train"].get("protocol_metadata", {})
    freeze_payload = {
        "configuration_digest": configuration_digest,
        "selected_config": {
            key: best[key]
            for key in ("config_id", "hidden_dim", "temperature", "dropout")
        },
        "checkpoint_records": selected_records,
        "candidate_names": list(manifests["train"].get("candidate_names", [])),
        "feature_names": list(manifests["train"].get("feature_names", [])),
        "context_feature_names": list(
            manifests["train"].get("context_feature_names", [])
        ),
        "cache_sha256": {
            name: str(manifests[name].get("cache_sha256", ""))
            for name in ("train", "calibration", "validation")
        },
        "defer_rule": {
            "ranking": "ensemble_mean_listwise_score",
            "execution": "mean_calibrated_improvement_minus_1.64_total_std_gt_0",
            "uncertainty": "ensemble_variance_plus_predicted_aleatoric_variance",
            "pareto": "observable_task_energy_proxy_non_dominated",
        },
        "source_git_sha": protocol_metadata.get("source_git_sha"),
        "world_checkpoint_sha256": protocol_metadata.get("world_checkpoint_sha256"),
        "policy_checkpoint_sha256": protocol_metadata.get("policy_checkpoint_sha256"),
    }
    frozen_manifest = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "configuration_frozen": freeze_passed,
        "configuration_digest": configuration_digest,
        "candidate_gate": validation_gate,
        "selector_validation_gate": selector_validation_gate,
        "selected_config": best,
        "selected_checkpoints": checkpoints[best["config_id"]],
        "selected_checkpoint_records": selected_records,
        "selector_freeze_payload": freeze_payload,
        "selector_freeze_digest": canonical_sha256(freeze_payload),
        "selection_split": "validation",
        "defer_calibration_split": "calibration",
        "matched_test_accessed": False,
        "result_kind": "deployable" if freeze_passed else "diagnostic_only",
    }
    frozen_path = args.output_dir / "frozen_selector_manifest.json"
    frozen_path.write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        **frozen_manifest,
        "runtime_seconds": float(time.time() - started),
        "outputs": {
            "grid": str(args.output_dir / "selector_grid_results.csv"),
            "comparison": str(args.output_dir / "selector_comparison.csv"),
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
