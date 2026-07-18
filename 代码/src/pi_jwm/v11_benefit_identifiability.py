"""Leakage-safe candidate-benefit identifiability diagnostics for PI-JWM v11."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from .v11_selector import CandidateBatch, CandidateOutcome, aggregate_selected_metrics


BENEFIT_EPSILON = 1e-6


@dataclass(frozen=True)
class BenefitAuditDataset:
    batch: CandidateBatch
    outcome: CandidateOutcome
    sample_ids: np.ndarray
    sample_seed: np.ndarray
    valid_sample: np.ndarray
    legal_candidate: np.ndarray
    candidate_benefit: np.ndarray
    candidate_positive: np.ndarray
    opportunity: np.ndarray
    opportunity_positive: np.ndarray
    flat_sample_index: np.ndarray
    flat_candidate_index: np.ndarray


@dataclass(frozen=True)
class BenefitFeatureGroup:
    opportunity_features: np.ndarray
    opportunity_feature_names: tuple[str, ...]
    candidate_features: np.ndarray | None
    candidate_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TrainNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.mean.shape[0]:
            raise ValueError("normalizer input feature dimension mismatch")
        return ((array - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class BenefitPredictions:
    opportunity_probability: np.ndarray
    candidate_sign_probability: np.ndarray
    predicted_benefit: np.ndarray


@dataclass(frozen=True)
class SafeThresholdSelection:
    opportunity_threshold: float
    sign_threshold: float
    choice: np.ndarray
    metrics: dict[str, float | int | None]
    status: str


@dataclass(frozen=True)
class FittedBenefitAuditModel:
    model_kind: str
    opportunity_model: Any
    sign_model: Any | None
    benefit_model: Any | None
    opportunity_normalizer: TrainNormalizer | None
    candidate_normalizer: TrainNormalizer | None
    opportunity_status: str
    candidate_status: str


class _ConstantProbability:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        count = np.asarray(values).shape[0]
        positive = np.full(count, self.probability, dtype=np.float32)
        return np.stack([1.0 - positive, positive], axis=1)


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(values).shape[0], self.value, dtype=np.float32)


def build_benefit_audit_dataset(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    sample_ids: np.ndarray,
    sample_seed: np.ndarray,
) -> BenefitAuditDataset:
    """Construct deployable-input audit targets without exposing them as features."""
    sample_count, candidate_count = batch.candidate_mask.shape
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    if ids.shape != (sample_count,) or seeds.shape != (sample_count,):
        raise ValueError("sample_ids and sample_seed must match candidate batch")
    if outcome.active_sse.shape != (sample_count, candidate_count):
        raise ValueError("candidate batch and outcome dimensions must match")

    default = int(outcome.default_index)
    if not np.all(batch.candidate_mask[:, default]):
        raise ValueError("ranked-allocation default must be available for every sample")
    applicable = (
        batch.candidate_mask
        if outcome.action_applicable is None
        else np.asarray(outcome.action_applicable, dtype=bool)
    )
    applied = (
        batch.candidate_mask
        if outcome.action_applied is None
        else np.asarray(outcome.action_applied, dtype=bool)
    )
    if applicable.shape != batch.candidate_mask.shape or applied.shape != batch.candidate_mask.shape:
        raise ValueError("action applicability and applied masks must match candidate batch")

    legal = batch.candidate_mask & applicable & applied
    legal[:, default] = batch.candidate_mask[:, default] & applicable[:, default]
    if not np.all(legal[:, default]):
        raise ValueError("ranked-allocation default must be legal for every sample")

    valid_sample = np.asarray(outcome.active_count, dtype=np.int64) > 0
    benefit = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    raw_benefit = (
        outcome.active_sse[:, default, None] - outcome.active_sse
    ).astype(np.float32)
    trainable = legal & valid_sample[:, None]
    benefit[trainable] = raw_benefit[trainable]
    positive = np.isfinite(benefit) & (benefit > BENEFIT_EPSILON)

    opportunity = np.full((sample_count,), np.nan, dtype=np.float32)
    for sample_index in np.flatnonzero(valid_sample):
        opportunity[sample_index] = max(
            0.0, float(np.nanmax(benefit[sample_index, legal[sample_index]]))
        )
    opportunity_positive = np.isfinite(opportunity) & (opportunity > BENEFIT_EPSILON)
    flat_sample, flat_candidate = np.where(trainable)

    return BenefitAuditDataset(
        batch=batch,
        outcome=outcome,
        sample_ids=ids,
        sample_seed=seeds,
        valid_sample=valid_sample,
        legal_candidate=legal,
        candidate_benefit=benefit,
        candidate_positive=positive,
        opportunity=opportunity,
        opportunity_positive=opportunity_positive,
        flat_sample_index=flat_sample.astype(np.int64),
        flat_candidate_index=flat_candidate.astype(np.int64),
    )


def fit_train_normalizer(values: np.ndarray) -> TrainNormalizer:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("train normalization requires a finite non-empty matrix")
    mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(array, axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return TrainNormalizer(mean=mean, scale=scale)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_") or "unknown"


def _stage_features(stages: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    categories = ("offload", "compute", "return", "unknown")
    normalized = np.asarray(
        [value if value in categories[:-1] else "unknown" for value in np.asarray(stages).astype(str)]
    )
    values = np.stack([normalized == category for category in categories], axis=1).astype(np.float32)
    return values, tuple(f"stage_{category}" for category in categories)


def _candidate_identity_features(
    dataset: BenefitAuditDataset,
) -> tuple[np.ndarray, tuple[str, ...]]:
    candidate_count = dataset.batch.candidate_mask.shape[1]
    values = np.eye(candidate_count, dtype=np.float32)[dataset.flat_candidate_index]
    names = dataset.batch.candidate_names or tuple(
        f"candidate_{index}" for index in range(candidate_count)
    )
    return values, tuple(f"candidate_id_{_safe_name(name)}" for name in names)


def _pooled_candidate_features(
    dataset: BenefitAuditDataset,
    columns: list[int],
) -> tuple[np.ndarray, tuple[str, ...]]:
    valid_indices = np.flatnonzero(dataset.valid_sample)
    if not columns:
        return np.zeros((valid_indices.shape[0], 0), dtype=np.float32), ()
    result = []
    for sample_index in valid_indices:
        legal_values = dataset.batch.candidate_features[
            sample_index, dataset.legal_candidate[sample_index]
        ][:, columns]
        result.append(
            np.concatenate(
                [
                    np.mean(legal_values, axis=0),
                    np.max(legal_values, axis=0),
                    np.min(legal_values, axis=0),
                ]
            )
        )
    base_names = [dataset.batch.feature_names[index] for index in columns]
    names = tuple(
        f"candidate_pool_{stat}_{name}"
        for stat in ("mean", "max", "min")
        for name in base_names
    )
    return np.asarray(result, dtype=np.float32), names


def build_benefit_feature_groups(
    dataset: BenefitAuditDataset,
) -> dict[str, BenefitFeatureGroup]:
    """Build the frozen, deployable feature groups used by the audit."""
    forbidden = ("seed", "future", "oracle", "benefit", "regret", "sse", "outcome")
    all_names = dataset.batch.context_feature_names + dataset.batch.feature_names
    if any(token in name.lower() for name in all_names for token in forbidden):
        raise ValueError("benefit audit feature protocol contains a forbidden field")

    feature_names = dataset.batch.feature_names
    action_columns = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith(("rb_", "cpu_", "offload_", "return_", "action_family_"))
        or name == "predicted_energy_proxy"
    ]
    forecast_columns = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("predicted_") and "delta" in name
    ]
    selected_columns = [
        index for index, name in enumerate(feature_names) if name.startswith("selected_")
    ]
    group_columns = {
        "prior_only": [],
        "context_only": [],
        "candidate_only": action_columns,
        "forecast_delta": forecast_columns,
        "selected_edge": selected_columns,
        "full_schema_v5": list(range(len(feature_names))),
    }

    valid_indices = np.flatnonzero(dataset.valid_sample)
    opportunity_stage, stage_names = _stage_features(dataset.batch.stage[valid_indices])
    flat_stage, _ = _stage_features(dataset.batch.stage[dataset.flat_sample_index])
    candidate_identity, candidate_identity_names = _candidate_identity_features(dataset)
    groups: dict[str, BenefitFeatureGroup] = {}
    for group_name, columns in group_columns.items():
        pooled, pooled_names = _pooled_candidate_features(dataset, columns)
        opportunity_blocks = [opportunity_stage]
        opportunity_names: tuple[str, ...] = stage_names
        if group_name in {"context_only", "full_schema_v5"}:
            opportunity_blocks.append(dataset.batch.context[valid_indices])
            opportunity_names += dataset.batch.context_feature_names
        if columns:
            opportunity_blocks.append(pooled)
            opportunity_names += pooled_names
        opportunity = np.concatenate(opportunity_blocks, axis=1).astype(np.float32)

        if group_name == "context_only":
            candidate = None
            candidate_names: tuple[str, ...] = ()
        else:
            candidate_blocks = [flat_stage, candidate_identity]
            candidate_names = stage_names + candidate_identity_names
            if group_name == "full_schema_v5":
                candidate_blocks.append(dataset.batch.context[dataset.flat_sample_index])
                candidate_names += dataset.batch.context_feature_names
            if columns:
                candidate_blocks.append(
                    dataset.batch.candidate_features[
                        dataset.flat_sample_index, dataset.flat_candidate_index
                    ][:, columns]
                )
                candidate_names += tuple(feature_names[index] for index in columns)
            candidate = np.concatenate(candidate_blocks, axis=1).astype(np.float32)
        groups[group_name] = BenefitFeatureGroup(
            opportunity_features=opportunity,
            opportunity_feature_names=opportunity_names,
            candidate_features=candidate,
            candidate_feature_names=candidate_names,
        )
    return groups


def seed_group_folds(
    sample_seed: np.ndarray,
    n_splits: int = 5,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    unique = np.unique(seeds)
    folds = int(n_splits)
    if folds < 2 or folds > unique.shape[0]:
        raise ValueError("n_splits must be between 2 and the number of unique seeds")
    result = []
    for fold_index in range(folds):
        validation_seeds = unique[fold_index::folds]
        validation = np.flatnonzero(np.isin(seeds, validation_seeds))
        train = np.flatnonzero(~np.isin(seeds, validation_seeds))
        result.append((train.astype(np.int64), validation.astype(np.int64)))
    return tuple(result)


def _validate_predictions(
    dataset: BenefitAuditDataset,
    predictions: BenefitPredictions,
) -> None:
    sample_count, candidate_count = dataset.batch.candidate_mask.shape
    opportunity = np.asarray(predictions.opportunity_probability, dtype=np.float32).reshape(-1)
    sign = np.asarray(predictions.candidate_sign_probability, dtype=np.float32)
    benefit = np.asarray(predictions.predicted_benefit, dtype=np.float32)
    if opportunity.shape != (sample_count,):
        raise ValueError("opportunity_probability must contain one value per sample")
    if sign.shape != (sample_count, candidate_count) or benefit.shape != sign.shape:
        raise ValueError("candidate predictions must match candidate dimensions")
    finite_opportunity = np.isfinite(opportunity[dataset.valid_sample])
    finite_candidate = np.isfinite(sign[dataset.legal_candidate & dataset.valid_sample[:, None]])
    finite_benefit = np.isfinite(benefit[dataset.legal_candidate & dataset.valid_sample[:, None]])
    if not np.all(finite_opportunity) or not np.all(finite_candidate) or not np.all(finite_benefit):
        raise ValueError("valid audit predictions must be finite")


def select_benefit_candidates(
    dataset: BenefitAuditDataset,
    predictions: BenefitPredictions,
    opportunity_threshold: float,
    sign_threshold: float,
) -> np.ndarray:
    _validate_predictions(dataset, predictions)
    opportunity = np.asarray(predictions.opportunity_probability, dtype=np.float32)
    sign = np.asarray(predictions.candidate_sign_probability, dtype=np.float32)
    predicted = np.asarray(predictions.predicted_benefit, dtype=np.float32)
    default = int(dataset.outcome.default_index)
    choice = np.full(dataset.batch.candidate_mask.shape[0], default, dtype=np.int64)
    for sample_index in np.flatnonzero(dataset.valid_sample):
        if opportunity[sample_index] < float(opportunity_threshold):
            continue
        eligible = (
            dataset.legal_candidate[sample_index]
            & (sign[sample_index] >= float(sign_threshold))
            & (predicted[sample_index] > 0.0)
        )
        eligible[default] = False
        candidates = np.flatnonzero(eligible)
        if candidates.size:
            scores = predicted[sample_index, candidates]
            choice[sample_index] = int(candidates[np.argmax(scores)])
    return choice


def evaluate_benefit_choice(
    dataset: BenefitAuditDataset,
    choice: np.ndarray,
) -> dict[str, float | int | None]:
    selected = np.asarray(choice, dtype=np.int64).reshape(-1)
    metrics = aggregate_selected_metrics(dataset.outcome, selected)
    default_choice = np.full_like(selected, dataset.outcome.default_index)
    default_metrics = aggregate_selected_metrics(dataset.outcome, default_choice)
    executed = dataset.valid_sample & (selected != dataset.outcome.default_index)
    rows = np.arange(selected.shape[0])
    selected_benefit = dataset.candidate_benefit[rows, selected]
    executed_count = int(np.sum(executed))
    positive_count = int(np.sum(selected_benefit[executed] > BENEFIT_EPSILON))
    negative_count = int(np.sum(selected_benefit[executed] < -BENEFIT_EPSILON))
    metrics.update(
        {
            "default_active_rate_rmse": default_metrics["active_rate_rmse"],
            "default_link_rmse": default_metrics["link_rmse"],
            "default_activity_f1": default_metrics["activity_f1"],
            "executed_count": executed_count,
            "defer_ratio": float(1.0 - executed_count / max(1, int(np.sum(dataset.valid_sample)))),
            "executed_positive_precision": (
                float(positive_count / executed_count) if executed_count else 0.0
            ),
            "negative_selection_rate": (
                float(negative_count / executed_count) if executed_count else 0.0
            ),
        }
    )
    return metrics


def calibrate_safe_thresholds(
    dataset: BenefitAuditDataset,
    predictions: BenefitPredictions,
    thresholds: tuple[float, ...] = (0.5, 0.65, 0.8, 0.9),
    minimum_positive_precision: float = 0.65,
    maximum_negative_selection_rate: float = 0.20,
) -> SafeThresholdSelection:
    candidates: list[SafeThresholdSelection] = []
    for opportunity_threshold in thresholds:
        for sign_threshold in thresholds:
            choice = select_benefit_candidates(
                dataset,
                predictions,
                opportunity_threshold=opportunity_threshold,
                sign_threshold=sign_threshold,
            )
            metrics = evaluate_benefit_choice(dataset, choice)
            if (
                int(metrics["executed_count"]) > 0
                and float(metrics["executed_positive_precision"]) >= minimum_positive_precision
                and float(metrics["negative_selection_rate"]) <= maximum_negative_selection_rate
            ):
                candidates.append(
                    SafeThresholdSelection(
                        opportunity_threshold=float(opportunity_threshold),
                        sign_threshold=float(sign_threshold),
                        choice=choice,
                        metrics=metrics,
                        status="safe_threshold",
                    )
                )
    if candidates:
        return min(
            candidates,
            key=lambda result: (
                float(result.metrics["active_rate_rmse"]),
                -float(result.metrics["defer_ratio"]),
                result.opportunity_threshold,
                result.sign_threshold,
            ),
        )
    default_choice = np.full(
        dataset.batch.candidate_mask.shape[0], dataset.outcome.default_index, dtype=np.int64
    )
    return SafeThresholdSelection(
        opportunity_threshold=float(max(thresholds)),
        sign_threshold=float(max(thresholds)),
        choice=default_choice,
        metrics=evaluate_benefit_choice(dataset, default_choice),
        status="no_safe_threshold",
    )


def _model_factories(model_kind: str, random_seed: int) -> tuple[Any, Any]:
    kind = str(model_kind).lower()
    if kind == "linear":
        from sklearn.linear_model import Ridge, SGDClassifier

        return (
            SGDClassifier(
                loss="log_loss",
                max_iter=30,
                tol=None,
                class_weight="balanced",
                random_state=int(random_seed),
                average=True,
            ),
            Ridge(alpha=1.0),
        )
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        common = {
            "n_estimators": 160,
            "max_depth": 16,
            "min_samples_leaf": 16,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": int(random_seed),
        }
        return RandomForestClassifier(class_weight="balanced", **common), RandomForestRegressor(
            **common
        )
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        common = {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "random_state": int(random_seed),
        }
        return HistGradientBoostingClassifier(class_weight="balanced", **common), HistGradientBoostingRegressor(
            **common
        )
    if kind == "xgb":
        from xgboost import XGBClassifier, XGBRegressor

        common = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": int(random_seed),
        }
        return XGBClassifier(eval_metric="logloss", **common), XGBRegressor(**common)
    raise ValueError("model_kind must be linear, rf, hgb, or xgb")


def _fit_probability_model(
    features: np.ndarray,
    target: np.ndarray,
    model_kind: str,
    random_seed: int,
) -> tuple[Any, str]:
    labels = np.asarray(target, dtype=bool).astype(np.int64)
    unique = np.unique(labels)
    if unique.size == 1:
        return _ConstantProbability(float(unique[0])), "constant_prior"
    classifier, _ = _model_factories(model_kind, random_seed)
    classifier.fit(features, labels)
    return classifier, "trained"


def fit_benefit_audit_model(
    dataset: BenefitAuditDataset,
    feature_group: BenefitFeatureGroup,
    model_kind: str,
    random_seed: int = 20260718,
) -> FittedBenefitAuditModel:
    valid_indices = np.flatnonzero(dataset.valid_sample)
    opportunity_x = feature_group.opportunity_features
    if opportunity_x.shape[0] != valid_indices.shape[0]:
        raise ValueError("opportunity features must match valid samples")
    normalize = str(model_kind).lower() == "linear"
    opportunity_normalizer = fit_train_normalizer(opportunity_x) if normalize else None
    opportunity_fit_x = (
        opportunity_normalizer.transform(opportunity_x)
        if opportunity_normalizer is not None
        else opportunity_x
    )
    opportunity_model, opportunity_status = _fit_probability_model(
        opportunity_fit_x,
        dataset.opportunity_positive[valid_indices],
        model_kind,
        random_seed,
    )

    if feature_group.candidate_features is None:
        return FittedBenefitAuditModel(
            model_kind=str(model_kind).lower(),
            opportunity_model=opportunity_model,
            sign_model=None,
            benefit_model=None,
            opportunity_normalizer=opportunity_normalizer,
            candidate_normalizer=None,
            opportunity_status=opportunity_status,
            candidate_status="unavailable",
        )

    candidate_x = feature_group.candidate_features
    if candidate_x.shape[0] != dataset.flat_sample_index.shape[0]:
        raise ValueError("candidate features must match flattened legal candidates")
    candidate_normalizer = fit_train_normalizer(candidate_x) if normalize else None
    candidate_fit_x = (
        candidate_normalizer.transform(candidate_x)
        if candidate_normalizer is not None
        else candidate_x
    )
    candidate_target = dataset.candidate_positive[
        dataset.flat_sample_index, dataset.flat_candidate_index
    ]
    sign_model, sign_status = _fit_probability_model(
        candidate_fit_x, candidate_target, model_kind, random_seed
    )
    _, benefit_model = _model_factories(model_kind, random_seed)
    benefit_target = dataset.candidate_benefit[
        dataset.flat_sample_index, dataset.flat_candidate_index
    ]
    if np.allclose(benefit_target, benefit_target[0]):
        benefit_model = _ConstantRegression(float(benefit_target[0]))
        benefit_status = "constant_prior"
    else:
        benefit_model.fit(candidate_fit_x, benefit_target)
        benefit_status = "trained"
    candidate_status = (
        "trained" if "trained" in {sign_status, benefit_status} else "constant_prior"
    )
    return FittedBenefitAuditModel(
        model_kind=str(model_kind).lower(),
        opportunity_model=opportunity_model,
        sign_model=sign_model,
        benefit_model=benefit_model,
        opportunity_normalizer=opportunity_normalizer,
        candidate_normalizer=candidate_normalizer,
        opportunity_status=opportunity_status,
        candidate_status=candidate_status,
    )


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(features), dtype=np.float32)
    if values.shape != (features.shape[0], 2):
        raise ValueError("probability model must return two-class probabilities")
    return values[:, 1]


def predict_benefit_audit_model(
    fitted: FittedBenefitAuditModel,
    dataset: BenefitAuditDataset,
    feature_group: BenefitFeatureGroup,
) -> BenefitPredictions:
    sample_count, candidate_count = dataset.batch.candidate_mask.shape
    valid_indices = np.flatnonzero(dataset.valid_sample)
    opportunity_x = feature_group.opportunity_features
    if fitted.opportunity_normalizer is not None:
        opportunity_x = fitted.opportunity_normalizer.transform(opportunity_x)
    opportunity = np.full(sample_count, np.nan, dtype=np.float32)
    opportunity[valid_indices] = _positive_probability(fitted.opportunity_model, opportunity_x)

    sign = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    benefit = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    if feature_group.candidate_features is None:
        sign[dataset.legal_candidate & dataset.valid_sample[:, None]] = 0.0
        benefit[dataset.legal_candidate & dataset.valid_sample[:, None]] = 0.0
    else:
        candidate_x = feature_group.candidate_features
        if fitted.candidate_normalizer is not None:
            candidate_x = fitted.candidate_normalizer.transform(candidate_x)
        flat_sign = _positive_probability(fitted.sign_model, candidate_x)
        flat_benefit = np.asarray(fitted.benefit_model.predict(candidate_x), dtype=np.float32)
        sign[dataset.flat_sample_index, dataset.flat_candidate_index] = flat_sign
        benefit[dataset.flat_sample_index, dataset.flat_candidate_index] = flat_benefit
    return BenefitPredictions(
        opportunity_probability=opportunity,
        candidate_sign_probability=sign,
        predicted_benefit=benefit,
    )


def _safe_binary_metric(function: Any, target: np.ndarray, score: np.ndarray) -> float | None:
    labels = np.asarray(target, dtype=np.int64)
    if np.unique(labels).size < 2:
        return None
    return float(function(labels, np.asarray(score, dtype=np.float64)))


def _improved_seed_count(dataset: BenefitAuditDataset, choice: np.ndarray) -> int:
    selected = np.asarray(choice, dtype=np.int64)
    rows = np.arange(selected.shape[0])
    improved = 0
    for seed in np.unique(dataset.sample_seed[dataset.valid_sample]):
        keep = dataset.valid_sample & (dataset.sample_seed == seed)
        count = int(np.sum(dataset.outcome.active_count[keep]))
        selected_sse = float(np.sum(dataset.outcome.active_sse[rows[keep], selected[keep]]))
        default_sse = float(
            np.sum(dataset.outcome.active_sse[keep, dataset.outcome.default_index])
        )
        if count > 0 and selected_sse < default_sse:
            improved += 1
    return improved


def evaluate_benefit_predictions(
    dataset: BenefitAuditDataset,
    predictions: BenefitPredictions,
    choice: np.ndarray,
) -> dict[str, float | int | None]:
    """Evaluate identifiability and deployable decision metrics from trace arrays."""
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    _validate_predictions(dataset, predictions)
    valid = dataset.valid_sample
    opportunity_target = dataset.opportunity_positive[valid]
    opportunity_score = predictions.opportunity_probability[valid]
    metrics: dict[str, float | int | None] = {
        "opportunity_roc_auc": _safe_binary_metric(
            roc_auc_score, opportunity_target, opportunity_score
        ),
        "opportunity_pr_auc": float(
            average_precision_score(opportunity_target.astype(np.int64), opportunity_score)
        ),
        "opportunity_brier": float(
            brier_score_loss(opportunity_target.astype(np.int64), opportunity_score)
        ),
    }

    legal = dataset.legal_candidate & valid[:, None]
    sign_target = dataset.candidate_positive[legal]
    sign_score = predictions.candidate_sign_probability[legal]
    metrics["candidate_sign_pr_auc"] = float(
        average_precision_score(sign_target.astype(np.int64), sign_score)
    )
    metrics["candidate_sign_roc_auc"] = _safe_binary_metric(
        roc_auc_score, sign_target, sign_score
    )
    actual_benefit = dataset.candidate_benefit[legal].astype(np.float64)
    predicted_benefit = predictions.predicted_benefit[legal].astype(np.float64)
    if np.unique(actual_benefit).size > 1 and np.unique(predicted_benefit).size > 1:
        metrics["benefit_pearson"] = float(np.corrcoef(actual_benefit, predicted_benefit)[0, 1])
        metrics["benefit_spearman"] = float(spearmanr(actual_benefit, predicted_benefit).statistic)
    else:
        metrics["benefit_pearson"] = None
        metrics["benefit_spearman"] = None

    top1_positive = []
    top1_regret = []
    sample_spearman = []
    default = int(dataset.outcome.default_index)
    for sample_index in np.flatnonzero(valid):
        candidates = np.flatnonzero(dataset.legal_candidate[sample_index])
        scores = predictions.predicted_benefit[sample_index, candidates]
        top_candidate = int(candidates[np.argmax(scores)])
        top_benefit = float(dataset.candidate_benefit[sample_index, top_candidate])
        oracle_benefit = float(np.max(dataset.candidate_benefit[sample_index, candidates]))
        top1_positive.append(top_benefit > BENEFIT_EPSILON)
        top1_regret.append(max(0.0, oracle_benefit - top_benefit))
        actual = dataset.candidate_benefit[sample_index, candidates]
        if (
            candidates.size > 1
            and np.unique(actual).size > 1
            and np.unique(scores).size > 1
        ):
            statistic = float(spearmanr(actual, scores).statistic)
            if np.isfinite(statistic):
                sample_spearman.append(statistic)
    metrics.update(
        {
            "top1_positive_ratio": float(np.mean(top1_positive)) if top1_positive else 0.0,
            "top1_mean_regret": float(np.mean(top1_regret)) if top1_regret else None,
            "sample_rank_spearman": (
                float(np.mean(sample_spearman)) if sample_spearman else None
            ),
        }
    )
    metrics.update(evaluate_benefit_choice(dataset, choice))
    metrics["improved_seed_count"] = _improved_seed_count(dataset, choice)
    metrics["default_candidate_index"] = default
    return metrics
