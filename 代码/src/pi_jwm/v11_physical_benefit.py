"""Leakage-safe physical-benefit supervision for the PI-JWM v11 selector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_PHYSICAL_FAMILIES = {
    "default": "identity_control",
    "rb_count": "rb_repair",
    "mixed_offload_rb": "offload_rb",
    "cpu_scale": "compute_cpu",
    "return_route": "return_route",
}

COMMON_DESCRIPTOR_NAMES = (
    "rb_total_sum",
    "rb_action_count",
    "cpu_total_sum",
    "cpu_action_count",
    "offload_action_count",
    "return_action_count",
    "intervention_start_step",
    "pattern_persistent",
    "pattern_decayed",
    "family_rb",
    "family_offload",
    "family_compute",
    "family_return",
    "family_control",
)

PHYSICAL_PREDICTION_FEATURES = (
    "physical_task_delta_mean",
    "physical_task_delta_std",
    "physical_task_delta_lcb",
    "physical_task_delta_ucb",
    "physical_energy_delta_mean",
    "physical_energy_delta_std",
    "physical_energy_delta_lcb",
    "physical_energy_delta_ucb",
)


def normalize_physical_family(name: str) -> str | None:
    """Map simulator candidate families to the shared selector vocabulary."""

    return SUPPORTED_PHYSICAL_FAMILIES.get(str(name).strip().lower())


def _time_key(value: Any, tolerance: float) -> float:
    tol = float(tolerance)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("time alignment tolerance must be finite and positive")
    digits = max(0, int(math.ceil(-math.log10(tol))))
    return round(float(value), digits)


def align_decision_points(
    points: Sequence[Mapping[str, Any]],
    sample_index_rows: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach sample IDs only when seed and input-end time match exactly."""

    by_key: dict[tuple[int, float], int] = {}
    for source in sample_index_rows:
        key = (
            int(source["seed"]),
            _time_key(source["input_end_time"], tolerance),
        )
        if key in by_key:
            raise ValueError(f"duplicate sample time for seed={key[0]} time={key[1]}")
        by_key[key] = int(source["sample_id"])

    aligned: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in points:
        row = dict(source)
        key = (
            int(row["seed"]),
            _time_key(row["decision_time"], tolerance),
        )
        sample_id = by_key.get(key)
        if sample_id is None:
            rejected.append({**row, "reason": "no_exact_input_end_time"})
        else:
            aligned.append({**row, "sample_id": sample_id})
    return aligned, rejected


def physical_action_descriptor(
    row: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the common action-only descriptor from an AirFogSim result row."""

    family = normalize_physical_family(str(row.get("action_family", "")))
    if family is None:
        raise ValueError(
            f"unsupported physical action family: {row.get('action_family', '')}"
        )
    pattern = str(row.get("temporal_pattern", "persistent")).strip().lower()
    if pattern not in {"persistent", "decayed"}:
        raise ValueError(f"unsupported temporal pattern: {pattern}")

    values = np.asarray(
        [
            float(row.get("rb_total", row.get("total_rb", 0.0))),
            float(row.get("num_rb_tasks", 0.0)),
            float(row.get("cpu_total", row.get("total_cpu", 0.0))),
            float(row.get("num_cpu_overrides", 0.0)),
            float(row.get("num_offload_overrides", 0.0)),
            float(row.get("num_return_route_overrides", 0.0)),
            float(row.get("intervention_start_step", 1.0)),
            float(pattern == "persistent"),
            float(pattern == "decayed"),
            float(family == "rb_repair"),
            float(family == "offload_rb"),
            float(family == "compute_cpu"),
            float(family == "return_route"),
            float(family == "identity_control"),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("physical action descriptor must be finite")
    return values, COMMON_DESCRIPTOR_NAMES


def selector_action_descriptors(batch: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    """Project selector candidates onto the shared action-only descriptor."""

    source_names = tuple(str(name) for name in batch.feature_names)
    source = np.asarray(batch.candidate_features, dtype=np.float32)
    required = (
        "rb_total_sum",
        "rb_action_count",
        "cpu_total_sum",
        "cpu_action_count",
        "offload_action_count",
        "return_action_count",
        "action_family_identity",
        "action_family_rb",
        "action_family_offload",
        "action_family_compute",
        "action_family_return",
        "action_family_historical",
    )
    missing = [name for name in required if name not in source_names]
    if missing:
        raise ValueError(f"selector batch is missing common descriptor fields: {missing}")

    def field(name: str) -> np.ndarray:
        return source[:, :, source_names.index(name)]

    candidate_names = tuple(str(name).lower() for name in batch.candidate_names)
    if not candidate_names:
        candidate_names = tuple("" for _ in range(source.shape[1]))
    decayed = np.asarray(
        ["decayed" in name for name in candidate_names], dtype=np.float32
    )
    decayed = np.broadcast_to(decayed[None, :], source.shape[:2])
    persistent = 1.0 - decayed
    family_rb = np.maximum(field("action_family_rb"), field("action_family_historical"))

    result = np.stack(
        (
            field("rb_total_sum"),
            field("rb_action_count"),
            field("cpu_total_sum"),
            field("cpu_action_count"),
            field("offload_action_count"),
            field("return_action_count"),
            np.ones(source.shape[:2], dtype=np.float32),
            persistent,
            decayed,
            family_rb,
            field("action_family_offload"),
            field("action_family_compute"),
            field("action_family_return"),
            field("action_family_identity"),
        ),
        axis=2,
    ).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("selector action descriptors must be finite")
    return result, COMMON_DESCRIPTOR_NAMES


@dataclass(frozen=True)
class PhysicalBenefitTrainingBatch:
    """Sparse paired physical outcomes aligned to observable selector context."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    task_delta: np.ndarray
    energy_delta: np.ndarray
    sample_ids: np.ndarray
    sample_seed: np.ndarray
    group_ids: np.ndarray
    normalized_family: np.ndarray
    stage: np.ndarray
    rejected_rows: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        task = np.asarray(self.task_delta, dtype=np.float32).reshape(-1)
        energy = np.asarray(self.energy_delta, dtype=np.float32).reshape(-1)
        sample_ids = np.asarray(self.sample_ids, dtype=np.int64).reshape(-1)
        seeds = np.asarray(self.sample_seed, dtype=np.int64).reshape(-1)
        groups = np.asarray(self.group_ids).astype(str).reshape(-1)
        families = np.asarray(self.normalized_family).astype(str).reshape(-1)
        stages = np.asarray(self.stage).astype(str).reshape(-1)
        size = features.shape[0] if features.ndim == 2 else -1
        if features.ndim != 2 or len(self.feature_names) != features.shape[1]:
            raise ValueError("physical bridge features must be [row,feature] with matching names")
        if any(values.shape[0] != size for values in (task, energy, sample_ids, seeds, groups, families, stages)):
            raise ValueError("physical bridge training fields must share the row dimension")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(task)) or not np.all(np.isfinite(energy)):
            raise ValueError("physical bridge features and targets must be finite")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "task_delta", task)
        object.__setattr__(self, "energy_delta", energy)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "sample_seed", seeds)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "normalized_family", families)
        object.__setattr__(self, "stage", stages)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_physical_training_batch(
    physical_rows: Sequence[Mapping[str, Any]],
    selector_sample_ids: np.ndarray,
    selector_sample_seed: np.ndarray,
    selector_batch: Any,
) -> PhysicalBenefitTrainingBatch:
    """Pair candidates with their group default and attach observable context."""

    ids = np.asarray(selector_sample_ids, dtype=np.int64).reshape(-1)
    seeds = np.asarray(selector_sample_seed, dtype=np.int64).reshape(-1)
    context = np.asarray(selector_batch.context, dtype=np.float32)
    if ids.shape != seeds.shape or ids.shape[0] != context.shape[0]:
        raise ValueError("selector sample ids, seeds, and context must align")
    if len(ids) != len(set(int(value) for value in ids)):
        raise ValueError("selector sample ids must be unique")
    context_names = tuple(str(name) for name in selector_batch.context_feature_names)
    if len(context_names) != context.shape[1]:
        raise ValueError("selector context feature names are required for physical alignment")
    sample_lookup = {
        int(sample_id): (index, int(seed))
        for index, (sample_id, seed) in enumerate(zip(ids, seeds))
    }

    grouped: dict[tuple[int, int, float], list[Mapping[str, Any]]] = {}
    for row in physical_rows:
        key = (
            int(row["seed"]),
            int(row["sample_id"]),
            round(float(row["decision_time"]), 8),
        )
        grouped.setdefault(key, []).append(row)

    feature_rows: list[np.ndarray] = []
    task_delta: list[float] = []
    energy_delta: list[float] = []
    output_ids: list[int] = []
    output_seeds: list[int] = []
    group_ids: list[str] = []
    families: list[str] = []
    stages: list[str] = []
    rejected: list[dict[str, Any]] = []
    for (seed, sample_id, decision_time), rows in sorted(grouped.items()):
        defaults = [row for row in rows if str(row.get("action_family", "")).lower() == "default"]
        if len(defaults) != 1:
            raise ValueError(
                f"physical group {(seed, sample_id, decision_time)} must contain exactly one default"
            )
        if sample_id not in sample_lookup:
            raise ValueError(f"physical sample_id missing from selector cache: {sample_id}")
        context_index, expected_seed = sample_lookup[sample_id]
        if expected_seed != seed:
            raise ValueError(f"physical sample {sample_id} seed differs from selector cache")
        baseline = defaults[0]
        if not _as_bool(baseline.get("action_applied", False)):
            raise ValueError("physical default action must be applied")
        for row in rows:
            if row is baseline:
                continue
            family = normalize_physical_family(str(row.get("action_family", "")))
            if family is None:
                rejected.append(
                    {"candidate_id": str(row.get("candidate_id", "")), "reason": "unsupported_action_family"}
                )
                continue
            if not _as_bool(row.get("action_applied", False)):
                rejected.append(
                    {"candidate_id": str(row.get("candidate_id", "")), "reason": "action_not_applied"}
                )
                continue
            descriptor, descriptor_names = physical_action_descriptor(row)
            feature_rows.append(np.concatenate((context[context_index], descriptor)).astype(np.float32))
            task_delta.append(float(row["task_utility"]) - float(baseline["task_utility"]))
            energy_delta.append(float(row["energy_total"]) - float(baseline["energy_total"]))
            output_ids.append(sample_id)
            output_seeds.append(seed)
            group_ids.append(f"{seed}:{sample_id}:{decision_time:.8f}")
            families.append(family)
            stages.append(str(selector_batch.stage[context_index]))

    feature_names = context_names + descriptor_names if feature_rows else context_names + COMMON_DESCRIPTOR_NAMES
    features = (
        np.stack(feature_rows).astype(np.float32)
        if feature_rows
        else np.empty((0, len(feature_names)), dtype=np.float32)
    )
    return PhysicalBenefitTrainingBatch(
        features=features,
        feature_names=feature_names,
        task_delta=np.asarray(task_delta, dtype=np.float32),
        energy_delta=np.asarray(energy_delta, dtype=np.float32),
        sample_ids=np.asarray(output_ids, dtype=np.int64),
        sample_seed=np.asarray(output_seeds, dtype=np.int64),
        group_ids=np.asarray(group_ids, dtype=str),
        normalized_family=np.asarray(families, dtype=str),
        stage=np.asarray(stages, dtype=str),
        rejected_rows=tuple(rejected),
    )


def audit_physical_bridge_protocol(
    feature_names: Sequence[str],
    split_seed_sets: Mapping[str, Sequence[int]],
    matched_test_accessed: bool,
    external_holdout_accessed: bool,
) -> dict[str, Any]:
    """Audit leakage tokens, seed isolation, and locked-split access."""

    forbidden_tokens = ("actual", "future", "oracle", "outcome", "seed")
    names = tuple(str(name) for name in feature_names)
    forbidden = [
        name for name in names if any(token in name.lower() for token in forbidden_tokens)
    ]
    normalized = {
        str(name): set(int(seed) for seed in values)
        for name, values in split_seed_sets.items()
    }
    overlap: set[int] = set()
    split_names = list(normalized)
    overlap_pairs = []
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            shared = normalized[left] & normalized[right]
            if shared:
                overlap.update(shared)
                overlap_pairs.append({"left": left, "right": right, "seeds": sorted(shared)})
    passed = (
        not forbidden
        and not overlap
        and not bool(matched_test_accessed)
        and not bool(external_holdout_accessed)
    )
    return {
        "passed": passed,
        "forbidden_features": forbidden,
        "split_overlap_count": len(overlap),
        "split_overlap_pairs": overlap_pairs,
        "matched_test_accessed": bool(matched_test_accessed),
        "external_holdout_accessed": bool(external_holdout_accessed),
    }


@dataclass(frozen=True)
class PhysicalBenefitPrediction:
    task_mean: np.ndarray
    task_std: np.ndarray
    task_lcb: np.ndarray
    task_ucb: np.ndarray
    energy_mean: np.ndarray
    energy_std: np.ndarray
    energy_lcb: np.ndarray
    energy_ucb: np.ndarray


@dataclass(frozen=True)
class FittedPhysicalBenefitBridge:
    feature_names: tuple[str, ...]
    task_models: tuple[Any, ...]
    energy_models: tuple[Any, ...]
    task_conformal_radius: float
    energy_conformal_radius: float
    fold_records: tuple[dict[str, Any], ...]
    oof_task_mean: np.ndarray
    oof_energy_mean: np.ndarray
    oof_fold_id: np.ndarray


def _family_stage_median_prediction(
    train_target: np.ndarray,
    train_stage: np.ndarray,
    train_family: np.ndarray,
    target_stage: np.ndarray,
    target_family: np.ndarray,
) -> np.ndarray:
    global_median = float(np.median(train_target))
    medians: dict[tuple[str, str], float] = {}
    for stage in np.unique(train_stage):
        for family in np.unique(train_family):
            mask = (train_stage == stage) & (train_family == family)
            if np.any(mask):
                medians[(str(stage), str(family))] = float(np.median(train_target[mask]))
    return np.asarray(
        [
            medians.get((str(stage), str(family)), global_median)
            for stage, family in zip(target_stage, target_family)
        ],
        dtype=np.float32,
    )


def _conformal_radius(y_true: np.ndarray, y_mean: np.ndarray, coverage: float = 0.9) -> float:
    residual = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_mean, dtype=np.float64))
    if residual.size == 0:
        raise ValueError("conformal calibration requires at least one row")
    rank = int(math.ceil((residual.size + 1) * float(coverage)))
    quantile = min(1.0, rank / residual.size)
    return float(np.quantile(residual, quantile, method="higher"))


def _ensemble_arrays(models: Sequence[Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    stacked = np.stack([model.predict(values) for model in models], axis=0).astype(np.float32)
    return stacked.mean(axis=0), stacked.std(axis=0)


def predict_physical_benefit(
    fitted: FittedPhysicalBenefitBridge,
    features: np.ndarray,
) -> PhysicalBenefitPrediction:
    """Predict calibrated task/energy intervals from observable bridge features."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != len(fitted.feature_names):
        raise ValueError("physical bridge feature dimension mismatch")
    output_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    task_mean, task_std = _ensemble_arrays(fitted.task_models, flat)
    energy_mean, energy_std = _ensemble_arrays(fitted.energy_models, flat)
    task_radius = float(fitted.task_conformal_radius) + 1.64 * task_std
    energy_radius = float(fitted.energy_conformal_radius) + 1.64 * energy_std

    def shaped(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float32).reshape(output_shape)

    return PhysicalBenefitPrediction(
        task_mean=shaped(task_mean),
        task_std=shaped(task_std),
        task_lcb=shaped(task_mean - task_radius),
        task_ucb=shaped(task_mean + task_radius),
        energy_mean=shaped(energy_mean),
        energy_std=shaped(energy_std),
        energy_lcb=shaped(energy_mean - energy_radius),
        energy_ucb=shaped(energy_mean + energy_radius),
    )


def augment_candidate_batch_with_physical_benefit(
    batch: Any,
    fitted: FittedPhysicalBenefitBridge,
    default_index: int,
) -> Any:
    """Return a new candidate batch with calibrated physical delta features."""

    from .v11_selector import CandidateBatch

    duplicate = [name for name in PHYSICAL_PREDICTION_FEATURES if name in batch.feature_names]
    if duplicate:
        raise ValueError(f"candidate batch already contains physical benefit fields: {duplicate}")
    default = int(default_index)
    if not 0 <= default < batch.candidate_features.shape[1]:
        raise ValueError("default_index outside candidate dimension")

    descriptors, descriptor_names = selector_action_descriptors(batch)
    context_names = tuple(str(name) for name in batch.context_feature_names)
    expected_names = context_names + descriptor_names
    if expected_names != fitted.feature_names:
        raise ValueError("candidate batch and physical bridge feature order mismatch")
    context = np.broadcast_to(
        np.asarray(batch.context, dtype=np.float32)[:, None, :],
        batch.candidate_features.shape[:2] + (batch.context.shape[1],),
    )
    bridge_features = np.concatenate((context, descriptors), axis=2).astype(np.float32)
    prediction = predict_physical_benefit(fitted, bridge_features)
    physical = np.stack(
        (
            prediction.task_mean,
            prediction.task_std,
            prediction.task_lcb,
            prediction.task_ucb,
            prediction.energy_mean,
            prediction.energy_std,
            prediction.energy_lcb,
            prediction.energy_ucb,
        ),
        axis=2,
    ).astype(np.float32)
    physical[:, default, :] = 0.0
    return CandidateBatch(
        context=batch.context,
        candidate_features=np.concatenate((batch.candidate_features, physical), axis=2),
        candidate_mask=batch.candidate_mask,
        stage=batch.stage,
        feature_names=tuple(batch.feature_names) + PHYSICAL_PREDICTION_FEATURES,
        candidate_names=tuple(batch.candidate_names),
        context_feature_names=context_names,
    )


def fit_physical_benefit_bridge(
    train: PhysicalBenefitTrainingBatch,
    calibration: PhysicalBenefitTrainingBatch,
) -> tuple[FittedPhysicalBenefitBridge, dict[str, Any]]:
    """Fit fixed five-fold task/energy ensembles and split-conformal intervals."""

    if train.feature_names != calibration.feature_names:
        raise ValueError("physical train/calibration feature order mismatch")
    from sklearn.ensemble import RandomForestRegressor

    from .v11_crossfit import build_seed_crossfit_folds
    from .v11_selector import DEFAULT_SELECTOR_SEEDS

    expected_train = set(DEFAULT_SELECTOR_SEEDS["train"])
    expected_calibration = set(DEFAULT_SELECTOR_SEEDS["calibration"])
    if set(int(value) for value in np.unique(train.sample_seed)) != expected_train:
        raise ValueError("formal physical bridge requires all 40 selector train seeds")
    if set(int(value) for value in np.unique(calibration.sample_seed)) != expected_calibration:
        raise ValueError("formal physical bridge requires all six calibration seeds")

    oof_task = np.full(train.task_delta.shape, np.nan, dtype=np.float32)
    oof_energy = np.full(train.energy_delta.shape, np.nan, dtype=np.float32)
    baseline_task = np.full(train.task_delta.shape, np.nan, dtype=np.float32)
    baseline_energy = np.full(train.energy_delta.shape, np.nan, dtype=np.float32)
    oof_fold = np.full(train.task_delta.shape, -1, dtype=np.int16)
    task_models = []
    energy_models = []
    fold_records = []
    for fold in build_seed_crossfit_folds():
        fit_mask = np.isin(train.sample_seed, fold.helper_train_seeds)
        held_mask = np.isin(train.sample_seed, fold.held_out_seeds)
        if not np.any(fit_mask) or not np.any(held_mask):
            raise ValueError(f"physical bridge fold {fold.fold_id} has no fit or held rows")

        def model(target_id: int) -> Any:
            return RandomForestRegressor(
                n_estimators=128,
                max_depth=8,
                min_samples_leaf=4,
                random_state=1700 + fold.fold_id * 10 + target_id,
                n_jobs=1,
            )

        task_model = model(0).fit(train.features[fit_mask], train.task_delta[fit_mask])
        energy_model = model(1).fit(train.features[fit_mask], train.energy_delta[fit_mask])
        oof_task[held_mask] = task_model.predict(train.features[held_mask]).astype(np.float32)
        oof_energy[held_mask] = energy_model.predict(train.features[held_mask]).astype(np.float32)
        baseline_task[held_mask] = _family_stage_median_prediction(
            train.task_delta[fit_mask],
            train.stage[fit_mask],
            train.normalized_family[fit_mask],
            train.stage[held_mask],
            train.normalized_family[held_mask],
        )
        baseline_energy[held_mask] = _family_stage_median_prediction(
            train.energy_delta[fit_mask],
            train.stage[fit_mask],
            train.normalized_family[fit_mask],
            train.stage[held_mask],
            train.normalized_family[held_mask],
        )
        oof_fold[held_mask] = fold.fold_id
        task_models.append(task_model)
        energy_models.append(energy_model)
        fold_records.append(
            {
                "fold_id": fold.fold_id,
                "held_out_seeds": list(fold.held_out_seeds),
                "model_train_seeds": list(fold.helper_train_seeds),
            }
        )
    if np.any(oof_fold < 0) or not np.all(np.isfinite(oof_task)) or not np.all(np.isfinite(oof_energy)):
        raise RuntimeError("physical bridge OOF coverage is incomplete")

    calibration_task_mean, _ = _ensemble_arrays(task_models, calibration.features)
    calibration_energy_mean, _ = _ensemble_arrays(energy_models, calibration.features)
    task_radius = _conformal_radius(calibration.task_delta, calibration_task_mean)
    energy_radius = _conformal_radius(calibration.energy_delta, calibration_energy_mean)
    fitted = FittedPhysicalBenefitBridge(
        feature_names=train.feature_names,
        task_models=tuple(task_models),
        energy_models=tuple(energy_models),
        task_conformal_radius=task_radius,
        energy_conformal_radius=energy_radius,
        fold_records=tuple(fold_records),
        oof_task_mean=oof_task,
        oof_energy_mean=oof_energy,
        oof_fold_id=oof_fold,
    )
    calibration_prediction = predict_physical_benefit(fitted, calibration.features)
    mae = lambda truth, prediction: float(np.mean(np.abs(truth - prediction)))
    report = {
        "oof_task_mae": mae(train.task_delta, oof_task),
        "oof_energy_mae": mae(train.energy_delta, oof_energy),
        "baseline_task_mae": mae(train.task_delta, baseline_task),
        "baseline_energy_mae": mae(train.energy_delta, baseline_energy),
        "calibration_task_coverage": float(
            np.mean(
                (calibration.task_delta >= calibration_prediction.task_lcb)
                & (calibration.task_delta <= calibration_prediction.task_ucb)
            )
        ),
        "calibration_energy_coverage": float(
            np.mean(
                (calibration.energy_delta >= calibration_prediction.energy_lcb)
                & (calibration.energy_delta <= calibration_prediction.energy_ucb)
            )
        ),
        "task_conformal_radius": task_radius,
        "energy_conformal_radius": energy_radius,
    }
    report["passed"] = bool(
        report["oof_task_mae"] < report["baseline_task_mae"]
        and report["oof_energy_mae"] < report["baseline_energy_mae"]
        and report["calibration_task_coverage"] >= 0.8
        and report["calibration_energy_coverage"] >= 0.8
    )
    return fitted, report
