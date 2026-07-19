"""Leakage-safe candidate ranking primitives for the PI-JWM v11 selector.

The module deliberately contains no AirFogSim execution code.  It defines the
candidate/outcome contracts, split lock, ranking model, and auditable defer
rule used by runnable scripts under ``代码/scripts``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_SELECTOR_SEEDS = {
    "train": tuple(range(0, 16)) + tuple(range(20, 44)),
    "calibration": tuple(range(44, 50)),
    "validation": tuple(range(50, 60)),
    "background": (16, 17),
    "matched_test": (18, 19),
    "external_holdout": tuple(range(60, 70)),
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible protocol payload with a stable representation."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selector_freeze_manifest(manifest: Mapping[str, Any]) -> str:
    payload = manifest.get("selector_freeze_payload")
    expected = str(manifest.get("selector_freeze_digest", ""))
    if not isinstance(payload, Mapping) or len(expected) != 64:
        raise ValueError("selector freeze digest payload is missing")
    actual = canonical_sha256(payload)
    if actual != expected:
        raise ValueError("selector freeze digest mismatch")
    return actual

FORBIDDEN_FEATURE_TOKENS = (
    "true_future",
    "future_truth",
    "actual_future",
    "counterfactual_outcome",
    "oracle",
    "sample_seed",
    "seed_id",
)


def build_selector_split(
    sample_seed: np.ndarray,
    seed_spec: Mapping[str, Sequence[int]] = DEFAULT_SELECTOR_SEEDS,
) -> dict[str, np.ndarray]:
    """Return sample indices for the fixed selector protocol."""
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    split = {
        str(name): np.flatnonzero(np.isin(seeds, np.asarray(values, dtype=np.int64)))
        for name, values in seed_spec.items()
    }
    seed_sets = {name: set(int(value) for value in values) for name, values in seed_spec.items()}
    names = list(seed_sets)
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            overlap = seed_sets[left] & seed_sets[right]
            if overlap:
                raise ValueError(f"selector seed split overlap between {left} and {right}: {sorted(overlap)}")
    return split


class SelectorProtocol:
    """Guard matched-test access until the complete configuration is frozen."""

    def __init__(
        self,
        sample_seed: np.ndarray,
        seed_spec: Mapping[str, Sequence[int]] = DEFAULT_SELECTOR_SEEDS,
    ) -> None:
        self._split = build_selector_split(sample_seed, seed_spec)
        self._configuration_digest: str | None = None

    @property
    def configuration_digest(self) -> str | None:
        return self._configuration_digest

    def freeze_configuration(self, configuration: Mapping[str, Any]) -> str:
        if self._configuration_digest is not None:
            raise RuntimeError("selector configuration is already frozen")
        serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._configuration_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self._configuration_digest

    def indices(self, split_name: str) -> np.ndarray:
        name = str(split_name)
        if name in {"matched_test", "external_holdout"} and self._configuration_digest is None:
            raise PermissionError(f"freeze selector configuration before accessing {name}")
        if name not in self._split:
            raise KeyError(f"unknown selector split: {name}")
        return self._split[name].copy()


def _is_forbidden_feature(name: str) -> bool:
    normalized = str(name).strip().lower()
    return normalized == "seed" or normalized.startswith("y_") or any(
        token in normalized for token in FORBIDDEN_FEATURE_TOKENS
    )


def audit_selector_protocol(
    feature_names: Iterable[str],
    split_seed_sets: Mapping[str, Iterable[int]],
) -> dict[str, Any]:
    """Audit feature leakage and split overlap without inspecting test outcomes."""
    names = tuple(str(name) for name in feature_names)
    forbidden = [name for name in names if _is_forbidden_feature(name)]
    normalized_sets = {
        str(name): set(int(seed) for seed in seeds) for name, seeds in split_seed_sets.items()
    }
    overlap = set()
    split_names = list(normalized_sets)
    overlap_pairs = []
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1 :]:
            shared = normalized_sets[left] & normalized_sets[right]
            if shared:
                overlap.update(shared)
                overlap_pairs.append({"left": left, "right": right, "seeds": sorted(shared)})
    return {
        "num_features": len(names),
        "forbidden_features": forbidden,
        "split_overlap_count": len(overlap),
        "split_overlap_pairs": overlap_pairs,
        "passed": not forbidden and not overlap,
    }


def align_sample_index(path: str | Path, sample_ids: np.ndarray) -> list[dict[str, Any]]:
    """Align source timing metadata to an arbitrary sample-id order."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in sample index: {sample_id}")
        converted: dict[str, Any] = dict(row)
        converted["sample_id"] = sample_id
        if "seed" in converted:
            converted["seed"] = int(converted["seed"])
        for field_name in ("input_end_time", "label_time"):
            if field_name in converted and converted[field_name] != "":
                converted[field_name] = float(converted[field_name])
        by_id[sample_id] = converted
    result = []
    for sample_id in np.asarray(sample_ids, dtype=np.int64).reshape(-1):
        key = int(sample_id)
        if key not in by_id:
            raise KeyError(f"sample_id missing from sample index: {key}")
        result.append(dict(by_id[key]))
    return result


@dataclass(frozen=True)
class CandidateBatch:
    context: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    stage: np.ndarray
    feature_names: tuple[str, ...]
    candidate_names: tuple[str, ...] = ()
    context_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        context = np.asarray(self.context, dtype=np.float32)
        candidate = np.asarray(self.candidate_features, dtype=np.float32)
        mask = np.asarray(self.candidate_mask, dtype=bool)
        stage = np.asarray(self.stage).reshape(-1)
        if candidate.ndim != 3:
            raise ValueError("candidate_features must have shape [sample, candidate, feature]")
        if context.ndim != 2 or context.shape[0] != candidate.shape[0]:
            raise ValueError("context and candidate_features must share sample dimension")
        if mask.shape != candidate.shape[:2]:
            raise ValueError("candidate_mask must match candidate dimensions")
        if stage.shape[0] != candidate.shape[0]:
            raise ValueError("stage must have one value per sample")
        if len(self.feature_names) != candidate.shape[2]:
            raise ValueError("feature_names must match candidate feature dimension")
        if self.candidate_names and len(self.candidate_names) != candidate.shape[1]:
            raise ValueError("candidate_names must match candidate dimension")
        if self.context_feature_names and len(self.context_feature_names) != context.shape[1]:
            raise ValueError("context_feature_names must match context feature dimension")
        if not np.all(np.isfinite(context)) or not np.all(np.isfinite(candidate)):
            raise ValueError("candidate batch features must be finite")
        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("every sample must contain at least one valid candidate")
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "candidate_features", candidate)
        object.__setattr__(self, "candidate_mask", mask)
        object.__setattr__(self, "stage", stage.astype(str))


def ablate_candidate_batch(batch: CandidateBatch, group: str) -> CandidateBatch:
    """Remove one observable feature group while preserving the cache contract."""
    name = str(group).lower()
    if name not in {"stage", "task", "resource", "energy"}:
        raise ValueError("ablation group must be stage, task, resource, or energy")
    features = batch.candidate_features.copy()
    context = batch.context.copy()
    stages = batch.stage.copy()
    selected: list[int] = []
    if name == "stage":
        stages[:] = "unknown"
    else:
        for index, feature_name in enumerate(batch.feature_names):
            normalized = str(feature_name).lower()
            if name == "task":
                match = normalized.startswith("predicted_task_")
            elif name == "energy":
                match = "energy" in normalized
            else:
                match = normalized.startswith(
                    ("rb_", "cpu_", "offload_action_", "return_action_")
                )
            if match:
                selected.append(index)
        if selected:
            features[:, :, selected] = 0.0
            if context.shape[1] == len(batch.feature_names):
                context[:, selected] = 0.0
        if batch.context_feature_names:
            context_selected = []
            for index, feature_name in enumerate(batch.context_feature_names):
                normalized = str(feature_name).lower()
                if name == "task":
                    match = "task" in normalized
                elif name == "energy":
                    match = "energy" in normalized
                else:
                    match = any(token in normalized for token in ("rb", "cpu", "resource", "action"))
                if match:
                    context_selected.append(index)
            if context_selected:
                context[:, context_selected] = 0.0
    return CandidateBatch(
        context=context,
        candidate_features=features,
        candidate_mask=batch.candidate_mask.copy(),
        stage=stages,
        feature_names=batch.feature_names,
        candidate_names=batch.candidate_names,
        context_feature_names=batch.context_feature_names,
    )


@dataclass(frozen=True)
class CandidateOutcome:
    active_sse: np.ndarray
    active_count: np.ndarray
    link_sse: np.ndarray | None = None
    link_count: np.ndarray | None = None
    activity_tp: np.ndarray | None = None
    activity_fp: np.ndarray | None = None
    activity_fn: np.ndarray | None = None
    activity_tn: np.ndarray | None = None
    action_applied: np.ndarray | None = None
    action_applicable: np.ndarray | None = None
    default_index: int = 0
    task_utility: np.ndarray | None = None
    energy_total: np.ndarray | None = None
    result_kind: str = "diagnostic_only"
    sample_rmse: np.ndarray = field(init=False, repr=False)
    improvement: np.ndarray = field(init=False, repr=False)
    regret: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        sse = np.asarray(self.active_sse, dtype=np.float32)
        count = np.asarray(self.active_count, dtype=np.int64).reshape(-1)
        if sse.ndim != 2 or count.shape[0] != sse.shape[0]:
            raise ValueError("active_sse must be [sample, candidate] and match active_count")
        if np.any(sse < 0.0) or not np.all(np.isfinite(sse)):
            raise ValueError("active_sse must be finite and non-negative")
        if np.any(count < 0):
            raise ValueError("active_count must be non-negative")
        default_index = int(self.default_index)
        if not 0 <= default_index < sse.shape[1]:
            raise ValueError("default_index outside candidate dimension")
        valid = count > 0
        mse = np.full_like(sse, np.nan, dtype=np.float32)
        mse[valid] = sse[valid] / count[valid, None]
        sample_rmse = np.sqrt(mse)
        improvement = np.full_like(sse, np.nan, dtype=np.float32)
        regret = np.full_like(sse, np.nan, dtype=np.float32)
        improvement[valid] = sample_rmse[valid, default_index, None] - sample_rmse[valid]
        regret[valid] = sample_rmse[valid] - np.min(sample_rmse[valid], axis=1, keepdims=True)
        candidate_fields = (
            ("link_sse", self.link_sse),
            ("activity_tp", self.activity_tp),
            ("activity_fp", self.activity_fp),
            ("activity_fn", self.activity_fn),
            ("activity_tn", self.activity_tn),
            ("task_utility", self.task_utility),
            ("energy_total", self.energy_total),
        )
        for name, values in candidate_fields:
            if values is not None and np.asarray(values).shape != sse.shape:
                raise ValueError(f"{name} must match active_sse shape")
        if self.link_count is not None:
            link_count = np.asarray(self.link_count, dtype=np.int64).reshape(-1)
            if link_count.shape != count.shape or np.any(link_count < 0):
                raise ValueError("link_count must match active_count and be non-negative")
            object.__setattr__(self, "link_count", link_count)
        if (self.link_sse is None) != (self.link_count is None):
            raise ValueError("link_sse and link_count must be provided together")
        activity_fields = (self.activity_tp, self.activity_fp, self.activity_fn, self.activity_tn)
        if any(value is not None for value in activity_fields) and not all(
            value is not None for value in activity_fields
        ):
            raise ValueError("all activity confusion fields must be provided together")
        if self.action_applied is not None:
            applied = np.asarray(self.action_applied, dtype=bool)
            if applied.shape != sse.shape:
                raise ValueError("action_applied must match active_sse shape")
            object.__setattr__(self, "action_applied", applied)
        if self.action_applicable is not None:
            applicable = np.asarray(self.action_applicable, dtype=bool)
            if applicable.shape != sse.shape:
                raise ValueError("action_applicable must match active_sse shape")
            object.__setattr__(self, "action_applicable", applicable)
        object.__setattr__(self, "active_sse", sse)
        object.__setattr__(self, "active_count", count)
        object.__setattr__(self, "default_index", default_index)
        object.__setattr__(self, "sample_rmse", sample_rmse)
        object.__setattr__(self, "improvement", improvement)
        object.__setattr__(self, "regret", regret)
        for name, values in candidate_fields:
            if values is None:
                continue
            dtype = np.float32 if name in {"link_sse", "task_utility", "energy_total"} else np.int64
            array = np.asarray(values, dtype=dtype)
            if np.any(array < 0) and name not in {"task_utility"}:
                raise ValueError(f"{name} must be non-negative")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, array)


def aggregate_selected_metrics(
    outcome: CandidateOutcome,
    choice: np.ndarray,
) -> dict[str, float | None]:
    """Reconstruct deployable metrics from per-sample selected outcomes."""
    selected = np.asarray(choice, dtype=np.int64).reshape(-1)
    sample_count, candidate_count = outcome.active_sse.shape
    if selected.shape[0] != sample_count:
        raise ValueError("choice must contain one candidate index per sample")
    if np.any(selected < 0) or np.any(selected >= candidate_count):
        raise ValueError("choice contains an invalid candidate index")
    rows = np.arange(sample_count)
    active_targets = int(np.sum(outcome.active_count))
    active_sse = float(np.sum(outcome.active_sse[rows, selected]))
    metrics: dict[str, float | None] = {
        "active_rate_rmse": float(np.sqrt(active_sse / active_targets)) if active_targets else None,
        "link_rmse": None,
        "activity_f1": None,
    }
    if outcome.link_sse is not None and outcome.link_count is not None:
        link_targets = int(np.sum(outcome.link_count))
        link_sse = float(np.sum(outcome.link_sse[rows, selected]))
        metrics["link_rmse"] = float(np.sqrt(link_sse / link_targets)) if link_targets else None
    if outcome.activity_tp is not None:
        tp = int(np.sum(outcome.activity_tp[rows, selected]))
        fp = int(np.sum(outcome.activity_fp[rows, selected]))
        fn = int(np.sum(outcome.activity_fn[rows, selected]))
        denominator = 2 * tp + fp + fn
        metrics["activity_f1"] = float(2 * tp / denominator) if denominator else 0.0
    return metrics


def observable_pareto_deltas(
    batch: CandidateBatch,
    default_index: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build test-time task/energy trade-offs from observable forecast features."""
    names = list(batch.feature_names)
    physical_task_name = "physical_task_delta_lcb"
    physical_energy_name = "physical_energy_delta_ucb"
    if physical_task_name in names and physical_energy_name in names:
        return (
            batch.candidate_features[:, :, names.index(physical_task_name)].astype(np.float32),
            batch.candidate_features[:, :, names.index(physical_energy_name)].astype(np.float32),
        )
    if physical_task_name in names and "predicted_energy_proxy" in names:
        default = int(default_index)
        energy = batch.candidate_features[
            :, :, names.index("predicted_energy_proxy")
        ].astype(np.float32)
        return (
            batch.candidate_features[:, :, names.index(physical_task_name)].astype(
                np.float32
            ),
            energy - energy[:, default : default + 1],
        )
    task_name = "predicted_task_delta_8"
    energy_name = "predicted_energy_proxy"
    if task_name not in names or energy_name not in names:
        return None, None
    default = int(default_index)
    if not 0 <= default < batch.candidate_features.shape[1]:
        raise ValueError("default_index outside candidate dimension")
    task = batch.candidate_features[:, :, names.index(task_name)].astype(np.float32)
    energy = batch.candidate_features[:, :, names.index(energy_name)].astype(np.float32)
    return (
        task - task[:, default : default + 1],
        energy - energy[:, default : default + 1],
    )


@dataclass(frozen=True)
class SelectorDecision:
    candidate_index: np.ndarray
    predicted_improvement: np.ndarray
    uncertainty: np.ndarray
    deferred: np.ndarray
    defer_reason: tuple[str, ...]

    def to_records(self, sample_ids: Sequence[int] | None = None) -> list[dict[str, Any]]:
        size = int(np.asarray(self.candidate_index).shape[0])
        ids = list(range(size)) if sample_ids is None else [int(value) for value in sample_ids]
        if len(ids) != size:
            raise ValueError("sample_ids must match decision count")
        return [
            {
                "sample_id": ids[index],
                "candidate_index": int(self.candidate_index[index]),
                "predicted_improvement": float(self.predicted_improvement[index]),
                "uncertainty": float(self.uncertainty[index]),
                "deferred": bool(self.deferred[index]),
                "defer_reason": str(self.defer_reason[index]),
            }
            for index in range(size)
        ]


def project_candidate_actions(
    actions: np.ndarray,
    baseline_actions: np.ndarray,
    valid_element_mask: np.ndarray,
    candidate_families: Sequence[str],
    stages: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Project candidate actions to resource and task-stage constraints."""
    projected = np.asarray(actions, dtype=np.float32).copy()
    baseline = np.asarray(baseline_actions, dtype=np.float32)
    valid = np.asarray(valid_element_mask, dtype=bool)
    if projected.ndim != 5 or baseline.shape != (projected.shape[0], *projected.shape[2:]):
        raise ValueError("actions must be [sample, candidate, step, element, dim]")
    if projected.shape[-1] < 6:
        raise ValueError("PI-JWM action vectors require at least six dimensions")
    if valid.shape != projected.shape[:1] + projected.shape[2:4]:
        raise ValueError("valid_element_mask must be [sample, step, element]")
    if len(candidate_families) != projected.shape[1]:
        raise ValueError("candidate_families must match candidate dimension")
    stage_values = np.asarray(stages).astype(str).reshape(-1)
    if stage_values.shape[0] != projected.shape[0]:
        raise ValueError("stages must match sample dimension")
    projected = np.clip(projected, 0.0, None)
    projected *= valid[:, None, :, :, None]
    for count_dim, total_dim in ((1, 2), (3, 4)):
        consistent = (projected[..., count_dim] > 0.0) & (projected[..., total_dim] > 0.0)
        projected[..., count_dim] = np.where(consistent, projected[..., count_dim], 0.0)
        projected[..., total_dim] = np.where(consistent, projected[..., total_dim], 0.0)
    for candidate_index, family in enumerate(candidate_families):
        normalized = str(family).lower()
        required_stage = "return" if "return" in normalized else "compute" if "compute" in normalized or "cpu" in normalized else None
        if required_stage is None:
            continue
        invalid_samples = stage_values != required_stage
        projected[invalid_samples, candidate_index] = baseline[invalid_samples]
    applied = np.any(np.abs(projected - baseline[:, None]) > 1e-8, axis=(2, 3, 4))
    return projected.astype(np.float32), applied


def audit_candidate_library(
    active_sse: np.ndarray,
    active_count: np.ndarray,
    action_applied: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    applicability_mask: np.ndarray | None = None,
    identity_index: int = 0,
    oracle_rmse_threshold: float = 190.0,
    min_nontrivial_ratio: float = 0.70,
    max_identity_win_ratio: float = 0.65,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    sse = np.asarray(active_sse, dtype=np.float64)
    count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    applied = np.asarray(action_applied, dtype=bool)
    available = np.ones_like(applied) if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    applicable = available.copy() if applicability_mask is None else np.asarray(applicability_mask, dtype=bool)
    if (
        sse.ndim != 2
        or count.shape[0] != sse.shape[0]
        or applied.shape != sse.shape
        or available.shape != sse.shape
        or applicable.shape != sse.shape
    ):
        raise ValueError("active_sse, active_count, and action_applied shapes are inconsistent")
    if not 0 <= int(identity_index) < sse.shape[1]:
        raise ValueError("identity_index outside candidate dimension")
    valid = count > 0
    if not np.any(valid):
        return {
            "sample_oracle_rmse": None,
            "nontrivial_ratio": 0.0,
            "identity_oracle_win_ratio": None,
            "action_applied_ratio": None,
            "num_valid_samples": 0,
            "failure_reason": "no_active_targets",
            "passed": False,
        }
    if np.any(available[valid].sum(axis=1) == 0):
        raise ValueError("each active sample requires at least one available candidate")
    masked_sse = np.where(available[valid], sse[valid], np.inf)
    oracle_sse = np.min(masked_sse, axis=1)
    oracle_rmse = math.sqrt(float(np.sum(oracle_sse)) / float(np.sum(count[valid])))
    maximum = np.max(np.where(available[valid], sse[valid], -np.inf), axis=1)
    spread = maximum - oracle_sse
    nontrivial = spread > float(tolerance)
    identity_is_oracle = sse[valid, int(identity_index)] <= oracle_sse + float(tolerance)
    non_identity = np.ones((sse.shape[1],), dtype=bool)
    non_identity[int(identity_index)] = False
    eligible_actions = applicable[valid][:, non_identity]
    applied_values = applied[valid][:, non_identity][eligible_actions]
    applied_ratio = float(np.mean(applied_values)) if applied_values.size else 1.0
    report = {
        "sample_oracle_rmse": oracle_rmse,
        "nontrivial_ratio": float(np.mean(nontrivial)),
        "identity_oracle_win_ratio": float(np.mean(identity_is_oracle)),
        "action_applied_ratio": applied_ratio,
        "num_valid_samples": int(np.sum(valid)),
    }
    report["passed"] = bool(
        report["sample_oracle_rmse"] < float(oracle_rmse_threshold)
        and report["nontrivial_ratio"] >= float(min_nontrivial_ratio)
        and report["identity_oracle_win_ratio"] < float(max_identity_win_ratio)
        and math.isclose(report["action_applied_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-12)
    )
    return report


class CandidateSetBenefitRanker(nn.Module):
    """Permutation-equivariant DeepSets ranker with improvement/risk heads."""

    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(int(candidate_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(nn.Linear(int(context_dim), hidden), nn.ReLU())
        self.stage_embedding = nn.Embedding(int(num_stages), hidden)
        joint_dim = hidden * 4
        self.score_head = nn.Sequential(nn.Linear(joint_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.improvement_head = nn.Sequential(nn.Linear(joint_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(joint_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        candidate_features: torch.Tensor,
        context: torch.Tensor,
        candidate_mask: torch.Tensor,
        stage_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_features.ndim != 3 or context.ndim != 2:
            raise ValueError("ranker inputs must be [sample,candidate,feature] and [sample,context]")
        mask = candidate_mask.to(dtype=torch.bool)
        encoded = self.candidate_encoder(candidate_features)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = torch.sum(encoded * weights, dim=1) / torch.clamp(torch.sum(weights, dim=1), min=1.0)
        context_encoded = self.context_encoder(context)
        if stage_ids is None:
            stage_ids = torch.zeros((candidate_features.shape[0],), dtype=torch.long, device=candidate_features.device)
        stage_encoded = self.stage_embedding(stage_ids.to(dtype=torch.long))
        candidate_count = candidate_features.shape[1]
        joint = torch.cat(
            [
                encoded,
                pooled[:, None, :].expand(-1, candidate_count, -1),
                context_encoded[:, None, :].expand(-1, candidate_count, -1),
                stage_encoded[:, None, :].expand(-1, candidate_count, -1),
            ],
            dim=-1,
        )
        score = self.score_head(joint).squeeze(-1)
        improvement = self.improvement_head(joint).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(joint).squeeze(-1)) + 1e-6
        score = score.masked_fill(~mask, -1e9)
        improvement = improvement.masked_fill(~mask, 0.0)
        uncertainty = uncertainty.masked_fill(~mask, 0.0)
        return {"score": score, "predicted_improvement": improvement, "uncertainty": uncertainty}


def listwise_regret_loss(
    scores: torch.Tensor,
    regret: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float = 0.25,
) -> torch.Tensor:
    if scores.shape != regret.shape or scores.shape != candidate_mask.shape:
        raise ValueError("scores, regret, and candidate_mask must share shape")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    mask = candidate_mask.to(dtype=torch.bool)
    valid_rows = mask.any(dim=1) & torch.isfinite(regret.masked_fill(~mask, 0.0)).all(dim=1)
    if not bool(valid_rows.any()):
        raise ValueError("listwise loss requires a valid candidate row")
    masked_regret = regret.masked_fill(~mask, float("inf"))
    target_logits = (-masked_regret / float(temperature)).masked_fill(~mask, -1e9)
    predicted_logits = scores.masked_fill(~mask, -1e9)
    target = torch.softmax(target_logits[valid_rows], dim=1)
    log_prob = torch.log_softmax(predicted_logits[valid_rows], dim=1)
    return -(target * log_prob).sum(dim=1).mean()


def _pairwise_margin_loss(
    scores: torch.Tensor,
    regret: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    regret_gap = regret[:, :, None] - regret[:, None, :]
    score_gap = scores[:, :, None] - scores[:, None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :] & (torch.abs(regret_gap) > 1e-8)
    if not bool(pair_mask.any()):
        return scores.sum() * 0.0
    desired = -torch.sign(regret_gap)
    return F.relu(float(margin) - desired * score_gap)[pair_mask].mean()


def _stage_ids(stages: np.ndarray) -> torch.Tensor:
    vocabulary = {"unknown": 0, "offload": 1, "compute": 2, "return": 3}
    return torch.tensor([vocabulary.get(str(value).lower(), 0) for value in stages], dtype=torch.long)


@dataclass
class FittedSelector:
    model: CandidateSetBenefitRanker
    history: list[dict[str, float]]
    stage_vocabulary: dict[str, int]
    target_scale: float = 1.0
    candidate_mean: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    candidate_scale: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    context_mean: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    context_scale: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))


def load_fitted_selector_checkpoint(
    path: str | Path,
    expected_configuration_digest: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[FittedSelector, float, dict[str, Any]]:
    """Restore a frozen selector and verify its experiment configuration."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    required = {
        "state_dict",
        "candidate_dim",
        "context_dim",
        "hidden_dim",
        "dropout",
        "calibration_bias",
        "configuration_digest",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"selector checkpoint missing fields: {missing}")
    digest = str(payload["configuration_digest"])
    if expected_configuration_digest is not None and digest != str(expected_configuration_digest):
        raise ValueError("selector checkpoint configuration digest mismatch")
    model = CandidateSetBenefitRanker(
        candidate_dim=int(payload["candidate_dim"]),
        context_dim=int(payload["context_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
    ).to(torch.device(device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    def normalization_array(name: str, dimension: int, fill: float) -> np.ndarray:
        value = payload.get(name)
        if value is None:
            return np.full((dimension,), fill, dtype=np.float32)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (dimension,):
            raise ValueError(f"checkpoint {name} dimension mismatch")
        return array

    fitted = FittedSelector(
        model=model,
        history=list(payload.get("history", [])),
        stage_vocabulary={"unknown": 0, "offload": 1, "compute": 2, "return": 3},
        target_scale=float(payload.get("target_scale", 1.0)),
        candidate_mean=normalization_array("candidate_mean", int(payload["candidate_dim"]), 0.0),
        candidate_scale=normalization_array("candidate_scale", int(payload["candidate_dim"]), 1.0),
        context_mean=normalization_array("context_mean", int(payload["context_dim"]), 0.0),
        context_scale=normalization_array("context_scale", int(payload["context_dim"]), 1.0),
    )
    metadata = {
        key: payload[key]
        for key in (
            "candidate_dim",
            "context_dim",
            "hidden_dim",
            "dropout",
            "temperature",
            "training_seed",
            "configuration_digest",
        )
        if key in payload
    }
    return fitted, float(payload["calibration_bias"]), metadata


def fit_listwise_selector(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    hidden_dim: int = 64,
    temperature: float = 0.25,
    dropout: float = 0.0,
    epochs: int = 20,
    learning_rate: float = 3e-3,
    seed: int = 17,
    device: str | torch.device = "cpu",
    group_ids: np.ndarray | None = None,
) -> FittedSelector:
    """Fit the fixed composite listwise objective deterministically on one split."""
    if batch.candidate_features.shape[:2] != outcome.active_sse.shape:
        raise ValueError("candidate batch and outcome shapes must match")
    external_groups = None
    if group_ids is not None:
        external_groups = np.asarray(group_ids).reshape(-1)
        if external_groups.shape[0] != batch.candidate_features.shape[0]:
            raise ValueError("group_ids must contain one group per sample")
    audit = audit_selector_protocol(batch.feature_names, {"fit": {0}})
    if not audit["passed"]:
        raise ValueError(f"selector feature protocol failed: {audit['forbidden_features']}")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = CandidateSetBenefitRanker(
        candidate_dim=batch.candidate_features.shape[2],
        context_dim=batch.context.shape[1],
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
    model = model.to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    valid_candidate = batch.candidate_features[batch.candidate_mask]
    candidate_mean = valid_candidate.mean(axis=0).astype(np.float32)
    candidate_scale = valid_candidate.std(axis=0).astype(np.float32)
    candidate_scale = np.where(candidate_scale < 1e-6, 1.0, candidate_scale).astype(np.float32)
    context_mean = batch.context.mean(axis=0).astype(np.float32)
    context_scale = batch.context.std(axis=0).astype(np.float32)
    context_scale = np.where(context_scale < 1e-6, 1.0, context_scale).astype(np.float32)
    normalized_candidate = np.clip(
        (batch.candidate_features - candidate_mean) / candidate_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    normalized_context = np.clip(
        (batch.context - context_mean) / context_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    candidate = torch.from_numpy(normalized_candidate).to(selected_device)
    context = torch.from_numpy(normalized_context).to(selected_device)
    mask = torch.from_numpy(batch.candidate_mask).to(selected_device)
    stage_ids = _stage_ids(batch.stage).to(selected_device)
    valid_numpy = batch.candidate_mask & (outcome.active_count > 0)[:, None]
    positive_regret = outcome.regret[valid_numpy & np.isfinite(outcome.regret) & (outcome.regret > 1e-8)]
    target_scale = float(np.median(positive_regret)) if positive_regret.size else 1.0
    target_scale = max(target_scale, 1e-6)
    regret = torch.from_numpy(outcome.regret / target_scale).to(selected_device)
    improvement = torch.from_numpy(outcome.improvement / target_scale).to(selected_device)
    valid_rows = torch.from_numpy(outcome.active_count > 0).to(selected_device)
    history = []
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(candidate, context, mask, stage_ids)
        valid_mask = mask & valid_rows[:, None]
        listwise = listwise_regret_loss(outputs["score"], regret, valid_mask, temperature)
        pairwise = _pairwise_margin_loss(outputs["score"], regret, valid_mask)
        residual = outputs["predicted_improvement"][valid_mask] - improvement[valid_mask]
        sigma = outputs["uncertainty"][valid_mask]
        calibration = torch.mean(0.5 * torch.square(residual / sigma) + torch.log(sigma))
        per_sample = []
        for stage_value in np.unique(batch.stage):
            group = torch.from_numpy((batch.stage == stage_value) & (outcome.active_count > 0)).to(selected_device)
            if bool(group.any()):
                group_mask = mask & group[:, None]
                per_sample.append(
                    listwise_regret_loss(outputs["score"], regret, group_mask, temperature)
                )
        if external_groups is not None:
            for group_value in np.unique(external_groups):
                group = torch.from_numpy(
                    (external_groups == group_value) & (outcome.active_count > 0)
                ).to(selected_device)
                if bool(group.any()):
                    group_mask = mask & group[:, None]
                    per_sample.append(
                        listwise_regret_loss(outputs["score"], regret, group_mask, temperature)
                    )
        worst_group = torch.stack(per_sample).max() if per_sample else listwise * 0.0
        loss = listwise + 0.25 * pairwise + 0.10 * worst_group + 0.05 * calibration
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(loss.detach()),
                "listwise": float(listwise.detach()),
                "pairwise": float(pairwise.detach()),
                "worst_group": float(worst_group.detach()),
                "calibration": float(calibration.detach()),
            }
        )
    return FittedSelector(
        model=model,
        history=history,
        stage_vocabulary={"unknown": 0, "offload": 1, "compute": 2, "return": 3},
        target_scale=target_scale,
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        context_mean=context_mean,
        context_scale=context_scale,
    )


def predict_fitted_selector(
    fitted: FittedSelector,
    batch: CandidateBatch,
) -> dict[str, np.ndarray]:
    """Run a fitted selector without exposing training outcomes."""
    fitted.model.eval()
    device = next(fitted.model.parameters()).device
    candidate_mean = (
        fitted.candidate_mean
        if fitted.candidate_mean.size
        else np.zeros((batch.candidate_features.shape[2],), dtype=np.float32)
    )
    candidate_scale = (
        fitted.candidate_scale
        if fitted.candidate_scale.size
        else np.ones((batch.candidate_features.shape[2],), dtype=np.float32)
    )
    context_mean = (
        fitted.context_mean
        if fitted.context_mean.size
        else np.zeros((batch.context.shape[1],), dtype=np.float32)
    )
    context_scale = (
        fitted.context_scale
        if fitted.context_scale.size
        else np.ones((batch.context.shape[1],), dtype=np.float32)
    )
    if candidate_mean.shape != (batch.candidate_features.shape[2],) or context_mean.shape != (
        batch.context.shape[1],
    ):
        raise ValueError("selector normalization statistics do not match batch dimensions")
    candidate = np.clip(
        (batch.candidate_features - candidate_mean) / candidate_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    context = np.clip((batch.context - context_mean) / context_scale, -10.0, 10.0).astype(np.float32)
    with torch.no_grad():
        outputs = fitted.model(
            torch.from_numpy(candidate).to(device),
            torch.from_numpy(context).to(device),
            torch.from_numpy(batch.candidate_mask).to(device),
            _stage_ids(batch.stage).to(device),
        )
    return {name: value.cpu().numpy().astype(np.float32) for name, value in outputs.items()}


def _pareto_dominated(task: np.ndarray, energy: np.ndarray) -> np.ndarray:
    candidate_count = task.shape[0]
    dominated = np.zeros((candidate_count,), dtype=bool)
    for candidate in range(candidate_count):
        no_worse = (task >= task[candidate]) & (energy <= energy[candidate])
        strictly_better = (task > task[candidate]) | (energy < energy[candidate])
        dominated[candidate] = bool(np.any(no_worse & strictly_better))
    return dominated


def select_with_defer(
    ensemble_rank_score: np.ndarray,
    default_index: int,
    ensemble_improvement: np.ndarray | None = None,
    ensemble_uncertainty: np.ndarray | None = None,
    task_delta: np.ndarray | None = None,
    energy_delta: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
    z_value: float = 1.64,
) -> SelectorDecision:
    """Choose the best safe candidate or defer to the ranked baseline."""
    rank_predictions = np.asarray(ensemble_rank_score, dtype=np.float64)
    if rank_predictions.ndim != 3 or rank_predictions.shape[0] < 1:
        raise ValueError("ensemble_rank_score must be [model, sample, candidate]")
    if not np.all(np.isfinite(rank_predictions)):
        raise ValueError("ensemble_rank_score must be finite")
    improvement_predictions = (
        rank_predictions
        if ensemble_improvement is None
        else np.asarray(ensemble_improvement, dtype=np.float64)
    )
    if improvement_predictions.shape != rank_predictions.shape or not np.all(
        np.isfinite(improvement_predictions)
    ):
        raise ValueError("ensemble_improvement must be finite and match ensemble_rank_score")
    predicted_uncertainty = None
    if ensemble_uncertainty is not None:
        predicted_uncertainty = np.asarray(ensemble_uncertainty, dtype=np.float64)
        if predicted_uncertainty.shape != rank_predictions.shape:
            raise ValueError("ensemble_uncertainty must match ensemble_rank_score")
        if np.any(predicted_uncertainty < 0.0) or not np.all(np.isfinite(predicted_uncertainty)):
            raise ValueError("ensemble_uncertainty must be finite and non-negative")
    _, sample_count, candidate_count = rank_predictions.shape
    default = int(default_index)
    if not 0 <= default < candidate_count:
        raise ValueError("default_index outside candidate dimension")
    mask = np.ones((sample_count, candidate_count), dtype=bool) if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    if mask.shape != (sample_count, candidate_count):
        raise ValueError("candidate_mask must match ensemble candidate dimensions")
    if not np.all(mask[:, default]):
        raise ValueError("default candidate must be valid for every sample")
    if (task_delta is None) != (energy_delta is None):
        raise ValueError("task_delta and energy_delta must be supplied together")
    if task_delta is not None:
        task = np.asarray(task_delta, dtype=np.float64)
        energy = np.asarray(energy_delta, dtype=np.float64)
        if task.shape != mask.shape or energy.shape != mask.shape:
            raise ValueError("task_delta and energy_delta must match candidate dimensions")
    else:
        task = energy = None
    rank_mean = rank_predictions.mean(axis=0)
    improvement_mean = improvement_predictions.mean(axis=0)
    epistemic_std = improvement_predictions.std(axis=0, ddof=0)
    if predicted_uncertainty is None:
        total_std = epistemic_std
    else:
        aleatoric_variance = np.mean(np.square(predicted_uncertainty), axis=0)
        total_std = np.sqrt(np.square(epistemic_std) + aleatoric_variance)
    chosen = np.full((sample_count,), default, dtype=np.int64)
    chosen_mean = np.zeros((sample_count,), dtype=np.float64)
    chosen_std = np.zeros((sample_count,), dtype=np.float64)
    deferred = np.ones((sample_count,), dtype=bool)
    reasons = []
    for sample in range(sample_count):
        allowed = mask[sample].copy()
        if task is not None and energy is not None:
            allowed &= ~_pareto_dominated(task[sample], energy[sample])
        allowed[default] = True
        scores = np.where(allowed, rank_mean[sample], -np.inf)
        candidate = int(np.argmax(scores))
        lower_bound = float(
            improvement_mean[sample, candidate] - float(z_value) * total_std[sample, candidate]
        )
        if candidate != default and lower_bound > 0.0:
            chosen[sample] = candidate
            chosen_mean[sample] = improvement_mean[sample, candidate]
            chosen_std[sample] = total_std[sample, candidate]
            deferred[sample] = False
            reasons.append("")
        else:
            chosen_mean[sample] = improvement_mean[sample, default]
            chosen_std[sample] = total_std[sample, default]
            reasons.append("nonpositive_lower_confidence_bound")
    return SelectorDecision(
        candidate_index=chosen,
        predicted_improvement=chosen_mean.astype(np.float32),
        uncertainty=chosen_std.astype(np.float32),
        deferred=deferred,
        defer_reason=tuple(reasons),
    )
