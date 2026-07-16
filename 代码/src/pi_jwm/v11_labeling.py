"""Candidate actual-rollout label cache for PI-JWM v11 selector experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .v11_selector import CandidateBatch, CandidateOutcome


CACHE_SCHEMA_VERSION = 4
SUPPORTED_CACHE_SCHEMA_VERSIONS = frozenset({1, 2, 3, CACHE_SCHEMA_VERSION})


def compute_rollout_outcome_metrics(
    predictions: dict[str, np.ndarray],
    activity_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Return additive per-sample metrics for one candidate rollout."""
    threshold = float(activity_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("activity_threshold must be in [0, 1]")

    def tensor(name: str) -> np.ndarray:
        values = np.asarray(predictions[name], dtype=np.float64)
        if values.shape[-1:] == (1,):
            values = values[..., 0]
        if values.ndim < 2:
            raise ValueError(f"{name} must have sample and target dimensions")
        return values

    truth_rate = tensor("link_rate_true")
    predicted_rate = tensor("link_rate_pred")
    truth_activity = tensor("link_activity_true") > 0.5
    predicted_activity = tensor("link_activity_prob") >= threshold
    if not (
        truth_rate.shape
        == predicted_rate.shape
        == truth_activity.shape
        == predicted_activity.shape
    ):
        raise ValueError("rollout link tensors must share shape")
    valid = np.isfinite(truth_rate) & np.isfinite(predicted_rate)
    axes = tuple(range(1, truth_rate.ndim))
    error = np.where(valid, (predicted_rate - truth_rate) ** 2, 0.0)
    return {
        "link_sse": np.sum(error, axis=axes).astype(np.float32),
        "link_count": np.sum(valid, axis=axes).astype(np.int64),
        "activity_tp": np.sum(valid & truth_activity & predicted_activity, axis=axes).astype(np.int64),
        "activity_fp": np.sum(valid & ~truth_activity & predicted_activity, axis=axes).astype(np.int64),
        "activity_fn": np.sum(valid & truth_activity & ~predicted_activity, axis=axes).astype(np.int64),
        "activity_tn": np.sum(valid & ~truth_activity & ~predicted_activity, axis=axes).astype(np.int64),
    }


def build_candidate_feature_batch(
    candidate_actions: np.ndarray,
    predictions_by_candidate: list[dict[str, np.ndarray]],
    action_families: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build observable candidate features from actions and PI-JWM predictions."""
    actions = np.asarray(candidate_actions, dtype=np.float32)
    if actions.ndim != 5 or actions.shape[-1] < 6:
        raise ValueError("candidate_actions must be [sample,candidate,step,edge,dim>=6]")
    sample_count, candidate_count, step_count = actions.shape[:3]
    if len(predictions_by_candidate) != candidate_count or len(action_families) != candidate_count:
        raise ValueError("candidate predictions/families must match candidate dimension")
    activity_rows = []
    rate_rows = []
    task_rows = []
    for predictions in predictions_by_candidate:
        activity = np.asarray(predictions["link_activity_prob"], dtype=np.float32)
        rate = np.asarray(predictions["link_rate_pred"], dtype=np.float32)
        task = np.asarray(predictions["task_pred"], dtype=np.float32)
        if activity.shape[-1:] == (1,):
            activity = activity[..., 0]
        if rate.shape[-1:] == (1,):
            rate = rate[..., 0]
        if activity.shape != (sample_count, step_count, actions.shape[3]):
            raise ValueError("link_activity_prob shape must match candidate actions")
        if rate.shape != activity.shape:
            raise ValueError("link_rate_pred shape must match link_activity_prob")
        if task.ndim != 3 or task.shape[:2] != (sample_count, step_count):
            raise ValueError("task_pred must be [sample,step,task_feature]")
        activity_rows.append(activity)
        rate_rows.append(rate)
        task_rows.append(task)
    activity = np.stack(activity_rows, axis=1)
    rate = np.stack(rate_rows, axis=1)
    task = np.stack(task_rows, axis=1)
    rb = np.clip(actions[..., 2], 0.0, None)
    cpu = np.clip(actions[..., 4], 0.0, None)
    activity_delta = activity - activity[:, :1]
    rate_delta = rate - rate[:, :1]
    task_final = task[:, :, -1]
    task_delta = task_final - task_final[:, :1]

    blocks: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (sample_count, candidate_count):
            raise ValueError(f"feature {name} must be [sample,candidate]")
        blocks.append(array[..., None])
        names.append(name)

    add("rb_total_sum", rb.sum(axis=(2, 3)))
    add("rb_total_max", rb.max(axis=(2, 3)))
    add("rb_action_count", (rb > 1e-8).sum(axis=(2, 3)))
    add("cpu_total_sum", cpu.sum(axis=(2, 3)))
    add("cpu_action_count", (cpu > 1e-8).sum(axis=(2, 3)))
    add("offload_action_count", np.clip(actions[..., 0], 0.0, None).sum(axis=(2, 3)))
    add("return_action_count", np.clip(actions[..., 5], 0.0, None).sum(axis=(2, 3)))
    add(
        "predicted_energy_proxy",
        rb.sum(axis=(2, 3)) + cpu.sum(axis=(2, 3)) + 0.1 * np.clip(actions[..., 5], 0.0, None).sum(axis=(2, 3)),
    )
    add("predicted_activity_mean", activity.mean(axis=(2, 3)))
    add("predicted_activity_max", activity.max(axis=(2, 3)))
    add("predicted_activity_mass", activity.sum(axis=(2, 3)))
    clipped_rate = np.clip(rate, 0.0, None)
    add("predicted_rate_sum", clipped_rate.sum(axis=(2, 3)))
    add("predicted_rate_max", clipped_rate.max(axis=(2, 3)))
    add("predicted_throughput_proxy", (activity * clipped_rate).sum(axis=(2, 3)))
    add("predicted_activity_delta_mean", activity_delta.mean(axis=(2, 3)))
    add("predicted_rate_delta_mean", rate_delta.mean(axis=(2, 3)))
    for step in range(step_count):
        add(f"rb_total_step_{step}", rb[:, :, step].sum(axis=2))
        add(f"cpu_total_step_{step}", cpu[:, :, step].sum(axis=2))
    for task_index in range(task_final.shape[2]):
        add(f"predicted_task_{task_index}", task_final[:, :, task_index])
        add(f"predicted_task_delta_{task_index}", task_delta[:, :, task_index])

    family_categories = ("identity", "rb", "offload", "compute", "return", "historical")
    normalized_families = [str(value).lower() for value in action_families]
    for category in family_categories:
        encoded = np.zeros((sample_count, candidate_count), dtype=np.float32)
        for candidate_index, family in enumerate(normalized_families):
            match = family == "identity" if category == "identity" else category in family
            encoded[:, candidate_index] = float(match)
        add(f"action_family_{category}", encoded)
    features = np.concatenate(blocks, axis=2).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("candidate features contain non-finite values")
    return features, tuple(names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".manifest.json")


def save_candidate_label_cache(
    path: str | Path,
    split_name: str,
    sample_ids: np.ndarray,
    sample_seed: np.ndarray,
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    configuration_digest: str | None,
    protocol_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an auditable, self-describing cache beside a SHA-256 manifest."""
    cache_path = Path(path)
    split = str(split_name)
    if split == "matched_test" and not configuration_digest:
        raise PermissionError("matched_test cache requires a frozen configuration digest")
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    sample_count = batch.candidate_features.shape[0]
    if ids.shape[0] != sample_count or seeds.shape[0] != sample_count:
        raise ValueError("sample_ids and sample_seed must match candidate batch")
    if outcome.active_sse.shape != batch.candidate_features.shape[:2]:
        raise ValueError("candidate batch and outcome shapes must match")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_ids=ids,
        sample_seed=seeds,
        context=batch.context,
        candidate_features=batch.candidate_features,
        candidate_mask=batch.candidate_mask,
        stage=batch.stage.astype(str),
        feature_names=np.asarray(batch.feature_names, dtype=str),
        candidate_names=np.asarray(batch.candidate_names, dtype=str),
        active_sse=outcome.active_sse,
        active_count=outcome.active_count,
        link_sse=np.asarray(outcome.link_sse if outcome.link_sse is not None else [], dtype=np.float32),
        link_count=np.asarray(outcome.link_count if outcome.link_count is not None else [], dtype=np.int64),
        activity_tp=np.asarray(outcome.activity_tp if outcome.activity_tp is not None else [], dtype=np.int64),
        activity_fp=np.asarray(outcome.activity_fp if outcome.activity_fp is not None else [], dtype=np.int64),
        activity_fn=np.asarray(outcome.activity_fn if outcome.activity_fn is not None else [], dtype=np.int64),
        activity_tn=np.asarray(outcome.activity_tn if outcome.activity_tn is not None else [], dtype=np.int64),
        action_applied=np.asarray(outcome.action_applied if outcome.action_applied is not None else [], dtype=bool),
        action_applicable=np.asarray(
            outcome.action_applicable if outcome.action_applicable is not None else [], dtype=bool
        ),
        default_index=np.asarray([outcome.default_index], dtype=np.int64),
        task_utility=np.asarray(outcome.task_utility if outcome.task_utility is not None else [], dtype=np.float32),
        energy_total=np.asarray(outcome.energy_total if outcome.energy_total is not None else [], dtype=np.float32),
    )
    digest = _sha256(cache_path)
    manifest = {
        "framework": "PI-JWM",
        "schema_version": CACHE_SCHEMA_VERSION,
        "split_name": split,
        "result_kind": "diagnostic_only",
        "configuration_digest": configuration_digest,
        "protocol_metadata": dict(protocol_metadata or {}),
        "num_samples": int(sample_count),
        "num_candidates": int(batch.candidate_features.shape[1]),
        "feature_names": list(batch.feature_names),
        "candidate_names": list(batch.candidate_names),
        "outcome_fields": [
            name
            for name, value in (
                ("active_sse", outcome.active_sse),
                ("link_sse", outcome.link_sse),
                ("activity_confusion", outcome.activity_tp),
                ("action_applied", outcome.action_applied),
                ("action_applicable", outcome.action_applicable),
                ("task_utility", outcome.task_utility),
                ("energy_total", outcome.energy_total),
            )
            if value is not None
        ],
        "seed_values": sorted(int(value) for value in np.unique(seeds)),
        "cache_sha256": digest,
        "cache_file": cache_path.name,
    }
    _manifest_path(cache_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_candidate_label_cache(
    path: str | Path,
    expected_configuration_digest: str | None = None,
) -> tuple[CandidateBatch, CandidateOutcome, dict[str, Any]]:
    cache_path = Path(path)
    manifest_path = _manifest_path(cache_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
        raise ValueError("unsupported candidate label cache schema")
    actual_sha = _sha256(cache_path)
    if actual_sha != str(manifest.get("cache_sha256")):
        raise ValueError("candidate label cache SHA-256 mismatch")
    if expected_configuration_digest is not None and str(manifest.get("configuration_digest")) != str(
        expected_configuration_digest
    ):
        raise ValueError("candidate label cache configuration digest mismatch")
    with np.load(cache_path, allow_pickle=False) as arrays:
        candidate_names = tuple(str(value) for value in arrays["candidate_names"].tolist())
        batch = CandidateBatch(
            context=arrays["context"],
            candidate_features=arrays["candidate_features"],
            candidate_mask=arrays["candidate_mask"],
            stage=arrays["stage"],
            feature_names=tuple(str(value) for value in arrays["feature_names"].tolist()),
            candidate_names=candidate_names,
        )
        def optional(name: str) -> np.ndarray | None:
            if name not in arrays.files:
                return None
            values = arrays[name]
            return None if values.size == 0 else values

        task_values = optional("task_utility")
        energy_values = optional("energy_total")
        outcome = CandidateOutcome(
            active_sse=arrays["active_sse"],
            active_count=arrays["active_count"],
            link_sse=optional("link_sse"),
            link_count=optional("link_count"),
            activity_tp=optional("activity_tp"),
            activity_fp=optional("activity_fp"),
            activity_fn=optional("activity_fn"),
            activity_tn=optional("activity_tn"),
            action_applied=optional("action_applied"),
            action_applicable=optional("action_applicable"),
            default_index=int(arrays["default_index"][0]),
            task_utility=task_values,
            energy_total=energy_values,
            result_kind=str(manifest.get("result_kind", "diagnostic_only")),
        )
    return batch, outcome, manifest
