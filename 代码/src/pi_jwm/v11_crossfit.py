"""Leakage-safe helper cross-fitting for PI-JWM v11 candidate labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .v11_selector import DEFAULT_SELECTOR_SEEDS, canonical_sha256


@dataclass(frozen=True)
class SeedCrossfitFold:
    """One held-out seed fold and the helper-training complement."""

    fold_id: int
    held_out_seeds: tuple[int, ...]
    helper_train_seeds: tuple[int, ...]


@dataclass(frozen=True)
class CrossfitExecution:
    """Resolved helper-training and label sample indices for one invocation."""

    mode: str
    fold_id: int | None
    held_out_seeds: tuple[int, ...]
    helper_train_seeds: tuple[int, ...]
    helper_train_indices: np.ndarray
    label_indices: dict[str, np.ndarray]


@dataclass(frozen=True)
class LoadedFoldCache:
    """One integrity-checked schema-v6 fold cache ready for merging."""

    path: Path
    sample_ids: np.ndarray
    sample_seed: np.ndarray
    fold_ids: np.ndarray
    batch: Any
    outcome: Any
    interactions: Any
    manifest: dict[str, Any]


def build_seed_crossfit_folds(
    train_seeds: Sequence[int] = DEFAULT_SELECTOR_SEEDS["train"],
    num_folds: int = 5,
) -> tuple[SeedCrossfitFold, ...]:
    """Build the frozen five-fold round-robin selector-train protocol."""
    ordered = tuple(sorted(int(seed) for seed in train_seeds))
    if len(ordered) != len(set(ordered)) or int(num_folds) != 5 or len(ordered) != 40:
        raise ValueError(
            "formal seed crossfit requires 40 unique train seeds and five folds"
        )
    folds = []
    for fold_id in range(int(num_folds)):
        held_out = ordered[fold_id:: int(num_folds)]
        held_out_set = set(held_out)
        helper_train = tuple(seed for seed in ordered if seed not in held_out_set)
        folds.append(SeedCrossfitFold(fold_id, held_out, helper_train))
    return tuple(folds)


def audit_seed_crossfit_folds(
    folds: Sequence[SeedCrossfitFold],
    seed_spec: Mapping[str, Sequence[int]] = DEFAULT_SELECTOR_SEEDS,
) -> dict[str, Any]:
    """Audit coverage, isolation, and locked-split exclusion."""
    train = set(int(seed) for seed in seed_spec["train"])
    held = [seed for fold in folds for seed in fold.held_out_seeds]
    locked = set()
    for name in (
        "calibration",
        "validation",
        "background",
        "matched_test",
        "external_holdout",
    ):
        locked.update(int(seed) for seed in seed_spec[name])
    errors = []
    if len(folds) != 5 or any(len(fold.held_out_seeds) != 8 for fold in folds):
        errors.append("fold_count_or_size")
    if len(held) != len(set(held)) or set(held) != train:
        errors.append("held_out_coverage")
    for fold in folds:
        held_out = set(fold.held_out_seeds)
        helper_train = set(fold.helper_train_seeds)
        if held_out & helper_train:
            errors.append(f"fold_{fold.fold_id}_overlap")
        if helper_train != train - held_out:
            errors.append(f"fold_{fold.fold_id}_helper_train")
        if helper_train & locked:
            errors.append(f"fold_{fold.fold_id}_locked_seed")
    return {
        "passed": not errors,
        "errors": errors,
        "held_out_seed_count": len(set(held)),
    }


def build_crossfit_protocol_manifest(
    base_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical global cross-fit protocol and its digest."""
    folds = build_seed_crossfit_folds()
    audit = audit_seed_crossfit_folds(folds)
    if not audit["passed"]:
        raise ValueError(f"invalid seed crossfit protocol: {audit}")
    payload = {
        "protocol_version": 1,
        "train_helper_mode": "seed_crossfit_5fold",
        "evaluation_helper_mode": "full_selector_train",
        "folds": [asdict(fold) for fold in folds],
        "base_configuration": dict(base_configuration),
    }
    return {
        "crossfit_protocol_payload": payload,
        "crossfit_protocol_digest": canonical_sha256(payload),
        "audit": audit,
    }


def resolve_crossfit_execution(
    sample_seed: np.ndarray,
    requested_splits: Sequence[str],
    fold_id: int | None,
) -> CrossfitExecution:
    """Resolve a leakage-safe train-fold or final-helper evaluation invocation."""
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    requested = tuple(str(name) for name in requested_splits)
    folds = build_seed_crossfit_folds()
    if "train" in requested:
        if requested != ("train",) or fold_id is None or int(fold_id) not in range(5):
            raise ValueError(
                "crossfit train execution requires exactly one valid fold and only train split"
            )
        fold = folds[int(fold_id)]
        helper = np.flatnonzero(np.isin(seeds, fold.helper_train_seeds))
        labels = {"train": np.flatnonzero(np.isin(seeds, fold.held_out_seeds))}
        return CrossfitExecution(
            mode="crossfit_train_fold",
            fold_id=fold.fold_id,
            held_out_seeds=fold.held_out_seeds,
            helper_train_seeds=fold.helper_train_seeds,
            helper_train_indices=helper,
            label_indices=labels,
        )
    if fold_id is not None:
        raise ValueError("evaluation helper execution does not accept a crossfit fold")
    allowed = {"calibration", "validation", "matched_test", "external_holdout"}
    if not requested or not set(requested).issubset(allowed):
        raise ValueError(
            "crossfit evaluation supports only frozen selector evaluation splits"
        )
    helper_seeds = tuple(DEFAULT_SELECTOR_SEEDS["train"])
    labels = {
        name: np.flatnonzero(np.isin(seeds, DEFAULT_SELECTOR_SEEDS[name]))
        for name in requested
    }
    return CrossfitExecution(
        mode="full_train_eval",
        fold_id=None,
        held_out_seeds=(),
        helper_train_seeds=helper_seeds,
        helper_train_indices=np.flatnonzero(np.isin(seeds, helper_seeds)),
        label_indices=labels,
    )


def _load_fold_cache(path: str | Path) -> LoadedFoldCache:
    from .v11_labeling import (
        load_candidate_interaction_cache,
        load_candidate_label_metadata,
    )

    cache_path = Path(path)
    batch, outcome, interactions, manifest = load_candidate_interaction_cache(
        cache_path
    )
    metadata = load_candidate_label_metadata(cache_path)
    fold_ids = metadata["sample_fold_id"]
    if fold_ids is None:
        raise ValueError(f"crossfit fold cache is missing sample provenance: {cache_path}")
    return LoadedFoldCache(
        path=cache_path,
        sample_ids=metadata["sample_ids"],
        sample_seed=metadata["sample_seed"],
        fold_ids=np.asarray(fold_ids, dtype=np.int16),
        batch=batch,
        outcome=outcome,
        interactions=interactions,
        manifest=manifest,
    )


def _contract_signature(item: LoadedFoldCache) -> tuple[Any, ...]:
    interaction = item.manifest.get("interaction", {})
    return (
        int(item.manifest.get("schema_version", -1)),
        tuple(item.batch.candidate_names),
        tuple(item.batch.feature_names),
        tuple(item.batch.context_feature_names),
        tuple(item.interactions.token_feature_names),
        tuple(item.interactions.pooled_feature_names),
        tuple(interaction.get("action_feature_names", ())),
        tuple(item.interactions.tokens.shape[1:]),
        tuple(item.interactions.pooled_features.shape[1:]),
    )


def merge_crossfit_label_caches(
    fold_cache_paths: Sequence[str | Path],
    output_path: str | Path,
    expected_sample_ids: np.ndarray,
    expected_sample_seed: np.ndarray,
    expected_crossfit_protocol_digest: str,
) -> dict[str, Any]:
    """Merge five out-of-fold schema-v6 train caches in stable sample order."""
    from .v11_interactions import CandidateInteractionBatch
    from .v11_labeling import save_candidate_interaction_cache
    from .v11_selector import CandidateBatch, CandidateOutcome

    items = tuple(_load_fold_cache(path) for path in fold_cache_paths)
    if not items:
        raise ValueError("crossfit merge requires fold caches")
    sample_ids = np.concatenate([item.sample_ids for item in items], axis=0)
    sample_seed = np.concatenate([item.sample_seed for item in items], axis=0)
    fold_ids = np.concatenate([item.fold_ids for item in items], axis=0)
    if len(sample_ids) != len(set(int(value) for value in sample_ids)):
        raise ValueError("duplicate sample id across crossfit folds")
    if len(items) != 5:
        raise ValueError("crossfit merge requires exactly five fold caches")

    expected_ids = np.asarray(expected_sample_ids, dtype=np.int64).reshape(-1)
    expected_seeds = np.asarray(expected_sample_seed, dtype=np.int64).reshape(-1)
    if expected_ids.shape != expected_seeds.shape:
        raise ValueError("expected sample ids and seeds must match")
    if len(expected_ids) != len(set(int(value) for value in expected_ids)):
        raise ValueError("expected sample ids must be unique")
    if set(sample_ids.tolist()) != set(expected_ids.tolist()):
        raise ValueError("crossfit sample coverage mismatch")
    expected_seed_by_id = {
        int(sample_id): int(seed)
        for sample_id, seed in zip(expected_ids, expected_seeds)
    }
    order = np.argsort(sample_ids, kind="stable")
    if any(
        expected_seed_by_id[int(sample_id)] != int(seed)
        for sample_id, seed in zip(sample_ids[order], sample_seed[order])
    ):
        raise ValueError("crossfit sample seed mismatch")

    expected_digest = str(expected_crossfit_protocol_digest)
    if len(expected_digest) != 64:
        raise ValueError("crossfit protocol digest must contain 64 characters")
    configurations = {
        str(item.manifest.get("configuration_digest", "")) for item in items
    }
    if len(configurations) != 1:
        raise ValueError("crossfit fold configuration digest mismatch")
    if len({_contract_signature(item) for item in items}) != 1:
        raise ValueError("crossfit fold cache contract mismatch")

    formal_folds = build_seed_crossfit_folds()
    observed_fold_ids = set()
    source_fold_caches = []
    for item in items:
        if str(item.manifest.get("split_name")) != "train":
            raise ValueError("crossfit fold cache must use train split")
        if int(item.manifest.get("schema_version", -1)) != 6:
            raise ValueError("crossfit fold cache must use schema 6")
        protocol = item.manifest.get("protocol_metadata", {})
        if str(protocol.get("crossfit_protocol_digest", "")) != expected_digest:
            raise ValueError("crossfit fold protocol digest mismatch")
        execution = protocol.get("helper_execution", {})
        if str(execution.get("mode")) != "crossfit_train_fold":
            raise ValueError("crossfit fold helper execution mode mismatch")
        fold_id = int(execution.get("fold_id", -1))
        if fold_id not in range(5) or fold_id in observed_fold_ids:
            raise ValueError("crossfit fold id coverage mismatch")
        observed_fold_ids.add(fold_id)
        formal = formal_folds[fold_id]
        if tuple(execution.get("held_out_seeds", ())) != formal.held_out_seeds:
            raise ValueError("crossfit held-out seed provenance mismatch")
        if tuple(execution.get("helper_train_seeds", ())) != formal.helper_train_seeds:
            raise ValueError("crossfit helper train seed provenance mismatch")
        if not np.all(item.fold_ids == fold_id):
            raise ValueError("crossfit sample fold provenance mismatch")
        if not set(int(value) for value in item.sample_seed).issubset(
            set(formal.held_out_seeds)
        ):
            raise ValueError("crossfit cache contains seed outside held-out fold")
        source_fold_caches.append(
            {
                "fold_id": fold_id,
                "path": str(item.path),
                "cache_sha256": str(item.manifest.get("cache_sha256", "")),
            }
        )
    if observed_fold_ids != set(range(5)):
        raise ValueError("crossfit fold id coverage mismatch")

    def cat_outcome(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(item.outcome, name)) for item in items], axis=0
        )

    def cat_optional(name: str) -> np.ndarray | None:
        values = [getattr(item.outcome, name) for item in items]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                f"crossfit outcome field is only partially present: {name}"
            )
        return np.concatenate([np.asarray(value) for value in values], axis=0)[order]

    first = items[0]
    merged_batch = CandidateBatch(
        context=np.concatenate([item.batch.context for item in items], axis=0)[order],
        candidate_features=np.concatenate(
            [item.batch.candidate_features for item in items], axis=0
        )[order],
        candidate_mask=np.concatenate(
            [item.batch.candidate_mask for item in items], axis=0
        )[order],
        stage=np.concatenate([item.batch.stage for item in items], axis=0)[order],
        feature_names=first.batch.feature_names,
        candidate_names=first.batch.candidate_names,
        context_feature_names=first.batch.context_feature_names,
    )
    merged_outcome = CandidateOutcome(
        active_sse=cat_outcome("active_sse")[order],
        active_count=cat_outcome("active_count")[order],
        link_sse=cat_optional("link_sse"),
        link_count=cat_optional("link_count"),
        activity_tp=cat_optional("activity_tp"),
        activity_fp=cat_optional("activity_fp"),
        activity_fn=cat_optional("activity_fn"),
        activity_tn=cat_optional("activity_tn"),
        action_applied=cat_optional("action_applied"),
        action_applicable=cat_optional("action_applicable"),
        default_index=first.outcome.default_index,
        task_utility=cat_optional("task_utility"),
        energy_total=cat_optional("energy_total"),
        result_kind="diagnostic_only",
    )
    merged_interactions = CandidateInteractionBatch(
        tokens=np.concatenate(
            [item.interactions.tokens for item in items], axis=0
        )[order],
        token_mask=np.concatenate(
            [item.interactions.token_mask for item in items], axis=0
        )[order],
        edge_index=np.concatenate(
            [item.interactions.edge_index for item in items], axis=0
        )[order],
        token_feature_names=first.interactions.token_feature_names,
        pooled_features=np.concatenate(
            [item.interactions.pooled_features for item in items], axis=0
        )[order],
        pooled_feature_names=first.interactions.pooled_feature_names,
    )
    source_fold_caches.sort(key=lambda row: int(row["fold_id"]))
    protocol_metadata = dict(first.manifest.get("protocol_metadata", {}))
    protocol_metadata["helper_execution"] = {
        "mode": "seed_crossfit_5fold_merged",
        "fold_ids": list(range(5)),
    }
    protocol_metadata["source_fold_caches"] = source_fold_caches
    action_names = tuple(
        first.manifest.get("interaction", {}).get("action_feature_names", ())
    )
    return save_candidate_interaction_cache(
        output_path,
        split_name="train",
        sample_ids=sample_ids[order],
        sample_seed=sample_seed[order],
        batch=merged_batch,
        outcome=merged_outcome,
        interactions=merged_interactions,
        action_feature_names=action_names,
        configuration_digest=next(iter(configurations)),
        protocol_metadata=protocol_metadata,
        sample_fold_id=fold_ids[order],
    )
