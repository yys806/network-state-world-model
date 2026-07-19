"""Leakage-safe helper cross-fitting for PI-JWM v11 candidate labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
