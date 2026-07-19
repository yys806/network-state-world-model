"""Leakage-safe helper cross-fitting for PI-JWM v11 candidate labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .v11_selector import DEFAULT_SELECTOR_SEEDS, canonical_sha256


@dataclass(frozen=True)
class SeedCrossfitFold:
    """One held-out seed fold and the helper-training complement."""

    fold_id: int
    held_out_seeds: tuple[int, ...]
    helper_train_seeds: tuple[int, ...]


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
