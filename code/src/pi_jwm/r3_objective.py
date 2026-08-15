"""Mask-aware multi-target objective for the PI-JWM R3 CPU preflight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F

from .r3_preflight_data import (
    CONTINUOUS_STATE_KEYS,
    FEATURE_MASK_BY_STATE,
    PRESENCE_BY_STATE,
    STATIC_VALID_BY_STATE,
    ExplicitStateBatch,
)
from .r3_world_model import R3RolloutOutput


R3_OBJECTIVE_SCHEMA = "PIJWM-R3-Masked-Objective-v1"


@dataclass(frozen=True)
class ObjectiveTerm:
    status: str
    value: float | None
    numerator: float | None
    denominator: float
    count: int
    reason: str | None


@dataclass
class R3ObjectiveReport:
    total: torch.Tensor
    terms: dict[str, ObjectiveTerm]
    schema_version: str = R3_OBJECTIVE_SCHEMA


def _static_entity_valid(
    batch: ExplicitStateBatch,
    key: str,
    target: torch.Tensor,
) -> torch.Tensor:
    valid_key = STATIC_VALID_BY_STATE.get(key)
    if valid_key is None:
        return torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device)
    if valid_key not in batch.static:
        raise ValueError(f"{key} is missing static {valid_key}")
    valid = batch.static[valid_key].bool()
    if valid.shape != (target.shape[0], target.shape[2]):
        raise ValueError(f"{valid_key} shape does not match {key}")
    return valid[:, None, :].expand(target.shape[0], target.shape[1], target.shape[2])


def _continuous_mask(
    batch: ExplicitStateBatch,
    key: str,
    target: torch.Tensor,
) -> torch.Tensor:
    presence_key = PRESENCE_BY_STATE[key]
    if presence_key not in batch.target:
        raise ValueError(f"{key} is missing {presence_key}")
    presence = batch.target[presence_key].bool()
    if presence.shape != target.shape[:-1]:
        raise ValueError(f"{presence_key} shape does not match {key}")
    mask = presence.unsqueeze(-1).expand_as(target).clone()
    feature_key = FEATURE_MASK_BY_STATE.get(key)
    if feature_key is not None:
        if feature_key not in batch.target:
            raise ValueError(f"{key} is missing {feature_key}")
        feature_mask = batch.target[feature_key].bool()
        if feature_mask.shape != target.shape:
            raise ValueError(f"{feature_key} shape does not match {key}")
        mask &= feature_mask
    mask &= _static_entity_valid(batch, key, target).unsqueeze(-1)
    return mask


def _term(
    loss_values: torch.Tensor,
    mask: torch.Tensor,
    *,
    reason: str,
) -> tuple[ObjectiveTerm, torch.Tensor | None]:
    mask = mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return (
            ObjectiveTerm(
                status="not_computable",
                value=None,
                numerator=None,
                denominator=0.0,
                count=0,
                reason=reason,
            ),
            None,
        )
    selected = loss_values[mask]
    if not torch.isfinite(selected).all():
        raise ValueError("objective contains NaN or Inf on an observed target")
    numerator = selected.sum()
    mean = numerator / float(count)
    return (
        ObjectiveTerm(
            status="computed",
            value=float(mean.detach().cpu().item()),
            numerator=float(numerator.detach().cpu().item()),
            denominator=float(count),
            count=count,
            reason=None,
        ),
        mean,
    )


def _presence_valid(
    batch: ExplicitStateBatch,
    presence_key: str,
    target: torch.Tensor,
) -> torch.Tensor:
    state_key = next(
        key for key, mapped in PRESENCE_BY_STATE.items() if mapped == presence_key
    )
    valid_key = STATIC_VALID_BY_STATE.get(state_key)
    if valid_key is None:
        return torch.ones_like(target, dtype=torch.bool)
    valid = batch.static[valid_key].bool()
    if valid.shape != (target.shape[0], target.shape[2]):
        raise ValueError(f"{valid_key} shape does not match {presence_key}")
    return valid[:, None, :].expand_as(target)


def compute_r3_objective(
    output: R3RolloutOutput,
    batch: ExplicitStateBatch,
) -> R3ObjectiveReport:
    """Compute only observed v3 targets and preserve explicit N/A terms."""

    terms: dict[str, ObjectiveTerm] = {}
    computed_losses: list[torch.Tensor] = []

    for key in CONTINUOUS_STATE_KEYS:
        if key not in output.predicted_explicit or key not in batch.target:
            raise ValueError(f"R3 continuous target is incomplete: {key}")
        predicted = output.predicted_explicit[key]
        target = batch.target[key][:, : predicted.shape[1]].to(predicted.dtype)
        if predicted.shape != target.shape:
            raise ValueError(f"prediction shape does not match target for {key}")
        mask = _continuous_mask(batch, key, target)
        term, loss = _term(
            torch.square(predicted - target),
            mask,
            reason=f"no observed target values for {key}",
        )
        terms[key] = term
        if loss is not None:
            computed_losses.append(loss)

    for presence_key in PRESENCE_BY_STATE.values():
        if presence_key not in output.predicted_logits or presence_key not in batch.target:
            raise ValueError(f"R3 presence target is incomplete: {presence_key}")
        logits = output.predicted_logits[presence_key]
        target = batch.target[presence_key][:, : logits.shape[1]].to(logits.dtype)
        if logits.shape != target.shape:
            raise ValueError(f"presence shape does not match target for {presence_key}")
        valid = _presence_valid(batch, presence_key, target)
        term, loss = _term(
            F.binary_cross_entropy_with_logits(logits, target, reduction="none"),
            valid,
            reason=f"no valid entity slots for {presence_key}",
        )
        terms[presence_key] = term
        if loss is not None:
            computed_losses.append(loss)

    for key in ("information_link_activity", "information_link_activity_mask"):
        if key not in batch.target:
            raise ValueError(f"R3 activity target is incomplete: {key}")
    activity_logits = output.predicted_logits["information_link_activity"]
    activity_target = batch.target["information_link_activity"][
        :, : activity_logits.shape[1]
    ].to(activity_logits.dtype)
    activity_mask = batch.target["information_link_activity_mask"][
        :, : activity_logits.shape[1]
    ].bool()
    if activity_logits.shape != activity_target.shape or activity_mask.shape != activity_target.shape:
        raise ValueError("information-link activity shapes are inconsistent")
    term, loss = _term(
        F.binary_cross_entropy_with_logits(
            activity_logits, activity_target, reduction="none"
        ),
        activity_mask,
        reason="no observed information-link activity targets",
    )
    terms["information_link_activity"] = term
    if loss is not None:
        computed_losses.append(loss)

    if "task_lifecycle_index" not in batch.target:
        raise ValueError("R3 lifecycle target is incomplete: task_lifecycle_index")
    lifecycle_logits = output.predicted_logits["task_lifecycle"]
    lifecycle = batch.target["task_lifecycle_index"][
        :, : lifecycle_logits.shape[1]
    ].long()
    illegal = (lifecycle < -1) | (lifecycle > 4)
    if illegal.any():
        raise ValueError("task_lifecycle_index must be -1 or one of 0..4")
    lifecycle_valid = lifecycle >= 0
    if "task_present" in batch.target:
        lifecycle_valid &= batch.target["task_present"][
            :, : lifecycle_logits.shape[1]
        ].bool()
    task_valid = batch.static["task_valid"].bool()
    lifecycle_valid &= task_valid[:, None, :].expand_as(lifecycle)
    safe_lifecycle = lifecycle.clamp_min(0)
    lifecycle_loss = F.cross_entropy(
        lifecycle_logits.reshape(-1, lifecycle_logits.shape[-1]),
        safe_lifecycle.reshape(-1),
        reduction="none",
    ).reshape_as(lifecycle)
    term, loss = _term(
        lifecycle_loss,
        lifecycle_valid,
        reason="no observed valid task lifecycle targets",
    )
    terms["task_lifecycle"] = term
    if loss is not None:
        computed_losses.append(loss)

    if not computed_losses:
        raise ValueError("R3 objective has no computable target terms")
    total = torch.stack(computed_losses).mean()
    if not torch.isfinite(total):
        raise ValueError("R3 total objective is NaN or Inf")
    return R3ObjectiveReport(total=total, terms=terms)


__all__ = [
    "ObjectiveTerm",
    "R3_OBJECTIVE_SCHEMA",
    "R3ObjectiveReport",
    "compute_r3_objective",
]
