"""Masked multi-task objective for the formal PI-JWM rollout interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F

from .formal_dual_graph_world_model_v1 import COMPONENT_FEATURES


@dataclass(frozen=True)
class FormalLossWeights:
    state_nll: float = 1.0
    state_mae: float = 0.05
    presence: float = 0.1
    sparse_event: float = 0.1
    lifecycle: float = 0.1
    dag: float = 0.1


def _component_valid_mask(name: str, static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if name == "node":
        return static["node_kind_index"] >= 0
    if name == "physical_edge":
        return torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
    if name == "flow":
        return static["flow_valid"].bool()
    if name == "task":
        return static["task_valid"].bool()
    raise KeyError(name)


def _with_batch(mask: torch.Tensor) -> torch.Tensor:
    return mask.unsqueeze(0) if mask.ndim == 1 else mask


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.to(value.dtype)
    return (value * numeric).sum() / numeric.sum().clamp_min(1.0)


def _gaussian_terms(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_mask = present.unsqueeze(-1).expand_as(target)
    squared_error = torch.square(mean - target)
    nll = 0.5 * (log_variance + squared_error * torch.exp(-log_variance))
    mae = torch.abs(mean - target)
    valid_count = present.sum()
    return _masked_mean(nll, feature_mask), _masked_mean(mae, feature_mask), valid_count


def _binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(
        logits,
        target.to(logits.dtype),
        pos_weight=logits.new_tensor(float(pos_weight)),
        reduction="none",
    )
    return _masked_mean(values, valid)


def formal_world_model_loss(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    static: Mapping[str, torch.Tensor],
    *,
    weights: FormalLossWeights = FormalLossWeights(),
    class_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one finite, auditable objective over all formal rollout heads."""

    class_weights = dict(class_weights or {})
    components: dict[str, torch.Tensor] = {}
    state_nll_terms: list[torch.Tensor] = []
    state_mae_terms: list[torch.Tensor] = []
    presence_terms: list[torch.Tensor] = []
    for name in COMPONENT_FEATURES:
        present = target[f"{name}_present"].bool()
        nll, mae, valid_count = _gaussian_terms(
            prediction[f"{name}_state_mean"],
            prediction[f"{name}_state_log_variance"],
            target[f"{name}_state"],
            present,
        )
        components[f"{name}_state_nll"] = nll.detach()
        components[f"{name}_state_mae"] = mae.detach()
        components[f"{name}_state_valid_count"] = valid_count.detach()
        state_nll_terms.append(nll)
        state_mae_terms.append(mae)

        entity_valid = _with_batch(_component_valid_mask(name, static))
        valid = entity_valid[:, None, :].expand_as(present)
        label_name = f"{name}_present"
        presence = _binary_loss(
            prediction[f"{name}_presence_logits"],
            present,
            valid,
            class_weights.get(label_name, 1.0),
        )
        components[f"{name}_presence_bce"] = presence.detach()
        presence_terms.append(presence)

    dag_present = target["task_dag_state_present"].bool()
    dag_nll, dag_mae, dag_valid_count = _gaussian_terms(
        prediction["task_dag_state_mean"],
        prediction["task_dag_state_log_variance"],
        target["task_dag_state"],
        dag_present,
    )
    components["task_dag_state_nll"] = dag_nll.detach()
    components["task_dag_state_mae"] = dag_mae.detach()
    components["task_dag_state_valid_count"] = dag_valid_count.detach()

    edge_valid = _with_batch(_component_valid_mask("physical_edge", static))
    link_valid = edge_valid[:, None, :].expand_as(target["link_activity"])
    link_activity = _binary_loss(
        prediction["link_activity_logits"],
        target["link_activity"].bool(),
        link_valid,
        class_weights.get("link_activity", 1.0),
    )
    components["link_activity_bce"] = link_activity.detach()

    task_valid = _with_batch(_component_valid_mask("task", static))
    lifecycle_target = target["task_lifecycle_index"].long()
    lifecycle_valid = (
        target["task_present"].bool()
        & task_valid[:, None, :]
        & (lifecycle_target >= 0)
    )
    lifecycle_values = F.cross_entropy(
        prediction["task_lifecycle_logits"].reshape(-1, 5),
        lifecycle_target.clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape_as(lifecycle_target)
    lifecycle = _masked_mean(lifecycle_values, lifecycle_valid)
    components["task_lifecycle_ce"] = lifecycle.detach()

    dag_release_target = target["task_dag_state"][..., 2] > 0.5
    dag_release_valid = dag_present & task_valid[:, None, :]
    dag_release = _binary_loss(
        prediction["dag_release_logits"],
        dag_release_target,
        dag_release_valid,
        class_weights.get("dag_release", 1.0),
    )
    components["dag_release_bce"] = dag_release.detach()

    dag_edge_valid = _with_batch(static["dag_edge_valid"].bool())
    dag_edge_mask = dag_edge_valid[:, None, :].expand_as(target["dag_edge_present"])
    dag_edge_presence = _binary_loss(
        prediction["dag_edge_presence_logits"],
        target["dag_edge_present"].bool(),
        dag_edge_mask,
        class_weights.get("dag_edge_present", 1.0),
    )
    components["dag_edge_presence_bce"] = dag_edge_presence.detach()

    state_nll = torch.stack(state_nll_terms).mean()
    state_mae = torch.stack(state_mae_terms).mean()
    presence = torch.stack(presence_terms).mean()
    total = (
        weights.state_nll * state_nll
        + weights.state_mae * state_mae
        + weights.presence * presence
        + weights.sparse_event * link_activity
        + weights.lifecycle * lifecycle
        + weights.dag * (dag_nll + dag_mae + dag_release + dag_edge_presence)
    )
    components["state_nll"] = state_nll.detach()
    components["state_mae"] = state_mae.detach()
    components["presence_bce"] = presence.detach()
    components["total_loss"] = total.detach()
    return total, components


def _count_binary(target: torch.Tensor, valid: torch.Tensor) -> tuple[int, int]:
    selected = target.bool()[valid.bool()]
    positive = int(selected.sum())
    return positive, int(selected.numel() - positive)


def compute_training_class_weights(
    train_dataset: Iterable[Mapping[str, Any]],
    *,
    max_pos_weight: float = 50.0,
) -> dict[str, Any]:
    """Compute sparse positive weights while rejecting non-training samples."""

    if max_pos_weight < 1.0:
        raise ValueError("max_pos_weight must be at least 1")
    counts = {
        "link_activity": [0, 0],
        "flow_present": [0, 0],
        "task_present": [0, 0],
        "dag_release": [0, 0],
        "dag_edge_present": [0, 0],
    }
    sample_count = 0
    for sample in train_dataset:
        if str(sample.get("split")) != "train":
            raise ValueError("class weights may use only train samples")
        sample_count += 1
        target = sample["target"]
        static = sample["static"]
        horizon = target["task_present"].shape[0]
        edge_valid = _component_valid_mask("physical_edge", static).unsqueeze(0).expand(horizon, -1)
        flow_valid = _component_valid_mask("flow", static).unsqueeze(0).expand(horizon, -1)
        task_valid = _component_valid_mask("task", static).unsqueeze(0).expand(horizon, -1)
        dag_valid = static["dag_edge_valid"].bool().unsqueeze(0).expand(horizon, -1)
        labels_and_masks = {
            "link_activity": (target["link_activity"], edge_valid),
            "flow_present": (target["flow_present"], flow_valid),
            "task_present": (target["task_present"], task_valid),
            "dag_release": (
                target["task_dag_state"][..., 2] > 0.5,
                target["task_dag_state_present"].bool() & task_valid,
            ),
            "dag_edge_present": (target["dag_edge_present"], dag_valid),
        }
        for name, (label, valid) in labels_and_masks.items():
            positive, negative = _count_binary(label, valid)
            counts[name][0] += positive
            counts[name][1] += negative

    pos_weight: dict[str, float] = {}
    for name, (positive, negative) in counts.items():
        ratio = 1.0 if positive == 0 else negative / positive
        pos_weight[name] = float(max(1.0, min(max_pos_weight, ratio)))
    return {
        "source_split": "train",
        "sample_count": sample_count,
        "counts": {
            name: {"positive": int(value[0]), "negative": int(value[1])}
            for name, value in counts.items()
        },
        "pos_weight": pos_weight,
    }


__all__ = [
    "FormalLossWeights",
    "compute_training_class_weights",
    "formal_world_model_loss",
]
