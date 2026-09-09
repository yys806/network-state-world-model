"""Masked multi-task objective for the formal PI-JWM rollout interface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F

from .airfogsim_tensor_v2 import EDGE_FEATURES, NODE_FEATURES, TASK_FEATURES
from .formal_dual_graph_world_model_v1 import COMPONENT_FEATURES


@dataclass(frozen=True)
class FormalLossWeights:
    state_nll: float = 1.0
    state_mae: float = 0.05
    presence: float = 0.1
    sparse_event: float = 0.1
    lifecycle: float = 0.1
    dag: float = 0.1
    active_rate_mae: float = 0.05
    rb_occupancy_mae: float = 0.05
    task_delay_mae: float = 0.05
    task_deadline_mae: float = 0.05
    uav_energy_nll: float = 0.05
    uav_energy_mae: float = 0.05
    rssm_kl: float = 0.0
    rssm_teacher_reconstruction: float = 0.0
    rssm_overshooting: float = 0.0
    rssm_kl_balance: float = 0.8
    node_x_residual_non_degradation: float = 0.0


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


def _normal_kl(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(q || p) for diagonal normal distributions."""

    q_variance = torch.exp(2.0 * q_log_std)
    p_variance = torch.exp(2.0 * p_log_std)
    value = (
        p_log_std
        - q_log_std
        + (q_variance + torch.square(q_mean - p_mean)) / (2.0 * p_variance)
        - 0.5
    )
    if mask is None:
        return value.mean()
    expanded = mask.bool().unsqueeze(-1).expand_as(value)
    return _masked_mean(value, expanded)


def _balanced_normal_kl(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
    balance: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 <= balance <= 1.0:
        raise ValueError("rssm_kl_balance must be in [0, 1]")
    dynamics = _normal_kl(
        q_mean.detach(), q_log_std.detach(), p_mean, p_log_std, mask
    )
    representation = _normal_kl(
        q_mean, q_log_std, p_mean.detach(), p_log_std.detach(), mask
    )
    return balance * dynamics + (1.0 - balance) * representation


def _aggregate_target(
    target: Mapping[str, torch.Tensor],
    name: str,
    fallback: torch.Tensor,
) -> torch.Tensor:
    value = target.get(name)
    return fallback if value is None else value


def _normalize_raw_aggregate_edge_target(
    raw_target: torch.Tensor,
    *,
    feature_index: int,
    normalization_stats: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Map a raw aggregate edge label into the model's normalized state space."""

    if normalization_stats is None:
        raise ValueError(
            "normalization_stats are required for observed raw aggregate edge targets"
        )
    try:
        feature_stats = normalization_stats["features"]["physical_edge_state"]
        mean = float(feature_stats["mean"][feature_index])
        scale = float(feature_stats["scale"][feature_index])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("physical-edge normalization statistics are incomplete") from error
    return (raw_target - raw_target.new_tensor(mean)) / raw_target.new_tensor(
        max(scale, 1e-6)
    )


def formal_world_model_loss(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    static: Mapping[str, torch.Tensor],
    *,
    weights: FormalLossWeights = FormalLossWeights(),
    class_weights: Mapping[str, float] | None = None,
    system_target: Mapping[str, torch.Tensor] | None = None,
    normalization_stats: Mapping[str, Any] | None = None,
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
    aggregate_activity = _aggregate_target(
        target, "aggregate_link_activity", target["link_activity"]
    ).bool()
    aggregate_activity_mask = _aggregate_target(
        target, "aggregate_link_activity_mask", edge_valid[:, None, :].expand_as(aggregate_activity)
    ).bool()
    link_valid = edge_valid[:, None, :].expand_as(aggregate_activity) & aggregate_activity_mask
    link_activity = _binary_loss(
        prediction["link_activity_logits"],
        aggregate_activity,
        link_valid,
        class_weights.get("aggregate_link_activity", class_weights.get("link_activity", 1.0)),
    )
    components["link_activity_bce"] = link_activity.detach()
    components["aggregate_link_activity_valid_count"] = link_valid.sum().detach()

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

    rate_index = list(EDGE_FEATURES).index("rate_sum")
    raw_aggregate_rate = _aggregate_target(
        target, "aggregate_link_rate_sum", target["physical_edge_state"][..., rate_index]
    )
    aggregate_rate_mask = _aggregate_target(
        target,
        "aggregate_link_rate_sum_mask",
        aggregate_activity & edge_valid[:, None, :],
    ).bool() & edge_valid[:, None, :]
    rate_mask = aggregate_rate_mask
    aggregate_rate = (
        _normalize_raw_aggregate_edge_target(
            raw_aggregate_rate,
            feature_index=rate_index,
            normalization_stats=normalization_stats,
        )
        if "aggregate_link_rate_sum" in target and torch.any(rate_mask)
        else raw_aggregate_rate
    )
    active_rate = _masked_mean(
        torch.abs(
            prediction["physical_edge_state_mean"][..., rate_index]
            - aggregate_rate
        ),
        rate_mask,
    )
    components["active_rate_mae"] = active_rate.detach()
    components["active_rate_valid_count"] = rate_mask.sum().detach()
    components["aggregate_link_rate_sum_valid_count"] = rate_mask.sum().detach()

    rb_index = list(EDGE_FEATURES).index("allocated_rb_count")
    aggregate_rb = target.get("aggregate_rb_occupancy")
    aggregate_rb_mask = target.get("aggregate_rb_occupancy_mask")
    if aggregate_rb is None or aggregate_rb_mask is None:
        rb_occupancy = active_rate.new_zeros(())
        rb_valid_count = torch.zeros((), device=active_rate.device, dtype=torch.long)
    else:
        rb_mask = aggregate_rb_mask.bool() & edge_valid[:, None, :]
        normalized_aggregate_rb = (
            _normalize_raw_aggregate_edge_target(
                aggregate_rb,
                feature_index=rb_index,
                normalization_stats=normalization_stats,
            )
            if torch.any(rb_mask)
            else aggregate_rb
        )
        rb_occupancy = _masked_mean(
            torch.abs(
                prediction["physical_edge_state_mean"][..., rb_index] - normalized_aggregate_rb
            ),
            rb_mask,
        )
        rb_valid_count = rb_mask.sum()
    components["aggregate_rb_occupancy_mae"] = rb_occupancy.detach()
    components["aggregate_rb_occupancy_valid_count"] = rb_valid_count.detach()

    timing_valid = target["task_present"].bool() & task_valid[:, None, :]
    delay_index = list(TASK_FEATURES).index("delay")
    deadline_index = list(TASK_FEATURES).index("deadline_remaining")
    task_delay = _masked_mean(
        torch.abs(
            prediction["task_state_mean"][..., delay_index]
            - target["task_state"][..., delay_index]
        ),
        timing_valid,
    )
    task_deadline = _masked_mean(
        torch.abs(
            prediction["task_state_mean"][..., deadline_index]
            - target["task_state"][..., deadline_index]
        ),
        timing_valid,
    )
    components["task_delay_mae"] = task_delay.detach()
    components["task_deadline_mae"] = task_deadline.detach()
    components["task_timing_valid_count"] = timing_valid.sum().detach()

    uav_energy_nll = task_delay.new_zeros(())
    uav_energy_mae = task_delay.new_zeros(())
    uav_energy_valid_count = torch.zeros((), device=task_delay.device, dtype=torch.long)
    if system_target is not None:
        if "uav_energy_delta_mean" not in prediction:
            raise ValueError("system_target requires the model UAV energy prediction head")
        energy_valid = system_target["uav_energy_valid"].bool()
        uav_energy_nll, uav_energy_mae, uav_energy_valid_count = _gaussian_terms(
            prediction["uav_energy_delta_mean"].unsqueeze(-1),
            prediction["uav_energy_delta_log_variance"].unsqueeze(-1),
            system_target["uav_energy_delta"].unsqueeze(-1),
            energy_valid,
        )
    components["uav_energy_nll"] = uav_energy_nll.detach()
    components["uav_energy_mae"] = uav_energy_mae.detach()
    components["uav_energy_valid_count"] = uav_energy_valid_count.detach()

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
        + weights.active_rate_mae * active_rate
        + weights.rb_occupancy_mae * rb_occupancy
        + weights.task_delay_mae * task_delay
        + weights.task_deadline_mae * task_deadline
        + weights.uav_energy_nll * uav_energy_nll
        + weights.uav_energy_mae * uav_energy_mae
    )
    components["state_nll"] = state_nll.detach()
    components["state_mae"] = state_mae.detach()
    components["presence_bce"] = presence.detach()
    correction = prediction.get("rssm_node_state_correction")
    if weights.node_x_residual_non_degradation > 0.0 and correction is None:
        raise ValueError(
            "node-x residual non-degradation requires an explicit RSSM node correction"
        )
    if isinstance(correction, torch.Tensor):
        x_index = list(NODE_FEATURES).index("x")
        correction_x = correction[..., x_index]
        full_x = prediction["node_state_mean"][..., x_index]
        target_x = target["node_state"][..., x_index]
        base_x = (full_x - correction_x).detach()
        safety_full_x = base_x + correction_x
        valid = target["node_present"].bool()
        node_x_non_degradation = _masked_mean(
            F.relu(
                torch.abs(safety_full_x - target_x)
                - torch.abs(base_x - target_x)
            ),
            valid,
        )
        components["node_x_residual_non_degradation"] = (
            node_x_non_degradation.detach()
        )
        components["node_x_residual_valid_count"] = valid.sum().detach()
        total = (
            total
            + weights.node_x_residual_non_degradation * node_x_non_degradation
        )
    rssm_keys = {
        "rssm_posterior_path_prior_mean",
        "rssm_posterior_path_prior_log_std",
        "rssm_posterior_mean",
        "rssm_posterior_log_std",
    }
    present_rssm_keys = rssm_keys.intersection(prediction)
    if present_rssm_keys and present_rssm_keys != rssm_keys:
        raise ValueError("RSSM probability parameters are incomplete")
    if present_rssm_keys:
        rssm_kl = _balanced_normal_kl(
            prediction["rssm_posterior_mean"],
            prediction["rssm_posterior_log_std"],
            prediction["rssm_posterior_path_prior_mean"],
            prediction["rssm_posterior_path_prior_log_std"],
            weights.rssm_kl_balance,
            prediction.get("rssm_latent_mask"),
        )
        components["rssm_kl"] = rssm_kl.detach()
        total = total + weights.rssm_kl * rssm_kl

        teacher_prediction = {
            key.removeprefix("training_"): value
            for key, value in prediction.items()
            if key.startswith("training_") and isinstance(value, torch.Tensor)
        }
        if not teacher_prediction:
            raise ValueError("RSSM posterior parameters require teacher predictions")
        teacher_weights = replace(
            weights,
            rssm_kl=0.0,
            rssm_teacher_reconstruction=0.0,
            rssm_overshooting=0.0,
            node_x_residual_non_degradation=0.0,
        )
        teacher_reconstruction, _ = formal_world_model_loss(
            teacher_prediction,
            target,
            static,
            weights=teacher_weights,
            class_weights=class_weights,
            system_target=system_target,
            normalization_stats=normalization_stats,
        )
        components["rssm_teacher_reconstruction"] = teacher_reconstruction.detach()
        total = total + weights.rssm_teacher_reconstruction * teacher_reconstruction

        overshooting_keys = {
            "rssm_overshooting_prior_mean",
            "rssm_overshooting_prior_log_std",
            "rssm_overshooting_posterior_mean",
            "rssm_overshooting_posterior_log_std",
        }
        if not overshooting_keys.issubset(prediction):
            raise ValueError("complete RSSM requires overshooting parameters")
        overshooting = _balanced_normal_kl(
            prediction["rssm_overshooting_posterior_mean"],
            prediction["rssm_overshooting_posterior_log_std"],
            prediction["rssm_overshooting_prior_mean"],
            prediction["rssm_overshooting_prior_log_std"],
            weights.rssm_kl_balance,
            prediction.get("rssm_overshooting_latent_mask"),
        )
        components["rssm_overshooting"] = overshooting.detach()
        total = total + weights.rssm_overshooting * overshooting
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
        aggregate_activity = target.get("aggregate_link_activity", target["link_activity"])
        aggregate_activity_mask = target.get(
            "aggregate_link_activity_mask", edge_valid
        ).bool()
        labels_and_masks = {
            "link_activity": (aggregate_activity, edge_valid & aggregate_activity_mask),
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
