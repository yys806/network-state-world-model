"""Rule baselines and method registry for the formal PI-JWM comparison."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .airfogsim_tensor_v2 import EDGE_FEATURES


def _repeat_last(value: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    repeats = [1] * value.ndim
    repeats[1] = horizon_steps
    return value[:, -1:].repeat(*repeats)


def _boolean_logits(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value.bool(), 20.0, -20.0).to(torch.float32)


def _normalized_zero(reference: torch.Tensor, stat: Mapping[str, Any]) -> torch.Tensor:
    mean = reference.new_tensor(stat["mean"])
    scale = reference.new_tensor(stat["scale"]).clamp_min(1e-6)
    zero = -mean / scale
    return zero.reshape(*([1] * (reference.ndim - 1)), -1).expand_as(reference).clone()


def _last_persistence(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    history = batch["history"]
    horizon = int(batch["future_action"]["task_action"].shape[1])
    output: dict[str, torch.Tensor] = {}
    for name in ("node", "physical_edge", "flow", "task"):
        mean = _repeat_last(history[f"{name}_state"], horizon)
        output[f"{name}_state_mean"] = mean
        output[f"{name}_state_log_variance"] = torch.zeros_like(mean)
        output[f"{name}_presence_logits"] = _boolean_logits(
            _repeat_last(history[f"{name}_present"], horizon)
        )
        output[f"{name}_state"] = mean

    dag_mean = _repeat_last(history["task_dag_state"], horizon)
    output["task_dag_state_mean"] = dag_mean
    output["task_dag_state_log_variance"] = torch.zeros_like(dag_mean)
    if "link_activity" in history:
        link_activity = history["link_activity"]
    else:
        activity_index = list(EDGE_FEATURES).index("active_task_count")
        link_activity = (
            history["physical_edge_state"][..., activity_index] > 0
        ) & history["physical_edge_present"].bool()
    output["link_activity_logits"] = _boolean_logits(_repeat_last(link_activity, horizon))
    lifecycle = _repeat_last(history["task_lifecycle_index"], horizon).long()
    lifecycle_logits = torch.full(
        (*lifecycle.shape, 5), -20.0, device=lifecycle.device, dtype=torch.float32
    )
    valid_lifecycle = lifecycle >= 0
    if torch.any(valid_lifecycle):
        lifecycle_logits[valid_lifecycle, lifecycle[valid_lifecycle]] = 20.0
    output["task_lifecycle_logits"] = lifecycle_logits
    release = (dag_mean[..., 2] > 0.5) & _repeat_last(
        history["task_dag_state_present"], horizon
    ).bool()
    output["dag_release_logits"] = _boolean_logits(release)
    output["dag_edge_presence_logits"] = _boolean_logits(
        _repeat_last(history["dag_edge_present"], horizon)
    )
    return output


def build_rule_prediction(
    method: str,
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build a unified prediction for one deterministic rule baseline."""

    if method not in {"zero_activity", "last_persistence"}:
        raise ValueError(f"unsupported rule baseline: {method}")
    output = _last_persistence(batch)
    if method == "last_persistence":
        return output

    output["link_activity_logits"].fill_(-20.0)
    output["flow_presence_logits"].fill_(-20.0)
    output["task_presence_logits"].fill_(-20.0)
    output["dag_release_logits"].fill_(-20.0)
    output["dag_edge_presence_logits"].fill_(-20.0)
    for name in ("flow", "task"):
        zero = _normalized_zero(output[f"{name}_state_mean"], stats["features"][f"{name}_state"])
        output[f"{name}_state_mean"] = zero
        output[f"{name}_state"] = zero
    edge = output["physical_edge_state_mean"].clone()
    edge_stat = stats["features"]["physical_edge_state"]
    mean = edge.new_tensor(edge_stat["mean"])
    scale = edge.new_tensor(edge_stat["scale"]).clamp_min(1e-6)
    for feature in ("rate_sum", "active_task_count", "allocated_rb_count"):
        index = list(EDGE_FEATURES).index(feature)
        edge[..., index] = -mean[index] / scale[index]
    output["physical_edge_state_mean"] = edge
    output["physical_edge_state"] = edge
    output["task_dag_state_mean"] = torch.zeros_like(output["task_dag_state_mean"])
    output["task_lifecycle_logits"].fill_(-20.0)
    output["task_lifecycle_logits"][..., 3] = 20.0
    return output


def method_registry() -> dict[str, dict[str, Any]]:
    return {
        "zero_activity": {
            "stage": "cpu_ready",
            "role": "rule_baseline",
            "distribution_output": False,
            "graph_message_passing": False,
        },
        "last_persistence": {
            "stage": "cpu_ready",
            "role": "rule_baseline",
            "distribution_output": False,
            "graph_message_passing": False,
        },
        "pooled_gru": {
            "stage": "cpu_ready",
            "role": "generic_learned_baseline",
            "distribution_output": True,
            "graph_message_passing": False,
        },
        "independent_dual_gnn": {
            "stage": "cpu_ready",
            "role": "strict_dual_graph_ablation",
            "distribution_output": True,
            "graph_message_passing": True,
            "cross_graph_coupling": False,
        },
        "coupled_dual_gnn": {
            "stage": "cpu_ready",
            "role": "pi_jwm_main_model",
            "distribution_output": True,
            "graph_message_passing": True,
            "cross_graph_coupling": True,
        },
        "coupled_jepa_bou_chaaya_2026": {
            "stage": "gpu_pending",
            "role": "complete_paper_method_adapter",
            "distribution_output": True,
            "citation": "C. Bou Chaaya et al., Learning Latent Multimodal Dynamics for Optimized Resource Planning, IEEE TWC, 2026.",
            "required_components": [
                "context_encoder",
                "target_encoder",
                "latent_predictor",
                "resource_planning_interface",
            ],
        },
    }


__all__ = ["build_rule_prediction", "method_registry"]
