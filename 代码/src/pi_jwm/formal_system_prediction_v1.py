"""Convert normalized PI-JWM rollout heads into real one-step system outcomes."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .airfogsim_tensor_v2 import FLOW_FEATURES, LIFECYCLE_TYPES, TASK_FEATURES


def _feature_to_real(
    value: torch.Tensor,
    stats: Mapping[str, Any],
    state_name: str,
    feature_index: int,
) -> torch.Tensor:
    feature_stats = stats["features"][state_name]
    mean = float(feature_stats["mean"][feature_index])
    scale = float(feature_stats["scale"][feature_index])
    return value * value.new_tensor(scale) + value.new_tensor(mean)


def system_predictions_from_batch(
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    horizon_index: int = 0,
) -> dict[str, torch.Tensor | None]:
    """Build de-normalized, non-duplicated k=1 outcomes for a batched window."""

    if horizon_index < 0:
        raise ValueError("horizon_index must be non-negative")
    system_target = batch["system_target"]
    if horizon_index >= system_target["task_completion_event"].shape[1]:
        raise IndexError("horizon_index exceeds the system target horizon")

    task_valid = batch["static"]["task_valid"].bool()
    previous_lifecycle = batch["history"]["task_lifecycle_index"][:, -1]
    predicted_lifecycle = prediction["task_lifecycle_logits"][:, horizon_index].argmax(dim=-1)
    finished_index = LIFECYCLE_TYPES.index("finished")
    predicted_completion = (
        (predicted_lifecycle == finished_index)
        & (previous_lifecycle != finished_index)
        & task_valid
    )

    delay_index = TASK_FEATURES.index("delay")
    predicted_delay = _feature_to_real(
        prediction["task_state_mean"][:, horizon_index, :, delay_index],
        stats,
        "task_state",
        delay_index,
    ).clamp_min(0.0)

    delivered_index = FLOW_FEATURES.index("delivered_this_slot")
    delivered_per_flow = _feature_to_real(
        prediction["flow_state_mean"][:, horizon_index, :, delivered_index],
        stats,
        "flow_state",
        delivered_index,
    ).clamp_min(0.0)
    flow_probability = torch.sigmoid(prediction["flow_presence_logits"][:, horizon_index])
    flow_valid = batch["static"]["flow_valid"].bool()
    predicted_delivered = (delivered_per_flow * flow_probability * flow_valid).sum(dim=-1)

    node_count = int(system_target["source_service_delta"].shape[-1])
    source_index = batch["history"]["task_node_index"][:, -1, :, 0].long()
    source_valid = (source_index >= 0) & (source_index < node_count) & predicted_completion
    predicted_source_service = predicted_delay.new_zeros(
        (predicted_delay.shape[0], node_count)
    )
    predicted_source_service.scatter_add_(
        1,
        source_index.clamp(0, max(node_count - 1, 0)),
        source_valid.to(predicted_source_service.dtype),
    )

    predicted_energy = prediction.get("uav_energy_delta_mean")
    if predicted_energy is not None:
        predicted_energy = predicted_energy[:, horizon_index].clamp_min(0.0)

    return {
        "true_completion_event": system_target["task_completion_event"][:, horizon_index].bool(),
        "predicted_completion_event": predicted_completion,
        "true_completed_delay": system_target["completed_task_delay"][:, horizon_index],
        "predicted_task_delay": predicted_delay,
        "completed_delay_valid": system_target["completed_task_delay_valid"][:, horizon_index].bool(),
        "true_delivered_data": system_target["delivered_data_total"][:, horizon_index],
        "predicted_delivered_data": predicted_delivered,
        "true_uav_energy": system_target["uav_energy_delta"][:, horizon_index],
        "predicted_uav_energy": predicted_energy,
        "uav_energy_valid": system_target["uav_energy_valid"][:, horizon_index].bool(),
        "true_source_service": system_target["source_service_delta"][:, horizon_index],
        "predicted_source_service": predicted_source_service,
        "source_population_valid": batch["system_static"]["source_population_valid"].bool(),
    }


__all__ = ["system_predictions_from_batch"]
