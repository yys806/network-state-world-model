"""Differentiable deterministic update boundary for formal PI-JWM rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


COMPONENTS = ("node", "physical_edge", "flow", "task")
TO_OFFLOAD, COMPUTING, RETURNING, FINISHED, FAILED = range(5)
INPUT_FLOW, RETURN_FLOW = 0, 1


class NormalizationAdapter(nn.Module):
    def __init__(self, stats: Mapping[str, Any]) -> None:
        super().__init__()
        features = stats.get("features", stats)
        for component in COMPONENTS:
            key = f"{component}_state"
            entry = features.get(key)
            if not isinstance(entry, Mapping):
                raise ValueError(f"missing normalization statistics for {key}")
            mean = torch.as_tensor(entry.get("mean", []), dtype=torch.float32)
            scale = torch.as_tensor(entry.get("scale", []), dtype=torch.float32)
            if mean.ndim != 1 or scale.shape != mean.shape or not mean.numel():
                raise ValueError(f"invalid normalization statistics for {key}")
            self.register_buffer(f"{component}_mean", mean)
            self.register_buffer(f"{component}_scale", scale.clamp_min(1e-6))

    def _pair(self, component: str) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f"{component}_mean"), getattr(self, f"{component}_scale")

    def to_physical(self, component: str, value: torch.Tensor) -> torch.Tensor:
        mean, scale = self._pair(component)
        return value * scale.to(value) + mean.to(value)

    def to_normalized(self, component: str, value: torch.Tensor) -> torch.Tensor:
        mean, scale = self._pair(component)
        return (value - mean.to(value)) / scale.to(value)


def resolve_action_endpoints(
    source_node_index: torch.Tensor,
    target_node_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return explicit action-record endpoints without inference."""

    if source_node_index.ndim not in (3, 4) or source_node_index.shape != target_node_index.shape:
        raise ValueError("explicit source/target endpoints must share [B,Q,4] or [B,H,Q,4]")
    if source_node_index.shape[-1] != 4:
        raise ValueError("explicit action endpoints must contain four semantic slots")
    return source_node_index, target_node_index


def _task_edges(dag_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if dag_edge_index.ndim != 3:
        raise ValueError("dag_edge_index must be rank 3")
    if dag_edge_index.shape[1] == 2:
        return dag_edge_index[:, 0], dag_edge_index[:, 1]
    if dag_edge_index.shape[2] == 2:
        return dag_edge_index[..., 0], dag_edge_index[..., 1]
    raise ValueError("dag_edge_index must use [B,2,D] or [B,D,2]")


def _scatter_task_sum(value: torch.Tensor, task_index: torch.Tensor, task_count: int) -> torch.Tensor:
    safe = task_index.clamp(0, max(task_count - 1, 0)).long()
    valid = (task_index >= 0) & (task_index < task_count)
    return torch.zeros(
        value.shape[0], task_count, device=value.device, dtype=value.dtype
    ).scatter_add(1, safe, value * valid.to(value.dtype))


def _capped_equal_share_tensor(
    demand: torch.Tensor,
    node_index: torch.Tensor,
    node_capacity: torch.Tensor,
) -> torch.Tensor:
    """GPU-safe capped equal sharing without batch/task scalar extraction."""

    node_count = node_capacity.shape[1]
    safe_node = node_index.clamp(0, max(node_count - 1, 0)).long()
    valid = (node_index >= 0) & (node_index < node_count) & (demand > 0)
    membership = torch.nn.functional.one_hot(safe_node, num_classes=node_count).to(demand.dtype)
    membership = membership * valid.unsqueeze(-1).to(demand.dtype)
    total_demand = (demand.unsqueeze(-1) * membership).sum(dim=1)
    target = torch.minimum(total_demand, node_capacity.clamp_min(0.0))
    low = torch.zeros_like(target)
    high = (demand.unsqueeze(-1) * membership).amax(dim=1)
    for _ in range(32):
        middle = 0.5 * (low + high)
        allocated = (torch.minimum(demand.unsqueeze(-1), middle.unsqueeze(1)) * membership).sum(dim=1)
        low = torch.where(allocated < target, middle, low)
        high = torch.where(allocated < target, high, middle)
    water_level = high.gather(1, safe_node)
    return torch.where(valid, torch.minimum(demand, water_level), torch.zeros_like(demand))


@dataclass
class RuleLayerResult:
    states: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    service_masks: dict[str, torch.Tensor]
    lifecycle_logits: torch.Tensor
    lifecycle_index: torch.Tensor
    task_node_index: torch.Tensor
    task_present: torch.Tensor
    flow_present: torch.Tensor
    dag_state: torch.Tensor
    service_outcome: dict[str, torch.Tensor]
    cpu_allocation: torch.Tensor
    cpu_served: torch.Tensor


class DeterministicRuleLayer(nn.Module):
    def __init__(self, stats: Mapping[str, Any], *, n_rb: int) -> None:
        super().__init__()
        if int(n_rb) <= 0:
            raise ValueError("n_rb must be positive")
        self.n_rb = int(n_rb)
        self.normalization = NormalizationAdapter(stats)

    def forward(
        self,
        proposed_means: Mapping[str, torch.Tensor],
        *,
        previous_states: Mapping[str, torch.Tensor],
        previous_task_present: torch.Tensor,
        previous_flow_present: torch.Tensor,
        action: torch.Tensor,
        action_present: torch.Tensor,
        source_node_index: torch.Tensor,
        target_node_index: torch.Tensor,
        task_node_index: torch.Tensor,
        previous_lifecycle_index: torch.Tensor,
        static: Mapping[str, torch.Tensor],
        service_outcome: Mapping[str, torch.Tensor],
        lifecycle_logits: torch.Tensor,
        target_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> RuleLayerResult:
        proposed = {name: self.normalization.to_physical(name, proposed_means[name]) for name in COMPONENTS}
        previous = {name: self.normalization.to_physical(name, previous_states[name]) for name in COMPONENTS}
        states = {name: value.clone() for name, value in proposed.items()}
        source, target = resolve_action_endpoints(source_node_index, target_node_index)
        if source.ndim == 4:
            if source.shape[1] != 1:
                raise ValueError("single rule update accepts one endpoint horizon step")
            source, target = source[:, 0], target[:, 0]

        batch_size, task_count, _ = action.shape
        task_valid = static["task_valid"].bool() & previous_task_present.bool()
        core_present = action_present.bool() & (action[..., :5].abs().sum(dim=-1) > 0)
        parents, children = _task_edges(static["dag_edge_index"])
        dag_valid = static["dag_edge_valid"].bool()
        safe_parent = parents.clamp(0, max(task_count - 1, 0)).long()
        safe_child = children.clamp(0, max(task_count - 1, 0)).long()
        valid_dag = dag_valid & (parents >= 0) & (parents < task_count) & (children >= 0) & (children < task_count)
        parent_count = torch.zeros(batch_size, task_count, device=action.device, dtype=action.dtype).scatter_add(
            1, safe_child, valid_dag.to(action.dtype)
        )
        unfinished = torch.zeros_like(parent_count).scatter_add(
            1,
            safe_child,
            ((previous_lifecycle_index.gather(1, safe_parent) != FINISHED) & valid_dag).to(action.dtype),
        )
        release_ready = (unfinished == 0) & task_valid

        updated_task_nodes = task_node_index.clone()
        offload = core_present & (action[..., 0] > 0) & release_ready & (target[..., 0] >= 0)
        updated_task_nodes[..., 1] = torch.where(offload, target[..., 0], updated_task_nodes[..., 1])
        updated_task_nodes[..., 2] = torch.where(offload, target[..., 0], updated_task_nodes[..., 2])
        return_action = core_present & (action[..., 2] > 0) & (target[..., 1] >= 0)
        updated_task_nodes[..., 3] = torch.where(return_action, target[..., 1], updated_task_nodes[..., 3])

        edge_endpoints = static["physical_edge_endpoint_index"].long()
        edge_valid = torch.all(edge_endpoints >= 0, dim=-1)
        rb_source = source[..., 2]
        rb_target = target[..., 2]
        pair_match = (
            (edge_endpoints[:, None, :, 0] == rb_source[:, :, None])
            & (edge_endpoints[:, None, :, 1] == rb_target[:, :, None])
        )
        rb_valid = core_present & (action[..., 1] > 0) & (rb_source >= 0) & (rb_target >= 0)
        rb_write = (
            pair_match
            * rb_valid.unsqueeze(-1)
            * action[..., 3].clamp_min(0.0).unsqueeze(-1)
        ).sum(dim=1).clamp(0.0, float(self.n_rb))
        edge_rate = service_outcome["edge_rate"].clamp_min(0.0)
        states["physical_edge"][..., 2] = edge_rate
        states["physical_edge"][..., 4] = rb_write

        flow_task = static["flow_task_index"].long()
        flow_type = static["flow_type_index"].long()
        flow_valid = static["flow_valid"].bool() & (flow_task >= 0) & (flow_task < task_count)
        safe_flow_task = flow_task.clamp(0, max(task_count - 1, 0))
        flow_endpoints = static["flow_endpoint_index"].long()
        offload_flow = (
            (flow_type == INPUT_FLOW)
            & offload.gather(1, safe_flow_task)
            & (flow_endpoints[..., 0] == source[..., 0].gather(1, safe_flow_task))
            & (flow_endpoints[..., 1] == target[..., 0].gather(1, safe_flow_task))
        )
        return_flow = (
            (flow_type == RETURN_FLOW)
            & return_action.gather(1, safe_flow_task)
            & (flow_endpoints[..., 0] == source[..., 1].gather(1, safe_flow_task))
            & (flow_endpoints[..., 1] == target[..., 1].gather(1, safe_flow_task))
        )
        created_flow = flow_valid & (offload_flow | return_flow)
        active_flow = flow_valid & (previous_flow_present.bool() | created_flow)
        flow_stage = previous_lifecycle_index.gather(1, safe_flow_task)
        flow_release = release_ready.gather(1, safe_flow_task)
        eligible_flow = active_flow & (
            ((flow_type == INPUT_FLOW) & (flow_stage == TO_OFFLOAD) & flow_release)
            | ((flow_type == RETURN_FLOW) & (flow_stage == RETURNING))
        )
        created_total = torch.where(
            offload_flow,
            previous["task"][..., 0].gather(1, safe_flow_task),
            previous["task"][..., 1].gather(1, safe_flow_task),
        ).clamp_min(0.0)
        flow_total = torch.where(created_flow, created_total, previous["flow"][..., 0]).clamp_min(0.0)
        flow_remaining = torch.where(created_flow, flow_total, previous["flow"][..., 1]).clamp_min(0.0)
        flow_cumulative = torch.where(
            created_flow,
            torch.zeros_like(previous["flow"][..., 2]),
            previous["flow"][..., 2].clamp_min(0.0),
        )
        flow_age = torch.where(
            created_flow,
            torch.zeros_like(previous["flow"][..., 4]),
            previous["flow"][..., 4].clamp_min(0.0),
        )
        if "flow_service_fraction" in service_outcome:
            delivered = service_outcome["flow_service_fraction"].clamp(0.0, 1.0) * flow_remaining
        else:
            delivered = torch.minimum(service_outcome["flow_delivered"].clamp_min(0.0), flow_remaining)
        delivered = delivered * eligible_flow.to(delivered.dtype)
        states["flow"] = previous["flow"].clone()
        states["flow"][..., 0] = flow_total
        states["flow"][..., 1] = flow_remaining - delivered
        states["flow"][..., 2] = flow_cumulative + delivered
        states["flow"][..., 3] = delivered
        slot = static["slot_seconds"].to(action).reshape(batch_size, 1)
        if torch.any(slot <= 0):
            raise ValueError("slot_seconds must be positive")
        states["flow"][..., 4] = flow_age + slot

        input_service = _scatter_task_sum(delivered * (flow_type == INPUT_FLOW), flow_task, task_count)
        return_cumulative = _scatter_task_sum(
            states["flow"][..., 2] * (flow_type == RETURN_FLOW) * flow_valid,
            flow_task,
            task_count,
        )
        task_state = previous["task"].clone()
        task_size = previous["task"][..., 0].clamp_min(0.0)
        task_cpu = previous["task"][..., 2].clamp_min(0.0)
        task_state[..., 3] = (previous["task"][..., 3] - slot).clamp_min(0.0)
        task_state[..., 5] = torch.minimum(task_size, previous["task"][..., 5].clamp_min(0.0) + input_service)
        task_state[..., 7] = previous["task"][..., 7].clamp_min(0.0) + slot

        post_communication_stage = torch.where(
            (previous_lifecycle_index == TO_OFFLOAD) & (task_state[..., 5] >= task_size - 1e-7) & release_ready,
            torch.full_like(previous_lifecycle_index, COMPUTING),
            previous_lifecycle_index,
        )
        execution_node = updated_task_nodes[..., 2]
        execution_node = torch.where(execution_node >= 0, execution_node, updated_task_nodes[..., 1])
        remaining_cpu = (task_cpu - previous["task"][..., 6].clamp_min(0.0)).clamp_min(0.0)
        cpu_eligible = task_valid & (post_communication_stage == COMPUTING) & (execution_node >= 0)
        demand_rate = torch.where(cpu_eligible, remaining_cpu / slot, torch.zeros_like(remaining_cpu))
        cpu_allocation = _capped_equal_share_tensor(demand_rate, execution_node, previous["node"][..., 5])
        cpu_served = torch.minimum(remaining_cpu, cpu_allocation * slot)
        task_state[..., 6] = torch.minimum(task_cpu, previous["task"][..., 6].clamp_min(0.0) + cpu_served)
        states["task"] = task_state

        lifecycle_index = post_communication_stage
        compute_complete = (lifecycle_index == COMPUTING) & (task_state[..., 6] >= task_cpu - 1e-7)
        has_return = previous["task"][..., 1] > 1e-7
        lifecycle_index = torch.where(compute_complete & has_return, torch.full_like(lifecycle_index, RETURNING), lifecycle_index)
        lifecycle_index = torch.where(compute_complete & ~has_return, torch.full_like(lifecycle_index, FINISHED), lifecycle_index)
        return_complete = (lifecycle_index == RETURNING) & (return_cumulative >= previous["task"][..., 1] - 1e-7)
        lifecycle_index = torch.where(return_complete, torch.full_like(lifecycle_index, FINISHED), lifecycle_index)
        expired = (task_state[..., 3] <= 0) & (lifecycle_index != FINISHED)
        lifecycle_index = torch.where(expired, torch.full_like(lifecycle_index, FAILED), lifecycle_index)
        lifecycle_index = torch.where(task_valid, lifecycle_index, torch.full_like(lifecycle_index, -1))
        lifecycle_logits = torch.nn.functional.one_hot(
            lifecycle_index.clamp(0, FAILED), num_classes=5
        ).to(lifecycle_logits.dtype) * 20.0

        unfinished_next = torch.zeros_like(parent_count).scatter_add(
            1,
            safe_child,
            ((lifecycle_index.gather(1, safe_parent) != FINISHED) & valid_dag).to(action.dtype),
        )
        dag_state = torch.stack((parent_count, unfinished_next, ((unfinished_next == 0) & task_valid).to(action.dtype)), dim=-1)

        masks = {name: torch.zeros_like(states[name], dtype=torch.bool) for name in COMPONENTS}
        service_masks = {name: torch.zeros_like(states[name], dtype=torch.bool) for name in COMPONENTS}
        masks["physical_edge"][..., 4] = edge_valid
        service_masks["physical_edge"][..., 2] = edge_valid
        masks["flow"] = active_flow.unsqueeze(-1).expand_as(masks["flow"]).clone()
        masks["task"] = task_valid.unsqueeze(-1).expand_as(masks["task"]).clone()
        if target_masks is not None:
            for name in COMPONENTS:
                if name in target_masks:
                    allowed = target_masks[name].bool()
                    masks[name] &= allowed
                    service_masks[name] &= allowed

        return RuleLayerResult(
            states=states,
            masks=masks,
            service_masks=service_masks,
            lifecycle_logits=lifecycle_logits,
            lifecycle_index=lifecycle_index,
            task_node_index=updated_task_nodes,
            task_present=previous_task_present.bool(),
            flow_present=active_flow,
            dag_state=dag_state,
            service_outcome={**service_outcome, "flow_delivered": delivered},
            cpu_allocation=cpu_allocation,
            cpu_served=cpu_served,
        )


__all__ = ["DeterministicRuleLayer", "NormalizationAdapter", "RuleLayerResult", "resolve_action_endpoints"]
