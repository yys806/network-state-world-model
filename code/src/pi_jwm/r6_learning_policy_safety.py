"""Shared action safety projection for R6 rule and learning policies."""

from __future__ import annotations

import torch
from torch import Tensor

from .r6_learning_policy_contract import (
    ActionSpec,
    ExecutableAction,
    PolicyState,
    ProjectionRecord,
    ProjectionResult,
    ProposedAction,
)


def _validate_shapes(state: PolicyState, proposed: ProposedAction, spec: ActionSpec) -> None:
    batch = state.batch_size
    if proposed.offload_index.shape != (batch,):
        raise ValueError("offload_index shape must be [batch]")
    if proposed.rb_index.shape != (batch,):
        raise ValueError("rb_index shape must be [batch]")
    if proposed.cpu_allocation.shape != (batch, spec.cpu_task_count):
        raise ValueError("cpu_allocation shape does not match ActionSpec")
    if state.offload_mask.shape != (batch, spec.offload_count):
        raise ValueError("offload_mask shape does not match ActionSpec")
    if state.rb_mask.shape != (batch, spec.rb_count):
        raise ValueError("rb_mask shape does not match ActionSpec")
    if state.cpu_task_mask.shape != (batch, spec.cpu_task_count):
        raise ValueError("cpu_task_mask shape does not match ActionSpec")


def _validate_finite(proposed: ProposedAction) -> None:
    for name, value in (
        ("offload_index", proposed.offload_index),
        ("rb_index", proposed.rb_index),
        ("cpu_allocation", proposed.cpu_allocation),
    ):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def _project_discrete(
    proposed: Tensor,
    mask: Tensor,
    noop_index: int,
    family: str,
) -> tuple[Tensor, list[ProjectionRecord], int]:
    result = proposed.detach().clone().long()
    records: list[ProjectionRecord] = []
    fallback_count = 0
    for batch_index in range(result.shape[0]):
        before = int(result[batch_index].item())
        legal = 0 <= before < mask.shape[1] and bool(mask[batch_index, before].item())
        if not legal:
            result[batch_index] = int(noop_index)
            reason = f"masked_{family}_to_noop"
            records.append(
                ProjectionRecord(batch_index, family, reason, float(before), float(noop_index))
            )
            fallback_count += 1
    return result, records, fallback_count


def _project_cpu(
    proposed: Tensor,
    mask: Tensor,
    capacity: Tensor,
    task_node_index: Tensor,
) -> tuple[Tensor, list[ProjectionRecord]]:
    result = proposed.detach().clone().to(torch.float32)
    records: list[ProjectionRecord] = []
    for batch_index in range(result.shape[0]):
        for task_index in range(result.shape[1]):
            before = float(result[batch_index, task_index].item())
            if not bool(mask[batch_index, task_index].item()) and before != 0.0:
                result[batch_index, task_index] = 0.0
                records.append(
                    ProjectionRecord(batch_index, "cpu", "masked_cpu_task", before, 0.0)
                )
            elif before < 0.0:
                result[batch_index, task_index] = 0.0
                records.append(
                    ProjectionRecord(batch_index, "cpu", "negative_cpu_clamped", before, 0.0)
                )
        for node_index in range(capacity.shape[1]):
            selected = mask[batch_index] & (task_node_index[batch_index] == node_index)
            total = float(result[batch_index, selected].sum().item())
            limit = float(capacity[batch_index, node_index].item())
            if total > limit + 1e-7:
                scale = 0.0 if total <= 0.0 else limit / total
                result[batch_index, selected] *= scale
                projected_total = float(result[batch_index, selected].sum().item())
                if projected_total > limit:
                    selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
                    largest_local_index = int(
                        torch.argmax(result[batch_index, selected_indices]).item()
                    )
                    largest_task_index = int(selected_indices[largest_local_index].item())
                    rounding_guard = torch.finfo(result.dtype).eps * max(abs(limit), 1.0)
                    correction = projected_total - limit + rounding_guard
                    result[batch_index, largest_task_index] = torch.clamp(
                        result[batch_index, largest_task_index] - correction,
                        min=0.0,
                    )
                records.append(
                    ProjectionRecord(
                        batch_index,
                        "cpu",
                        "cpu_capacity_projection",
                        total,
                        float(result[batch_index, selected].sum().item()),
                    )
                )
    return result, records


class SafetyProjector:
    """Apply the frozen R6 projection order and preserve an audit trail."""

    def project(
        self,
        state: PolicyState,
        proposed: ProposedAction,
        spec: ActionSpec,
    ) -> ProjectionResult:
        _validate_shapes(state, proposed, spec)
        _validate_finite(proposed)
        offload, offload_rows, offload_fallback = _project_discrete(
            proposed.offload_index,
            state.offload_mask,
            spec.offload_noop_index,
            "offload",
        )
        rb, rb_rows, rb_fallback = _project_discrete(
            proposed.rb_index,
            state.rb_mask,
            spec.rb_noop_index,
            "rb",
        )
        cpu, cpu_rows = _project_cpu(
            proposed.cpu_allocation,
            state.cpu_task_mask,
            state.cpu_capacity,
            state.cpu_task_node_index,
        )
        if (cpu < -1e-7).any():
            raise ValueError("post-projection CPU allocation is negative")
        for batch_index in range(state.batch_size):
            for node_index in range(state.cpu_capacity.shape[1]):
                selected = state.cpu_task_mask[batch_index] & (
                    state.cpu_task_node_index[batch_index] == node_index
                )
                if cpu[batch_index, selected].sum() > state.cpu_capacity[batch_index, node_index] + 1e-7:
                    raise ValueError("post-projection CPU capacity violation")
        return ProjectionResult(
            action=ExecutableAction(offload, rb, cpu),
            records=tuple(offload_rows + rb_rows + cpu_rows),
            fallback_count=offload_fallback + rb_fallback,
        )
