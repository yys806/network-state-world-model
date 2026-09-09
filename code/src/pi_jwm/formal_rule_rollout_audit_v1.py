"""Rule-rollout invariant auditing for the formal PI-JWM aggregate baseline."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import torch

from .formal_deterministic_rule_layer_v1 import FAILED, FINISHED, _task_edges


def _count(mask: torch.Tensor) -> int:
    return int(mask.detach().sum().item())


def audit_rule_step(
    *,
    result: Any,
    previous_states: dict[str, torch.Tensor],
    static: dict[str, torch.Tensor],
    n_rb: int,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Check one physical-unit deterministic rule update without mutation."""

    if n_rb <= 0:
        raise ValueError("n_rb must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    violations: list[str] = []
    details: dict[str, Any] = {}
    flow = result.states["flow"]
    active_flow = result.flow_present.bool()
    flow_total = flow[..., 0]
    flow_remaining = flow[..., 1]
    flow_cumulative = flow[..., 2]
    flow_delivered = flow[..., 3]
    flow_balance_error = (flow_total - flow_remaining - flow_cumulative).abs()
    active_balance_error = flow_balance_error.masked_select(active_flow)
    details["max_active_flow_balance_error"] = (
        float(active_balance_error.max().item()) if active_balance_error.numel() else 0.0
    )
    if active_balance_error.numel() and bool((active_balance_error > tolerance).any()):
        violations.append("flow_conservation")
    active_flow_values = torch.stack(
        (flow_total, flow_remaining, flow_cumulative, flow_delivered), dim=-1
    ).masked_select(active_flow.unsqueeze(-1).expand(-1, -1, 4))
    if active_flow_values.numel() and bool((active_flow_values < -tolerance).any()):
        violations.append("flow_nonnegative")
    service_delivered_error = (flow_delivered - result.service_outcome["flow_delivered"]).abs()
    active_service_error = service_delivered_error.masked_select(active_flow)
    details["max_flow_service_output_error"] = (
        float(active_service_error.max().item()) if active_service_error.numel() else 0.0
    )
    if active_service_error.numel() and bool((active_service_error > tolerance).any()):
        violations.append("flow_service_output_mismatch")

    rb_occupancy = result.states["physical_edge"][..., 4]
    details["max_rb_occupancy"] = float(rb_occupancy.max().item()) if rb_occupancy.numel() else 0.0
    if bool((rb_occupancy < -tolerance).any()) or bool((rb_occupancy > n_rb + tolerance).any()):
        violations.append("rb_capacity")

    task_valid = static["task_valid"].bool() & result.task_present.bool()
    lifecycle = result.lifecycle_index
    if bool(((lifecycle < -1) | (lifecycle > FAILED)).any()):
        violations.append("lifecycle_domain")
    if bool((lifecycle.masked_select(~task_valid) != -1).any()):
        violations.append("lifecycle_invalid_task")

    parents, children = _task_edges(static["dag_edge_index"])
    dag_valid = static["dag_edge_valid"].bool()
    task_count = lifecycle.shape[1]
    valid_edges = dag_valid & (parents >= 0) & (parents < task_count) & (children >= 0) & (children < task_count)
    safe_parent = parents.clamp(0, max(task_count - 1, 0))
    safe_child = children.clamp(0, max(task_count - 1, 0))
    expected_parent_count = torch.zeros_like(result.dag_state[..., 0]).scatter_add(
        1, safe_child, valid_edges.to(result.dag_state.dtype)
    )
    expected_unfinished = torch.zeros_like(expected_parent_count).scatter_add(
        1,
        safe_child,
        ((lifecycle.gather(1, safe_parent) != FINISHED) & valid_edges).to(result.dag_state.dtype),
    )
    dag_error = torch.stack(
        (
            (result.dag_state[..., 0] - expected_parent_count).abs(),
            (result.dag_state[..., 1] - expected_unfinished).abs(),
            (result.dag_state[..., 2] - ((expected_unfinished == 0) & task_valid).to(result.dag_state.dtype)).abs(),
        ),
        dim=-1,
    )
    details["max_dag_state_error"] = float(dag_error.max().item()) if dag_error.numel() else 0.0
    if bool((dag_error > tolerance).any()):
        violations.append("dag_recursion")

    slot = static["slot_seconds"].to(result.cpu_allocation).reshape(-1, 1)
    allocation = result.cpu_allocation
    served = result.cpu_served
    previous_task = previous_states["task"]
    remaining_cpu = (previous_task[..., 2] - previous_task[..., 6]).clamp_min(0.0)
    details["max_cpu_service_error"] = float((served - torch.minimum(remaining_cpu, allocation * slot)).abs().max().item())
    if bool((allocation < -tolerance).any()) or bool((served < -tolerance).any()):
        violations.append("cpu_nonnegative")
    if bool((served - allocation * slot > tolerance).any()) or bool((served - remaining_cpu > tolerance).any()):
        violations.append("cpu_service_cap")
    task_nodes = result.task_node_index
    execution_node = torch.where(task_nodes[..., 2] >= 0, task_nodes[..., 2], task_nodes[..., 1])
    node_capacity = previous_states["node"][..., 5].clamp_min(0.0)
    safe_node = execution_node.clamp(0, max(node_capacity.shape[1] - 1, 0))
    allocated_by_node = torch.zeros_like(node_capacity).scatter_add(
        1, safe_node, allocation * (execution_node >= 0).to(allocation.dtype)
    )
    details["max_cpu_capacity_excess"] = float((allocated_by_node - node_capacity).clamp_min(0.0).max().item())
    if bool((allocated_by_node - node_capacity > tolerance).any()):
        violations.append("cpu_capacity")

    checked_tensors = [flow, rb_occupancy, lifecycle, result.dag_state, allocation, served]
    if any(not torch.isfinite(value.float()).all() for value in checked_tensors):
        violations.append("nonfinite_rule_output")
    details["active_flow_count"] = _count(active_flow)
    details["active_task_count"] = _count(task_valid)
    return {"passed": not violations, "violations": violations, "details": details}


def summarize_rule_rollout_reports(
    reports: Sequence[Mapping[str, Any]], *, expected_step_count: int
) -> dict[str, Any]:
    """Summarize captured rule steps and reject incomplete recursive traces."""

    if expected_step_count < 0:
        raise ValueError("expected_step_count must be nonnegative")
    violation_counts = Counter(
        violation for report in reports for violation in report.get("violations", [])
    )
    observed_step_count = len(reports)
    missing_steps = max(expected_step_count - observed_step_count, 0)
    return {
        "audit_passed": observed_step_count == expected_step_count and not violation_counts,
        "expected_step_count": expected_step_count,
        "observed_step_count": observed_step_count,
        "missing_step_count": missing_steps,
        "unexpected_step_count": max(observed_step_count - expected_step_count, 0),
        "violation_counts": dict(sorted(violation_counts.items())),
        "active_flow_count": sum(
            int(report.get("details", {}).get("active_flow_count", 0)) for report in reports
        ),
        "active_task_count": sum(
            int(report.get("details", {}).get("active_task_count", 0)) for report in reports
        ),
    }


def audit_rule_rollout(model: Any, batch: Mapping[str, Any], *, n_rb: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one model batch while capturing every deterministic rule invocation."""

    rule_layer = getattr(model, "deterministic_rules", None)
    if rule_layer is None:
        raise ValueError("rule rollout audit requires deterministic_rule_layer=True")
    reports: list[dict[str, Any]] = []

    def _capture(_module: Any, _args: tuple[Any, ...], kwargs: Mapping[str, Any], result: Any) -> None:
        physical_previous_states = {
            name: rule_layer.normalization.to_physical(name, value)
            for name, value in kwargs["previous_states"].items()
        }
        reports.append(
            audit_rule_step(
                result=result,
                previous_states=physical_previous_states,
                static=kwargs["static"],
                n_rb=n_rb,
            )
        )

    handle = rule_layer.register_forward_hook(_capture, with_kwargs=True)
    try:
        prediction = model(batch)
    finally:
        handle.remove()
    expected_steps = int(getattr(getattr(model, "config", None), "horizon_steps", 0))
    return prediction, summarize_rule_rollout_reports(reports, expected_step_count=expected_steps)


__all__ = ["audit_rule_rollout", "audit_rule_step", "summarize_rule_rollout_reports"]
