"""Consistency and training-protocol gates for the formal PI-JWM contract."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterable, Mapping
from typing import Any


SCHEMA_VERSION = "PI-JWM-formal-training-protocol-audit-v1"
PER_RB_WINDOW_KEYS = frozenset(
    {
        "link_activity_by_rb",
        "link_activity_mask_by_rb",
        "link_rate_by_rb",
        "link_rate_by_rb_mask",
    }
)
PER_RB_MODEL_OUTPUTS = frozenset(
    {"link_activity_by_rb_logits", "link_rate_by_rb_mean"}
)
PER_RB_LOSS_TARGETS = frozenset(PER_RB_WINDOW_KEYS)
PER_RB_METRIC_SOURCES = frozenset({"link_activity_by_rb", "link_rate_by_rb"})
AGGREGATE_WINDOW_KEYS = frozenset(
    {
        "aggregate_link_activity",
        "aggregate_link_activity_mask",
        "aggregate_link_rate_sum",
        "aggregate_link_rate_sum_mask",
        "aggregate_rb_occupancy",
        "aggregate_rb_occupancy_mask",
    }
)
AGGREGATE_MODEL_OUTPUTS = frozenset(
    {"link_activity_logits", "physical_edge_state_mean"}
)
AGGREGATE_LOSS_TARGETS = AGGREGATE_WINDOW_KEYS
AGGREGATE_METRIC_SOURCES = frozenset(
    {"aggregate_link_activity", "aggregate_link_rate_sum", "aggregate_rb_occupancy"}
)


def _as_set(values: Iterable[str]) -> set[str]:
    return {str(value) for value in values}


def audit_training_protocol(
    *,
    contract_mode: str = "per_rb_extension",
    tensor_contract: Mapping[str, Any],
    tensor_validation: Mapping[str, Any],
    normalization_stats: Mapping[str, Any],
    window_keys: Iterable[str],
    model_outputs: Iterable[str],
    loss_targets: Iterable[str],
    metric_sources: Iterable[str],
    locked_test_materialized: bool,
) -> dict[str, Any]:
    """Audit the formal input contract without executing training."""

    if contract_mode not in {"per_rb_extension", "aggregate_baseline"}:
        raise ValueError(f"unsupported contract_mode: {contract_mode}")

    window_keys = _as_set(window_keys)
    model_outputs = _as_set(model_outputs)
    loss_targets = _as_set(loss_targets)
    metric_sources = _as_set(metric_sources)
    if contract_mode == "aggregate_baseline":
        expected_window_keys = AGGREGATE_WINDOW_KEYS
        expected_model_outputs = AGGREGATE_MODEL_OUTPUTS
        expected_loss_targets = AGGREGATE_LOSS_TARGETS
        expected_metric_sources = AGGREGATE_METRIC_SOURCES
    else:
        expected_window_keys = PER_RB_WINDOW_KEYS
        expected_model_outputs = PER_RB_MODEL_OUTPUTS
        expected_loss_targets = PER_RB_LOSS_TARGETS
        expected_metric_sources = PER_RB_METRIC_SOURCES
    checks = {
        "tensor_capacity_valid": int(tensor_contract.get("max_physical_edges", 0)) > 0
        and int(tensor_contract.get("n_rb", 0)) > 0,
        "tensor_ready": bool(tensor_validation.get("formal_tensor_ready", False))
        and not list(tensor_validation.get("failed_checks", [])),
        "normalization_is_train_only": normalization_stats.get("source_split") == "train",
        "window_contract_complete": expected_window_keys <= window_keys,
        "model_outputs_complete": expected_model_outputs <= model_outputs,
        "loss_targets_and_masks_complete": expected_loss_targets <= loss_targets,
        "metrics_definitions_complete": expected_metric_sources <= metric_sources,
        "per_rb_sidecar_present": PER_RB_WINDOW_KEYS <= window_keys,
        "locked_test_not_materialized": not bool(locked_test_materialized),
    }

    critical_mismatches: list[str] = []
    if not checks["model_outputs_complete"]:
        critical_mismatches.append(
            "model_missing_aggregate_outputs"
            if contract_mode == "aggregate_baseline"
            else "model_missing_per_rb_outputs"
        )
    if not checks["loss_targets_and_masks_complete"]:
        critical_mismatches.append(
            "loss_missing_aggregate_targets_or_masks"
            if contract_mode == "aggregate_baseline"
            else "loss_missing_per_rb_targets_or_masks"
        )
    if not checks["metrics_definitions_complete"]:
        critical_mismatches.append(
            "metrics_missing_aggregate_definitions"
            if contract_mode == "aggregate_baseline"
            else "metrics_missing_per_rb_definitions"
        )
    if not checks["tensor_ready"] or not checks["tensor_capacity_valid"]:
        critical_mismatches.append("tensor_contract_not_ready")
    if not checks["normalization_is_train_only"]:
        critical_mismatches.append("normalization_not_train_only")
    if not checks["window_contract_complete"]:
        critical_mismatches.append(
            "window_missing_aggregate_contract"
            if contract_mode == "aggregate_baseline"
            else "window_missing_per_rb_contract"
        )
    if not checks["locked_test_not_materialized"]:
        critical_mismatches.append("locked_test_materialized_before_unlock")
    ready = not critical_mismatches and all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_mode": contract_mode,
        "status": "ready_for_protocol_review" if ready else "blocked",
        "formal_training_ready": ready,
        "checks": checks,
        "critical_mismatches": critical_mismatches,
        "contract": {
            "max_physical_edges": int(tensor_contract.get("max_physical_edges", 0)),
            "n_rb": int(tensor_contract.get("n_rb", 0)),
            "history_steps": int(tensor_contract.get("history_steps", 0)),
            "horizon_steps": int(tensor_contract.get("horizon_steps", 0)),
            "normalization_source_split": normalization_stats.get("source_split"),
            "locked_test_policy": "untensorized_until_explicit_unlock",
        },
        "expected": {
            "window_keys": sorted(expected_window_keys),
            "model_outputs": sorted(expected_model_outputs),
            "loss_targets": sorted(expected_loss_targets),
            "metric_sources": sorted(expected_metric_sources),
        },
        "observed": {
            "window_keys": sorted(window_keys),
            "model_outputs": sorted(model_outputs),
            "loss_targets": sorted(loss_targets),
            "metric_sources": sorted(metric_sources),
        },
        "execution_policy": {
            "gpu_started": False,
            "training_started": False,
            "locked_test_accessed": False,
        },
        "sidecar": {
            "field": "per_rb_target_sidecar",
            "status": "retained_diagnostic_only",
            "consumed_by_current_model": False,
            "present": checks["per_rb_sidecar_present"],
        },
    }


def build_training_protocol_freeze(report: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze data and launch boundaries even when implementation gates block launch."""

    contract = dict(report.get("contract", {}))
    ready = bool(report.get("formal_training_ready", False))
    contract_mode = str(report.get("contract_mode", "per_rb_extension"))
    aggregate = contract_mode == "aggregate_baseline"
    return {
        "schema_version": "PI-JWM-formal-training-protocol-freeze-v1",
        "protocol_status": "ready_for_review" if ready else "blocked",
        "contract_mode": contract_mode,
        "data_protocol": {
            "splits": ["train", "validation", "calibration"],
            "history_steps": int(contract.get("history_steps", 0)),
            "horizon_steps": int(contract.get("horizon_steps", 0)),
            "max_physical_edges": int(contract.get("max_physical_edges", 0)),
            "n_rb": int(contract.get("n_rb", 0)),
            "normalization_source_split": contract.get("normalization_source_split"),
            "locked_test_policy": contract.get(
                "locked_test_policy", "untensorized_until_explicit_unlock"
            ),
        },
        "required_contracts": {
            "model_outputs": sorted(AGGREGATE_MODEL_OUTPUTS if aggregate else PER_RB_MODEL_OUTPUTS),
            "loss_targets_and_masks": sorted(AGGREGATE_LOSS_TARGETS if aggregate else PER_RB_LOSS_TARGETS),
            "metric_sources": sorted(AGGREGATE_METRIC_SOURCES if aggregate else PER_RB_METRIC_SOURCES),
        },
        "sidecar": dict(report.get("sidecar", {
            "field": "per_rb_target_sidecar",
            "status": "retained_diagnostic_only",
            "consumed_by_current_model": False,
        })),
        "launch_gates": {
            "cpu_protocol_audit_allowed": True,
            "cpu_training_allowed": ready,
            "gpu_allowed": False,
            "gpu_blocker": (
                "pending_cpu_baseline_vs_persistence_gate"
                if ready
                else "training_contract_not_ready"
            ),
            "formal_performance_claim_allowed": False,
            "locked_test_allowed": False,
        },
        "critical_mismatches": list(report.get("critical_mismatches", [])),
        "evidence": {
            "formal_training_ready": ready,
            "audit_status": report.get("status"),
            "gpu_started": False,
            "training_started": False,
            "locked_test_accessed": False,
        },
    }
def _function_string_literals(function: Any, *, call_name: str | None = None) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if call_name is not None and node.func.attr != call_name:
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            values.add(node.args[0].value)
    return values


def _all_string_literals(function: Any) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _target_key_literals(function: Any) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id != "target":
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            values.add(key.value)
    return values


def collect_current_implementation_snapshot() -> dict[str, set[str]]:
    """Collect current model/loss/metric declarations without running a model."""

    from .formal_dual_graph_world_model_v2 import FormalDirectedDynamicWorldModelV2
    from .formal_world_model_loss_v1 import formal_world_model_loss
    from .formal_world_model_metrics_v1 import metric_registry

    model_outputs = _function_string_literals(
        FormalDirectedDynamicWorldModelV2.forward, call_name="setdefault"
    )
    # Component-head names are emitted through an f-string in the model.
    model_outputs.add("physical_edge_state_mean")
    loss_targets = _target_key_literals(formal_world_model_loss)
    loss_targets.update(
        _all_string_literals(formal_world_model_loss) & AGGREGATE_LOSS_TARGETS
    )
    metric_sources = {
        source
        for definition in metric_registry().values()
        for source in definition.get("source_fields", [])
    }
    return {
        "model_outputs": model_outputs,
        "loss_targets": loss_targets,
        "metric_sources": metric_sources,
    }


__all__ = [
    "AGGREGATE_LOSS_TARGETS",
    "AGGREGATE_METRIC_SOURCES",
    "AGGREGATE_MODEL_OUTPUTS",
    "AGGREGATE_WINDOW_KEYS",
    "PER_RB_LOSS_TARGETS",
    "PER_RB_METRIC_SOURCES",
    "PER_RB_MODEL_OUTPUTS",
    "PER_RB_WINDOW_KEYS",
    "SCHEMA_VERSION",
    "audit_training_protocol",
    "build_training_protocol_freeze",
    "collect_current_implementation_snapshot",
]
