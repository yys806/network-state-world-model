"""Causal, temporally executable candidate policies for PI-JWM rollouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


CAUSAL_POLICY_PROTOCOL = "causal_policy_v1"
LEGACY_TASK_PROTOCOL = "precomputed_task_v0"
SUPPORTED_FAMILIES = {
    "default",
    "rb_count",
    "offload_target",
    "mixed_offload_rb",
    "cpu_scale",
    "return_route",
}
TASK_ID_ACTION_FIELDS = (
    "rb_plan",
    "offload_overrides",
    "cpu_overrides",
    "return_route_overrides",
)


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def validate_temporal_candidate(candidate: Mapping[str, Any]) -> None:
    """Validate the causal policy schema without consulting simulator state."""

    if str(candidate.get("action_protocol")) != CAUSAL_POLICY_PROTOCOL:
        raise ValueError(f"action_protocol must be {CAUSAL_POLICY_PROTOCOL}")
    family = str(candidate.get("action_family", ""))
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported action_family: {family}")
    populated = [name for name in TASK_ID_ACTION_FIELDS if candidate.get(name)]
    if populated:
        raise ValueError(f"causal policy contains task-ID payload: {populated}")

    coverage = int(candidate.get("policy_coverage", 0))
    if family == "default":
        if coverage != 0:
            raise ValueError("default policy_coverage must be zero")
        return
    if coverage <= 0:
        raise ValueError("non-default policy_coverage must be positive")
    if family in {"rb_count", "mixed_offload_rb"}:
        _positive_finite(candidate.get("rb_scale", 1.0), "rb_scale")
    if family == "cpu_scale":
        _positive_finite(candidate.get("cpu_scale", 1.0), "cpu_scale")
    if int(candidate.get("policy_rank", 0)) < 0:
        raise ValueError("policy_rank must be nonnegative")
    if family == "return_route" and candidate.get("return_route_mode") not in {
        "nearest_rsu",
        "uav_relay",
    }:
        raise ValueError("unsupported return_route_mode")


def _policy_candidate_id(family: str, candidate: Mapping[str, Any], coverage: int) -> str:
    rank = int(candidate.get("policy_rank", 1))
    if family == "default":
        return "default"
    if family == "rb_count":
        return f"rb_scale_{float(candidate.get('rb_scale', 1.0)):g}_k{coverage}"
    if family == "offload_target":
        return f"offload_rank_{rank}_k{coverage}"
    if family == "mixed_offload_rb":
        return (
            f"mixed_offload_rank_{rank}_rb_{float(candidate.get('rb_scale', 1.0)):g}"
            f"_k{coverage}"
        )
    if family == "cpu_scale":
        return f"cpu_scale_{float(candidate.get('cpu_scale', 1.0)):g}_k{coverage}"
    mode = str(candidate.get("return_route_mode", "nearest_rsu"))
    return f"return_{mode}_rank_{rank}_k{coverage}"


def to_causal_policy_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a decision-time descriptor into a task-ID-free policy intent."""

    result = dict(candidate)
    source_id = str(result.get("candidate_id", ""))
    family = (
        "default"
        if source_id in {"default", "default_no_rb"}
        else str(result.get("action_family", ""))
    )
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported action_family: {family}")
    result["action_family"] = family

    requested = [
        int(result.get("num_offload_overrides", 0)),
        int(result.get("num_cpu_overrides", 0)),
        int(result.get("num_return_route_overrides", 0)),
        len(result.get("rb_plan", {})),
    ]
    coverage = 0 if family == "default" else max(1, max(requested, default=0))
    result["action_protocol"] = CAUSAL_POLICY_PROTOCOL
    result["policy_coverage"] = coverage
    result["policy_rank"] = int(result.get("policy_rank", 1 if family != "default" else 0))

    if family == "return_route" and "return_route_mode" not in result:
        routes = list(result.get("return_route_overrides", {}).values())
        result["return_route_mode"] = (
            "uav_relay" if any(len(route) > 1 for route in routes) else "nearest_rsu"
        )
    for field in TASK_ID_ACTION_FIELDS:
        result[field] = {}
    result["candidate_id"] = _policy_candidate_id(family, result, coverage)
    validate_temporal_candidate(result)
    return result


def project_scaled_counts(
    default_counts: Mapping[str, int],
    scale: float,
    capacity: int,
) -> dict[str, int]:
    """Scale integer allocations with deterministic largest-remainder projection."""

    factor = _positive_finite(scale, "scale")
    limit = max(0, int(capacity))
    positive = {
        str(key): max(0, int(value))
        for key, value in default_counts.items()
        if int(value) > 0
    }
    if not positive or limit == 0:
        return {}

    keys = sorted(positive)
    desired = [positive[key] * factor for key in keys]
    minimum_total = min(len(keys), limit)
    target_total = min(limit, max(minimum_total, int(math.floor(sum(desired) + 0.5))))
    weight_total = sum(desired)
    ideal = [value / weight_total * target_total for value in desired]
    base = [int(math.floor(value)) for value in ideal]

    if target_total >= len(keys):
        base = [max(1, value) for value in base]
    while sum(base) > target_total:
        candidates = [index for index, value in enumerate(base) if value > 1]
        index = min(candidates, key=lambda item: (ideal[item] - base[item], keys[item]))
        base[index] -= 1
    while sum(base) < target_total:
        index = max(
            range(len(keys)),
            key=lambda item: (ideal[item] - base[item], tuple(-ord(ch) for ch in keys[item])),
        )
        base[index] += 1
    return {key: value for key, value in zip(keys, base) if value > 0}


def summarize_active_action_steps(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Aggregate active-step legality separately from actual action change."""

    return {
        "action_applicable": bool(rows)
        and all(bool(row.get("action_applicable")) for row in rows),
        "action_changed": any(bool(row.get("action_changed")) for row in rows),
    }
