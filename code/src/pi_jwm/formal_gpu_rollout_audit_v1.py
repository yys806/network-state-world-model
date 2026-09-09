"""Horizon-wise audit for completed formal non-locked GPU evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HORIZONS = ("k=1", "k=2", "k=3")
METRICS = {
    "node_x_mae": "state.node.x.mae",
    "link_active_rate_mae": "link.active_only_rate.mae",
    "throughput_mae": "system.communication_throughput.mae",
    "rb_occupancy_mae": "resource.rb_occupancy.mae",
    "node_x_coverage_95": "uncertainty.node.x.coverage_95",
}


def _value(metrics: dict[str, Any], name: str) -> float:
    item = metrics.get(name)
    if not isinstance(item, dict) or item.get("value") is None:
        raise ValueError(f"missing rollout metric: {name}")
    return float(item["value"])


def _optional_value(metrics: dict[str, Any], name: str) -> float | None:
    item = metrics.get(name)
    if not isinstance(item, dict) or item.get("value") is None:
        return None
    return float(item["value"])


def _read_method(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("horizons"), dict):
        raise ValueError(f"invalid rollout metrics: {path}")
    return value


def audit_gpu_rollout_metrics(
    metrics_dir: str | Path,
    *,
    learned_method: str = "coupled_dual_gnn_residual",
    persistence_method: str = "last_persistence",
) -> dict[str, Any]:
    """Compare the learned model and persistence across the three predicted steps."""

    directory = Path(metrics_dir)
    if "locked_test" in str(directory).lower():
        raise ValueError("locked_test metrics are forbidden")
    learned = _read_method(directory / f"{learned_method}__validation.json")
    persistence = _read_method(directory / f"{persistence_method}__validation.json")
    learned_horizons = learned["horizons"]
    persistence_horizons = persistence["horizons"]
    if any(h not in learned_horizons or h not in persistence_horizons for h in HORIZONS):
        raise ValueError("missing horizon rollout metric")
    rows: dict[str, Any] = {}
    for horizon in HORIZONS:
        learned_metrics = learned_horizons[horizon].get("metrics", {})
        persistence_metrics = persistence_horizons[horizon].get("metrics", {})
        row: dict[str, float] = {}
        for output_name, source_name in METRICS.items():
            learned_value = _value(learned_metrics, source_name)
            persistence_value = _optional_value(persistence_metrics, source_name)
            row[f"learned_{output_name}"] = learned_value
            row[f"persistence_{output_name}"] = persistence_value
            row[f"{output_name}_delta"] = (
                learned_value - persistence_value if persistence_value is not None else None
            )
        rows[horizon] = row
    first = rows[HORIZONS[0]]["learned_node_x_mae"]
    last = rows[HORIZONS[-1]]["learned_node_x_mae"]
    persistence_first = rows[HORIZONS[0]]["persistence_node_x_mae"]
    persistence_last = rows[HORIZONS[-1]]["persistence_node_x_mae"]
    return {
        "schema_version": "PI-JWM-formal-gpu-rollout-audit-v1",
        "audit_passed": True,
        "formal_performance_claim_ready": False,
        "learned_method": learned_method,
        "persistence_method": persistence_method,
        "horizons": list(HORIZONS),
        "per_horizon": rows,
        "learned_node_x_error_growth": last / first if first else None,
        "persistence_node_x_error_growth": persistence_last / persistence_first if persistence_first else None,
        "execution_policy": {
            "locked_test_accessed": False,
            "result_boundary": "Nonlocked aggregate-baseline horizon-wise rollout diagnosis only.",
        },
    }


__all__ = ["audit_gpu_rollout_metrics"]
