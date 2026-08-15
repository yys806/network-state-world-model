"""Seed-safe window and metric aggregation for AirFogSim dual-graph v2."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "PI-JWM-AirFogSim-multiseed-dataset-v2"


def _validated_time_grid(values: Sequence[float]) -> tuple[list[float], float]:
    times = sorted({float(value) for value in values})
    if len(times) < 2:
        raise ValueError("each seed requires at least two distinct time points")
    interval = times[1] - times[0]
    if interval <= 0 or any(
        not math.isclose(right - left, interval, rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(times, times[1:])
    ):
        raise ValueError("each seed must use a uniform time grid")
    return times, interval


def build_multiseed_window_index(
    times_by_seed: Mapping[int, Sequence[float]],
    split_by_seed: Mapping[int, str],
    *,
    history_steps: int,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    """Create windows whose history and labels stay inside one seed trajectory."""

    history_steps = int(history_steps)
    horizon_steps = int(horizon_steps)
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be positive")
    if set(times_by_seed) != set(split_by_seed):
        raise ValueError("times_by_seed and split_by_seed must contain the same seeds")

    rows: list[dict[str, Any]] = []
    for seed in sorted(times_by_seed):
        times, interval = _validated_time_grid(times_by_seed[seed])
        window_count = max(len(times) - history_steps - horizon_steps + 1, 0)
        for start in range(window_count):
            input_end = start + history_steps
            label_end = input_end + horizon_steps
            rows.append(
                {
                    "sample_id": f"seed{int(seed):03d}::window{start:06d}",
                    "seed": int(seed),
                    "split": str(split_by_seed[seed]),
                    "simulation_interval": interval,
                    "input_start_index": start,
                    "input_end_index": input_end,
                    "label_start_index": input_end,
                    "label_end_index": label_end,
                    "history_start_time": times[start],
                    "decision_time": times[input_end - 1],
                    "label_start_time": times[input_end],
                    "label_end_time": times[label_end - 1],
                }
            )
    return rows


def validate_multiseed_window_index(
    rows: Sequence[Mapping[str, Any]],
    times_by_seed: Mapping[int, Sequence[float]],
    split_by_seed: Mapping[int, str],
    *,
    history_steps: int,
    horizon_steps: int,
) -> dict[str, Any]:
    grids = {int(seed): _validated_time_grid(times)[0] for seed, times in times_by_seed.items()}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    sample_ids = [str(row.get("sample_id")) for row in rows]
    add("sample_ids_unique", len(sample_ids) == len(set(sample_ids)), "Every window ID is unique.")
    add(
        "seed_split_isolation",
        all(
            int(row.get("seed", -1)) in split_by_seed
            and str(row.get("split")) == str(split_by_seed[int(row["seed"])])
            for row in rows
        ),
        "Each complete seed belongs to one development split.",
    )
    bounds_valid = True
    times_valid = True
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed not in grids:
            bounds_valid = False
            times_valid = False
            continue
        times = grids[seed]
        input_start = int(row.get("input_start_index", -1))
        input_end = int(row.get("input_end_index", -1))
        label_start = int(row.get("label_start_index", -1))
        label_end = int(row.get("label_end_index", -1))
        valid = (
            0 <= input_start < input_end == label_start < label_end <= len(times)
            and input_end - input_start == int(history_steps)
            and label_end - label_start == int(horizon_steps)
        )
        bounds_valid &= valid
        if not valid:
            times_valid = False
            continue
        times_valid &= all(
            math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-9)
            for observed, expected in (
                (row.get("history_start_time"), times[input_start]),
                (row.get("decision_time"), times[input_end - 1]),
                (row.get("label_start_time"), times[label_start]),
                (row.get("label_end_time"), times[label_end - 1]),
            )
        )
    add("window_bounds_valid", bounds_valid, "History and label indices have the requested lengths.")
    add("window_times_aligned", times_valid, "Window timestamps match the source seed time grid.")
    failed = [row["name"] for row in checks if not row["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "window_index_valid": not failed,
        "failed_checks": failed,
        "checks": checks,
        "window_count": len(rows),
        "seed_count": len(times_by_seed),
    }


def aggregate_metric_reports(
    reports_by_seed: Mapping[int, Mapping[str, Any]],
    split_by_seed: Mapping[int, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Flatten per-seed metrics and summarize only directly available values."""

    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    for seed in sorted(reports_by_seed):
        if seed not in split_by_seed:
            raise ValueError(f"missing split for seed {seed}")
        for metric in reports_by_seed[seed].get("metrics", []):
            row = dict(metric)
            row["seed"] = int(seed)
            row["split"] = str(split_by_seed[seed])
            rows.append(row)
            metric_names.add(str(row.get("name")))

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_name[str(row.get("name"))].append(row)
    summary: dict[str, dict[str, Any]] = {}
    total_seed_count = len(reports_by_seed)
    for name in sorted(metric_names):
        metric_rows = by_name[name]
        values = [
            float(row["value"])
            for row in metric_rows
            if row.get("status") == "available" and row.get("value") is not None
        ]
        not_computable = sum(row.get("status") == "not_computable" for row in metric_rows)
        not_applicable = sum(row.get("status") == "not_applicable" for row in metric_rows)
        summary[name] = {
            "available_seed_count": len(values),
            "not_computable_seed_count": not_computable,
            "not_applicable_seed_count": not_applicable,
            "missing_seed_count": total_seed_count - len(metric_rows),
            "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "unit": next((str(row.get("unit")) for row in metric_rows), None),
        }
    return rows, summary
