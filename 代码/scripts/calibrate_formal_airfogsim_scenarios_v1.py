from __future__ import annotations

"""Probe candidate PI-JWM scenarios with short, real AirFogSim trajectories."""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.formal_airfogsim_dataset_v1 import DEFAULT_SCENARIOS, TrajectorySpec


SCHEMA_VERSION = "PI-JWM-AirFogSim-formal-calibration-v1"
LOAD_ORDER = ("low", "medium", "high")
DENSITY_ORDER = ("sparse", "dense")
STAT_FIELDS = (
    "task_count",
    "mean_concurrent_tasks",
    "physical_node_count",
    "physical_edge_count",
    "link_active_rate",
    "cpu_utilization",
)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def summarize_calibration(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate probes and accept only complete, monotonic scenario evidence."""

    expected_seed_set = {int(seed) for seed in expected_seeds}
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        by_scenario[str(row["scenario_id"])].append(row)

    scenario_summaries: list[dict[str, Any]] = []
    for scenario in DEFAULT_SCENARIOS:
        rows = by_scenario.get(scenario.scenario_id, [])
        scenario_summaries.append(
            {
                "scenario_id": scenario.scenario_id,
                "load_level": scenario.load_level,
                "density_level": scenario.density_level,
                "seeds": sorted({int(row["seed"]) for row in rows}),
                "probe_count": len(rows),
                **{
                    f"mean_{field}": _mean(rows, field) if rows else None
                    for field in STAT_FIELDS
                },
            }
        )

    summary_by_pair = {
        (row["load_level"], row["density_level"]): row
        for row in scenario_summaries
    }
    load_task_count_monotonic = all(
        all(
            summary_by_pair[(left, density)]["mean_task_count"]
            <= summary_by_pair[(right, density)]["mean_task_count"]
            for left, right in zip(LOAD_ORDER, LOAD_ORDER[1:])
        )
        for density in DENSITY_ORDER
    ) if all(
        summary_by_pair[(load, density)]["mean_task_count"] is not None
        for load in LOAD_ORDER
        for density in DENSITY_ORDER
    ) else False
    density_node_count_monotonic = all(
        summary_by_pair[(load, "sparse")]["mean_physical_node_count"]
        <= summary_by_pair[(load, "dense")]["mean_physical_node_count"]
        for load in LOAD_ORDER
    ) if all(
        summary_by_pair[(load, density)]["mean_physical_node_count"] is not None
        for load in LOAD_ORDER
        for density in DENSITY_ORDER
    ) else False
    two_repetitions_per_scenario = all(
        set(row["seeds"]) == expected_seed_set
        and row["probe_count"] == len(expected_seed_set)
        for row in scenario_summaries
    )
    nonempty_observations = all(
        row["mean_task_count"] is not None
        and row["mean_task_count"] > 0
        and row["mean_physical_node_count"] is not None
        and row["mean_physical_node_count"] > 0
        for row in scenario_summaries
    )
    checks = {
        "six_scenarios_present": len(by_scenario) == 6
        and all(row["probe_count"] > 0 for row in scenario_summaries),
        "two_repetitions_per_scenario": two_repetitions_per_scenario,
        "nonempty_observations": nonempty_observations,
        "load_task_count_monotonic": load_task_count_monotonic,
        "density_node_count_monotonic": density_node_count_monotonic,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        **checks,
        "calibration_ready": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "scenario_summaries": scenario_summaries,
    }


def _extract_probe_row(
    runtime: Mapping[str, Any], spec: TrajectorySpec
) -> dict[str, Any]:
    source = runtime["source_bundle"]
    resource = runtime["bundle"]
    task_snapshots = list(source.get("task_snapshots", []))
    active_by_time: dict[float, int] = defaultdict(int)
    for row in task_snapshots:
        if row.get("observed_time") is None:
            continue
        state = str(row.get("lifecycle_state", ""))
        if state not in {"finished", "failed", "removed"}:
            active_by_time[round(float(row["observed_time"]), 6)] += 1
    edge_snapshots = list(source.get("physical_edge_snapshots", []))
    active_edges = sum(
        float(row.get("rate_sum", 0.0)) > 0.0
        or int(row.get("active_task_count", 0)) > 0
        for row in edge_snapshots
    )
    cpu_rows = list(resource.get("cpu_ledger", []))
    cpu_fractions = [
        float(row.get("allocated_fraction", 0.0)) for row in cpu_rows
    ]
    return {
        "scenario_id": spec.scenario.scenario_id,
        "load_level": spec.scenario.load_level,
        "density_level": spec.scenario.density_level,
        "seed": int(spec.seed),
        "task_count": len(source.get("information_nodes", [])),
        "mean_concurrent_tasks": statistics.fmean(active_by_time.values())
        if active_by_time
        else 0.0,
        "physical_node_count": len(source.get("physical_nodes", [])),
        "physical_edge_count": len(source.get("physical_edges", [])),
        "link_active_rate": active_edges / len(edge_snapshots)
        if edge_snapshots
        else 0.0,
        "cpu_utilization": statistics.fmean(cpu_fractions) if cpu_fractions else 0.0,
        "cpu_action_count": len(cpu_rows),
        "dag_edge_count": len(source.get("information_edges", [])),
    }


def run_calibration(
    *,
    seeds: Sequence[int],
    seconds: float,
    runtime_runner: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(DEFAULT_SCENARIOS):
        for repetition, base_seed in enumerate(seeds):
            spec = TrajectorySpec(
                trajectory_id=f"calibration__{scenario.scenario_id}__s{int(base_seed)}",
                seed=int(base_seed) + scenario_index * 1000,
                repetition=repetition,
                split="calibration",
                cpu_policy="equal_share",
                scenario=scenario,
            )
            runtime = runtime_runner(spec, max_time=float(seconds))
            row = _extract_probe_row(runtime, spec)
            row["seed"] = int(base_seed)
            row["runtime_seed"] = int(spec.seed)
            rows.append(row)
    return rows, summarize_calibration(rows, expected_seeds=seeds)


def _write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "probe_rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_output_dir() -> Path:
    return CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1_calibration"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate formal AirFogSim scenarios.")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[900, 901])
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    from formal_airfogsim_runtime_v1 import run_formal_airfogsim_trajectory

    rows, report = run_calibration(
        seeds=args.seeds,
        seconds=args.seconds,
        runtime_runner=run_formal_airfogsim_trajectory,
    )
    _write_outputs(args.output_dir, rows, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["calibration_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
