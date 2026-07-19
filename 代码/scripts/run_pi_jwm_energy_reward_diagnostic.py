"""Run the local PI-JWM UAV-energy and step-reward diagnostic loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_diagnostics import (
    ENERGY_COMPONENTS,
    audit_diagnostic_rows,
    energy_step_metrics,
    paired_candidate_effects,
    reward_components,
    summarize_candidate_steps,
)
from pi_jwm.airfogsim_runtime import capture_energy_manager_snapshot
from pi_jwm.v11_physical_benefit import align_decision_points
from run_airfogsim_counterfactual_action_smoke_v0 import (
    channel_throughput,
    current_counts,
    make_env,
    step_default_until,
)
from run_airfogsim_counterfactual_multifamily_v0 import (
    apply_cpu_overrides,
    apply_offload_overrides,
    apply_return_route_overrides,
    build_extended_candidates,
    collect_extended_context,
    discover_decision_points,
)


DEFAULT_OUTPUT_DIR = CODE_ROOT / "artifacts" / "reports" / "pi_jwm_energy_reward_diagnostic_20260713"
EFFECT_METRICS = ("task_utility", "throughput_delta", "rb_total", "cpu_total", "energy_total")
SENSITIVITY_LAMBDAS = (0.0, 0.25, 0.5, 1.0)


def temporal_pattern_active(step, intervention_start_step, temporal_pattern):
    """Return whether a precomputed intervention is active at this rollout step."""

    current = int(step)
    start = int(intervention_start_step)
    pattern = str(temporal_pattern).strip().lower()
    if pattern == "persistent":
        return current >= start
    if pattern == "decayed":
        return current == start
    raise ValueError(f"unknown temporal pattern: {temporal_pattern}")


def align_points_to_sample_index(points, sample_index_rows):
    """Expose the framework's exact alignment contract to the runner."""

    return align_decision_points(points, sample_index_rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run PI-JWM energy/reward counterfactual diagnostics.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--scan-step-limit", type=int, default=80)
    parser.add_argument("--decision-times-per-seed", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--sample-index-csv", type=Path)
    parser.add_argument("--intervention-start-step", type=int, default=0)
    parser.add_argument(
        "--temporal-patterns",
        nargs="+",
        choices=("persistent", "decayed"),
        default=["persistent"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def build_reproduction_command(args):
    """Serialize every physical-label protocol argument into one command."""

    tokens = [
        "conda",
        "run",
        "-n",
        "airfogsim",
        "python",
        f"代码/scripts/{Path(__file__).name}",
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--max-time",
        f"{args.max_time:g}",
        "--scan-step-limit",
        str(args.scan_step_limit),
        "--decision-times-per-seed",
        str(args.decision_times_per_seed),
        "--horizon",
        str(args.horizon),
        "--max-candidates",
        str(args.max_candidates),
        "--intervention-start-step",
        str(args.intervention_start_step),
        "--temporal-patterns",
        *[str(value) for value in args.temporal_patterns],
        "--output-dir",
        str(args.output_dir),
    ]
    if args.sample_index_csv is not None:
        insert_at = tokens.index("--output-dir")
        tokens[insert_at:insert_at] = [
            "--sample-index-csv",
            str(args.sample_index_csv),
        ]
    return shlex.join(tokens)


def select_balanced_candidates(candidates, max_candidates):
    """Keep the default first, then one candidate per action family before filling."""
    limit = max(1, int(max_candidates))
    normalized = []
    for item in candidates:
        item = dict(item)
        if str(item.get("candidate_id")) in {"default", "default_no_rb"}:
            item["action_family"] = "default"
        normalized.append(item)
    defaults = [item for item in normalized if str(item.get("action_family")) == "default"]
    if len(defaults) != 1:
        raise ValueError("candidate set must contain exactly one default")
    selected = [defaults[0]]
    remaining = [item for item in normalized if item is not defaults[0]]
    seen_families = {"default"}
    for item in remaining:
        family = str(item.get("action_family"))
        if family in seen_families:
            continue
        selected.append(item)
        seen_families.add(family)
        if len(selected) >= limit:
            return selected
    for item in remaining:
        if item not in selected:
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected


def expand_temporal_candidates(
    candidates,
    intervention_start_step,
    temporal_patterns,
    max_candidates,
):
    """Create unique delayed variants while retaining exactly one default."""

    start = int(intervention_start_step)
    patterns = tuple(str(value).strip().lower() for value in temporal_patterns)
    if start != 1:
        raise ValueError("formal temporal candidate expansion requires start step 1")
    if not patterns or any(value not in {"persistent", "decayed"} for value in patterns):
        raise ValueError("temporal patterns must be persistent or decayed")
    expanded = []
    for source in candidates:
        candidate = dict(source)
        is_default = str(candidate.get("candidate_id")) in {"default", "default_no_rb"}
        if is_default:
            candidate["action_family"] = "default"
            candidate["candidate_id"] = "default"
            candidate["intervention_start_step"] = start
            candidate["temporal_pattern"] = "persistent"
            expanded.append(candidate)
            continue
        for pattern in patterns:
            variant = dict(candidate)
            variant["candidate_id"] = f"{candidate['candidate_id']}__{pattern}"
            variant["intervention_start_step"] = start
            variant["temporal_pattern"] = pattern
            expanded.append(variant)
    return select_balanced_candidates(expanded, max_candidates=max_candidates)


def select_decision_points(points, max_points):
    """Prefer stage coverage, then fill by simulation time."""
    ordered = sorted(points, key=lambda item: float(item["decision_time"]))
    selected = []
    for stage in ("offload_rb", "compute", "return_route"):
        match = next((item for item in ordered if item.get("decision_stage") == stage), None)
        if match is not None and match not in selected:
            selected.append(match)
        if len(selected) >= max_points:
            return selected
    for item in ordered:
        if item not in selected:
            selected.append(item)
            if len(selected) >= max_points:
                break
    return selected


def select_aligned_decision_points(points, sample_index_rows, max_points):
    """Discard unobservable times before applying stage-coverage selection."""

    aligned, rejected = align_points_to_sample_index(points, sample_index_rows)
    return select_decision_points(aligned, max_points), rejected


def _recording_cpu_callback(env):
    original = env.alloc_cpu_callback
    record = {"total": 0.0}
    if original is None:
        return record

    def callback(computing_tasks, **kwargs):
        allocations = original(computing_tasks, **kwargs)
        record["total"] = float(sum(float(value) for value in allocations.values()))
        return allocations

    env.alloc_cpu_callback = callback
    return record


def _action_applied(candidate, applied_counts):
    family = str(candidate["action_family"])
    if family == "default":
        return True
    checks = []
    for name in ("offload", "cpu", "return_route"):
        requested = int(candidate.get(f"num_{name}_overrides", 0))
        if requested:
            checks.append(int(applied_counts[name]) == requested)
    if family in {"rb_count", "mixed_offload_rb"}:
        checks.append(bool(candidate.get("rb_plan")))
    return bool(checks and all(checks))


def prepare_candidate_step(env, algorithm, candidate, active):
    """Schedule the shared default, then optionally overlay a precomputed action."""

    algorithm.scheduleStep(env)
    if not bool(active):
        return {"offload": 0, "cpu": 0, "return_route": 0}
    applied_counts = {
        "offload": apply_offload_overrides(env, algorithm, candidate),
        "cpu": apply_cpu_overrides(env, candidate),
        "return_route": apply_return_route_overrides(env, algorithm, candidate),
    }
    env.activated_offloading_tasks_with_RB_Nos = {
        task_id: list(rbs) for task_id, rbs in candidate.get("rb_plan", {}).items()
    }
    return applied_counts


def candidate_action_metadata(candidate, horizon):
    fields = (
        "rb_scale",
        "cpu_scale",
        "total_rb",
        "total_cpu",
        "num_offload_overrides",
        "num_cpu_overrides",
        "num_return_route_overrides",
        "offload_default_distance",
        "offload_alternative_distance",
        "offload_distance_delta",
        "offload_distance_ratio",
        "offload_default_is_vehicle",
        "offload_default_is_uav",
        "offload_default_is_rsu",
        "offload_default_is_cloud",
        "offload_alternative_is_vehicle",
        "offload_alternative_is_uav",
        "offload_alternative_is_rsu",
        "offload_alternative_is_cloud",
        "offload_target_type_changed",
    )
    result = {field: float(candidate.get(field, 0.0)) for field in fields}
    for field in ("total_rb", "num_offload_overrides", "num_cpu_overrides", "num_return_route_overrides"):
        result[field] = int(result[field])
    result["num_rb_tasks"] = int(len(candidate.get("rb_plan", {})))
    result["horizon"] = int(horizon)
    result["intervention_start_step"] = int(candidate.get("intervention_start_step", 0))
    result["temporal_pattern"] = str(candidate.get("temporal_pattern", "persistent"))
    return result


def _uav_rows(before, after, identifiers):
    rows = []
    before_uavs = before.get("uavs", {})
    after_uavs = after.get("uavs", {})
    for uav_id in sorted(set(before_uavs) | set(after_uavs)):
        before_item = before_uavs.get(uav_id, {})
        after_item = after_uavs.get(uav_id, {})
        consumption = after_item.get("last_consumption", {})
        row = dict(identifiers)
        row.update(
            {
                "uav_id": uav_id,
                "status": after_item.get("status", before_item.get("status", "unknown")),
                "energy_before": before_item.get("remaining_energy", np.nan),
                "energy_after": after_item.get("remaining_energy", np.nan),
            }
        )
        for name in ENERGY_COMPONENTS:
            row[f"energy_{name}"] = float(consumption.get(name, 0.0))
        row["energy_total"] = sum(row[f"energy_{name}"] for name in ENERGY_COMPONENTS)
        rows.append(row)
    return rows


def run_candidate(seed, point, candidate, horizon, max_time):
    start = time.perf_counter()
    decision_time = float(point["decision_time"])
    env, algorithm = make_env(seed, max_time=max(max_time, decision_time + horizon + 1.0))
    step_rows = []
    uav_rows = []
    try:
        step_default_until(env, algorithm, decision_time)
        context = collect_extended_context(env, algorithm)
        intervention_start_step = int(candidate.get("intervention_start_step", 0))
        temporal_pattern = str(candidate.get("temporal_pattern", "persistent"))
        active_application_results = []
        observed_applied_counts = {"offload": 0, "cpu": 0, "return_route": 0}

        for step in range(int(horizon)):
            candidate_action_step = str(candidate.get("action_family")) != "default" and temporal_pattern_active(
                step, intervention_start_step, temporal_pattern
            )
            applied_counts = prepare_candidate_step(
                env, algorithm, candidate, active=candidate_action_step
            )
            if candidate_action_step:
                observed_applied_counts = {
                    name: max(observed_applied_counts[name], int(value))
                    for name, value in applied_counts.items()
                }
                active_application_results.append(
                    _action_applied(candidate, applied_counts)
                )
            step_action_applied = (
                _action_applied(candidate, applied_counts)
                if candidate_action_step
                else True
            )
            counts_before = current_counts(env)
            throughput_before = channel_throughput(env)
            energy_before = capture_energy_manager_snapshot(env.energy_manager)
            rb_total = float(sum(len(rbs) for rbs in env.activated_offloading_tasks_with_RB_Nos.values()))
            cpu_record = _recording_cpu_callback(env)
            simulation_time_before = float(env.simulation_time)

            env.step()

            counts_after = current_counts(env)
            throughput_after = channel_throughput(env)
            energy_after = capture_energy_manager_snapshot(env.energy_manager)
            reward = reward_components(
                counts_after["done"] - counts_before["done"],
                counts_after["failed"] - counts_before["failed"],
                throughput_after - throughput_before,
            )
            energy = energy_step_metrics(energy_before, energy_after)
            identifiers = {
                "seed": int(seed),
                "sample_id": int(point.get("sample_id", -1)),
                "decision_time": decision_time,
                "step": int(step),
                "candidate_id": str(candidate["candidate_id"]),
                "action_family": str(candidate["action_family"]),
            }
            row = {
                **identifiers,
                "decision_stage": str(point.get("decision_stage", "offload_rb")),
                "simulation_time_before": simulation_time_before,
                "simulation_time_after": float(env.simulation_time),
                "rb_total": rb_total,
                "cpu_total": float(cpu_record["total"]),
                "action_applied": bool(step_action_applied),
                "candidate_action_step": bool(candidate_action_step),
                "intervention_start_step": intervention_start_step,
                "temporal_pattern": temporal_pattern,
                **reward,
                **energy,
            }
            step_rows.append(row)
            uav_rows.extend(_uav_rows(energy_before, energy_after, identifiers))
            if env.isDone():
                break
        action_applied = bool(active_application_results) and all(
            active_application_results
        )
        if str(candidate.get("action_family")) == "default":
            action_applied = True
        metadata = {
            "seed": int(seed),
            "sample_id": int(point.get("sample_id", -1)),
            "decision_time": decision_time,
            "candidate_id": str(candidate["candidate_id"]),
            "action_family": str(candidate["action_family"]),
            "decision_stage": str(point.get("decision_stage", "offload_rb")),
            "requested_offload_overrides": int(candidate.get("num_offload_overrides", 0)),
            "applied_offload_overrides": int(observed_applied_counts["offload"]),
            "requested_cpu_overrides": int(candidate.get("num_cpu_overrides", 0)),
            "applied_cpu_overrides": int(observed_applied_counts["cpu"]),
            "requested_return_route_overrides": int(candidate.get("num_return_route_overrides", 0)),
            "applied_return_route_overrides": int(observed_applied_counts["return_route"]),
            "action_applied": bool(action_applied),
            "runtime_ms": float((time.perf_counter() - start) * 1000.0),
            "context_num_to_offload_tasks": int(context.get("num_to_offload_tasks", 0)),
            "context_num_computing_tasks": int(context.get("num_computing_tasks", 0)),
            "context_num_waiting_return_tasks": int(context.get("num_waiting_return_tasks", 0)),
        }
        metadata.update(candidate_action_metadata(candidate, horizon))
        return step_rows, uav_rows, metadata
    finally:
        env.close()


def generate_rows(args):
    step_rows = []
    uav_rows = []
    metadata_rows = []
    point_rows = []
    sample_index_rows = None
    sample_index_csv = getattr(args, "sample_index_csv", None)
    if sample_index_csv is not None:
        sample_index_rows = pd.read_csv(sample_index_csv).to_dict(orient="records")
        if int(getattr(args, "intervention_start_step", 0)) != 1:
            raise ValueError("selector-timed physical labels require intervention start step 1")
    for seed in args.seeds:
        discovery_limit = int(args.decision_times_per_seed)
        if sample_index_rows is not None:
            discovery_limit = max(12, discovery_limit * 4)
        discovered = discover_decision_points(
            seed,
            args.max_time,
            discovery_limit,
            args.scan_step_limit,
            include_compute_return=True,
        )
        rejected_points = []
        if sample_index_rows is None:
            points = select_decision_points(discovered, args.decision_times_per_seed)
        else:
            points, rejected_points = select_aligned_decision_points(
                discovered, sample_index_rows, args.decision_times_per_seed
            )
        for rejected in rejected_points:
            point_rows.append(
                {
                    "seed": int(seed),
                    "sample_id": -1,
                    "decision_time": float(rejected["decision_time"]),
                    "point_index": -1,
                    "decision_stage": str(rejected.get("decision_stage", "unknown")),
                    "num_candidates": 0,
                    "candidate_families": "",
                    "alignment_status": str(rejected["reason"]),
                }
            )
        for point_index, point in enumerate(points):
            raw_candidates = build_extended_candidates(point, max_candidates=max(32, args.max_candidates * 4))
            if sample_index_rows is None:
                candidates = select_balanced_candidates(raw_candidates, args.max_candidates)
                candidates = [
                    {
                        **candidate,
                        "intervention_start_step": int(
                            getattr(args, "intervention_start_step", 0)
                        ),
                        "temporal_pattern": str(
                            getattr(args, "temporal_patterns", ["persistent"])[0]
                        ),
                    }
                    for candidate in candidates
                ]
            else:
                candidates = expand_temporal_candidates(
                    raw_candidates,
                    intervention_start_step=args.intervention_start_step,
                    temporal_patterns=args.temporal_patterns,
                    max_candidates=args.max_candidates,
                )
            point_rows.append(
                {
                    "seed": int(seed),
                    "sample_id": int(point.get("sample_id", -1)),
                    "decision_time": float(point["decision_time"]),
                    "point_index": int(point_index),
                    "decision_stage": str(point.get("decision_stage", "offload_rb")),
                    "num_candidates": len(candidates),
                    "candidate_families": ",".join(sorted({item["action_family"] for item in candidates})),
                    "alignment_status": "exact" if sample_index_rows is not None else "legacy_unaligned",
                }
            )
            for candidate in candidates:
                candidate_steps, candidate_uavs, metadata = run_candidate(
                    seed, point, candidate, args.horizon, args.max_time
                )
                step_rows.extend(candidate_steps)
                uav_rows.extend(candidate_uavs)
                metadata_rows.append(metadata)
    return step_rows, uav_rows, metadata_rows, point_rows


def build_candidate_summary(step_rows, metadata_rows):
    summaries = summarize_candidate_steps(
        step_rows,
        group_fields=("seed", "sample_id", "decision_time", "candidate_id", "action_family"),
    )
    metadata = {
        (row["seed"], row["sample_id"], row["decision_time"], row["candidate_id"], row["action_family"]): row
        for row in metadata_rows
    }
    for row in summaries:
        key = (row["seed"], row["sample_id"], row["decision_time"], row["candidate_id"], row["action_family"])
        row.update(metadata[key])
    return summaries


def add_energy_sensitivity(effect_rows):
    for row in effect_rows:
        baseline_energy = max(abs(float(row["baseline_energy_total"])), 1e-12)
        overhead = float(row["effect_energy_total"]) / baseline_energy
        row["energy_overhead_ratio"] = overhead
        for value in SENSITIVITY_LAMBDAS:
            row[f"energy_aware_utility_lambda_{value:g}"] = float(row["task_utility"]) - value * overhead
    return effect_rows


def make_group_audit(candidate_df):
    rows = []
    for (seed, decision_time), part in candidate_df.groupby(["seed", "decision_time"], dropna=False):
        spread = float(part["task_utility"].max() - part["task_utility"].min())
        rows.append(
            {
                "seed": int(seed),
                "decision_time": float(decision_time),
                "num_candidates": int(len(part)),
                "utility_spread": spread,
                "is_nontrivial": bool(spread > 1e-8),
                "num_invalid_actions": int((~part["action_applied"].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_family_summary(effects_df):
    columns = [
        "action_family",
        "num_candidates",
        "mean_effect_task_utility",
        "min_effect_task_utility",
        "max_effect_task_utility",
        "positive_utility_ratio",
        "mean_effect_energy_total",
        "min_effect_energy_total",
        "max_effect_energy_total",
    ]
    if effects_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for family, part in effects_df.groupby("action_family", dropna=False):
        rows.append(
            {
                "action_family": str(family),
                "num_candidates": int(len(part)),
                "mean_effect_task_utility": float(part["effect_task_utility"].mean()),
                "min_effect_task_utility": float(part["effect_task_utility"].min()),
                "max_effect_task_utility": float(part["effect_task_utility"].max()),
                "positive_utility_ratio": float((part["effect_task_utility"] > 0).mean()),
                "mean_effect_energy_total": float(part["effect_energy_total"].mean()),
                "min_effect_energy_total": float(part["effect_energy_total"].min()),
                "max_effect_energy_total": float(part["effect_energy_total"].max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def format_effect_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def markdown_table(frame):
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float) and math.isnan(value):
                text = ""
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def metric_definitions_markdown():
    return """# PI-JWM 能耗与逐步 Reward 指标定义

| 字段 | 定义 | 单位/方向 | 使用边界 |
| --- | --- | --- | --- |
| `delta_done` | 当前步新增完成任务数 | count，越大越好 | 原始物理量 |
| `delta_failed` | 当前步新增超时/失败任务数 | count，越小越好 | 原始物理量 |
| `throughput_delta` | 当前步新增传输数据量，负值截为 0 | simulator data unit，越大越好 | 原始物理量 |
| `reward_done` | 等于 `delta_done` | reward component | 可直接重建 |
| `reward_failed` | 等于 `-delta_failed` | reward component | 可直接重建 |
| `reward_throughput` | `0.01 * log(1 + throughput_delta)` | reward component | 固定权重，不按结果调参 |
| `task_utility` | 三个 reward component 之和；候选级为逐步值求和 | 越大越好 | 主任务指标，不含能耗权重 |
| `rb_total` | 每一步实际激活的 RB 数量，候选级为逐步求和 | RB count，越小越节省 | 与静态 `total_rb` 区分 |
| `cpu_total` | 每一步 CPU callback 实际分配总量，候选级为逐步求和 | simulator CPU unit | 原始资源量 |
| `energy_fly/hover/sensing/receive/send` | AirFogSim EnergyManager 配置与当步状态计算出的五类 UAV 消耗 | simulator energy unit，越小越好 | 原始能耗分项 |
| `energy_total` | 五类能耗分项之和 | simulator energy unit，越小越好 | 必须满足分项守恒 |
| `energy_balance_error` | `(energy_before - energy_after) - energy_total` | 应为 0 | 非零时实验质量门失败 |
| `effect_*` | 同 seed、同决策时间下候选值减默认动作值 | 正负按对应指标解释 | 配对归因，不是跨场景相关性 |
| `energy_aware_utility_lambda_*` | `task_utility - lambda * energy_overhead_ratio` | 敏感性指标 | lambda 不按 val/test 事后选择 |

`task_utility` 与原始任务、资源、能耗指标分开报告。综合能耗权重只用于敏感性分析，不能替代原始物理量，也不能据此声称存在统一最优策略。
"""


def make_figures(step_df, candidate_df, effects_df, family_df, output_dir):
    figures = {}
    first = step_df.sort_values(["seed", "decision_time", "candidate_id", "step"])
    first_key = first.iloc[0][["seed", "decision_time", "candidate_id"]].tolist()
    trace = first[
        (first["seed"] == first_key[0])
        & (first["decision_time"] == first_key[1])
        & (first["candidate_id"] == first_key[2])
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(trace["step"], trace["reward_done"], label="completed")
    ax.bar(trace["step"], trace["reward_failed"], bottom=trace["reward_done"], label="failed")
    bottom = trace["reward_done"] + trace["reward_failed"]
    ax.bar(trace["step"], trace["reward_throughput"], bottom=bottom, label="throughput")
    ax.set(xlabel="rollout step", ylabel="reward component", title="Step-wise reward decomposition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    figures["reward_decomposition"] = _save_figure(fig, output_dir / "figure_1_reward_decomposition.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)
    energy_cols = [f"energy_{name}" for name in ENERGY_COMPONENTS]
    trace.set_index("step")[energy_cols].plot(kind="bar", stacked=True, ax=ax1)
    ax1.set(ylabel="energy used", title="UAV energy components by rollout step")
    ax1.grid(axis="y", alpha=0.25)
    ax2.plot(trace["step"], trace["energy_after"], marker="o")
    ax2.set(xlabel="rollout step", ylabel="remaining energy")
    ax2.grid(alpha=0.25)
    figures["energy_components"] = _save_figure(fig, output_dir / "figure_2_uav_energy_components.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for family, part in candidate_df.groupby("action_family", dropna=False):
        ax.scatter(part["energy_total"], part["task_utility"], label=str(family), alpha=0.8, s=45)
    ax.set(xlabel="UAV energy used", ylabel="step-wise task utility", title="Task utility and UAV-energy trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    figures["utility_energy_tradeoff"] = _save_figure(fig, output_dir / "figure_3_utility_energy_tradeoff.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    if family_df.empty:
        ax1.text(0.5, 0.5, "No paired effects", ha="center", va="center")
        ax2.text(0.5, 0.5, "No paired effects", ha="center", va="center")
    else:
        labels = family_df["action_family"]
        ax1.barh(labels, family_df["mean_effect_task_utility"])
        ax2.barh(labels, family_df["mean_effect_energy_total"])
    ax1.set(xlabel="mean paired effect", title="Task utility coupling")
    ax2.set(xlabel="mean paired effect", title="UAV energy coupling")
    format_effect_axis(ax1)
    format_effect_axis(ax2)
    ax1.grid(axis="x", alpha=0.25)
    ax2.grid(axis="x", alpha=0.25)
    figures["paired_coupling"] = _save_figure(fig, output_dir / "figure_4_paired_coupling.png")
    return figures


def write_report(summary, family_df, group_df, output_dir):
    audit = summary["quality_audit"]
    facts = [
        f"Collected {summary['num_step_rows']} step rows from {summary['num_seeds']} fixed seeds.",
        f"Evaluated {summary['num_candidates']} candidate rollouts across {summary['num_decision_groups']} paired decision groups.",
        f"Energy/reward audit passed: {audit['passed']} (balance errors={audit['energy_balance_errors']}, reward errors={audit['reward_reconstruction_errors']}).",
        f"Non-trivial task-utility groups: {summary['num_nontrivial_groups']}/{summary['num_decision_groups']}.",
    ]
    interpretations = [
        "Paired effects compare candidates only against the default action at the same seed and decision time.",
        "Raw task, resource, and energy quantities are primary; energy-aware scalar scores are sensitivity diagnostics only.",
        "With three to five seeds, effect directions are preliminary and are not treated as significance claims.",
    ]
    hypotheses = [
        "If candidate utility spread is small, candidate construction or the short rollout horizon should be examined before selector retraining.",
        "If energy-aware rankings change across penalty values, reward preference specification is a bottleneck rather than evidence of a universally better policy.",
        "PI-JWM selector attribution requires a protocol-matched learned ranking evaluation; this simulator-only diagnostic does not promote oracle or proxy rankings.",
    ]
    lines = [
        "# PI-JWM UAV Energy and Step-wise Reward Diagnostic",
        "",
        "AirFogSim is used only as the simulator and data source. PI-JWM remains the research framework.",
        "",
        "## Observed Facts",
        "",
        *[f"- {item}" for item in facts],
        "",
        "## Reasonable Interpretations",
        "",
        *[f"- {item}" for item in interpretations],
        "",
        "## Hypotheses Requiring Further Validation",
        "",
        *[f"- {item}" for item in hypotheses],
        "",
        "## Action-family Summary",
        "",
        markdown_table(family_df) if not family_df.empty else "No paired action-family effects were available.",
        "",
        "## Decision-group Audit",
        "",
        markdown_table(group_df) if not group_df.empty else "No decision groups were available.",
        "",
        "## Result Boundary",
        "",
        "- `result_kind`: `diagnostic_only`",
        "- No test-best, oracle, true-future, or proxy result is presented as deployable.",
        "- Historical BC/non-BC results are not merged into the new energy table because their protocols differ.",
        "",
        "## Reproduction",
        "",
        "See `reproduction_command.txt` and `summary.json` in this directory.",
    ]
    path = output_dir / "diagnostic_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main():
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows, uav_rows, metadata_rows, point_rows = generate_rows(args)
    if not step_rows:
        raise RuntimeError("no valid counterfactual decision rows were generated")
    candidate_rows = build_candidate_summary(step_rows, metadata_rows)
    effect_rows = paired_candidate_effects(
        candidate_rows,
        metric_fields=EFFECT_METRICS,
        group_fields=("seed", "sample_id", "decision_time"),
    )
    metadata_by_candidate = {
        (
            int(row["seed"]),
            int(row["sample_id"]),
            float(row["decision_time"]),
            str(row["candidate_id"]),
        ): row
        for row in candidate_rows
    }
    for row in effect_rows:
        source = metadata_by_candidate[
            (
                int(row["seed"]),
                int(row["sample_id"]),
                float(row["decision_time"]),
                str(row["candidate_id"]),
            )
        ]
        row["intervention_start_step"] = int(source["intervention_start_step"])
        row["temporal_pattern"] = str(source["temporal_pattern"])
        row["action_applied"] = bool(source["action_applied"])
    add_energy_sensitivity(effect_rows)

    step_df = pd.DataFrame(step_rows)
    uav_df = pd.DataFrame(uav_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    effects_df = pd.DataFrame(effect_rows)
    points_df = pd.DataFrame(point_rows)
    group_df = make_group_audit(candidate_df)
    family_df = make_family_summary(effects_df)
    quality = audit_diagnostic_rows(step_rows)

    outputs = {
        "step_metrics_csv": output_dir / "step_metrics.csv",
        "uav_energy_steps_csv": output_dir / "uav_energy_steps.csv",
        "candidate_summary_csv": output_dir / "candidate_summary.csv",
        "coupling_effects_csv": output_dir / "coupling_effects.csv",
        "action_family_summary_csv": output_dir / "action_family_summary.csv",
        "decision_group_audit_csv": output_dir / "decision_group_audit.csv",
        "decision_points_csv": output_dir / "decision_points.csv",
        "metric_definitions": output_dir / "metric_definitions.md",
    }
    for frame, key in (
        (step_df, "step_metrics_csv"),
        (uav_df, "uav_energy_steps_csv"),
        (candidate_df, "candidate_summary_csv"),
        (effects_df, "coupling_effects_csv"),
        (family_df, "action_family_summary_csv"),
        (group_df, "decision_group_audit_csv"),
        (points_df, "decision_points_csv"),
    ):
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")
    outputs["metric_definitions"].write_text(metric_definitions_markdown(), encoding="utf-8")

    figures = make_figures(step_df, candidate_df, effects_df, family_df, output_dir)
    summary = {
        "framework": "PI-JWM",
        "simulator": "AirFogSim",
        "result_kind": "diagnostic_only",
        "seeds": [int(seed) for seed in args.seeds],
        "num_seeds": int(candidate_df["seed"].nunique()),
        "num_step_rows": int(len(step_df)),
        "num_candidates": int(len(candidate_df)),
        "num_decision_groups": int(len(group_df)),
        "num_nontrivial_groups": int(group_df["is_nontrivial"].sum()),
        "action_families": sorted(candidate_df["action_family"].unique().tolist()),
        "quality_audit": quality,
        "experiment": {
            "max_time": float(args.max_time),
            "horizon": int(args.horizon),
            "decision_times_per_seed": int(args.decision_times_per_seed),
            "max_candidates": int(args.max_candidates),
            "scan_step_limit": int(args.scan_step_limit),
            "energy_penalty_lambdas": list(SENSITIVITY_LAMBDAS),
            "intervention_protocol": (
                "selector_timed" if args.sample_index_csv is not None else "immediate_legacy"
            ),
            "intervention_start_step": int(args.intervention_start_step),
            "temporal_patterns": list(args.temporal_patterns),
            "sample_index_csv": (
                str(args.sample_index_csv) if args.sample_index_csv is not None else None
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "figures": {key: str(path) for key, path in figures.items()},
    }
    report_path = write_report(summary, family_df, group_df, output_dir)
    summary["outputs"]["diagnostic_report"] = str(report_path)
    command = build_reproduction_command(args)
    command_path = output_dir / "reproduction_command.txt"
    command_path.write_text(command + "\n", encoding="utf-8")
    summary["outputs"]["reproduction_command"] = str(command_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
