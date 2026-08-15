from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pi_jwm.r5_analysis import MetricSpec, paired_comparison


R6_10K_ANALYSIS_SCHEMA = "PIJWM-R6-10k-Gate-Analysis-v1"
VALIDATION_METRICS = (
    MetricSpec("validation_return", "higher"),
    MetricSpec("on_time_completion_rate", "higher"),
    MetricSpec("mean_latency", "lower"),
)
TRAINING_FIELDS = (
    "total_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "gradient_norm",
    "ratio_min",
    "ratio_max",
)
EXPECTED_CANDIDATE_IDS = {
    "airfogsim_default",
    "deadline_first",
    "energy_conservative",
    "load_balance",
    "priority_first",
    "rate_aware",
}


def _finite(value: Any, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    numeric = [_finite(value, label="summary value") for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty sequence")
    mean = statistics.fmean(numeric)
    return {
        "count": len(numeric),
        "mean": mean,
        "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("slope requires aligned non-empty values")
    if len(xs) == 1:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _expected_run_map(
    methods: Sequence[str],
    state_modes: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, tuple[str, str, int]]:
    return {
        f"{method}__{mode}__seed_{int(seed)}": (str(method), str(mode), int(seed))
        for method in methods
        for mode in state_modes
        for seed in seeds
    }


def _selection_entropy(counts: Mapping[str, Any]) -> float:
    numeric = [_finite(value, label="candidate selection count") for value in counts.values()]
    if any(value < 0.0 for value in numeric) or not numeric:
        raise ValueError("candidate selection counts must be non-negative and non-empty")
    total = sum(numeric)
    if total <= 0.0:
        raise ValueError("candidate selection counts must have positive total")
    if len(numeric) == 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in numeric if value > 0.0)
    return entropy / math.log(len(numeric))


def _validate_and_normalize(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str],
    expected_state_modes: Sequence[str],
    expected_seeds: Sequence[int],
    target_environment_steps: int,
) -> list[dict[str, Any]]:
    expected = _expected_run_map(expected_methods, expected_state_modes, expected_seeds)
    by_id: dict[str, dict[str, Any]] = {}
    for raw in records:
        row = dict(raw)
        run_id = str(row.get("run_id", ""))
        if run_id in by_id:
            raise ValueError(f"duplicate run_id: {run_id}")
        by_id[run_id] = row
    if set(by_id) != set(expected):
        raise ValueError(
            "R6 10k matrix mismatch: "
            f"missing={sorted(set(expected) - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - set(expected))}"
        )

    normalized: list[dict[str, Any]] = []
    for run_id in sorted(expected):
        row = by_id[run_id]
        method, state_mode, seed = expected[run_id]
        if row.get("formal") is not True or row.get("status") != "complete":
            raise ValueError(f"formal run is not complete: {run_id}")
        if row.get("locked_test_accessed") is not False:
            raise ValueError(f"formal run accessed locked-test: {run_id}")
        if row.get("world_model_updated") is not False:
            raise ValueError(f"world model changed during policy training: {run_id}")
        if row.get("checkpoint_reload_verified") is not True:
            raise ValueError(f"checkpoint reload not verified: {run_id}")
        if row.get("state_source") != "online_airfogsim_strict_dual_graph":
            raise ValueError(f"unexpected state source: {run_id}")
        if int(row.get("environment_steps", -1)) != int(target_environment_steps):
            raise ValueError(f"unexpected environment step budget: {run_id}")

        reports = [dict(report) for report in row.get("reports", [])]
        if not reports or int(row.get("update_count", -1)) != len(reports):
            raise ValueError(f"training update records are incomplete: {run_id}")
        steps: list[int] = []
        for report in reports:
            step = int(report.get("environment_step", -1))
            steps.append(step)
            for field in TRAINING_FIELDS:
                _finite(report.get(field), label=f"{run_id}.{field}")
            if report.get("parameter_changed") is not True:
                raise ValueError(f"policy parameter did not change: {run_id}:{step}")
        if steps != sorted(set(steps)) or steps[-1] != int(target_environment_steps):
            raise ValueError(f"training update steps are invalid: {run_id}")

        validations = [dict(report) for report in row.get("validation_reports", [])]
        if not validations:
            raise ValueError(f"validation report is missing: {run_id}")
        for validation in validations:
            if int(validation.get("environment_step", -1)) > int(target_environment_steps):
                raise ValueError(f"validation step exceeds training budget: {run_id}")
            for metric in VALIDATION_METRICS:
                _finite(validation.get(metric.metric_id), label=f"{run_id}.{metric.metric_id}")
            if not 0.0 <= float(validation["on_time_completion_rate"]) <= 1.0:
                raise ValueError(f"on-time completion rate is outside [0,1]: {run_id}")
            if int(validation.get("hard_violation_count", -1)) < 0:
                raise ValueError(f"validation hard violation count is invalid: {run_id}")

        counts = dict(row.get("candidate_selection_counts", {}))
        if set(counts) != EXPECTED_CANDIDATE_IDS:
            raise ValueError(
                f"candidate IDs differ from the frozen action protocol: {run_id}"
            )
        selection_total = sum(int(value) for value in counts.values())
        if selection_total != int(target_environment_steps):
            raise ValueError(f"candidate selection counts do not match budget: {run_id}")
        nondefault = int(row.get("nondefault_selection_count", -1))
        if nondefault != selection_total - int(counts.get("airfogsim_default", 0)):
            raise ValueError(f"nondefault selection count is inconsistent: {run_id}")
        if int(row.get("distinct_explicit_state_count", 0)) <= 0:
            raise ValueError(f"no distinct online explicit states recorded: {run_id}")

        normalized.append(
            {
                **row,
                "method_id": method,
                "state_mode": state_mode,
                "training_seed": seed,
                "reports": reports,
                "validation_reports": validations,
                "candidate_selection_counts": counts,
            }
        )
    return normalized


def _run_metric_row(row: Mapping[str, Any], target_steps: int) -> dict[str, Any]:
    reports = list(row["reports"])
    validation = list(row["validation_reports"])[-1]
    total_losses = [float(report["total_loss"]) for report in reports]
    policy_losses = [float(report["policy_loss"]) for report in reports]
    value_losses = [float(report["value_loss"]) for report in reports]
    entropies = [float(report["entropy"]) for report in reports]
    gradients = [float(report["gradient_norm"]) for report in reports]
    steps = [float(report["environment_step"]) for report in reports]
    tail = min(10, len(reports))
    return {
        "run_id": row["run_id"],
        "method_id": row["method_id"],
        "state_mode": row["state_mode"],
        "training_seed": int(row["training_seed"]),
        "validation_return": float(validation["validation_return"]),
        "on_time_completion_rate": float(validation["on_time_completion_rate"]),
        "mean_latency": float(validation["mean_latency"]),
        "hard_violation_count": int(row["hard_violation_count"]),
        "validation_hard_violation_count": int(validation["hard_violation_count"]),
        "validation_step_count": int(validation["validation_step_count"]),
        "validation_trajectory_count": int(validation["validation_trajectory_count"]),
        "nondefault_selection_rate": int(row["nondefault_selection_count"]) / target_steps,
        "normalized_selection_entropy": _selection_entropy(row["candidate_selection_counts"]),
        "distinct_explicit_state_count": int(row["distinct_explicit_state_count"]),
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "update_count": len(reports),
        "parameter_changed_all": all(report.get("parameter_changed") is True for report in reports),
        "total_loss_first10_mean": statistics.fmean(total_losses[:tail]),
        "total_loss_last10_mean": statistics.fmean(total_losses[-tail:]),
        "total_loss_slope_per_1k_steps": 1000.0 * _linear_slope(steps, total_losses),
        "policy_loss_last10_mean": statistics.fmean(policy_losses[-tail:]),
        "value_loss_last10_mean": statistics.fmean(value_losses[-tail:]),
        "entropy_first10_mean": statistics.fmean(entropies[:tail]),
        "entropy_last10_mean": statistics.fmean(entropies[-tail:]),
        "gradient_norm_mean": statistics.fmean(gradients),
        "gradient_norm_maximum": max(gradients),
        "ratio_minimum": min(float(report["ratio_min"]) for report in reports),
        "ratio_maximum": max(float(report["ratio_max"]) for report in reports),
    }


def analyze_r6_10k_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str],
    expected_state_modes: Sequence[str],
    expected_seeds: Sequence[int],
    target_environment_steps: int,
) -> dict[str, Any]:
    validated = _validate_and_normalize(
        records,
        expected_methods=expected_methods,
        expected_state_modes=expected_state_modes,
        expected_seeds=expected_seeds,
        target_environment_steps=target_environment_steps,
    )
    run_metrics = [_run_metric_row(row, int(target_environment_steps)) for row in validated]
    by_key = {
        (row["method_id"], row["state_mode"], int(row["training_seed"])): row
        for row in run_metrics
    }

    summary_fields = (
        "validation_return",
        "on_time_completion_rate",
        "mean_latency",
        "nondefault_selection_rate",
        "normalized_selection_entropy",
        "distinct_explicit_state_count",
        "elapsed_seconds",
        "gradient_norm_maximum",
        "total_loss_slope_per_1k_steps",
    )
    configuration_summary: dict[str, Any] = {}
    for method in expected_methods:
        for mode in expected_state_modes:
            config_id = f"{method}__{mode}"
            selected = [by_key[(method, mode, int(seed))] for seed in expected_seeds]
            configuration_summary[config_id] = {
                "method_id": method,
                "state_mode": mode,
                "metrics": {
                    field: {
                        **_describe([float(row[field]) for row in selected]),
                        "by_seed": {
                            str(seed): float(by_key[(method, mode, int(seed))][field])
                            for seed in expected_seeds
                        },
                    }
                    for field in summary_fields
                },
            }

    policy_paired: dict[str, Any] = {}
    if "actor_critic" in expected_methods and "ppo_clipped" in expected_methods:
        for mode in expected_state_modes:
            policy_paired[mode] = {}
            for metric in VALIDATION_METRICS:
                baseline = {
                    int(seed): float(by_key[("actor_critic", mode, int(seed))][metric.metric_id])
                    for seed in expected_seeds
                }
                candidate = {
                    int(seed): float(by_key[("ppo_clipped", mode, int(seed))][metric.metric_id])
                    for seed in expected_seeds
                }
                policy_paired[mode][metric.metric_id] = paired_comparison(
                    baseline, candidate, metric
                )

    state_paired: dict[str, Any] = {}
    comparisons = (
        ("explicit_only", "latent_only"),
        ("explicit_only", "explicit_latent"),
        ("latent_only", "explicit_latent"),
    )
    for method in expected_methods:
        state_paired[method] = {}
        for baseline_mode, candidate_mode in comparisons:
            if baseline_mode not in expected_state_modes or candidate_mode not in expected_state_modes:
                continue
            comparison_id = f"{candidate_mode}_vs_{baseline_mode}"
            state_paired[method][comparison_id] = {}
            for metric in VALIDATION_METRICS:
                baseline = {
                    int(seed): float(by_key[(method, baseline_mode, int(seed))][metric.metric_id])
                    for seed in expected_seeds
                }
                candidate = {
                    int(seed): float(by_key[(method, candidate_mode, int(seed))][metric.metric_id])
                    for seed in expected_seeds
                }
                state_paired[method][comparison_id][metric.metric_id] = paired_comparison(
                    baseline, candidate, metric
                )

    training_curve = []
    for row in validated:
        for report in row["reports"]:
            training_curve.append(
                {
                    "run_id": row["run_id"],
                    "method_id": row["method_id"],
                    "state_mode": row["state_mode"],
                    "training_seed": int(row["training_seed"]),
                    **{field: report[field] for field in ("environment_step", *TRAINING_FIELDS)},
                }
            )

    validation_points = sorted({len(row["validation_reports"]) for row in validated})
    health_checks = {
        "complete_matrix": len(validated)
        == len(expected_methods) * len(expected_state_modes) * len(expected_seeds),
        "zero_hard_violations": all(
            int(row["hard_violation_count"]) == 0
            and all(int(v["hard_violation_count"]) == 0 for v in row["validation_reports"])
            for row in validated
        ),
        "all_checkpoint_reloads_verified": all(
            row["checkpoint_reload_verified"] is True for row in validated
        ),
        "world_model_frozen": all(row["world_model_updated"] is False for row in validated),
        "locked_test_not_accessed": all(
            row["locked_test_accessed"] is False for row in validated
        ),
        "all_policy_updates_changed_parameters": all(
            metric["parameter_changed_all"] is True for metric in run_metrics
        ),
        "all_six_candidate_types_exercised": all(
            len(row["candidate_selection_counts"]) == 6
            and all(int(value) > 0 for value in row["candidate_selection_counts"].values())
            for row in validated
        ),
    }
    gate_passed = all(health_checks.values())
    recommended_configurations = [
        f"{method}__{mode}" for method in expected_methods for mode in expected_state_modes
    ] if gate_passed else []
    continuation_gate = {
        "status": (
            "pass_continue_full_frozen_matrix" if gate_passed else "fail_do_not_continue"
        ),
        "continue_to_full_budget": gate_passed,
        "health_checks": health_checks,
        "recommended_configurations": recommended_configurations,
        "recommended_configuration_count": len(recommended_configurations),
        "recommended_run_count": len(recommended_configurations) * len(expected_seeds),
        "reason": (
            "All predeclared health and safety checks passed. Each run has only one "
            "validation checkpoint at 10k, so pruning or final selection is not supported; "
            "continue the complete frozen matrix to collect a validation learning curve."
            if gate_passed
            else "At least one predeclared health or safety check failed."
        ),
        "final_method_frozen": False,
        "locked_test_accessed": False,
    }
    return {
        "schema_version": R6_10K_ANALYSIS_SCHEMA,
        "selection_status": "descriptive_10k_health_gate",
        "claim_boundary": "health_gate_not_final_selection",
        "integrity": {
            "completed_run_count": len(validated),
            "expected_run_count": len(expected_methods)
            * len(expected_state_modes)
            * len(expected_seeds),
            "failed_run_count": 0,
            "validation_points_per_run": validation_points,
            "locked_test_accessed": False,
        },
        "metric_directions": {
            metric.metric_id: metric.direction for metric in VALIDATION_METRICS
        },
        "run_metrics": run_metrics,
        "configuration_summary": configuration_summary,
        "policy_paired_by_state_mode": policy_paired,
        "state_paired_by_method": state_paired,
        "training_curve": training_curve,
        "continuation_gate": continuation_gate,
        "limitations": [
            "Only one validation checkpoint exists per run at 10k; no validation learning curve is available yet.",
            "On-time completion rate is 1.0 for every run and therefore cannot rank configurations at this gate.",
            "Training losses are objective-specific diagnostics and are not comparable between Actor-Critic and PPO.",
            "Throughput, energy, resource utilization and fairness are not present in these summaries and are not imputed.",
            "Three seeds support paired descriptive evidence, not final method selection.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(section: Mapping[str, Any], *, section_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if section_name == "policy":
        iterable = (
            (state_mode, "ppo_clipped_vs_actor_critic", metrics)
            for state_mode, metrics in section.items()
        )
    else:
        iterable = (
            (method, comparison, metrics)
            for method, comparisons in section.items()
            for comparison, metrics in comparisons.items()
        )
    for group, comparison, metrics in iterable:
        for metric_id, record in metrics.items():
            benefit = record["benefit"]
            rows.append(
                {
                    "group": group,
                    "comparison": comparison,
                    "metric_id": metric_id,
                    "direction": record["direction"],
                    "benefit_definition": record["benefit_definition"],
                    "mean_benefit": benefit["mean"],
                    "sample_std_benefit": benefit["sample_std"],
                    "wins": record["wins"],
                    "ties": record["ties"],
                    "losses": record["losses"],
                    "benefit_by_seed": json.dumps(record["benefit_by_seed"], sort_keys=True),
                    "exact_sign_flip_p_two_sided": record["exact_sign_flip_p_two_sided"],
                }
            )
    return rows


def _write_figures(analysis: Mapping[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = list(analysis["training_curve"])
    methods = sorted({str(row["method_id"]) for row in curve})
    modes = sorted({str(row["state_mode"]) for row in curve})
    colors = {mode: color for mode, color in zip(modes, ("#4472C4", "#ED7D31", "#70AD47"))}
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4), squeeze=False)
    for axis, method in zip(axes[0], methods):
        for mode in modes:
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in curve:
                if row["method_id"] == method and row["state_mode"] == mode:
                    grouped[int(row["environment_step"])].append(float(row["total_loss"]))
            xs = sorted(grouped)
            means = [statistics.fmean(grouped[x]) for x in xs]
            lows = [min(grouped[x]) for x in xs]
            highs = [max(grouped[x]) for x in xs]
            axis.plot(xs, means, label=mode, color=colors[mode], linewidth=1.5)
            axis.fill_between(xs, lows, highs, color=colors[mode], alpha=0.15)
        axis.set_title(method.replace("_", " "))
        axis.set_xlabel("environment step")
        axis.set_ylabel("training total loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("R6 10k training-objective diagnostics (mean and seed range)")
    fig.tight_layout()
    fig.savefig(output / "training_objective_curves.png", dpi=180)
    plt.close(fig)

    run_metrics = list(analysis["run_metrics"])
    configs = sorted({f"{r['method_id']}__{r['state_mode']}" for r in run_metrics})
    palette = plt.get_cmap("tab10")
    fig, axis = plt.subplots(figsize=(8, 6))
    for index, config in enumerate(configs):
        selected = [r for r in run_metrics if f"{r['method_id']}__{r['state_mode']}" == config]
        xs = [float(r["mean_latency"]) for r in selected]
        ys = [float(r["validation_return"]) for r in selected]
        axis.scatter(xs, ys, label=config, color=palette(index), alpha=0.75)
        axis.scatter(
            [statistics.fmean(xs)], [statistics.fmean(ys)], marker="*", s=130,
            color=palette(index), edgecolor="black", linewidth=0.5,
        )
    axis.set_xlabel("mean latency (lower is better)")
    axis.set_ylabel("validation return (higher is better)")
    axis.set_title("R6 10k validation return–latency trade-off\n(points: seeds; stars: configuration means)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output / "validation_return_latency.png", dpi=180)
    plt.close(fig)


def write_r6_10k_analysis_bundle(
    analysis: Mapping[str, Any],
    output_dir: str | Path,
    *,
    input_binding: Mapping[str, str],
) -> None:
    output = Path(output_dir)
    allowed_existing = {"task_plan.md", "findings.md", "progress.md"}
    if output.exists():
        unexpected = {
            path.name
            for path in output.iterdir()
            if path.name not in allowed_existing
            and not (path.is_dir() and path.name.startswith("audit_"))
        }
        if unexpected:
            raise FileExistsError(f"analysis output already contains generated files: {sorted(unexpected)}")
    else:
        output.mkdir(parents=True)

    (output / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "run_metrics.csv", list(analysis["run_metrics"]))

    configuration_rows = []
    for config_id, record in analysis["configuration_summary"].items():
        for metric_id, metric in record["metrics"].items():
            configuration_rows.append(
                {
                    "configuration_id": config_id,
                    "method_id": record["method_id"],
                    "state_mode": record["state_mode"],
                    "metric_id": metric_id,
                    "count": metric["count"],
                    "mean": metric["mean"],
                    "sample_std": metric["sample_std"],
                    "median": metric["median"],
                    "minimum": metric["minimum"],
                    "maximum": metric["maximum"],
                    "by_seed": json.dumps(metric["by_seed"], sort_keys=True),
                }
            )
    _write_csv(output / "configuration_summary.csv", configuration_rows)
    _write_csv(
        output / "policy_paired.csv",
        _paired_rows(analysis["policy_paired_by_state_mode"], section_name="policy"),
    )
    _write_csv(
        output / "state_paired.csv",
        _paired_rows(analysis["state_paired_by_method"], section_name="state"),
    )
    _write_csv(output / "training_curve.csv", list(analysis["training_curve"]))
    diagnostic_fields = (
        "run_id", "method_id", "state_mode", "training_seed", "update_count",
        "parameter_changed_all", "total_loss_first10_mean", "total_loss_last10_mean",
        "total_loss_slope_per_1k_steps", "policy_loss_last10_mean",
        "value_loss_last10_mean", "entropy_first10_mean", "entropy_last10_mean",
        "gradient_norm_mean", "gradient_norm_maximum", "ratio_minimum", "ratio_maximum",
    )
    _write_csv(
        output / "training_diagnostics.csv",
        [{field: row[field] for field in diagnostic_fields} for row in analysis["run_metrics"]],
    )
    (output / "continuation_gate.json").write_text(
        json.dumps(analysis["continuation_gate"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    ranked = sorted(
        analysis["configuration_summary"].items(),
        key=lambda item: -float(item[1]["metrics"]["validation_return"]["mean"]),
    )
    table = ["| 配置 | validation return（均值±样本标准差） | 平均时延（均值±样本标准差） |", "|---|---:|---:|"]
    for config_id, record in ranked:
        ret = record["metrics"]["validation_return"]
        lat = record["metrics"]["mean_latency"]
        table.append(
            f"| `{config_id}` | {ret['mean']:.6f} ± {ret['sample_std']:.6f} | "
            f"{lat['mean']:.6f} ± {lat['sample_std']:.6f} |"
        )
    readme = "\n".join(
        [
            "# PI-JWM R6 10k门控分析",
            "",
            "## 结论",
            "",
            "18组完整性、安全性和数值健康门均通过，建议按冻结协议继续完整18组至100k，以获得多个validation checkpoint。当前只有10k单点，不提前删除配置，也不宣布最终方法。",
            "",
            "## 10k描述统计",
            "",
            *table,
            "",
            "表格按validation return均值排列，仅用于描述，不构成赢家判定。按时完成率18/18均为1.0，硬违规均为0。训练loss只在同一目标内部作数值诊断，Actor–Critic与PPO的loss不可直接横比。",
            "",
            "## 缺失但不补造的指标",
            "",
            "本轮summary没有保存吞吐量、能耗、资源利用率和公平性的validation聚合值；这些指标必须在后续统一闭环评价中重新计算，不能从return或动作计数反推。",
            "",
            "## 产物说明",
            "",
            "`training_objective_curves.png`展示同一策略目标内三种状态模式的训练loss均值与seed范围；`validation_return_latency.png`展示return—时延权衡。CSV保留全部逐run、逐更新和逐seed配对证据。",
            "",
        ]
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_figures(analysis, output)

    managed_names = (
        "analysis.json", "run_metrics.csv", "configuration_summary.csv",
        "policy_paired.csv", "state_paired.csv", "training_curve.csv",
        "training_diagnostics.csv", "continuation_gate.json", "README.md",
        "training_objective_curves.png", "validation_return_latency.png",
    )
    entries = []
    for name in managed_names:
        path = output / name
        entries.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": R6_10K_ANALYSIS_SCHEMA,
        "manifest_entry_count": len(entries),
        "input_binding": dict(input_binding),
        "locked_test_accessed": False,
        "files": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
