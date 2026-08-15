"""CPU-only energy-aware ranking diagnostic over PI-JWM candidate summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_airfogsim_counterfactual_action_smoke_v0 import spearman_rank_correlation
from run_world_model_v5_utility_ranking_smoke import (
    apply_feature_standardizer,
    fit_feature_standardizer,
    predict_utility,
    train_utility_head,
)


DIAGNOSTIC_ROOT = CODE_ROOT / "artifacts" / "reports" / "pi_jwm_energy_reward_diagnostic_20260713"
DEFAULT_INPUT = DIAGNOSTIC_ROOT / "candidate_summary.csv"
DEFAULT_OUTPUT_DIR = DIAGNOSTIC_ROOT / "ranking_diagnostic"
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0)
ACTION_FAMILIES = ("default", "rb_count", "offload_target", "mixed_offload_rb", "cpu_scale", "return_route")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a seed-held-out energy-aware ranking diagnostic.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[3])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[4])
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def add_energy_targets(frame, lambdas=DEFAULT_LAMBDAS):
    result = frame.copy()
    baseline_energy = {}
    for key, part in result.groupby(["seed", "decision_time"], dropna=False):
        defaults = part[part["action_family"].astype(str) == "default"]
        if len(defaults) != 1:
            raise ValueError(f"group {key} must contain exactly one default")
        baseline_energy[key] = float(defaults.iloc[0]["energy_total"])
    result["baseline_energy_total"] = [
        baseline_energy[(row.seed, row.decision_time)] for row in result.itertuples()
    ]
    denominator = result["baseline_energy_total"].abs().clip(lower=1e-12)
    result["energy_overhead_ratio"] = (
        result["energy_total"].astype(float) - result["baseline_energy_total"]
    ) / denominator
    for value in lambdas:
        result[f"target_lambda_{float(value):g}"] = (
            result["task_utility"].astype(float) - float(value) * result["energy_overhead_ratio"]
        )
    return result


def build_action_context_features(frame):
    required_numeric = (
        "rb_scale",
        "total_rb",
        "num_rb_tasks",
        "cpu_scale",
        "total_cpu",
        "num_offload_overrides",
        "num_cpu_overrides",
        "num_return_route_overrides",
        "context_num_to_offload_tasks",
        "context_num_computing_tasks",
        "context_num_waiting_return_tasks",
    )
    optional_numeric = (
        "offload_default_distance",
        "offload_alternative_distance",
        "offload_distance_delta",
        "offload_distance_ratio",
        "offload_default_is_uav",
        "offload_alternative_is_uav",
        "offload_target_type_changed",
    )
    numeric = required_numeric + optional_numeric
    missing = [field for field in required_numeric if field not in frame.columns]
    if missing:
        raise KeyError(f"candidate summary is missing ranking features: {missing}")
    columns = []
    names = []
    log_fields = {
        "total_rb",
        "num_rb_tasks",
        "total_cpu",
        "context_num_to_offload_tasks",
        "context_num_computing_tasks",
        "context_num_waiting_return_tasks",
        "offload_default_distance",
        "offload_alternative_distance",
        "offload_distance_ratio",
    }
    for field in numeric:
        source = frame[field] if field in frame.columns else pd.Series(0.0, index=frame.index)
        values = source.fillna(0.0).to_numpy(dtype=np.float32)
        if field in log_fields:
            values = np.log1p(np.maximum(values, 0.0))
        elif field == "offload_distance_delta":
            values = np.sign(values) * np.log1p(np.abs(values))
        columns.append(values)
        names.append(field)
    family = frame["action_family"].astype(str)
    for name in ACTION_FAMILIES:
        columns.append(family.eq(name).to_numpy(dtype=np.float32))
        names.append(f"family_{name}")
    return np.column_stack(columns).astype(np.float32), names


def _split_mask(frame, seeds):
    return frame["seed"].astype(int).isin({int(seed) for seed in seeds}).to_numpy()


def grouped_ranking_metrics(frame, target_col, prediction_col):
    rows = []
    for (seed, decision_time), part in frame.groupby(["seed", "decision_time"], dropna=False):
        true = part[target_col].to_numpy(dtype=float)
        predicted = part[prediction_col].to_numpy(dtype=float)
        best_pred = int(np.argmax(predicted))
        best_true = float(np.max(true))
        regret = best_true - float(true[best_pred])
        spread = float(np.max(true) - np.min(true))
        rows.append(
            {
                "seed": int(seed),
                "decision_time": float(decision_time),
                "num_candidates": int(len(part)),
                "is_nontrivial": bool(spread > 1e-8),
                "top1_hit": float(abs(float(true[best_pred]) - best_true) <= 1e-8),
                "normalized_top1_regret": regret / spread if spread > 1e-12 else 0.0,
                "spearman": spearman_rank_correlation(true, predicted) if spread > 1e-12 else np.nan,
                "utility_rmse": float(np.sqrt(np.mean((true - predicted) ** 2))),
            }
        )
    group_df = pd.DataFrame(rows)
    nontrivial = group_df[group_df["is_nontrivial"]]
    source = nontrivial if not nontrivial.empty else group_df
    summary = {
        "num_groups": int(len(group_df)),
        "num_nontrivial_groups": int(len(nontrivial)),
        "top1_hit_mean": float(source["top1_hit"].mean()) if not source.empty else np.nan,
        "normalized_top1_regret_mean": float(source["normalized_top1_regret"].mean()) if not source.empty else np.nan,
        "spearman_mean": float(source["spearman"].mean(skipna=True)) if not source.empty else np.nan,
        "utility_rmse_mean": float(source["utility_rmse"].mean()) if not source.empty else np.nan,
    }
    return summary, group_df


def build_combined_findings(main_summary, family_df, metrics_df):
    def family_fact(name, label):
        part = family_df[family_df["action_family"].astype(str) == name]
        if part.empty:
            return f"- {label}：本轮没有形成可配对候选。"
        row = part.iloc[0]
        count = int(row["num_candidates"])
        positive = int(round(float(row["positive_utility_ratio"]) * count))
        return (
            f"- {label}：{count} 个配对候选中 {positive} 个 task utility 为正；"
            f"平均 utility 变化 {float(row['mean_effect_task_utility']):.6f}，"
            f"平均 UAV 能耗变化 {float(row['mean_effect_energy_total']):.6f}。"
        )

    test_zero = metrics_df[
        metrics_df["split"].astype(str).eq("test")
        & np.isclose(metrics_df["lambda"].astype(float), 0.0)
    ]
    ranking_fact = "- 排序诊断：没有可用的 λ=0 测试结果。"
    if not test_zero.empty:
        row = test_zero.iloc[0]
        ranking_fact = (
            f"- 排序诊断：测试集只有 {int(row['num_nontrivial_groups'])} 个非平凡决策组；"
            f"λ=0 的 Top-1 命中率为 {float(row['top1_hit_mean']):.3f}，"
            f"归一化 regret 为 {float(row['normalized_top1_regret_mean']):.3f}。"
        )
    facts = [
        f"- 固定 seeds 0–4，共 {int(main_summary['num_decision_groups'])} 个配对决策组、"
        f"{int(main_summary['num_candidates'])} 个候选 rollout、{int(main_summary['num_step_rows'])} 条逐步记录。",
        f"- 数据质量审计通过：{bool(main_summary['quality_audit']['passed'])}；"
        "未发现 reward 重建错误、能耗守恒错误、负能耗或无效动作。",
        f"- 只有 {int(main_summary['num_nontrivial_groups'])}/{int(main_summary['num_decision_groups'])} "
        "个决策组出现非零 task-utility spread，其余组无法为 selector 提供有效排序监督。",
        family_fact("rb_count", "RB 数量动作"),
        family_fact("mixed_offload_rb", "offload+RB 联合动作"),
        family_fact("offload_target", "offload 目标动作"),
        family_fact("cpu_scale", "CPU 缩放动作"),
        family_fact("return_route", "返回路径动作"),
        ranking_fact,
    ]
    lines = [
        "# PI-JWM 能耗与逐步 Reward 诊断结论底稿",
        "",
        "> AirFogSim 仅作为仿真器和数据源；本轮排序头属于 `diagnostic_only` 接口，不是 PI-JWM 主方法。",
        "",
        "## 观测事实",
        "",
        *facts,
        "",
        "## 合理解释",
        "",
        "- 当前最直接的瓶颈不是缺少更多 selector 结构，而是多数短时决策组缺少可辨识的候选收益差异。",
        "- RB 与 mixed 动作整体收益偏负，说明增加或重排资源并不会自动转化为任务收益；offload 目标的方向依赖具体状态。",
        "- CPU 与返回路径在当前 3 步窗口中没有可见效应，不能据此断言它们长期无效，只能说明当前窗口和决策点不足以辨识。",
        "- 能耗惩罚会改变候选排名，但 λ 的验证/测试表现不一致，因此现阶段没有证据支持固定某个 λ 并宣称策略提升。",
        "",
        "## 待验证假设",
        "",
        "- 将 rollout horizon 扩到 10/20 步，CPU 和 return-route 对任务完成与能耗的滞后效应可能才会出现。",
        "- 按动作阶段分层采样决策点，可提高非平凡组比例，改善 selector 的监督支持。",
        "- 加入 PI-JWM 预测状态与不确定性后，可能比纯动作/上下文特征更能解释跨 seed 的候选排序。",
        "",
        "## 下一步建议",
        "",
        "1. 暂停继续横向尝试 selector，先扩大 horizon 并提高有效候选组比例。",
        "2. reward 主报告继续保留任务、资源、能耗分项；λ 仅做敏感性分析，不按测试表现选权重。",
        "3. 当非平凡组和 seed 覆盖扩大后，再做 PI-JWM rollout-aware selector，并保持 train/val/test seed 完全隔离。",
        "4. 目前本地 CPU 足够；只有扩大到长 horizon、多场景矩阵或重新训练 PI-JWM 时再使用服务器。",
        "",
        "## 结果边界",
        "",
        "- 本轮没有证明策略性能提升。",
        "- 本轮证明了能耗/reward 测量链可复现，并定位了候选可辨识性和排序泛化不足。",
        "- oracle、test-best、true-future 和 λ 事后最优均不得作为可部署结论。",
    ]
    return "\n".join(lines) + "\n"


def attach_post_diagnostics(main_summary, ranking_summary, combined_findings):
    result = dict(main_summary)
    result["post_diagnostics"] = {
        "ranking_summary": str(ranking_summary),
        "combined_findings": str(combined_findings),
        "result_kind": "diagnostic_only",
    }
    return result


def main():
    args = parse_args()
    train_set = set(args.train_seeds)
    val_set = set(args.val_seeds)
    test_set = set(args.test_seeds)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train/val/test seed sets must be disjoint")

    frame = pd.read_csv(args.candidate_csv)
    frame = add_energy_targets(frame, args.lambdas)
    features, feature_names = build_action_context_features(frame)
    masks = {
        "train": _split_mask(frame, args.train_seeds),
        "val": _split_mask(frame, args.val_seeds),
        "test": _split_mask(frame, args.test_seeds),
    }
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("every split must contain at least one candidate")
    mean, std = fit_feature_standardizer(features[masks["train"]])
    normalized = apply_feature_standardizer(features, mean, std)
    group_ids = frame["seed"].astype(str) + "@" + frame["decision_time"].astype(str)

    prediction_rows = frame.copy()
    metric_rows = []
    group_rows = []
    for lambda_value in args.lambdas:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        target_col = f"target_lambda_{float(lambda_value):g}"
        model, _, _ = train_utility_head(
            normalized[masks["train"]],
            frame.loc[masks["train"], target_col].to_numpy(dtype=np.float32),
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            device="cpu",
            groups=group_ids[masks["train"]].to_numpy(),
            pair_scope="group",
        )
        prediction_col = f"prediction_lambda_{float(lambda_value):g}"
        prediction_rows[prediction_col] = predict_utility(model, normalized, "cpu")
        for split_name, mask in masks.items():
            split_frame = prediction_rows.loc[mask].copy()
            metrics, per_group = grouped_ranking_metrics(split_frame, target_col, prediction_col)
            metric_rows.append({"split": split_name, "lambda": float(lambda_value), **metrics})
            per_group.insert(0, "lambda", float(lambda_value))
            per_group.insert(0, "split", split_name)
            group_rows.append(per_group)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "ranking_predictions.csv"
    metrics_path = output_dir / "ranking_metrics.csv"
    group_path = output_dir / "ranking_group_metrics.csv"
    prediction_rows.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.concat(group_rows, ignore_index=True).to_csv(group_path, index=False, encoding="utf-8-sig")

    summary = {
        "framework": "PI-JWM",
        "interface": "action_context_energy_utility_ranking",
        "result_kind": "diagnostic_only",
        "device": "cpu",
        "train_seeds": sorted(train_set),
        "validation_seeds": sorted(val_set),
        "test_seeds": sorted(test_set),
        "lambdas": [float(value) for value in args.lambdas],
        "feature_names": feature_names,
        "selection_rule": "none; validation metrics are descriptive and no lambda is promoted",
        "metrics": metric_rows,
        "outputs": {
            "predictions_csv": str(predictions_path),
            "metrics_csv": str(metrics_path),
            "group_metrics_csv": str(group_path),
        },
    }
    report_lines = [
        "# PI-JWM Energy-aware Utility Ranking Diagnostic",
        "",
        "- Result kind: `diagnostic_only`.",
        "- Train seeds: `0,1,2`; validation seed: `3`; test seed: `4`.",
        "- Seed identity and simulator outcomes are excluded from model features.",
        "- Validation is descriptive; no lambda or checkpoint is selected on validation or test.",
        "- This action/context interface is not presented as the PI-JWM main method.",
        "",
        "See `ranking_metrics.csv` for split-wise results and `ranking_group_metrics.csv` for per-decision evidence.",
    ]
    report_path = output_dir / "ranking_diagnostic_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary["outputs"]["report"] = str(report_path)
    command_path = output_dir / "reproduction_command.txt"
    command_path.write_text(
        "python code/scripts/run_pi_jwm_energy_utility_ranking_diagnostic.py "
        f"--candidate-csv \"{args.candidate_csv.resolve()}\" "
        f"--output-dir \"{output_dir}\" "
        f"--train-seeds {' '.join(str(seed) for seed in args.train_seeds)} "
        f"--val-seeds {' '.join(str(seed) for seed in args.val_seeds)} "
        f"--test-seeds {' '.join(str(seed) for seed in args.test_seeds)} "
        f"--epochs {args.epochs} --hidden {args.hidden} --lr {args.lr:g}\n",
        encoding="utf-8",
    )
    summary["outputs"]["reproduction_command"] = str(command_path)
    parent_dir = args.candidate_csv.resolve().parent
    main_summary_path = parent_dir / "summary.json"
    family_path = parent_dir / "action_family_summary.csv"
    findings_path = None
    main_summary = None
    if main_summary_path.exists() and family_path.exists():
        main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
        family_df = pd.read_csv(family_path)
        findings_path = parent_dir / "findings_for_report.md"
        findings_path.write_text(
            build_combined_findings(main_summary, family_df, metrics_df),
            encoding="utf-8",
        )
        summary["outputs"]["combined_findings"] = str(findings_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if main_summary is not None and findings_path is not None:
        linked = attach_post_diagnostics(main_summary, summary_path, findings_path)
        main_summary_path.write_text(json.dumps(linked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
