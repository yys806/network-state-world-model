import argparse
import json
from pathlib import Path

import pandas as pd

from run_airfogsim_counterfactual_action_smoke_v0 import (
    OUTPUT_DIR as SMOKE_OUTPUT_DIR,
    build_candidate_plans,
    default_rb_plan,
    display_path,
    make_env,
    plot_candidates,
    run_candidate,
    world_model_proxy_utility,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "airfogsim_counterfactual_label_dataset_v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a compact AirFogSim counterfactual candidate label dataset.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--scan-step-limit", type=int, default=80)
    parser.add_argument("--decision-times-per-seed", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def summarize_label_dataset(df):
    if df.empty:
        return {
            "num_rows": 0,
            "num_seeds": 0,
            "num_decision_groups": 0,
            "mean_candidates_per_group": 0.0,
        }
    groups = df.groupby(["seed", "decision_time"], dropna=False)
    return {
        "num_rows": int(len(df)),
        "num_seeds": int(df["seed"].nunique()),
        "num_decision_groups": int(groups.ngroups),
        "mean_candidates_per_group": float(groups.size().mean()),
        "min_candidates_per_group": int(groups.size().min()),
        "max_candidates_per_group": int(groups.size().max()),
        "utility_min": float(df["airfogsim_utility"].min()),
        "utility_max": float(df["airfogsim_utility"].max()),
    }


def discover_rb_decision_points(seed, max_time, max_points, scan_step_limit):
    env, algorithm = make_env(seed, max_time=max_time)
    points = []
    try:
        steps = 0
        while (not env.isDone()) and steps < scan_step_limit and len(points) < max_points:
            decision_time = float(env.simulation_time)
            plan = default_rb_plan(env, algorithm)
            if plan:
                n_rb = algorithm.commScheduler.getNumberOfRB(env)
                points.append({"seed": seed, "decision_time": decision_time, "default_plan": plan, "n_rb": n_rb})
            env.step()
            steps += 1
    finally:
        env.close()
    return points


def generate_label_rows(seeds, max_time, decision_times_per_seed, horizon, max_candidates, scan_step_limit):
    rows = []
    point_rows = []
    for seed in seeds:
        points = discover_rb_decision_points(seed, max_time, decision_times_per_seed, scan_step_limit)
        for point_idx, point in enumerate(points):
            candidates = build_candidate_plans(point["default_plan"], point["n_rb"], max_candidates)
            point_rows.append(
                {
                    "seed": int(seed),
                    "decision_time": float(point["decision_time"]),
                    "num_candidates": int(len(candidates)),
                    "point_index": int(point_idx),
                }
            )
            for candidate in candidates:
                row = run_candidate(seed, point["decision_time"], horizon, max_time, candidate)
                row["decision_group_id"] = f"seed{seed}_t{point['decision_time']:.3f}"
                row["point_index"] = int(point_idx)
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(point_rows)


def write_report(summary, labels_df, points_df, output_dir):
    lines = [
        "# AirFogSim counterfactual label dataset v0",
        "",
        "## Goal",
        "",
        "This dataset expands the single counterfactual smoke test into multiple seed/time action-candidate labels for v5 utility/ranking training.",
        "",
        "## Summary",
        "",
        pd.DataFrame([summary["dataset"]]).to_markdown(index=False),
        "",
        "## Decision Points",
        "",
        points_df.to_markdown(index=False) if not points_df.empty else "No decision points found.",
        "",
        "## Label Preview",
        "",
        labels_df.head(20).to_markdown(index=False) if not labels_df.empty else "No labels generated.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path = output_dir / "airfogsim_counterfactual_label_dataset_v0_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_df, points_df = generate_label_rows(
        args.seeds,
        args.max_time,
        args.decision_times_per_seed,
        args.horizon,
        args.max_candidates,
        args.scan_step_limit,
    )
    if not labels_df.empty:
        labels_df["world_model_utility"] = labels_df.apply(world_model_proxy_utility, axis=1)
    labels_path = args.output_dir / "airfogsim_counterfactual_label_dataset_v0_labels.csv"
    points_path = args.output_dir / "airfogsim_counterfactual_label_dataset_v0_points.csv"
    labels_df.to_csv(labels_path, index=False, encoding="utf-8-sig")
    points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
    plot_path = plot_candidates(labels_df) if not labels_df.empty else ""
    summary = {
        "seeds": [int(seed) for seed in args.seeds],
        "max_time": float(args.max_time),
        "horizon": int(args.horizon),
        "max_candidates": int(args.max_candidates),
        "decision_times_per_seed": int(args.decision_times_per_seed),
        "dataset": summarize_label_dataset(labels_df),
        "outputs": {
            "labels_csv": display_path(labels_path),
            "points_csv": display_path(points_path),
            "summary_plot": display_path(plot_path) if plot_path else "",
        },
    }
    report_path = write_report(summary, labels_df, points_df, args.output_dir)
    summary["outputs"]["report_md"] = display_path(report_path)
    summary_path = args.output_dir / "airfogsim_counterfactual_label_dataset_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
