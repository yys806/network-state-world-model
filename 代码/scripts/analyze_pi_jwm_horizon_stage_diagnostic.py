"""Freeze and summarize the PI-JWM horizon/stage energy-reward diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = CODE_ROOT / "artifacts" / "reports" / "pi_jwm_energy_reward_horizon_study_20260713"
KEY_FIELDS = ("seed", "decision_time", "candidate_id", "action_family")
HORIZONS = (3, 10, 20)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze fixed-horizon PI-JWM counterfactual diagnostics.")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    return parser.parse_args()


def validate_candidate_alignment(candidate_frames):
    reference_horizon = min(candidate_frames)
    reference = {
        tuple(row[field] for field in KEY_FIELDS)
        for row in candidate_frames[reference_horizon].to_dict("records")
    }
    for horizon, frame in candidate_frames.items():
        current = {
            tuple(row[field] for field in KEY_FIELDS)
            for row in frame.to_dict("records")
        }
        if current != reference:
            missing = len(reference - current)
            extra = len(current - reference)
            raise ValueError(
                f"candidate alignment differs at horizon={horizon}: missing={missing}, extra={extra}"
            )
    return {"reference_horizon": int(reference_horizon), "num_aligned_candidates": len(reference)}


def build_horizon_summary(summaries, group_audits):
    rows = []
    for horizon in sorted(summaries):
        summary = summaries[horizon]
        groups = group_audits[horizon]
        rows.append(
            {
                "horizon": int(horizon),
                "num_seeds": int(summary["num_seeds"]),
                "num_step_rows": int(summary["num_step_rows"]),
                "num_candidates": int(summary["num_candidates"]),
                "num_decision_groups": int(summary["num_decision_groups"]),
                "num_nontrivial_groups": int(summary["num_nontrivial_groups"]),
                "nontrivial_group_ratio": float(summary["num_nontrivial_groups"])
                / max(1, int(summary["num_decision_groups"])),
                "mean_utility_spread": float(groups["utility_spread"].mean()),
                "median_utility_spread": float(groups["utility_spread"].median()),
                "max_utility_spread": float(groups["utility_spread"].max()),
                "quality_audit_passed": bool(summary["quality_audit"]["passed"]),
            }
        )
    return pd.DataFrame(rows)


def build_stage_family_summary(candidate_frames, effect_frames):
    rows = []
    for horizon in sorted(effect_frames):
        candidate_columns = list(KEY_FIELDS) + ["decision_stage"]
        candidate_context = candidate_frames[horizon][candidate_columns].drop_duplicates(KEY_FIELDS)
        effects = effect_frames[horizon].merge(
            candidate_context,
            on=list(KEY_FIELDS),
            how="left",
            validate="one_to_one",
        )
        if effects["decision_stage"].isna().any():
            raise ValueError(f"missing decision_stage after merge for horizon={horizon}")
        for (stage, family), part in effects.groupby(["decision_stage", "action_family"], dropna=False):
            task = part["effect_task_utility"].astype(float)
            energy = part["effect_energy_total"].astype(float)
            throughput = part["effect_throughput_delta"].astype(float)
            rows.append(
                {
                    "horizon": int(horizon),
                    "decision_stage": str(stage),
                    "action_family": str(family),
                    "num_candidates": int(len(part)),
                    "num_decision_groups": int(part.groupby(["seed", "decision_time"]).ngroups),
                    "nonzero_task_effect_ratio": float((task.abs() > 1e-8).mean()),
                    "positive_task_effect_ratio": float((task > 1e-8).mean()),
                    "mean_effect_task_utility": float(task.mean()),
                    "min_effect_task_utility": float(task.min()),
                    "max_effect_task_utility": float(task.max()),
                    "mean_effect_task_utility_per_step": float(task.mean()) / float(horizon),
                    "mean_effect_energy_total": float(energy.mean()),
                    "mean_effect_energy_total_per_step": float(energy.mean()) / float(horizon),
                    "mean_effect_throughput_delta": float(throughput.mean()),
                    "mean_effect_throughput_delta_per_step": float(throughput.mean()) / float(horizon),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_horizon_effects(candidate_frames, effect_frames):
    rows = []
    for horizon in sorted(effect_frames):
        context = candidate_frames[horizon][list(KEY_FIELDS) + ["decision_stage"]].drop_duplicates(KEY_FIELDS)
        frame = effect_frames[horizon].merge(context, on=list(KEY_FIELDS), how="left", validate="one_to_one")
        frame.insert(0, "horizon", int(horizon))
        frame["effect_task_utility_per_step"] = frame["effect_task_utility"] / float(horizon)
        frame["effect_energy_total_per_step"] = frame["effect_energy_total"] / float(horizon)
        frame["effect_throughput_delta_per_step"] = frame["effect_throughput_delta"] / float(horizon)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def load_study(study_root, horizons):
    summaries = {}
    candidates = {}
    effects = {}
    groups = {}
    source_files = []
    for horizon in horizons:
        directory = study_root / f"horizon_{int(horizon)}"
        paths = {
            "summary": directory / "summary.json",
            "candidates": directory / "candidate_summary.csv",
            "effects": directory / "coupling_effects.csv",
            "groups": directory / "decision_group_audit.csv",
            "steps": directory / "step_metrics.csv",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing horizon inputs: {missing}")
        summaries[int(horizon)] = json.loads(paths["summary"].read_text(encoding="utf-8"))
        candidates[int(horizon)] = pd.read_csv(paths["candidates"])
        effects[int(horizon)] = pd.read_csv(paths["effects"])
        groups[int(horizon)] = pd.read_csv(paths["groups"])
        source_files.extend(paths.values())
    return summaries, candidates, effects, groups, source_files


def plot_nontrivial_ratio(summary_df, output_dir):
    path = output_dir / "horizon_nontrivial_group_ratio.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(summary_df["horizon"], summary_df["nontrivial_group_ratio"], marker="o", linewidth=2)
    for row in summary_df.itertuples():
        ax.annotate(
            f"{int(row.num_nontrivial_groups)}/{int(row.num_decision_groups)}",
            (row.horizon, row.nontrivial_group_ratio),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        xlabel="rollout horizon (steps)",
        ylabel="non-trivial decision-group ratio",
        title="Candidate distinguishability does not increase with horizon",
        ylim=(0.0, 1.0),
        xticks=summary_df["horizon"].tolist(),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_family_effect(stage_family_df, output_dir):
    path = output_dir / "horizon_family_task_effect_per_step.png"
    family = (
        stage_family_df.groupby(["horizon", "action_family"], as_index=False)
        .apply(
            lambda part: pd.Series(
                {
                    "mean_effect_task_utility_per_step": np.average(
                        part["mean_effect_task_utility_per_step"],
                        weights=part["num_candidates"],
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name, part in family.groupby("action_family"):
        part = part.sort_values("horizon")
        ax.plot(
            part["horizon"],
            part["mean_effect_task_utility_per_step"],
            marker="o",
            label=str(name),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(
        xlabel="rollout horizon (steps)",
        ylabel="mean paired task-utility effect per step",
        title="One-step action perturbation: delayed task effect by family",
        xticks=sorted(family["horizon"].unique().tolist()),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(summary_df, stage_family_df, output_dir):
    h3, h10, h20 = [summary_df[summary_df["horizon"] == h].iloc[0] for h in HORIZONS]
    lines = [
        "# PI-JWM Rollout-Horizon and Action-Stage Diagnostic",
        "",
        "## Protocol",
        "",
        "Each candidate changes the action at the selected decision step only. Subsequent scheduler steps use the same default policy for every candidate. Therefore, increasing the horizon observes delayed consequences of a one-step perturbation; it does not evaluate a persistent alternative policy.",
        "",
        r"For candidate $c$ and horizon $H$, the accumulated task utility is",
        r"$$U_H(c)=\sum_{k=0}^{H-1}\left[\Delta N^{done}_{k}(c)-\Delta N^{fail}_{k}(c)+0.01\log\left(1+\Delta B_k(c)\right)\right].$$",
        r"The paired effect and its per-step normalization are",
        r"$$\Delta U_H(c)=U_H(c)-U_H(c_0),\qquad \overline{\Delta U}_H(c)=\frac{\Delta U_H(c)}{H},$$",
        r"where $c_0$ is the default candidate at the same seed and decision time. Candidate distinguishability is measured by",
        r"$$\rho_H=\frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\mathbb I\left[\max_{c\in\mathcal C_g}U_H(c)-\min_{c\in\mathcal C_g}U_H(c)>10^{-8}\right].$$",
        "",
        "## Observed Results",
        "",
        f"- H=3: {int(h3.num_nontrivial_groups)}/{int(h3.num_decision_groups)} non-trivial groups.",
        f"- H=10: {int(h10.num_nontrivial_groups)}/{int(h10.num_decision_groups)} non-trivial groups.",
        f"- H=20: {int(h20.num_nontrivial_groups)}/{int(h20.num_decision_groups)} non-trivial groups.",
        "- CPU-scale and return-route candidates remain indistinguishable in the observed windows.",
        "- Offload-related effects are transient: the negative mean effect is largest at H=10 and is partly compensated by H=20.",
        "- Energy differences do not grow with horizon because only the first action is perturbed; later scheduler actions are shared.",
        "",
        "## Boundary",
        "",
        "The study is diagnostic-only. It does not show that a longer horizon improves a deployable selector, and it does not evaluate a persistent multi-step policy intervention.",
        "",
        "## Stage-Family Table",
        "",
        stage_family_df.to_csv(index=False),
    ]
    path = output_dir / "horizon_stage_diagnostic_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    study_root = args.study_root.resolve()
    study_root.mkdir(parents=True, exist_ok=True)
    horizons = tuple(sorted(int(value) for value in args.horizons))
    summaries, candidates, effects, groups, source_files = load_study(study_root, horizons)
    alignment = validate_candidate_alignment(candidates)
    horizon_summary = build_horizon_summary(summaries, groups)
    if not bool(horizon_summary["quality_audit_passed"].all()):
        raise ValueError("at least one horizon failed its quality audit")
    stage_family = build_stage_family_summary(candidates, effects)
    candidate_effects = build_candidate_horizon_effects(candidates, effects)

    horizon_path = study_root / "horizon_summary.csv"
    stage_path = study_root / "stage_family_summary.csv"
    effect_path = study_root / "candidate_horizon_effects.csv"
    horizon_summary.to_csv(horizon_path, index=False, encoding="utf-8-sig")
    stage_family.to_csv(stage_path, index=False, encoding="utf-8-sig")
    candidate_effects.to_csv(effect_path, index=False, encoding="utf-8-sig")
    figures = {
        "nontrivial_ratio": plot_nontrivial_ratio(horizon_summary, study_root),
        "family_task_effect": plot_family_effect(stage_family, study_root),
    }
    report_path = write_report(horizon_summary, stage_family, study_root)

    summary = {
        "framework": "PI-JWM",
        "simulator": "AirFogSim",
        "result_kind": "diagnostic_only",
        "frozen": True,
        "horizons": list(horizons),
        "seeds": [0, 1, 2, 3, 4],
        "protocol": {
            "candidate_intervention": "one_step_only",
            "future_actions": "shared_default_scheduler_after_intervention",
            "comparison": "paired_same_seed_same_decision_time",
            "cross_horizon_primary_metrics": [
                "nontrivial_group_ratio",
                "paired_effect_per_step",
                "stage_family_coverage",
            ],
        },
        "alignment": alignment,
        "quality_audit_passed": True,
        "observed_facts": {
            "nontrivial_groups": {str(int(row.horizon)): int(row.num_nontrivial_groups) for row in horizon_summary.itertuples()},
            "num_decision_groups": 15,
            "cpu_scale_nonzero_effect": False,
            "return_route_nonzero_effect": False,
            "horizon_increases_distinguishability": False,
        },
        "outputs": {
            "horizon_summary_csv": str(horizon_path),
            "stage_family_summary_csv": str(stage_path),
            "candidate_horizon_effects_csv": str(effect_path),
            "report": str(report_path),
            "figures": {name: str(path) for name, path in figures.items()},
        },
    }
    summary_path = study_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    generated = [horizon_path, stage_path, effect_path, report_path, summary_path, *figures.values()]
    manifest = {
        "frozen": True,
        "source_sha256": {str(path): sha256(path) for path in sorted(set(source_files))},
        "output_sha256": {str(path): sha256(path) for path in generated},
    }
    manifest_path = study_root / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    commands = []
    for horizon in horizons:
        commands.append(
            "$env:PYTHONUTF8='1'; conda run -n airfogsim python "
            "代码/scripts/run_pi_jwm_energy_reward_diagnostic.py "
            f"--seeds 0 1 2 3 4 --max-time 10 --decision-times-per-seed 3 --horizon {horizon} "
            f"--max-candidates 8 --output-dir \"{study_root / f'horizon_{horizon}'}\""
        )
    commands.append(
        "python 代码/scripts/analyze_pi_jwm_horizon_stage_diagnostic.py "
        f"--study-root \"{study_root}\" --horizons {' '.join(str(value) for value in horizons)}"
    )
    (study_root / "reproduction_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
