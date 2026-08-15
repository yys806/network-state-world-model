"""Audit PI-JWM v11 candidate value/magnitude bottlenecks.

This script is intentionally light-weight. It consolidates existing experiment
artifacts and decides whether the remaining gap is mostly support/ranking,
aggregate total, or per-edge RB value/magnitude representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_value_gap_audit_20260628"


def _safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_gap(current: float, oracle: float, target: float) -> dict[str, float | bool]:
    current = float(current)
    oracle = float(oracle)
    target = float(target)
    return {
        "current": current,
        "oracle": oracle,
        "target": target,
        "needed_to_target": current - target,
        "oracle_headroom": target - oracle,
        "current_to_oracle": current - oracle,
        "oracle_can_meet_target": oracle < target,
    }


def classify_bottleneck(
    learned_rmse: float,
    oracle_rmse: float,
    value_pearson: float,
    support_corr: float,
    target_rmse: float,
) -> dict[str, Any]:
    gap = relative_gap(learned_rmse, oracle_rmse, target_rmse)
    reasons: list[str] = []
    value_corr = _safe_float(value_pearson)
    support = _safe_float(support_corr)

    if gap["oracle_can_meet_target"] and gap["needed_to_target"] > 0:
        reasons.append("oracle true-value path can meet target but learned path cannot")
    if math.isfinite(value_corr) and abs(value_corr) < 0.1:
        reasons.append("value predictor correlation is near zero")
    if math.isfinite(support) and abs(support) < 0.1:
        reasons.append("support/ranking correlation is near zero")

    if math.isfinite(value_corr) and abs(value_corr) < 0.1:
        primary = "value_magnitude_representation"
    elif math.isfinite(support) and abs(support) < 0.1:
        primary = "support_ranking_generalization"
    elif gap["needed_to_target"] > 0 and gap["oracle_can_meet_target"]:
        primary = "candidate_action_family_or_rollout_selector"
    else:
        primary = "gpu_ready_or_no_clear_gap"

    return {
        "primary_bottleneck": primary,
        "reasons": reasons,
        "gap": gap,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(label: str, summary_path: Path, payload: dict) -> dict:
    best = payload.get("best_val") or {}
    test = payload.get("matched_test_for_best_val") or {}
    diagnostics = payload.get("diagnostics") or {}
    return {
        "label": label,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "mode": payload.get("mode", ""),
        "candidate": test.get("candidate") or best.get("candidate", ""),
        "val_active_rate_rmse": _safe_float(best.get("active_rate_rmse")),
        "test_active_rate_rmse": _safe_float(test.get("active_rate_rmse")),
        "test_link_rmse": _safe_float(test.get("link_rmse")),
        "test_f1": _safe_float(test.get("activity_f1")),
        "val_value_pearson": _best_metric(diagnostics, "val_", "value_pearson"),
        "test_value_pearson": _best_metric(diagnostics, "test_", "value_pearson"),
        "val_value_mae_nonzero_true": _best_metric(diagnostics, "val_", "value_mae_nonzero_true", prefer_abs_min=True),
        "test_value_mae_nonzero_true": _best_metric(diagnostics, "test_", "value_mae_nonzero_true", prefer_abs_min=True),
        "val_support_corr": _first_metric(diagnostics, "val_", ["support_pearson", "support_score_corr", "support_score_corr_label", "diffusion_score_corr_label", "cem_score_corr_label"]),
        "test_support_corr": _first_metric(diagnostics, "test_", ["support_pearson", "support_score_corr", "support_score_corr_label", "diffusion_score_corr_label", "cem_score_corr_label"]),
    }


def _best_metric(diagnostics: dict, prefix: str, suffix: str, prefer_abs_min: bool = False) -> float:
    values = []
    for key, value in diagnostics.items():
        if str(key).startswith(prefix) and str(key).endswith(suffix):
            number = _safe_float(value)
            if math.isfinite(number):
                values.append(number)
    if not values:
        return math.nan
    if prefer_abs_min:
        return float(min(values, key=lambda item: abs(item)))
    return float(max(values, key=lambda item: abs(item)))


def _first_metric(diagnostics: dict, prefix: str, suffixes: list[str]) -> float:
    for suffix in suffixes:
        for key, value in diagnostics.items():
            if str(key).startswith(prefix) and str(key).endswith(suffix):
                number = _safe_float(value)
                if math.isfinite(number):
                    return number
    return math.nan


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        "triage_best_learned": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_branch_template_value_selector_cpu_triage_20260628/summary.json",
        "oracle_true_value": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_oracle_value_scope_diagnostic_20260622/summary.json",
        "rf_value_reconstruction": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rb_total_value_reconstruction_rf_confirm_1024_20260622/summary.json",
        "step_support_value": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_step_support_value_reconstruction_confirm_1024_20260622/summary.json",
        "xgb_value_ranked": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_xgb_value_ranked_allocation_tiny_20260628/summary.json",
        "xgb_support_rank": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_xgb_rank_support_256_20260628/summary.json",
        "diffusion_support": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_diffusion_support_generator_256_20260628/summary.json",
        "cem_refiner": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_cem_planner_refiner_256_20260628/summary.json",
        "rollout_template": PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_rollout_template_selector_narrow_256_20260628/summary.json",
    }

    rows = []
    payloads = {}
    for label, path in summaries.items():
        if not path.exists():
            continue
        payload = _load_json(path)
        payloads[label] = payload
        if label == "triage_best_learned":
            learned = payload.get("best_learned_by_validation") or {}
            test = learned.get("test") or {}
            val = learned.get("val") or {}
            rows.append({
                "label": label,
                "summary_path": str(path.relative_to(PROJECT_ROOT)),
                "mode": payload.get("mode", ""),
                "candidate": test.get("candidate") or val.get("candidate", ""),
                "val_active_rate_rmse": _safe_float(val.get("active_rate_rmse")),
                "test_active_rate_rmse": _safe_float(test.get("active_rate_rmse")),
                "test_link_rmse": _safe_float(test.get("link_rmse")),
                "test_f1": _safe_float(test.get("activity_f1")),
                "val_value_pearson": math.nan,
                "test_value_pearson": math.nan,
                "val_value_mae_nonzero_true": math.nan,
                "test_value_mae_nonzero_true": math.nan,
                "val_support_corr": math.nan,
                "test_support_corr": math.nan,
            })
        else:
            rows.append(_summary_row(label, path, payload))

    oracle_payload = payloads.get("oracle_true_value", {})
    triage_payload = payloads.get("triage_best_learned", {})
    oracle_rmse = _safe_float((oracle_payload.get("best_val") or {}).get("active_rate_rmse"), default=float(args.oracle_active_rmse))
    learned_test = ((triage_payload.get("best_learned_by_validation") or {}).get("test") or {})
    learned_rmse = _safe_float(learned_test.get("active_rate_rmse"), default=float(args.current_active_rmse))

    value_rows = [row for row in rows if math.isfinite(row.get("test_value_pearson", math.nan))]
    if value_rows:
        best_value_row = max(value_rows, key=lambda row: abs(float(row["test_value_pearson"])))
        value_pearson = float(best_value_row["test_value_pearson"])
    else:
        best_value_row = {}
        value_pearson = math.nan

    support_rows = [row for row in rows if math.isfinite(row.get("test_support_corr", math.nan))]
    if support_rows:
        best_support_row = max(support_rows, key=lambda row: abs(float(row["test_support_corr"])))
        support_corr = float(best_support_row["test_support_corr"])
    else:
        best_support_row = {}
        support_corr = math.nan

    verdict = classify_bottleneck(
        learned_rmse=learned_rmse,
        oracle_rmse=oracle_rmse,
        value_pearson=value_pearson,
        support_corr=support_corr,
        target_rmse=float(args.target_rmse),
    )

    _write_csv(output_dir / "value_gap_audit_rows.csv", rows)
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "value_gap_audit",
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "target_rmse": float(args.target_rmse),
        "learned_reference_active_rmse": learned_rmse,
        "oracle_true_value_active_rmse": oracle_rmse,
        "best_value_predictability_row": best_value_row,
        "best_support_predictability_row": best_support_row,
        "verdict": verdict,
        "rows_csv": str((output_dir / "value_gap_audit_rows.csv").relative_to(PROJECT_ROOT)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# PI-JWM v11 Candidate Value Gap Audit",
        "",
        f"- Learned reference active RMSE: `{learned_rmse}`",
        f"- Oracle true-value active RMSE: `{oracle_rmse}`",
        f"- Target active RMSE: `{float(args.target_rmse)}`",
        f"- Primary bottleneck: `{verdict['primary_bottleneck']}`",
        f"- Reasons: {', '.join(verdict['reasons'])}",
        "",
        "## Decision",
        "",
        "Do not scale GPU until the next method changes value/magnitude representation or trains a policy to emit calibrated RB values directly.",
        "",
    ]
    (output_dir / "value_gap_audit.md").write_text("\n".join(md), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-rmse", type=float, default=200.0)
    parser.add_argument("--current-active-rmse", type=float, default=222.26801009733668)
    parser.add_argument("--oracle-active-rmse", type=float, default=107.6383284059626)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
