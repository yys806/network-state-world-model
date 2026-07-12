#!/usr/bin/env bash
set -euo pipefail

ROOT=${PI_JWM_ROOT:-/root/autodl-tmp/pi_jwm_gpu_m5}
PY=${PYTHON_BIN:-/root/miniconda3/bin/python}
cd "$ROOT"

OUT="$ROOT/artifacts/experiments/pi_jwm_v9_existing_checkpoint_operating_search_20260617"
mkdir -p "$OUT"

"$PY" scripts/evaluate_v9_hurdle_gate_calibration.py \
  --device cuda \
  --batch-size 64 \
  --output-dir "$OUT" \
  --selection-metric constrained_active_rate \
  --min-f1 0.027 \
  --max-link-rmse 90.0 \
  --f1-penalty-weight 1000.0 \
  --link-penalty-weight 10.0 \
  --temperatures 0.5,0.75,1.0,1.25,1.5,2.0 \
  --powers 0.0,0.25,0.5,0.75,1.0,1.25 \
  --positive-rate-scales 0.5,0.75,1.0,1.25,1.5 \
  --rate-gate-modes soft \
  --selective-min-activity-probs 0.0,0.01,0.03,0.05,0.1 \
  --selective-min-positive-rates 0.0,100.0,300.0,600.0 \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_candidate_hurdle_event_20260615/c8a_hurdle_only" \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_hurdle_refine_20260615/r0_hurdle_baseline_repeat" \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_train_gate_power_20260617/q3_train_p05_eval_p05_active" \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_high_rate_weight_20260617/h2_high_w4" \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_high_rate_link_balance_20260617/b2_w4_inactive010" \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_positive_rate_specialist_20260616/p2_log1p_norm_mse" \
  --experiment "$ROOT/artifacts/experiments/pi_jwm_v9_c9_repair_20260617/r0_c8a_original_seed20260614" \
  > "$OUT/search_stdout.json"

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

out = Path("/root/autodl-tmp/pi_jwm_gpu_m5/artifacts/experiments/pi_jwm_v9_existing_checkpoint_operating_search_20260617")
summary = json.loads((out / "gate_calibration_summary.json").read_text())
anchor = {
    "experiment": "c8a_anchor_reference",
    "test_active": 292.83030251003066,
    "test_f1": 0.027108433734939756,
    "test_link": 89.14184698976241,
}
rows = []
for exp in summary["experiments"]:
    name = Path(exp["experiment"]).name
    best = exp["best_val"]
    test = exp["test_calibrated"]
    unc = exp["test_uncalibrated_gate"]
    row = {
        "experiment": name,
        "val_meets_constraints": best.get("meets_constraints"),
        "val_score": best.get("score"),
        "temperature": test["temperature"],
        "power": test["power"],
        "positive_rate_scale": test["positive_rate_scale"],
        "rate_gate_mode": test["rate_gate_mode"],
        "min_activity_prob": test["min_activity_prob"],
        "min_positive_rate": test["min_positive_rate"],
        "threshold": test["threshold"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_active": test["active_rate_rmse"],
        "test_positive_active": test["positive_rate_active_rmse"],
        "test_link": test["link_rate_rmse"],
        "uncal_f1": unc["f1"],
        "uncal_active": unc["active_rate_rmse"],
        "uncal_link": unc["link_rate_rmse"],
    }
    row["active_delta_vs_c8a"] = row["test_active"] - anchor["test_active"]
    row["f1_delta_vs_c8a"] = row["test_f1"] - anchor["test_f1"]
    row["link_delta_vs_c8a"] = row["test_link"] - anchor["test_link"]
    row["reportable"] = row["test_active"] < anchor["test_active"] and row["test_f1"] >= anchor["test_f1"] and row["test_link"] <= 90.0
    rows.append(row)

rows.sort(key=lambda r: (not r["reportable"], r["test_active"]))
fields = list(rows[0])
with (out / "operating_search_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)

with (out / "operating_search_comparison.md").open("w") as f:
    f.write("# PI-JWM v9 Existing Checkpoint Operating-Point Search\n\n")
    f.write("Selection: validation constrained active-rate, target F1 >= 0.027 and link RMSE <= 90. Reportable requires test active-rate < C8a, test F1 >= C8a, test link <= 90.\n\n")
    f.write("| Experiment | val ok | temp | power | scale | min prob | min pos | thr | Test P | Test R | Test F1 | Test active | Test pos active | Test link | Uncal active | Uncal F1 | Uncal link | d active | d F1 | d link | reportable |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        f.write("| " + " | ".join([
            fmt(r["experiment"]),
            fmt(r["val_meets_constraints"]),
            fmt(r["temperature"]),
            fmt(r["power"]),
            fmt(r["positive_rate_scale"]),
            fmt(r["min_activity_prob"]),
            fmt(r["min_positive_rate"]),
            fmt(r["threshold"]),
            fmt(r["test_precision"]),
            fmt(r["test_recall"]),
            fmt(r["test_f1"]),
            fmt(r["test_active"]),
            fmt(r["test_positive_active"]),
            fmt(r["test_link"]),
            fmt(r["uncal_active"]),
            fmt(r["uncal_f1"]),
            fmt(r["uncal_link"]),
            fmt(r["active_delta_vs_c8a"]),
            fmt(r["f1_delta_vs_c8a"]),
            fmt(r["link_delta_vs_c8a"]),
            fmt(r["reportable"]),
        ]) + " |\n")
print(out / "operating_search_comparison.md")
PY

nvidia-smi > "$OUT/final_nvidia_smi.txt" || true
