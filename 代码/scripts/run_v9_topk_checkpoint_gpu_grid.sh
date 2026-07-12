#!/usr/bin/env bash
set -euo pipefail

ROOT=${PI_JWM_ROOT:-/root/autodl-tmp/pi_jwm_gpu_m5}
PY=${PYTHON_BIN:-/root/miniconda3/bin/python}
cd "$ROOT"

DATA="$ROOT/artifacts/experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v1"
OUT="$ROOT/artifacts/experiments/pi_jwm_v9_topk_checkpoint_20260617"
mkdir -p "$OUT"

METRIC_CHECKPOINTS="val_active_rate_rmse,val_link_rate_rmse,val_activity_f1,val_precision_constrained_composite,val_link_f1_constrained_active_rate,val_link_f1_constrained_composite"

COMMON=(
  --dataset-dir "$DATA"
  --max-train-samples 999999 --max-val-samples 999999 --max-test-samples 999999
  --batch-size 64 --hidden-dim 64 --device cuda
  --train-seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --val-seeds 16,17 --test-seeds 18,19
  --fusion-mode cross_attention --fusion-num-heads 4
  --history-encoder mean --latent-transition-mode recurrent
  --active-rate-auxiliary --active-rate-auxiliary-weight 0.3
  --active-rate-head-mode mlp
  --model-rate-output-mode hurdle_soft
  --rate-output-mode main --rate-loss-mode active_mixed --inactive-rate-weight 0.05
  --node-loss-weight 1.0 --activity-loss-weight 1.5 --rate-loss-weight 0.3 --task-loss-weight 1.0
  --activity-loss-mode focal --activity-pos-weight 160 --activity-focal-gamma 2.0
  --inactive-loss-sample-ratio 0.25
  --hurdle-train-gate-mode predicted
  --hurdle-train-gate-power 1.0
  --eval-hurdle-gate-power 1.0
  --best-min-precision 0.01 --best-min-recall 0.05
  --best-min-f1 0.027 --best-max-link-rmse 90.0
  --best-f1-penalty-weight 1000.0 --best-link-penalty-weight 10.0
  --metric-checkpoints "$METRIC_CHECKPOINTS"
)

run_one() {
  local name="$1"
  shift
  echo "===== START ${name} $(date '+%F %T %z') =====" | tee -a "$OUT/queue.log"
  "$PY" scripts/run_world_model_v8_full_training.py \
    "${COMMON[@]}" \
    "$@" \
    --output-dir "$OUT/$name" \
    2>&1 | tee "$OUT/${name}.log"
  echo "===== END ${name} $(date '+%F %T %z') =====" | tee -a "$OUT/queue.log"
}

run_one k0_hurdle_seed20260614 \
  --seed 20260614 \
  --epochs 160 \
  --best-metric val_precision_constrained_composite

run_one k1_hurdle_seed20260615 \
  --seed 20260615 \
  --epochs 160 \
  --best-metric val_link_f1_constrained_active_rate

run_one k2_hurdle_seed20260617 \
  --seed 20260617 \
  --epochs 160 \
  --best-metric val_link_f1_constrained_composite

run_one k3_hurdle_seed20260614_long \
  --seed 20260614 \
  --epochs 220 \
  --best-metric val_link_f1_constrained_active_rate

SEARCH_OUT="$OUT/operating_search_all_checkpoints"
mkdir -p "$SEARCH_OUT"
"$PY" scripts/evaluate_v9_hurdle_gate_calibration.py \
  --device cuda \
  --batch-size 64 \
  --output-dir "$SEARCH_OUT" \
  --checkpoint-glob "v8_dual_best*.pt" \
  --selection-metric constrained_active_rate \
  --min-f1 0.027 \
  --max-link-rmse 90.0 \
  --f1-penalty-weight 1000.0 \
  --link-penalty-weight 10.0 \
  --temperatures 0.75,1.0,1.25 \
  --powers 0.5,0.75,1.0 \
  --thresholds 0.5,0.7,0.83,0.9,0.97 \
  --positive-rate-scales 0.75,1.0,1.25 \
  --rate-gate-modes soft \
  --selective-min-activity-probs 0.0,0.03 \
  --selective-min-positive-rates 0.0,300.0 \
  --experiment "$OUT/k0_hurdle_seed20260614" \
  --experiment "$OUT/k1_hurdle_seed20260615" \
  --experiment "$OUT/k2_hurdle_seed20260617" \
  --experiment "$OUT/k3_hurdle_seed20260614_long" \
  > "$SEARCH_OUT/search_stdout.json"

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

out = Path("/root/autodl-tmp/pi_jwm_gpu_m5/artifacts/experiments/pi_jwm_v9_topk_checkpoint_20260617")
search_out = out / "operating_search_all_checkpoints"
summary = json.loads((search_out / "gate_calibration_summary.json").read_text())
anchor = {
    "experiment": "c8a_hurdle_anchor",
    "checkpoint": "historical_c8a",
    "test_active": 292.83030251003066,
    "test_f1": 0.027108433734939756,
    "test_link": 89.14184698976241,
    "uncal_active": 292.83030251003066,
    "uncal_f1": 0.027108433734939756,
    "uncal_link": 89.14184698976241,
    "reportable": True,
}
rows = []
for exp in summary["experiments"]:
    label = exp.get("experiment_label") or Path(exp["experiment"]).name
    best = exp["best_val"]
    test = exp["test_calibrated"]
    unc = exp["test_uncalibrated_gate"]
    row = {
        "experiment": label,
        "checkpoint": exp.get("checkpoint_name", ""),
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
    row["reportable"] = (
        row["test_active"] < anchor["test_active"]
        and row["test_f1"] >= anchor["test_f1"]
        and row["test_link"] <= 90.0
    )
    rows.append(row)

rows.sort(key=lambda r: (not r["reportable"], r["test_active"], -r["test_f1"], r["test_link"]))
fields = list(rows[0])
with (out / "topk_operating_search_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

with (out / "topk_operating_search_comparison.md").open("w") as f:
    f.write("# PI-JWM v9 Top-k Checkpoint Operating Search\n\n")
    f.write("Training: original hurdle_soft, active-heavy v1 split, multi-metric checkpoint retention. Selection: validation constrained active-rate; reportable requires test active-rate < C8a, F1 >= C8a, and link RMSE <= 90.\n\n")
    f.write("C8a anchor: active-rate=292.830303, F1=0.027108, link=89.141847.\n\n")
    f.write("| Experiment/checkpoint | val ok | temp | power | scale | min prob | min pos | thr | Test P | Test R | Test F1 | Test active | Test pos active | Test link | Uncal active | Uncal F1 | Uncal link | d active | d F1 | d link | reportable |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in rows:
        f.write("| " + " | ".join([
            fmt(row["experiment"]),
            fmt(row["val_meets_constraints"]),
            fmt(row["temperature"]),
            fmt(row["power"]),
            fmt(row["positive_rate_scale"]),
            fmt(row["min_activity_prob"]),
            fmt(row["min_positive_rate"]),
            fmt(row["threshold"]),
            fmt(row["test_precision"]),
            fmt(row["test_recall"]),
            fmt(row["test_f1"]),
            fmt(row["test_active"]),
            fmt(row["test_positive_active"]),
            fmt(row["test_link"]),
            fmt(row["uncal_active"]),
            fmt(row["uncal_f1"]),
            fmt(row["uncal_link"]),
            fmt(row["active_delta_vs_c8a"]),
            fmt(row["f1_delta_vs_c8a"]),
            fmt(row["link_delta_vs_c8a"]),
            fmt(row["reportable"]),
        ]) + " |\n")
print(out / "topk_operating_search_comparison.md")
PY

nvidia-smi > "$OUT/final_nvidia_smi.txt" || true
