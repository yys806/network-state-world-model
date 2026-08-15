#!/usr/bin/env bash
set -euo pipefail

ROOT=${PI_JWM_ROOT:-/root/autodl-tmp/pi_jwm_gpu_m5}
PY=${PYTHON_BIN:-/root/miniconda3/bin/python}
cd "$ROOT"

DATA="$ROOT/artifacts/experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v1"
OUT="$ROOT/artifacts/experiments/pi_jwm_v9_active_mass_raw_target_20260617"
mkdir -p "$OUT"

COMMON=(
  --dataset-dir "$DATA"
  --epochs 140
  --max-train-samples 999999 --max-val-samples 999999 --max-test-samples 999999
  --batch-size 64 --hidden-dim 64 --device cuda
  --train-seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --val-seeds 16,17 --test-seeds 18,19
  --fusion-mode cross_attention --fusion-num-heads 4
  --history-encoder mean --latent-transition-mode recurrent
  --active-rate-auxiliary --active-rate-auxiliary-weight 0.3
  --active-rate-head-mode mlp
  --rate-loss-mode active_mixed --inactive-rate-weight 0.05
  --best-metric val_link_f1_constrained_active_rate
  --best-min-f1 0.027 --best-max-link-rmse 90.0
  --best-f1-penalty-weight 1000.0 --best-link-penalty-weight 10.0
  --node-loss-weight 1.0 --activity-loss-weight 1.5 --rate-loss-weight 0.3 --task-loss-weight 1.0
  --activity-loss-mode focal --activity-pos-weight 160 --activity-focal-gamma 2.0
  --inactive-loss-sample-ratio 0.25
  --hurdle-train-gate-mode predicted
  --hurdle-train-gate-power 1.0
  --eval-hurdle-gate-power 1.0
  --high-rate-threshold 600.0
  --model-rate-output-mode hurdle_mass
  --rate-output-mode active_mass_alloc
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

run_one r0_norm_w03 \
  --seed 20260614 \
  --active-mass-loss-weight 0.3 \
  --active-mass-target-mode normalized

run_one r1_raw_w03 \
  --seed 20260614 \
  --active-mass-loss-weight 0.3 \
  --active-mass-target-mode raw

run_one r2_raw_w01 \
  --seed 20260614 \
  --active-mass-loss-weight 0.1 \
  --active-mass-target-mode raw

run_one r3_raw_w003 \
  --seed 20260614 \
  --active-mass-loss-weight 0.03 \
  --active-mass-target-mode raw

run_one r4_raw_w01_seed20260617 \
  --seed 20260617 \
  --active-mass-loss-weight 0.1 \
  --active-mass-target-mode raw

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

out = Path("/root/autodl-tmp/pi_jwm_gpu_m5/artifacts/experiments/pi_jwm_v9_active_mass_raw_target_20260617")
anchor = {
    "experiment": "c8a_hurdle_anchor",
    "seed": 20260615,
    "active_mass_target_mode": "",
    "active_mass_loss_weight": 0.0,
    "best_epoch": 25,
    "precision": 0.019693654266958426,
    "recall": 0.043478260869565216,
    "f1": 0.027108433734939756,
    "active_rate_rmse": 292.83030251003066,
    "positive_rate_active_rmse": 293.3975945890951,
    "active_mass_active_rmse": None,
    "link_rate_rmse": 89.14184698976241,
    "node_rmse": 20.20379323733833,
    "task_rmse": 3.419967911572385,
}
rows = [anchor]
names = [
    "r0_norm_w03",
    "r1_raw_w03",
    "r2_raw_w01",
    "r3_raw_w003",
    "r4_raw_w01_seed20260617",
]
for name in names:
    summary_path = out / name / "v8_full_training_summary.json"
    if not summary_path.exists():
        rows.append({"experiment": name, "error": "missing summary", "summary_path": str(summary_path)})
        continue
    summary = json.loads(summary_path.read_text())
    cfg = summary["config"]
    metrics = summary["best_test_eval"]
    row = {
        "experiment": name,
        "seed": cfg.get("seed"),
        "active_mass_target_mode": cfg.get("active_mass_target_mode"),
        "active_mass_loss_weight": cfg.get("active_mass_loss_weight"),
        "best_metric": cfg.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
        "best_metric_value": summary.get("best_metric_value"),
        "precision": metrics["activity"]["precision"],
        "recall": metrics["activity"]["recall"],
        "f1": metrics["activity"]["f1"],
        "active_rate_rmse": metrics["active_rate"]["active_rmse"],
        "positive_rate_active_rmse": metrics.get("positive_rate_active", {}).get("active_rmse"),
        "active_mass_active_rmse": metrics.get("active_mass", {}).get("active_rmse"),
        "link_rate_rmse": metrics["link_rate"]["rmse"],
        "node_rmse": metrics["node"]["rmse"],
        "task_rmse": metrics["task"]["rmse"],
        "summary_path": str(summary_path),
    }
    row["active_rate_delta_vs_c8a"] = row["active_rate_rmse"] - anchor["active_rate_rmse"]
    row["link_rate_delta_vs_c8a"] = row["link_rate_rmse"] - anchor["link_rate_rmse"]
    row["f1_delta_vs_c8a"] = row["f1"] - anchor["f1"]
    row["meets_reportable_gate"] = (
        row["active_rate_rmse"] < anchor["active_rate_rmse"]
        and row["link_rate_rmse"] <= 90.0
        and row["f1"] >= anchor["f1"]
    )
    rows.append(row)

fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with (out / "active_mass_raw_target_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

with (out / "active_mass_raw_target_comparison.md").open("w") as f:
    f.write("# PI-JWM v9 Active-Mass Raw-Target GPU Grid\n\n")
    f.write("Purpose: test whether active-mass amplitude collapse comes from supervising total mass in normalized rate space instead of raw physical rate space.\n\n")
    f.write("C8a anchor: active-rate=292.830303, F1=0.027108, link=89.141847.\n\n")
    f.write("| Experiment | seed | target mode | mass w | best epoch | P | R | F1 | Active RMSE | Pos-active RMSE | Mass-active RMSE | Link RMSE | Node RMSE | Task RMSE | d active C8a | d link C8a | d F1 C8a | reportable |\n")
    f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in rows:
        f.write("| " + " | ".join([
            fmt(row.get("experiment")),
            fmt(row.get("seed")),
            fmt(row.get("active_mass_target_mode")),
            fmt(row.get("active_mass_loss_weight")),
            fmt(row.get("best_epoch")),
            fmt(row.get("precision")),
            fmt(row.get("recall")),
            fmt(row.get("f1")),
            fmt(row.get("active_rate_rmse")),
            fmt(row.get("positive_rate_active_rmse")),
            fmt(row.get("active_mass_active_rmse")),
            fmt(row.get("link_rate_rmse")),
            fmt(row.get("node_rmse")),
            fmt(row.get("task_rmse")),
            fmt(row.get("active_rate_delta_vs_c8a")),
            fmt(row.get("link_rate_delta_vs_c8a")),
            fmt(row.get("f1_delta_vs_c8a")),
            fmt(row.get("meets_reportable_gate")),
        ]) + " |\n")
print(out / "active_mass_raw_target_comparison.csv")
PY

nvidia-smi > "$OUT/final_nvidia_smi.txt" || true
