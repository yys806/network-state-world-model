#!/usr/bin/env bash
set -euo pipefail

ROOT=${PI_JWM_ROOT:-/root/autodl-tmp/pi_jwm_gpu_m5}
PY=${PYTHON_BIN:-/root/miniconda3/bin/python}
cd "$ROOT"

DATA="$ROOT/artifacts/experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v1"
OUT="$ROOT/artifacts/experiments/pi_jwm_v9_c9_dual_head_20260617"
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
  --best-metric val_link_f1_constrained_composite
  --best-min-f1 0.027 --best-max-link-rmse 90.0
  --best-f1-penalty-weight 1000.0 --best-link-penalty-weight 10.0
  --node-loss-weight 1.0 --activity-loss-weight 1.5 --rate-loss-weight 0.3 --task-loss-weight 1.0
  --activity-loss-mode focal --activity-pos-weight 160 --activity-focal-gamma 2.0
  --inactive-loss-sample-ratio 0.25
  --hurdle-train-gate-power 1.0
  --eval-hurdle-gate-power 1.0
  --positive-rate-specialist-weight 0.3
  --positive-rate-target-mode log1p
  --positive-rate-loss-mode huber
  --high-rate-threshold 600.0
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

run_one c9a_hurdle_soft_anchor_repeat \
  --seed 20260617 \
  --model-rate-output-mode hurdle_soft \
  --rate-output-mode main \
  --hurdle-train-gate-mode predicted \
  --positive-rate-specialist-weight 0.0 \
  --high-rate-weight 1.0

run_one c9b_dual_main_conservative \
  --seed 20260617 \
  --model-rate-output-mode hurdle_dual \
  --rate-output-mode main \
  --hurdle-train-gate-mode none \
  --high-rate-weight 1.0

run_one c9c_dual_gated_eval \
  --seed 20260617 \
  --model-rate-output-mode hurdle_dual \
  --rate-output-mode hurdle_gate \
  --hurdle-train-gate-mode none \
  --high-rate-weight 1.0

run_one c9d_dual_blend_eval \
  --seed 20260617 \
  --model-rate-output-mode hurdle_dual \
  --rate-output-mode dual_soft_blend \
  --hurdle-train-gate-mode none \
  --high-rate-weight 1.0

run_one c9e_dual_gated_highrate \
  --seed 20260619 \
  --model-rate-output-mode hurdle_dual \
  --rate-output-mode hurdle_gate \
  --hurdle-train-gate-mode none \
  --high-rate-weight 2.0

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

out = Path("/root/autodl-tmp/pi_jwm_gpu_m5/artifacts/experiments/pi_jwm_v9_c9_dual_head_20260617")
anchor = {
    "experiment": "c8a_hurdle_anchor",
    "seed": 20260615,
    "model_rate_output_mode": "hurdle_soft",
    "rate_output_mode": "main",
    "best_epoch": 25,
    "precision": 0.019693654266958426,
    "recall": 0.043478260869565216,
    "f1": 0.027108433734939756,
    "active_rate_rmse": 292.83030251003066,
    "positive_rate_active_rmse": 293.3975945890951,
    "link_rate_rmse": 89.14184698976241,
    "node_rmse": 20.20379323733833,
    "task_rmse": 3.419967911572385,
}
rows = [anchor]
names = [
    "c9a_hurdle_soft_anchor_repeat",
    "c9b_dual_main_conservative",
    "c9c_dual_gated_eval",
    "c9d_dual_blend_eval",
    "c9e_dual_gated_highrate",
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
        "model_rate_output_mode": cfg.get("model_rate_output_mode"),
        "rate_output_mode": cfg.get("rate_output_mode"),
        "hurdle_train_gate_mode": cfg.get("hurdle_train_gate_mode"),
        "positive_rate_specialist_weight": cfg.get("positive_rate_specialist_weight"),
        "high_rate_weight": cfg.get("high_rate_weight"),
        "best_metric": cfg.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
        "best_metric_value": summary.get("best_metric_value"),
        "precision": metrics["activity"]["precision"],
        "recall": metrics["activity"]["recall"],
        "f1": metrics["activity"]["f1"],
        "active_rate_rmse": metrics["active_rate"]["active_rmse"],
        "positive_rate_active_rmse": metrics.get("positive_rate_active", {}).get("active_rmse"),
        "link_rate_rmse": metrics["link_rate"]["rmse"],
        "node_rmse": metrics["node"]["rmse"],
        "task_rmse": metrics["task"]["rmse"],
        "summary_path": str(summary_path),
    }
    rows.append(row)

for row in rows[1:]:
    if "error" in row:
        continue
    row["active_rate_delta_vs_c8a"] = row["active_rate_rmse"] - anchor["active_rate_rmse"]
    row["positive_rate_active_delta_vs_c8a"] = (
        row["positive_rate_active_rmse"] - anchor["positive_rate_active_rmse"]
        if row["positive_rate_active_rmse"] is not None else None
    )
    row["link_rate_delta_vs_c8a"] = row["link_rate_rmse"] - anchor["link_rate_rmse"]
    row["f1_delta_vs_c8a"] = row["f1"] - anchor["f1"]
    row["meets_reportable_gate"] = (
        row["active_rate_rmse"] < anchor["active_rate_rmse"]
        and row["link_rate_rmse"] <= 90.0
        and row["f1"] >= anchor["f1"]
    )

fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with (out / "c9_dual_head_comparison.csv").open("w", newline="") as f:
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

with (out / "c9_dual_head_comparison.md").open("w") as f:
    f.write("# PI-JWM v9 C9 Dual-Head GPU Grid\n\n")
    f.write("Reportable gate: active-rate < C8a, link RMSE <= 90, F1 >= C8a.\n\n")
    f.write("| Experiment | seed | model mode | eval rate | train gate | pos w | high w | best epoch | P | R | F1 | Active-rate RMSE | Positive-rate active RMSE | Link RMSE | Node RMSE | Task RMSE | d active | d positive | d link | d F1 | reportable |\n")
    f.write("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in rows:
        f.write("| " + " | ".join([
            fmt(row.get("experiment")),
            fmt(row.get("seed")),
            fmt(row.get("model_rate_output_mode")),
            fmt(row.get("rate_output_mode")),
            fmt(row.get("hurdle_train_gate_mode")),
            fmt(row.get("positive_rate_specialist_weight")),
            fmt(row.get("high_rate_weight")),
            fmt(row.get("best_epoch")),
            fmt(row.get("precision")),
            fmt(row.get("recall")),
            fmt(row.get("f1")),
            fmt(row.get("active_rate_rmse")),
            fmt(row.get("positive_rate_active_rmse")),
            fmt(row.get("link_rate_rmse")),
            fmt(row.get("node_rmse")),
            fmt(row.get("task_rmse")),
            fmt(row.get("active_rate_delta_vs_c8a")),
            fmt(row.get("positive_rate_active_delta_vs_c8a")),
            fmt(row.get("link_rate_delta_vs_c8a")),
            fmt(row.get("f1_delta_vs_c8a")),
            fmt(row.get("meets_reportable_gate")),
        ]) + " |\n")
print(out / "c9_dual_head_comparison.csv")
PY

nvidia-smi > "$OUT/final_nvidia_smi.txt" || true
