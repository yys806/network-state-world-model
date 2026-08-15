#!/usr/bin/env bash
set -euo pipefail

ROOT=${PI_JWM_ROOT:-/root/autodl-tmp/pi_jwm_gpu_m5}
PY=${PYTHON_BIN:-/root/miniconda3/bin/python}
cd "$ROOT"

DATA="$ROOT/artifacts/experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v1"
OUT="$ROOT/artifacts/experiments/pi_jwm_v9_adaptive_sparse_edge_20260616"
mkdir -p "$OUT"

COMMON=(
  --dataset-dir "$DATA"
  --epochs 120
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
  --best-metric val_precision_constrained_composite
  --best-min-precision 0.01 --best-min-recall 0.05
  --node-loss-weight 1.0 --activity-loss-weight 1.5 --rate-loss-weight 0.3 --task-loss-weight 1.0
  --activity-loss-mode focal --activity-pos-weight 160 --activity-focal-gamma 2.0
  --inactive-loss-sample-ratio 0.25
  --hurdle-train-gate-mode predicted
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

run_one a0_hurdle_anchor_repeat --adaptive-edge-context none
run_one a1_sparse_edge_topk4 --adaptive-edge-context sparse_attention --adaptive-edge-topk 4
run_one a2_sparse_edge_topk8 --adaptive-edge-context sparse_attention --adaptive-edge-topk 8

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

out = Path("/root/autodl-tmp/pi_jwm_gpu_m5/artifacts/experiments/pi_jwm_v9_adaptive_sparse_edge_20260616")
anchor = {
    "experiment": "c8a_hurdle_anchor",
    "adaptive_edge_context": "none_previous",
    "adaptive_edge_topk": "",
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
for name in ["a0_hurdle_anchor_repeat", "a1_sparse_edge_topk4", "a2_sparse_edge_topk8"]:
    summary_path = out / name / "v8_full_training_summary.json"
    if not summary_path.exists():
        rows.append({"experiment": name, "error": "missing summary", "summary_path": str(summary_path)})
        continue
    summary = json.loads(summary_path.read_text())
    cfg = summary["config"]
    metrics = summary["best_test_eval"]
    row = {
        "experiment": name,
        "adaptive_edge_context": cfg.get("adaptive_edge_context"),
        "adaptive_edge_topk": cfg.get("adaptive_edge_topk"),
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
    row["positive_rate_active_delta_vs_c8a"] = row["positive_rate_active_rmse"] - anchor["positive_rate_active_rmse"]
    row["link_rate_delta_vs_c8a"] = row["link_rate_rmse"] - anchor["link_rate_rmse"]
    row["f1_delta_vs_c8a"] = row["f1"] - anchor["f1"]

fields = []
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with (out / "adaptive_sparse_edge_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

with (out / "adaptive_sparse_edge_comparison.md").open("w") as f:
    f.write("# PI-JWM v9 Adaptive Sparse Edge Context GPU Grid\n\n")
    f.write("| Experiment | context | topk | best epoch | P | R | F1 | Active-rate RMSE | Positive-rate active RMSE | Link RMSE | Node RMSE | Task RMSE | d active | d positive | d link | d F1 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write("| " + " | ".join([
            fmt(row.get("experiment")),
            fmt(row.get("adaptive_edge_context")),
            fmt(row.get("adaptive_edge_topk")),
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
        ]) + " |\n")
print(out / "adaptive_sparse_edge_comparison.csv")
PY

nvidia-smi > "$OUT/final_nvidia_smi.txt" || true
