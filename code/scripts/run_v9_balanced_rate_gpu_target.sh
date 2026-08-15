#!/usr/bin/env bash
set -euo pipefail

ROOT=${PI_JWM_ROOT:-/root/autodl-tmp/pi_jwm_gpu_m5}
PY=${PYTHON_BIN:-/root/miniconda3/bin/python}
RATE_MODE=${RATE_MODE:-bmc}
cd "$ROOT"

DATA="$ROOT/artifacts/experiments/airfogsim_v0/datasets/world_model_dataset_active_heavy_v1"
OUT="$ROOT/artifacts/experiments/pi_jwm_v9_balanced_rate_gpu_20260619"
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
  --metric-checkpoints val_active_rate_rmse,val_link_rate_rmse,val_activity_f1,val_link_f1_constrained_active_rate
  --node-loss-weight 1.0 --activity-loss-weight 1.5 --rate-loss-weight 0.3 --task-loss-weight 1.0
  --activity-loss-mode focal --activity-pos-weight 160 --activity-focal-gamma 2.0
  --inactive-loss-sample-ratio 0.25
  --hurdle-train-gate-mode predicted --hurdle-train-gate-power 1.0
  --eval-hurdle-gate-power 1.0
)

case "$RATE_MODE" in
  bmc)
    NAME="bmc_sigma5"
    RATE_ARGS=(--active-rate-reweight-mode bmc --bmc-noise-sigma 5.0 --bmc-minimum-count 3)
    ;;
  lds)
    NAME="lds_mild_clip3"
    RATE_ARGS=(
      --active-rate-reweight-mode lds
      --lds-bin-width 50 --lds-kernel-size 5 --lds-sigma 2
      --lds-weight-min 0.5 --lds-weight-max 3.0 --lds-tail-quantile 0.995
    )
    ;;
  *)
    echo "RATE_MODE must be bmc or lds" >&2
    exit 2
    ;;
esac

echo "===== START ${NAME} $(date '+%F %T %z') =====" | tee -a "$OUT/queue.log"
"$PY" scripts/run_world_model_v8_full_training.py \
  "${COMMON[@]}" \
  "${RATE_ARGS[@]}" \
  --output-dir "$OUT/$NAME" \
  2>&1 | tee "$OUT/${NAME}.log"
echo "===== END ${NAME} $(date '+%F %T %z') =====" | tee -a "$OUT/queue.log"

"$PY" - "$OUT/$NAME/v8_full_training_summary.json" "$OUT/${NAME}_vs_c8a.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text())
metrics = summary["best_test_eval"]
anchor = {
    "active_rate_rmse": 292.83030251003066,
    "f1": 0.027108433734939756,
    "link_rate_rmse": 89.14184698976241,
}
result = {
    "experiment": summary_path.parent.name,
    "best_epoch": summary["best_epoch"],
    "active_rate_rmse": metrics["active_rate"]["active_rmse"],
    "f1": metrics["activity"]["f1"],
    "precision": metrics["activity"]["precision"],
    "recall": metrics["activity"]["recall"],
    "link_rate_rmse": metrics["link_rate"]["rmse"],
    "positive_rate_active_rmse": metrics.get("positive_rate_active", {}).get("active_rmse"),
    "anchor": anchor,
}
result["delta_active_rate"] = result["active_rate_rmse"] - anchor["active_rate_rmse"]
result["delta_f1"] = result["f1"] - anchor["f1"]
result["delta_link_rate"] = result["link_rate_rmse"] - anchor["link_rate_rmse"]
result["strict_promote"] = (
    result["active_rate_rmse"] < anchor["active_rate_rmse"]
    and result["f1"] >= anchor["f1"]
    and result["link_rate_rmse"] <= 90.0
)
output_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PY

nvidia-smi > "$OUT/${NAME}_final_nvidia_smi.txt" || true
