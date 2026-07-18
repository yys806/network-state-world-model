#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

PYTHON="${PYTHON:-python}"
CACHE="${CACHE_DIR:-${CODE_ROOT}/artifacts/reports/pi_jwm_v11_selector_refinement_20260717/label_cache_schema5}"
REPORT="${REPORT_DIR:-${CODE_ROOT}/artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/selector_training}"

for split in train calibration validation; do
  test -f "${CACHE}/candidate_labels_${split}.npz"
  test -f "${CACHE}/candidate_labels_${split}.npz.manifest.json"
done

"${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the formal selector run")
print({"cuda": True, "device": torch.cuda.get_device_name(0)})
PY

mkdir -p "${REPORT}"
sha256sum \
  "${CACHE}/candidate_labels_train.npz" \
  "${CACHE}/candidate_labels_calibration.npz" \
  "${CACHE}/candidate_labels_validation.npz" \
  > "${REPORT}/input_cache_sha256.txt"
git rev-parse HEAD > "${REPORT}/source_git_sha.txt"

"${PYTHON}" scripts/train_v11_objective_aligned_selector.py \
  --train-cache "${CACHE}/candidate_labels_train.npz" \
  --calibration-cache "${CACHE}/candidate_labels_calibration.npz" \
  --validation-cache "${CACHE}/candidate_labels_validation.npz" \
  --output-dir "${REPORT}" \
  --device cuda \
  --hidden-dim 64 128 \
  --weight-cap 5 10 \
  --training-seeds 17 29 41 \
  --epochs 200 \
  --learning-rate 0.003 \
  --run-xgboost \
  --xgboost-estimators 300 \
  --run-ablations \
  2>&1 | tee "${REPORT}/formal_run.log"

"${PYTHON}" - "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
required = (
    "selector_grid_results.csv",
    "selector_comparison.csv",
    "opportunity_calibration.csv",
    "decision_trace_validation.csv",
    "gain_concentration.csv",
    "feature_and_decision_ablation.csv",
    "frozen_selector_manifest.json",
)
missing = [name for name in required if not (report / name).is_file()]
if missing:
    raise SystemExit(f"formal selector outputs missing: {missing}")
if summary.get("xgboost_status") not in {
    "completed",
    "skipped_dependency_unavailable",
}:
    raise SystemExit("XGBoost baseline did not reach a terminal status")
if summary.get("ablation_status") != "completed":
    raise SystemExit("selector ablations did not complete")
print(
    json.dumps(
        {
            "classification": summary["classification"],
            "validation_rmse": summary["selected_config"]["validation_rmse"],
            "report": str(report),
        },
        indent=2,
    )
)
PY
