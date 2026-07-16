#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${PI_JWM_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${PI_JWM_MKL_NUM_THREADS:-8}"

# Validation-only refinement after the original matched-test split was consumed.
# This script deliberately contains no locked evaluation split.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-200}"
RF_TREES="${RF_TREES:-160}"
REPORT_ROOT="${REPORT_ROOT:-artifacts/reports/pi_jwm_v11_selector_refinement_20260717}"
LABEL_DIR="${REPORT_ROOT}/label_cache_schema5"
TRAIN_DIR="${REPORT_ROOT}/selector_training"
LOG_DIR="${REPORT_ROOT}/logs"
mkdir -p "${LABEL_DIR}" "${TRAIN_DIR}" "${LOG_DIR}"

run_logged() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

COMMON_LABEL_ARGS=(
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --helper-train-limit 0
  --split-sample-limit 0
  --rf-trees "${RF_TREES}"
  --stats-chunk-size 512
  --output-dir "${LABEL_DIR}"
)

run_logged "01_validation_schema5_labels" \
  "${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --splits validation \
  "${COMMON_LABEL_ARGS[@]}"

"${PYTHON}" -c \
  'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["candidate_gate"]["passed"] is True, "validation candidate gate failed: {}".format(p["candidate_gate"])' \
  "${LABEL_DIR}/summary_validation.json"

run_logged "02_train_calibration_schema5_labels" \
  "${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --splits train calibration \
  "${COMMON_LABEL_ARGS[@]}"

run_logged "03_validation_selector_refinement" \
  "${PYTHON}" scripts/train_v11_candidate_set_selector.py \
  --train-cache "${LABEL_DIR}/candidate_labels_train.npz" \
  --calibration-cache "${LABEL_DIR}/candidate_labels_calibration.npz" \
  --validation-cache "${LABEL_DIR}/candidate_labels_validation.npz" \
  --output-dir "${TRAIN_DIR}" \
  --device "${DEVICE}" \
  --hidden-dim 64 128 \
  --temperature 0.1 0.25 0.5 \
  --dropout 0 0.1 \
  --training-seeds 17 29 41 \
  --epochs "${EPOCHS}"

FROZEN_MANIFEST="${TRAIN_DIR}/frozen_selector_manifest.json"
run_logged "04_validation_feature_group_ablation" \
  "${PYTHON}" scripts/run_v11_selector_feature_ablation.py \
  --train-cache "${LABEL_DIR}/candidate_labels_train.npz" \
  --calibration-cache "${LABEL_DIR}/candidate_labels_calibration.npz" \
  --validation-cache "${LABEL_DIR}/candidate_labels_validation.npz" \
  --frozen-manifest "${FROZEN_MANIFEST}" \
  --output-dir "${REPORT_ROOT}/selector_ablation" \
  --device "${DEVICE}" \
  --training-seeds 17 29 41 \
  --epochs "${EPOCHS}"

echo "PI-JWM v11 selector refinement completed without opening a locked evaluation split: ${REPORT_ROOT}"
