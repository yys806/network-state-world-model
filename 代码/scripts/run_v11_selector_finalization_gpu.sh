#!/usr/bin/env bash
set -euo pipefail

# PI-JWM v11 selector finalization.  Validation is generated and selected
# before the matched-test split can be opened.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
RF_TREES="${RF_TREES:-160}"
REPORT_ROOT="${REPORT_ROOT:-artifacts/reports/pi_jwm_v11_selector_finalization_20260719}"
LABEL_DIR="${REPORT_ROOT}/label_cache_full"
TRAIN_DIR="${REPORT_ROOT}/selector_training_full"
EVAL_DIR="${REPORT_ROOT}/frozen_evaluation"
LOG_DIR="${REPORT_ROOT}/logs"
mkdir -p "${LABEL_DIR}" "${TRAIN_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

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

run_logged "01_validation_candidate_labels" \
  "${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --splits validation \
  "${COMMON_LABEL_ARGS[@]}"

"${PYTHON}" -c \
  'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["candidate_gate"]["passed"] is True, "validation candidate gate failed: {}".format(p["candidate_gate"])' \
  "${LABEL_DIR}/summary_validation.json"

run_logged "02_full_train_calibration_labels_after_gate" \
  "${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --splits train calibration \
  "${COMMON_LABEL_ARGS[@]}"

run_logged "03_validation_selected_selector_training" \
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
"${PYTHON}" -c \
  'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p.get("configuration_frozen") is True, "validation candidate gate did not pass"' \
  "${FROZEN_MANIFEST}"

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

run_logged "05_matched_test_labels_after_freeze" \
  "${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --splits matched_test \
  --frozen-config-manifest "${FROZEN_MANIFEST}" \
  "${COMMON_LABEL_ARGS[@]}"

run_logged "06_one_shot_matched_test_evaluation" \
  "${PYTHON}" scripts/evaluate_v11_frozen_selector.py \
  --frozen-manifest "${FROZEN_MANIFEST}" \
  --cache "${LABEL_DIR}/candidate_labels_matched_test.npz" \
  --output-dir "${EVAL_DIR}" \
  --device "${DEVICE}"

run_logged "07_freeze_result_bundle" \
  "${PYTHON}" scripts/finalize_v11_selector_report.py \
  --report-root "${REPORT_ROOT}" \
  --frozen-manifest "${FROZEN_MANIFEST}" \
  --matched-summary "${EVAL_DIR}/summary_matched_test.json" \
  --matched-decision-trace "${EVAL_DIR}/decision_trace_matched_test.csv"

echo "PI-JWM v11 matched-test bundle completed: ${REPORT_ROOT}"
echo "A-level finalization remains pending until external seeds 60-69 and actual AirFogSim safety evidence are attached."
