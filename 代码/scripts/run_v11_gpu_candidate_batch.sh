#!/usr/bin/env bash
set -u -o pipefail

# PI-JWM v11 candidate GPU pilot batch.
# Run from the project root or from scripts/. Outputs stay under artifacts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-artifacts/experiments/pi_jwm_v11_gpu_candidate_batch_${RUN_TAG}}"

TRAIN="${TRAIN:-512}"
VAL="${VAL:-256}"
TEST="${TEST:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STATS_CHUNK="${STATS_CHUNK:-512}"

EPOCHS="${EPOCHS:-8}"
HIDDEN="${HIDDEN:-128}"
RF_TREES="${RF_TREES:-50}"
DIFFUSION_MAX_TRAIN_ROWS="${DIFFUSION_MAX_TRAIN_ROWS:-120000}"
DIFFUSION_BATCH_ROWS="${DIFFUSION_BATCH_ROWS:-4096}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-4}"
CEM_ITERATIONS="${CEM_ITERATIONS:-3}"
CEM_SAMPLES="${CEM_SAMPLES:-32}"

mkdir -p "${OUT_ROOT}/logs"
STATUS_FILE="${OUT_ROOT}/candidate_status.tsv"
printf "candidate\tstatus\tlog\n" > "${STATUS_FILE}"

run_candidate() {
  local name="$1"
  shift
  local log_file="${OUT_ROOT}/logs/${name}.log"
  echo
  echo "===== ${name} ====="
  echo "Log: ${log_file}"
  printf "%q " "$@" | tee "${log_file}"
  echo | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
  local status="${PIPESTATUS[0]}"
  if [[ "${status}" -eq 0 ]]; then
    printf "%s\tPASS\t%s\n" "${name}" "${log_file}" >> "${STATUS_FILE}"
  else
    printf "%s\tFAIL_%s\t%s\n" "${name}" "${status}" "${log_file}" >> "${STATUS_FILE}"
  fi
  return "${status}"
}

FAILED=0

COMMON_ARGS=(
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --max-train-samples "${TRAIN}"
  --max-val-samples "${VAL}"
  --max-test-samples "${TEST}"
  --limit-after-stats
  --streaming-stats
  --stats-chunk-size "${STATS_CHUNK}"
)

run_candidate "01_ranked_constrained_allocation_anchor" \
  "${PYTHON}" scripts/diagnose_v11_scheduler_ranked_allocation.py \
  --output-dir "${OUT_ROOT}/01_ranked_constrained_allocation_anchor" \
  "${COMMON_ARGS[@]}" \
  --rf-trees "${RF_TREES}" \
  --top-k 16 32 \
  --blend-alpha 0.95 1.0 \
  --step-total-cap-scale 1.1 1.15 \
  --edge-value-cap-scale 1.15 1.25 \
  --risk-weight 0.05 || FAILED=1

run_candidate "02_diffusion_support_mask_generator" \
  "${PYTHON}" scripts/compare_v11_diffusion_support_generator.py \
  --output-dir "${OUT_ROOT}/02_diffusion_support_mask_generator" \
  "${COMMON_ARGS[@]}" \
  --rank-target-mode gain_norm \
  --target-top-k 4 \
  --diffusion-epochs "${EPOCHS}" \
  --diffusion-hidden-dim "${HIDDEN}" \
  --diffusion-batch-rows "${DIFFUSION_BATCH_ROWS}" \
  --diffusion-max-train-rows "${DIFFUSION_MAX_TRAIN_ROWS}" \
  --diffusion-sampling-steps "${DIFFUSION_STEPS}" \
  --top-k 4 8 16 \
  --selection-score-modes diffusion diffusion_value \
  --selection-group-modes baseline_active_count \
  --blend-alpha 1.0 \
  --step-total-cap-scale 1.05 1.1 \
  --edge-value-cap-scale 1.15 \
  --new-edge-value-cap 2.0 || FAILED=1

run_candidate "03_gnn_guided_greedy_scheduler" \
  "${PYTHON}" scripts/compare_v11_gnn_greedy_scheduler.py \
  --output-dir "${OUT_ROOT}/03_gnn_guided_greedy_scheduler" \
  "${COMMON_ARGS[@]}" \
  --rank-target-mode gain_norm \
  --target-top-k 4 \
  --gnn-epochs "${EPOCHS}" \
  --gnn-hidden-dim "${HIDDEN}" \
  --gnn-layers 2 \
  --gnn-batch-groups 64 \
  --top-k 4 8 16 \
  --selection-score-modes gnn gnn_value \
  --selection-group-modes baseline_active_count \
  --blend-alpha 1.0 \
  --step-total-cap-scale 1.05 1.1 \
  --edge-value-cap-scale 1.15 \
  --new-edge-value-cap 2.0 || FAILED=1

run_candidate "04_pointer_support_generator" \
  "${PYTHON}" scripts/compare_v11_pointer_support_generator.py \
  --output-dir "${OUT_ROOT}/04_pointer_support_generator" \
  "${COMMON_ARGS[@]}" \
  --rank-target-mode gain_norm \
  --target-top-k 4 \
  --pointer-epochs "${EPOCHS}" \
  --pointer-hidden-dim "${HIDDEN}" \
  --pointer-batch-groups 64 \
  --top-k 4 8 16 \
  --selection-score-modes pointer pointer_value \
  --selection-group-modes baseline_active_count \
  --blend-alpha 1.0 \
  --step-total-cap-scale 1.05 1.1 \
  --edge-value-cap-scale 1.15 \
  --new-edge-value-cap 2.0 || FAILED=1

run_candidate "05_cem_planner_refiner" \
  "${PYTHON}" scripts/compare_v11_cem_planner_refiner.py \
  --output-dir "${OUT_ROOT}/05_cem_planner_refiner" \
  "${COMMON_ARGS[@]}" \
  --cem-iterations "${CEM_ITERATIONS}" \
  --cem-samples-per-group "${CEM_SAMPLES}" \
  --cem-elite-frac 0.25 \
  --cem-noise-std 0.7 \
  --cem-momentum 0.7 \
  --top-k 4 8 16 \
  --selection-score-modes cem cem_value \
  --selection-group-modes baseline_active_count \
  --blend-alpha 1.0 \
  --step-total-cap-scale 1.05 1.1 \
  --edge-value-cap-scale 1.15 \
  --new-edge-value-cap 2.0 || FAILED=1

echo
echo "Batch output root: ${OUT_ROOT}"
echo "Candidate status:"
cat "${STATUS_FILE}"

if "${PYTHON}" scripts/summarize_v11_gpu_candidate_batch.py --batch-dir "${OUT_ROOT}"; then
  echo "Metric summary written under ${OUT_ROOT}"
else
  echo "Metric summary failed" >&2
  FAILED=1
fi

exit "${FAILED}"
