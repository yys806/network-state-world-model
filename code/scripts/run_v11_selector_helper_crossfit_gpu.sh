#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${PI_JWM_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${PI_JWM_MKL_NUM_THREADS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RF_TREES="${RF_TREES:-160}"
REPORT_ROOT="${REPORT_ROOT:-artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719}"
LABEL_ROOT="${REPORT_ROOT}/label_cache_schema6"
FOLD_ROOT="${LABEL_ROOT}/folds"
EVAL_DIR="${LABEL_ROOT}/eval"
MERGED_DIR="${LABEL_ROOT}/merged"
LOG_DIR="${REPORT_ROOT}/logs"
SAMPLE_INDEX_CSV="${SAMPLE_INDEX_CSV:-artifacts/experiments/airfogsim_v0/datasets/dataset_multiseed_active_heavy_v2_60seed_20260619/sample_index.csv}"
WORLD_CHECKPOINT="${WORLD_CHECKPOINT:-artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt}"
mkdir -p "${FOLD_ROOT}" "${EVAL_DIR}" "${MERGED_DIR}" "${LOG_DIR}"

if [[ "${DEVICE}" != "cuda" ]]; then
  echo "formal selector crossfit labels require DEVICE=cuda" >&2
  exit 2
fi
if [[ ! -f "${WORLD_CHECKPOINT}" || ! -f "${POLICY_CHECKPOINT}" || ! -f "${SAMPLE_INDEX_CSV}" ]]; then
  echo "required checkpoint or sample index is missing" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked source tree must be clean before formal generation" >&2
  exit 2
fi

SOURCE_GIT_SHA="$(git rev-parse HEAD)"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

for FOLD in 0 1 2 3 4; do
  FOLD_DIR="${FOLD_ROOT}/fold_${FOLD}"
  mkdir -p "${FOLD_DIR}"
  "${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
    --world-checkpoint "${WORLD_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --splits train \
    --helper-protocol seed_crossfit_5fold \
    --crossfit-fold "${FOLD}" \
    --cache-schema-version 6 \
    --helper-train-limit 0 \
    --split-sample-limit 0 \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --rf-trees "${RF_TREES}" \
    --stats-chunk-size 512 \
    --output-dir "${FOLD_DIR}" \
    2>&1 | tee "${LOG_DIR}/fold_${FOLD}.log"
done

"${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --world-checkpoint "${WORLD_CHECKPOINT}" \
  --policy-checkpoint "${POLICY_CHECKPOINT}" \
  --splits calibration validation \
  --helper-protocol seed_crossfit_5fold \
  --cache-schema-version 6 \
  --helper-train-limit 0 \
  --split-sample-limit 0 \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --rf-trees "${RF_TREES}" \
  --stats-chunk-size 512 \
  --output-dir "${EVAL_DIR}" \
  2>&1 | tee "${LOG_DIR}/evaluation.log"

CONFIGURATION_DIGEST="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["configuration_digest"])' "${FOLD_ROOT}/fold_0/candidate_label_run_summary.json")"
CROSSFIT_PROTOCOL_DIGEST="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["crossfit_protocol_digest"])' "${FOLD_ROOT}/fold_0/crossfit_protocol.json")"

"${PYTHON}" scripts/merge_v11_selector_crossfit_labels.py \
  --fold-cache "${FOLD_ROOT}/fold_0/candidate_labels_train.npz" \
  --fold-cache "${FOLD_ROOT}/fold_1/candidate_labels_train.npz" \
  --fold-cache "${FOLD_ROOT}/fold_2/candidate_labels_train.npz" \
  --fold-cache "${FOLD_ROOT}/fold_3/candidate_labels_train.npz" \
  --fold-cache "${FOLD_ROOT}/fold_4/candidate_labels_train.npz" \
  --output-cache "${MERGED_DIR}/candidate_labels_train.npz" \
  --sample-index-csv "${SAMPLE_INDEX_CSV}" \
  --sample-limit-per-fold 0 \
  --expected-configuration-digest "${CONFIGURATION_DIGEST}" \
  --expected-crossfit-protocol-digest "${CROSSFIT_PROTOCOL_DIGEST}" \
  2>&1 | tee "${LOG_DIR}/merge.log"

"${PYTHON}" - "${LABEL_ROOT}" "${SOURCE_GIT_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from pi_jwm.v11_labeling import (
    load_candidate_interaction_cache,
    load_candidate_label_metadata,
)

label_root = Path(sys.argv[1])
source_git_sha = sys.argv[2]
paths = {
    "train": label_root / "merged/candidate_labels_train.npz",
    "calibration": label_root / "eval/candidate_labels_calibration.npz",
    "validation": label_root / "eval/candidate_labels_validation.npz",
}
expected_counts = {"train": 15600, "calibration": 2340, "validation": 3900}
expected_seeds = {
    "train": set(range(16)) | set(range(20, 44)),
    "calibration": set(range(44, 50)),
    "validation": set(range(50, 60)),
}
manifests = {}
metadata = {}
for split, path in paths.items():
    _, _, _, manifest = load_candidate_interaction_cache(path)
    meta = load_candidate_label_metadata(path)
    if int(manifest["schema_version"]) != 6:
        raise RuntimeError(f"{split}: schema mismatch")
    if int(manifest["interaction"]["overflow_count"]) != 0:
        raise RuntimeError(f"{split}: token overflow")
    if int(manifest["num_samples"]) != expected_counts[split]:
        raise RuntimeError(f"{split}: sample count mismatch")
    if set(int(value) for value in meta["sample_seed"]) != expected_seeds[split]:
        raise RuntimeError(f"{split}: seed coverage mismatch")
    if manifest["protocol_metadata"].get("source_git_sha") != source_git_sha:
        raise RuntimeError(f"{split}: source Git SHA mismatch")
    manifests[split] = manifest
    metadata[split] = meta
if set(int(value) for value in metadata["train"]["sample_fold_id"]) != set(range(5)):
    raise RuntimeError("train fold provenance mismatch")
configuration_digests = {m["configuration_digest"] for m in manifests.values()}
crossfit_digests = {
    m["protocol_metadata"]["crossfit_protocol_digest"] for m in manifests.values()
}
if len(configuration_digests) != 1 or len(crossfit_digests) != 1:
    raise RuntimeError("cross-split protocol digest mismatch")
summary = {
    "framework": "PI-JWM",
    "result_kind": "diagnostic_only",
    "source_git_sha": source_git_sha,
    "sample_count": expected_counts,
    "train_fold_ids": [0, 1, 2, 3, 4],
    "configuration_digest_count": len(configuration_digests),
    "crossfit_protocol_digest_count": len(crossfit_digests),
    "configuration_digest": next(iter(configuration_digests)),
    "crossfit_protocol_digest": next(iter(crossfit_digests)),
    "overflow_count": {name: 0 for name in paths},
    "locked_split_accessed": False,
    "cache_sha256": {name: manifest["cache_sha256"] for name, manifest in manifests.items()},
}
(label_root / "gpu_crossfit_handoff_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "PI-JWM selector helper crossfit labels verified: ${LABEL_ROOT}"
