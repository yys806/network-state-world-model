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
REPORT_ROOT="${REPORT_ROOT:-artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718}"
LABEL_DIR="${REPORT_ROOT}/label_cache_schema6"
LOG_DIR="${REPORT_ROOT}/logs"
WORLD_CHECKPOINT="${WORLD_CHECKPOINT:-artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt}"
mkdir -p "${LABEL_DIR}" "${LOG_DIR}"

if [[ "${DEVICE}" != "cuda" ]]; then
  echo "schema-v6 full-label handoff requires DEVICE=cuda" >&2
  exit 2
fi
if [[ ! -f "${WORLD_CHECKPOINT}" || ! -f "${POLICY_CHECKPOINT}" ]]; then
  echo "required frozen checkpoint is missing" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked source tree must be clean before full-label generation" >&2
  exit 2
fi

SOURCE_GIT_SHA="$(git rev-parse HEAD)"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
"${PYTHON}" -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.__version__)'

echo "[$(date --iso-8601=seconds)] generating complete schema-v6 label caches"
"${PYTHON}" scripts/run_v11_selector_candidate_labels.py \
  --world-checkpoint "${WORLD_CHECKPOINT}" \
  --policy-checkpoint "${POLICY_CHECKPOINT}" \
  --output-dir "${LABEL_DIR}" \
  --splits train calibration validation \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --helper-train-limit 0 \
  --split-sample-limit 0 \
  --rf-trees "${RF_TREES}" \
  --stats-chunk-size 512 \
  --cache-schema-version 6 \
  2>&1 | tee "${LOG_DIR}/01_complete_schema6_labels.log"

"${PYTHON}" - "${LABEL_DIR}" "${SOURCE_GIT_SHA}" <<'PY'
import json
import sys
from pathlib import Path

from pi_jwm.v11_labeling import load_candidate_interaction_cache

label_dir = Path(sys.argv[1])
source_git_sha = sys.argv[2]
expected_seeds = {
    "train": set(range(16)) | set(range(20, 44)),
    "calibration": set(range(44, 50)),
    "validation": set(range(50, 60)),
}
manifests = {}
for split, seeds in expected_seeds.items():
    cache = label_dir / f"candidate_labels_{split}.npz"
    _, _, interactions, manifest = load_candidate_interaction_cache(cache)
    if int(manifest["schema_version"]) != 6:
        raise RuntimeError(f"{split}: expected schema 6")
    if set(int(value) for value in manifest["seed_values"]) != seeds:
        raise RuntimeError(f"{split}: seed protocol mismatch")
    if int(manifest["interaction"]["overflow_count"]) != 0:
        raise RuntimeError(f"{split}: token overflow detected")
    if int(manifest["interaction"]["token_count_max"]) > int(
        manifest["interaction"]["token_capacity"]
    ):
        raise RuntimeError(f"{split}: token capacity violated")
    if manifest["protocol_metadata"].get("source_git_sha") != source_git_sha:
        raise RuntimeError(f"{split}: source SHA mismatch")
    manifests[split] = manifest

digests = {manifest["configuration_digest"] for manifest in manifests.values()}
if len(digests) != 1:
    raise RuntimeError("split configuration digests differ")
protocols = {
    json.dumps(
        {
            key: manifest["interaction"][key]
            for key in (
                "token_capacity",
                "token_dimension",
                "pooled_dimension",
                "token_feature_names",
                "pooled_feature_names",
                "action_feature_names",
            )
        },
        sort_keys=True,
    )
    for manifest in manifests.values()
}
if len(protocols) != 1:
    raise RuntimeError("split interaction protocols differ")

summary = {
    "framework": "PI-JWM",
    "result_kind": "diagnostic_only",
    "cache_schema_version": 6,
    "source_git_sha": source_git_sha,
    "configuration_digest": next(iter(digests)),
    "split_cache_sha256": {
        split: manifest["cache_sha256"] for split, manifest in manifests.items()
    },
    "token_count_max": {
        split: int(manifest["interaction"]["token_count_max"])
        for split, manifest in manifests.items()
    },
    "overflow_count": {
        split: int(manifest["interaction"]["overflow_count"])
        for split, manifest in manifests.items()
    },
}
(label_dir / "gpu_label_handoff_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "PI-JWM schema-v6 complete unlocked label caches verified: ${LABEL_DIR}"
