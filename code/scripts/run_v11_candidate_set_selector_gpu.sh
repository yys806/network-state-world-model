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
CACHE_DIR="${CACHE_DIR:-artifacts/reports/pi_jwm_v11_physical_benefit_bridge_20260719/formal_h10/task_bridge}"
BRIDGE_MANIFEST="${BRIDGE_MANIFEST:-${CACHE_DIR}/summary.json}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/pi_jwm_v11_selector_finalization_20260719/selector_training_task_bridge_h10}"
EPOCHS="${EPOCHS:-20}"

if [[ "${DEVICE}" != "cuda" ]]; then
  echo "formal CandidateSet selector training requires DEVICE=cuda" >&2
  exit 2
fi
for split in train calibration validation; do
  test -f "${CACHE_DIR}/candidate_labels_${split}_physical.npz"
  test -f "${CACHE_DIR}/candidate_labels_${split}_physical.npz.manifest.json"
done
test -f "${BRIDGE_MANIFEST}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked source tree must be clean before selector training" >&2
  exit 2
fi

"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print({"device": torch.cuda.get_device_name(0), "torch": torch.__version__})'
mkdir -p "${REPORT_DIR}"
sha256sum \
  "${CACHE_DIR}/candidate_labels_train_physical.npz" \
  "${CACHE_DIR}/candidate_labels_calibration_physical.npz" \
  "${CACHE_DIR}/candidate_labels_validation_physical.npz" \
  > "${REPORT_DIR}/input_cache_sha256.txt"
git rev-parse HEAD > "${REPORT_DIR}/source_git_sha.txt"

"${PYTHON}" scripts/train_v11_candidate_set_selector.py \
  --train-cache "${CACHE_DIR}/candidate_labels_train_physical.npz" \
  --calibration-cache "${CACHE_DIR}/candidate_labels_calibration_physical.npz" \
  --validation-cache "${CACHE_DIR}/candidate_labels_validation_physical.npz" \
  --physical-bridge-manifest "${BRIDGE_MANIFEST}" \
  --output-dir "${REPORT_DIR}" \
  --device "${DEVICE}" \
  --hidden-dim 64 128 \
  --temperature 0.1 0.25 0.5 \
  --dropout 0 0.1 \
  --training-seeds 17 29 41 \
  --epochs "${EPOCHS}" \
  2>&1 | tee "${REPORT_DIR}/formal_run.log"

"${PYTHON}" - "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
required = ("selector_grid_results.csv", "selector_comparison.csv", "frozen_selector_manifest.json")
missing = [name for name in required if not (report / name).is_file()]
if missing:
    raise SystemExit(f"selector outputs missing: {missing}")
print(json.dumps({
    "configuration_frozen": bool(summary.get("configuration_frozen")),
    "validation_rmse": summary.get("selected_config", {}).get("validation_rmse"),
    "report": str(report),
}, ensure_ascii=False, indent=2))
PY
