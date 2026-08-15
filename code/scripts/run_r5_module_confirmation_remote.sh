#!/usr/bin/env bash
set -uo pipefail

BASE="${PIJWM_CONFIRMATION_BASE:-/root/autodl-tmp/pi_jwm_r5_confirmation}"
PYTHON="${PIJWM_PYTHON:-/root/miniconda3/bin/python}"
OUTPUT="${BASE}/artifacts/pi_jwm_r5_module_confirmation_v1"
SMOKE_OUTPUT="${BASE}/artifacts/pi_jwm_r5_module_confirmation_smoke_v1"
SMOKE_GATE="${BASE}/artifacts/r5_module_confirmation_smoke_gate.json"
STATUS="${BASE}/artifacts/r5_module_confirmation_launcher_status.json"
LOCK="${BASE}/artifacts/r5_module_confirmation_launcher.lock"
LOG="${BASE}/artifacts/r5_module_confirmation_training.log"
MAX_ATTEMPTS="${PIJWM_MAX_ATTEMPTS:-3}"

mkdir -p "${BASE}/artifacts"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another confirmation launcher already owns ${LOCK}" >&2
  exit 3
fi

smoke_resume_args=()
if [[ -f "${SMOKE_OUTPUT}/training_protocol.json" ]]; then
  smoke_resume_args+=(--resume)
fi
cd "${BASE}/code/scripts"
set +e
PYTHONPATH="${BASE}/code/src:${BASE}/code/scripts" \
  "${PYTHON}" -u run_r5_module_confirmation_training.py \
    --dataset-root "${BASE}/data" \
    --evaluation-root "${BASE}/evaluation" \
    --r4-screening-root "${BASE}/r4_screening" \
    --existing-r5-root "${BASE}/r5_existing" \
    --output-dir "${SMOKE_OUTPUT}" \
    --device cuda:0 \
    --hidden-dim 16 \
    --micro-batch-size 4 \
    --combination F \
    --seed 20260803 \
    --smoke-epochs 1 \
    "${smoke_resume_args[@]}" 2>&1 | tee -a "${LOG}"
smoke_exit_code=${PIPESTATUS[0]}
if [[ ${smoke_exit_code} -eq 0 ]]; then
  "${PYTHON}" verify_r5_module_confirmation_bundle.py --smoke "${SMOKE_OUTPUT}" > "${SMOKE_GATE}.tmp"
  smoke_exit_code=$?
  if [[ ${smoke_exit_code} -eq 0 ]]; then
    mv "${SMOKE_GATE}.tmp" "${SMOKE_GATE}"
  fi
fi
set -e
if [[ ${smoke_exit_code} -ne 0 ]]; then
  rm -f "${SMOKE_GATE}.tmp"
  "${PYTHON}" - "${STATUS}" "${smoke_exit_code}" "${LOG}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "state": "smoke_failed",
    "training_complete": False,
    "child_exit_code": int(sys.argv[2]),
    "log_path": sys.argv[3],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  exit "${smoke_exit_code}"
fi

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  resume_args=()
  if [[ -f "${OUTPUT}/training_protocol.json" ]]; then
    resume_args+=(--resume)
  fi

  "${PYTHON}" - "${STATUS}" "${attempt}" "${LOG}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "attempt": int(sys.argv[2]),
    "state": "running",
    "training_complete": False,
    "log_path": sys.argv[3],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

  cd "${BASE}/code/scripts"
  set +e
  PYTHONPATH="${BASE}/code/src:${BASE}/code/scripts" \
    "${PYTHON}" -u run_r5_module_confirmation_training.py \
      --dataset-root "${BASE}/data" \
      --evaluation-root "${BASE}/evaluation" \
      --r4-screening-root "${BASE}/r4_screening" \
      --existing-r5-root "${BASE}/r5_existing" \
      --output-dir "${OUTPUT}" \
      --device cuda:0 \
      --hidden-dim 16 \
      --micro-batch-size 4 \
      "${resume_args[@]}" 2>&1 | tee -a "${LOG}"
  exit_code=${PIPESTATUS[0]}
  set -e

  complete=$("${PYTHON}" - "${OUTPUT}/training_summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    print("false")
else:
    payload = json.loads(path.read_text())
    print(str(bool(payload.get("r5_module_confirmation_complete"))).lower())
PY
)

  "${PYTHON}" - "${STATUS}" "${attempt}" "${exit_code}" "${complete}" "${LOG}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "attempt": int(sys.argv[2]),
    "child_exit_code": int(sys.argv[3]),
    "training_complete": sys.argv[4] == "true",
    "state": "complete" if sys.argv[4] == "true" else "retry_pending",
    "log_path": sys.argv[5],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

  if [[ "${complete}" == "true" ]]; then
    "${PYTHON}" verify_r5_module_confirmation_bundle.py "${OUTPUT}"
    exit 0
  fi
  sleep 30
done

"${PYTHON}" - "${STATUS}" "${MAX_ATTEMPTS}" "${LOG}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "attempt": int(sys.argv[2]),
    "state": "failed_attempts_exhausted",
    "training_complete": False,
    "log_path": sys.argv[3],
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

exit 1
