#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/pi_jwm_r5_20260805"
PYTHON="/root/miniconda3/bin/python"
OUTPUT="${BASE}/artifacts/pi_jwm_r5_gpu_training_v1"
STATUS="${BASE}/artifacts/r5_formal_launcher_status.json"
LOCK="${BASE}/artifacts/r5_formal_launcher.lock"
MAX_ATTEMPTS=3

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another R5 formal launcher already owns ${LOCK}" >&2
  exit 3
fi

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  resume_args=()
  if [[ -f "${OUTPUT}/training_protocol.json" ]]; then
    resume_args+=(--resume)
  fi

  cd "${BASE}/code/scripts"
  PYTHONPATH="${BASE}/code/src:${BASE}/code/scripts" \
    "${PYTHON}" -u run_r5_gpu_training.py \
      --dataset-root "${BASE}/data" \
      --evaluation-root "${BASE}/evaluation" \
      --r4-screening-root "${BASE}/r4_screening" \
      --output-dir "${OUTPUT}" \
      --device cuda:0 \
      --hidden-dim 16 \
      --micro-batch-size 4 \
      "${resume_args[@]}"
  exit_code=$?

  complete=$("${PYTHON}" - "${OUTPUT}/training_summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    print("false")
else:
    print(str(bool(json.loads(path.read_text())["r5_gpu_training_complete"])).lower())
PY
)

  "${PYTHON}" - "${STATUS}" "${attempt}" "${exit_code}" "${complete}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "attempt": int(sys.argv[2]),
    "child_exit_code": int(sys.argv[3]),
    "training_complete": sys.argv[4] == "true",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

  if [[ "${complete}" == "true" ]]; then
    exit 0
  fi
  sleep 30
done

exit 1
