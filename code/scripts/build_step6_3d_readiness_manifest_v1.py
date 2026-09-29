"""Hash the bounded 6.3D CPU readiness evidence without claiming selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929"
FILES = (
    "01_train_anchor_manifest.json",
    "02_validation_anchor_manifest.json",
    "03_selected_deadline_sidecars.json",
    "04_selected_deadline_alignment_receipt.json",
    "05_cpu_contract_smoke.json",
    "06_cpu_budget256_probe.json",
    "07_validation_cpu_contract_smoke.json",
    "09_selected_side_state_readiness.json",
    "10_scem_cpu_contract_smoke.json",
    "11_pre_diagnostic_formal_train_probe.json",
    "12_exact_oracle_receipt.json",
    "14_pre_input_identity_formal_train_probe.json",
)


def main() -> None:
    rows = []
    for name in FILES:
        path = OUT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        rows.append({"filename": name, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
    payload = {
        "step": "STEP 6.3D",
        "status": "CPU_READINESS_ONLY_FORMAL_COMPARISON_INCOMPLETE",
        "method_selected": None,
        "train_tuning_complete": False,
        "validation_comparison_complete": False,
        "files": rows,
        "locked_test": False,
    }
    (OUT / "13_cpu_readiness_manifest.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"status": payload["status"], "file_count": len(rows)}))


if __name__ == "__main__":
    main()
