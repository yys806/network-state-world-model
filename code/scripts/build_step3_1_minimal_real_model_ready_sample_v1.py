"""Build and validate one small model-ready sample from frozen Raw evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pi_jwm.model_ready_sample_contract_v1 import (  # noqa: E402
    build_sample,
    load_sample,
    validate_sample,
    write_sample,
)


RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919"
SAMPLE = OUT / "model_ready_sample.json"
MANIFEST = OUT / "manifest.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    sample = build_sample(raw, anchor_step=2)
    checks = validate_sample(sample)
    write_sample(sample, SAMPLE)
    roundtrip = load_sample(SAMPLE)
    if roundtrip != sample:
        raise AssertionError("serialize/load round-trip changed the sample")
    artifact_hash = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    source_files = {}
    for source in (
        ROOT / "code/src/pi_jwm/model_ready_sample_contract_v1.py",
        Path(__file__),
        ROOT / "code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py",
        ROOT / "code/scripts/run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py",
        ROOT / "code/tests/test_model_ready_sample_contract_v1.py",
        ROOT / "docs/contracts_PIJWM_MODEL_READY_SAMPLE_TENSOR_CONTRACT_V1.md",
        ROOT / "code/scripts/audit_step3_1f_future_action_references_v1.py",
    ):
        source_files[str(source.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "PI-JWM-Step-3.1F-Minimal-Real-Sample-Manifest-v3",
        "artifact": SAMPLE.name,
        "artifact_sha256": artifact_hash,
        "passed": True,
        "checks": checks,
        "source_files": source_files,
        "raw_source_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "scope": {"raw_source": str(RAW.relative_to(ROOT)), "H": 2, "L": 2, "locked_test": False, "training": False, "gpu": False},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
