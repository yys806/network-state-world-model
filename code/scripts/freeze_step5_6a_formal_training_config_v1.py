"""Materialize the explicitly approved Formal Training Config v1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1


def freeze(manifest: Path, output_dir: Path) -> dict:
    config = formal_training_config_v1(manifest)
    payload = config.as_manifest()
    payload["formal_training_config"] = "FROZEN"
    payload["formal_training_readiness"] = "READY_TO_START"
    payload["scope"] = {
        "formal_training": False, "gpu_training_verified": False,
        "locked_test_accessed": False, "baseline": False,
        "planner": False, "performance_claim": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "formal_training_config_v1.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "PI-JWM-STEP-5.6A-Formal-Training-Config-Freeze-v1",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_id": config.dataset_id,
        "dataset_manifest_sha256": config.dataset_manifest_sha256,
        "formal_training_config": "FROZEN",
        "formal_training_readiness": "READY_TO_START",
        "researcher_decision": True,
        "source": "explicit researcher-approved STEP 5.6A-CONFIG-FREEZE values",
        "formal_training_started": False,
        "gpu_training_verified": False,
        "locked_test_accessed": False,
        "scope": payload["scope"],
        "passed": True,
    }
    (output_dir / "config_freeze_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.manifest, args.output_dir), ensure_ascii=False, sort_keys=True))
