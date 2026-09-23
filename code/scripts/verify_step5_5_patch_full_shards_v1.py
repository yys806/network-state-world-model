"""Read every accepted Formal Dataset payload shard without whole-data RAM."""
from __future__ import annotations

import json
from pathlib import Path

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
OUTPUT = ROOT / "code/artifacts/audit/pi_jwm_step5_5_patch_20260923/full_shard_traversal_receipt.json"


if __name__ == "__main__":
    interface = FormalTrainingInterface.from_manifest(MANIFEST)
    dataset = FullFormalShardDataset(interface)
    result = dataset.audit_all_shards()
    receipt = {"schema_version": "PI-JWM-Step-5.5-PATCH-Full-Shard-Traversal-v1", "passed": result["all_payload_hashes_verified"] and result["unique_windows"] == 5520, "dataset_manifest_hash": interface.dataset_manifest_hash, **result, "scope": {"cpu": True, "gpu": False, "formal_training": False, "locked_test_accessed": False}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
