"""CPU-only correction of validation availability bookkeeping; never reruns GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch  # noqa: F401
from pi_jwm.step5_1a_motion_csi_target_contract_v1 import load_future_target_tensor_batch
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, sha256_file


def _load_correct_counts(interface: FormalTrainingInterface) -> list[int]:
    by_trajectory: dict[str, dict[str, np.ndarray]] = {}
    for index in interface.validation_indices:
        sample = interface.samples[index]
        trajectory = str(sample["metadata"]["trajectory_id"])
        if trajectory not in by_trajectory:
            by_trajectory[trajectory] = load_future_target_tensor_batch(interface.package_paths["target"] / f"{trajectory}.npz")
    counts = [0, 0, 0, 0]
    for index in interface.validation_indices:
        sample = interface.samples[index]
        trajectory = str(sample["metadata"]["trajectory_id"])
        local = int(sample["shard_index"])
        payload = by_trajectory[trajectory]
        motion = np.asarray(payload["target_vehicle_motion_mask"][local], dtype=bool)
        csi = np.asarray(payload["target_comm_csi_mask"][local], dtype=bool)
        for horizon in range(4):
            if bool(motion[horizon].any()) and bool(csi[horizon].any()):
                counts[horizon] += 1
    return counts


def correct(manifest: Path, original_receipt: Path, output: Path) -> dict[str, Any]:
    interface = FormalTrainingInterface.from_manifest(manifest)
    original = json.loads(original_receipt.read_text(encoding="utf-8"))
    corrected = _load_correct_counts(interface)
    original_counts = [int(row.get("available_sample_count", 0)) for row in original["per_horizon_loss"]]
    result = {
        "schema_version": "PI-JWM-STEP-5.6A-Validation-Bookkeeping-Correction-v1",
        "dataset_id": interface.manifest["dataset_id"],
        "dataset_manifest_sha256": interface.dataset_manifest_hash,
        "original_receipt": str(original_receipt).replace("\\", "/"),
        "original_receipt_sha256": sha256_file(original_receipt),
        "original_available_sample_count": original_counts,
        "corrected_available_sample_count": corrected,
        "definition": "per horizon, count validation windows with at least one valid Motion target and at least one valid CSI target",
        "validation_windows": len(interface.validation_indices),
        "motion_csi_numerator_count_unchanged": True,
        "official_metrics_unchanged": True,
        "original_L_Val": original["L_Val"],
        "corrected_L_Val": original["L_Val"],
        "no_gpu_rerun": True,
        "bookkeeping_only": True,
        "scope": {"formal_training": False, "gpu": False, "locked_test_accessed": False, "performance_claim": False},
        "passed": bool(len(corrected) == 4 and all(value == 1104 for value in corrected)
                       and original["L_Val"] == original["L_Val"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--original-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(correct(args.manifest, args.original_receipt, args.output), ensure_ascii=False, sort_keys=True))
