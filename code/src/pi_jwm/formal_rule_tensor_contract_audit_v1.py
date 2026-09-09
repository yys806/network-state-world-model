"""Full unlocked-tensor audit for the deterministic rule-layer contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rule_tensor_contract(tensor_root: str | Path) -> dict[str, Any]:
    root = Path(tensor_root)
    if "locked_test" in str(root).lower():
        raise ValueError("locked_test path is forbidden")
    contract = json.loads((root / "tensor_contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failures: set[str] = set()
    contract_fields = {
        "action_source_endpoint_field": "task_action_source_node_index",
        "flow_task_mapping_field": "flow_task_index",
        "slot_duration_field": "slot_seconds",
    }
    for key, expected in contract_fields.items():
        if contract.get(key) != expected:
            failures.add(f"contract_{key}")

    manifest_mismatches: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(expected["size_bytes"]) or _sha256(path) != expected["sha256"]:
            manifest_mismatches.append(relative)
    if manifest_mismatches:
        failures.add("manifest_integrity")

    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        failures.add("seed_count")
    source_missing = {"offload": 0, "rb": 0, "return": 0, "cpu": 0}
    flow_mapping_missing = 0
    slots: list[float] = []
    for seed_dir in seed_dirs:
        with np.load(seed_dir / "trajectory_tensors.npz", allow_pickle=False) as arrays:
            required = {
                "task_action",
                "task_action_source_node_index",
                "flow_valid",
                "flow_task_index",
                "task_valid",
                "slot_seconds",
            }
            absent = required - set(arrays.files)
            if absent:
                failures.update(f"array_{name}_missing" for name in absent)
                continue
            action = arrays["task_action"]
            source = arrays["task_action_source_node_index"]
            if source.shape != (*action.shape[:-1], 4):
                failures.add("action_source_shape")
                continue
            checks = (("offload", 0, 0), ("rb", 1, 2), ("return", 2, 1), ("cpu", 5, 3))
            for name, action_column, endpoint_column in checks:
                active = action[..., action_column] > 0
                source_missing[name] += int(np.count_nonzero(active & (source[..., endpoint_column] < 0)))
            flow_valid = arrays["flow_valid"].astype(bool)
            flow_mapping_missing += int(np.count_nonzero(flow_valid & (arrays["flow_task_index"] < 0)))
            slot = float(arrays["slot_seconds"])
            slots.append(slot)
            if not np.isfinite(slot) or slot <= 0:
                failures.add("slot_seconds_invalid")
    for name, count in source_missing.items():
        if count:
            failures.add(f"{name}_source_missing")
    if flow_mapping_missing:
        failures.add("flow_task_mapping_missing")
    if slots and not np.allclose(slots, slots[0], rtol=1e-6, atol=1e-8):
        failures.add("slot_seconds_inconsistent")
    locked_materialized = (root / "locked_test").exists()
    if locked_materialized:
        failures.add("locked_test_materialized")

    return {
        "schema_version": "PI-JWM-formal-rule-tensor-contract-audit-v1",
        "rule_layer_tensor_contract_ready": not failures,
        "failed_checks": sorted(failures),
        "seed_count": len(seed_dirs),
        "slot_seconds_values": sorted(set(slots)),
        "source_missing_counts": source_missing,
        "flow_task_mapping_missing_count": flow_mapping_missing,
        "manifest_mismatch_count": len(manifest_mismatches),
        "manifest_mismatches": manifest_mismatches,
        "locked_test_accessed": False,
        "locked_test_materialized": locked_materialized,
    }


__all__ = ["audit_rule_tensor_contract"]
