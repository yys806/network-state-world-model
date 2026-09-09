from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _fixture(root: Path, *, corrupt_source: bool = False) -> None:
    contract = {
        "action_source_endpoint_field": "task_action_source_node_index",
        "flow_task_mapping_field": "flow_task_index",
        "slot_duration_field": "slot_seconds",
    }
    (root / "tensor_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    seed = root / "seed_001"
    seed.mkdir()
    action = np.zeros((2, 1, 8), dtype=np.float32)
    action[0, 0, 0] = 1.0
    source = np.full((2, 1, 4), -1, dtype=np.int32)
    source[0, 0, 0] = -1 if corrupt_source else 0
    path = seed / "trajectory_tensors.npz"
    np.savez_compressed(
        path,
        task_action=action,
        task_action_source_node_index=source,
        flow_valid=np.asarray([True]),
        flow_task_index=np.asarray([0], dtype=np.int32),
        task_valid=np.asarray([True]),
        slot_seconds=np.asarray(0.1, dtype=np.float32),
    )
    files = {}
    for file_path in (root / "tensor_contract.json", path):
        files[file_path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            "size_bytes": file_path.stat().st_size,
        }
    (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


class FormalRuleTensorContractAuditV1Tests(unittest.TestCase):
    def test_accepts_complete_nonlocked_contract(self):
        from pi_jwm.formal_rule_tensor_contract_audit_v1 import audit_rule_tensor_contract

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root)
            report = audit_rule_tensor_contract(root)
        self.assertTrue(report["rule_layer_tensor_contract_ready"])
        self.assertEqual(1, report["seed_count"])
        self.assertFalse(report["locked_test_accessed"])

    def test_rejects_missing_action_source(self):
        from pi_jwm.formal_rule_tensor_contract_audit_v1 import audit_rule_tensor_contract

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root, corrupt_source=True)
            report = audit_rule_tensor_contract(root)
        self.assertFalse(report["rule_layer_tensor_contract_ready"])
        self.assertIn("offload_source_missing", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
