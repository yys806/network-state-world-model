from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_repository_layout_migration.py"
SPEC = importlib.util.spec_from_file_location("audit_repository_layout_migration", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load migration audit script: {SCRIPT_PATH}")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class RepositoryLayoutMigrationAuditTest(unittest.TestCase):
    def test_validate_entries_rejects_duplicate_targets(self) -> None:
        entries = [
            {"source": "a.txt", "target": "same.txt", "size_bytes": 1, "sha256": "0" * 64},
            {"source": "b.txt", "target": "same.txt", "size_bytes": 1, "sha256": "1" * 64},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            audit.validate_entries(entries)

    def test_verify_entries_detects_size_and_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "new.txt"
            target.write_text("changed\n", encoding="utf-8")
            entries = [
                {
                    "source": "old.txt",
                    "target": "new.txt",
                    "size_bytes": 4,
                    "sha256": hashlib.sha256(b"old\n").hexdigest(),
                }
            ]
            result = audit.verify_entries(root, entries, phase="after")
            self.assertEqual(result["missing_targets"], [])
            self.assertEqual(result["sources_still_present"], [])
            self.assertEqual(result["size_mismatches"], ["new.txt"])
            self.assertEqual(result["hash_mismatches"], ["new.txt"])

    def test_classify_gate_delta_separates_existing_and_layout_errors(self) -> None:
        before = ["missing source: docs/superpowers/plans/deleted.md"]
        after = before + ["missing source: code/src/new.py"]
        result = audit.classify_gate_delta(before, after)
        self.assertEqual(result["pre_existing"], before)
        self.assertEqual(result["resolved"], [])
        self.assertEqual(result["layout_induced"], ["missing source: code/src/new.py"])

    def test_classify_gate_delta_treats_code_alias_as_same_error(self) -> None:
        before = ["source hash mismatch: 代码/src/example.py"]
        after = ["source hash mismatch: code/src/example.py"]
        result = audit.classify_gate_delta(before, after)
        self.assertEqual(result["pre_existing"], before)
        self.assertEqual(result["resolved"], [])
        self.assertEqual(result["layout_induced"], [])


if __name__ == "__main__":
    unittest.main()
