from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from governance_quarantine import (  # noqa: E402
    MANIFEST_FIELDS,
    ensure_within_root,
    quarantine_tree,
    sha256_file,
)


class GovernanceQuarantineTest(unittest.TestCase):
    def test_ensure_within_root_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()

            with self.assertRaises(ValueError):
                ensure_within_root(root.parent / "outside.txt", root)

    def test_dry_run_writes_complete_manifest_without_moving_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            destination = base / "quarantine" / "source"
            manifest = base / "manifest.csv"
            source.mkdir()
            (source / "a.txt").write_text("alpha", encoding="utf-8")
            (source / "nested").mkdir()
            (source / "nested" / "b.txt").write_text("beta", encoding="utf-8")

            rows = quarantine_tree(
                source=source,
                destination=destination,
                manifest_path=manifest,
                reason="test dry run",
                dry_run=True,
            )

            self.assertEqual(len(rows), 2)
            self.assertTrue(source.exists())
            self.assertFalse(destination.exists())
            with manifest.open(newline="", encoding="utf-8-sig") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(tuple(written[0]), MANIFEST_FIELDS)
            self.assertEqual({row["status"] for row in written}, {"planned"})
            self.assertTrue(all(row["sha256"] for row in written))

    def test_move_preserves_hashes_and_removes_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            destination = base / "quarantine" / "source"
            manifest = base / "manifest.csv"
            source.mkdir()
            original = source / "payload.bin"
            original.write_bytes(b"payload" * 1024)
            expected_hash = sha256_file(original)

            rows = quarantine_tree(
                source=source,
                destination=destination,
                manifest_path=manifest,
                reason="test move",
                dry_run=False,
            )

            moved = destination / "payload.bin"
            self.assertFalse(source.exists())
            self.assertTrue(moved.exists())
            self.assertEqual(sha256_file(moved), expected_hash)
            self.assertEqual(rows[0]["status"], "verified")

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            destination = base / "quarantine" / "source"
            source.mkdir()
            destination.mkdir(parents=True)
            (source / "a.txt").write_text("new", encoding="utf-8")
            existing = destination / "a.txt"
            existing.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                quarantine_tree(
                    source=source,
                    destination=destination,
                    manifest_path=base / "manifest.csv",
                    reason="collision",
                    dry_run=False,
                )

            self.assertEqual(existing.read_text(encoding="utf-8"), "existing")
            self.assertTrue((source / "a.txt").exists())


if __name__ == "__main__":
    unittest.main()
