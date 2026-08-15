from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pi_jwm.path_compat import load_exact_mapping, load_source_changes, resolve_repository_path


class RepositoryPathCompatibilityTest(unittest.TestCase):
    def test_maps_stable_legacy_prefixes_and_backslashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {
                r"代码\src\example.py": root / "code" / "src" / "example.py",
                "文档/文献/paper.pdf": root / "literature" / "paper.pdf",
                "文档/组会/slides.pptx": root / "meeting" / "slides.pptx",
                "文档/知识库/PIJWM主文档.md": root / "记录" / "PIJWM主文档.md",
            }
            for raw, target in expected.items():
                with self.subTest(raw=raw):
                    self.assertEqual(resolve_repository_path(root, raw), target)

    def test_research_progress_uses_exact_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {
                "文档/研究进展/design.md": "记录/研究进展/design.md",
                "文档/研究进展/draft.tex": "paper/draft.tex",
            }
            self.assertEqual(
                resolve_repository_path(
                    root,
                    "文档/研究进展/draft.tex",
                    exact_mapping=mapping,
                ),
                root / "paper" / "draft.tex",
            )
            self.assertEqual(
                resolve_repository_path(root, "文档/研究进展/unmapped.md"),
                root / "文档" / "研究进展" / "unmapped.md",
            )

    def test_does_not_invent_deleted_superpowers_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolved = resolve_repository_path(
                root,
                "docs/superpowers/plans/deleted.md",
            )
            self.assertEqual(resolved, root / "docs" / "superpowers" / "plans" / "deleted.md")
            self.assertFalse(resolved.exists())

    def test_rejects_absolute_escape_and_mapping_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for raw in ("../outside.txt", "/absolute.txt", "C:/absolute.txt"):
                with self.subTest(raw=raw):
                    with self.assertRaisesRegex(ValueError, "repository-relative"):
                        resolve_repository_path(root, raw)
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                resolve_repository_path(
                    root,
                    "old.txt",
                    exact_mapping={"old.txt": "../outside.txt"},
                )

    def test_load_exact_mapping_reads_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            path.write_text(
                '{"schema":"pi_jwm_repository_layout_migration_v1",'
                '"entries":[{"source":"old.txt","target":"new.txt"}]}',
                encoding="utf-8",
            )
            self.assertEqual(load_exact_mapping(path), {"old.txt": "new.txt"})

    def test_load_source_changes_requires_reviewed_current_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "changes.json"
            path.write_text(
                '{"schema":"pi_jwm_repository_layout_source_changes_v1",'
                '"entries":[{"source":"old.txt","new_sha256":"' + "a" * 64 + '"}]}',
                encoding="utf-8",
            )
            self.assertEqual(load_source_changes(path), {"old.txt": "a" * 64})


if __name__ == "__main__":
    unittest.main()
