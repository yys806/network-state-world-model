import subprocess
import sys
import unittest
from pathlib import Path


class ResearchProgressAuditScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.archive_root = cls.repo_root / "记录" / "研究进展" / "归档"
        cls.audit_dir = cls.repo_root / "docs" / "templates"
        cls.overview_dir = cls.archive_root / "旧版主文档"
        cls.paper_dir = cls.repo_root / "paper" / "archive" / "pi_jwm_ton_draft_zh"

    def test_archived_audit_scripts_resolve_archived_documents(self):
        expected_documents = (
            self.overview_dir / "research_progress_overview.tex",
            self.paper_dir / "pi_jwm_ton_draft_zh.tex",
        )
        for document in expected_documents:
            with self.subTest(document=document.name):
                self.assertTrue(document.is_file(), document)

        for script_name in ("audit_tables.py", "audit_table_numbers.py"):
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [sys.executable, str(self.audit_dir / script_name)],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
