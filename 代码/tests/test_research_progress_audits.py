import subprocess
import sys
import unittest
from pathlib import Path


class ResearchProgressAuditScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.progress_dir = cls.repo_root / "文档" / "研究进展"

    def test_audit_scripts_resolve_documents_from_their_own_directory(self):
        for script_name in ("audit_tables.py", "audit_table_numbers.py"):
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [sys.executable, str(self.progress_dir / script_name)],
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
