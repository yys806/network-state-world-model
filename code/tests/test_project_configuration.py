from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"


class ProjectConfigurationTest(unittest.TestCase):
    def test_pyproject_declares_pi_jwm_src_layout_and_core_dependencies(self) -> None:
        pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            config = tomllib.load(handle)

        project = config["project"]
        self.assertEqual(project["name"], "pi-jwm")
        self.assertEqual(project["requires-python"], ">=3.10,<3.14")
        dependencies = set(project["dependencies"])
        self.assertIn("numpy>=1.24", dependencies)
        self.assertIn("torch>=2.1", dependencies)
        self.assertEqual(config["tool"]["setuptools"]["package-dir"], {"": "code/src"})

    def test_dependency_files_separate_core_and_experiment_packages(self) -> None:
        core = [
            line
            for line in (CODE_ROOT / "requirements-core.txt").read_text(encoding="utf-8").splitlines()
            if line
        ]
        experiments = [
            line
            for line in (CODE_ROOT / "requirements-experiments.txt").read_text(encoding="utf-8").splitlines()
            if line
        ]

        self.assertEqual(core, ["numpy>=1.24", "torch>=2.1"])
        self.assertIn("-r requirements-core.txt", experiments)
        self.assertIn("scikit-learn>=1.3", experiments)
        self.assertIn("pandas>=2.0", experiments)
        self.assertIn("matplotlib>=3.7", experiments)

    def test_gitignore_covers_generated_files_without_hiding_evidence(self) -> None:
        patterns = {
            line.strip()
            for line in (WORKSPACE_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        for required in {"__pycache__/", "*.py[cod]", ".venv/", "*.pid", "*.log", ".pytest_cache/"}:
            self.assertIn(required, patterns)
        self.assertNotIn("*.json", patterns)
        self.assertNotIn("*.csv", patterns)

    def test_current_python_is_within_declared_range(self) -> None:
        self.assertGreaterEqual(sys.version_info[:2], (3, 10))
        self.assertLess(sys.version_info[:2], (3, 14))

    def test_repository_uses_canonical_top_level_directories(self) -> None:
        for directory in (
            "code",
            "记录",
            "paper",
            "literature",
            "meeting",
            "docs",
        ):
            self.assertTrue((WORKSPACE_ROOT / directory).is_dir(), directory)
        self.assertFalse((WORKSPACE_ROOT / "代码").exists())
        self.assertFalse((WORKSPACE_ROOT / "文档").exists())

        root_markdown = sorted(path.name for path in WORKSPACE_ROOT.glob("*.md"))
        self.assertEqual(root_markdown, ["AGENTS.md", "README.md"])

        for script_path in (CODE_ROOT / "scripts").glob("*.py"):
            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn(
                ' / "开会"',
                content,
                msg=f"Legacy meeting path remains in {script_path.name}",
            )


if __name__ == "__main__":
    unittest.main()
