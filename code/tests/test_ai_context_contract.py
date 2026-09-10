from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
AI_CONTEXT_ROOT = WORKSPACE_ROOT / "AI_CONTEXT"
REGISTRY_ROOT = WORKSPACE_ROOT / "docs" / "registries"
BUILDER_PATH = WORKSPACE_ROOT / "code" / "scripts" / "build_project_knowledge_index_v1.py"
SPEC = importlib.util.spec_from_file_location("build_project_knowledge_index_v1", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load project knowledge index builder: {BUILDER_PATH}")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


REQUIRED_FILES = (
    "00_PROJECT_STATE.md",
    "01_RESEARCH_CONTEXT.md",
    "02_ARCHITECTURE.md",
    "03_DATA_FLOW.md",
    "04_MODULE_MAP.md",
    "05_EXPERIMENTS.md",
    "06_DECISIONS.md",
    "07_KNOWN_ISSUES.md",
    "08_CHANGELOG.md",
)


class AIContextContractTests(unittest.TestCase):
    def test_required_context_files_exist_and_builder_accepts_them(self) -> None:
        self.assertEqual(REQUIRED_FILES, builder.AI_CONTEXT_FILES)
        for name in REQUIRED_FILES:
            self.assertTrue((AI_CONTEXT_ROOT / name).is_file(), name)
        report = builder.validate_ai_context(WORKSPACE_ROOT)
        self.assertEqual([], report["errors"])
        self.assertEqual(9, report["validated_file_count"])

    def test_context_has_navigation_evidence_and_boundary_markers(self) -> None:
        combined = "\n".join(
            (AI_CONTEXT_ROOT / name).read_text(encoding="utf-8") for name in REQUIRED_FILES
        )
        for phrase in (
            "Source of truth",
            "Unverified",
            "Research Rationale",
            "Awaiting Researcher Decision",
            "locked_test_accessed=false",
            "formal_performance_claim_ready=false",
            "code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py",
            "code/scripts/run_formal_p4_entity_rssm_gpu_v1.py",
        ):
            self.assertIn(phrase, combined)

    def test_context_is_registered_as_chatgpt_first_entrypoint(self) -> None:
        authority = json.loads(
            (REGISTRY_ROOT / "document_authority.json").read_text(encoding="utf-8")
        )
        self.assertEqual("AI_CONTEXT/00_PROJECT_STATE.md", authority["chatgpt_entrypoint"])
        paths = {row["path"] for row in authority["documents"]}
        for name in REQUIRED_FILES:
            self.assertIn(f"AI_CONTEXT/{name}", paths)

        routes = json.loads((REGISTRY_ROOT / "question_routes.json").read_text(encoding="utf-8"))
        route = next(row for row in routes["routes"] if row["id"] == "ROUTE-CHATGPT-ONBOARDING")
        self.assertEqual("AI_CONTEXT/00_PROJECT_STATE.md", route["primary_sources"][0])

    def test_permanent_rules_cover_three_roles_context_git_and_private_notes(self) -> None:
        agents = (WORKSPACE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for phrase in (
            "Research Engineer",
            "ChatGPT Web",
            "Context Consistency Check",
            "AI_CONTEXT/",
            "commit + push",
            "Private Notes Prohibition",
            "only when the user explicitly asks to modify AGENTS.md",
        ):
            self.assertIn(phrase, agents)

    def test_validator_rejects_an_incomplete_context_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AI_CONTEXT").mkdir()
            report = builder.validate_ai_context(root)
        self.assertEqual(0, report["validated_file_count"])
        self.assertTrue(any("missing" in error for error in report["errors"]))

    def test_backticked_repository_paths_resolve(self) -> None:
        prefixes = ("AI_CONTEXT/", "code/", "docs/", "记录/", "meeting/")
        for name in REQUIRED_FILES:
            text = (AI_CONTEXT_ROOT / name).read_text(encoding="utf-8")
            for token in re.findall(r"`([^`]+)`", text):
                if not token.startswith(prefixes):
                    continue
                relative = token.split("::", 1)[0].rstrip("/。，；：")
                self.assertTrue(
                    (WORKSPACE_ROOT / relative).exists(),
                    f"AI_CONTEXT/{name}:{relative}",
                )


if __name__ == "__main__":
    unittest.main()
