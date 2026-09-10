from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = WORKSPACE_ROOT / "code" / "scripts" / "query_project_knowledge_v1.py"
SPEC = importlib.util.spec_from_file_location("query_project_knowledge_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load project knowledge query tool: {SCRIPT_PATH}")
query_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(query_tool)


class ProjectKnowledgeQueryTests(unittest.TestCase):
    def _ids(self, question: str) -> set[str]:
        result = query_tool.query_knowledge(WORKSPACE_ROOT, question, limit=10)
        self.assertTrue(result["verification_required"])
        self.assertFalse(result["locked_test_accessed"])
        return {row["id"] for row in result["matches"]}

    def test_finds_current_method_and_entrypoint(self) -> None:
        ids = self._ids("现在使用的模型和训练入口在哪里")
        self.assertIn("ROUTE-CURRENT-METHOD", ids)
        self.assertIn("P4-EARSSM-SEED-20260831", ids)

    def test_handles_a_colloquial_current_model_question(self) -> None:
        ids = self._ids("我们现在模型是啥")
        self.assertIn("ROUTE-CURRENT-METHOD", ids)

    def test_finds_why_historical_complete_rssm_was_replaced(self) -> None:
        ids = self._ids("之前的完整 RSSM 为什么不用了")
        self.assertIn("HIST-P4-GLOBAL-COMPLETE-RSSM", ids)

    def test_finds_formal_seed_result_provenance(self) -> None:
        ids = self._ids("seed 20260831 的指标结果从哪里来")
        self.assertIn("RES-P4-20260831-SINGLE-SEED", ids)

    def test_finds_third_seed_deferred_guard(self) -> None:
        ids = self._ids("第三个 seed 什么时候运行")
        self.assertIn("P4-SEED-20260832", ids)

    def test_finds_an_exact_historical_artifact_name(self) -> None:
        ids = self._ids("pi_jwm_v11_method_potential_compare_20260626 这个实验做过吗")
        self.assertIn(
            "ARTIFACT::experiments/pi_jwm_v11_method_potential_compare_20260626",
            ids,
        )

    def test_finds_an_exact_code_filename(self) -> None:
        ids = self._ids("formal_world_model_loss_v1.py 在哪里")
        self.assertIn("FILE::code/src/pi_jwm/formal_world_model_loss_v1.py", ids)

    def test_finds_chatgpt_onboarding_entrypoint(self) -> None:
        ids = self._ids("ChatGPT 新对话从哪里开始读项目")
        self.assertIn("ROUTE-CHATGPT-ONBOARDING", ids)


if __name__ == "__main__":
    unittest.main()
