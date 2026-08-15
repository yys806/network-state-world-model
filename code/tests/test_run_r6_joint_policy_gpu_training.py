from __future__ import annotations

import ast
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_r6_joint_policy_gpu_training.py"


class R6JointPolicyGPUTrainingSourceTest(unittest.TestCase):
    def test_collection_uses_online_airfogsim_state_not_frozen_slot_state(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        collect = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_collect_trajectory"
        )
        names = {
            node.id
            for node in ast.walk(collect)
            if isinstance(node, ast.Name)
        }
        self.assertIn("_OnlineAirFogSimRecorder", names)
        self.assertIn("_state_from_online_history", names)
        self.assertNotIn("_state_for_slot", names)
        self.assertNotIn("_assert_time_alignment", names)

    def test_online_state_builder_rejects_non_online_batch_metadata(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('state_source"] != "online_airfogsim_strict_dual_graph"', source)
        self.assertIn("contract_from_dict", source)

    def test_summary_persists_online_state_action_and_reload_evidence(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"state_source": "online_airfogsim_strict_dual_graph"', source)
        self.assertIn('"candidate_selection_counts"', source)
        self.assertIn('"distinct_explicit_state_count"', source)
        self.assertIn('"checkpoint_reload_verified"', source)
        self.assertIn('"online_capture_counts"', source)

    def test_formal_runner_has_atomic_resume_checkpoint(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('resume_checkpoint = run_root / "resume_checkpoint.pt"', source)
        self.assertIn("_save_resume_checkpoint", source)
        self.assertIn("os.replace(temporary, path)", source)
        self.assertIn('"resumed_from_environment_step"', source)

    def test_final_partial_rollout_is_flushed_before_resume_checkpoint(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "len(pending_transitions) >= int(args.minibatch_size)\n"
            "            or (total_steps >= target_steps and bool(pending_transitions))",
            source,
        )


if __name__ == "__main__":
    unittest.main()
