from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_learning_policy_contract import PolicyIdentity, PolicyState  # noqa: E402
from pi_jwm.r6_learning_policy_preflight import (  # noqa: E402
    load_real_frozen_policy_state,
    policy_latent_from_belief,
    run_policy_cpu_smoke,
    validate_nonlocked_splits,
    write_preflight_bundle,
)


def _state() -> PolicyState:
    return PolicyState.create(
        explicit=torch.tensor([[1.0, 2.0, 3.0]]),
        latent=torch.tensor([[0.5, -0.5]]),
        offload_mask=torch.ones(1, 1, dtype=torch.bool),
        rb_mask=torch.ones(1, 1, dtype=torch.bool),
        cpu_task_mask=torch.tensor([[True, True, False]]),
        cpu_capacity=torch.tensor([[5.0, 10.0]]),
        cpu_task_node_index=torch.tensor([[0, 1, -1]]),
        identities=(PolicyIdentity("s0", 6, 1, "validation", "frozen-r6"),),
    )


class R6LearningPolicyPreflightTest(unittest.TestCase):
    def test_rssm_policy_latent_concatenates_deterministic_and_stochastic_state(self) -> None:
        class Belief:
            deterministic = torch.tensor([[1.0, 2.0]])
            stochastic = torch.tensor([[3.0, 4.0]])

        latent = policy_latent_from_belief(Belief())
        self.assertTrue(torch.equal(latent, torch.tensor([[1.0, 2.0, 3.0, 4.0]])))
        self.assertFalse(latent.requires_grad)

    def test_locked_test_is_rejected_before_loading(self) -> None:
        self.assertEqual(validate_nonlocked_splits(("train", "validation")), ("train", "validation"))
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_nonlocked_splits(("locked_test",))

    def test_real_builder_uses_nonlocked_frozen_b_state(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        state, audit, bindings = load_real_frozen_policy_state(
            dataset_root=code_root / "artifacts/datasets/airfogsim_teacher_aligned_v3",
            evaluation_root=code_root / "artifacts/evaluation/pi_jwm_eval_protocol_v3",
            r5_training_root=code_root / "artifacts/formal_training/pi_jwm_r5_gpu_training_v1",
            r5_analysis_root=code_root / "artifacts/formal_training/pi_jwm_r5_module_confirmation_analysis_v1",
            r6_paired_root=code_root / "artifacts/formal_training/pi_jwm_r6_cpu_paired_closed_loop_v1",
        )
        self.assertEqual(state.identities[0].split, "validation")
        self.assertGreater(state.latent.shape[1], 0)
        self.assertGreater(state.cpu_task_mask.sum().item(), 0)
        self.assertFalse(audit["world_model_updated"])
        self.assertIn("frozen_b_checkpoint", bindings)

    def test_cpu_smoke_runs_actor_critic_and_ppo_without_gpu(self) -> None:
        result = run_policy_cpu_smoke(_state(), hidden_dim=8, seed=20260808)
        self.assertTrue(result["r6_learning_policy_cpu_ready"])
        self.assertFalse(result["gpu_started"])
        self.assertFalse(result["world_model_updated"])
        self.assertEqual(result["hard_constraint_violation_count"], 0)
        self.assertEqual(set(result["training_reports"]), {"actor_critic", "ppo_clipped"})

    def test_bundle_is_new_and_manifest_is_self_verifying(self) -> None:
        smoke = run_policy_cpu_smoke(_state(), hidden_dim=8, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            write_preflight_bundle(
                output,
                summary=smoke,
                bindings={"fixture": "a" * 64},
                state_audit={"source": "unit_test"},
                action_rows=[{"policy_id": "actor_critic", "legal": True}],
                failures=[],
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_entry_count"], len(manifest["files"]))
            for name, metadata in manifest["files"].items():
                data = (output / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"])
            with self.assertRaises(FileExistsError):
                write_preflight_bundle(
                    output,
                    summary=smoke,
                    bindings={},
                    state_audit={},
                    action_rows=[],
                    failures=[],
                )


if __name__ == "__main__":
    unittest.main()
