from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_joint_policy_preflight import (  # noqa: E402
    GPUReadinessEvidence,
    assess_gpu_readiness,
    write_gpu_readiness_bundle,
)


def _evidence(**overrides) -> GPUReadinessEvidence:
    values = {
        "joint_candidate_contract_passed": True,
        "offload_nonnoop_count": 1,
        "rb_nonnoop_count": 1,
        "cpu_nonnoop_count": 1,
        "hard_constraint_rejection_passed": True,
        "actor_critic_update_passed": True,
        "ppo_update_passed": True,
        "real_rollout_transition_count": 4,
        "identity_continuity_passed": True,
        "reward_recomputation_passed": True,
        "gae_reference_passed": True,
        "world_model_sha256_before": "a" * 64,
        "world_model_sha256_after": "a" * 64,
        "locked_test_accessed": False,
        "gpu_used": False,
        "dataset_regenerated": False,
        "world_model_retrained": False,
        "regression_passed": True,
    }
    values.update(overrides)
    return GPUReadinessEvidence(**values)


class R6JointPolicyPreflightTest(unittest.TestCase):
    def test_all_nine_gates_are_required_for_ready(self) -> None:
        ready = assess_gpu_readiness(_evidence())
        self.assertTrue(ready.r6_gpu_strategy_training_ready)
        self.assertEqual((), ready.blockers)
        blocked = assess_gpu_readiness(_evidence(rb_nonnoop_count=0))
        self.assertFalse(blocked.r6_gpu_strategy_training_ready)
        self.assertIn("real_nonnoop_action_evidence", blocked.blockers)

    def test_locked_gpu_or_world_model_mutation_is_never_ready(self) -> None:
        for value in (
            _evidence(locked_test_accessed=True),
            _evidence(gpu_used=True),
            _evidence(world_model_sha256_after="b" * 64),
            _evidence(dataset_regenerated=True),
            _evidence(world_model_retrained=True),
        ):
            result = assess_gpu_readiness(value)
            self.assertFalse(result.r6_gpu_strategy_training_ready)
            self.assertTrue(result.blockers)

    def test_bundle_manifest_hashes_every_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assessment = assess_gpu_readiness(_evidence())
            write_gpu_readiness_bundle(
                root,
                assessment=assessment,
                evidence=_evidence(),
                input_bindings={"r1_manifest": "c" * 64},
                protocol_payload={"schema_version": "fixture"},
                action_rows=[{"slot": 2, "candidate_id": "default"}],
                transition_rows=[{"slot": 2, "reward": 1.0}],
            )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(6, len(manifest["files"]))
            self.assertTrue(manifest["self_check_passed"])
            self.assertTrue((root / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
