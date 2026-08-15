from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_learning_policy import MaskedActorCritic  # noqa: E402
from pi_jwm.r6_learning_policy_contract import (  # noqa: E402
    ActionSpec,
    PolicyIdentity,
    PolicyState,
)
from pi_jwm.r6_learning_policy_training import (  # noqa: E402
    PolicyTrainingBatch,
    actor_critic_cpu_step,
    ppo_cpu_step,
)


class _FrozenWorldModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _setup_batch():
    torch.manual_seed(8)
    world_model = _FrozenWorldModel()
    explicit = torch.tensor([[1.0, 2.0, 3.0]])
    latent = world_model(explicit)
    state = PolicyState.create(
        explicit=explicit,
        latent=latent,
        offload_mask=torch.tensor([[True, False, True]]),
        rb_mask=torch.tensor([[True, True, False]]),
        cpu_task_mask=torch.tensor([[True, True]]),
        cpu_capacity=torch.tensor([[10.0]]),
        cpu_task_node_index=torch.tensor([[0, 0]]),
        identities=(PolicyIdentity("s0", 6, 1, "validation", "frozen-r6"),),
    )
    spec = ActionSpec(3, 3, 2, 0, 0)
    policy = MaskedActorCritic(explicit_dim=3, latent_dim=4, hidden_dim=16, spec=spec)
    decision = policy.act(state, deterministic=False, seed=9)
    with torch.no_grad():
        old_log_prob = policy.evaluate(state, decision.proposed).log_prob.clone()
    batch = PolicyTrainingBatch(
        state=state,
        action=decision.proposed,
        advantage=torch.tensor([1.25]),
        returns=torch.tensor([0.75]),
        old_log_prob=old_log_prob,
    )
    return world_model, policy, batch


class R6LearningPolicyTrainingTest(unittest.TestCase):
    def test_actor_critic_step_updates_policy_without_touching_world_model(self) -> None:
        world_model, policy, batch = _setup_batch()
        world_before = tuple(value.detach().clone() for value in world_model.parameters())
        report = actor_critic_cpu_step(
            policy=policy,
            batch=batch,
            optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3),
        )
        self.assertTrue(report.policy_parameter_changed)
        self.assertTrue(all(math.isfinite(value) for value in report.numeric_values()))
        self.assertTrue(all(parameter.grad is None for parameter in world_model.parameters()))
        self.assertTrue(
            all(torch.equal(before, after) for before, after in zip(world_before, world_model.parameters()))
        )

    def test_ppo_step_reports_finite_ratio_clip_and_gradient(self) -> None:
        _, policy, batch = _setup_batch()
        report = ppo_cpu_step(
            policy=policy,
            batch=batch,
            optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3),
            clip_epsilon=0.2,
        )
        self.assertTrue(report.policy_parameter_changed)
        self.assertTrue(all(math.isfinite(value) for value in report.numeric_values()))
        self.assertLessEqual(report.ratio_min, report.ratio_max)
        self.assertEqual(report.objective_id, "ppo_clipped")

    def test_nonfinite_training_targets_are_rejected(self) -> None:
        _, policy, batch = _setup_batch()
        bad = PolicyTrainingBatch(
            state=batch.state,
            action=batch.action,
            advantage=torch.tensor([float("nan")]),
            returns=batch.returns,
            old_log_prob=batch.old_log_prob,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            actor_critic_cpu_step(
                policy=policy,
                batch=bad,
                optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3),
            )


if __name__ == "__main__":
    unittest.main()
