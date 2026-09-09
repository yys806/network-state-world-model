from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
TEST_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class RecordingWorldModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def __call__(self, batch):
        action = batch["future_action"]["task_action"]
        self.calls.append(action.detach().clone())
        return {"state": action.clone(), "task": action[..., 0].sum()}


class FormalCandidateRolloutPlannerV1Tests(unittest.TestCase):
    def _small_batch(self):
        zeros = torch.zeros(1, 2, 1, 8)
        return {
            "history": {"node_state": torch.zeros(1, 2, 1, 7)},
            "future_action": {
                "task_action": zeros.clone(),
                "task_action_present": torch.zeros(1, 2, 1, dtype=torch.bool),
                "task_action_node_index": torch.zeros(1, 2, 1, 3, dtype=torch.long),
                "task_action_source_node_index": torch.zeros(1, 2, 1, 3, dtype=torch.long),
            },
            "static": {"node_kind_index": torch.zeros(1, 1, dtype=torch.long)},
        }

    def test_runs_each_legal_candidate_from_one_common_belief_and_replans(self):
        from pi_jwm.formal_candidate_rollout_planner_v1 import (
            CandidateAction,
            FormalCandidateRolloutPlanner,
        )

        model = RecordingWorldModel()
        base = self._small_batch()

        def generate(batch):
            first = copy.deepcopy(batch["future_action"])
            second = copy.deepcopy(batch["future_action"])
            second["task_action"][..., 0] = 2.0
            return (CandidateAction("stay", first), CandidateAction("serve", second))

        def extract(outputs, _candidate):
            task = float(outputs["task"])
            return {
                "future_state": {"task_state_mean": outputs["state"]},
                "task_outcome": task,
                "cost": abs(task),
                "risk": 0.0,
            }

        planner = FormalCandidateRolloutPlanner(
            world_model=model,
            candidate_generator=generate,
            prediction_extractor=extract,
            objective=lambda prediction: prediction.cost,
        )
        decision = planner.plan(base)

        self.assertEqual("prototype_only", decision.mechanism_status)
        self.assertEqual(2, len(model.calls))
        self.assertEqual("stay", decision.selected_candidate_id)
        self.assertEqual(2, len(decision.records))
        self.assertEqual(
            {record.common_belief_fingerprint for record in decision.records},
            {decision.common_belief_fingerprint},
        )
        self.assertEqual((1, 1, 1, 8), tuple(decision.selected_first_action["task_action"].shape))

        changed = copy.deepcopy(base)
        changed["history"]["node_state"].fill_(1.0)
        replanned = planner.replan(changed)
        self.assertEqual(1, replanned.replan_count)
        self.assertNotEqual(decision.common_belief_fingerprint, replanned.common_belief_fingerprint)
        self.assertTrue(model.training)

    def test_wraps_action_conditioned_formal_world_model_without_target_access(self):
        from pi_jwm.formal_dual_graph_world_model_v1 import (
            FormalDualGraphWorldModel,
            FormalWorldModelConfig,
        )
        from pi_jwm.formal_candidate_rollout_planner_v1 import (
            CandidateAction,
            FormalCandidateRolloutPlanner,
        )
        from test_formal_dual_graph_world_model_v1 import fake_formal_batch

        batch = fake_formal_batch()
        source = torch.zeros_like(batch["future_action"]["task_action_node_index"])
        batch["future_action"]["task_action_source_node_index"] = source
        model = FormalDualGraphWorldModel(
            FormalWorldModelConfig(
                mode="coupled_dual_gnn",
                hidden_dim=8,
                history_steps=3,
                horizon_steps=2,
                use_system_energy_head=True,
            )
        )

        def generate(candidate_batch):
            baseline = {key: value.clone() for key, value in candidate_batch["future_action"].items()}
            changed = {key: value.clone() for key, value in candidate_batch["future_action"].items()}
            changed["task_action"][:, 0, 0, 0] = 1.0
            changed["task_action_present"][:, 0, 0] = True
            return (CandidateAction("baseline", baseline), CandidateAction("action_0", changed))

        def extract(outputs, _candidate):
            task_probability = outputs["task_lifecycle_logits"].softmax(-1)[..., 3].mean()
            cost = outputs["uav_energy_delta_mean"].mean()
            risk = outputs["task_state_log_variance"].exp().mean()
            return {
                "future_state": {"task_state_mean": outputs["task_state_mean"]},
                "task_outcome": task_probability,
                "cost": cost,
                "risk": risk,
            }

        planner = FormalCandidateRolloutPlanner(
            world_model=model,
            candidate_generator=generate,
            prediction_extractor=extract,
            objective=lambda prediction: prediction.cost + prediction.risk - prediction.task_outcome,
        )
        decision = planner.plan(batch)

        self.assertEqual(2, len(decision.records))
        self.assertIn(decision.selected_candidate_id, {"baseline", "action_0"})
        self.assertTrue(all(torch.isfinite(record.prediction.future_state["task_state_mean"]).all() for record in decision.records))
        self.assertTrue(all(record.prediction.cost >= 0.0 for record in decision.records))
        self.assertTrue(all(record.prediction.risk >= 0.0 for record in decision.records))
        self.assertEqual(set(decision.selected_first_action), {
            "task_action",
            "task_action_present",
            "task_action_node_index",
            "task_action_source_node_index",
        })


if __name__ == "__main__":
    unittest.main()
