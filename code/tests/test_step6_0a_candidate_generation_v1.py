"""SYNTHETIC_CONTRACT_EVIDENCE: no simulator or model rollout."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "src"), str(ROOT / "code" / "scripts")]

from build_step5_1d_unified_model_chain_v1 import build_action
from pi_jwm.step6_0a_candidate_generation_v1 import (
    Backend, CandidateActionSequence, CandidateActionStep, CandidatePool, Constraint,
    ConstraintStatus, HybridCompositionBackend, LearnedProposalBackend,
    PlannerCandidateContext, SearchStubBackend, compile_candidate, rule_fallback,
    shift_warm_start,
)


def fixture():
    state = {
        "position": torch.zeros((1, 2, 3)), "entity_presence": torch.tensor([[True, True]]),
        "uav_mask": torch.tensor([[True, False]]), "rb_active_mask": torch.zeros((1, 1, 2), dtype=torch.bool),
        "flow_task_index": torch.tensor([[0]]), "flow_presence": torch.tensor([[True]]),
        "flow_known": torch.tensor([[True]]), "flow_identity_index": torch.tensor([[1]]),
        "flow_comm_relation_index": torch.tensor([[0]]),
        "carrying_hop_source_index": torch.tensor([[0]]),
        "carrying_hop_destination_index": torch.tensor([[1]]),
        "comm_source_index": torch.tensor([[0]]), "comm_target_index": torch.tensor([[1]]),
        "comm_presence": torch.tensor([[True]]), "comm_validity": torch.tensor([[True]]),
    }
    context = PlannerCandidateContext(
        {"input_entity_index": {"physical": {"uav": 0, "edge": 1},
                                "task": {"task": 0}, "logical_flow": {"flow::task::Input::0": 0}}},
        ({"tasks": [{"task_id": "task", "task_index": 0, "presence": True,
                     "current_node_id": "uav"}], "logical_flows": []},), state, "synthetic-current")
    return context


def sequence(context, step, constraints=()):
    return CandidateActionSequence((step,), "candidate", Backend.SEARCH, 7,
                                   context.causal_provenance, frozenset({"SEARCH"}), constraints)


class CandidateContractTests(unittest.TestCase):
    def test_four_action_exact_adapter_equivalence(self):
        ctx = fixture()
        step = CandidateActionStep(
            route=({"task_id": "task", "task_index": 0, "task_node_index": 0,
                    "target_node_index": 1, "route_node_indices": [1], "route_kind": "offload"},),
            comm=({"task_id": "task", "task_index": 0, "relation_index": 0, "rb_indices": [0, 1]},),
            comp=({"task_id": "task", "node_id": "edge", "allocated_cpu_per_s": 2.5},),
            mob=({"uav_index": 0, "azimuth_rad": 0.25, "elevation_rad": -0.5,
                  "speed_mps": 3.0},))
        candidate = sequence(ctx, step)
        self.assertEqual({c.constraint_name for c in candidate.unresolved_constraints},
                         {"dynamic_available_cpu", "mobility_numeric_bounds"})
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            compile_candidate(candidate, ctx, (ctx.current_state,))
        actual, mappings = compile_candidate(candidate, ctx, (ctx.current_state,), synthetic_contract_test_only=True)
        expected, old_mapping = build_action({"static": ctx.static, "history": list(ctx.history),
                                               "future_action": [step.frame()]}, ctx.current_state, 0)
        self.assertEqual(set(actual[0]), set(expected))
        for field in expected:
            self.assertTrue(torch.equal(actual[0][field], expected[field]), field)
        self.assertEqual(mappings[0], old_mapping)
        self.assertEqual(actual[0]["route_values"][0, 0].tolist(), [0.0, 1.0, 2.0, 0.0])
        self.assertEqual(actual[0]["comm_allocation_mask"][0, 0].tolist(), [True, True])
        self.assertEqual(float(actual[0]["comp_values"][0, 0, 0]), 2.5)
        self.assertEqual(actual[0]["mobility_values"][0, 0].tolist(), [0.25, -0.5, 3.0, 1.0])

    def test_noop_and_horizon(self):
        ctx = fixture()
        fallback = rule_fallback(ctx, 4)
        tensors, _ = compile_candidate(fallback, ctx, (ctx.current_state,) * 4)
        self.assertTrue(all(int(t["route_task_index"][0, 0]) == -1 and
                            t["comp_values"].shape[1] == 0 and
                            t["mobility_values"].shape[1] == 0 for t in tensors))
        for size in (0, 5):
            with self.assertRaises(ValueError):
                rule_fallback(ctx, size)

    def test_tristate_fixed_support_and_values(self):
        ctx = fixture()
        unknown = Constraint("dynamic_cpu", ConstraintStatus.UNKNOWN,
                             "NO_CAUSAL_SOURCE", "STEP_4_2A_AUDIT")
        candidate = sequence(ctx, CandidateActionStep(), (unknown,))
        self.assertEqual(candidate.constraint_status, ConstraintStatus.UNKNOWN)
        self.assertEqual(candidate.unresolved_constraints, (unknown,))
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            compile_candidate(candidate, ctx, (ctx.current_state,))
        with self.assertRaisesRegex(ValueError, "violated"):
            sequence(ctx, CandidateActionStep(), (Constraint("x", ConstraintStatus.VIOLATED, "X", "fixture"),))
        bad = CandidateActionStep(route=({"task_id": "task", "task_index": 0,
            "route_kind": "return", "target_node_index": 1, "route_node_indices": [1]},))
        with self.assertRaisesRegex(ValueError, "Return birth"):
            compile_candidate(sequence(ctx, bad), ctx, (ctx.current_state,), synthetic_contract_test_only=True)
        bad = CandidateActionStep(comp=({"node_id": "new", "task_id": "task", "allocated_cpu_per_s": 1},))
        with self.assertRaisesRegex(ValueError, "current support"):
            compile_candidate(sequence(ctx, bad), ctx, (ctx.current_state,), synthetic_contract_test_only=True)
        bad = CandidateActionStep(mob=({"uav_index": 1, "azimuth_rad": 0, "elevation_rad": 0, "speed_mps": 1},))
        with self.assertRaisesRegex(ValueError, "current UAV"):
            compile_candidate(sequence(ctx, bad), ctx, (ctx.current_state,), synthetic_contract_test_only=True)

    def test_backends_pool_warm_start(self):
        ctx = fixture()
        noop = rule_fallback(ctx, 2)
        learned = LearnedProposalBackend(lambda context, horizon, budget, seed:
            self._pool(CandidateActionSequence(noop.steps, "learned", Backend.LEARNED, seed,
                       context.causal_provenance, frozenset({"LEARNED"}))))
        pool = HybridCompositionBackend(learned, SearchStubBackend()).generate(ctx, 2, 3, 7)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool.candidates[0].source_tags, frozenset({"LEARNED", "SEARCH", "RULE_FALLBACK"}))
        self.assertEqual({x["backend"] for x in pool.candidates[0].generation_metadata["duplicate_sources"]},
                         {"LEARNED", "SEARCH", "RULE_FALLBACK"})
        self.assertEqual(pool.candidates[0].fingerprint, noop.fingerprint)
        first = CandidateActionStep(comp=({"task_id": "task", "node_id": "edge", "allocated_cpu_per_s": 1.0},))
        previous = CandidateActionSequence((first, CandidateActionStep()), "previous", Backend.SEARCH,
                                           7, ctx.causal_provenance)
        tail = CandidateActionStep(mob=({"uav_index": 0, "azimuth_rad": 0.0,
                                        "elevation_rad": 0.0, "speed_mps": 0.0},))
        shifted = shift_warm_start(previous, ctx, lambda _: tail, 7)
        self.assertEqual(shifted.steps, (CandidateActionStep(), tail))
        self.assertNotIn(first, shifted.steps)
        self.assertEqual(shifted.generation_metadata["parent_candidate_id"], previous.candidate_id)
        self.assertEqual(shifted.generation_metadata["shift_count"], 1)

    def test_identity_and_context_violation(self):
        ctx = fixture()
        route = {"task_id": "task", "task_index": 0, "route_kind": "offload",
                 "task_node_index": 0, "target_node_index": 99, "route_node_indices": [99]}
        with self.assertRaisesRegex(ValueError, "current support"):
            compile_candidate(sequence(ctx, CandidateActionStep(route=(route,))), ctx, (ctx.current_state,))
        violated_context = PlannerCandidateContext(ctx.static, ctx.history, ctx.current_state,
            ctx.causal_provenance, (Constraint("eligibility", ConstraintStatus.VIOLATED, "DENIED", "fixture"),))
        with self.assertRaisesRegex(ValueError, "violated"):
            compile_candidate(sequence(ctx, CandidateActionStep()), violated_context, (ctx.current_state,))
        self.assertEqual(rule_fallback(ctx, 1).fingerprint, SearchStubBackend().generate(ctx, 1, 1, 123).candidates[0].fingerprint)

    def test_comm_relation_rb_and_mobility_negatives(self):
        ctx = fixture()
        row = {"task_id": "task", "task_index": 0, "relation_index": 0, "rb_indices": [0]}
        ctx.current_state["comm_validity"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "Comm relation"):
            compile_candidate(sequence(ctx, CandidateActionStep(comm=(row,))), ctx, (ctx.current_state,))
        ctx.current_state["comm_validity"][0, 0] = True
        for invalid in ([0, 0], [2]):
            with self.assertRaisesRegex(ValueError, "RB assignment"):
                compile_candidate(sequence(ctx, CandidateActionStep(comm=({**row, "rb_indices": invalid},))),
                                  ctx, (ctx.current_state,))
        with self.assertRaisesRegex(ValueError, "duplicate comm"):
            compile_candidate(sequence(ctx, CandidateActionStep(comm=(row, row))), ctx, (ctx.current_state,))
        with self.assertRaises(ValueError):
            compile_candidate(sequence(ctx, CandidateActionStep(mob=({"uav_index": 0,
                "azimuth_rad": float("nan"), "elevation_rad": 0, "speed_mps": 1},))),
                ctx, (ctx.current_state,))

    @staticmethod
    def _pool(candidate):
        pool = CandidatePool()
        pool.add(candidate)
        return pool


if __name__ == "__main__":
    unittest.main()
