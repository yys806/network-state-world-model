"""Bounded synthetic causal-state checks for STEP 6.3B; no model or optimizer."""
import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pi_jwm.step6_0a_candidate_generation_v1 import (
    Backend, CandidateActionSequence, CandidateActionStep, PlannerCandidateContext,
)
from pi_jwm.step6_0c_planner_action_domain_v1 import (
    PlannerActionDomainContext, PlannerComputeBudgetEvidence,
    PlannerMobilityControlState,
)
from pi_jwm.step6_3b_candidate_grammar_v1 import (
    CandidateGrammarViolation, CommBlockChoice, StructuredStepChoice,
    admit_structured_candidate, bind_structured_step,
)
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog


def fixture():
    state = {
        "entity_presence": torch.tensor([[True, True, True, True]]),
        "uav_mask": torch.tensor([[True, True, False, False]]),
        "task_presence": torch.tensor([[True, True]]),
        "task_completed": torch.tensor([[False, False]]),
        "task_lifecycle_index": torch.tensor([[4, 5]]),
        "task_work_remaining": torch.tensor([[2.0, 1.0]]),
        "task_agent_validity": torch.tensor([[True]]),
        "task_agent_task_index": torch.tensor([[1]]),
        "task_agent_agent_index": torch.tensor([[2]]),
        "task_agent_relation_type_index": torch.tensor([[4]]),
        "flow_known": torch.tensor([[True]]),
        "flow_presence": torch.tensor([[True]]),
        "carrying_active": torch.tensor([[True]]),
        "flow_task_index": torch.tensor([[0]]),
        "flow_comm_relation_index": torch.tensor([[0]]),
        "carrying_hop_source_index": torch.tensor([[3]]),
        "carrying_hop_destination_index": torch.tensor([[2]]),
        "comm_presence": torch.tensor([[True]]),
        "comm_validity": torch.tensor([[True]]),
        "comm_wireless_mask": torch.tensor([[True]]),
        "comm_source_index": torch.tensor([[3]]),
        "comm_target_index": torch.tensor([[2]]),
        "rb_active_mask": torch.zeros((1, 1, 50), dtype=torch.bool),
    }
    static = {"input_entity_index": {"physical": {"UAV_0": 0, "UAV_1": 1,
                                                   "RSU_0": 2, "vehicle_0": 3},
                                     "task": {"input": 0, "compute": 1},
                                     "logical_flow": {"flow::input::Input::0": 0}}}
    context = PlannerCandidateContext(static, (), state, "synthetic-current")
    control = {
        0: PlannerMobilityControlState("UAV_0", 0, 0.5, 0.1, True, True),
        1: PlannerMobilityControlState("UAV_1", 1, 1.0, 0.2, True, True),
    }
    domain = PlannerActionDomainContext(
        {"RSU_0": PlannerComputeBudgetEvidence("RSU_0", 10.0, True)},
        control, context.causal_provenance)
    catalog = TrainStructuralSupportCatalog.from_json(
        ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json")
    return context, domain, state, control, catalog


class GrammarTests(unittest.TestCase):
    def test_multirow_cyclic_comm_comp_scale_and_shared_hold(self):
        context, domain, state, control, catalog = fixture()
        choice = StructuredStepChoice((CommBlockChoice("input", 0, 1),
                                       CommBlockChoice("input", 49, 2)),
                                      "SCALE_0.5", "PROFILE_HOLD")
        bound = bind_structured_step(choice, context, domain, state, control, catalog)
        self.assertEqual(bound.structural_signature, ("rows:1,2", "tasks:1;alpha:0.5", "HOLD,HOLD"))
        self.assertTrue(bound.support.formal_pool_admitted)
        self.assertEqual([list(row["rb_indices"]) for row in bound.action.comm], [[0], [0, 49]])
        self.assertEqual([row["relation_index"] for row in bound.action.comm], [0, 0])
        self.assertEqual(bound.action.comp[0]["allocated_cpu_per_s"], 5.0)
        self.assertEqual([row["speed_mps"] for row in bound.action.mob], [0.0, 0.0])
        self.assertEqual(bound.action.route, ())

    def test_next_state_rebinds_and_recomputes_cpu_base(self):
        context, domain, state, control, catalog = fixture()
        first_choice = StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                                            "SCALE_1.0", "PROFILE_1")
        first = bind_structured_step(first_choice, context, domain, state, control, catalog)
        next_state = copy.deepcopy(state)
        next_state["flow_presence"][0, 0] = False
        next_state["carrying_active"][0, 0] = False
        next_state["task_work_remaining"][0, 1] = 0.5
        second = bind_structured_step(StructuredStepChoice((), "SCALE_0.5", "PROFILE_HOLD"),
                                      context, domain, next_state, first.next_mobility_control,
                                      catalog, (first.structural_signature,))
        self.assertEqual(second.action.comm, ())
        self.assertEqual(second.action.comp[0]["allocated_cpu_per_s"], 2.5)
        self.assertEqual(second.action.mob[0]["azimuth_rad"], first.action.mob[0]["azimuth_rad"])
        candidate = CandidateActionSequence((first.action, second.action), "two-step",
                                            Backend.SEARCH, 7, context.causal_provenance)
        admission = admit_structured_candidate(candidate, context, domain,
                                               (state, next_state), catalog)
        self.assertTrue(admission.admitted, admission.reason_codes)
        self.assertEqual(len(admission.per_step), 2)
        self.assertIsNone(admission.h_sup)
        with self.assertRaisesRegex(CandidateGrammarViolation, "COMM_TASK_COVERAGE"):
            bind_structured_step(first_choice, context, domain, next_state,
                                 first.next_mobility_control, catalog)

    def test_rejects_unseen_rb_pair_and_unbound_task(self):
        context, domain, state, control, catalog = fixture()
        for choice, reason in (
            (StructuredStepChoice((CommBlockChoice("input", 0, 2),), "SCALE_1.0", "PROFILE_HOLD"),
             "COMM_START_WIDTH_PAIR_UNSEEN_IN_TRAIN"),
            (StructuredStepChoice((CommBlockChoice("missing", 0, 1),), "SCALE_1.0", "PROFILE_HOLD"),
             "COMM_TASK_COVERAGE"),
        ):
            with self.assertRaisesRegex(CandidateGrammarViolation, reason):
                bind_structured_step(choice, context, domain, state, control, catalog)

    def test_backend_independent_admission_and_rejections(self):
        context, domain, state, control, catalog = fixture()
        choice = StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                                      "SCALE_1.0", "PROFILE_HOLD")
        bound = bind_structured_step(choice, context, domain, state, control, catalog)

        def verdict(step):
            candidate = CandidateActionSequence((step,), "test", Backend.SEARCH, 1,
                                                context.causal_provenance)
            return admit_structured_candidate(candidate, context, domain, (state,), catalog)

        valid = verdict(bound.action)
        self.assertTrue(valid.admitted)
        self.assertIsNone(valid.h_sup)
        self.assertEqual(valid.per_step[0].structural_signature, bound.structural_signature)
        bad_relation = dict(bound.action.comm[0], relation_index=42)
        self.assertIn("COMM_CAUSAL_RELATION_BINDING_DIFFERS", verdict(
            CandidateActionStep(comm=(bad_relation,), comp=bound.action.comp,
                                mob=bound.action.mob)).reason_codes)
        self.assertIn("ROUTE_MUST_BE_EXPLICIT_NOOP", verdict(
            CandidateActionStep(route=({"flow_id": "anything"},), comm=bound.action.comm,
                                comp=bound.action.comp, mob=bound.action.mob)).reason_codes)
        self.assertIn("COMM_TASK_COVERAGE", verdict(
            CandidateActionStep(comp=bound.action.comp, mob=bound.action.mob)).reason_codes[0])
        self.assertIn("ELIGIBLE_COMP_REQUIRES_TRAIN_ALPHA", verdict(
            CandidateActionStep(comm=bound.action.comm, mob=bound.action.mob)).reason_codes)
        asym = list(bound.action.mob)
        asym[1] = dict(asym[1], azimuth_rad=asym[1]["azimuth_rad"] - 0.2,
                       speed_mps=5.0)
        self.assertIn("ASYMMETRIC_MOBILITY_OUTSIDE_PLANNER_V1", verdict(
            CandidateActionStep(comm=bound.action.comm, comp=bound.action.comp,
                                mob=tuple(asym))).reason_codes)

    def test_comp_modes_follow_causal_base_and_reject_mixed_alpha(self):
        context, domain, state, control, catalog = fixture()
        state["flow_presence"][0, 0] = False
        state["carrying_active"][0, 0] = False
        state["task_lifecycle_index"][0, 0] = 5
        state["task_agent_validity"] = torch.tensor([[True, True]])
        state["task_agent_task_index"] = torch.tensor([[0, 1]])
        state["task_agent_agent_index"] = torch.tensor([[2, 2]])
        state["task_agent_relation_type_index"] = torch.tensor([[4, 4]])
        for alpha in (0.5, 0.75, 1.0):
            bound = bind_structured_step(StructuredStepChoice((), f"SCALE_{alpha}",
                                                             "PROFILE_HOLD"),
                                         context, domain, state, control, catalog)
            self.assertEqual(len(bound.action.comp), 2)
            self.assertEqual(bound.structural_signature[1], f"tasks:2;alpha:{alpha}")
        mixed = list(bound.action.comp)
        mixed[0] = dict(mixed[0], allocated_cpu_per_s=mixed[0]["allocated_cpu_per_s"] * 0.5)
        candidate = CandidateActionSequence((CandidateActionStep(comp=tuple(mixed),
                                      mob=bound.action.mob),), "mixed", Backend.SEARCH, 1,
                                      context.causal_provenance)
        result = admit_structured_candidate(candidate, context, domain, (state,), catalog)
        self.assertFalse(result.admitted)
        self.assertIn("COMP_MIXED_OR_OUTSIDE_TRAIN_ALPHA", result.reason_codes)

    def test_comp_empty_only_without_computing_task(self):
        context, domain, state, control, catalog = fixture()
        with self.assertRaisesRegex(CandidateGrammarViolation, "ELIGIBLE_COMP_REQUIRES_TRAIN_ALPHA"):
            bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                                  "NOOP", "PROFILE_HOLD"), context, domain, state,
                                 control, catalog)
        state["task_lifecycle_index"][0, 1] = 4
        bound = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                                      "NOOP", "PROFILE_HOLD"), context, domain, state,
                                     control, catalog)
        self.assertEqual(bound.action.comp, ())

    def test_comm_width_three_reuse_and_invalid_width(self):
        context, domain, state, control, catalog = fixture()
        for name, value in (
            ("flow_known", True), ("flow_presence", True), ("carrying_active", True),
            ("comm_presence", True), ("comm_validity", True), ("comm_wireless_mask", True),
        ):
            state[name] = torch.tensor([[value, value]])
        for name, values in (
            ("flow_task_index", [0, 1]), ("flow_comm_relation_index", [0, 1]),
            ("carrying_hop_source_index", [3, 3]),
            ("carrying_hop_destination_index", [2, 2]),
            ("comm_source_index", [3, 3]), ("comm_target_index", [2, 2]),
        ):
            state[name] = torch.tensor([values])
        state["rb_active_mask"] = torch.zeros((1, 2, 50), dtype=torch.bool)
        choice = StructuredStepChoice((CommBlockChoice("input", 49, 3),
                                       CommBlockChoice("compute", 49, 3)),
                                      "SCALE_1.0", "PROFILE_HOLD")
        bound = bind_structured_step(choice, context, domain, state, control, catalog)
        self.assertEqual([row["rb_indices"] for row in bound.action.comm], [[0, 1, 49]] * 2)
        self.assertEqual([row["relation_index"] for row in bound.action.comm], [1, 0])
        with self.assertRaisesRegex(CandidateGrammarViolation, "COMM_BLOCK_OUTSIDE_TRAIN_CORE"):
            bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 49, 4),
                            CommBlockChoice("compute", 49, 3)), "SCALE_1.0", "PROFILE_HOLD"),
                                 context, domain, state, control, catalog)

    def test_no_present_uav_has_one_empty_mobility_choice(self):
        context, domain, state, control, catalog = fixture()
        state["uav_mask"] = torch.tensor([[False, False, False, False]])
        with self.assertRaisesRegex(CandidateGrammarViolation,
                                    "MOBILITY_PROFILE_WITHOUT_PRESENT_UAV"):
            bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),),
                               "SCALE_1.0", "PROFILE_1"), context, domain, state,
                                 control, catalog)

    def test_noncanonical_multirow_order_rejected_for_stable_identity(self):
        context, domain, state, control, catalog = fixture()
        bound = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", 0, 1),
                                      CommBlockChoice("input", 49, 2)), "SCALE_0.5",
                                     "PROFILE_HOLD"), context, domain, state, control, catalog)
        reversed_rows = CandidateActionStep(comm=tuple(reversed(bound.action.comm)),
                                            comp=bound.action.comp, mob=bound.action.mob)
        candidate = CandidateActionSequence((reversed_rows,), "order", Backend.SEARCH, 1,
                                            context.causal_provenance)
        admission = admit_structured_candidate(candidate, context, domain,
                                               (state,), catalog)
        self.assertEqual(admission.reason_codes,
                         ("CANDIDATE_ACTION_NOT_CANONICAL_GRAMMAR_OUTPUT",))


if __name__ == "__main__":
    unittest.main()
