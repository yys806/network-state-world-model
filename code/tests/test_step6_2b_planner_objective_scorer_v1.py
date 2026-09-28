"""Observable CPU contract for the frozen Planner objective."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi_jwm.step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence, CandidateActionStep
from pi_jwm.step6_1_trained_candidate_rollout_v1 import CandidateRolloutTrace
from pi_jwm.step6_2a_planner_objective_side_state_v1 import PlannerObjectiveCausalSideState, PlannerTaskCausalSideState
from pi_jwm.step6_2b_planner_objective_scorer_v1 import (
    BurdenSemanticsBlocked, ScorerStateInconsistency, score_candidate_set, objective_sort_key,
)


def state(*, work=10., done=False, host=1):
    return {
        "task_presence": torch.tensor([[True]]), "task_completed": torch.tensor([[done]]),
        "task_work_remaining": torch.tensor([[work]]), "return_flow_index": torch.tensor([[-1]]),
        "flow_known": torch.zeros((1, 0), dtype=torch.bool),
        "flow_presence": torch.zeros((1, 0), dtype=torch.bool),
        "flow_task_index": torch.zeros((1, 0), dtype=torch.long),
        "task_agent_task_index": torch.tensor([[0]]),
        "task_agent_agent_index": torch.tensor([[host]]),
        "task_agent_relation_type_index": torch.tensor([[4]]),
        "task_agent_validity": torch.tensor([[True]]),
        "comm_presence": torch.tensor([[False]]), "comm_validity": torch.tensor([[False]]),
        "comm_wireless_mask": torch.tensor([[False]]),
        "csi_mask": torch.tensor([[[False]]]),
        "rb_active_mask": torch.zeros((1, 1, 1), dtype=torch.bool),
    }


def side(*, deadline=4., return_size=0.):
    task = PlannerTaskCausalSideState("q", 0, 0., 0., deadline, deadline,
                                      return_size, 1, False, False, 0.)
    return PlannerObjectiveCausalSideState(0., 0., (task,), (),
                                            (False, True, False), (None, 10., None), "test")


def pair(name, works, dones=None, *, states=None):
    dones = dones or [False] * len(works)
    noop = CandidateActionStep(comp=({"task_id": "q", "node_id": "host", "allocated_cpu_per_s": 0.},)) if name == "b" else CandidateActionStep()
    steps = (noop, *(CandidateActionStep() for _ in works[1:]))
    candidate = CandidateActionSequence(steps, name, Backend.RULE_FALLBACK, 0, "anchor",
                                         generation_metadata={"planner_action_domain": "V1"})
    trace = CandidateRolloutTrace(name, {"anchor": "same"}, (), (), (), (), (),
                                  tuple(states if states is not None else
                                        (state(work=w, done=d) for w, d in zip(works, dones))), (), ())
    return candidate, trace


def with_flow(row, *, remaining, present=True):
    row = dict(row)
    row.update({
        "flow_known": torch.tensor([[True]]),
        "flow_presence": torch.tensor([[present]]),
        "flow_task_index": torch.tensor([[0]]),
        "flow_type_index": torch.tensor([[2]]),
        "flow_remaining": torch.tensor([[remaining]]),
        "hop_remaining": torch.tensor([[remaining]]),
        "flow_status_index": torch.tensor([[2 if present else 3]]),
        "carrying_active": torch.tensor([[present]]),
        "route_node_mask": torch.tensor([[[True]]]),
        "route_node_indices": torch.tensor([[[1]]]),
        "current_hop_index": torch.tensor([[0]]),
        "flow_destination_index": torch.tensor([[1]]),
    })
    return row


class ObjectiveScorerTests(unittest.TestCase):
    def test_common_horizon_and_order_independence(self):
        a = pair("a", [9., 8., 7., 6.])
        b = pair("b", [9., 0., 0., 0.], [False, True, True, True])
        first = score_candidate_set(state(), side(), (a, b), slot_duration_s=1.)
        second = score_candidate_set(state(), side(), (b, a), slot_duration_s=1.)
        self.assertEqual(first.H_eff, 4)
        self.assertEqual(first.status, "SCOREABLE")
        self.assertEqual({x.candidate_id: x.objective_tuple for x in first.scores},
                         {x.candidate_id: x.objective_tuple for x in second.scores})
        self.assertEqual([x.candidate_id for x in first.scores], [x.candidate_id for x in second.scores])
        self.assertLess(first.scores[0].J_Delay, first.scores[1].J_Delay)

    def test_deadline_failure_latches_even_if_model_completes_later(self):
        late = pair("late", [9., 8., 0., 0.], [False, False, True, True])
        result = score_candidate_set(state(), side(deadline=1.), (late,), slot_duration_s=1.)
        score = result.scores[0]
        self.assertEqual(score.N_DDL, 1)
        self.assertEqual([row.deadline_violation for row in score.per_horizon_rows], [0, 1, 1, 1])
        self.assertEqual(score.J_Delay, 1.)

    def test_empty_cohort_is_unscoreable(self):
        result = score_candidate_set(state(done=True), side(), (pair("a", [10.] ),), slot_duration_s=1.)
        self.assertEqual(result.status, "OBJECTIVE_UNSCOREABLE_EMPTY_COHORT")

    def test_exact_lexicographic_priority(self):
        self.assertLess(objective_sort_key((0, .2, .8, 1., 1.), "z"),
                        objective_sort_key((1, 0., 0., 0., 0.), "a"))
        self.assertLess(objective_sort_key((0, 0., .2, .9, 1.), "z"),
                        objective_sort_key((0, 0., .3, 0., 0.), "a"))
        self.assertLess(objective_sort_key((0, 0., 0., 0., .1), "z"),
                        objective_sort_key((0, 0., 0., 0., .2), "a"))
        self.assertLess(objective_sort_key((0, 0., 0., 0., 0.), "a"),
                        objective_sort_key((0, 0., 0., 0., 0.), "z"))

    def test_support_horizon_is_shared_and_unsupported_states_excluded(self):
        a = pair("a", [9., 8., 7., 6.])
        b = pair("b", [9., 8., 0., 0.], states=[state(work=9.), state(work=8.),
                 state(work=0., host=2), state(work=0., host=2)])
        result = score_candidate_set(state(), side(return_size=1.), (a, b), slot_duration_s=1.)
        self.assertEqual(result.H_eff, 2)
        self.assertEqual(result.support_horizons, {"a": 4, "b": 2})
        self.assertTrue(all(len(score.per_horizon_rows) == 2 for score in result.scores))
        self.assertEqual([score.objective_tuple for score in result.scores],
                         [score.objective_tuple for score in
                          score_candidate_set(state(), side(return_size=1.), (b, a), slot_duration_s=1.).scores])
        at_h1 = pair("a", [0., 0., 0., 0.], states=[state(work=0., host=2)] * 4)
        zero = score_candidate_set(state(), side(return_size=1.), (at_h1,), slot_duration_s=1.)
        self.assertEqual(zero.status, "OBJECTIVE_UNSCOREABLE")
        self.assertEqual(zero.H_eff, 0)
        self.assertFalse(zero.scores)

    def test_deadline_completion_boundaries(self):
        timely = score_candidate_set(state(), side(deadline=1.),
                                     (pair("a", [0.], [True]),), slot_duration_s=1.)
        self.assertEqual(timely.scores[0].objective_tuple[:3], (0, 0., 0.))
        tolerance = score_candidate_set(state(), side(deadline=1.),
                                        (pair("a", [0.], [True]),), slot_duration_s=1.000005)
        self.assertEqual(tolerance.scores[0].N_DDL, 0)
        return_state = state(work=0., done=True)
        return_state.update({"return_flow_index": torch.tensor([[0]]),
                             "flow_known": torch.tensor([[True]]),
                             "flow_task_index": torch.tensor([[0]]),
                             "flow_type_index": torch.tensor([[3]])})
        strict = score_candidate_set(state(), side(deadline=1., return_size=1.),
                                     (pair("a", [0.], states=[return_state]),),
                                     slot_duration_s=1.000005)
        self.assertEqual(strict.scores[0].N_DDL, 1)

    def test_burden_partial_service_completion_and_no_mask_block(self):
        anchor = with_flow(state(), remaining=10.)
        partial = pair("a", [10., 10.], states=[with_flow(state(), remaining=8.),
                                               with_flow(state(), remaining=4.)])
        done = pair("a", [10., 10.], states=[with_flow(state(), remaining=8.),
                                            with_flow(state(), remaining=0., present=False)])
        left = score_candidate_set(anchor, side(), (partial,), slot_duration_s=1.).scores[0]
        right = score_candidate_set(anchor, side(), (done,), slot_duration_s=1.).scores[0]
        self.assertLess(right.J_Burden, left.J_Burden)
        self.assertEqual(right.per_horizon_rows[1].tasks[0]["tx_burden"], 0.)
        with self.assertRaisesRegex(BurdenSemanticsBlocked, "BURDEN_SEMANTICS_BLOCKED"):
            score_candidate_set(state(work=0.), side(), (pair("a", [0.]),), slot_duration_s=1.)
        less_cpu = score_candidate_set(state(), side(), (pair("a", [9., 8.]),), slot_duration_s=1.)
        more_cpu = score_candidate_set(state(), side(), (pair("a", [5., 4.]),), slot_duration_s=1.)
        self.assertLess(more_cpu.scores[0].J_Burden, less_cpu.scores[0].J_Burden)

    def test_future_truth_mutation_does_not_change_score(self):
        candidate, trace = pair("a", [9., 8.])
        original = score_candidate_set(state(), side(), ((candidate, trace),), slot_duration_s=1.)
        changed = tuple({**row, "future_target": torch.tensor([[999.]]),
                         "failed_label": torch.tensor([[True]])} for row in trace.states)
        altered = CandidateRolloutTrace(trace.candidate_id, trace.initial_fingerprints,
                                        trace.input_fingerprints, trace.output_fingerprints,
                                        trace.actions, trace.mappings, trace.latents, changed,
                                        trace.graphs, trace.model_traces)
        audit = score_candidate_set(state(), side(), ((candidate, altered),), slot_duration_s=1.)
        self.assertEqual(original.scores[0].objective_tuple, audit.scores[0].objective_tuple)
        revised = tuple({**row, "flow_route_revision": torch.tensor([[999]])} for row in trace.states)
        revised_trace = CandidateRolloutTrace(trace.candidate_id, trace.initial_fingerprints,
                                             trace.input_fingerprints, trace.output_fingerprints,
                                             trace.actions, trace.mappings, trace.latents, revised,
                                             trace.graphs, trace.model_traces)
        revision_score = score_candidate_set(state(), side(), ((candidate, revised_trace),), slot_duration_s=1.)
        self.assertEqual(original.scores[0].J_Effort, revision_score.scores[0].J_Effort)

    def test_effort_is_last_and_anchor_applicability_does_not_disappear(self):
        from dataclasses import replace
        base = state()
        base["comm_presence"] = torch.tensor([[True]])
        base["comm_validity"] = torch.tensor([[True]])
        base["comm_wireless_mask"] = torch.tensor([[True]])
        base["csi_mask"] = torch.tensor([[[True, True, True, True]]])
        base["rb_active_mask"] = torch.zeros((1, 1, 4), dtype=torch.bool)
        frozen = replace(side(), effort_component_mask=(True, True, True),
                         effort_denominators=(4., 10., 15.))
        quiet, quiet_trace = pair("a", [9.])
        loud_step = CandidateActionStep(
            comm=({"relation_index": 0, "rb_indices": [0, 1]},),
            comp=({"task_id": "q", "node_id": "host", "allocated_cpu_per_s": 2.},),
            mob=({"uav_index": 1, "speed_mps": 5.},))
        loud = CandidateActionSequence((loud_step,), "loud", Backend.RULE_FALLBACK, 0,
                                        "anchor", generation_metadata={"planner_action_domain": "V1"})
        loud_trace = replace(quiet_trace, candidate_id="loud")
        result = score_candidate_set(base, frozen, ((loud, loud_trace), (quiet, quiet_trace)),
                                     slot_duration_s=1.)
        self.assertEqual(result.scores[0].candidate_id, "a")
        self.assertEqual(result.scores[0].J_Effort, 0.)
        self.assertEqual(result.scores[1].per_horizon_rows[0].effort_components,
                         {"Comm": .5, "Comp": .2, "Mob": 5. / 15.})
        self.assertEqual(result.scores[0].objective_tuple[:4], result.scores[1].objective_tuple[:4])
        self.assertEqual(len(result.scores[0].per_horizon_rows[0].effort_components), 3)

    def test_pending_route_with_no_flow_is_not_silently_scored(self):
        from dataclasses import replace
        candidate, trace = pair("a", [9.])
        route_step = CandidateActionStep(route=({"task_id": "q", "task_index": 0,
            "target_node_index": 1, "route_node_indices": [1], "route_kind": "offload"},))
        candidate = replace(candidate, steps=(route_step,))
        trace = replace(trace, mappings=({"route": [{"mode": "pending_flow", "flow_index": -1}]},))
        with self.assertRaisesRegex(ScorerStateInconsistency,
                                    "UNSUPPORTED_PENDING_FLOW_ROUTE_OBJECTIVE"):
            score_candidate_set(state(), side(), ((candidate, trace),), slot_duration_s=1.)


if __name__ == "__main__":
    unittest.main()
