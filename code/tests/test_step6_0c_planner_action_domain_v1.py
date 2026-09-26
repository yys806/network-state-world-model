"""SYNTHETIC_CONTRACT_EVIDENCE; no simulator, model rollout or GPU."""
import sys
import json
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "src"), str(ROOT / "code" / "scripts")]

from build_step5_1d_unified_model_chain_v1 import build_action
from pi_jwm.step6_0a_candidate_generation_v1 import (
    Backend, CandidateActionSequence, CandidateActionStep, CandidatePool, ConstraintStatus, SearchStubBackend,
    PlannerCandidateContext, compile_candidate,
)
from pi_jwm.step6_0c_planner_action_domain_v1 import (
    CPU_POLICY, MOB_POLICY, PROFILES, PlannerComputeBudgetEvidence,
    admit_candidate_pool_v1, annotate_domain, context_from_current_raw, rule_fallback_v1,
    validate_domain_sequence,
)


def fixture(two_uavs=False, capacity=10.0, heading=1.0):
    physical = {"uav": 0, "edge": 1}
    if two_uavs:
        physical["uav2"] = 2
    size = len(physical)
    state = {
        "position": torch.zeros((1, size, 3)), "entity_presence": torch.ones((1, size), dtype=torch.bool),
        "uav_mask": torch.tensor([[True, False] + ([True] if two_uavs else [])]),
        "rb_active_mask": torch.zeros((1, 1, 2), dtype=torch.bool),
        "flow_task_index": torch.tensor([[0]]), "flow_presence": torch.tensor([[True]]),
        "flow_known": torch.tensor([[True]]), "flow_identity_index": torch.tensor([[1]]),
        "flow_comm_relation_index": torch.tensor([[0]]),
        "carrying_hop_source_index": torch.tensor([[0]]),
        "carrying_hop_destination_index": torch.tensor([[1]]),
        "comm_source_index": torch.tensor([[0]]), "comm_target_index": torch.tensor([[1]]),
        "comm_presence": torch.tensor([[True]]), "comm_validity": torch.tensor([[True]]),
    }
    static = {"input_entity_index": {"physical": physical, "task": {"task": 0, "task2": 1},
               "logical_flow": {"flow::task::Input::0": 0}},
              "agent_static_capability": [{"entity_id": node, "agent_index": slot,
                 "cpu_capacity_per_s": {"value": capacity, "normalized_value": 0.01,
                                        "observed_mask": capacity is not None}}
                 for node, slot in physical.items()]}
    base = PlannerCandidateContext(static, ({"tasks": [{"task_id": "task", "task_index": 0,
        "presence": True, "current_node_id": "uav"}], "logical_flows": []},), state, "synthetic-current")
    raw = {"node_cpu_capacity_observation_rows": [{"node_id": node, "capacity_per_s": capacity,
             "observed_mask": capacity is not None, "missing_reason": None if capacity is not None else "MISSING"}
             for node in physical],
           "entities": [{"entity_id": "uav", "entity_type": "uav", "heading": heading,
                          "heading_unit": "rad" if heading is not None else None, "elevation_rad": 0.3}]}
    if two_uavs:
        raw["entities"].append({"entity_id": "uav2", "entity_type": "uav", "heading": 2.0,
                                 "heading_unit": "rad", "elevation_rad": -0.1})
    return base, raw, context_from_current_raw(base, raw)


def seq(base, *steps):
    return CandidateActionSequence(tuple(steps), "c", Backend.SEARCH, 1, base.causal_provenance)


def mob(slot=0, heading=1.0, elevation=0.3, profile=0):
    _, delta, speed = PROFILES[profile]
    return {"uav_index": slot, "azimuth_rad": heading + delta,
            "elevation_rad": elevation, "speed_mps": speed}


def step(comp=(), mobs=None):
    return CandidateActionStep(comp=tuple(comp), mob=(mob(),) if mobs is None else tuple(mobs))


class PlannerActionDomainTests(unittest.TestCase):
    def test_cpu_budget_and_provenance(self):
        base, _, domain = fixture()
        self.assertEqual(domain.compute_budgets["edge"].unit, "AirFogSim CPU-work-unit/s")
        self.assertEqual(domain.compute_budgets["edge"].observation_provenance, domain.raw_decision_identity)
        self.assertFalse(domain.dynamic_available_cpu_available)
        self.assertFalse(domain.planner_requires_dynamic_available_cpu)
        for amounts, expected in [([5], ConstraintStatus.SATISFIED),
                                  ([4, 6], ConstraintStatus.SATISFIED),
                                  ([4, 6.01], ConstraintStatus.VIOLATED)]:
            rows = [{"node_id": "edge", "task_id": task, "allocated_cpu_per_s": amount}
                    for task, amount in zip(("task", "task2"), amounts)]
            result = validate_domain_sequence(seq(base, step(comp=rows)), base, domain)
            self.assertEqual(result.status, expected)
            self.assertIn(CPU_POLICY, result.constraints[0].constraint_name)
            self.assertIn(domain.raw_decision_identity, result.constraints[0].source)
        both = [{"node_id": "edge", "task_id": "task", "allocated_cpu_per_s": 9},
                {"node_id": "uav", "task_id": "task", "allocated_cpu_per_s": 9}]
        self.assertEqual(validate_domain_sequence(seq(base, step(comp=both)), base, domain).status,
                         ConstraintStatus.SATISFIED)

    def test_missing_capacity_and_raw_units(self):
        base, raw, domain = fixture(capacity=None)
        requested = {"node_id": "edge", "task_id": "task", "allocated_cpu_per_s": 1}
        result = validate_domain_sequence(seq(base, step(comp=(requested,))), base, domain)
        self.assertEqual(result.status, ConstraintStatus.VIOLATED)
        self.assertEqual(result.constraints[0].reason_code, "STATIC_CPU_CAPACITY_UNOBSERVED")
        self.assertEqual(validate_domain_sequence(seq(base, step()), base, domain).status,
                         ConstraintStatus.SATISFIED)
        with self.assertRaisesRegex(ValueError, "raw CPU"):
            PlannerComputeBudgetEvidence("edge", 0.01, True, unit="normalized")
        base.static["agent_static_capability"][0]["cpu_capacity_per_s"]["normalized_value"] = 9999
        raw["node_cpu_capacity_observation_rows"][0]["capacity_per_s"] = 10.0
        # A normalized placeholder cannot make an unobserved Raw source valid.
        raw["node_cpu_capacity_observation_rows"][0]["observed_mask"] = False
        self.assertFalse(context_from_current_raw(base, raw).compute_budgets["uav"].observed_mask)

    def test_negative_allocation_is_named_violation(self):
        base, _, domain = fixture()
        row = {"node_id": "edge", "task_id": "task", "allocated_cpu_per_s": -1}
        result = validate_domain_sequence(seq(base, step(comp=(row,))), base, domain)
        self.assertEqual(result.status, ConstraintStatus.VIOLATED)
        self.assertIn("NEGATIVE_COMP_ALLOCATION", [c.reason_code for c in result.constraints])

    def test_mobility_profiles_and_rejections(self):
        base, _, domain = fixture()
        for index in range(6):
            result = validate_domain_sequence(seq(base, step(mobs=(mob(profile=index),))), base, domain)
            self.assertEqual(result.status, ConstraintStatus.SATISFIED)
            self.assertEqual(result.constraints[-1].reason_code, PROFILES[index][0])
            self.assertIn(domain.raw_decision_identity, result.constraints[-1].source)
        bad = [dict(mob(), speed_mps=7.0), dict(mob(), azimuth_rad=1.15),
               dict(mob(), elevation_rad=0.4)]
        for row in bad:
            result = validate_domain_sequence(seq(base, step(mobs=(row,))), base, domain)
            self.assertEqual(result.status, ConstraintStatus.VIOLATED)
            self.assertEqual(result.constraints[-1].reason_code, "OUTSIDE_PLANNER_MOBILITY_CORE_DOMAIN_V1")
        self.assertEqual(validate_domain_sequence(seq(base, CandidateActionStep()), base, domain).status,
                         ConstraintStatus.VIOLATED)

    def test_missing_control_state_and_unit(self):
        base, raw, _ = fixture()
        raw["entities"][0]["heading"] = None
        missing = context_from_current_raw(base, raw)
        self.assertEqual(validate_domain_sequence(seq(base, step()), base, missing).status,
                         ConstraintStatus.UNKNOWN)
        raw["entities"][0]["heading"] = 1.0
        raw["entities"][0]["elevation_rad"] = None
        missing = context_from_current_raw(base, raw)
        self.assertEqual(validate_domain_sequence(seq(base, step()), base, missing).status,
                         ConstraintStatus.UNKNOWN)
        raw["entities"][0]["heading_unit"] = "degree"
        with self.assertRaisesRegex(ValueError, "unit must be rad"):
            context_from_current_raw(base, raw)
        raw["entities"][0]["entity_type"] = "vehicle"
        result = validate_domain_sequence(seq(base, step()), base, context_from_current_raw(base, raw))
        self.assertEqual(result.status, ConstraintStatus.UNKNOWN)

    def test_raw_anchor_and_unit_cannot_be_substituted(self):
        base, raw, _ = fixture()
        anchored = PlannerCandidateContext(base.static,
            ({"frame_index": 8, "simulation_time_s": 0.8},), base.current_state, base.causal_provenance)
        raw["frame_index"] = 9
        raw["simulation_time_s"] = 0.8
        with self.assertRaisesRegex(ValueError, "frame differs"):
            context_from_current_raw(anchored, raw)
        raw["frame_index"] = 8
        raw["simulation_time_s"] = 0.9
        with self.assertRaisesRegex(ValueError, "time differs"):
            context_from_current_raw(anchored, raw)
        raw["simulation_time_s"] = 0.8
        base.static["agent_static_capability"][0]["cpu_capacity_per_s"]["unit"] = "normalized"
        with self.assertRaisesRegex(ValueError, "raw CPU"):
            context_from_current_raw(anchored, raw)

    def test_multi_uav_joint_class_and_no_wrap(self):
        base, _, domain = fixture(two_uavs=True)
        same = seq(base, step(mobs=(mob(profile=2), mob(2, 2.0, -0.1, 2))))
        mixed = seq(base, step(mobs=(mob(profile=2), mob(2, 2.0, -0.1, 4))))
        self.assertEqual(validate_domain_sequence(same, base, domain).joint_behavior_support_class,
                         ("EXACT_COLLECTION_SHARED_PROFILE",))
        self.assertEqual(validate_domain_sequence(mixed, base, domain).joint_behavior_support_class,
                         ("PER_UAV_MARGINAL_SUPPORT_COMPOSITION",))
        annotated = annotate_domain(mixed, base, domain)
        self.assertEqual(annotated.constraint_status, ConstraintStatus.SATISFIED)
        self.assertEqual(annotated.unresolved_constraints, ())
        self.assertEqual(annotated.generation_metadata["joint_behavior_support_class"],
                         ["PER_UAV_MARGINAL_SUPPORT_COMPOSITION"])
        self.assertEqual(annotated.generation_metadata["domain_observation_provenance"], domain.raw_decision_identity)
        base2, _, domain2 = fixture(heading=3.1)
        candidate = seq(base2, step(mobs=(mob(heading=3.1, profile=5),)))
        self.assertAlmostEqual(validate_domain_sequence(candidate, base2, domain2).final_mobility_states[0].heading_rad, 3.3)

    def test_h4_side_state_and_fallback(self):
        base, _, domain = fixture()
        steps = []
        heading = 1.0
        for profile in (5, 4, 3, 0):
            command = mob(heading=heading, profile=profile)
            steps.append(step(mobs=(command,)))
            heading = command["azimuth_rad"]
        result = validate_domain_sequence(seq(base, *steps), base, domain)
        self.assertEqual(result.status, ConstraintStatus.SATISFIED)
        self.assertAlmostEqual(result.final_mobility_states[0].heading_rad, heading)
        fallback = rule_fallback_v1(base, domain, 4)
        self.assertEqual(fallback.generator_backend, Backend.RULE_FALLBACK)
        self.assertFalse(fallback.generation_metadata["safe"])
        self.assertTrue(all(len(s.mob) == 1 and s.mob[0]["speed_mps"] == 0 and not s.comp for s in fallback.steps))
        self.assertEqual(fallback.steps[0].mob[0]["azimuth_rad"], 1.0)
        self.assertEqual(fallback.generation_metadata["joint_behavior_support_class"],
                         ["EXACT_COLLECTION_SHARED_PROFILE"] * 4)
        self.assertNotEqual(fallback.fingerprint, seq(base, CandidateActionStep()).fingerprint)
        multi, _, multi_domain = fixture(two_uavs=True)
        multi_fallback = rule_fallback_v1(multi, multi_domain, 1)
        self.assertEqual(len(multi_fallback.steps[0].mob), 2)
        self.assertEqual({row["uav_index"] for row in multi_fallback.steps[0].mob}, {0, 2})

    def test_receipts_are_reproducible_and_isolated(self):
        from build_step6_0c_planner_action_domain_v1 import OUT, build
        for name, expected in build().items():
            self.assertEqual(json.loads((OUT / name).read_text(encoding="utf-8")), expected, name)
        receipt = build()["planner_action_domain_acceptance_receipt.json"]
        for key in ("step5_6b_remote_contacted", "ssh_used", "gpu_used", "checkpoint_consumed",
                    "formal_training_modified", "formal_training_config_modified", "formal_dataset_modified",
                    "world_model_rollout_performed", "planner_objective_computed", "locked_test_accessed"):
            self.assertFalse(receipt[key], key)

    def test_generic_backend_pool_is_admitted_with_explicit_hold(self):
        base, _, domain = fixture()
        generic = SearchStubBackend().generate(base, 1, 2, 7)
        admission = admit_candidate_pool_v1(generic, base, domain, 1, 7)
        self.assertEqual(len(admission.rejected), 1)
        self.assertIn("NO_MOBILITY_COMMAND_IS_NOT_HOLD", admission.rejected[0]["reason"])
        self.assertEqual(len(admission.pool), 1)
        only = admission.pool.candidates[0]
        self.assertEqual(only.generator_backend, Backend.RULE_FALLBACK)
        self.assertEqual(only.steps[0].mob[0]["speed_mps"], 0)
        source = CandidatePool()
        source.add(seq(base, step(mobs=(mob(profile=1),))))
        source.add(CandidateActionSequence((step(mobs=(mob(profile=1),)),), "learned_same",
            Backend.LEARNED, 2, base.causal_provenance, frozenset({"LEARNED"})))
        admitted = admit_candidate_pool_v1(source, base, domain, 1)
        self.assertEqual(len(admitted.rejected), 0)
        self.assertEqual(len(admitted.pool), 2)  # intervention + explicit HOLD

    def test_compiler_equivalence_and_gate(self):
        base, _, domain = fixture()
        candidate = rule_fallback_v1(base, domain, 1)
        self.assertEqual(candidate.constraint_status, ConstraintStatus.SATISFIED)
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            compile_candidate(candidate, base, (base.current_state,))
        actual, mapping = compile_candidate(candidate, base, (base.current_state,), planner_domain_context=domain)
        expected, old_mapping = build_action({"static": base.static, "history": list(base.history),
                                              "future_action": [candidate.steps[0].frame()]}, base.current_state, 0)
        self.assertEqual(set(actual[0]), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[0][name], expected[name]), name)
        self.assertEqual(mapping[0], old_mapping)
        self.assertEqual(actual[0]["mobility_values"][0, 0].tolist(), [1.0, 0.30000001192092896, 0.0, 1.0])
        self.assertEqual(actual[0]["comp_values"].shape[1], 0)
        bad = seq(base, step(mobs=(dict(mob(), speed_mps=7),)))
        with self.assertRaisesRegex(ValueError, "VIOLATED"):
            compile_candidate(bad, base, (base.current_state,), planner_domain_context=domain)

    def test_four_family_compiler_exact_equivalence(self):
        base, _, domain = fixture()
        action = CandidateActionStep(
            route=({"task_id": "task", "task_index": 0, "task_node_index": 0,
                    "target_node_index": 1, "route_node_indices": [1], "route_kind": "offload"},),
            comm=({"task_id": "task", "task_index": 0, "relation_index": 0, "rb_indices": [0, 1]},),
            comp=({"task_id": "task", "node_id": "edge", "allocated_cpu_per_s": 2.5},),
            mob=(mob(profile=1),))
        candidate = annotate_domain(seq(base, action), base, domain)
        actual, mapping = compile_candidate(candidate, base, (base.current_state,), planner_domain_context=domain)
        expected, old_mapping = build_action({"static": base.static, "history": list(base.history),
                                              "future_action": [action.frame()]}, base.current_state, 0)
        self.assertEqual(set(actual[0]), set(expected))
        self.assertEqual(len(actual[0]), 11)
        for name in expected:
            self.assertTrue(torch.equal(actual[0][name], expected[name]), name)
        self.assertEqual(mapping[0], old_mapping)
        self.assertEqual(float(actual[0]["comp_values"][0, 0, 0]), 2.5)
        self.assertAlmostEqual(float(actual[0]["mobility_values"][0, 0, 0]), 0.8)


if __name__ == "__main__":
    unittest.main()
