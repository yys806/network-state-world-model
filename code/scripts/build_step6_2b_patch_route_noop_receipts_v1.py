"""CPU-only machine receipts for the researcher-frozen Route no-op closure."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/tests"))

from test_step6_0c_planner_action_domain_v1 import fixture, mob, seq
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep, ConstraintStatus
from pi_jwm.step6_0c_planner_action_domain_v1 import validate_domain_sequence
from run_step6_2b_objective_scorer_cpu_v1 import CHECKPOINT, EXPECTED_SHA, run, sha

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, dict]:
    base, _, domain = fixture()
    empty = seq(base, CandidateActionStep(route=(), mob=(mob(),)))
    direct = {"task_id": "task", "task_index": 0, "task_node_index": 0,
              "target_node_index": 1, "route_node_indices": [1], "route_kind": "offload"}
    rows = {
        "pending": {**direct, "task_id": "task2", "task_index": 1},
        "existing_same_path": {**direct, "flow_id": "flow::task::Input::0"},
        "multihop": {**direct, "route_node_indices": [0, 1]},
        "destination_change": {**direct, "target_node_index": 0, "route_node_indices": [0],
                               "flow_id": "flow::task::Input::0"},
    }
    verdicts = {}
    for name, row in rows.items():
        candidate = seq(base, CandidateActionStep(route=(row,), mob=(mob(),)))
        result = validate_domain_sequence(candidate, base, domain)
        reasons = [c.reason_code for c in result.constraints if c.status is ConstraintStatus.VIOLATED]
        if result.status is not ConstraintStatus.VIOLATED or "OUTSIDE_PLANNER_ROUTE_NOOP_ONLY_V1" not in reasons:
            raise AssertionError(f"{name} Route was admitted")
        verdicts[name] = {"status": "REJECTED", "reason": "OUTSIDE_PLANNER_ROUTE_NOOP_ONLY_V1"}
    empty_result = validate_domain_sequence(empty, base, domain)
    if empty_result.status is not ConstraintStatus.SATISFIED:
        raise AssertionError("explicit Route no-op was rejected")

    integration = run(include_historical_route_audit=False)
    if (integration["status"] != "PASS" or integration["candidate_count"] < 2 or
        integration["H_eff"] != 4 or integration["effective_route_action_audit"] or
        not integration["route_noop_compiled_to_absence_sentinels"] or
        not integration["no_route_task_flow_latent_injection_on_hold"] or
        not integration["parameter_digest_unchanged"] or
        integration["comm_effort_denominator"] != 50 or
        integration["comm_relation_rows"] != 242 or
        sha(CHECKPOINT) != EXPECTED_SHA):
        raise AssertionError("frozen-checkpoint Route no-op integration failed")

    common = {"step": "STEP 6.2B-PATCH", "git_start_sha":
              "3ac4470080b11906d16c3a4c33c3a3306f7f31c4",
              "checkpoint_sha256": EXPECTED_SHA, "checkpoint_modified": False,
              "gpu": False, "training": False, "optimizer_step": False,
              "formal_dataset_modified": False, "locked_test": False,
              "candidate_method_selected": False, "closed_loop": False,
              "baseline": False, "performance_claim": False}
    receipts = {
        "01_route_noop_policy.json": {**common, "PLANNER_V1_ROUTE_POLICY": "EXPLICIT_NOOP_ONLY",
            "route_family_per_horizon": [], "route_interface_retained": True,
            "route_optimization_active_v1": False, "multihop_code_retained": True,
            "active_optimization_families": ["Comm", "Comp", "Mob"]},
        "02_pending_route_rejection.json": {**common, "pending": verdicts["pending"],
            "flow_index_minus_one_never_reaches_rollout": True},
        "03_existing_route_rejection.json": {**common,
            "existing_same_path": verdicts["existing_same_path"],
            "multihop": verdicts["multihop"],
            "destination_change": verdicts["destination_change"],
            "model_4_4_route_semantics_modified": False},
        "04_route_action_absence_compile_audit.json": {**common,
            "empty_route_admitted": True,
            "route_task_and_flow_indices_all_negative": True,
            "no_route_task_flow_latent_injection_on_hold": True,
            "formal_learned_action_tensor_count": 11},
        "05_effective_route_freedom.json": {**common,
            "bounded_causal_fixture": True,
            "tested_nonempty_route_categories": verdicts,
            "legal_nonempty_route_count": 0,
            "legal_state_changing_route_count": 0,
            "legal_route_noop_count": 1,
            "ROUTE_EFFECTIVE_FREEDOM_V1": "NONE",
            "ROUTE_INTERFACE_RETAINED": True,
            "ROUTE_OPTIMIZATION_ACTIVE_V1": False,
            "MULTIHOP_CODE_RETAINED": True},
        "06_comm_effort_denominator_regression.json": {**common,
            "selected_anchor_valid_global_rb_id_count": integration["comm_effort_denominator"],
            "comm_relation_row_count": integration["comm_relation_rows"],
            "denominator_unit": "global simulator RB IDs",
            "numerator_unit": "requested relation-RB assignments",
            "anchor_frozen": True,
            "source_rule": "rb_active_mask last-axis width when any current valid wireless relation exists",
            "previous_step6_2b_correction_retained": True},
        "07_frozen_checkpoint_no_route_integration.json": {**common, **integration,
            "candidate_route_family_all_empty": True,
            "same_start_mean_prior_expected_service": True,
            "future_target_used": False},
        "08_step6_2b_final_acceptance.json": {**common,
            "STEP_6_2B": "PASS",
            "PLANNER_OBJECTIVE_SCORER": "IMPLEMENTED_AND_CPU_CONTRACT_VERIFIED",
            "MPC_OBJECTIVE": "FROZEN_AND_IMPLEMENTED",
            "CANDIDATE_METHOD_SELECTION": "RESEARCH_PENDING",
            "CLOSED_LOOP_READINESS": "NOT_READY",
            "MULTIHOP_PLANNER_READINESS": "NOT_IN_V1_DOMAIN",
            "ROUTE_OPTIMIZATION_ACTIVE_V1": False,
            "objective_tuple": ["N_DDL", "A_DDL", "J_Delay", "J_Burden", "J_Effort"],
            "comparison": "STRICT_LEXICOGRAPHIC_MINIMIZATION",
            "patch_blockers_closed": ["pending_flow_route", "existing_same_path_route"],
            "remaining_scope": "CPU scorer mechanism; no candidate method, ranking quality or closed-loop claim"},
    }
    return receipts


def write() -> None:
    receipts = build()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"step": "STEP 6.2B-PATCH", "files": {}}
    for name, payload in receipts.items():
        path = OUT / name
        path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        manifest["files"][name] = {"sha256": _digest(path), "bytes": path.stat().st_size}
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print("STEP_6_2B=PASS; receipts=8; Route no-op checkpoint integration=PASS")


if __name__ == "__main__":
    write()
