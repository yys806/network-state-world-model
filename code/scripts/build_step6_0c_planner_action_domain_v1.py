"""Write/check CPU-only Planner v1 action-domain contract receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_0c_planner_action_domain_v1_20260926"
SOURCE_SHA = "1c24fc4b4f919b06352ad86f94174dc0020682cb"
P = "code/artifacts/protocols/pi_jwm_step6_0b_planner_action_feasibility_audit_v1_20260926/planner_action_feasibility_source_audit.json"


def _sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def build() -> dict[str, dict]:
    prior = json.loads((ROOT / P).read_text(encoding="utf-8"))
    source_hashes = {
        "raw_adapter": _sha("code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py"),
        "formal_collector": _sha("code/scripts/collect_step5_5_formal_raw_v1.py"),
        "formal_action_adapter": _sha("code/scripts/build_step5_1d_unified_model_chain_v1.py"),
        "generic_candidate": _sha("code/src/pi_jwm/step6_0a_candidate_generation_v1.py"),
        "planner_domain": _sha("code/src/pi_jwm/step6_0c_planner_action_domain_v1.py"),
    }
    provenance = {
        "pi_jwm_start_sha": SOURCE_SHA,
        "step6_0b_receipt": P,
        "step6_0b_receipt_sha256": _sha(P),
        "step6_0b_airfogsim_local_relevant_source_sha256": prior["airfogsim_local_relevant_source_sha256"],
        "airfogsim_source_sha": None,
        "airfogsim_git_identity_note": "Local third-party directory has no independent Git metadata; 6.0B records inspected source hashes.",
        "source_file_sha256": source_hashes,
    }
    compute = {
        "schema": "PIJWM_PLANNER_COMPUTE_BUDGET_POLICY_V1",
        "policy": "STATIC_PER_SLOT_BUDGET_V1",
        "source": "current Raw decision.node_cpu_capacity_observation_rows; cross-check causal Sample static.agent_static_capability.value",
        "tensor_raw_correspondence": "agent_cpu_capacity_raw + agent_cpu_capacity_mask",
        "tensor_normalized_not_used": "agent_cpu_capacity",
        "graph_correspondence": "agent_nodes.cpu_capacity + mask",
        "unit": "AirFogSim CPU-work-unit/s",
        "formula": "for each current node n and decision slot t: sum_i allocated_cpu_per_s(i,n,t) <= observed C_static(n)",
        "missing_capacity_nonzero_request": "VIOLATED: STATIC_CPU_CAPACITY_UNOBSERVED",
        "missing_capacity_no_comp_request": "SATISFIED",
        "dynamic_available_cpu_available": False,
        "planner_requires_dynamic_available_cpu": False,
        "simulator_native_capacity_enforcement": False,
        "provenance": provenance,
    }
    mobility = {
        "schema": "PIJWM_FORMAL_DATASET_MOBILITY_CORE_DOMAIN_V1",
        "policy": "FORMAL_DATASET_MOBILITY_CORE_DOMAIN_V1",
        "source": "code/scripts/collect_step5_5_formal_raw_v1.py; profile=(frame+policy_seed)%5",
        "source_status_before_decision": "DATASET_BEHAVIOR_SUPPORT_ONLY",
        "status_after_researcher_decision": "PLANNER_V1_OPERATIONAL_DOMAIN",
        "profiles": [{"name": name, "delta_azimuth_rad": angle, "delta_elevation_rad": 0.0, "speed_mps": speed}
                     for name, angle, speed in (("PROFILE_HOLD", 0.0, 0.0), ("PROFILE_1", -0.2, 5.0),
                         ("PROFILE_2", -0.1, 8.0), ("PROFILE_3", 0.05, 10.0),
                         ("PROFILE_4", 0.1, 12.0), ("PROFILE_5", 0.2, 15.0))],
        "current_control_side_state": "current Raw entities UAV heading(rad), elevation_rad; Planner-only, no World Model feature change",
        "state_update": "next heading=command azimuth; next elevation=command elevation; no physical prediction",
        "hold": "one explicit speed=0 row for every current UAV; empty mobility is NO_MOBILITY_COMMAND",
        "angle_wrap_or_clamp": False,
        "joint_behavior_support_classes": ["EXACT_COLLECTION_SHARED_PROFILE", "PER_UAV_MARGINAL_SUPPORT_COMPOSITION"],
        "simulator_hard_bound": False,
        "safety_or_geofence_claim": False,
        "provenance": provenance,
    }
    contract = {
        "schema": "PIJWM_PLANNER_ACTION_DOMAIN_V1",
        "status": "FROZEN_RESEARCHER_OPERATIONAL_DOMAIN_CPU_CONTRACT",
        "compute_policy": compute["policy"],
        "mobility_policy": mobility["policy"],
        "route_comm": "6.0A fixed-support and current relation validation unchanged",
        "horizon_min": 1, "horizon_max": 4,
        "formal_action_tensor_count": 11,
        "candidate_generation_final_method": "RESEARCH_PENDING",
        "final_optimal_action_space": False,
        "provenance": provenance,
    }
    fallback = {
        "schema": "PIJWM_RULE_FALLBACK_V1_RECEIPT",
        "route": "no-op", "comm": "no-op", "comp": "no allocation",
        "mobility": "explicit PROFILE_HOLD row for every current present UAV",
        "requires_current_uav_heading_elevation": True,
        "safe": False, "future_feasibility_claim": False,
        "evidence_kind": "SYNTHETIC_CONTRACT_EVIDENCE",
    }
    receipt = {
        "schema": "PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_ACCEPTANCE_V1",
        "status": "CPU_CONTRACT_TESTED",
        "evidence_kind": "SYNTHETIC_CONTRACT_EVIDENCE",
        "focused_test_file": "code/tests/test_step6_0c_planner_action_domain_v1.py",
        "focused_test_count": 12,
        "focused_test_status": "PASSED",
        "four_family_exact_formal_tensor_equivalence_test": True,
        "multi_step_control_side_horizon": 4,
        "dynamic_available_cpu_available": False,
        "planner_requires_dynamic_available_cpu": False,
        "unknown_constraints_remaining": ["dynamic_available_cpu as simulator fact; not Planner v1 legality input",
                                          "spatial_workspace_constraint", "geofence_safety"],
        "step5_6b_remote_contacted": False, "ssh_used": False, "gpu_used": False,
        "checkpoint_consumed": False, "formal_training_modified": False,
        "formal_training_config_modified": False, "formal_dataset_modified": False,
        "world_model_rollout_performed": False, "planner_objective_computed": False,
        "proposal_training": False, "closed_loop": False, "baseline": False,
        "locked_test_accessed": False,
        "provenance": provenance,
    }
    return {"planner_action_domain_v1.json": contract,
            "compute_budget_policy_v1.json": compute,
            "mobility_core_domain_v1.json": mobility,
            "planner_action_domain_acceptance_receipt.json": receipt,
            "rule_fallback_v1_receipt.json": fallback}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build()
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for name, content in expected.items():
        payload = json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != payload:
                raise SystemExit(f"receipt mismatch: {path}")
        else:
            path.write_text(payload, encoding="utf-8")
    print(f"STEP 6.0C receipts {'checked' if args.check else 'written'}: {len(expected)}/{len(expected)}")


if __name__ == "__main__":
    main()
