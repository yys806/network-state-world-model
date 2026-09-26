"""Rebuild deterministic CPU-only STEP 6.0A contract receipts."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "src"), str(ROOT / "code" / "scripts"), str(ROOT / "code" / "tests")]
from pi_jwm.step6_0a_candidate_generation_v1 import Backend, ConstraintStatus

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_0a_candidate_generation_contract_v1_20260926"
ADAPTER = ROOT / "code/scripts/build_step5_1d_unified_model_chain_v1.py"
MODULE = ROOT / "code/src/pi_jwm/step6_0a_candidate_generation_v1.py"
TEST = ROOT / "code/tests/test_step6_0a_candidate_generation_v1.py"
FIELDS = ("mobility_entity_index", "mobility_values", "comm_relation_index", "comm_values",
          "comm_allocation_mask", "comp_agent_index", "comp_task_index", "comp_values",
          "route_task_index", "route_flow_index", "route_values")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    source = MODULE.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    suite = unittest.defaultTestLoader.loadTestsFromName("test_step6_0a_candidate_generation_v1")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    checks = {
        "focused_tests": result.wasSuccessful() and result.testsRun >= 5,
        "current_action_fields_identified": all(f'"{name}"' in adapter for name in FIELDS),
        "training_adapter_wrapped": "from build_step5_1d_unified_model_chain_v1 import build_action" in source,
        "obsolete_p6_fields_absent": all(name not in source for name in
            ("task_action_present", "task_action_node_index", "task_action_source_node_index")),
        "no_model_rollout_optimizer_training_cuda": all(term not in source for term in
            (".rollout(", "optimizer.step(", ".cuda(", "torch.cuda", "future_target", "locked_test")),
        "tri_state": set(ConstraintStatus.__members__) == {"SATISFIED", "VIOLATED", "UNKNOWN"},
        "backend_names": set(Backend.__members__) == {"SEARCH", "LEARNED", "HYBRID", "RULE_FALLBACK"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "candidate_generation_contract.json", {
        "schema": "PIJWM_STEP_06_0A_CANDIDATE_GENERATION_V1", "evidence_kind": "SYNTHETIC_CONTRACT_EVIDENCE",
        "horizon": {"min": 1, "max": 4, "basis": "formal H1-H4 supervision/validation boundary; no optimality claim"},
        "action_families": ["Route", "Comm", "Comp", "Mob"], "formal_tensor_fields": FIELDS,
        "adapter": str(ADAPTER.relative_to(ROOT)).replace("\\", "/"), "adapter_sha256": sha(ADAPTER),
        "candidate_module_sha256": sha(MODULE), "test_sha256": sha(TEST),
        "unknown_constraint_policy": "retain in pool, reject compilation; researcher decision pending",
        "dynamic_cpu_available": "UNKNOWN", "mobility_bounds": "UNKNOWN",
        "future_return_birth": "fixed-support blocked", "proposal_training": False,
        "world_model_rollout": False, "planner_performance": False,
    })
    write(OUT / "action_adapter_equivalence_receipt.json", {
        "passed": checks["focused_tests"], "evidence_kind": "SYNTHETIC_CONTRACT_EVIDENCE",
        "fixture": "test_four_action_exact_adapter_equivalence",
        "comparison": "torch.equal for all 11 tensor fields plus mapping equality; explicit semantic assertions",
        "fields": FIELDS, "real_simulator_candidate_coverage": False,
    })
    write(OUT / "candidate_generation_acceptance_receipt.json", {
        "passed": all(checks.values()), "checks": checks, "focused_test_count": result.testsRun,
        "evidence_kind": "SYNTHETIC_CONTRACT_EVIDENCE", "cpu_only": True,
        "step5_6b_remote_contacted": False, "ssh_used": False, "gpu_used": False,
        "formal_training_process_modified": False, "formal_training_config_modified": False,
        "formal_dataset_modified": False, "world_model_training_code_modified": False,
        "checkpoint_consumed_for_planner": False, "locked_test_accessed": False,
        "world_model_rollout_performed": False, "proposal_training": False,
        "planner_objective_computed": False, "closed_loop_performed": False,
    })
    if not all(checks.values()):
        raise SystemExit("STEP 6.0A acceptance failed")


if __name__ == "__main__":
    main()
