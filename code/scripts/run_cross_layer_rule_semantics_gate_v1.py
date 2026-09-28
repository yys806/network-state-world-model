"""Mandatory source-fingerprinted CPU gate before any future formal training."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "code/tests"
sys.path[:0] = [str(TESTS), str(ROOT / "code/src"), str(ROOT / "code/scripts")]
RECEIPT = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928/12_cross_layer_semantics_gate.json"
SOURCES = (
    "code/src/pi_jwm/step4_2c_b_causal_flow_ledger_raw_v1.py",
    "code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py",
    "code/src/pi_jwm/step4_3a_typed_dual_graph_builder_v1.py",
    "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
    "code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py",
    "code/src/pi_jwm/step6_2a_route_rule_metadata_v1.py",
    "code/scripts/build_step5_1d_unified_model_chain_v1.py",
    "code/scripts/run_cross_layer_rule_semantics_gate_v1.py",
    "code/tests/test_step6_2a_route_recovery_v1.py",
)
PATTERNS = (
    "test_step4_2c_b_causal_flow_ledger_raw_v1.py",
    "test_step4_2c_c_flow_sample_tensor_v1.py",
    "test_step4_3a_typed_dual_graph_builder_v1.py",
    "test_step4_4_structured_rssm_world_model_v1.py",
    "test_step5_1d_unified_model_chain_v1.py",
    "test_step6_2a_patch_side_state_v1.py",
    "test_step6_2a_route_recovery_v1.py",
)
COVERAGE = {
    "raw_to_tensor_to_graph_to_rule_two_hop_input": "test_real_two_hop_input_tensor_state_rule_alignment",
    "normal_multihop_partial_and_completion": "test_two_hop_partial_and_terminal_lifecycle",
    "same_destination_full_path_route": "test_same_destination_reroute_writes_full_destination_list",
    "return": "test_return_is_independent_and_multihop",
    "task_completion": "test_computation_finished_waits_for_existing_return_flow",
    "fixed_support_birth": "test_missing_required_return_support_blocks_final_completion",
    "comm_rebind": "test_hop_completion_rebinds_current_comm_relation",
    "cpu_rule": "test_computation_finished_waits_for_existing_return_flow",
    "mob_rule": "test_vehicle_uav_static_stochastic_eligibility",
    "e2e_useful_throughput_extractor": "test_shared_real_throughput_uses_ledger_e2e_delta",
}


def fingerprint() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="reject stale or failed gate")
    args = parser.parse_args()
    source_hashes = fingerprint()
    if args.check:
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if receipt["status"] != "PASS" or receipt["source_sha256"] != source_hashes:
            raise ValueError("CROSS_LAYER_RULE_SEMANTICS_GATE is missing, failed or stale")
        print("CROSS_LAYER_RULE_SEMANTICS_GATE=PASS")
        return
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for pattern in PATTERNS:
        suite.addTests(loader.discover(str(TESTS), pattern=pattern))
    names = []
    def walk(node):
        for test in node:
            if isinstance(test, unittest.TestSuite):
                walk(test)
            else:
                names.append(test.id().split(".")[-1])
    walk(suite)
    missing = {key: name for key, name in COVERAGE.items() if name not in names}
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
    status = "PASS" if result.wasSuccessful() and not missing else "FAIL"
    receipt = {"status": status, "mandatory_before_future_formal_training": True,
               "source_sha256": source_hashes, "test_patterns": PATTERNS,
               "test_count": result.testsRun, "failures": [str(t) for t, _ in result.failures],
               "errors": [str(t) for t, _ in result.errors], "missing_coverage": missing,
               "coverage": COVERAGE, "gpu": False, "locked_test": False,
               "training": False, "output": output.getvalue()}
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CROSS_LAYER_RULE_SEMANTICS_GATE={status}; tests={result.testsRun}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
