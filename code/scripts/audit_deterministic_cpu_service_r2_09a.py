"""Run the bounded R2-09A deterministic CPU lifecycle acceptance.

The audit intentionally imports only the synthetic R2-08 fixture and the local
unit tests.  It does not import AirFogSim, read V2U4Real, use GPU/validation/
test/locked_test data, or start a detector model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
TESTS = ROOT / "code" / "tests"
CONFIG = ROOT / "code" / "scripts" / "configs" / "pi_jwm_r2_09a_deterministic_cpu_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_unit_suite() -> tuple[dict, str]:
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(TESTS))
    module = importlib.import_module("test_deterministic_cpu_service_lifecycle_r2_09a")
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    return {
        "status": "PASS" if result.wasSuccessful() else "BLOCKED",
        "tests_run": result.testsRun,
        "failures": [str(item[0]) for item in result.failures],
        "errors": [str(item[0]) for item in result.errors],
        "skipped": [str(item[0]) for item in result.skipped],
    }, output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "code/artifacts/audit/pi_jwm_r2_09a_deterministic_cpu_service_20261027")
    args = parser.parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    suite_result, log = run_unit_suite()
    (output_dir / "execution.log").write_text(log, encoding="utf-8")
    source_paths = [
        SRC / "pi_jwm" / "deterministic_cpu_service_lifecycle_v1.py",
        SRC / "pi_jwm" / "scenario_contract_v1.py",
        SRC / "pi_jwm" / "airfogsim_service_interface_v1.py",
        SRC / "pi_jwm" / "v2u4real_airfogsim_adapter_v1.py",
        TESTS / "test_deterministic_cpu_service_lifecycle_r2_09a.py",
        CONFIG,
        ROOT / "code/scripts/configs/pi_jwm_scenario_v1.synthetic_test.json",
    ]
    result = {
        "schema_version": "PIJWM-R2-09A-Acceptance-v1",
        "step": "R2-09A",
        "status": suite_result["status"],
        "scope": "deterministic_cpu_service_and_task_lifecycle_only",
        "profile": "synthetic_contract_test_only",
        "input_config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "unit_test": suite_result,
        "execution": {
            "command": "python code/scripts/audit_deterministic_cpu_service_r2_09a.py --output code/artifacts/audit/pi_jwm_r2_09a_deterministic_cpu_service_20261027",
            "elapsed_s": round(time.time() - started, 6),
            "python": sys.version,
            "gpu": False,
            "locked_test": False,
            "validation_or_test_data_access": False,
            "airfogsim_wireless_service": False,
            "real_detector": False,
        },
        "acceptance_matrix": {
            "single_slot_complete": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "multi_slot_partial_and_accumulation": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "observation_boundary_persistence": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "deadline_inclusive_and_exclusive_engineering_strategies": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "four_failure_classes_distinct": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "multi_task_and_multi_packet": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "invalid_input_and_duplicate_guards": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "deterministic_replay": "PASS" if suite_result["status"] == "PASS" else "BLOCKED",
            "real_wireless": "NOT_RUN",
            "real_detection": "NOT_RUN",
            "spatial_reconstruction": "NOT_RUN",
        },
        "terminal_reason_vocabulary": [
            "REQUEST_NOT_DELIVERED",
            "FEATURE_PACKET_NOT_DELIVERED",
            "RSU_PROCESSING_TIMEOUT",
            "DETECTION_FAILED",
            "DETECTION_NOT_EXECUTED",
            "COMPLETED",
        ],
        "physical_boundary": {
            "service_receipts": "SYNTHETIC_ONLY",
            "theoretical_capacity_is_delivery": False,
            "airfogsim_source_modified": False,
            "formal_research_parameters_frozen": False,
        },
        "source_fingerprints": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in source_paths
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if suite_result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
