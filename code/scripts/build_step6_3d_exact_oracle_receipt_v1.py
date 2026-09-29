"""Write machine-readable evidence for the bounded STEP 6.3D exact oracle."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/tests")]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929"


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "test_step6_3d_exact_oracle_v1.ExactOracleTests")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 1:
        raise AssertionError(stream.getvalue())
    sources = (
        "code/tests/test_step6_3d_exact_oracle_v1.py",
        "code/src/pi_jwm/step6_3d_structured_proposal_v1.py",
        "code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py",
        "code/src/pi_jwm/step6_3c_candidate_domain_v1.py",
        "code/src/pi_jwm/step6_3c_search_protocol_v1.py",
    )
    receipt = {
        "gate": "EXACT_ORACLE_CORRECTNESS",
        "verdict": "PASS",
        "synthetic_h4_space_count": 16,
        "known_best_objective": [0, 0.0, 0.0, 0.0, 0.0],
        "methods_asserted_to_recover_optimum": ["HRS", "S-CEM", "MH-CEM"],
        "B_WM_per_method": 30,
        "unique_transition_evals_asserted_per_method": 30,
        "cache_hits_asserted_positive_per_method": True,
        "tests_run": result.testsRun,
        "test_output": stream.getvalue(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        "formal_performance_evidence": False,
        "locked_test": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "12_exact_oracle_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": receipt["verdict"], "tests_run": result.testsRun}))


if __name__ == "__main__":
    main()
