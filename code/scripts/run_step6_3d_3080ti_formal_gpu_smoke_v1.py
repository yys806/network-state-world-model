"""Bounded TRAIN-only smoke of the formal CUDA matrix solve path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

import torch

from run_step6_3d_formal_cpu_matrix_v1 import (
    execution_identity, read, selected_ids, solve_or_resume,
    validate_resume_result,
)
from run_step6_3d_one_cpu_solve_v1 import EXPECTED_SHA, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen = read(args.execution_config)
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    source = source_hashes()
    execution = execution_identity(device="cuda", gpu_model=torch.cuda.get_device_name(0),
        batch_size=int(frozen["batch_size"]), checkpoint_sha256=EXPECTED_SHA,
        source_sha256=source)
    validate_resume_result(frozen, {}, execution)
    anchor = selected_ids("15_train_anchor_manifest_objective_eligible.json", "dev_train", 32)[2]
    cases = []
    for method, seed, iterations, elite_ratio in (
        ("HRS", 63901, 1, None), ("S-CEM", 63902, 3, 0.1),
    ):
        result = solve_or_resume("smoke_3080ti_cpu_storage", anchor, method, seed, 256, iterations,
                                 elite_ratio, source, execution)
        resumed = solve_or_resume("smoke_3080ti_cpu_storage", anchor, method, seed, 256, iterations,
                                  elite_ratio, source, execution)
        if json.loads(json.dumps(result)) != resumed:
            raise AssertionError("smoke resume changed result")
        budget = result["outcome"]["budget_receipt"]
        if budget["N_unique_transition_evals"] != 256 or not result["gpu"] or \
                result["split"] != "dev_train" or result["locked_test"]:
            raise AssertionError("bounded formal GPU solve identity or budget mismatch")
        if result["outcome"]["complete_sequence_count"] == 0 or \
                result["outcome"]["h4_scoreable_count"] == 0:
            raise AssertionError("bounded formal GPU solve did not exercise H4 scorer")
        cases.append({"method": method, "seed": seed, "budget": 256,
                      "unique_transitions": budget["N_unique_transition_evals"],
                      "cache_hits": budget["N_cache_hits"],
                      "complete_h4": result["outcome"]["complete_sequence_count"],
                      "scoreable_h4": result["outcome"]["h4_scoreable_count"],
                      "score_residuals": result["score_residuals"],
                      "resume_equal": True,
                      "execution_config_id": result["execution_config_id"]})
    tampered = dict(result, batch_size=result["batch_size"] + 1)
    try:
        validate_resume_result(tampered, {"sample_id": anchor, "method": "S-CEM",
            "seed": 63902, "budget": 256, "iterations": 3, "elite_ratio": 0.1}, execution)
    except ValueError:
        tamper_rejected = True
    else:
        tamper_rejected = False
    if not tamper_rejected:
        raise AssertionError("resume accepted a different batch")
    if sum(case["complete_h4"] for case in cases) == 0 or \
            sum(case["scoreable_h4"] for case in cases) == 0:
        raise AssertionError("formal smoke did not exercise the H4 scorer")
    receipt = {"verdict": "PASS", "purpose": "bounded TRAIN CUDA smoke only",
               "sample_id": anchor, "cases": cases, "resume_batch_tamper_rejected": True,
               **execution, "formal_train_tuning": False,
               "validation_comparison": False, "locked_test": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"verdict": "PASS", "cases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
