"""TRAIN-only HRS B_WM=64 H4 support diagnostic; never selects anchors/methods."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/src")]
from run_step6_3d_one_cpu_solve_v1 import OUT, run, source_hashes, EXPECTED_SHA

SEED = 6391
B_WM = 64
RESULTS = OUT / "h4_train_preflight_results"


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    manifest_path = OUT / "15_train_anchor_manifest_objective_eligible.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["split"] != "dev_train" or len(manifest["selected"]) != 32:
        raise ValueError("fixed corrected TRAIN selection required")
    identity = source_hashes()
    per_anchor = []
    for selected in manifest["selected"]:
        sample_id = selected["sample_id"]
        name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16] + ".json"
        path = RESULTS / name
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row["sample_id"] != sample_id or row["split"] != "dev_train" or
                row["method"] != "HRS" or row["seed"] != SEED or row["budget"] != B_WM or
                row["source_sha256"] != identity or row["checkpoint_sha256"] != EXPECTED_SHA):
                raise ValueError(f"preflight resume identity mismatch: {path}")
        else:
            row = run(sample_id, "HRS", SEED, B_WM, 1, None, batch_size=1)
            if row["source_sha256"] != identity:
                raise ValueError("preflight source changed during solve")
            write_atomic(path, row)
        outcome = row["outcome"]
        budget = outcome["budget_receipt"]
        reasons = row["score_residuals"]
        per_anchor.append({
            "sample_id": sample_id,
            "result_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unique_transitions": budget["N_unique_transition_evals"],
            "complete_h4_sequences": outcome["complete_sequence_count"],
            "h4_scoreable_count": outcome["h4_scoreable_count"],
            "h4_unscoreable_count": outcome["h4_unscoreable_count"],
            "grammar_dead_end_count": budget["N_dead_end_branches"],
            "support_horizon_counts": row["support_horizon_counts"],
            "scorer_residual_reasons": reasons,
        })
        print(f"TRAIN H4 preflight {len(per_anchor)}/32: {sample_id} "
              f"scoreable={outcome['h4_scoreable_count']}", flush=True)
    if len(per_anchor) != 32 or any(row["unique_transitions"] != B_WM for row in per_anchor):
        raise ValueError("TRAIN preflight budget or anchor count incomplete")
    reason_counts = Counter()
    for row in per_anchor:
        reason_counts.update(row["scorer_residual_reasons"])
    success = sum(row["h4_scoreable_count"] > 0 for row in per_anchor)
    result = {
        "preflight": "TRAIN_ONLY_HRS_H4_SCOREABILITY",
        "diagnostic_seed": SEED, "B_WM_per_anchor": B_WM,
        "maximum_total_B_WM": 32 * B_WM,
        "actual_unique_transitions": sum(row["unique_transitions"] for row in per_anchor),
        "selected_train_anchor_count": 32,
        "anchors_with_h4_scoreable_candidate": success,
        "anchors_without_h4_scoreable_candidate": 32 - success,
        "support_boundary_and_scorer_reason_counts": dict(sorted(reason_counts.items())),
        "unsupported_future_return_birth_count": sum(value for key, value in reason_counts.items()
                                                     if "UNSUPPORTED_FUTURE_RETURN_BIRTH" in key),
        "empty_cohort_residual_count": sum(value for key, value in reason_counts.items()
                                            if "OBJECTIVE_UNSCOREABLE_EMPTY_COHORT" in key),
        "grammar_dead_end_total": sum(row["grammar_dead_end_count"] for row in per_anchor),
        "rows": per_anchor,
        "train_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": identity,
        "checkpoint_sha256": EXPECTED_SHA,
        "validation_search_executed": False,
        "optimizer_comparison": False,
        "anchor_replacement_based_on_preflight": False,
        "locked_test": False,
    }
    write_atomic(OUT / "22_train_h4_scoreability_preflight.json", result)
    print(json.dumps({key: result[key] for key in
                      ("anchors_with_h4_scoreable_candidate",
                       "anchors_without_h4_scoreable_candidate",
                       "unsupported_future_return_birth_count",
                       "grammar_dead_end_total")}, sort_keys=True))


if __name__ == "__main__":
    main()
