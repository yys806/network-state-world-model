"""Resumable, paired CPU execution of the researcher-frozen STEP 6.3D matrix.

Each solve is atomically stored with source/checkpoint identity. TRAIN tuning
finishes and freezes separate CEM configs before Validation may begin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
from pi_jwm.step6_3d_method_selection_v1 import (
    cluster_bootstrap_advantage, paired_outcome, select_method, tune_cem_config,
)
from run_step6_3d_one_cpu_solve_v1 import EXPECTED_SHA, OUT, run, source_hashes

TRAIN_SEEDS = (6301, 6302, 6303)
VALIDATION_SEEDS = (6311, 6312, 6313, 6314, 6315)
BUDGETS = (256, 512, 1024)
CONFIGS = ((3, 0.1), (3, 0.2), (4, 0.1), (4, 0.2))
RESULTS = OUT / "solve_results"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def result_path(phase: str, sample_id: str, method: str, seed: int,
                budget: int, k: int, rho: float | None) -> Path:
    anchor_key = hashlib.sha256(sample_id.encode()).hexdigest()[:16]
    suffix = "uniform" if rho is None else f"k{k}_rho{str(rho).replace('.', 'p')}"
    return RESULTS / phase / f"{anchor_key}_{method}_{seed}_b{budget}_{suffix}.json"


def solve_or_resume(phase: str, sample_id: str, method: str, seed: int,
                    budget: int, k: int, rho: float | None, expected_source: dict) -> dict:
    path = result_path(phase, sample_id, method, seed, budget, k, rho)
    if path.is_file():
        result = read(path)
        if (result["sample_id"], result["method"], result["seed"], result["budget"],
            result["iterations"], result["elite_ratio"], result["checkpoint_sha256"],
            result["source_sha256"]) != (sample_id, method, seed, budget, k, rho,
                                          EXPECTED_SHA, expected_source):
            raise ValueError(f"resume identity mismatch: {path}")
        return result
    if source_hashes() != expected_source:
        raise ValueError("search source changed during formal matrix")
    result = run(sample_id, method, seed, budget, k, rho)
    if result["source_sha256"] != expected_source:
        raise ValueError("search source changed inside solve")
    write_atomic(path, result)
    print(f"{phase}: {sample_id} {method} seed={seed} B_WM={budget} K={k} rho={rho}"
          f" H4={result['outcome']['h4_scoreable_count']}", flush=True)
    return result


def selected_ids(name: str, split: str, expected: int) -> tuple[str, ...]:
    manifest = read(OUT / name)
    if manifest["split"] != split or len(manifest["selected"]) != expected:
        raise ValueError("anchor manifest identity or count mismatch")
    source = ROOT / manifest["source_receipt"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("anchor manifest source receipt changed")
    return tuple(row["sample_id"] for row in manifest["selected"])


def train() -> None:
    anchors = selected_ids("15_train_anchor_manifest_objective_eligible.json", "dev_train", 32)
    expected_source = source_hashes()
    grids = {}
    for method in ("S-CEM", "MH-CEM"):
        config_results = {}
        for k, rho in CONFIGS:
            cases = {}
            for sample_id in anchors:
                for seed in TRAIN_SEEDS:
                    result = solve_or_resume("train", sample_id, method, seed, 512,
                                             k, rho, expected_source)
                    cases[(sample_id, seed)] = result["outcome"]["best_objective"]
            config_results[(k, rho)] = cases
        grids[method] = tune_cem_config(config_results)
    frozen = {method: {"K": grids[method]["selected_config"][0],
                       "elite_ratio": grids[method]["selected_config"][1]}
              for method in ("S-CEM", "MH-CEM")}
    write_atomic(OUT / "05_train_hyperparameter_tuning_receipt.json", {
        "train_anchor_count": 32, "seeds": TRAIN_SEEDS, "B_WM": 512,
        "grids": grids, "validation_used": False, "source_sha256": expected_source,
        "checkpoint_sha256": EXPECTED_SHA, "locked_test": False})
    write_atomic(OUT / "06_frozen_selected_cem_configs.json", {
        "configs": frozen, "selected_from": "Formal TRAIN only",
        "train_anchor_count": 32, "source_sha256": expected_source,
        "checkpoint_sha256": EXPECTED_SHA, "locked_test": False})


def validation() -> None:
    anchors = selected_ids("16_validation_anchor_manifest_objective_eligible.json", "dev_validation", 64)
    frozen = read(OUT / "06_frozen_selected_cem_configs.json")
    expected_source = source_hashes()
    if frozen["source_sha256"] != expected_source or frozen["checkpoint_sha256"] != EXPECTED_SHA:
        raise ValueError("TRAIN selected configs do not match current source/checkpoint")
    configs = {"HRS": (1, None), **{method: (row["K"], row["elite_ratio"])
                                     for method, row in frozen["configs"].items()}}
    all_results = {}
    for budget in BUDGETS:
        for sample_id in anchors:
            for seed in VALIDATION_SEEDS:
                for method in ("HRS", "S-CEM", "MH-CEM"):
                    k, rho = configs[method]
                    all_results[(budget, sample_id, seed, method)] = solve_or_resume(
                        "validation", sample_id, method, seed, budget, k, rho, expected_source)
    summary = {}
    primary_ci = {}
    for budget in BUDGETS:
        success = {method: sum(all_results[(budget, anchor, seed, method)]["outcome"]["best_objective"]
                               is not None for anchor in anchors for seed in VALIDATION_SEEDS)
                   for method in configs}
        method_diagnostics = {}
        for method in configs:
            cases = [all_results[(budget, anchor, seed, method)]
                     for anchor in anchors for seed in VALIDATION_SEEDS]
            objectives = Counter(json.dumps(case["outcome"]["best_objective"])
                                 for case in cases if case["outcome"]["best_objective"] is not None)
            keys = ("N_complete_sequences", "N_dead_end_branches", "N_unique_transition_evals",
                    "N_cache_hits", "N_proposed_steps", "N_admitted_steps", "N_rejected_steps")
            method_diagnostics[method] = {
                "h4_scoreable_success_rate": success[method] / len(cases),
                "best_objective_tuple_distribution": [
                    {"objective": json.loads(value), "count": count}
                    for value, count in sorted(objectives.items())],
                "total_h4_scoreable_candidates": sum(case["outcome"]["h4_scoreable_count"]
                                                     for case in cases),
                "total_h4_unscoreable_sequences": sum(case["outcome"]["h4_unscoreable_count"]
                                                     for case in cases),
                "budget_cache_and_proposal_totals": {key: sum(
                    case["outcome"]["budget_receipt"][key] for case in cases) for key in keys},
                "wall_clock_seconds_total_diagnostic_only": sum(
                    case["outcome"]["budget_receipt"]["wall_clock_seconds_diagnostic_only"]
                    for case in cases),
            }
        paired = {}
        for left, right in (("S-CEM", "HRS"), ("MH-CEM", "HRS"), ("MH-CEM", "S-CEM")):
            per_anchor = {anchor: [paired_outcome(
                all_results[(budget, anchor, seed, left)]["outcome"]["best_objective"],
                all_results[(budget, anchor, seed, right)]["outcome"]["best_objective"])
                for seed in VALIDATION_SEEDS] for anchor in anchors}
            flat = [item for values in per_anchor.values() for item in values]
            key = f"{left}_vs_{right}"
            paired[key] = {"win": flat.count(1), "tie": flat.count(0),
                           "loss": flat.count(-1),
                           "cluster_bootstrap": cluster_bootstrap_advantage(per_anchor)}
            if budget == 1024:
                primary_ci[(left, right)] = paired[key]["cluster_bootstrap"]
        summary[str(budget)] = {"h4_success_count": success,
                                "method_diagnostics": method_diagnostics,
                                "paired_outcomes": paired,
                                "paired_cases_per_method": len(anchors) * len(VALIDATION_SEEDS)}
    winner = select_method(primary_ci)
    write_atomic(OUT / "07_validation_paired_comparison_receipt.json", {
        "selected_anchor_count": 64, "static_empty_anchors_separate": 6,
        "seeds": VALIDATION_SEEDS, "budgets": BUDGETS,
        "comparison": summary, "primary_budget": 1024,
        "train_configs_frozen": configs, "validation_retuning": False,
        "source_sha256": expected_source, "checkpoint_sha256": EXPECTED_SHA,
        "locked_test": False})
    write_atomic(OUT / "08_selected_method.json", {
        "selected_method": winner, "rule": "frozen STEP 6.3D primary-budget bootstrap simplicity rule",
        "primary_budget": 1024, "pairwise_ci": {
            f"{left}_vs_{right}": value for (left, right), value in primary_ci.items()},
        "validation_retuning": False, "locked_test": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    if args.phase == "train":
        train()
    else:
        validation()


if __name__ == "__main__":
    main()
