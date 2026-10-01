"""Resumable, paired execution of the researcher-frozen STEP 6.3D matrix.

Each solve is atomically stored with source/checkpoint identity. TRAIN tuning
finishes and freezes separate CEM configs before Validation may begin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
from pi_jwm.step6_3d_method_selection_v1 import (
    cluster_bootstrap_advantage, paired_outcome, select_method, tune_cem_config,
)
from run_step6_3d_one_cpu_solve_v1 import (
    EXPECTED_SHA, OUT, execution_identity, run, source_hashes,
)

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


def validate_resume_result(result: dict, solve_fields: dict,
                           expected_execution: dict) -> None:
    for key, value in {**solve_fields, **expected_execution}.items():
        if key not in result or result[key] != value:
            raise ValueError(f"resume identity mismatch: {key}")


def solve_or_resume(phase: str, sample_id: str, method: str, seed: int,
                    budget: int, k: int, rho: float | None, expected_source: dict,
                    execution: dict) -> dict:
    path = result_path(phase, sample_id, method, seed, budget, k, rho)
    solve_fields = {"sample_id": sample_id, "method": method, "seed": seed,
                    "budget": budget, "iterations": k, "elite_ratio": rho}
    if path.is_file():
        result = read(path)
        validate_resume_result(result, solve_fields, execution)
        return result
    if source_hashes() != expected_source:
        raise ValueError("search source changed during formal matrix")
    result = run(sample_id, method, seed, budget, k, rho,
                 batch_size=execution["batch_size"], device=execution["execution_device"])
    validate_resume_result(result, solve_fields, execution)
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


def train(execution: dict) -> None:
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
                                             k, rho, expected_source, execution)
                    cases[(sample_id, seed)] = result["outcome"]["best_objective"]
            config_results[(k, rho)] = cases
        grids[method] = tune_cem_config(config_results)
    frozen = {method: {"K": grids[method]["selected_config"][0],
                       "elite_ratio": grids[method]["selected_config"][1]}
              for method in ("S-CEM", "MH-CEM")}
    write_atomic(OUT / "05_train_hyperparameter_tuning_receipt.json", {
        "train_anchor_count": 32, "seeds": TRAIN_SEEDS, "B_WM": 512,
        "grids": grids, "validation_used": False, "source_sha256": expected_source,
        **execution, "locked_test": False})
    write_atomic(OUT / "06_frozen_selected_cem_configs.json", {
        "configs": frozen, "selected_from": "Formal TRAIN only",
        "train_anchor_count": 32, **execution, "locked_test": False})


METHODS = ("HRS", "S-CEM", "MH-CEM")
PAIRS = (("S-CEM", "HRS"), ("MH-CEM", "HRS"), ("MH-CEM", "S-CEM"))
FROZEN_CONFIGS = {"HRS": (1, None), "S-CEM": (4, 0.1), "MH-CEM": (4, 0.1)}
QUALIFICATION = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/05_selected_execution_config.json"
SUMMARY_SCHEMA = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/07_validation_output_schema_receipt.json"
RUNNER_SOURCE = "code/scripts/run_step6_3d_formal_cpu_matrix_v1.py"
EXECUTION_KEYS = tuple(execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
    batch_size=16, checkpoint_sha256=EXPECTED_SHA, source_sha256={}))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_stage_budgets(stage: str) -> tuple[int, ...]:
    if stage == "primary":
        return (1024,)
    if stage == "diagnostic":
        return (256, 512)
    raise ValueError("explicit primary or diagnostic Validation stage required")


def validate_validation_provenance(config: dict, execution: dict) -> dict:
    """Validate the historical selection separately from the new execution."""
    current = execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
        batch_size=16, checkpoint_sha256=EXPECTED_SHA, source_sha256=source_hashes())
    validate_resume_result(execution, {}, current)
    validate_resume_result(config, {}, current)
    runtime = {"name": "FORMAL_STEP_6_3D_VALIDATION_EXECUTION_CONFIG_3080TI",
        "amp": False, "bf16": False, "fp16": False, "quantization": False,
        "torch_compile": False, "prior_mode": "mean", "service_mode": "expectation",
        "stages": {"primary": [1024], "diagnostic": [256, 512]}, "stage_a_hard_stop": True}
    validate_resume_result(config, {}, runtime)
    expected_paths = {"train_frozen_configs": OUT / "06_frozen_selected_cem_configs.json",
        "train_tuning_receipt": OUT / "05_train_hyperparameter_tuning_receipt.json",
        "train_closure": OUT / "train_tuning_closure_acceptance.json",
        "qualification_config": QUALIFICATION, "summary_schema": SUMMARY_SCHEMA}
    parents = config.get("parents", {})
    if set(parents) != set(expected_paths):
        raise ValueError("Validation provenance parents incomplete")
    for key, path in expected_paths.items():
        if parents[key] != {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)}:
            raise ValueError(f"Validation parent identity mismatch: {key}")
    old = read(QUALIFICATION)
    old_execution = {key: old[key] for key in EXECUTION_KEYS}
    frozen = read(expected_paths["train_frozen_configs"])
    tuning = read(expected_paths["train_tuning_receipt"])
    validate_resume_result(frozen, {}, old_execution)
    validate_resume_result(tuning, {}, old_execution)
    if frozen["configs"] != {m: {"K": 4, "elite_ratio": .1} for m in ("S-CEM", "MH-CEM")}:
        raise ValueError("TRAIN-selected configs changed")
    closure = read(expected_paths["train_closure"])
    if closure.get("cases") != 768 or not all(closure.get("checks", {}).values()):
        raise ValueError("TRAIN closure not accepted")
    if closure["frozen_configs_sha256"] != file_sha(expected_paths["train_frozen_configs"]):
        raise ValueError("TRAIN frozen config no longer matches closure")
    changed = {key for key in current["source_sha256"]
        if current["source_sha256"][key] != old["source_sha256"].get(key)}
    if set(current["source_sha256"]) != set(old["source_sha256"]) or changed != {RUNNER_SOURCE}:
        raise ValueError("Validation patch must change only formal runner source")
    if config["execution_config_id"] == old["execution_config_id"]:
        raise ValueError("Validation must have its own execution identity")
    return {"train_selection_execution_config_id": old["execution_config_id"],
        "validation_execution_config_id": current["execution_config_id"],
        "parents": parents, "source_bridge_changed_files": sorted(changed),
        "summary_schema_version": read(SUMMARY_SCHEMA)["schema_version"]}


def summarize_validation_budget(all_results: dict, anchors: tuple, seeds: tuple, budget: int) -> dict:
    if len(anchors) != 64 or len(set(anchors)) != 64 or tuple(seeds) != VALIDATION_SEEDS:
        raise ValueError("exact 64 anchors and five frozen seeds required")
    expected = {(budget, a, s, m) for a in anchors for s in seeds for m in METHODS}
    supplied = {key for key in all_results if key[0] == budget}
    if supplied != expected:
        raise ValueError("incomplete or extra paired budget cases")
    diagnostics = {}
    for method in METHODS:
        cases = [all_results[(budget, a, seed, method)] for a in anchors for seed in seeds]
        totals = {key: sum(c["outcome"]["budget_receipt"][key] for c in cases)
            for key in ("N_complete_sequences", "N_dead_end_branches", "N_unique_transition_evals",
                "N_cache_hits", "N_proposed_steps", "N_admitted_steps", "N_rejected_steps")}
        scoreable = sum(c["outcome"]["best_objective"] is not None for c in cases)
        exceptions = sum(c["support_horizon_counts"].get("SCORER_EXCEPTION", 0) for c in cases)
        residual_exceptions = sum(n for c in cases for reason, n in c["score_residuals"].items()
            if reason.startswith(("ScorerStateInconsistency:", "BurdenSemanticsBlocked:")))
        if exceptions != residual_exceptions:
            raise ValueError("scorer exception diagnostics disagree")
        diagnostics[method] = {"case_count": len(cases), "h4_scoreable_success_count": scoreable,
            "h4_scoreable_success_rate": scoreable / 320,
            "total_complete_h4": sum(c["outcome"]["complete_sequence_count"] for c in cases),
            "total_distinct_scoreable_h4": sum(c["outcome"]["h4_scoreable_count"] for c in cases),
            "total_unscoreable_h4": sum(c["outcome"]["h4_unscoreable_count"] for c in cases),
            "return_birth_support_boundary_count": sum(n for c in cases for reason, n in c["score_residuals"].items()
                if reason.startswith("H4_SUPPORT_BOUNDARY:") and "UNSUPPORTED_FUTURE_RETURN_BIRTH" in reason),
            "grammar_dead_end_count": totals["N_dead_end_branches"], "scorer_exception_count": exceptions,
            "nominal_B_WM_total": 320 * budget, "N_unique_transition_evals": totals["N_unique_transition_evals"],
            "cache_hits": totals["N_cache_hits"], "N_cache_hits": totals["N_cache_hits"],
            "proposed": totals["N_proposed_steps"], "admitted": totals["N_admitted_steps"],
            "rejected": totals["N_rejected_steps"], "budget_cache_and_proposal_totals": totals,
            "wall_clock_diagnostic": {"in_solve_seconds_sum": sum(c["outcome"]["budget_receipt"]
                ["wall_clock_seconds_diagnostic_only"] for c in cases), "is_elapsed_matrix_runtime": False}}
    paired = {}
    for left, right in PAIRS:
        bins = dict.fromkeys(("only_left_scoreable", "only_right_scoreable", "both_scoreable_left_win",
            "both_scoreable_right_win", "both_scoreable_tie", "both_unscoreable"), 0)
        per_anchor = {}
        for anchor in anchors:
            values = []
            for seed in seeds:
                a = all_results[(budget, anchor, seed, left)]["outcome"]["best_objective"]
                b = all_results[(budget, anchor, seed, right)]["outcome"]["best_objective"]
                value = paired_outcome(a, b)
                values.append(value)
                key = ("both_unscoreable" if a is None and b is None else
                    "only_right_scoreable" if a is None else "only_left_scoreable" if b is None else
                    "both_scoreable_left_win" if value == 1 else "both_scoreable_right_win" if value == -1 else
                    "both_scoreable_tie")
                bins[key] += 1
            per_anchor[anchor] = values
        if sum(bins.values()) != 320:
            raise ValueError("six-category paired count must equal 320")
        flat = [v for row in per_anchor.values() for v in row]
        paired[f"{left}_vs_{right}"] = {"six_category_counts": bins, "total_paired_cases": 320,
            "win": flat.count(1), "tie": flat.count(0), "loss": flat.count(-1),
            "paired_seed_order": seeds, "per_anchor_seed_outcomes": per_anchor,
            "cluster_bootstrap": cluster_bootstrap_advantage(per_anchor)}
    return {"method_diagnostics": diagnostics, "paired_outcomes": paired, "paired_cases_per_method": 320}


def sensitivity_diagnostic(summary: dict) -> dict:
    """Preregistered simultaneous sensitivity; does not enter select_method."""
    import random
    rows = [summary["paired_outcomes"][f"{a}_vs_{b}"]["per_anchor_seed_outcomes"] for a, b in PAIRS]
    anchors = sorted(rows[0])
    means = [[sum(row[a]) / 5 for a in anchors] for row in rows]
    rng = random.Random(6316)
    boot = [[] for _ in PAIRS]
    for _ in range(10000):
        indices = [rng.randrange(64) for _ in range(64)]
        for k in range(3):
            boot[k].append(sum(means[k][i] for i in indices) / 64)
    bounds = {}
    tail = .05 / (2 * 3)
    for pair, values in zip(PAIRS, boot):
        values.sort()
        bounds[f"{pair[0]}_vs_{pair[1]}"] = {"lower": values[int(tail*10000)],
            "upper": values[int((1-tail)*10000)], "percentile_level": 1-.05/3}
    return {"diagnostic_only": True, "changes_primary_selection": False, "replications": 10000,
        "seed": 6316, "shared_anchor_resample_indices": True, "bonferroni_percentile_intervals": bounds,
        "limitation": "Bootstrap sensitivity does not guarantee finite-sample simultaneous coverage; Holm not enabled."}


def execute_validation_stage(execution: dict, stage: str, anchors: tuple, configs: dict,
                             expected_source: dict, provenance: dict) -> None:
    budgets = validation_stage_budgets(stage)
    if configs != FROZEN_CONFIGS or len(anchors) != 64 or len(set(anchors)) != 64:
        raise ValueError("frozen methods/configs and exact anchor count required")
    primary_path = OUT / "07_validation_stage_a_primary_comparison_receipt.json"
    selected_path = OUT / "08_selected_method.json"
    primary_parent = None
    if stage == "diagnostic":
        if not primary_path.is_file() or not selected_path.is_file():
            raise ValueError("Stage B requires accepted Stage A artifacts and separate researcher authorization")
        primary, selected = read(primary_path), read(selected_path)
        validate_resume_result(primary, {"stage": "primary", "budgets": [1024]}, execution)
        validate_resume_result(selected, {"primary_budget": 1024}, execution)
        if (primary["selected_method"] != selected["selected_method"] or
                selected["selected_method"] not in METHODS or
                selected.get("primary_comparison_sha256") != file_sha(primary_path) or
                primary.get("case_count") != 960 or primary.get("nominal_B_WM_total") != 983040 or
                primary.get("hard_stop_after_stage") is not True):
            raise ValueError("Stage A method selection artifacts disagree")
        primary_parent = {"path": primary_path.name, "sha256": file_sha(primary_path),
            "selected_method_sha256": file_sha(selected_path), "selected_method": selected["selected_method"]}
    from datetime import datetime, timezone
    import time
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    all_results = {}
    for budget in budgets:
        for sample_id in anchors:
            for seed in VALIDATION_SEEDS:
                for method in METHODS:
                    k, rho = configs[method]
                    all_results[(budget, sample_id, seed, method)] = solve_or_resume(
                        "validation", sample_id, method, seed, budget, k, rho, expected_source, execution)
    summaries = {str(b): summarize_validation_budget(all_results, anchors, VALIDATION_SEEDS, b) for b in budgets}
    receipt = {"stage": stage, "budgets": budgets, "selected_anchor_count": 64,
        "seeds": VALIDATION_SEEDS, "case_count": len(all_results),
        "nominal_B_WM_total": sum(budgets)*960, "primary_budget": 1024,
        "comparison": summaries, "train_configs_frozen": configs, "validation_retuning": False,
        "provenance_bridge": provenance, **execution, "locked_test": False,
        "runtime_diagnostic": {"invocation_start_utc": started, "invocation_finish_utc": datetime.now(timezone.utc).isoformat(),
            "invocation_elapsed_seconds": time.perf_counter()-timer,
            "includes_resume_reads": True, "is_full_elapsed_across_resumes": False}}
    if stage == "primary":
        ci = {pair: summaries["1024"]["paired_outcomes"][f"{pair[0]}_vs_{pair[1]}"]["cluster_bootstrap"] for pair in PAIRS}
        winner = select_method(ci)
        receipt.update(selected_method=winner, hard_stop_after_stage=True,
            next_stage_auto_start=False, sensitivity_diagnostic=sensitivity_diagnostic(summaries["1024"]))
        write_atomic(primary_path, receipt)
        write_atomic(selected_path, {"selected_method": winner, "primary_budget": 1024,
            "rule": "frozen STEP 6.3D primary-budget bootstrap simplicity rule",
            "pairwise_ci": {f"{a}_vs_{b}": value for (a,b),value in ci.items()},
            "primary_comparison_sha256": file_sha(primary_path), **execution,
            "validation_retuning": False, "locked_test": False})
        print("Stage A complete: STOP. Researcher review required; Stage B not started.", flush=True)
        return
    receipt.update(primary_parent=primary_parent, changes_primary_selection=False)
    write_atomic(OUT / "09_validation_stage_b_budget_diagnostic_receipt.json", receipt)
    return


def validation(execution: dict, *, stage: str, execution_config: dict) -> None:
    provenance = validate_validation_provenance(execution_config, execution)
    anchors = selected_ids("16_validation_anchor_manifest_objective_eligible.json", "dev_validation", 64)
    execute_validation_stage(execution, stage, anchors, FROZEN_CONFIGS, source_hashes(), provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "validation"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--execution-config", type=Path)
    parser.add_argument("--validation-stage", choices=("primary", "diagnostic"))
    args = parser.parse_args()
    if args.phase == "validation" and (args.validation_stage is None or args.execution_config is None):
        parser.error("Validation requires --validation-stage and new --execution-config")
    if args.phase == "train" and args.validation_stage is not None:
        parser.error("Validation stages cannot run TRAIN")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available() or args.execution_config is None:
            raise ValueError("CUDA and a frozen execution config are required")
        gpu_model = torch.cuda.get_device_name(0)
    else:
        gpu_model = None
        if args.execution_config is not None:
            raise ValueError("CPU execution does not use a GPU config")
    execution = execution_identity(device=args.device, gpu_model=gpu_model,
        batch_size=args.batch_size, checkpoint_sha256=EXPECTED_SHA,
        source_sha256=source_hashes())
    if args.execution_config is not None:
        frozen = read(args.execution_config)
        validate_resume_result(frozen, {}, execution)
    if args.phase == "train":
        train(execution)
    else:
        validation(execution, stage=args.validation_stage, execution_config=read(args.execution_config))


if __name__ == "__main__":
    main()
