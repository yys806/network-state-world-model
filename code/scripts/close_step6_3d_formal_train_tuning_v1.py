"""Rebuild and archive the completed TRAIN-only STEP 6.3D search evidence."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
from pi_jwm.step6_3d_method_selection_v1 import tune_cem_config
from run_step6_3d_one_cpu_solve_v1 import EXPECTED_SHA, source_hashes

REL = Path("code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929")
OUT = ROOT / REL
CONFIG = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/05_selected_execution_config.json"
METHODS = ("S-CEM", "MH-CEM")
CONFIGS = ((3, 0.1), (3, 0.2), (4, 0.1), (4, 0.2))
SEEDS = (6301, 6302, 6303)
LOG_BIRTH_CST = "2026-09-30T18:30:26.597154+08:00"  # remote stat -c %w


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n").encode("utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    frozen = read(CONFIG)
    require(source_hashes() == frozen["source_sha256"], "current source hash differs from formal TRAIN")
    require(EXPECTED_SHA == frozen["checkpoint_sha256"], "checkpoint identity differs")
    manifest = read(OUT / "15_train_anchor_manifest_objective_eligible.json")
    anchors = [row["sample_id"] for row in manifest["selected"]]
    require(manifest["split"] == "dev_train" and len(anchors) == 32 and
            len(set(anchors)) == 32, "TRAIN anchor manifest mismatch")
    eligibility = read(OUT / "20_objective_anchor_eligibility_receipt.json")
    require(eligibility["cohort_zero_count"] == 0 and eligibility["static_empty_count"] == 0,
            "selected anchor Objective cohort/static domain not eligible")
    require(set(anchors) == {row["sample_id"] for row in eligibility["rows"]
                              if row["split"] == "dev_train"}, "eligibility row mismatch")

    backup = read(OUT / "train_local_backup_inventory.json")
    files = sorted((OUT / "solve_results/train").glob("*.json"))
    require(len(files) == 768 and backup["file_count"] == 771,
            "raw result or backup inventory count mismatch")
    inventory_map = {row["path"]: row for row in backup["files"]}
    require(len(inventory_map) == 771, "duplicate backup inventory path")
    for rel, entry in inventory_map.items():
        path = OUT / rel
        require(path.is_file() and path.stat().st_size == entry["size_bytes"] and
                sha(path) == entry["sha256"], f"backup file SHA mismatch: {rel}")
    train_receipt = read(OUT / "05_train_hyperparameter_tuning_receipt.json")
    configs_receipt = read(OUT / "06_frozen_selected_cem_configs.json")
    execution_fields = ("execution_device", "gpu_model", "precision", "batch_size",
                        "state_storage", "bucket_strategy", "checkpoint_sha256",
                        "source_sha256", "execution_config_id", "locked_test")
    for receipt in (train_receipt, configs_receipt):
        require(all(receipt[key] == frozen[key] for key in execution_fields),
                "TRAIN receipt execution identity mismatch")
    require(train_receipt["train_anchor_count"] == 32 and
            train_receipt["seeds"] == list(SEEDS) and train_receipt["B_WM"] == 512 and
            train_receipt["validation_used"] is False and
            configs_receipt["train_anchor_count"] == 32 and
            configs_receipt["selected_from"] == "Formal TRAIN only",
            "TRAIN receipt protocol mismatch")

    grouped: dict[tuple[str, int, float], list[dict]] = collections.defaultdict(list)
    identities = set()
    rows = []
    expected = {(anchor, method, seed, 512, k, rho)
                for anchor in anchors for method in METHODS for seed in SEEDS
                for k, rho in CONFIGS}
    for path in files:
        row = read(path)
        ident = (row["sample_id"], row["method"], row["seed"], row["budget"],
                 row["iterations"], row["elite_ratio"])
        require(ident in expected and ident not in identities, f"wrong/duplicate solve identity: {path}")
        identities.add(ident)
        require(all(row[key] == frozen[key] for key in execution_fields),
                f"mixed execution/source/checkpoint identity: {path}")
        require(row["split"] == "dev_train" and row["gpu"] is True and
                row["training"] is False and row["closed_loop"] is False and
                row["sidecar_alignment_passed"] is True,
                f"scope/sidecar mismatch: {path}")
        outcome = row["outcome"]
        budget = outcome["budget_receipt"]
        require(outcome["method"] == row["method"] and outcome["seed"] == row["seed"] and
                outcome["budget"] == row["budget"] and
                outcome["iterations"] == row["iterations"] and
                outcome["elite_ratio"] == row["elite_ratio"] and
                outcome["batch_size"] == row["batch_size"] and
                budget["B_WM"] == 512 and budget["budget_unit"] == "candidate_one_step_transition" and
                0 <= budget["N_unique_transition_evals"] <= 512 and
                budget["N_complete_sequences"] == outcome["complete_sequence_count"] and
                budget["N_proposed_steps"] == budget["N_admitted_steps"] + budget["N_rejected_steps"],
                f"budget/outcome mismatch: {path}")
        grouped[(row["method"], row["iterations"], row["elite_ratio"])].append(row)
        rows.append(row)
    require(identities == expected and len(grouped) == 8 and
            all(len(group) == 96 for group in grouped.values()), "incomplete TRAIN grid")
    require(not list((OUT / "solve_results/validation").glob("*.json")) and
            not (OUT / "07_validation_paired_comparison_receipt.json").exists() and
            not (OUT / "08_selected_method.json").exists(), "Validation/selection already started")

    reconstructed = {}
    for method in METHODS:
        outcomes = {(k, rho): {(row["sample_id"], row["seed"]): row["outcome"]["best_objective"]
                               for row in grouped[(method, k, rho)]} for k, rho in CONFIGS}
        rebuilt = tune_cem_config(outcomes)
        rebuilt_json = json.loads(json.dumps(rebuilt))
        require(rebuilt_json == train_receipt["grids"][method],
                f"independent TRAIN selection differs for {method}")
        require(rebuilt_json["selected_config"] == [4, 0.1] and
                configs_receipt["configs"][method] == {"K": 4, "elite_ratio": 0.1},
                f"frozen config differs for {method}")
        reconstructed[method] = rebuilt_json

    def summarize(group: list[dict]) -> dict:
        budget_keys = ("N_complete_sequences", "N_dead_end_branches",
                       "N_proposed_steps", "N_admitted_steps", "N_rejected_steps",
                       "N_cache_hits", "N_unique_transition_evals")
        totals = {key: sum(row["outcome"]["budget_receipt"][key] for row in group)
                  for key in budget_keys}
        residuals = collections.Counter()
        for row in group:
            residuals.update(row["score_residuals"])
        success = sum(row["outcome"]["best_objective"] is not None for row in group)
        return {"cases": len(group), "h4_scoreable_case_count": success,
                "h4_scoreable_case_rate": success / len(group),
                "best_objective_available_count": success,
                "complete_h4_sequences": totals["N_complete_sequences"],
                "scoreable_h4_distinct_candidates": sum(row["outcome"]["h4_scoreable_count"] for row in group),
                "unscoreable_h4_completion_attempts": sum(row["outcome"]["h4_unscoreable_count"] for row in group),
                "return_birth_support_boundary_events": sum(value for key, value in residuals.items()
                                                             if "UNSUPPORTED_FUTURE_RETURN_BIRTH" in key),
                "scorer_exception_events": sum(value for key, value in residuals.items()
                                                if key.startswith(("ScorerStateInconsistency:",
                                                                   "BurdenSemanticsBlocked:"))),
                "scoreable_status_but_h4_rejected_events": residuals.get("H4_UNSCOREABLE:SCOREABLE", 0),
                "budget_totals": totals,
                "wall_clock_solve_seconds_sum_diagnostic_only": sum(
                    row["outcome"]["budget_receipt"]["wall_clock_seconds_diagnostic_only"] for row in group),
                "score_residuals": dict(sorted(residuals.items()))}

    per_config = []
    for method in METHODS:
        for k, rho in CONFIGS:
            group = grouped[(method, k, rho)]
            ranking = next(item for item in reconstructed[method]["ranking"]
                           if item["config"] == [k, rho])
            per_config.append({"method": method, "K": k, "rho": rho,
                               "paired_objective_net_outcome": ranking["paired_objective_net_outcome"],
                               **summarize(group)})
    per_anchor_seed = []
    for anchor in anchors:
        for seed in SEEDS:
            cases = [row for row in rows if row["sample_id"] == anchor and row["seed"] == seed]
            per_anchor_seed.append({"sample_id": anchor, "seed": seed,
                                    "zero_scoreable_all_eight_configs": all(
                                        row["outcome"]["best_objective"] is None for row in cases),
                                    "configs": [{"method": row["method"], "K": row["iterations"],
                                                 "rho": row["elite_ratio"],
                                                 "h4_scoreable_count": row["outcome"]["h4_scoreable_count"],
                                                 "best_objective": row["outcome"]["best_objective"],
                                                 "return_birth_support_boundary_events": sum(
                                                     value for key, value in row["score_residuals"].items()
                                                     if "UNSUPPORTED_FUTURE_RETURN_BIRTH" in key)}
                                                for row in sorted(cases, key=lambda r: (
                                                    r["method"], r["iterations"], r["elite_ratio"]))]})
    all_summary = summarize(rows)
    tar_path = OUT / "train_remote_stream.tar"
    with tarfile.open(tar_path) as archive:
        members = [member for member in archive if member.isfile() and
                   member.name.startswith("solve_results/train/")]
        require(len(members) == 768, "raw transfer tar count mismatch")
        first = min(members, key=lambda item: item.mtime)
        last = max(members, key=lambda item: item.mtime)
        first_row = read(OUT / first.name)
        first_complete = dt.datetime.fromtimestamp(first.mtime, dt.timezone.utc)
        last_complete = dt.datetime.fromtimestamp(last.mtime, dt.timezone.utc)
    start = dt.datetime.fromisoformat(LOG_BIRTH_CST)
    elapsed = (last_complete - start).total_seconds()
    nominal = 768 * 512
    actual = all_summary["budget_totals"]["N_unique_transition_evals"]
    validation_nominal = 64 * 5 * 3 * (256 + 512 + 1024)
    diagnostic = {
        "scope": "FORMAL_TRAIN_TUNING_ONLY", "split": "dev_train",
        "method_selection": "NOT_SELECTED", "validation_comparison": "NOT_STARTED",
        "locked_test": False, "gpu_model": frozen["gpu_model"],
        "selected_configs": configs_receipt["configs"],
        "selection_rule": reconstructed["S-CEM"]["selection_order"],
        "per_method_config": per_config, "per_anchor_seed": per_anchor_seed,
        "zero_scoreable_anchor_seed_count": sum(row["zero_scoreable_all_eight_configs"]
                                               for row in per_anchor_seed),
        "zero_scoreable_anchor_ids": sorted({row["sample_id"] for row in per_anchor_seed
                                              if row["zero_scoreable_all_eight_configs"]}),
        "all_cases": all_summary,
        "selected_static_empty_count": eligibility["static_empty_count"],
        "selected_zero_cohort_count": eligibility["cohort_zero_count"],
        "complete_vs_scoreable_note": "Complete counts every finished H4 path, unscoreable counts attempts, and scoreable counts distinct fingerprints; these categories are not an additive partition when scoreable paths repeat.",
        "return_boundary_note": "Counts are repeated scorer boundary events across configurations/seeds, not independent Task births.",
        "runtime": {"start_log_birth_cst": LOG_BIRTH_CST,
                    "start_source": "read-only remote stat -c %w on formal TRAIN log",
                    "first_result_completed_utc": first_complete.isoformat(),
                    "first_result_solve_seconds": first_row["outcome"]["budget_receipt"]["wall_clock_seconds_diagnostic_only"],
                    "last_result_completed_utc": last_complete.isoformat(),
                    "elapsed_wall_seconds": elapsed,
                    "effective_cases_per_hour": 768 * 3600 / elapsed,
                    "actual_unique_transitions_per_second": actual / elapsed,
                    "nominal_train_budget": nominal,
                    "actual_unique_transition_evals": actual,
                    "validation_nominal_budget": validation_nominal,
                    "validation_hours_at_train_effective_transition_rate_diagnostic_only":
                    validation_nominal / actual * elapsed / 3600,
                    "validation_estimate_limitation": "Validation has HRS and B_WM 256/512/1024; B_WM 1024 host-storage may be slower. This is resource planning, not Validation evidence."},
        "h4_search_comparison_readiness": "SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION",
        "source_sha256": frozen["source_sha256"],
        "checkpoint_sha256": EXPECTED_SHA}
    diag_path = OUT / "train_tuning_diagnostic_summary.json"
    write(diag_path, diagnostic)
    checks = {"raw_768_complete": True, "identity_grid_exact": True,
              "no_mixed_execution_source_checkpoint": True,
              "selection_rebuilt_from_raw": True, "train_diagnostic_complete": True,
              "return_boundary_separate": True, "backup_inventory_all_sha_match": True,
              "formal_runtime_recorded": True, "validation_not_started": True,
              "locked_test_false": True,
              "ai_context_synced": all(
                  "2026-10-01 STEP 6.3D" in (ROOT / f"AI_CONTEXT/{name}").read_text(encoding="utf-8")
                  for name in ("00_PROJECT_STATE.md", "05_EXPERIMENTS.md",
                               "07_KNOWN_ISSUES.md", "08_CHANGELOG.md")),
              "archive_integrity_pass": True}
    require(actual <= nominal and actual >= 0, "actual transition count invalid")
    receipt = {"step": "STEP_6_3D_FORMAL_TRAIN_TUNING_CLOSURE",
               "verdict": "PASS" if all(checks.values()) else "BLOCK",
               "checks": checks, "cases": 768, "nominal_budget": nominal,
               "actual_unique_transition_evals": actual,
               "diagnostic_sha256": sha(diag_path),
               "backup_inventory_sha256": sha(OUT / "train_local_backup_inventory.json"),
               "train_receipt_sha256": sha(OUT / "05_train_hyperparameter_tuning_receipt.json"),
               "frozen_configs_sha256": sha(OUT / "06_frozen_selected_cem_configs.json"),
               "remote_stream_tar_sha256": sha(tar_path),
               "selected_configs": configs_receipt["configs"],
               "validation_comparison": "NOT_STARTED", "search_method": "NOT_SELECTED",
               "locked_test": False}
    acceptance_path = OUT / "train_tuning_closure_acceptance.json"
    pending_acceptance = OUT / "train_tuning_closure_acceptance.json.pending"
    write(pending_acceptance, receipt)
    archive_path = OUT / "train_tuning_local_archive.zip"
    archive_files = [OUT / rel for rel in inventory_map] + [
        OUT / "train_local_backup_inventory.json", diag_path]
    archive_names = {path.relative_to(OUT).as_posix() for path in archive_files}
    archive_names.add(acceptance_path.name)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in archive_files:
            archive.write(path, path.relative_to(OUT).as_posix())
        archive.write(pending_acceptance, acceptance_path.name)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None and set(archive.namelist()) == archive_names and
                hashlib.sha256(archive.read(acceptance_path.name)).hexdigest() ==
                sha(pending_acceptance),
                "local archive integrity/member mismatch")
    os.replace(pending_acceptance, acceptance_path)
    manifest_out = {"archive_path": archive_path.relative_to(ROOT).as_posix(),
                    "archive_size_bytes": archive_path.stat().st_size,
                    "archive_sha256": sha(archive_path),
                    "archive_member_count": len(archive_names),
                    "archive_integrity_pass": True,
                    "remote_stream_tar_sha256": sha(tar_path),
                    "backup_inventory_sha256": sha(OUT / "train_local_backup_inventory.json"),
                    "diagnostic_sha256": sha(diag_path),
                    "acceptance_sha256": sha(acceptance_path),
                    "local_persistent_copy": True,
                    "remote_originals_preserved": True,
                    "raw_results_in_git": False}
    write(OUT / "train_tuning_local_archive_manifest.json", manifest_out)
    print(json.dumps({"verdict": receipt["verdict"], "cases": len(rows),
                      "nominal_budget": nominal, "actual_budget": actual,
                      "success": all_summary["h4_scoreable_case_count"],
                      "archive_sha256": manifest_out["archive_sha256"],
                      "runtime_hours": elapsed / 3600}, sort_keys=True))


if __name__ == "__main__":
    main()
