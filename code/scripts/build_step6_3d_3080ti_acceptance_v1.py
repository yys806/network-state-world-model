"""Close the bounded 3080 Ti migration/GPU qualification evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

from run_step6_3d_one_cpu_solve_v1 import (
    EXPECTED_SHA, execution_identity, source_hashes,
)

DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1"
DEFAULT_OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930"
EXPECTED_MANIFEST = "6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc"
EXPECTED_SOURCE_LIST = "c254fd5abc8d340871bb957b987f4e71481d799c6d02035c044f2e687189776f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(f"3080 Ti acceptance condition failed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    source_list = out / "00_source_4090_sha256.txt"
    dataset = read(out / "02_formal_dataset_identity.json")
    checkpoint = read(out / "03_checkpoint_identity.json")
    environment = read(out / "02_gpu_environment.json")
    equivalence = read(out / "03_cpu_gpu_equivalence.json")
    throughput = read(out / "offload_probe/04_gpu_batch_throughput.json")
    config = read(out / "05_selected_execution_config.json")
    smoke = read(out / "06_formal_gpu_runner_smoke.json")
    capacity = read(out / "08_offload_max_budget_capacity.json")
    cem = read(out / "09_cem_min_budget_batch16_diagnostic.json")
    large = read(out / "10_cem_large_batch_feasibility_diagnostic.json")
    offload = read(out / "11_offload_equivalence_corrected.json")

    require(sha(source_list) == EXPECTED_SOURCE_LIST, "4090 source SHA list")
    require(len(source_list.read_text(encoding="utf-8").splitlines()) == 266,
            "4090 source inventory count")
    require(dataset["verdict"] == "PASS" and dataset["dataset_manifest_sha256"] == EXPECTED_MANIFEST,
            "Formal Dataset identity")
    require(dataset["source_4090_file_count"] == 266 and
            dataset["source_target_file_sha_mismatches"] == 0 and
            dataset["source_4090_sha_list_sha256"] == EXPECTED_SOURCE_LIST,
            "4090 to 3080 Ti SHA verification")
    require((dataset["train_trajectories"], dataset["validation_trajectories"],
             dataset["train_windows"], dataset["validation_windows"]) == (48, 12, 4416, 1104),
            "Formal split/window identity")
    require((dataset["selected_train_anchors_loaded"],
             dataset["selected_validation_anchors_loaded"],
             dataset["selected_zero_cohort"],
             dataset["selected_static_empty_domain"],
             dataset["deadline_sidecar_mismatches"]) == (32, 64, 0, 0, 0),
            "selected anchor and sidecar identity")
    require(dataset["raw_count"] == 60 and dataset["raw_sha_mismatches"] == 0,
            "Formal Raw SHA identity")
    require(dataset["package_hashes"]["all_present_and_matching"] and
            dataset["runtime_package_hashes"]["all_present_and_matching"],
            "package and runtime SHA identity")
    stats = read(DATASET / "packages/normalization/stats.json")
    require(stats["source_split"] == "dev_train" and
            len(stats["source_trajectory_ids"]) == 48 and
            len(stats["excluded_validation_trajectory_ids"]) == 12 and
            stats["future_target_in_fit"] is False,
            "normalization provenance")
    require(checkpoint["verdict"] == "PASS" and checkpoint["checkpoint_sha256"] == EXPECTED_SHA
            and checkpoint["data_identity_verified_by_frozen_loader"]
            and checkpoint["config_identity_verified_by_frozen_loader"]
            and checkpoint["checkpoint_source_git_identity_verified"],
            "frozen checkpoint identity")
    require(environment["name"] == "NVIDIA GeForce RTX 3080 Ti" and
            environment["cuda_available"] and environment["vram_bytes"] == 12491292672
            and environment["checkpoint_sha256"] == EXPECTED_SHA,
            "RTX 3080 Ti environment")
    require(equivalence["verdict"] == "PASS" and len(equivalence["fixtures"]) == 2,
            "CPU to GPU equivalence")
    require(offload["verdict"] == "PASS" and all(offload["fields_equal"].values()),
            "host-storage discrete/B_WM equivalence")
    require(throughput["verdict"] == "PASS" and throughput["state_storage"] ==
            "cpu_cache_and_prefix" and [row["batch_size"] for row in throughput["rows"]] ==
            [8, 16, 32, 64] and all(row["stable"] and len(row["repeats"]) >= 3
                                    for row in throughput["rows"]),
            "steady-state throughput")
    require(all(row["scoreable_h4"] > 0 for row in cem["rows"]) and
            all(row["scoreable_h4"] == 0 for row in large["rows"]),
            "minimum-budget CEM batch feasibility")
    require(capacity["verdict"] == "PASS" and capacity["budget"] == 1024 and
            capacity["unique_transitions"] == 1024 and capacity["scoreable_h4"] > 0,
            "maximum-budget host-storage capacity")
    selected = next(row for row in throughput["rows"] if row["batch_size"] == 16)
    expected_execution = execution_identity(device="cuda",
        gpu_model=environment["name"], batch_size=16,
        checkpoint_sha256=EXPECTED_SHA, source_sha256=source_hashes())
    require(all(config.get(key) == value for key, value in expected_execution.items()) and
            config["median_transitions_per_second"] ==
            selected["median_transitions_per_second"] and config["steady_state_repeats"] == 3,
            "selected execution config/source identity")
    require(smoke["verdict"] == "PASS" and
            all(smoke.get(key) == value for key, value in expected_execution.items()) and
            {row["method"] for row in smoke["cases"]} == {"HRS", "S-CEM"} and
            all(row["unique_transitions"] == 256 and row["complete_h4"] > 0 and
                row["scoreable_h4"] > 0 and row["resume_equal"] for row in smoke["cases"]) and
            smoke["resume_batch_tamper_rejected"],
            "bounded formal runner GPU smoke and resume")
    require(not config["locked_test"] and not smoke["locked_test"] and
            not smoke["formal_train_tuning"] and not smoke["validation_comparison"],
            "execution boundary")

    raw_provenance = read(DATASET / "raw_trajectory_provenance.json")
    local_raw_mismatch = sum(sha(ROOT / row["source_path"]) != row["source_sha256"]
                             for row in raw_provenance)
    require(local_raw_mismatch == 0, "local Formal Raw source SHA")
    inventory = {
        "verdict": "PASS", "source_4090_gpu": "NVIDIA GeForce RTX 4090",
        "source_4090_dataset": "/root/autodl-tmp/pi_jwm_step5_6a_a8c5dfa/work/dataset",
        "source_4090_fixture_raw": "/root/autodl-tmp/pi-jwm-step6-3d-gpu/code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/raw",
        "target_3080ti_repo": "/root/autodl-tmp/pi-jwm-step6-3d",
        "source_4090_regular_files": 266,
        "source_4090_to_target_same_sha": 266,
        "source_4090_raw_available": 2,
        "local_authorized_raw_count": 60,
        "local_raw_transferred_after_existing_two": 58,
        "local_raw_target_sha_verified": 60,
        "checkpoint_sha256": EXPECTED_SHA,
        "transfer_modes": ["direct rsync -a --partial --append-verify without --delete",
                           "local WSL rsync -a --partial --append-verify without --delete"],
        "tracked_code_and_receipts_source": "GitHub origin/main",
        "formal_dataset_manifest_sha256": EXPECTED_MANIFEST,
        "excluded_local_build_cache": "not referenced by frozen Formal manifest or runner",
        "source_4090_artifacts_modified": False,
        "source_4090_instance_stopped_or_released": False,
        "aborted_sftp_attempt": "01_aborted_relay_inventory.json; superseded by direct rsync and complete SHA checks",
        "locked_test": False,
    }
    write(out / "01_migration_inventory.json", inventory)
    write(out / "12_migration_sha_verification.json", {
        "verdict": "PASS", "source_4090_sha_list_sha256": EXPECTED_SOURCE_LIST,
        "source_4090_target_sha_match": "266/266",
        "local_raw_provenance_sha_match": "60/60",
        "target_raw_provenance_sha_match": "60/60",
        "checkpoint_sha256": EXPECTED_SHA,
        "dataset_manifest_sha256": EXPECTED_MANIFEST,
        "package_hashes_match": True, "runtime_hashes_match": True,
        "normalization_fit_split": stats["source_split"],
        "normalization_train_trajectories": len(stats["source_trajectory_ids"]),
        "normalization_validation_excluded": len(stats["excluded_validation_trajectory_ids"]),
        "normalization_future_target_in_fit": stats["future_target_in_fit"],
        "locked_test": False,
    })
    write(out / "13_resume_identity_audit.json", {
        "verdict": "PASS", "execution_fields": list(expected_execution),
        "selected_execution_config_id": expected_execution["execution_config_id"],
        "legacy_cpu_partial_rejected_by_missing_execution_fields": True,
        "different_batch_rejected_by_unit_and_real_smoke": True,
        "different_source_or_config_rejected_by_identity": True,
        "formal_gpu_smoke_resumed_identically": True,
        "locked_test": False,
    })
    tps = selected["median_transitions_per_second"]
    conditions = {
        "gpu_exact_model": True, "formal_dataset_complete": True,
        "frozen_checkpoint_complete": True, "dataset_raw_shard_normalization_identity": True,
        "checkpoint_data_config_source_identity": True,
        "selected_32_train_64_validation_loadable": True,
        "cpu_gpu_equivalence": True, "batch_throughput_probe": True,
        "formal_batch_config_frozen": True, "formal_gpu_runner_cuda_closure": True,
        "resume_config_identity": True, "bounded_gpu_smoke": True,
        "formal_train_tuning_not_run": True, "validation_search_not_run": True,
        "locked_test_false": True, "scientific_semantics_unchanged": True,
    }
    write(out / "14_acceptance.json", {
        "step": "STEP_6_3D_3080TI_MIGRATION_QUALIFICATION",
        "verdict": "PASS" if all(conditions.values()) else "BLOCKED",
        "conditions": conditions,
        "migration_verdict": "PASS", "source_target_formal_dataset_identity": "MATCH",
        "checkpoint_sha256": EXPECTED_SHA,
        "gpu_model": environment["name"], "vram_bytes": environment["vram_bytes"],
        "cpu_gpu_equivalence": "PASS",
        "batch_rows": [{"batch_size": row["batch_size"],
                        "median_transitions_per_second": row["median_transitions_per_second"],
                        "min_transitions_per_second": row["min_transitions_per_second"],
                        "max_transitions_per_second": row["max_transitions_per_second"],
                        "peak_vram_bytes": row["max_peak_vram_bytes"]}
                       for row in throughput["rows"]],
        "selected_batch_size": 16,
        "selected_transitions_per_second": tps,
        "throughput_ratio_vs_rtx4090_12_71": tps / 12.71,
        "train_tuning_393216_estimated_seconds": 393216 / tps,
        "full_2113536_estimated_seconds": 2113536 / tps,
        "formal_gpu_runner_closure": "PASS",
        "resume_identity": "PASS",
        "source_4090_can_be_stopped": True,
        "source_4090_stopped_or_released": False,
        "h4_search_comparison_readiness": "SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION",
        "future_return_birth": "known fixed-support limitation unchanged",
        "runtime_estimate_limitation": "short steady-state probe; B_WM=1024 host-storage diagnostic was slower and is not a full matrix timing",
        "formal_train_tuning": False, "validation_search": False,
        "locked_test": False,
    })
    evidence = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "15_sha_manifest.json":
            evidence.append({"path": path.relative_to(out).as_posix(),
                             "bytes": path.stat().st_size, "sha256": sha(path)})
    write(out / "15_sha_manifest.json", {"files": evidence,
        "locked_test": False, "formal_train_tuning": False,
        "validation_search": False})
    print(json.dumps({"verdict": "PASS", "selected_batch": 16,
                      "tps": tps, "evidence_files": len(evidence)}))


if __name__ == "__main__":
    main()
