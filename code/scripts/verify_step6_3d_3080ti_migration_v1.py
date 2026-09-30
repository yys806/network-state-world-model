"""Read-only identity gate before the RTX 3080 Ti GPU qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, sha256_file
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from run_step6_3d_one_cpu_solve_v1 import (
    CATALOG, CHECKPOINT, DATASET, EXPECTED_SHA, OUT, SOURCE_SHA,
    load_frozen_runtime, load_selected, objective_side,
)

EXPECTED_DATASET_SHA = "6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-sha-list", type=Path, required=True)
    args = parser.parse_args()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         cwd=ROOT, text=True).strip()
    if git_status:
        raise ValueError("target has tracked source changes before migration identity gate")
    if sha256_file(DATASET) != EXPECTED_DATASET_SHA:
        raise ValueError("Formal Dataset manifest SHA mismatch")
    if sha256_file(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("frozen checkpoint SHA mismatch")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    packages = interface.verify_packages()
    runtime = interface.verify_runtime_packages()
    if not packages["all_present_and_matching"] or not runtime["all_present_and_matching"]:
        raise ValueError("Formal package or runtime hash mismatch")
    shards = FullFormalShardDataset(interface)
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    if catalog.dataset_manifest_sha256 != interface.dataset_manifest_hash:
        raise ValueError("structural support catalog Dataset identity mismatch")

    source_rows = args.source_sha_list.read_text(encoding="utf-8").splitlines()
    if len(source_rows) != 266:
        raise ValueError("4090 source file inventory is incomplete")
    source_bad = []
    for row in source_rows:
        expected, relative = row.split("  ", 1)
        if sha256_file(DATASET.parent / relative) != expected:
            source_bad.append(relative)
    if source_bad:
        raise ValueError(f"4090/3080 Ti file SHA mismatch: {source_bad[:3]}")

    raw_rows = read(DATASET.parent / "raw_trajectory_provenance.json")
    if len(raw_rows) != 60:
        raise ValueError("Formal Raw inventory is incomplete")
    raw_bad = []
    for row in raw_rows:
        raw_path = DATASET.parent / "raw" / Path(row["source_path"]).name
        if sha256_file(raw_path) != row["source_sha256"]:
            raw_bad.append(str(raw_path))
    if raw_bad:
        raise ValueError(f"Formal Raw SHA mismatch: {raw_bad[:3]}")

    train = read(OUT / "15_train_anchor_manifest_objective_eligible.json")
    validation = read(OUT / "16_validation_anchor_manifest_objective_eligible.json")
    sidecars = read(OUT / "19_formal_selected_deadline_sidecars.json")
    if len(train["selected"]) != 32 or len(validation["selected"]) != 64 or len(sidecars) != 96:
        raise ValueError("selected anchor or deadline sidecar count mismatch")
    selected = train["selected"] + validation["selected"]
    if len({row["sample_id"] for row in selected}) != 96:
        raise ValueError("selected anchor IDs are not unique")
    cohort_zero = static_empty = deadline_bad = 0
    loaded = []
    for row in selected:
        sample_id = row["sample_id"]
        sample, _, _, state, _, context, operational, decision, meta = load_selected(
            sample_id, interface, shards)
        deadline = sidecars[sample_id]
        if not deadline["alignment_passed"] or deadline["trajectory_id"] != meta["trajectory_id"] or \
                int(deadline["frame_index"]) != int(meta["anchor_decision_frame"]):
            deadline_bad += 1
        side = objective_side(sample, state, context, decision, deadline)
        cohort = sum(not task.terminal_at_anchor and
                     bool(state["task_presence"][0, task.task_index]) and
                     not bool(state["task_completed"][0, task.task_index])
                     for task in side.tasks)
        domain = CandidateDomain.from_state(context, operational, state,
            operational.mobility_states, catalog)
        cohort_zero += cohort == 0
        static_empty += domain.is_empty or domain.exact_unique_single_step_count != row["concrete_count"]
        loaded.append({"sample_id": sample_id, "split": meta["split"],
                       "cohort_count": cohort,
                       "static_candidate_count": domain.exact_unique_single_step_count})
    if cohort_zero or static_empty or deadline_bad:
        raise ValueError("selected anchor eligibility or sidecar alignment mismatch")
    if sum(row["split"] == "dev_train" for row in loaded) != 32 or \
            sum(row["split"] == "dev_validation" for row in loaded) != 64:
        raise ValueError("selected anchor split mismatch")

    model, encoder, prepared, side, loaded_catalog, protocol, meta, parameter_digest = \
        load_frozen_runtime(train["selected"][0]["sample_id"], device="cpu")
    del model, encoder, prepared, side, loaded_catalog, protocol, meta
    import torch
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload["git_commit"] != SOURCE_SHA:
        raise ValueError("checkpoint training-source Git identity mismatch")

    write(args.out_dir / "02_formal_dataset_identity.json", {
        "verdict": "PASS", "target_git_head": git_head,
        "dataset_manifest_sha256": interface.dataset_manifest_hash,
        "train_trajectories": len(shards.train_ids),
        "validation_trajectories": len(shards.validation_ids),
        "train_windows": len(interface.train_indices),
        "validation_windows": len(interface.validation_indices),
        "package_hashes": packages, "runtime_package_hashes": runtime,
        "source_4090_file_count": len(source_rows),
        "source_4090_sha_list_sha256": sha256_file(args.source_sha_list),
        "source_target_file_sha_mismatches": 0,
        "raw_count": len(raw_rows), "raw_sha_mismatches": 0,
        "normalization_source_split": "dev_train",
        "selected_train_anchors_loaded": 32,
        "selected_validation_anchors_loaded": 64,
        "selected_zero_cohort": cohort_zero,
        "selected_static_empty_domain": static_empty,
        "deadline_sidecar_mismatches": deadline_bad,
        "structural_catalog_dataset_sha256": catalog.dataset_manifest_sha256,
        "locked_test": False,
    })
    write(args.out_dir / "03_checkpoint_identity.json", {
        "verdict": "PASS", "checkpoint_sha256": EXPECTED_SHA,
        "data_identity_verified_by_frozen_loader": True,
        "config_identity_verified_by_frozen_loader": True,
        "checkpoint_source_git_commit": payload["git_commit"],
        "checkpoint_source_git_identity_verified": True,
        "parameter_digest": parameter_digest, "locked_test": False,
    })
    print(json.dumps({"verdict": "PASS", "dataset_sha": interface.dataset_manifest_hash,
                      "source_target_files": len(source_rows), "raw": len(raw_rows),
                      "selected_train": 32, "selected_validation": 64,
                      "checkpoint_sha": EXPECTED_SHA}, sort_keys=True))


if __name__ == "__main__":
    main()
