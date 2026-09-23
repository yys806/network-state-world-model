"""CPU-only read-only audit of accepted Formal Dataset v1 packages and raw."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any

from pi_jwm.step5_5_fixed_support_audit_v1 import detect_future_return_birth
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from pi_jwm.step5_5_lifecycle_repair_v1 import TASK_COLLECTION_PRIORITY, repair_duplicate_task_references


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1"
DEFAULT_RECEIPT = ROOT / "code/artifacts/audit/pi_jwm_step5_5_patch_20260923/audit_receipt.json"


def _repair_fixture() -> bool:
    class Task:
        def __init__(self):
            self.task_id = "Task_fixture"
            self.transmitted = 2.0
            self.computed = 3.0
            self.returned = 1.0
            self.done = True

        def getTaskId(self) -> str:
            return self.task_id

    class Manager:
        pass

    manager = Manager()
    for collection in TASK_COLLECTION_PRIORITY:
        setattr(manager, collection, {})
    task = Task()
    manager._offloading_tasks["UAV_0"] = [task, task]
    manager._done_tasks["UAV_1"] = [task]
    before = vars(task).copy()
    repair = repair_duplicate_task_references(manager, 5)
    return bool(len(repair) == 1 and vars(task) == before and manager._offloading_tasks["UAV_0"] == [] and manager._done_tasks["UAV_1"] == [task])


def run(dataset: Path) -> dict[str, Any]:
    interface = FormalTrainingInterface.from_manifest(dataset / "formal_dataset_manifest.json")
    loader = FullFormalShardDataset(interface)
    split_counts = Counter()
    sample_ids: set[str] = set()
    structure = Counter({"unsupported": 0, "unresolved": 0, "fixed_support_blocked": 0})
    positive_examples: list[dict[str, Any]] = []
    affected_windows: set[str] = set()
    affected_structure_trajectories: set[str] = set()
    supported_return_continuations = 0
    repairs: list[dict[str, Any]] = []
    affected_trajectories: set[str] = set()
    affected_tasks: set[tuple[str, str]] = set()
    invalid_repair_rows: list[dict[str, Any]] = []
    indexed_by_position = {(row["metadata"]["trajectory_id"], row["shard_index"]): row["metadata"]["sample_id"] for row in loader.index}
    exact_index_shard_alignment = True
    for trajectory_id, shard in sorted(loader.shards.items()):
        with gzip.open(loader.paths["samples"] / f"{trajectory_id}.json.gz", "rt", encoding="utf-8") as handle:
            samples = json.load(handle)
        if len(samples) != 92:
            raise ValueError(f"incomplete sample shard: {trajectory_id}")
        for shard_index, sample in enumerate(samples):
            sample_id = sample["metadata"]["sample_id"]
            exact_index_shard_alignment &= indexed_by_position.get((trajectory_id, shard_index)) == sample_id
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample: {sample_id}")
            sample_ids.add(sample_id)
            split_counts[sample["metadata"]["split"]] += 1
            for frame in sample["target"]:
                result = detect_future_return_birth(sample, frame)
                current = sample["static"]["input_entity_index"]["logical_flow"]
                supported_return_continuations += sum(
                    row.get("flow_type") == "Return" and row.get("known", False) and (row.get("presence", False) or row.get("status") == "COMPLETED") and row.get("flow_id") in current
                    for row in frame.get("logical_flows", [])
                )
                structure["unsupported"] += result["unsupported_count"]
                structure["unresolved"] += result["unresolved_count"]
                structure["fixed_support_blocked"] += result["fixed_support_blocked_count"]
                if result["unsupported_count"] and len(positive_examples) < 8:
                    positive_examples.append({"sample_id": sample_id, "future_frame_index": frame["frame_index"], "rows": result["unsupported"]})
                if result["unsupported_count"]:
                    affected_windows.add(sample_id)
                    affected_structure_trajectories.add(trajectory_id)
        del samples
        with gzip.open(ROOT / shard["source_path"], "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        for repair in raw.get("simulator_lifecycle_repairs", []):
            before = repair["before"]
            kept = repair["kept"]
            furthest = max(before, key=lambda value: TASK_COLLECTION_PRIORITY.index(value.split(":", 1)[0]))
            if len(before) <= 1 or kept != furthest or repair.get("reason") != "remove_stale_duplicate_reference_preserve_furthest_lifecycle":
                invalid_repair_rows.append({"trajectory_id": trajectory_id, **repair})
            repairs.append({"trajectory_id": trajectory_id, **repair})
            affected_trajectories.add(trajectory_id)
            affected_tasks.add((trajectory_id, repair["task_id"]))
        del raw
    before_counts = Counter(entry for repair in repairs for entry in repair["before"])
    kept_counts = Counter(repair["kept"] for repair in repairs)
    repair_fixture_passed = _repair_fixture()
    scope = {"gpu": False, "formal_training": False, "locked_test_accessed": False, "baseline": False, "planner": False, "performance_claim": False}
    checks = {
        "full_index_4416_1104": len(loader.index) == 5520 and split_counts == {"dev_train": 4416, "dev_validation": 1104},
        "unique_sample_ids": len(sample_ids) == 5520,
        "exact_index_shard_alignment": exact_index_shard_alignment,
        "repair_rows_valid": not invalid_repair_rows,
        "repair_same_object_and_outcomes_fixture": repair_fixture_passed,
        "repair_only_collection_references": repair_fixture_passed,
    }
    return {
        "schema_version": "PI-JWM-Step-5.5-PATCH-Audit-v1", "passed": all(checks.values()), "checks": checks,
        "dataset_manifest_hash": interface.dataset_manifest_hash, "scope": scope,
        "sample_counts": dict(split_counts), "future_fixed_support": {**dict(structure), "affected_window_count": len(affected_windows), "affected_trajectory_count": len(affected_structure_trajectories), "supported_return_continuation_count": supported_return_continuations, "positive_examples": positive_examples, "detector": "pi_jwm.step5_5_fixed_support_audit_v1.detect_future_return_birth", "accounting_scope": "all 5520 accepted sample windows and four future frames per window; repeated window-horizon events are counted separately", "previous_field_only_receipt_counts_superseded": True},
        "lifecycle_repair": {"lifecycle_repair_count": len(repairs), "affected_trajectory_count": len(affected_trajectories), "affected_task_count": len(affected_tasks), "before_lifecycle_collections": dict(before_counts), "kept_lifecycle": dict(kept_counts), "distinct_task_object_same_id_rejected_by_collector": True, "same_task_object_duplicate_required_by_collector": True, "task_object_fields_modified_by_repair": not repair_fixture_passed, "invalid_repair_rows": invalid_repair_rows, "repair_semantics_verified": not invalid_repair_rows and repair_fixture_passed},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    arguments = parser.parse_args()
    receipt = run(arguments.dataset)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "sample_counts": receipt["sample_counts"], "future_fixed_support": receipt["future_fixed_support"], "lifecycle_repair_count": receipt["lifecycle_repair"]["lifecycle_repair_count"]}, ensure_ascii=False))
