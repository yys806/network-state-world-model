"""All-anchor static CandidateDomain audit; TRAIN freezes support, val describes.

Reads current Sample/Tensor/Graph and current Raw decision only. Never loads a
Future Target package, model checkpoint, optimizer, GPU or locked test split.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/scripts"))

from build_step5_1d_unified_model_chain_v1 import (
    ENTITY_UAV, COMM_WIRELESS, _current, _graph_current, build_state,
)
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES, _one
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain

MANIFEST = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
CATALOG = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name: str, value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes((json.dumps(value, sort_keys=True, indent=2,
                                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def quantiles(values: list[int]) -> dict[str, int | float]:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("empty anchor distribution")
    return {"min": sorted_values[0], "median": statistics.median(sorted_values),
            "p90": sorted_values[int(0.9 * (len(values) - 1))],
            "p95": sorted_values[int(0.95 * (len(values) - 1))],
            "max": sorted_values[-1]}


def project_domain_state(tensor: dict, graph: dict, index: int) -> dict[str, torch.Tensor]:
    """The exact build_state fields used by 6.3B; no dense physical graph work."""
    entity_type = _current(tensor, "entity_type_index", index, torch.long)
    comm_src = _graph_current(graph, "comm_relations", "source_index", index, torch.long)
    comm_dst = _graph_current(graph, "comm_relations", "target_index", index, torch.long)
    flow_presence = _graph_current(graph, "flow_relations", "presence", index, torch.bool)
    carrying = graph["blocks"]["flow_carrying_state"]
    hop_src = torch.as_tensor(carrying["hop_source_index"][index:index + 1], dtype=torch.long)
    hop_dst = torch.as_tensor(carrying["hop_destination_index"][index:index + 1], dtype=torch.long)
    flow_comm = torch.full(flow_presence.shape, -1, dtype=torch.long)
    for flow in range(flow_presence.shape[1]):
        hits = torch.nonzero((comm_src[0] == hop_src[0, flow]) &
                             (comm_dst[0] == hop_dst[0, flow]), as_tuple=False)
        if len(hits):
            flow_comm[0, flow] = hits[0, 0]
    work = _current(tensor, "task_history_extended_raw_features", index, torch.float32)
    comm_type = _graph_current(graph, "comm_relations", "relation_type_index", index, torch.long)
    task_presence = _graph_current(graph, "task_nodes", "presence", index, torch.bool)
    return {
        "entity_presence": _current(tensor, "entity_presence", index, torch.bool),
        "uav_mask": entity_type == ENTITY_UAV,
        "comm_presence": _graph_current(graph, "comm_relations", "presence", index, torch.bool),
        "comm_validity": _graph_current(graph, "comm_relations", "validity", index, torch.bool),
        "comm_wireless_mask": comm_type == COMM_WIRELESS,
        "comm_source_index": comm_src, "comm_target_index": comm_dst,
        "rb_active_mask": torch.zeros_like(_graph_current(graph, "comm_relations", "csi_mask", index, torch.bool)),
        "flow_known": _graph_current(graph, "flow_relations", "known", index, torch.bool),
        "flow_presence": flow_presence,
        "flow_task_index": _graph_current(graph, "flow_relations", "task_index", index, torch.long),
        "flow_comm_relation_index": flow_comm,
        "carrying_active": torch.as_tensor(carrying["active"][index:index + 1], dtype=torch.bool),
        "carrying_hop_source_index": hop_src,
        "carrying_hop_destination_index": hop_dst,
        "task_presence": task_presence,
        "task_completed": torch.zeros_like(task_presence),
        "task_lifecycle_index": _graph_current(graph, "task_nodes", "lifecycle_index", index, torch.long),
        "task_work_remaining": (work[..., 0] - work[..., 1]).clamp_min(0),
        "task_agent_validity": _graph_current(graph, "task_agent_relations", "validity", index, torch.bool),
        "task_agent_task_index": _graph_current(graph, "task_agent_relations", "task_index", index, torch.long),
        "task_agent_agent_index": _graph_current(graph, "task_agent_relations", "agent_index", index, torch.long),
        "task_agent_relation_type_index": _graph_current(graph, "task_agent_relations", "relation_type_index", index, torch.long),
    }


def audit_split(interface: FormalTrainingInterface, shards: FullFormalShardDataset,
                catalog: TrainStructuralSupportCatalog, split: str) -> dict:
    rows = [row for row in interface.samples if row["metadata"]["split"] == split]
    by_tid: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_tid[str(row["metadata"]["trajectory_id"])].append(row)
    result = []
    for tid, samples_index in sorted(by_tid.items()):
        info = shards.shards[tid]
        for family in ("samples", "tensor", "graph"):
            entry = info["files"][family]
            if digest(shards.paths[family] / entry["path"]) != entry["sha256"]:
                raise ValueError(f"current {family} shard hash mismatch: {tid}")
        with gzip.open(shards.paths["samples"] / f"{tid}.json.gz", "rt", encoding="utf-8") as stream:
            samples = json.load(stream)
        tensors = load_flow_tensor_batch(shards.paths["tensor"] / f"{tid}.npz")
        graph_inputs = load_typed_dual_graph_batch(shards.paths["graph"] / f"{tid}.npz")
        raw_path = ROOT / samples_index[0]["metadata"]["source_path"]
        if digest(raw_path) != samples_index[0]["metadata"]["source_sha256"]:
            raise ValueError(f"Raw source hash mismatch: {tid}")
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        current_decisions = {int(decision["frame_index"]): decision for decision in raw["decisions"]}
        for row in samples_index:
            meta = row["metadata"]
            slot = int(row["shard_index"])
            sample = samples[slot]
            if sample["metadata"]["sample_id"] != meta["sample_id"]:
                raise ValueError("sample identity mismatch")
            tensor = _one(tensors, slot, info["sample_count"])
            graph_input = _one(graph_inputs, slot, info["sample_count"])
            state = project_domain_state(tensors, graph_inputs, slot)
            if slot == 0:
                state_tensor = dict(tensor)
                state_tensor["sample_metadata"] = [{**tensor["sample_metadata"][0],
                    "source_path": str(raw_path.resolve())}]
                full_state, _ = build_state(state_tensor, graph_input, 0,
                                            wired_edges=FORMAL_V1_WIRED_EDGES)
                if any(not torch.equal(value, full_state[key]) for key, value in state.items()):
                    raise AssertionError(f"projected domain state differs from build_state: {tid}")
            context = PlannerCandidateContext.from_sample(sample, state, meta["sample_id"])
            frame = int(meta["anchor_decision_frame"])
            current = current_decisions[frame]
            domain_context = context_from_current_raw(context, current)
            domain = CandidateDomain.from_state(context, domain_context, state,
                                                domain_context.mobility_states, catalog)
            result.append({"sample_id": meta["sample_id"],
                           "trajectory_id": tid,
                           "mode_count": len(domain.modes),
                           "concrete_count": domain.exact_unique_single_step_count,
                           "comm_concrete_count": domain.comm_concrete_count,
                           "comp_mode_count": domain.comp_mode_count,
                           "mob_profile_count": domain.mobility_profile_count,
                           "eligible_wireless_tasks": len(domain.wireless_task_to_relation),
                           "eligible_compute_tasks": len(domain.compute_base),
                           "present_uavs": len(domain.present_uav_slots),
                           "empty_reason": domain.empty_reason})
        print(f"{split}: {len(result)}/{len(rows)} current anchors audited", flush=True)
        del samples, tensors, graph_inputs, raw
    if len(result) != len(rows):
        raise AssertionError("anchor audit did not cover full split")
    result.sort(key=lambda row: row["sample_id"])
    hist = collections.Counter(str(row["concrete_count"]) for row in result)
    reasons = collections.Counter(row["empty_reason"] for row in result if row["empty_reason"])
    return {"split": split, "anchor_count": len(result),
            "empty_domain_count": sum(row["concrete_count"] == 0 for row in result),
            "empty_reasons": dict(sorted(reasons.items())),
            "quantiles": {name: quantiles([row[key] for row in result])
                          for name, key in (("compatible_structural_modes", "mode_count"),
                                            ("exact_unique_concrete_candidates", "concrete_count"),
                                            ("comm_concrete_candidates", "comm_concrete_count"),
                                            ("comp_modes", "comp_mode_count"),
                                            ("mob_profiles", "mob_profile_count"))},
            "cardinality_histogram": dict(sorted(hist.items(), key=lambda item: int(item[0]))),
            "anchors": result,
            "future_target_used": False,
            "checkpoint_loaded": False,
            "locked_test": False, "gpu": False, "training": False}


def main() -> None:
    interface = FormalTrainingInterface.from_manifest(MANIFEST)
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    if interface.dataset_manifest_hash != catalog.dataset_manifest_sha256:
        raise ValueError("TRAIN structural catalog dataset identity mismatch")
    shards = FullFormalShardDataset(interface)
    train = audit_split(interface, shards, catalog, "dev_train")
    write("01_formal_train_domain_feasibility.json", train)
    validation = audit_split(interface, shards, catalog, "dev_validation")
    validation["descriptive_only"] = True
    validation["validation_changed_train_catalog_or_policy"] = False
    write("02_formal_validation_domain_descriptive.json", validation)
    print(json.dumps({"train_anchors": train["anchor_count"],
                      "train_empty": train["empty_domain_count"],
                      "validation_anchors": validation["anchor_count"],
                      "validation_empty": validation["empty_domain_count"]}))


if __name__ == "__main__":
    main()
