"""Measure numeric Comp differences among projected TRAIN H1 structural passes."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import statistics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

from run_step6_3bc_train_self_replay_v1 import (
    CATALOG, MANIFEST, OUT, ROOT, project_comm, project_domain_state,
)
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep, PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3b_candidate_grammar_v1 import (
    _decode_choice, _wireless_bindings, bind_structured_step,
)
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog


def main() -> None:
    replay_path = OUT / "06_train_h1_projected_self_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    targets = {row["sample_id"] for row in replay["per_anchor"]
               if row["projected_semantic_replay_rejection"] == "PROJECTED_COMP_AMOUNT_DIFFERS"}
    interface = FormalTrainingInterface.from_manifest(MANIFEST)
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    shards = FullFormalShardDataset(interface)
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in interface.samples:
        if row["metadata"]["sample_id"] in targets:
            groups[str(row["metadata"]["trajectory_id"])].append(row)
    deltas: list[float] = []
    relatives: list[float] = []
    examples: list[dict] = []
    affected = set()
    for tid, rows in sorted(groups.items()):
        with gzip.open(shards.paths["samples"] / f"{tid}.json.gz", "rt", encoding="utf-8") as stream:
            samples = json.load(stream)
        tensors = load_flow_tensor_batch(shards.paths["tensor"] / f"{tid}.npz")
        graphs = load_typed_dual_graph_batch(shards.paths["graph"] / f"{tid}.npz")
        raw_path = ROOT / rows[0]["metadata"]["source_path"]
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        decisions = {int(item["frame_index"]): item for item in raw["decisions"]}
        for row in rows:
            meta = row["metadata"]
            slot = int(row["shard_index"])
            sample = samples[slot]
            state = project_domain_state(tensors, graphs, slot)
            context = PlannerCandidateContext.from_sample(sample, state, meta["sample_id"])
            domain = context_from_current_raw(context, decisions[int(meta["anchor_decision_frame"])])
            action = sample["future_action"][0]
            wireless = _wireless_bindings(state, context.static["input_entity_index"]["task"])
            comm, _ = project_comm(action, wireless)
            step = CandidateActionStep(route=(), comm=tuple(comm),
                comp=tuple(action["comp"]["entries"]), mob=tuple(action["mobility"]["entries"]))
            choice = _decode_choice(step, context, domain, state, domain.mobility_states, 0.1)
            bound = bind_structured_step(choice, context, domain, state, domain.mobility_states, catalog)
            expected = {str(item["task_id"]): float(item["allocated_cpu_per_s"])
                        for item in bound.action.comp}
            for item in step.comp:
                old = float(item["allocated_cpu_per_s"])
                new = expected[str(item["task_id"])]
                delta = abs(old - new)
                if delta > 1e-7:
                    affected.add(meta["sample_id"])
                    deltas.append(delta)
                    relatives.append(delta / max(abs(old), 1.0))
                    if len(examples) < 5:
                        examples.append({"sample_id": meta["sample_id"],
                                         "task_id": item["task_id"],
                                         "raw_amount": old, "bound_amount": new,
                                         "absolute_delta": delta})
        print(f"diagnosed {tid}: {len(rows)} targets", flush=True)
    if affected != targets:
        raise ValueError("Comp residual diagnostic did not reproduce all affected anchors")
    result = {
        "source_receipt_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        "affected_anchor_count": len(affected),
        "differing_comp_row_count": len(deltas),
        "absolute_delta_min": min(deltas),
        "absolute_delta_median": statistics.median(deltas),
        "absolute_delta_max": max(deltas),
        "relative_delta_max": max(relatives),
        "examples": examples,
        "acceptance_tolerance_changed": False,
        "gpu": False, "training": False, "locked_test": False,
    }
    (OUT / "08_comp_amount_residual_diagnostic.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "affected_anchor_count", "differing_comp_row_count",
        "absolute_delta_min", "absolute_delta_median", "absolute_delta_max",
        "relative_delta_max")}, sort_keys=True))


if __name__ == "__main__":
    main()
