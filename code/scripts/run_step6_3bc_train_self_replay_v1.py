"""Formal TRAIN H1 raw-to-Planner-v1 projected self-replay audit."""
from __future__ import annotations
import collections, gzip, hashlib, json, math, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src")); sys.path.insert(0, str(ROOT / "code/scripts"))
from build_step5_1d_unified_model_chain_v1 import ENTITY_UAV, COMM_WIRELESS, _current, _graph_current, build_state
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES, _one
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep, PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3b_candidate_grammar_v1 import (_decode_choice, bind_structured_step, _wireless_bindings,
                                                   CandidateGrammarViolation)
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from run_step6_3c_candidate_domain_audit_v1 import project_domain_state

MANIFEST = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
CATALOG = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def project_comm(action, wireless):
    route_tasks = {str(row["task_id"]) for row in action["route"]["entries"]}
    included, excluded = [], []
    for row in action["comm"]["entries"]:
        task_id = str(row["task_id"])
        if task_id in wireless:
            included.append(row)
        else:
            reason = ("ROUTE_CREATED_FLOW_NOT_AVAILABLE_IN_CURRENT_STATE" if task_id in route_tasks
                      else "NO_UNIQUE_CURRENT_WIRELESS_FLOW_WITHOUT_ROUTE_ROW")
            excluded.append({"row": row, "reason": reason})
    return included, excluded

def comm_projection_reason(rows, wireless, catalog, n_rb):
    if any(str(row["task_id"]) not in wireless for row in rows):
        return "COMM_TASK_OUTSIDE_CAUSAL_ELIGIBILITY"
    signature = "NOOP" if not rows else "rows:" + ",".join(map(str, sorted(len(row["rb_indices"]) for row in rows)))
    if signature not in catalog.comm_selected_task_counts:
        return "COMM_STRUCTURAL_SIGNATURE_UNSEEN_AFTER_PROJECTION"
    if len({str(row["task_id"]) for row in rows}) not in catalog.comm_selected_task_counts.get(signature, frozenset()):
        return "COMM_SELECTED_TASK_COUNT_OUTSIDE_TRAIN_SUPPORT"
    for row in rows:
        rb = tuple(sorted(map(int, row["rb_indices"])))
        starts = [s for s in range(n_rb) if tuple(sorted((s+i)%n_rb for i in range(len(rb)))) == rb]
        if len(starts) != 1 or (starts[0], len(rb)) not in catalog.comm_start_width_pairs:
            return "COMM_BLOCK_PAIR_UNSUPPORTED_OR_NONCYCLIC"
    return None

def semantic_replay_reason(projected, bound):
    """Compare the shared action fields after causal rebinding, ignoring row order."""
    if projected["route"] or bound.action.route:
        return "PROJECTED_ROUTE_NOT_EMPTY"
    raw_comm = sorted((str(row["task_id"]), int(row["task_index"]),
                       tuple(sorted(map(int, row["rb_indices"])))) for row in projected["comm"])
    bound_comm = sorted((str(row["task_id"]), int(row["task_index"]),
                         tuple(sorted(map(int, row["rb_indices"])))) for row in bound.action.comm)
    if raw_comm != bound_comm:
        return "PROJECTED_COMM_ROWS_DIFFER"
    raw_comp = sorted((str(row["task_id"]), str(row["node_id"]),
                       float(row["allocated_cpu_per_s"])) for row in projected["comp"])
    bound_comp = sorted((str(row["task_id"]), str(row["node_id"]),
                         float(row["allocated_cpu_per_s"])) for row in bound.action.comp)
    if len(raw_comp) != len(bound_comp) or any(a[:2] != b[:2] for a, b in zip(raw_comp, bound_comp)):
        return "PROJECTED_COMP_ROWS_DIFFER"
    if any(not math.isclose(a[2], b[2], rel_tol=0, abs_tol=1e-7)
           for a, b in zip(raw_comp, bound_comp)):
        return "PROJECTED_COMP_AMOUNT_DIFFERS"
    raw_mob = sorted(projected["mobility"], key=lambda row: int(row["uav_index"]))
    bound_mob = sorted(bound.action.mob, key=lambda row: int(row["uav_index"]))
    if len(raw_mob) != len(bound_mob):
        return "PROJECTED_MOB_ROWS_DIFFER"
    for a, b in zip(raw_mob, bound_mob):
        if int(a["uav_index"]) != int(b["uav_index"]):
            return "PROJECTED_MOB_ROWS_DIFFER"
        if any(not math.isclose(float(a[field]), float(b[field]), rel_tol=0, abs_tol=1e-7)
               for field in ("azimuth_rad", "elevation_rad", "speed_mps")):
            return "PROJECTED_MOB_VALUES_DIFFER"
    return None

def main():
    interface = FormalTrainingInterface.from_manifest(MANIFEST)
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    shards = FullFormalShardDataset(interface)
    rows = [r for r in interface.samples if r["metadata"]["split"] == "dev_train"]
    by_tid = collections.defaultdict(list)
    for row in rows: by_tid[str(row["metadata"]["trajectory_id"])].append(row)
    comm_fail = collections.Counter(); full_fail = collections.Counter(); excluded_reasons = collections.Counter()
    comm_pass = 0; full_pass = 0; semantic_pass = 0; details=[]
    semantic_fail = collections.Counter()
    raw_comm_rows = projected_comm_rows = raw_route_rows = 0
    for tid, samples in sorted(by_tid.items()):
        info = shards.shards[tid]
        for family in ("samples", "tensor", "graph"):
            entry = info["files"][family]
            if digest(shards.paths[family] / entry["path"]) != entry["sha256"]:
                raise ValueError(f"{family} shard hash mismatch: {tid}")
        with gzip.open(shards.paths["samples"] / f"{tid}.json.gz", "rt", encoding="utf8") as f: sample_rows=json.load(f)
        tensors=load_flow_tensor_batch(shards.paths["tensor"] / f"{tid}.npz")
        graphs=load_typed_dual_graph_batch(shards.paths["graph"] / f"{tid}.npz")
        raw_path=ROOT / samples[0]["metadata"]["source_path"]
        if digest(raw_path) != samples[0]["metadata"]["source_sha256"]:
            raise ValueError(f"Raw source hash mismatch: {tid}")
        with gzip.open(raw_path,"rt",encoding="utf8") as f: raw=json.load(f)
        decisions={int(x["frame_index"]):x for x in raw["decisions"]}
        for row in samples:
            meta=row["metadata"]; slot=int(row["shard_index"]); sample=sample_rows[slot]
            tensor=_one(tensors,slot,info["sample_count"]); graph=_one(graphs,slot,info["sample_count"])
            state=project_domain_state(tensors,graphs,slot)
            if slot==0:
                st=dict(tensor); st["sample_metadata"]=[{**tensor["sample_metadata"][0],"source_path":str(raw_path.resolve())}]
                full,_=build_state(st,graph,0,wired_edges=FORMAL_V1_WIRED_EDGES)
                if any(not torch.equal(v,full[k]) for k,v in state.items()): raise AssertionError("state projection mismatch")
            context=PlannerCandidateContext.from_sample(sample,state,meta["sample_id"])
            domain=context_from_current_raw(context,decisions[int(meta["anchor_decision_frame"])])
            action=sample["future_action"][0]
            wireless=_wireless_bindings(state,context.static["input_entity_index"]["task"])
            projected_comm, excluded_comm = project_comm(action, wireless)
            raw_comm_rows += len(action["comm"]["entries"])
            projected_comm_rows += len(projected_comm)
            raw_route_rows += len(action["route"]["entries"])
            excluded_reasons.update(row["reason"] for row in excluded_comm)
            projected_action = {"route": [], "comm": projected_comm,
                                "comp": action["comp"]["entries"],
                                "mobility": action["mobility"]["entries"]}
            step=CandidateActionStep(route=(),comm=tuple(projected_comm),comp=tuple(projected_action["comp"]),mob=tuple(projected_action["mobility"]))
            comm_reason = comm_projection_reason(projected_comm, wireless, catalog,
                                                 int(state["rb_active_mask"].shape[-1]))
            if comm_reason is None:
                comm_pass += 1
            else:
                comm_fail[comm_reason] += 1
            full_reason = None
            semantic_reason = None
            try:
                choice=_decode_choice(step,context,domain,state,domain.mobility_states,0.1)
                bound=bind_structured_step(choice,context,domain,state,domain.mobility_states,catalog)
                if not bound.support.formal_pool_admitted: raise CandidateGrammarViolation(",".join(bound.support.reason_codes))
                full_pass += 1
                semantic_reason = semantic_replay_reason(projected_action, bound)
                if semantic_reason is None:
                    semantic_pass += 1
                else:
                    semantic_fail[semantic_reason] += 1
            except (ValueError, KeyError, TypeError) as exc:
                full_reason=str(exc).split(":")[0]
                full_fail[full_reason] += 1
            details.append({"sample_id":meta["sample_id"],
                            "raw_historical_action":action,
                            "planner_v1_projected_action":projected_action,
                            "excluded_comm_rows":excluded_comm,
                            "excluded_route_rows":[{"row":r,"reason":"PLANNER_V1_ROUTE_EXPLICIT_NOOP"}
                                                   for r in action["route"]["entries"]],
                            "comm_projection_rejection":comm_reason,
                            "full_projected_rejection":full_reason,
                            "projected_semantic_replay_rejection":semantic_reason})
        del sample_rows,tensors,graphs,raw
        print(f"audited {tid}: {len(samples)} H1 anchors", flush=True)
    out={"source":"FORMAL_TRAIN_ONLY","anchor_count":len(rows),
         "raw_comm_row_count":raw_comm_rows,"projected_comm_row_count":projected_comm_rows,
         "raw_route_row_count":raw_route_rows,"excluded_comm_row_reasons":dict(sorted(excluded_reasons.items())),
         "comm_task_selection_rejections":comm_fail.get("COMM_SELECTED_TASK_COUNT_OUTSIDE_TRAIN_SUPPORT",0),
         "comm_projection_admitted_count":comm_pass,"comm_projection_failures":dict(sorted(comm_fail.items())),
         "full_projected_admitted_count":full_pass,"full_projected_failures":dict(sorted(full_fail.items())),
         "full_projected_pass_meaning":"decode + rebind + support admission",
         "projected_semantic_replay_pass":semantic_pass,
         "projected_semantic_replay_mismatches":dict(sorted(semantic_fail.items())),
         "semantic_comparison":"shared Comm/Comp/Mob action fields; row order ignored; numeric absolute tolerance 1e-7",
         "per_anchor":details,"route_projected_empty":True,"future_target_used":False,
         "validation_used":False,"locked_test":False,"gpu":False,"training":False}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"06_train_h1_projected_self_replay.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps({k:out[k] for k in ("anchor_count","excluded_comm_row_reasons","comm_task_selection_rejections","comm_projection_admitted_count","comm_projection_failures","full_projected_admitted_count","full_projected_failures","projected_semantic_replay_pass","projected_semantic_replay_mismatches")},sort_keys=True))
if __name__ == "__main__": main()
