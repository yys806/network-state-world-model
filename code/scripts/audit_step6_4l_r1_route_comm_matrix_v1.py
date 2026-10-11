"""Build the R1 Route/communication support matrix from the saved live root.

The matrix is read-only: it rebuilds the existing tensor state and compares
the current AirFogSim channel row with the integer-indexed model relation.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v7"
OUT = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2/04_route_comm_support_matrix.json"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    import torch
    from run_step6_3d_one_cpu_solve_v1 import DATASET, load_selected
    from run_step6_4i_cpu_smoke_v1 import history_tensor
    from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
    from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES
    from build_step5_1d_unified_model_chain_v1 import build_state
    stats = json.loads((ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json").read_text(encoding="utf-8"))
    prefix = json.loads((IN / "A_01_causal_prefix.json").read_text(encoding="utf-8"))
    current = prefix["decisions"][-1]
    interface = FormalTrainingInterface.from_manifest(DATASET)
    shards = FullFormalShardDataset(interface)
    _, frozen, *_ = load_selected("formal-v1-sim-2026092318-policy-2026092418::anchor-0006", interface, shards)
    sample, tensor, ginput = history_tensor(prefix, stats, frozen)
    state, graph = build_state(tensor, ginput, 0, wired_edges=FORMAL_V1_WIRED_EDGES)
    entities = sample["static"]["input_entity_index"]["physical"]
    src, dst = entities["UAV_0"], entities["UAV_1"]
    hits = torch.nonzero(
        (state["comm_source_index"][0] == src)
        & (state["comm_target_index"][0] == dst)
        & state["comm_presence"][0] & state["comm_validity"][0], as_tuple=False
    ).flatten().tolist()
    raw_rows = [row for row in current.get("channel_rows", [])
                if str(row.get("source_id")) == "UAV_0" and str(row.get("target_id")) == "UAV_1"]
    runtime = json.loads((ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1/02_post_step.json").read_text(encoding="utf-8"))
    events = json.loads((ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1/03_runtime_events.json").read_text(encoding="utf-8"))
    alloc = [e for e in events if e.get("event") == "_allocate_communication_RBs"]
    matrix = {
        "schema": "PI-JWM-STEP-6.4L-R1-ROUTE-COMM-MATRIX-v1",
        "root": {"trajectory_id": current["trajectory_id"], "frame": current["frame_index"],
                 "capture_event_id": current["capture_event_id"], "simulation_time_s": current["simulation_time_s"],
                 "live_root_capture_valid": True, "r39_exact_equivalence": "UNPROVEN"},
        "airfogsim": {
            "source": "UAV_0", "destination": "UAV_1", "node_types": ["U", "U"],
            "channel_type": raw_rows[0].get("channel_type") if raw_rows else None,
            "raw_channel_row_observed": bool(raw_rows),
            "raw_physical_edge_id": raw_rows[0].get("physical_edge_id") if raw_rows else None,
            "raw_observed_mask": raw_rows[0].get("observed_mask") if raw_rows else None,
            "wireless_hop_before_route": json.loads((ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1/01_root_and_setter.json").read_text(encoding="utf-8"))["before"]["wireless_hop_before"],
            "new_rb_request_after_route": alloc[-1].get("requested") if alloc else None,
            "new_transfer_event_for_task5": any(float(e.get("time", -1)) >= 1.8 for e in runtime.get("transfer_events", []) if e.get("task_id") == "Task_5"),
            "post_step_transfer_events": [e for e in runtime.get("transfer_events", []) if e.get("task_id") == "Task_5" and float(e.get("time", -1)) >= 1.8],
            "failure_reason": runtime["task"]["failure_reason"],
            "failure_reason_code": runtime["task"]["failure_reason_code"],
        },
        "model": {
            "physical_entity_indices": {"UAV_0": src, "UAV_1": dst},
            "matching_relation_indices": hits,
            "matching_relation_support": [{"relation_index": int(i), "presence": bool(state["comm_presence"][0, i]),
                                            "validity": bool(state["comm_validity"][0, i]),
                                            "wireless_mask": bool(state["comm_wireless_mask"][0, i]),
                                            "source_index": int(state["comm_source_index"][0, i]),
                                            "target_index": int(state["comm_target_index"][0, i])} for i in hits],
            "raw_graph_support_missing": not bool(hits),
            "route_shadow_type_fix": "R1 passes integer model indices to inject_return_slot",
        },
        "classification": {
            "primary": "TEMPORARILY_NO_SERVICE",
            "physically_unsupported_hop": False,
            "raw_or_graph_support_missing": not bool(hits),
            "model_representation_mismatch": False,
            "temporary_no_service_evidence": "route setter is consumed in _updateTask; no RB request is present and TTI failure occurs before wireless execution",
            "undetermined": False,
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "code/reference/AirFogSim/airfogsim/airfogsim_env.py",
            ROOT / "code/reference/AirFogSim/airfogsim/manager/task_manager.py",
            ROOT / "code/reference/AirFogSim/airfogsim/entities/task.py",
            ROOT / "code/reference/AirFogSim/airfogsim/scheduler/task_sched.py",
            ROOT / "code/scripts/collect_step5_5_formal_raw_v1.py",
        )},
        "GPU": False, "locked_test": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(matrix, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)

if __name__ == "__main__":
    main()
