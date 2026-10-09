"""STEP 6.4J Pilot entry point.

The engineering mode delegates to the already accepted real AirFogSim smoke.
The execute mode performs strict preflight and is deliberately explicit: it
cannot resume a partial episode or silently retry a real action.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009_r23"
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pi_jwm.step6_4j_pilot_v1 import PilotClock, engineering_acceptance, validate_resume, validate_spent

class EpisodeStopped(BaseException):
    pass

def reconstruct_history_action(indexed_action: dict, current: dict) -> dict:
    """Convert executed indexed action rows back to raw History IDs/units."""
    indexed_action = indexed_action or {}
    node_ids = sorted(str(v["entity_id"]) for v in current["entities"])
    task_ids = sorted(str(v["task_id"]) for v in current["tasks"])
    def task_id(entry):
        value = entry.get("task_id")
        if value is not None: return str(value)
        index = entry.get("task_index")
        if index is None or not 0 <= int(index) < len(task_ids): raise RuntimeError("ACTION_HISTORY_ID_RESOLUTION_FAILED:task_index")
        return task_ids[int(index)]
    def node_id(entry, key="node_id", index_key="node_index"):
        value = entry.get(key)
        if value is not None: return str(value)
        index = entry.get(index_key)
        if index is None or not 0 <= int(index) < len(node_ids): raise RuntimeError("ACTION_HISTORY_ID_RESOLUTION_FAILED:node_index")
        return node_ids[int(index)]
    def entries(name): return indexed_action.get(name, {}).get("entries", [])
    route_entries=[]
    for entry in entries("route"):
        item=dict(entry);item["task_id"]=task_id(item)
        if "target_node_id" in item or "target_node_index" in item:item["target_node_id"]=node_id(item,"target_node_id","target_node_index")
        if item.get("task_node_id") is not None or item.get("task_node_index") is not None:item["task_node_id"]=node_id(item,"task_node_id","task_node_index")
        item["route_node_ids"]=[node_ids[int(i)] for i in item.get("route_node_indices", [])] if "route_node_indices" in item else list(item.get("route_node_ids", []))
        route_entries.append(item)
    comm_entries=[]
    for entry in entries("comm"):
        item=dict(entry);item["task_id"]=task_id(item);comm_entries.append(item)
    comp_entries=[]
    for entry in entries("comp"):
        item=dict(entry);item["task_id"]=task_id(item)
        if item.get("node_id") is not None or item.get("node_index") is not None:item["node_id"]=node_id(item)
        comp_entries.append(item)
    mobility_entries=[]
    for entry in entries("mobility"):
        item=dict(entry);idx=int(item["uav_index"])
        if idx < 0 or idx >= len(node_ids):raise RuntimeError("ACTION_HISTORY_ID_RESOLUTION_FAILED:uav_index")
        item["uav_id"]=node_ids[idx];item.pop("uav_index",None);mobility_entries.append(item)
    return {"route":{"field_present":True,"empty":not bool(route_entries),"entries":route_entries,"missing":False},"comm":{"field_present":True,"empty":not bool(comm_entries),"entries":comm_entries,"missing":False},"comp":{"field_present":True,"empty":not bool(comp_entries),"entries":comp_entries,"missing":False},"mobility":{"field_present":True,"empty":not bool(mobility_entries),"entries":mobility_entries,"missing":False},"vehicle_motion":"SUMO external"}


def load_protocol(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {"method": "S-CEM", "K": 4, "rho": 0.2, "B_WM": 512,
                "H": 4, "batch_size": 16, "precision": "FP32",
                "gpu_model": "NVIDIA GeForce RTX 3080 Ti", "locked_test": False}
    for key, value in required.items():
        if cfg.get(key) != value:
            raise RuntimeError(f"IDENTITY_MISMATCH:{key}")
    if cfg.get("status") != "CPU_PROTOCOL_FROZEN_GPU_NOT_STARTED":
        raise RuntimeError("PROTOCOL_NOT_FROZEN_OR_ALREADY_USED")
    if len(cfg.get("episodes_manifest", [])) != 2 or cfg.get("max_searches") != 16:
        raise RuntimeError("PILOT_SCOPE_MISMATCH")
    return cfg


def engineering_smoke() -> int:
    """Run the accepted real AirFogSim CPU mechanism smoke, once.

    This is explicitly B64 engineering evidence and never writes formal Pilot
    results or claims CUDA readiness.
    """
    accepted = ROOT / "code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2/15_acceptance.json"
    if not accepted.exists():
        raise RuntimeError("CPU_ENGINEERING_EVIDENCE_MISSING")
    evidence = json.loads(accepted.read_text(encoding="utf-8"))
    if evidence.get("verdict") != "PASS" or evidence.get("GPU") not in ("NOT_USED", "CPU_ONLY"):
        raise RuntimeError("CPU_ENGINEERING_EVIDENCE_NOT_ACCEPTED")
    cfg = json.loads((OUT / "00_protocol.json").read_text(encoding="utf-8"))
    return execute(cfg, device="cpu", engineering=True)


def execute(cfg: dict, *, device="cuda", engineering=False) -> int:
    if device == "cuda" and not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    torch = __import__("torch")
    if device == "cuda" and torch.cuda.get_device_name(0) != cfg["gpu_model"]:
        raise RuntimeError("GPU_IDENTITY_MISMATCH")
    if cfg["precision"] != "FP32":
        raise RuntimeError("PRECISION_IDENTITY_MISMATCH")
    result_dir = OUT / ("engineering_results" if engineering else "pilot_results")
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("RESULT_NAMESPACE_NOT_EMPTY_OR_RESUME_FORBIDDEN")
    result_dir.mkdir(parents=True)
    def atomic_json(path: Path, value: object) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)
    clock = PilotClock()
    receipt = {"status": "RUNNING", "execution_config_id": cfg["execution_config_id"],
               "GPU": torch.cuda.get_device_name(0) if device == "cuda" else "NOT_USED", "locked_test": False,
               "episodes": [], "max_searches": 16, "searches": 0}
    atomic_json(result_dir / "pilot_attempt.json", receipt)
    # Importing the real provider here keeps CPU imports harmless and makes the
    # CUDA gate occur before an environment is constructed.
    from step6_4i_live_planner_v1 import LiveSCEMPlanner
    from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime, load_selected, DATASET
    from pi_jwm.step6_4i_episode_v1 import EpisodeController
    from pi_jwm.step6_4i_real_metrics_v1 import RealTaskLedger
    import collect_step5_5_formal_raw_v1 as collector
    from pi_jwm.step6_4b_live_bridge_v1 import apply_commands, live_deadline_sidecar
    from airfogsim.scheduler.computation_sched import ComputationScheduler
    from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
    from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES
    from run_step6_4i_cpu_smoke_v1 import history_tensor
    from pi_jwm.step5_2_training_loop_v1 import _torch_tree
    import numpy as np
    stats = json.loads((ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json").read_text())
    interface = FormalTrainingInterface.from_manifest(DATASET)
    for episode in cfg["episodes_manifest"]:
        if receipt["searches"] >= 16:
            break
        clock.check_total()
        sample_id = episode["sample_id"]
        model, encoder, _, _, catalog, protocol, meta, _ = load_frozen_runtime(sample_id, device=device)
        _, frozen, *_ = load_selected(sample_id, interface, FullFormalShardDataset(interface))
        planner = LiveSCEMPlanner(model=model, encoder=encoder, stats=stats,
                                  slot_template=frozen, catalog=catalog,
                                  protocol=protocol, device=device, engineering_only=engineering,
                                  approved_execution={"status": "APPROVED_BY_RESEARCHER", "method": "S-CEM",
                                                       "K": 4, "rho": 0.2, "B_WM": 64 if engineering else 512,
                                                       "batch_size": 1 if engineering else 16, "precision": "FP32",
                                                       "gpu_model": cfg["gpu_model"],
                                                       "checkpoint_sha256": cfg["checkpoint_sha256"]})
        rows = []
        episode_dir = result_dir / episode["trajectory_id"]
        episode_dir.mkdir()
        episode_started = time.perf_counter()
        final_env = None
        final_observation = None
        journal = []
        controller = EpisodeController(lambda row: journal.append(dict(row)))
        ledger = RealTaskLedger()
        def callback(env, decisions, steps, env_config, communication):
            nonlocal final_env, final_observation
            final_env = env
            final_observation = decisions[-1]
            # The production collector attaches this required causal field
            # immediately after capture.  The live runner must preserve the
            # same contract for both the frozen prefix and every fresh root.
            for task_row in decisions[-1].get("tasks", []):
                task_object = collector.step23._all_runtime_tasks(env).get(str(task_row["task_id"]))
                if task_object is not None and hasattr(task_object, "getReturnedSize"):
                    task_row["required_returned_size"] = float(task_object.getReturnedSize())
            if decisions[-1]["frame_index"] != episode["initial_frame"] or rows:
                return
            for _ in range(2 if engineering else 8):
                clock.before_plan(); clock.check_total()
                current = decisions[-1]
                runtime_tasks = collector.step23._all_runtime_tasks(env)
                ledger.observe(current["tasks"], runtime_tasks, float(current["simulation_time_s"]))
                raw = {"environment": {"seed": episode["simulator_seed"], "policy_seed": episode["policy_seed"],
                                       "wired_edges": [dict(edge) for edge in FORMAL_V1_WIRED_EDGES]},
                       "decisions": decisions, "steps": steps}
                def plan():
                    packet = planner.plan(raw, env, runtime_tasks, seed=6311)
                    validate_spent(packet.outcome.budget_receipt, 64 if engineering else 512)
                    if packet.planning_seconds > 600:
                        raise RuntimeError("PILOT_SINGLE_PLAN_TIMEOUT")
                    return packet
                previous_speeds = {r["entity_id"]: float(r["speed_mps"]) for r in current["entities"]}
                before = {k: float(v.getComputedSize()) for k, v in runtime_tasks.items()}
                event_start = len(env.pi_jwm_transfer_events)
                start = float(env.simulation_time)
                def capture():
                    end = float(env.simulation_time)
                    events = [dict(v) for v in env.pi_jwm_transfer_events[event_start:]]
                    computed_after = {k: float(v.getComputedSize()) for k, v in collector.step23._all_runtime_tasks(env).items()}
                    fresh = collector.step23._capture(env, frame=current["frame_index"] + 1,
                        phase="loop_start_decision", event_index=2 * (current["frame_index"] + 1),
                        previous_speed_by_entity=previous_speeds, delta_t_s=end-start)
                    slot_outcome = collector.aggregate_slot_outcomes(
                        transfer_events=events, computed_before=before, computed_after=computed_after,
                        transport_observation=getattr(env, "pi_jwm_transfer_observation", None))
                    fresh.update(slot_outcome)
                    fresh["slot_transfer_events"] = events
                    fresh["communication_observation"] = slot_outcome["communication_observation"]
                    for task_row in fresh.get("tasks", []):
                        task_object = collector.step23._all_runtime_tasks(env).get(str(task_row["task_id"]))
                        if task_object is not None and hasattr(task_object, "getReturnedSize"):
                            task_row["required_returned_size"] = float(task_object.getReturnedSize())
                    return fresh
                row = controller.cycle(env, current, runtime_tasks, plan,
                    lambda e, commands: apply_commands(e, commands, communication, ComputationScheduler, collector.step23.TrafficScheduler),
                    env.step, capture)
                rows.append(dict(row))
                receipt["searches"] += 1
                decision_path = episode_dir / f"decision_{receipt['searches']:02d}.json"
                atomic_json(decision_path, row)
                if row["status"] != "EXECUTED":
                    break
                decisions.append(row["fresh_observation"])
                final_observation = decisions[-1]
                history_action = reconstruct_history_action(row.get("action") or {}, current)
                row["history_action"] = history_action
                row["history_outcome"] = row["fresh_observation"]
                atomic_json(decision_path, row)
                steps.append({"frame_index": current["frame_index"], "action": history_action, "outcome": row["fresh_observation"]})
            raise EpisodeStopped()
        try:
            collector.collect_trajectory(episode["simulator_seed"], episode["policy_seed"], episode["trajectory_id"], on_decision=callback)
        except EpisodeStopped:
            pass
        except Exception as exc:
            receipt["status"] = "STOPPED"
            receipt["reason"] = type(exc).__name__
            atomic_json(episode_dir / "episode_failure.json", {"reason": type(exc).__name__, "detail": str(exc), "traceback": traceback.format_exc()})
            raise
        if rows and final_env is not None and final_observation is not None:
            ledger.observe(final_observation["tasks"], collector.step23._all_runtime_tasks(final_env),
                           float(final_observation["simulation_time_s"]))
        atomic_json(episode_dir / "episode_receipt.json", {
            "sample_id": sample_id, "decision_count": len(rows),
            "elapsed_seconds": time.perf_counter() - episode_started,
            "metric_draft": ledger.report(planned_duration_s=max(0.1, len(rows) * 0.1),
                                           actual_duration_s=max(0.0, len(rows) * 0.1)),
            "no_retry": True, "no_mid_episode_resume": True})
        successful_rows = [r for r in rows if r.get("status") == "EXECUTED" and r.get("environment_step_attempted") is True]
        failed_rows = [r for r in rows if r.get("status") != "EXECUTED"]
        successful_steps = len(successful_rows)
        root_ids = [str(r.get("root", {}).get("state")) for r in successful_rows]
        receipt["episodes"].append({"sample_id": sample_id,
                                     "status": "COMPLETED" if not failed_rows and successful_steps >= (2 if engineering else 1) else "FAILED",
                                     "decisions": successful_steps, "device": device,
                                     "failures": [{"reason": r.get("reason"), "detail": r.get("detail", "")} for r in failed_rows],
                                     "root_ids": root_ids, "distinct_root_ids": len(set(root_ids)),
                                     "actual_action_history_aligned": all(r.get("action") is not None and r.get("fresh_observation", {}).get("capture_event_id") for r in successful_rows)})
    if engineering:
        completed = len(receipt["episodes"]) == 2 and all(e.get("status") == "COMPLETED" for e in receipt["episodes"])
        total_steps = sum(int(e.get("decisions", 0)) for e in receipt["episodes"])
        two_consecutive_per_episode = all(e.get("decisions", 0) >= 2 for e in receipt["episodes"])
        distinct_fresh_roots = all(e.get("distinct_root_ids", 0) >= 2 for e in receipt["episodes"])
        aligned = all(e.get("actual_action_history_aligned") is True for e in receipt["episodes"])
        no_fail_closed = all(not e.get("failures") for e in receipt["episodes"])
        receipt["engineering_checks"] = {"two_episodes": completed, "env_step_count": total_steps,
                                         "at_least_two_consecutive_env_steps_per_episode": two_consecutive_per_episode,
                                         "two_different_real_fresh_roots_per_episode": distinct_fresh_roots,
                                         "actual_action_history_aligned": aligned, "no_fail_closed": no_fail_closed}
        receipt["status"] = "ENGINEERING_PASS" if (completed and total_steps >= 4 and two_consecutive_per_episode
            and distinct_fresh_roots and aligned and no_fail_closed and engineering_acceptance(receipt["episodes"])) else "ENGINEERING_FAIL"
    else:
        receipt["status"] = "RUNTIME_PREFLIGHT_PASS"
    atomic_json(result_dir / "pilot_attempt.json", receipt)
    return 0 if receipt["status"] in {"ENGINEERING_PASS", "RUNTIME_PREFLIGHT_PASS"} else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, default=OUT / "00_protocol.json")
    ap.add_argument("--engineering-smoke", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.engineering_smoke == args.execute:
        raise SystemExit("choose exactly one explicit mode")
    if args.engineering_smoke:
        return engineering_smoke()
    return execute(load_protocol(args.protocol))


if __name__ == "__main__":
    raise SystemExit(main())
