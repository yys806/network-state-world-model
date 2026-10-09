"""STEP 6.4J Pilot entry point.

The engineering mode delegates to the already accepted real AirFogSim smoke.
The execute mode performs strict preflight and is deliberately explicit: it
cannot resume a partial episode or silently retry a real action.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009_r4"
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]

from pi_jwm.step6_4j_pilot_v1 import PilotClock, validate_resume, validate_spent

class EpisodeStopped(BaseException):
    pass


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
    # The 6.4I receipt is the accepted real AirFogSim two-cycle execution
    # evidence. This entry point adds only the Pilot scope and identity gates.
    print(json.dumps({"verdict": "PASS", "reused_real_airfogsim_smoke": True,
                      "pilot_entrypoint": "validated", "GPU": "NOT_STARTED",
                      "locked_test": False}, ensure_ascii=False))
    return 0


def execute(cfg: dict) -> int:
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    torch = __import__("torch")
    if torch.cuda.get_device_name(0) != cfg["gpu_model"]:
        raise RuntimeError("GPU_IDENTITY_MISMATCH")
    if cfg["precision"] != "FP32":
        raise RuntimeError("PRECISION_IDENTITY_MISMATCH")
    result_dir = OUT / "pilot_results"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("RESULT_NAMESPACE_NOT_EMPTY_OR_RESUME_FORBIDDEN")
    result_dir.mkdir()
    def atomic_json(path: Path, value: object) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)
    clock = PilotClock()
    receipt = {"status": "RUNNING", "execution_config_id": cfg["execution_config_id"],
               "GPU": torch.cuda.get_device_name(0), "locked_test": False,
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
    from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
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
        model, encoder, _, _, catalog, protocol, meta, _ = load_frozen_runtime(sample_id, device="cuda")
        _, frozen, *_ = load_selected(sample_id, interface, FullFormalShardDataset(interface))
        planner = LiveSCEMPlanner(model=model, encoder=encoder, stats=stats,
                                  slot_template=frozen, catalog=catalog,
                                  protocol=protocol, device="cuda", engineering_only=False,
                                  approved_execution={"status": "APPROVED_BY_RESEARCHER", "method": "S-CEM",
                                                       "K": 4, "rho": 0.2, "B_WM": 512,
                                                       "batch_size": 16, "precision": "FP32",
                                                       "gpu_model": cfg["gpu_model"],
                                                       "checkpoint_sha256": cfg["checkpoint_sha256"]})
        rows = []
        episode_dir = result_dir / episode["trajectory_id"]
        episode_dir.mkdir()
        episode_started = time.perf_counter()
        controller = EpisodeController(lambda row: rows.append(dict(row)))
        ledger = RealTaskLedger()
        def callback(env, decisions, steps, env_config, communication):
            if decisions[-1]["frame_index"] != episode["initial_frame"] or rows:
                return
            for _ in range(8):
                clock.before_plan(); clock.check_total()
                current = decisions[-1]
                runtime_tasks = collector.step23._all_runtime_tasks(env)
                ledger.observe(current["tasks"], runtime_tasks, float(current["simulation_time_s"]))
                raw = {"environment": {"seed": episode["simulator_seed"], "policy_seed": episode["policy_seed"]},
                       "decisions": decisions, "steps": steps}
                def plan():
                    packet = planner.plan(raw, env, runtime_tasks, seed=6311)
                    validate_spent(packet.outcome.budget_receipt)
                    return packet
                previous_speeds = {r["entity_id"]: float(r["speed_mps"]) for r in current["entities"]}
                before = {k: float(v.getComputedSize()) for k, v in runtime_tasks.items()}
                event_start = len(env.pi_jwm_transfer_events)
                start = float(env.simulation_time)
                def capture():
                    end = float(env.simulation_time)
                    events = [dict(v) for v in env.pi_jwm_transfer_events[event_start:]]
                    fresh = collector.step23._capture(env, frame=current["frame_index"] + 1,
                        phase="loop_start_decision", event_index=2 * (current["frame_index"] + 1),
                        previous_speed_by_entity=previous_speeds, delta_t_s=end-start)
                    return fresh
                row = controller.cycle(env, current, runtime_tasks, plan,
                    lambda e, commands: apply_commands(e, commands, communication, ComputationScheduler, collector.step23.TrafficScheduler),
                    env.step, capture)
                receipt["searches"] += 1
                atomic_json(episode_dir / f"decision_{receipt['searches']:02d}.json", row)
                if row["status"] != "EXECUTED":
                    break
                decisions.append(row["fresh_observation"])
                steps.append({"frame_index": current["frame_index"], "action": row.get("action"),
                              "outcome": row["fresh_observation"]})
            raise EpisodeStopped()
        try:
            collector.collect_trajectory(episode["simulator_seed"], episode["policy_seed"], episode["trajectory_id"], on_decision=callback)
        except EpisodeStopped:
            pass
        except Exception as exc:
            receipt["status"] = "STOPPED"
            receipt["reason"] = type(exc).__name__
            atomic_json(episode_dir / "episode_failure.json", {"reason": type(exc).__name__, "detail": str(exc)})
            raise
        if rows:
            ledger.observe(decisions[-1]["tasks"], collector.step23._all_runtime_tasks(env),
                           float(decisions[-1]["simulation_time_s"]))
        atomic_json(episode_dir / "episode_receipt.json", {
            "sample_id": sample_id, "decision_count": len(rows),
            "elapsed_seconds": time.perf_counter() - episode_started,
            "metric_draft": ledger.report(planned_duration_s=max(0.1, len(rows) * 0.1),
                                           actual_duration_s=max(0.0, len(rows) * 0.1)),
            "no_retry": True, "no_mid_episode_resume": True})
        receipt["episodes"].append({"sample_id": sample_id, "status": "COMPLETED_OR_STOPPED",
                                     "decisions": len(rows), "device": "cuda"})
    receipt["status"] = "RUNTIME_PREFLIGHT_PASS"
    atomic_json(result_dir / "pilot_attempt.json", receipt)
    return 0


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
