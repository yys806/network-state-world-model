"""Run the frozen real AirFogSim warmup to each Pilot decision root."""
import json, os, sys, traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "code/reference/AirFogSim/examples"
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts"),
                str(EXAMPLES), str(ROOT / "code/reference/AirFogSim")]


class _ReachedDecisionRoot(BaseException):
    def __init__(self, row):
        self.row = row


def run(receipt_path: Path, episodes: list[dict]) -> dict:
    os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
    receipt = {"status": "STARTING", "sumo_home": os.environ.get("SUMO_HOME"),
               "config": "sumo_wujiaochang/osm.sumocfg", "wm_searches": 0,
               "episodes": []}
    try:
        os.chdir(EXAMPLES)
        import traci, sumolib
        try:
            traci_version = version("traci")
            sumolib_version = version("sumolib")
        except PackageNotFoundError:
            raise RuntimeError("SUMO_PYTHON_DISTRIBUTION_VERSION_UNAVAILABLE")
        receipt.update(traci_version=traci_version, sumolib_version=sumolib_version,
                       traci_path=str(traci.__file__), sumolib_path=str(sumolib.__file__))
        if traci_version != "1.12.0" or sumolib_version != "1.12.0":
            raise RuntimeError(f"SUMO_PYTHON_VERSION_MISMATCH: traci={traci_version};sumolib={sumolib_version}")
        from collect_step5_5_formal_raw_v1 import collect_trajectory

        for item in episodes:
            target = int(item["initial_frame"])
            observed = {}

            def stop_at_root(env, decisions, steps, config, communication_scheduler):
                frame = len(decisions) - 1
                if frame != target:
                    return
                decision = decisions[-1]
                connection = getattr(env, "traci_connection", None)
                if connection is None or not connection.isConnected():
                    raise RuntimeError("PREFLIGHT_TRACI_DISCONNECTED_AT_ROOT")
                row = {
                    "trajectory_id": item["trajectory_id"],
                    "simulator_seed": int(item["simulator_seed"]),
                    "policy_seed": int(item["policy_seed"]),
                    "target_frame": target,
                    "capture_event_id": decision.get("capture_event_id"),
                    "simulation_time_s": float(decision["simulation_time_s"]),
                    "vehicle_count": len(env.vehicles),
                    "task_count": len(decision.get("tasks", [])),
                    "entity_count": len(decision.get("entities", [])),
                    "decision_fields": sorted(decision),
                    "history_present": hasattr(env, "History"),
                    "traci_connected_at_root": True,
                    "wm_searches": 0,
                }
                observed.update(row)
                raise _ReachedDecisionRoot(row)

            try:
                collect_trajectory(int(item["simulator_seed"]), int(item["policy_seed"]),
                                   str(item["trajectory_id"]), on_decision=stop_at_root)
                raise RuntimeError("PREFLIGHT_DECISION_ROOT_NOT_REACHED")
            except _ReachedDecisionRoot as reached:
                receipt["episodes"].append({**reached.row, "status": "PASS", "closed": True})
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        if receipt["episodes"]:
            receipt["episodes"][-1]["status"] = "FAIL"
    finally:
        os.chdir(ROOT)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt
