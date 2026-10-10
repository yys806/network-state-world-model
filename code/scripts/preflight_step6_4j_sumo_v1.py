"""Real AirFogSim/SUMO startup gate for the 6.4J GPU Pilot."""
import argparse, json, os, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "code/reference/AirFogSim/examples"
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts"), str(ROOT / "code/reference/AirFogSim")]

def run(receipt_path: Path, seed: int = 2026092326) -> dict:
    os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
    os.chdir(EXAMPLES)
    receipt = {"status": "STARTING", "sumo_home": os.environ.get("SUMO_HOME"),
               "cwd": str(Path.cwd()), "config": "sumo_wujiaochang/osm.sumocfg"}
    env = None
    try:
        from run_p2_single_step_collector_preflight_v1 import _build_environment
        env, *_ = _build_environment(seed, 5.0)
        receipt.update(status="CREATED", simulation_time_s=float(env.simulation_time),
                       traci_connected=getattr(env, "traci_connection", None) is not None)
        env.alloc_cpu_callback = lambda _: {}
        env.step()
        receipt.update(status="STEPPED", simulation_time_after_s=float(env.simulation_time))
    except Exception as exc:
        receipt.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            if env is not None:
                if hasattr(env, "close"):
                    env.close()
                elif getattr(env, "traci_connection", None) is not None:
                    env.traci_connection.close()
        except Exception as exc:
            receipt["close_error"] = repr(exc)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    result = run(ap.parse_args().receipt)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "STEPPED" and result.get("traci_connected") else 1

if __name__ == "__main__":
    raise SystemExit(main())
