"""Build the deterministic STEP 4.2C-A observation-only audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi_jwm.step4_2c_a_causal_flow_ledger_feasibility_v1 import (
    build_causal_flow_ledger_feasibility_audit,
    validate_causal_flow_ledger_feasibility_audit,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_a_causal_flow_ledger_feasibility_v1_20260920"
SOURCE_FILES = [
    "code/reference/AirFogSim/airfogsim/entities/task.py",
    "code/reference/AirFogSim/airfogsim/manager/task_manager.py",
    "code/reference/AirFogSim/airfogsim/airfogsim_env.py",
    "code/reference/AirFogSim/airfogsim/manager/wired_manager.py",
    "code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
    "code/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py",
]
SYMBOLS = {
    SOURCE_FILES[0]: [
        {"symbol": "Task.transmit_to_Node", "semantic_claim": "stage-local transmitted size updates and resets", "anchor": "symbol:Task.transmit_to_Node"},
        {"symbol": "Task.startToReturn", "semantic_claim": "return stage reuses transmission mechanism", "anchor": "symbol:Task.startToReturn"},
    ],
    SOURCE_FILES[1]: [{"symbol": "TaskManager._task_dependencies", "semantic_claim": "DAG dependency gating, not payload transfer", "anchor": "attribute:TaskManager._task_dependencies"}],
    SOURCE_FILES[2]: [
        {"symbol": "AirFogSimEnv._updateWirelessCommunication", "semantic_claim": "wireless transfer event source", "anchor": "symbol:AirFogSimEnv._updateWirelessCommunication"},
        {"symbol": "AirFogSimEnv._updateWiredCommunication", "semantic_claim": "wired transfer event source", "anchor": "symbol:AirFogSimEnv._updateWiredCommunication"},
    ],
    SOURCE_FILES[3]: [{"symbol": "WiredNetworkManager.step", "semantic_claim": "wired per-task service source", "anchor": "symbol:WiredNetworkManager.step"}],
    SOURCE_FILES[4]: [{"symbol": "_extract_tasks", "semantic_claim": "decision-time task/current-node snapshot", "anchor": "symbol:_extract_tasks"}],
    SOURCE_FILES[5]: [{"symbol": "build_transfer_events", "semantic_claim": "collector transfer-event assembly", "anchor": "symbol:build_transfer_events"}],
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    provenance = [{"path": path, "sha256": sha256(ROOT / path), "symbols": SYMBOLS[path]} for path in SOURCE_FILES]
    report = build_causal_flow_ledger_feasibility_audit(source_provenance=provenance)
    report["validation"] = validate_causal_flow_ledger_feasibility_audit(report)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "causal_flow_ledger_feasibility_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "PI-JWM-Step4.2C-A-Audit-Manifest-v1",
        "files": [{"path": target.name, "sha256": sha256(target)}],
        "source_files": provenance,
        "observation_only": True,
        "formal_dataset": False,
        "training": False,
        "gpu": False,
        "locked_test": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "verdict": report["verdict"], "validation_passed": report["validation"]["passed"]}, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
