"""Build the deterministic STEP 4.2B source audit artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi_jwm.step4_2b_stateful_flow_source_audit_v1 import (
    build_stateful_flow_source_audit,
    validate_stateful_flow_source_audit,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2b_stateful_flow_source_audit_v1_20260920"
SOURCE_FILES = [
    "code/reference/AirFogSim/airfogsim/entities/task.py",
    "code/reference/AirFogSim/airfogsim/manager/task_manager.py",
    "code/reference/AirFogSim/airfogsim/airfogsim_env.py",
    "code/reference/AirFogSim/airfogsim/manager/wired_manager.py",
    "code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
    "code/src/pi_jwm/airfogsim_full_dual_graph_frame_builder_v1.py",
    "code/src/pi_jwm/full_dual_graph_collector_contract_v1.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    provenance = [
        {"path": path, "sha256": sha256(ROOT / path)}
        for path in SOURCE_FILES
    ]
    report = build_stateful_flow_source_audit(source_provenance=provenance)
    report["validation"] = validate_stateful_flow_source_audit(report)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "stateful_flow_source_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "PI-JWM-Step4.2B-Audit-Manifest-v1",
        "files": [{"path": target.name, "sha256": sha256(target)}],
        "source_files": provenance,
        "observation_only": True,
        "formal_dataset": False,
        "training": False,
        "gpu": False,
        "locked_test": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "verdict": report["verdict"], "failed_checks": report["failed_checks"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
