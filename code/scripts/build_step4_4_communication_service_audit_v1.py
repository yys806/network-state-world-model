"""Build the STEP 4.4 communication-service sufficiency gate artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pi_jwm.step4_4_communication_service_audit_v1 import (
    build_communication_service_audit,
    validate_communication_service_audit,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_4_communication_service_audit_v1_20260921"
AUTHORITY_PATH = Path(r"D:\shen\OB\科研\PIJWM\04世界模型预测边界与当前模型.md")
SOURCE_SPECS = {
    "code/reference/AirFogSim/airfogsim/manager/channel_manager_cp.py": [
        {"symbol": "ChannelManagerCP.computeRate", "semantic_claim": "derives SINR/rate from signal, interference, noise and RB bandwidth, then applies a random per-RB outage mask", "anchor": "symbol:ChannelManagerCP.computeRate"},
        {"symbol": "ChannelManagerCP.getCSI", "semantic_claim": "returns per-RB channel attenuation including fast fading", "anchor": "symbol:ChannelManagerCP.getCSI"},
    ],
    "code/reference/AirFogSim/airfogsim/channel_callback/outage_callback.py": [
        {"symbol": "rayleigh_outage_prob", "semantic_claim": "maps SINR and threshold to outage probability, not a deterministic outage realization", "anchor": "symbol:rayleigh_outage_prob"},
    ],
    "code/reference/AirFogSim/airfogsim/airfogsim_env.py": [
        {"symbol": "AirFogSimEnv._compute_communication_rate", "semantic_claim": "updates fast fading before computeRate", "anchor": "symbol:AirFogSimEnv._compute_communication_rate"},
        {"symbol": "AirFogSimEnv._execute_communication", "semantic_claim": "actual transferred data is summed rate times simulation interval", "anchor": "symbol:AirFogSimEnv._execute_communication"},
        {"symbol": "AirFogSimEnv._updateWiredCommunication", "semantic_claim": "wired runtime enqueues active task flows and consumes WiredNetworkManager.step results", "anchor": "symbol:AirFogSimEnv._updateWiredCommunication"},
    ],
    "code/reference/AirFogSim/airfogsim/manager/wired_manager.py": [
        {"symbol": "WiredNetworkManager.step", "semantic_claim": "shares configured link capacity equally among active flows on that link", "anchor": "symbol:WiredNetworkManager.step"},
    ],
    "code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py": [
        {"symbol": "_physical_structure", "semantic_claim": "Decision snapshot exposes per-RB CSI but not the future outage realization", "anchor": "symbol:_physical_structure"},
    ],
    "code/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py": [
        {"symbol": "_capture_wireless_profile_rows", "semantic_claim": "outage and actual rate are recorded after runtime channel computation as outcome-only evidence", "anchor": "symbol:_capture_wireless_profile_rows"},
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_claim_checks() -> dict[str, bool]:
    channel = (ROOT / "code/reference/AirFogSim/airfogsim/manager/channel_manager_cp.py").read_text(encoding="utf-8")
    env = (ROOT / "code/reference/AirFogSim/airfogsim/airfogsim_env.py").read_text(encoding="utf-8")
    wired = (ROOT / "code/reference/AirFogSim/airfogsim/manager/wired_manager.py").read_text(encoding="utf-8")
    collector = (ROOT / "code/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py").read_text(encoding="utf-8")
    return {
        "wireless_nominal_formula_found": "cp.log2(1 + V2V_SINR_linear)" in channel and "avg_band * self.V2V_Rate" in channel,
        "per_rb_random_outage_draw_found": "cp.random.rand(*self.V2V_SINR.shape)" in channel,
        "outage_zeroes_rate_found": "cp.where(self.is_V2V_outage, 0, self.V2V_Rate)" in channel,
        "outage_is_outcome_only_in_collector": '"outage": bool' in collector and '"temporal_role": "outcome_only_not_same_frame_decision_input"' in collector,
        "wired_capacity_share_formula_found": "per_flow_capacity = total_capacity / len(task_ids)" in wired and "self.wired_manager.step(self.simulation_interval)" in env,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.refresh_existing)
    provenance = [
        {"path": path, "sha256": sha256(ROOT / path), "symbols": symbols}
        for path, symbols in SOURCE_SPECS.items()
    ]
    report = build_communication_service_audit(source_provenance=provenance, source_claim_checks=source_claim_checks())
    report["authority_source"] = {"path": str(AUTHORITY_PATH), "sha256": sha256(AUTHORITY_PATH), "access": "read_only"}
    report["validation"] = validate_communication_service_audit(report)
    if not report["validation"]["passed"]:
        raise RuntimeError("communication-service audit validation failed")
    audit_path = args.output_dir / "communication_service_sufficiency_audit.json"
    write_json(audit_path, report)
    manifest = {
        "schema_version": "PI-JWM-Step-4.4-Communication-Service-Audit-Manifest-v1",
        "evidence_class": "SOURCE_AUDIT_ONLY_NO_WORLD_MODEL_IMPLEMENTATION",
        "files": {audit_path.name: sha256(audit_path)},
        "source_files": provenance,
        "authority_source": report["authority_source"],
        "scope": report["scope"],
        "world_model_implementation_started": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"passed": True, "verdict": report["verdict"], "research_stop": report["research_stop"], "artifact": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
