"""Verify the frozen single-variable complete-RSSM P4 sentinel protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_METHOD = "complete_rssm_node_x_safe_dual_graph_v1"
EXPECTED_CHANGE = "node_x_residual_non_degradation_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gate(
    *,
    protocol_path: str | Path,
    base_protocol_path: str | Path,
    preflight_audit_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path)
    base_protocol_path = Path(base_protocol_path)
    preflight_audit_path = Path(preflight_audit_path)
    repository_root = Path(repository_root)
    output_dir = Path(output_dir)
    for path in (protocol_path, base_protocol_path, preflight_audit_path):
        if "locked_test" in str(path).lower():
            raise ValueError("locked_test path is forbidden")
    protocol = _read_json(protocol_path)
    base_protocol = _read_json(base_protocol_path)
    preflight = _read_json(preflight_audit_path)
    source_mismatches = []
    for relative, expected in protocol.get("source_sha256", {}).items():
        source = repository_root / relative
        if not source.is_file() or _sha256(source) != expected:
            source_mismatches.append(relative)
    expected_gates = {
        "validation_link_f1_delta_min": -0.05,
        "node_x_mae_ratio_max": 1.25,
        "throughput_mae_delta_max": 0.0,
        "rb_occupancy_mae_delta_max": 0.0,
        "task_delay_mae_delta_max": 0.0,
    }
    boundaries = protocol.get("boundaries", {})
    change = protocol.get("changed_variable", {})
    checks = {
        "method_identity": protocol.get("method") == EXPECTED_METHOD,
        "single_change_identity": change.get("name") == EXPECTED_CHANGE,
        "single_change_weight_frozen": float(change.get("weight", -1.0)) == 1.0,
        "run_contract_reused_exactly": protocol.get("run") == base_protocol.get("run"),
        "training_semantics_reused_exactly": protocol.get("training_semantics")
        == base_protocol.get("training_semantics"),
        "acceptance_gates_frozen": protocol.get("acceptance_gates") == expected_gates,
        "preflight_status_ready": preflight.get("status")
        == "ready_for_protocol_freeze",
        "preflight_hash_bound": protocol.get("preflight", {}).get("sha256")
        == _sha256(preflight_audit_path),
        "source_hashes_match": not source_mismatches,
        "followup_seeds_closed": boundaries.get("followup_seeds_allowed") is False,
        "probability_gate_closed": boundaries.get(
            "probability_gate_allowed_before_performance_pass"
        )
        is False,
        "gpu_launch_pending": boundaries.get("gpu_launch_pending_availability") is True,
        "locked_test_sealed": boundaries.get("locked_test_accessed") is False,
        "formal_claim_closed": boundaries.get("formal_performance_claim_ready") is False,
    }
    ready = all(checks.values())
    report = {
        "schema_version": "PI-JWM-P4-complete-RSSM-node-x-safe-protocol-gate-v1",
        "status": "ready_when_gpu_available" if ready else "blocked",
        "checks": checks,
        "source_mismatches": source_mismatches,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path),
        },
        "execution_policy": {
            "gpu_started": False,
            "followup_seeds_allowed": False,
            "probability_gate_allowed": False,
            "locked_test_allowed": False,
            "formal_performance_claim_allowed": False,
        },
        "result_boundary": "Protocol identity evidence only; no GPU execution or performance claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate(
        protocol_path=args.protocol,
        base_protocol_path=args.base_protocol,
        preflight_audit_path=args.preflight_audit,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready_when_gpu_available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
