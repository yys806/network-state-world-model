"""Observation-only audit of future action references in available Raw windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pi_jwm.model_ready_sample_contract_v1 import audit_future_action_references  # noqa: E402


RAW_PATHS = (
    ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json",
    ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/real_communication_outcome_semantics.json",
    ROOT / "code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/real_raw_contract_finalization.json",
    ROOT / "code/artifacts/protocols/pi_jwm_raw_multi_decision_step_real_airfogsim_v2_20260919/real_multi_step.json",
)
OUT = ROOT / "code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/future_action_reference_audit.json"


def main() -> None:
    records = []
    seen = set()
    for path in RAW_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        scope = payload.get("scope", {})
        if scope.get("locked_test") or scope.get("training") or scope.get("gpu"):
            continue
        records.append({
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": digest,
            "schema_version": payload.get("schema_version"),
            "scope": scope,
            "audit": audit_future_action_references(payload, history_steps=2, horizon_steps=2),
        })
    aggregate = {
        "schema_version": "PI-JWM-Step-3.1F-Future-Action-Reference-Audit-v1",
        "window_policy": {"history_steps": 2, "horizon_steps": 2, "future_offsets_audited": [1]},
        "source_count": len(records),
        "records": records,
        "summary": {
            "constructible_window_count": sum(item["audit"]["constructible_window_count"] for item in records),
            "affected_window_count": sum(item["audit"]["affected_window_count"] for item in records),
            "unresolved_future_reference_count": sum(item["audit"]["unresolved_future_reference_count"] for item in records),
            "by_action_family": {
                family: {
                    "windows": sum(item["audit"]["by_action_family"][family]["windows"] for item in records),
                    "references": sum(item["audit"]["by_action_family"][family]["references"] for item in records),
                }
                for family in ("route", "comm", "comp", "mobility")
            },
            "by_object_kind": {
                kind: {
                    "windows": sum(item["audit"]["by_object_kind"][kind]["windows"] for item in records),
                    "references": sum(item["audit"]["by_object_kind"][kind]["references"] for item in records),
                }
                for kind in ("task", "physical_entity")
            },
            "affected_window_rate": (
                sum(item["audit"]["affected_window_count"] for item in records)
                / sum(item["audit"]["constructible_window_count"] for item in records)
                if sum(item["audit"]["constructible_window_count"] for item in records) else None
            ),
        },
        "observation_only": True,
        "researcher_decision_required_if_nonzero": True,
        "locked_test_accessed": False,
        "training": False,
        "gpu": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
