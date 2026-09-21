"""Build the non-locked Step 5.1A development target artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pi_jwm.step5_1a_motion_csi_target_contract_v1 import (
    build_future_target_tensor_batch,
    extend_future_motion_csi_targets,
    future_target_digest,
    save_future_target_tensor_batch,
    validate_current_model_slot_alignment,
    validate_step5_1a_acceptance,
)
from pi_jwm.step4_2a_graph_input_extension_v1 import load_extended_tensor_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920")
    parser.add_argument("--stats", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920/encoder_normalization_stats.json")
    parser.add_argument("--output", type=Path, default=root / "code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = json.loads((args.input / "normalized_samples.json").read_text(encoding="utf-8"))
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    raw = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (args.input / "raw_amendments").glob("*.json")}
    current_tensor = load_extended_tensor_batch(args.input / "tensor.npz")
    extended = [extend_future_motion_csi_targets(s, raw[s["metadata"]["trajectory_id"]], stats) for s in samples]
    tensor = build_future_target_tensor_batch(extended, stats)
    rebuilt = [extend_future_motion_csi_targets(s, raw[s["metadata"]["trajectory_id"]], stats) for s in samples]
    rebuilt_tensor = build_future_target_tensor_batch(rebuilt, stats)
    digest = future_target_digest(extended, tensor)
    deterministic_rebuild = digest == future_target_digest(rebuilt, rebuilt_tensor)
    real_trajectory_verified = bool(samples) and all(
        sample.get("metadata", {}).get("trajectory_id") in raw
        and raw[sample["metadata"]["trajectory_id"]].get("steps")
        and raw[sample["metadata"]["trajectory_id"]].get("environment", {}).get("airfogsim_source") == "code/reference/AirFogSim"
        and raw[sample["metadata"]["trajectory_id"]].get("scope", {}).get("non_locked") is True
        and raw[sample["metadata"]["trajectory_id"]].get("scope", {}).get("training") is False
        and raw[sample["metadata"]["trajectory_id"]].get("scope", {}).get("gpu") is False
        and raw[sample["metadata"]["trajectory_id"]].get("scope", {}).get("locked_test") is False
        and any(step.get("outcome", {}).get("channel_rows") for step in raw[sample["metadata"]["trajectory_id"]].get("steps", []))
        for sample in samples
    ) and "locked_test" not in str(args.input).lower()
    save_future_target_tensor_batch(tensor, args.output / "tensor.npz")
    (args.output / "extended_samples.json").write_text(json.dumps(extended, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = validate_step5_1a_acceptance(
        extended,
        tensor,
        deterministic_rebuild=deterministic_rebuild,
        real_trajectory_verified=real_trajectory_verified,
        current_model_slot_checks=validate_current_model_slot_alignment(extended, current_tensor),
    )
    receipt.update({"step": "5.1A-PATCH", "artifact_kind": "non_locked_development_target_contract_patch", "sample_count": len(extended), "digest": digest, "deterministic_rebuild_digest": future_target_digest(rebuilt, rebuilt_tensor), "stats_source": str(args.stats), "source_artifact": str(args.input)})
    (args.output / "acceptance_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    files = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = {
        "schema_version": "PI-JWM-Step5.1A-Future-Target-Artifact-Manifest-v2-local-step-stable-slot",
        "artifact_kind": "non_locked_development_target_contract_patch",
        "source_artifact": str(args.input),
        "stats_source": str(args.stats),
        "sample_count": len(extended),
        "future_target_digest": digest,
        "scope": receipt["scope"],
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "sample_count": len(extended), "digest": receipt["digest"]}, ensure_ascii=False))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
