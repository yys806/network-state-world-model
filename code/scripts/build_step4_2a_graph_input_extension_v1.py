"""Build the Step 4.2A existing-source additive graph-input evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pi_jwm.step3_2_batch_preprocessing_v1 import (
    RawSource,
    apply_normalization as apply_base_normalization,
    fit_train_normalization_stats as fit_base_normalization_stats,
)
from pi_jwm.step4_2a_graph_input_extension_v1 import (
    DATASET_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
    TENSOR_SCHEMA_VERSION,
    apply_extension_normalization,
    build_extended_tensor_batch,
    build_extension_batch,
    fit_extension_normalization_stats,
    save_extended_tensor_batch,
    validate_extension_acceptance,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920"
SOURCES = (
    RawSource(ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json", "dev_train"),
    RawSource(ROOT / "code/artifacts/protocols/pi_jwm_step3_2_raw_dev_seed1_20260919/real_communication_outcome_semantics.json", "dev_train"),
    RawSource(ROOT / "code/artifacts/protocols/pi_jwm_step3_2_raw_dev_seed2_20260919/real_communication_outcome_semantics.json", "dev_validation"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    # Repository JSON artifacts are LF-normalized.  Writing bytes avoids
    # Windows newline translation so manifest SHA-256 values also match the
    # exact blobs stored by Git.
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def semantic_digest(bundle: dict[str, Any], stats: dict[str, Any], normalized: list[dict[str, Any]], tensor: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in (bundle, stats, normalized, tensor["contract"], tensor["sample_ids"], tensor["sample_static"], tensor["sample_metadata"]):
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(key for key, value in tensor.items() if isinstance(value, np.ndarray)):
        value = tensor[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_all() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    bundle = build_extension_batch(SOURCES)
    base_stats = fit_base_normalization_stats(bundle["samples"])
    base_normalized = apply_base_normalization(bundle["samples"], base_stats)
    extension_stats = fit_extension_normalization_stats(base_normalized)
    normalized = apply_extension_normalization(base_normalized, extension_stats)
    tensor = build_extended_tensor_batch(normalized, stats=extension_stats)
    stats = {
        **extension_stats,
        "base_step3_2_stats": base_stats,
        "fit_order": "Step 3.2 frozen fields and Step 4.2A additive fields are both fit on dev_train only",
    }
    return bundle, stats, normalized, tensor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.refresh_existing)

    bundle, stats, normalized, tensor = build_all()
    rebuilt_bundle, rebuilt_stats, rebuilt_normalized, rebuilt_tensor = build_all()
    first_digest = semantic_digest(bundle, stats, normalized, tensor)
    second_digest = semantic_digest(rebuilt_bundle, rebuilt_stats, rebuilt_normalized, rebuilt_tensor)
    deterministic = first_digest == second_digest
    checks = validate_extension_acceptance(bundle, stats, tensor, deterministic_rebuild=deterministic)
    if not checks["passed"]:
        raise ValueError(f"Step 4.2A validation failed: {checks}")

    raw_dir = args.output_dir / "raw_amendments"
    raw_dir.mkdir(exist_ok=args.refresh_existing)
    raw_files: dict[str, str] = {}
    for raw in bundle["raw_amendments"]:
        trajectory_id = str(raw["decisions"][0]["trajectory_id"])
        path = raw_dir / f"{trajectory_id}.json"
        write_json(path, raw)
        raw_files[str(path.relative_to(args.output_dir)).replace("\\", "/")] = sha256(path)

    artifact_bundle = {key: value for key, value in bundle.items() if key != "raw_amendments"}
    write_json(args.output_dir / "batch.json", artifact_bundle)
    write_json(args.output_dir / "train_normalization_stats.json", stats)
    write_json(args.output_dir / "normalized_samples.json", normalized)
    write_json(args.output_dir / "tensor_schema.json", tensor["contract"])
    write_json(args.output_dir / "step4_1_gap_resolution.json", bundle["step4_1_gap_resolution"])
    save_extended_tensor_batch(tensor, args.output_dir / "tensor.npz")

    report = {
        "schema_version": "PI-JWM-Step-4.2A-Validation-Receipt-v1",
        "passed": bool(all(value for key, value in checks.items() if key != "passed")),
        "required_checks": checks,
        "fields_resolved": bundle["fields_resolved"],
        "fields_still_blocked": bundle["fields_still_blocked"],
        "remaining_gap_classification": bundle["remaining_gap_classification"],
        "raw_to_sample_to_tensor_provenance": bundle["provenance"],
        "new_normalization_stats": stats["features"],
        "communication_relation_semantics": {
            "wireless": "decision channel_rows; structural relation presence/validity is independent of per-RB CSI observability; missing CSI stays masked with zero placeholder and explicit reason",
            "wired": "pre-action environment.wired_edges expanded with WiredNetworkManager directed hasLink semantics; valid relation with CSI null/mask=false",
            "execution_outcome_used": False,
            "relation_validity_independent_of_feature_observability": checks["communication_mask_counterfactual"],
        },
        "task_agent_relation_semantics": {
            "src": "task_node_id",
            "host": "current decision current_node_id",
            "exec": "current_node_id only when lifecycle=computing",
            "ret": "current non-null return_destination_id",
            "future_route_action_used": False,
        },
        "graph_readiness": {
            "ready": False,
            "reason": "stable stateful Flow and other Raw-insufficient fields remain blocked",
            "graph_builder_implemented": False,
        },
        "deterministic_semantic_digest": first_digest,
        "sample_count": len(bundle["samples"]),
        "trajectory_count": len(bundle["provenance"]),
        "scope": bundle["scope"],
    }
    write_json(args.output_dir / "validation_report.json", report)

    output_files = [
        "batch.json",
        "train_normalization_stats.json",
        "normalized_samples.json",
        "tensor_schema.json",
        "tensor.npz",
        "validation_report.json",
        "step4_1_gap_resolution.json",
    ]
    source_paths = (
        ROOT / "code/src/pi_jwm/step4_2a_graph_input_extension_v1.py",
        Path(__file__),
        ROOT / "code/tests/test_step4_2a_graph_input_extension_v1.py",
        ROOT / "code/reference/AirFogSim/airfogsim/manager/wired_manager.py",
    )
    manifest = {
        "schema_versions": {
            "raw": RAW_SCHEMA_VERSION,
            "sample": SAMPLE_SCHEMA_VERSION,
            "dataset_preprocessing": DATASET_SCHEMA_VERSION,
            "tensor": TENSOR_SCHEMA_VERSION,
        },
        "input_sources": {
            str(source.path.relative_to(ROOT)).replace("\\", "/"): sha256(source.path)
            for source in SOURCES
        },
        "source_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
        "files": {
            name: sha256(args.output_dir / name)
            for name in output_files
        },
        "raw_amendment_files": raw_files,
        "sample_count": len(bundle["samples"]),
        "trajectory_count": len(bundle["provenance"]),
        "array_shapes_dtypes": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensor.items()
            if isinstance(value, np.ndarray)
        },
        "normalization_units": {key: value["unit"] for key, value in stats["features"].items()},
        "fields_resolved": bundle["fields_resolved"],
        "fields_still_blocked": bundle["fields_still_blocked"],
        "remaining_gap_classification": bundle["remaining_gap_classification"],
        "deterministic_rebuild": deterministic,
        "deterministic_semantic_digest": first_digest,
        "validation_passed": checks["passed"],
        "scope": bundle["scope"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "passed": checks["passed"], "sample_count": len(bundle["samples"]), "deterministic": deterministic}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
