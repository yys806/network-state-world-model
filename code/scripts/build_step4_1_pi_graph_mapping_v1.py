"""Build the machine-readable Step 4.1 PI graph mapping evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pi_jwm.step4_1_pi_graph_mapping_v1 import (
    build_mapping,
    save_mapping,
    validate_mapping_checks,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
DEFAULT_TENSOR_SCHEMA = ROOT / "code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/tensor_schema.json"
DEFAULT_TENSOR_MANIFEST = ROOT / "code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/manifest.json"
DEFAULT_OUTPUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_1_pi_graph_mapping_v1_20260919"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--tensor-schema", type=Path, default=DEFAULT_TENSOR_SCHEMA)
    parser.add_argument("--tensor-manifest", type=Path, default=DEFAULT_TENSOR_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    tensor_schema = json.loads(args.tensor_schema.read_text(encoding="utf-8"))
    mapping = build_mapping()
    checks = validate_mapping_checks(mapping, raw, tensor_schema)
    if not checks["passed"]:
        raise ValueError(f"Step 4.1 mapping validation failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mapping_schema.json": mapping,
        "current_data_to_graph_role.json": mapping["current_data_to_graph_role"],
        "required_additive_data_extensions.json": mapping["required_additive_data_extensions"],
        "forbidden_wrong_placements.json": mapping["forbidden_wrong_placements"],
        "old_implementation_reuse_conflict.json": mapping["old_implementation_reuse_conflict"],
        "validation_report.json": {
            "schema_version": mapping["schema_version"],
            "passed": all(value for key, value in checks.items() if key != "passed"),
            "required_checks": checks,
            "evidence_source": {
                "raw": str(args.raw.relative_to(ROOT)).replace("\\", "/"),
                "tensor_schema": str(args.tensor_schema.relative_to(ROOT)).replace("\\", "/"),
                "validator": "pi_jwm.step4_1_pi_graph_mapping_v1.validate_mapping_checks",
            },
        },
    }
    for name, value in outputs.items():
        if name == "mapping_schema.json":
            save_mapping(value, args.output_dir / name)
        else:
            write_json(args.output_dir / name, value)

    source_paths = (
        ROOT / "code/src/pi_jwm/step4_1_pi_graph_mapping_v1.py",
        Path(__file__),
        ROOT / "code/tests/test_step4_1_pi_graph_mapping_v1.py",
    )
    manifest = {
        "schema_version": mapping["schema_version"],
        "scope": mapping["scope"],
        "graph_readiness": mapping["graph_readiness"],
        "input_evidence": {
            str(args.raw.relative_to(ROOT)).replace("\\", "/"): sha256(args.raw),
            str(args.tensor_schema.relative_to(ROOT)).replace("\\", "/"): sha256(args.tensor_schema),
            str(args.tensor_manifest.relative_to(ROOT)).replace("\\", "/"): sha256(args.tensor_manifest),
        },
        "definition_basis": mapping["definition_basis"],
        "source_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
        "files": {
            name: sha256(args.output_dir / name)
            for name in outputs
        },
        "counts": {
            "field_mappings": len(mapping["field_mappings"]),
            "relations": len(mapping["relations"]),
            "required_additive_data_extensions": len(mapping["required_additive_data_extensions"]),
            "forbidden_wrong_placements": len(mapping["forbidden_wrong_placements"]),
            "old_implementation_reuse_conflict": len(mapping["old_implementation_reuse_conflict"]),
        },
        "validation_passed": checks["passed"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "passed": checks["passed"], "files": sorted(outputs) + ["manifest.json"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
