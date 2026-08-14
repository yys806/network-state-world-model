"""Run or verify the CPU-only PI-JWM P2-C scale/distribution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.p2c_scale_distribution_audit_v1 import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    FORMAL_CONFIG_SCHEMA_VERSION,
    REQUIRED_ARTIFACT_FILES,
    audit_bundle,
)


AUDIT_REPORT_NAME = "p2c_scale_distribution_audit_v1.json"
FORMAL_CONFIG_NAME = "p2c_formal_data_config_candidate_v1.json"
MANIFEST_NAME = "manifest.json"
CANONICAL_SOURCE_PATHS = (
    CODE_ROOT / "src" / "pi_jwm" / "p2c_scale_distribution_audit_v1.py",
    Path(__file__).resolve(),
    CODE_ROOT / "tests" / "test_p2c_scale_distribution_audit_v1.py",
    CODE_ROOT / "tests" / "test_run_p2c_scale_distribution_audit_v1.py",
    PROJECT_ROOT / "docs" / "superpowers" / "plans" / "2026-08-14-p2-c-scale-distribution-audit.md",
    PROJECT_ROOT / "文档" / "研究进展" / "2026-08-13-PI-JWM-v4全双图采集器设计.md",
    PROJECT_ROOT
    / "文档"
    / "研究进展"
    / "2026-08-14-PI-JWM-P2-C正式数据规模与分布审计.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _portable_source_key(path: Path) -> str:
    try:
        key = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"audit source is outside project root: {path}") from exc
    if key.startswith(".worktrees/") or Path(key).is_absolute():
        raise ValueError(f"non-portable audit source key: {key}")
    return key


def _source_hashes(source_paths: Sequence[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"audit source is missing: {path}")
        key = _portable_source_key(path)
        if key in hashes:
            raise ValueError(f"duplicate audit source key: {key}")
        hashes[key] = _sha256(path)
    return {key: hashes[key] for key in sorted(hashes)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the CPU-only P2-B bundle without generating data or running training"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def publish_audit_bundle(
    input_bundle: str | Path,
    output_dir: str | Path,
    *,
    source_paths: Sequence[Path] = CANONICAL_SOURCE_PATHS,
) -> dict[str, object]:
    source_root = Path(input_bundle)
    target = Path(output_dir)
    if source_root.absolute() == target.absolute():
        raise ValueError("input bundle and audit output directory must be different")
    if target.exists():
        raise FileExistsError(f"audit output already exists; use --verify-only: {target}")
    report = audit_bundle(source_root, project_root=PROJECT_ROOT)
    candidate = report["candidate_formal_data_config"]
    if not isinstance(candidate, Mapping):
        raise ValueError("audit report lacks candidate formal-data configuration")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{target.name}.partial-", dir=target.parent))
    try:
        _write_json(temporary / AUDIT_REPORT_NAME, report)
        _write_json(temporary / FORMAL_CONFIG_NAME, candidate)
        manifest = {
            "schema_version": "PIJWM-P2C-Audit-Manifest-v1",
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "formal_config_schema_version": FORMAL_CONFIG_SCHEMA_VERSION,
            "input_hashes": {
                name: _sha256(source_root / name) for name in REQUIRED_ARTIFACT_FILES
            },
            "artifact_hashes": {
                name: _sha256(temporary / name)
                for name in (AUDIT_REPORT_NAME, FORMAL_CONFIG_NAME)
            },
            "source_hashes": _source_hashes(tuple(Path(path) for path in source_paths)),
            "status": {
                "audit_completed": True,
                "audit_status": report["audit_status"],
                "formal_data_approved": False,
                "training_eligible": False,
                "gpu_started": False,
                "locked_test_accessed": False,
            },
        }
        _write_json(temporary / MANIFEST_NAME, manifest)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "audit_status": report["audit_status"],
        "formal_data_approved": False,
        "blocking_reasons": list(report["blocking_reasons"]),
        "output_file_count": 3,
    }


def verify_audit_bundle(
    input_bundle: str | Path,
    output_dir: str | Path,
    *,
    source_paths: Sequence[Path] = CANONICAL_SOURCE_PATHS,
) -> dict[str, object]:
    source_root = Path(input_bundle)
    target = Path(output_dir)
    errors: list[str] = []
    for name in (AUDIT_REPORT_NAME, FORMAL_CONFIG_NAME, MANIFEST_NAME):
        if not (target / name).is_file():
            errors.append(f"missing audit artifact: {name}")
    if errors:
        return {"passed": False, "errors": errors}
    try:
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        report = json.loads((target / AUDIT_REPORT_NAME).read_text(encoding="utf-8"))
        candidate = json.loads((target / FORMAL_CONFIG_NAME).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"passed": False, "errors": [f"audit JSON parse failure: {exc}"]}
    if manifest.get("schema_version") != "PIJWM-P2C-Audit-Manifest-v1":
        errors.append("manifest schema mismatch")
    if report.get("schema_version") != AUDIT_SCHEMA_VERSION:
        errors.append("audit report schema mismatch")
    if candidate.get("schema_version") != FORMAL_CONFIG_SCHEMA_VERSION:
        errors.append("candidate config schema mismatch")
    if candidate.get("formal_data_approved") is not False:
        errors.append("candidate config improperly approves formal data")
    for name in REQUIRED_ARTIFACT_FILES:
        if not (source_root / name).is_file():
            errors.append(f"missing audit input: {name}")
            continue
        expected = manifest.get("input_hashes", {}).get(name)
        if expected != _sha256(source_root / name):
            errors.append(f"input hash mismatch: {name}")
    for name in (AUDIT_REPORT_NAME, FORMAL_CONFIG_NAME):
        expected = manifest.get("artifact_hashes", {}).get(name)
        if expected != _sha256(target / name):
            errors.append(f"artifact hash mismatch: {name}")
    try:
        expected_sources = _source_hashes(tuple(Path(path) for path in source_paths))
    except (OSError, ValueError) as exc:
        errors.append(f"source verification failure: {exc}")
    else:
        recorded_sources = manifest.get("source_hashes")
        if recorded_sources != expected_sources:
            errors.append("source hash mismatch")
        if any(Path(key).is_absolute() or str(key).startswith(".worktrees/") for key in expected_sources):
            errors.append("non-portable source key")
    status = manifest.get("status", {})
    for flag in ("formal_data_approved", "training_eligible", "gpu_started", "locked_test_accessed"):
        if status.get(flag) is not False:
            errors.append(f"status flag must remain false: {flag}")
    return {"passed": not errors, "errors": errors}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        result = verify_audit_bundle(args.bundle, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    result = publish_audit_bundle(args.bundle, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
