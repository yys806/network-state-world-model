from __future__ import annotations

"""Publish the machine-readable audit for the frozen formal protocol."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs  # noqa: E402
from pi_jwm.formal_airfogsim_protocol_audit_v1 import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    audit_formal_protocol,
)


DEFAULT_FREEZE_PROPOSAL = (
    CODE_ROOT
    / "artifacts"
    / "audit"
    / "pi_jwm_p2c_scenario_calibration_20260819"
    / "freeze_proposal_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    CODE_ROOT / "artifacts" / "audit" / "pi_jwm_p2c_formal_protocol_audit_20260819"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def run_protocol_audit(
    *,
    freeze_proposal_path: Path = DEFAULT_FREEZE_PROPOSAL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Audit and atomically publish the frozen protocol without data generation."""

    freeze_proposal_path = Path(freeze_proposal_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"protocol audit output already exists: {output_dir}")
    proposal = json.loads(freeze_proposal_path.read_text(encoding="utf-8"))
    report = audit_formal_protocol(
        build_formal_trajectory_specs(),
        proposal,
        project_root=PROJECT_ROOT,
    )
    source_paths = [
        freeze_proposal_path,
        CODE_ROOT / "src" / "pi_jwm" / "formal_airfogsim_dataset_v1.py",
        CODE_ROOT / "src" / "pi_jwm" / "formal_airfogsim_protocol_audit_v1.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {_relative(path): _sha256(path) for path in source_paths}
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _write_json(temporary / "protocol_audit.json", report)
        manifest = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_report": "protocol_audit.json",
            "artifact_hashes": {"protocol_audit.json": _sha256(temporary / "protocol_audit.json")},
            "source_hashes": source_hashes,
            "status": {
                "audit_ready": bool(report["audit_ready"]),
                "formal_data_approved": False,
                "training_eligible": False,
                "gpu_started": False,
                "locked_test_accessed": False,
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        if _sha256(temporary / "protocol_audit.json") != manifest["artifact_hashes"]["protocol_audit.json"]:
            raise RuntimeError("protocol audit artifact changed before publication")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "audit_ready": bool(report["audit_ready"]),
        "failed_checks": list(report["failed_checks"]),
        "output_dir": str(output_dir),
        "formal_data_approved": False,
        "training_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-proposal", type=Path, default=DEFAULT_FREEZE_PROPOSAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_protocol_audit(
        freeze_proposal_path=args.freeze_proposal,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["audit_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
