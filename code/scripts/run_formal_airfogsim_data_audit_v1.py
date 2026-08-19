from __future__ import annotations

"""Run and publish the independent formal AirFogSim data acceptance audit."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_data_audit_v1 import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    audit_formal_dataset,
)


DEFAULT_DATASET_DIR = CODE_ROOT / "artifacts" / "formal_data" / "pi_jwm_v4_formal_candidate_v2"
DEFAULT_OUTPUT_DIR = (
    CODE_ROOT / "artifacts" / "audit" / "pi_jwm_p2c_formal_data_audit_20260819"
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


def run_data_audit(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    deep_graph_validation: bool = True,
) -> dict[str, Any]:
    """Audit the dataset and atomically publish the acceptance evidence."""

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"data audit output already exists: {output_dir}")
    report = audit_formal_dataset(
        dataset_dir,
        deep_graph_validation=deep_graph_validation,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "data_audit.json", report)
        manifest = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_report": "data_audit.json",
            "dataset_root": str(dataset_dir.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
                "\\", "/"
            )
            if dataset_dir.resolve().is_relative_to(PROJECT_ROOT.resolve())
            else str(dataset_dir.resolve()),
            "artifact_hashes": {
                "data_audit.json": _sha256(temporary / "data_audit.json")
            },
            "status": {
                "audit_ready": bool(report["audit_ready"]),
                "formal_data_approved": bool(report["formal_data_approved"]),
                "training_eligible": False,
                "gpu_started": False,
                "locked_test_accessed": False,
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "audit_ready": bool(report["audit_ready"]),
        "failed_checks": list(report["failed_checks"]),
        "output_dir": str(output_dir),
        "formal_data_approved": bool(report["formal_data_approved"]),
        "training_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-deep-graph-validation",
        action="store_true",
        help="skip re-running the graph validator; keep the stored validation report checks",
    )
    args = parser.parse_args()
    result = run_data_audit(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        deep_graph_validation=not args.skip_deep_graph_validation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["audit_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
