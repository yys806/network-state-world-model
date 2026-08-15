"""Freeze the auditable PI-JWM R5.1 confirmation matrix before GPU training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_module_confirmation import write_confirmation_bundle
from run_r4_gpu_screening import _information_rate_stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-r5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--update-existing", action="store_true")
    args = parser.parse_args()
    manifest = args.existing_r5_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("existing R5 manifest is missing")
    normalization = json.loads(
        (args.evaluation_root / "evaluation_normalization_stats.json").read_text(encoding="utf-8")
    )
    rate_mean, rate_scale = _information_rate_stats(normalization)
    write_confirmation_bundle(
        args.output_dir,
        existing_r5_manifest_sha256=_sha256(manifest),
        update_existing=args.update_existing,
        information_rate_mean=rate_mean,
        information_rate_scale=rate_scale,
    )


if __name__ == "__main__":
    main()
