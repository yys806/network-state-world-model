"""Move files into a reversible quarantine with a verifiable CSV manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


MANIFEST_FIELDS = (
    "source_path",
    "quarantine_path",
    "size_bytes",
    "sha256",
    "reason",
    "status",
)


def ensure_within_root(path: Path, root: Path) -> Path:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the allowed root: {resolved_path} not under {resolved_root}") from exc
    return resolved_path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _inventory(source: Path, destination: Path, reason: str) -> list[dict[str, str]]:
    rows = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        rows.append(
            {
                "source_path": str(source_file.resolve()),
                "quarantine_path": str((destination / relative).resolve()),
                "size_bytes": str(source_file.stat().st_size),
                "sha256": sha256_file(source_file),
                "reason": reason,
                "status": "planned",
            }
        )
    return rows


def _write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def quarantine_tree(
    source: Path,
    destination: Path,
    manifest_path: Path,
    reason: str,
    dry_run: bool = True,
) -> list[dict[str, str]]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Quarantine destination already exists: {destination}")

    rows = _inventory(source, destination, reason)
    _write_manifest(rows, manifest_path)
    if dry_run:
        return rows

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(source), str(destination))
        for row in rows:
            moved_path = Path(row["quarantine_path"])
            expected_size = int(row["size_bytes"])
            if not moved_path.is_file():
                raise FileNotFoundError(f"Moved file is missing: {moved_path}")
            if moved_path.stat().st_size != expected_size:
                raise IOError(f"Moved file size changed: {moved_path}")
            if sha256_file(moved_path) != row["sha256"]:
                raise IOError(f"Moved file hash changed: {moved_path}")
            row["status"] = "verified"
    except Exception:
        for row in rows:
            if row["status"] != "verified":
                row["status"] = "failed"
        _write_manifest(rows, manifest_path)
        raise

    _write_manifest(rows, manifest_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--allowed-source-root", type=Path, required=True)
    parser.add_argument("--allowed-destination-root", type=Path, required=True)
    parser.add_argument("--move", action="store_true", help="Perform the move; default is dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ensure_within_root(args.source, args.allowed_source_root)
    destination = ensure_within_root(args.destination, args.allowed_destination_root)
    rows = quarantine_tree(
        source=source,
        destination=destination,
        manifest_path=args.manifest,
        reason=args.reason,
        dry_run=not args.move,
    )
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    mode = "moved" if args.move else "planned"
    print(f"{mode}: files={len(rows)} bytes={total_bytes} manifest={args.manifest.resolve()}")


if __name__ == "__main__":
    main()
