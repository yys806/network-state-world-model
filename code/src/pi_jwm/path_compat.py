"""Read-only compatibility for repository-relative paths frozen before layout changes."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Mapping


STABLE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("文档/知识库/", "记录/"),
    ("文档/文献/", "literature/"),
    ("文档/组会/", "meeting/"),
    ("代码/", "code/"),
)


def _repository_relative(raw_path: str) -> str:
    key = str(raw_path).replace("\\", "/")
    pure = PurePosixPath(key)
    if (
        not key
        or pure.is_absolute()
        or ".." in pure.parts
        or re.match(r"^[A-Za-z]:", key)
    ):
        raise ValueError("path must be repository-relative")
    return pure.as_posix()


def load_exact_mapping(mapping_path: Path) -> dict[str, str]:
    """Load old-to-new file identities from a migration manifest."""

    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "pi_jwm_repository_layout_migration_v1":
        raise ValueError("unsupported repository migration mapping schema")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("migration mapping entries must be an array")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("migration mapping entry must be an object")
        source = _repository_relative(str(entry.get("source", "")))
        target = _repository_relative(str(entry.get("target", "")))
        if source in result:
            raise ValueError(f"duplicate source in migration mapping: {source}")
        result[source] = target
    return result


def load_source_changes(changes_path: Path) -> dict[str, str]:
    """Load explicitly reviewed old-key to current-SHA migration changes."""

    payload = json.loads(Path(changes_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "pi_jwm_repository_layout_source_changes_v1":
        raise ValueError("unsupported repository source-change schema")
    result: dict[str, str] = {}
    for entry in payload.get("entries", []):
        if not isinstance(entry, Mapping):
            raise ValueError("source-change entry must be an object")
        source = _repository_relative(str(entry.get("source", "")))
        new_sha = str(entry.get("new_sha256", ""))
        if len(new_sha) != 64 or any(char not in "0123456789abcdef" for char in new_sha):
            raise ValueError(f"invalid migrated source SHA-256 for {source}")
        prior = result.get(source)
        if prior is not None and prior != new_sha:
            raise ValueError(f"conflicting migrated source hashes for {source}")
        result[source] = new_sha
    return result


def resolve_repository_path(
    project_root: Path,
    raw_path: str,
    *,
    exact_mapping: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a safe current path without asserting that the file exists.

    Historical identifiers remain the manifest identity.  This function only
    chooses the current read location and deliberately does not create, copy,
    or substitute a missing source.
    """

    key = _repository_relative(raw_path)
    mapped = None
    if exact_mapping is not None:
        mapped = exact_mapping.get(key)
    if mapped is None:
        for old_prefix, new_prefix in STABLE_PREFIXES:
            if key.startswith(old_prefix):
                mapped = new_prefix + key[len(old_prefix) :]
                break
    mapped_key = _repository_relative(mapped if mapped is not None else key)
    root = Path(project_root).resolve()
    candidate = root.joinpath(*PurePosixPath(mapped_key).parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must be repository-relative") from exc
    return candidate
