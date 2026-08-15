"""Build and verify the one-time PI-JWM repository layout migration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


MAPPING_SCHEMA = "pi_jwm_repository_layout_migration_v1"
BASELINE_SCHEMA = "pi_jwm_repository_layout_gate_baseline_v1"
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts):
            yield path


def _target_for_document(source: Path, project_root: Path) -> tuple[str, str]:
    rel = source.relative_to(project_root / "文档").as_posix()
    if rel == "README.md":
        return "docs/archive/原文档README.md", "docs"
    if rel == "知识库/PIJWM主文档.md":
        return "记录/PIJWM主文档.md", "records"
    if rel == "知识库/8.12之后推进.md":
        return "记录/8.12之后推进.md", "records"
    if rel == "知识库/README.md":
        return "docs/archive/原知识库README.md", "docs"
    if rel.startswith("文献/"):
        return "literature/" + rel[len("文献/") :], "literature"
    if rel.startswith("组会/"):
        return "meeting/" + rel[len("组会/") :], "meeting"
    prefix = "研究进展/"
    if not rel.startswith(prefix):
        raise ValueError(f"unclassified document path: 文档/{rel}")
    progress_rel = rel[len(prefix) :]
    if progress_rel.startswith("归档/工具与模板/"):
        suffix = progress_rel[len("归档/工具与模板/") :]
        return "docs/templates/" + suffix, "docs"
    if progress_rel.startswith("归档/项目说明/"):
        suffix = progress_rel[len("归档/项目说明/") :]
        return "docs/project-notes/" + suffix, "docs"
    if progress_rel.startswith("归档/旧版主文档/"):
        name = PurePosixPath(progress_rel).name
        if name in {"pi_jwm_ton_draft_zh.pdf", "pi_jwm_ton_draft_zh.tex", "IEEEtran.cls"}:
            return "paper/archive/pi_jwm_ton_draft_zh/" + name, "paper"
        return "记录/研究进展/归档/旧版主文档/" + name, "records"
    return "记录/研究进展/" + progress_rel, "records"


def build_default_entries(project_root: Path) -> list[dict[str, object]]:
    root = Path(project_root).resolve()
    moves: list[tuple[Path, str, str]] = []
    code_root = root / "代码"
    for source in _iter_files(code_root):
        rel = source.relative_to(code_root).as_posix()
        moves.append((source, "code/" + rel, "code"))
    document_root = root / "文档"
    for source in _iter_files(document_root):
        target, category = _target_for_document(source, root)
        moves.append((source, target, category))
    root_moves = {
        "本地计划表.md": ("记录/本地计划表.md", "records"),
        "新对话接续说明_20260728.md": ("记录/接续记录/新对话接续说明_20260728.md", "records"),
        "新对话接续说明_20260811.md": ("记录/接续记录/新对话接续说明_20260811.md", "records"),
        "新对话接续说明_20260815.md": ("记录/接续记录/新对话接续说明_20260815.md", "records"),
        "task_plan.md": ("记录/工作日志/task_plan.md", "records"),
        "findings.md": ("记录/工作日志/findings.md", "records"),
        "progress.md": ("记录/工作日志/progress.md", "records"),
    }
    for name, (target, category) in root_moves.items():
        source = root / name
        if source.is_file():
            moves.append((source, target, category))
    entries = [
        {
            "source": _relative(source, root),
            "target": target,
            "category": category,
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source, target, category in moves
    ]
    entries.sort(key=lambda item: str(item["source"]))
    validate_entries(entries)
    return entries


def validate_entries(entries: Sequence[Mapping[str, object]]) -> None:
    sources: set[str] = set()
    targets: set[str] = set()
    for entry in entries:
        source = str(entry.get("source", "")).replace("\\", "/")
        target = str(entry.get("target", "")).replace("\\", "/")
        for label, value in (("source", source), ("target", target)):
            pure = PurePosixPath(value)
            if not value or pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
                raise ValueError(f"{label} must be repository-relative: {value}")
        if source in sources:
            raise ValueError(f"duplicate source: {source}")
        if target in targets:
            raise ValueError(f"duplicate target: {target}")
        sources.add(source)
        targets.add(target)
        size = entry.get("size_bytes")
        digest = str(entry.get("sha256", ""))
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid size for {source}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 for {source}")


def verify_entries(
    project_root: Path,
    entries: Sequence[Mapping[str, object]],
    *,
    phase: str,
) -> dict[str, object]:
    if phase not in {"before", "after"}:
        raise ValueError("phase must be before or after")
    validate_entries(entries)
    root = Path(project_root).resolve()
    result: dict[str, list[str]] = {
        "missing_sources": [],
        "targets_already_present": [],
        "sources_still_present": [],
        "missing_targets": [],
        "size_mismatches": [],
        "hash_mismatches": [],
    }
    for entry in entries:
        source_key = str(entry["source"])
        target_key = str(entry["target"])
        source = root.joinpath(*PurePosixPath(source_key).parts)
        target = root.joinpath(*PurePosixPath(target_key).parts)
        if phase == "before":
            if not source.is_file():
                result["missing_sources"].append(source_key)
                continue
            if target.exists() and source.resolve() != target.resolve():
                result["targets_already_present"].append(target_key)
            candidate = source
        else:
            if source.exists():
                result["sources_still_present"].append(source_key)
            if not target.is_file():
                result["missing_targets"].append(target_key)
                continue
            candidate = target
        if candidate.stat().st_size != int(entry["size_bytes"]):
            result["size_mismatches"].append(target_key if phase == "after" else source_key)
        if sha256_file(candidate) != str(entry["sha256"]):
            result["hash_mismatches"].append(target_key if phase == "after" else source_key)
    errors = sum((len(values) for values in result.values()), 0)
    return {"phase": phase, "passed": errors == 0, "error_count": errors, **result}


def _normalized_error(error: str) -> str:
    value = str(error).replace("\\", "/")
    value = re.sub(r"[^ ]*docs/superpowers/[^ ]+", "docs/superpowers/<deleted-source>", value)
    replacements = (
        ("文档/知识库/", "记录/"),
        ("文档/文献/", "literature/"),
        ("文档/组会/", "meeting/"),
        ("代码/", "code/"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def classify_gate_delta(before: Sequence[str], after: Sequence[str]) -> dict[str, list[str]]:
    before_by_normalized = {_normalized_error(item): str(item) for item in before}
    after_by_normalized = {_normalized_error(item): str(item) for item in after}
    shared = sorted(set(before_by_normalized) & set(after_by_normalized))
    resolved = sorted(set(before_by_normalized) - set(after_by_normalized))
    introduced = sorted(set(after_by_normalized) - set(before_by_normalized))
    return {
        "pre_existing": [before_by_normalized[key] for key in shared],
        "resolved": [before_by_normalized[key] for key in resolved],
        "layout_induced": [after_by_normalized[key] for key in introduced],
    }


def _gate_commands(project_root: Path) -> dict[str, list[str]]:
    root = Path(project_root).resolve()
    code_name = "code" if (root / "code").is_dir() else "代码"
    code = root / code_name
    preflight = code / "artifacts" / "preflight"
    audit = code / "artifacts" / "audit"
    return {
        "p2b_v1": [
            sys.executable,
            str(code / "scripts" / "run_p2_full_dual_graph_collector_preflight_v1.py"),
            "--output-dir",
            str(preflight / "pi_jwm_p2_full_dual_graph_collector_v1"),
            "--verify-only",
        ],
        "p2b_v2_candidate": [
            sys.executable,
            str(code / "scripts" / "run_p2_full_dual_graph_collector_preflight_v2.py"),
            "--output-dir",
            str(preflight / "pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814"),
            "--verify-only",
        ],
        "p2c_v1": [
            sys.executable,
            str(code / "scripts" / "run_p2c_scale_distribution_audit_v1.py"),
            "--bundle",
            str(preflight / "pi_jwm_p2_full_dual_graph_collector_v1"),
            "--output-dir",
            str(audit / "pi_jwm_p2c_scale_distribution_audit_v1"),
            "--verify-only",
        ],
        "p2c_v2_pre_document": [
            sys.executable,
            str(code / "scripts" / "run_p2c_scale_distribution_audit_v2.py"),
            "--bundle",
            str(preflight / "pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814"),
            "--output-dir",
            str(audit / "pi_jwm_p2c_scale_distribution_audit_v2_pre_document_closure_20260814"),
            "--verify-only",
        ],
    }


def _collect_error_strings(value: object, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        direct = value.get("errors")
        if isinstance(direct, list):
            errors.extend(str(item) for item in direct)
        for key, child in value.items():
            if key == "errors":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, list) and child and any(
                token in str(key).lower() for token in ("missing", "mismatch", "error")
            ):
                errors.extend(f"{child_prefix}: {item}" for item in child)
            elif isinstance(child, Mapping):
                errors.extend(_collect_error_strings(child, child_prefix))
    return sorted(dict.fromkeys(errors))


def capture_gates(project_root: Path) -> dict[str, object]:
    gates: dict[str, object] = {}
    for name, command in _gate_commands(project_root).items():
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        report: object | None = None
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = None
        errors = _collect_error_strings(report) if report is not None else []
        if completed.returncode != 0 and not errors:
            errors = [line for line in completed.stderr.splitlines() if line.strip()]
        gates[name] = {
            "command": command,
            "exit_code": completed.returncode,
            "errors": errors,
            "report": report,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return {
        "schema": BASELINE_SCHEMA,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gates": gates,
    }


def write_json(path: Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate-mapping", "verify-before", "verify-after", "capture-gates", "compare-gates"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    return parser


def _load_mapping(path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != MAPPING_SCHEMA or not isinstance(payload.get("entries"), list):
        raise ValueError("invalid migration mapping")
    return payload["entries"]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-mapping":
        if args.output is None:
            raise ValueError("--output is required")
        entries = build_default_entries(args.project_root)
        write_json(
            args.output,
            {
                "schema": MAPPING_SCHEMA,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "entries": entries,
            },
        )
        print(json.dumps({"entry_count": len(entries)}, ensure_ascii=False))
        return 0
    if args.command in {"verify-before", "verify-after"}:
        if args.mapping is None:
            raise ValueError("--mapping is required")
        result = verify_entries(
            args.project_root,
            _load_mapping(args.mapping),
            phase="before" if args.command == "verify-before" else "after",
        )
        if args.output is not None:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "capture-gates":
        if args.output is None:
            raise ValueError("--output is required")
        payload = capture_gates(args.project_root)
        write_json(args.output, payload)
        print(json.dumps({"gate_count": len(payload["gates"])}, ensure_ascii=False))
        return 0
    if args.before is None or args.after is None:
        raise ValueError("--before and --after are required")
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    comparison: dict[str, object] = {}
    for name, before_gate in before["gates"].items():
        after_gate = after["gates"][name]
        comparison[name] = {
            "before_exit_code": before_gate["exit_code"],
            "after_exit_code": after_gate["exit_code"],
            **classify_gate_delta(before_gate["errors"], after_gate["errors"]),
        }
    payload = {"schema": "pi_jwm_repository_layout_gate_delta_v1", "gates": comparison}
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(not item["layout_induced"] for item in comparison.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
