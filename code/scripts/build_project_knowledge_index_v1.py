"""Build retrieval-oriented PI-JWM project registries without changing research artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "PI-JWM-project-knowledge-index-v2"
GENERATED_PREFIX = "docs/registries/generated/"
AI_CONTEXT_FILES = (
    "00_PROJECT_STATE.md",
    "01_RESEARCH_CONTEXT.md",
    "02_ARCHITECTURE.md",
    "03_DATA_FLOW.md",
    "04_MODULE_MAP.md",
    "05_EXPERIMENTS.md",
    "06_DECISIONS.md",
    "07_KNOWN_ISSUES.md",
    "08_CHANGELOG.md",
)

EXPERIMENT_REQUIRED_FIELDS = {
    "id",
    "name",
    "date",
    "research_question",
    "method",
    "code",
    "code_version",
    "configuration",
    "parameters",
    "protocol",
    "data",
    "split",
    "seed",
    "checkpoint",
    "result",
    "metrics",
    "audit",
    "status",
    "conclusion",
    "claim_boundary",
    "locked_test_accessed",
    "field_notes",
}
RESULT_METRIC_TO_GATE = {
    "validation_link_f1_delta": "validation_link_f1_delta",
    "calibration_link_f1_delta": "calibration_link_f1_delta",
    "node_x_overall_ratio": "validation_node_x_mae_ratio",
    "node_x_h5_ratio": "validation_node_x_h5_mae_ratio",
    "node_x_h10_ratio": "validation_node_x_h10_mae_ratio",
    "node_x_h20_ratio": "validation_node_x_h20_mae_ratio",
    "throughput_ratio": "validation_throughput_mae_ratio",
    "rb_occupancy_ratio": "validation_rb_occupancy_mae_ratio",
    "task_delay_ratio": "validation_task_delay_mae_ratio",
}

CURRENT_MODULES = {
    "formal_deterministic_rule_layer_v1",
    "formal_dual_graph_world_model_v1",
    "formal_entity_aligned_rssm_world_model_v1",
    "formal_motion_state_v1",
    "formal_p4_gate_v1",
    "formal_rb_targets_v1",
    "formal_world_model_baselines_v1",
    "formal_world_model_loss_v1",
    "formal_world_model_metrics_v1",
}
CURRENT_SCRIPTS = {
    "build_project_knowledge_index_v1",
    "build_formal_causal_motion_tensor_v1",
    "freeze_formal_p4_entity_rssm_protocol_v1",
    "run_formal_dual_graph_gpu_train_v1",
    "run_formal_entity_aligned_rssm_consistency_audit_v1",
    "run_formal_p4_entity_rssm_gpu_batch_probe_v1",
    "run_formal_p4_entity_rssm_gpu_v1",
    "query_project_knowledge_v1",
}
PROTOTYPE_MODULES = {
    "formal_candidate_rollout_planner_audit_v1",
    "formal_candidate_rollout_planner_v1",
}
PROTOTYPE_SCRIPTS = {"run_formal_candidate_rollout_planner_audit_v1"}
HISTORICAL_PREFIXES = (
    "r3_",
    "r4_",
    "r5_",
    "r6_",
    "v6_",
    "v7_",
    "v8_",
    "v9_",
    "v10_",
    "v11_",
    "analyze_r",
    "analyze_v",
    "audit_v",
    "diagnose_v",
    "evaluate_v",
    "finalize_v",
    "launch_r",
    "merge_v",
    "sweep_v",
    "train_v",
)
ARTIFACT_FAMILIES = (
    "analysis",
    "audit",
    "datasets",
    "evaluation",
    "experiments",
    "formal_data",
    "formal_tensor",
    "formal_training",
    "main_experiment_readiness",
    "preflight",
    "protocol",
    "protocols",
    "reports",
    "research_notes",
    "small_experiments",
)
CONTROL_FILE_PRIORITY = (
    "single_seed_acceptance.json",
    "run_summary.json",
    "audit.json",
    "validation_report.json",
    "protocol.json",
    "result.json",
    "report.json",
    "config.json",
)


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stem(path: str) -> str:
    return Path(path).stem


def classify_tracked_path(path: str) -> dict[str, str]:
    """Return a conservative engineering/evidence classification for one path."""
    normalized = path.replace("\\", "/")
    stem = _stem(normalized)

    if normalized.startswith("code/src/pi_jwm/"):
        category = "framework_source"
        if stem in CURRENT_MODULES:
            status = "current"
        elif stem in PROTOTYPE_MODULES:
            status = "prototype"
        elif stem.startswith(HISTORICAL_PREFIXES):
            status = "historical"
        else:
            status = "support"
        evidence_role = "implementation"
    elif normalized.startswith("code/scripts/"):
        category = "runnable_script"
        if stem in CURRENT_SCRIPTS:
            status = "current"
        elif stem in PROTOTYPE_SCRIPTS:
            status = "prototype"
        elif stem.startswith(HISTORICAL_PREFIXES):
            status = "historical"
        else:
            status = "support"
        evidence_role = "execution_entry"
    elif normalized.startswith("code/tests/"):
        category = "test"
        subject = stem.removeprefix("test_")
        if subject in CURRENT_MODULES or subject in CURRENT_SCRIPTS:
            status = "current"
        elif subject in PROTOTYPE_MODULES or subject in PROTOTYPE_SCRIPTS:
            status = "prototype"
        elif subject.startswith(HISTORICAL_PREFIXES):
            status = "historical"
        else:
            status = "support"
        evidence_role = "test_evidence"
    elif normalized.startswith("code/artifacts/"):
        category = "artifact_metadata"
        status = "support"
        evidence_role = "machine_evidence_index"
    elif normalized in {
        "记录/本地计划表.md",
        "记录/PIJWM主文档.md",
        "记录/8.12之后推进.md",
        "记录/双约束门矩阵_20260826.json",
        "AGENTS.md",
    }:
        category = "governance"
        status = "current"
        evidence_role = "authority"
    elif normalized.startswith("docs/") or normalized in {
        "README.md",
        "PROJECT_CONTEXT.md",
    }:
        category = "documentation"
        status = "current" if "/archive/" not in normalized else "historical"
        evidence_role = "navigation"
    elif normalized.startswith("记录/"):
        category = "research_record"
        status = "historical" if "归档/" in normalized else "support"
        evidence_role = "record"
    else:
        category = "project_file"
        status = "support"
        evidence_role = "support"

    return {
        "category": category,
        "lifecycle_status": status,
        "evidence_role": evidence_role,
    }


def extract_local_imports(source: str) -> set[str]:
    """Extract pi_jwm imports from Python source."""
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("pi_jwm."))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pi_jwm":
                imports.update(f"pi_jwm.{alias.name}" for alias in node.names)
            elif node.module and node.module.startswith("pi_jwm."):
                imports.add(node.module)
    return imports


def _all_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _git_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = [value.decode("utf-8") for value in result.stdout.split(b"\0") if value]
    return sorted(path for path in paths if not path.startswith(GENERATED_PREFIX))


def build_file_inventory(repo_root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for relative in _git_paths(repo_root):
        path = repo_root / relative
        if not path.is_file():
            continue
        classification = classify_tracked_path(relative)
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                **classification,
            }
        )
    return rows


def _module_name(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if normalized.startswith("code/src/pi_jwm/") and normalized.endswith(".py"):
        return f"pi_jwm.{Path(normalized).stem}"
    if normalized.startswith("code/scripts/") and normalized.count("/") == 2 and normalized.endswith(".py"):
        return f"scripts.{Path(normalized).stem}"
    if normalized.startswith("code/tests/") and normalized.count("/") == 2 and normalized.endswith(".py"):
        return f"tests.{Path(normalized).stem}"
    return None


def build_dependency_map(repo_root: Path, inventory: Iterable[dict[str, Any]]) -> dict[str, Any]:
    python_paths = sorted(
        row["path"]
        for row in inventory
        if str(row["path"]).startswith(("code/src/pi_jwm/", "code/scripts/", "code/tests/"))
        and str(row["path"]).endswith(".py")
        and _module_name(str(row["path"])) is not None
    )
    module_by_stem = {
        Path(path).stem: _module_name(path)
        for path in python_paths
        if path.startswith("code/scripts/")
    }
    path_by_module = {_module_name(path): path for path in python_paths}
    imports_by_module: dict[str, set[str]] = {}
    parse_errors: list[dict[str, str]] = []

    for path in python_paths:
        module = _module_name(path)
        if module is None:
            continue
        try:
            source = (repo_root / path).read_text(encoding="utf-8-sig")
            raw_imports = _all_imports(source)
            local = extract_local_imports(source)
            for imported in raw_imports:
                first = imported.split(".", 1)[0]
                if first in module_by_stem:
                    local.add(str(module_by_stem[first]))
            imports_by_module[module] = {
                value for value in local if value in path_by_module
            }
        except (SyntaxError, UnicodeDecodeError) as exc:
            imports_by_module[module] = set()
            parse_errors.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})

    imported_by: dict[str, set[str]] = defaultdict(set)
    for module, imports in imports_by_module.items():
        for imported in imports:
            imported_by[imported].add(module)

    nodes: list[dict[str, Any]] = []
    for module in sorted(path_by_module):
        path = path_by_module[module]
        classification = classify_tracked_path(path)
        test_files = sorted(
            path_by_module[value]
            for value in imported_by.get(module, set())
            if value.startswith("tests.")
        )
        subject = module.rsplit(".", 1)[-1]
        conventional_test = f"code/tests/test_{subject}.py"
        if conventional_test in python_paths and conventional_test not in test_files:
            test_files.append(conventional_test)
        nodes.append(
            {
                "module": module,
                "path": path,
                **classification,
                "imports": sorted(imports_by_module.get(module, set())),
                "imported_by": sorted(imported_by.get(module, set())),
                "test_files": sorted(test_files),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "nodes": nodes, "parse_errors": parse_errors}


def build_archive_candidates(
    inventory: Iterable[dict[str, Any]], dependencies: dict[str, Any]
) -> list[dict[str, str | int]]:
    """Build a non-destructive, per-file candidate map for historical Python paths."""
    inventory_by_path = {str(row["path"]): row for row in inventory}
    nodes = list(dependencies.get("nodes", []))
    status_by_module = {
        str(node["module"]): str(node.get("lifecycle_status", "support")) for node in nodes
    }
    rows: list[dict[str, str | int]] = []
    for node in nodes:
        if node.get("lifecycle_status") != "historical":
            continue
        path = str(node["path"])
        inventory_row = inventory_by_path.get(path, {})
        imported_by = list(node.get("imported_by", []))
        current_imported_by = [
            module for module in imported_by if status_by_module.get(str(module)) == "current"
        ]
        rows.append(
            {
                "path": path,
                "module": str(node["module"]),
                "size_bytes": int(inventory_row.get("size_bytes", 0)),
                "sha256": str(inventory_row.get("sha256", "")),
                "imported_by_count": len(imported_by),
                "current_imported_by_count": len(current_imported_by),
                "test_file_count": len(node.get("test_files", [])),
                "proposed_action": "keep_in_place_logical_archive",
                "proposed_target": "",
                "rollback_path": path,
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def _read_json(path: Path) -> tuple[Any | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), ""
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _find_value(value: Any, keys: tuple[str, ...]) -> Any | None:
    if isinstance(value, dict):
        for key in keys:
            if key in value and value[key] is not None:
                return value[key]
        for child in value.values():
            found = _find_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_value(child, keys)
            if found is not None:
                return found
    return None


def _scalar(value: Any | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _control_json_files(artifact: Path) -> list[Path]:
    direct = sorted(artifact.glob("*.json"))
    by_name = {path.name: path for path in direct}
    ordered = [by_name[name] for name in CONTROL_FILE_PRIORITY if name in by_name]
    ordered.extend(path for path in direct if path not in ordered)
    return ordered


def build_artifact_record(artifact_root: Path, artifact: Path) -> dict[str, str | int]:
    """Summarize control evidence only; checkpoints and predictions are never hashed here."""
    relative = _as_posix(artifact.relative_to(artifact_root))
    control_files = _control_json_files(artifact)
    values: list[Any] = []
    errors: list[str] = []
    for path in control_files:
        value, error = _read_json(path)
        if value is not None:
            values.append(value)
        if error:
            errors.append(f"{path.name}: {error}")

    combined: Any = values
    status = _find_value(combined, ("status", "gate_status", "audit_status", "decision"))
    seed = _find_value(combined, ("seed", "training_seed"))
    method = _find_value(combined, ("method", "model", "candidate", "method_id"))
    locked = _find_value(combined, ("locked_test_accessed",))

    all_files: list[Path] = []
    try:
        all_files = [path for path in artifact.rglob("*") if path.is_file()]
    except OSError as exc:
        errors.append(f"walk: {type(exc).__name__}: {exc}")
    modified_ns = max((path.stat().st_mtime_ns for path in all_files), default=artifact.stat().st_mtime_ns)

    manifest = artifact / "manifest.json"
    config = artifact / "config.json"
    primary = next((path for path in control_files if path.name in CONTROL_FILE_PRIORITY), None)
    hashed: list[str] = []
    for path in (manifest, config, primary):
        if path is None or not path.is_file() or path in {item[0] for item in []}:
            continue
        try:
            rel = _as_posix(path.relative_to(artifact_root))
            value = f"{rel}:{_sha256(path)}"
            if value not in hashed:
                hashed.append(value)
        except OSError as exc:
            errors.append(f"hash {path.name}: {type(exc).__name__}: {exc}")

    return {
        "artifact_id": relative,
        "family": relative.split("/", 1)[0],
        "status": _scalar(status) or ("unreadable" if errors and not values else "unclassified"),
        "seed": _scalar(seed),
        "method": _scalar(method),
        "locked_test_accessed": _scalar(locked),
        "file_count": len(all_files),
        "size_bytes": sum(path.stat().st_size for path in all_files),
        "modified_time_ns": modified_ns,
        "config_path": _as_posix(config.relative_to(artifact_root)) if config.is_file() else "",
        "manifest_path": _as_posix(manifest.relative_to(artifact_root)) if manifest.is_file() else "",
        "primary_result_path": _as_posix(primary.relative_to(artifact_root)) if primary else "",
        "hashed_control_files": ";".join(hashed),
        "read_errors": " | ".join(errors),
    }


def build_artifact_catalog(repo_root: Path) -> list[dict[str, str | int]]:
    artifact_root = repo_root / "code" / "artifacts"
    rows: list[dict[str, str | int]] = []
    for family in ARTIFACT_FAMILIES:
        root = artifact_root / family
        if not root.is_dir():
            continue
        for artifact in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
            rows.append(build_artifact_record(artifact_root, artifact))
    return rows


def validate_deferred_work(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "PI-JWM-deferred-work-v1":
        raise ValueError("unsupported deferred-work schema")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("deferred-work items must be a non-empty list")
    ids: set[str] = set()
    for item in items:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("deferred-work ids must be unique non-empty strings")
        ids.add(identifier)
        if item.get("status") != "deferred":
            raise ValueError(f"{identifier}: status must be deferred")
        if item.get("authorization_required") is not True:
            raise ValueError(f"{identifier}: authorization_required must be true")
        if item.get("auto_start") is not False:
            raise ValueError(f"{identifier}: auto_start must be false")
        if item.get("locked_test_accessed") is not False:
            raise ValueError(f"{identifier}: locked_test_accessed must be false")


def validate_ai_context(repo_root: Path) -> dict[str, Any]:
    """Validate the stable ChatGPT context entrypoints and evidence boundaries."""

    errors: list[str] = []
    validated = 0
    root = repo_root / "AI_CONTEXT"
    if not root.is_dir():
        return {
            "validated_file_count": 0,
            "errors": ["missing AI_CONTEXT directory"],
        }
    required_markers = {
        "00_PROJECT_STATE.md": ("Current State Snapshot", "Source of truth", "Latest commit"),
        "01_RESEARCH_CONTEXT.md": ("Research Rationale", "Unverified"),
        "02_ARCHITECTURE.md": ("formal_entity_aligned_rssm_world_model_v1.py", "Source of truth"),
        "03_DATA_FLOW.md": ("FormalAirFogSimWindowDataset", "locked_test_accessed=false"),
        "04_MODULE_MAP.md": ("formal_world_model_loss_v1.py", "Source of truth"),
        "05_EXPERIMENTS.md": ("P4-EARSSM-SEED-20260832", "Unverified"),
        "06_DECISIONS.md": ("Researcher Decision", "Unverified"),
        "07_KNOWN_ISSUES.md": ("Status: Awaiting Researcher Decision", "Actual Implementation"),
        "08_CHANGELOG.md": ("Context Consistency Check", "Source of truth"),
    }
    for name in AI_CONTEXT_FILES:
        path = root / name
        if not path.is_file():
            errors.append(f"missing AI_CONTEXT/{name}")
            continue
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in required_markers[name] if marker not in text]
        if missing:
            errors.append(f"AI_CONTEXT/{name}: missing markers {missing}")
            continue
        validated += 1
    return {"validated_file_count": validated, "errors": errors}


def _load_required_registry(repo_root: Path, name: str) -> dict[str, Any]:
    path = repo_root / "docs" / "registries" / name
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        raise ValueError(f"cannot read {name}: {error}")
    return payload


def _validate_unique_ids(rows: list[dict[str, Any]], *, label: str) -> None:
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError(f"{label} ids must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} ids must be unique")


def _assert_existing(repo_root: Path, relative: str, *, label: str) -> None:
    path_text = relative.split("#", 1)[0]
    if not (repo_root / path_text).exists():
        raise ValueError(f"{label}: missing path {relative}")


def validate_experiment_registry(repo_root: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "PI-JWM-experiment-registry-v2":
        raise ValueError("unsupported experiment-registry schema")
    rows = payload.get("experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("experiment registry must contain experiments")
    _validate_unique_ids(rows, label="experiment")
    for row in rows:
        identifier = row["id"]
        missing = sorted(EXPERIMENT_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"{identifier}: missing fields {missing}")
        notes = row.get("field_notes")
        if not isinstance(notes, dict):
            raise ValueError(f"{identifier}: field_notes must be an object")
        for field in EXPERIMENT_REQUIRED_FIELDS - {"field_notes"}:
            if row[field] is None and not notes.get(field):
                raise ValueError(f"{identifier}: null {field} requires field_notes")
        if row.get("locked_test_accessed") is not False:
            raise ValueError(f"{identifier}: locked_test_accessed must be false")
        for field in ("code", "configuration", "protocol", "data", "checkpoint", "result", "audit"):
            relative = row.get(field)
            if isinstance(relative, str) and relative:
                _assert_existing(repo_root, relative, label=f"{identifier}:{field}")
        code_version = row.get("code_version")
        if not isinstance(code_version, dict) or not code_version.get("identity_type"):
            raise ValueError(f"{identifier}: code_version must record an identity type")
        manifest = code_version.get("manifest")
        if isinstance(manifest, str) and manifest:
            _assert_existing(repo_root, manifest, label=f"{identifier}:code_version.manifest")
            expected_hash = code_version.get("manifest_sha256")
            if expected_hash and _sha256(repo_root / manifest) != str(expected_hash).lower():
                raise ValueError(f"{identifier}: code-version manifest SHA-256 mismatch")


def validate_historical_method_registry(repo_root: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "PI-JWM-historical-method-registry-v1":
        raise ValueError("unsupported historical-method-registry schema")
    rows = payload.get("methods")
    if not isinstance(rows, list) or not rows:
        raise ValueError("historical method registry must contain methods")
    _validate_unique_ids(rows, label="historical method")
    for row in rows:
        identifier = row["id"]
        if row.get("status") == "current" or not row.get("why_not_current"):
            raise ValueError(f"{identifier}: historical status and reason are required")
        for field in ("experiment_paths", "evidence_paths"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"{identifier}: {field} must be non-empty")
            for relative in values:
                _assert_existing(repo_root, relative, label=f"{identifier}:{field}")


def validate_question_routes(repo_root: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "PI-JWM-question-routes-v1":
        raise ValueError("unsupported question-routes schema")
    rows = payload.get("routes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("question routes must be non-empty")
    _validate_unique_ids(rows, label="question route")
    required_ids = {
        "ROUTE-CURRENT-METHOD",
        "ROUTE-EXPERIMENT-HISTORY",
        "ROUTE-RESULT-PROVENANCE",
        "ROUTE-DEFERRED-WORK",
        "ROUTE-CHATGPT-ONBOARDING",
    }
    if not required_ids.issubset({row["id"] for row in rows}):
        raise ValueError("required question routes are missing")
    for row in rows:
        if not row.get("keywords") or not row.get("primary_sources"):
            raise ValueError(f"{row['id']}: keywords and primary_sources are required")
        for relative in list(row["primary_sources"]) + list(row.get("verification_sources", [])):
            _assert_existing(repo_root, relative, label=row["id"])


def validate_result_registry_against_evidence(
    repo_root: Path,
    result_payload: dict[str, Any],
    experiment_payload: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare registered formal numbers with their original acceptance JSON."""
    mismatches: list[str] = []
    if result_payload.get("schema_version") != "PI-JWM-results-registry-v2":
        mismatches.append("unsupported results-registry schema")
        return {"validated_result_count": 0, "mismatches": mismatches}
    experiments = {row["id"]: row for row in experiment_payload.get("experiments", [])}
    results = result_payload.get("results", [])
    if not isinstance(results, list):
        return {"validated_result_count": 0, "mismatches": ["results must be a list"]}
    for result in results:
        identifier = str(result.get("id", "<missing-id>"))
        experiment = experiments.get(result.get("experiment_id"))
        if experiment is None:
            mismatches.append(f"{identifier}: experiment link missing")
            continue
        audit_relative = result.get("audit")
        if not isinstance(audit_relative, str):
            mismatches.append(f"{identifier}: audit path missing")
            continue
        audit_path = repo_root / audit_relative
        audit, error = _read_json(audit_path)
        if error or not isinstance(audit, dict):
            mismatches.append(f"{identifier}: cannot read audit: {error}")
            continue
        if _sha256(audit_path) != str(result.get("audit_sha256", "")).lower():
            mismatches.append(f"{identifier}: audit SHA-256 mismatch")
        if result.get("seed") != audit.get("seed") or result.get("seed") != experiment.get("seed"):
            mismatches.append(f"{identifier}: seed mismatch")
        if result.get("selected_epoch") != audit.get("selection", {}).get("best_epoch"):
            mismatches.append(f"{identifier}: selected epoch mismatch")
        if audit.get("status") != "passed" or result.get("status") != "passed_single_seed":
            mismatches.append(f"{identifier}: acceptance status mismatch")
        if audit.get("locked_test_accessed") is not False or result.get("locked_test_accessed") is not False:
            mismatches.append(f"{identifier}: locked-test boundary mismatch")
        if audit.get("formal_performance_claim_ready") is not False:
            mismatches.append(f"{identifier}: formal claim boundary mismatch")
        gate_rows = audit.get("gate_recompute", {}).get("recomputed", {}).get("gates", [])
        gate_values = {row.get("name"): row.get("value") for row in gate_rows}
        registered_metrics = result.get("metrics", {})
        for metric_name, gate_name in RESULT_METRIC_TO_GATE.items():
            registered = registered_metrics.get(metric_name)
            original = gate_values.get(gate_name)
            if not isinstance(registered, (int, float)) or not isinstance(original, (int, float)):
                mismatches.append(f"{identifier}: missing numeric metric {metric_name}")
            elif abs(float(registered) - float(original)) > tolerance:
                mismatches.append(f"{identifier}: metric mismatch {metric_name}")
    return {"validated_result_count": len(results), "mismatches": mismatches}


def _csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def build_outputs(repo_root: Path) -> dict[str, str]:
    inventory = build_file_inventory(repo_root)
    dependencies = build_dependency_map(repo_root, inventory)
    artifacts = build_artifact_catalog(repo_root)
    archive_candidates = build_archive_candidates(inventory, dependencies)
    deferred = _load_required_registry(repo_root, "deferred_work.json")
    experiments = _load_required_registry(repo_root, "experiment_registry.json")
    results = _load_required_registry(repo_root, "results_registry.json")
    historical_methods = _load_required_registry(repo_root, "historical_method_registry.json")
    question_routes = _load_required_registry(repo_root, "question_routes.json")
    ai_context_validation = validate_ai_context(repo_root)
    if ai_context_validation["errors"]:
        raise ValueError(
            "AI_CONTEXT validation failed: " + "; ".join(ai_context_validation["errors"])
        )
    validate_deferred_work(deferred)
    validate_experiment_registry(repo_root, experiments)
    validate_historical_method_registry(repo_root, historical_methods)
    validate_question_routes(repo_root, question_routes)
    result_validation = validate_result_registry_against_evidence(
        repo_root, results, experiments
    )
    if result_validation["mismatches"]:
        raise ValueError(
            "results registry does not match original evidence: "
            + "; ".join(result_validation["mismatches"])
        )

    inventory_fields = [
        "path",
        "size_bytes",
        "sha256",
        "category",
        "lifecycle_status",
        "evidence_role",
    ]
    artifact_fields = [
        "artifact_id",
        "family",
        "status",
        "seed",
        "method",
        "locked_test_accessed",
        "file_count",
        "size_bytes",
        "modified_time_ns",
        "config_path",
        "manifest_path",
        "primary_result_path",
        "hashed_control_files",
        "read_errors",
    ]
    archive_fields = [
        "path",
        "module",
        "size_bytes",
        "sha256",
        "imported_by_count",
        "current_imported_by_count",
        "test_file_count",
        "proposed_action",
        "proposed_target",
        "rollback_path",
    ]
    lifecycle_counts: dict[str, int] = defaultdict(int)
    for row in inventory:
        lifecycle_counts[str(row["lifecycle_status"])] += 1
    artifact_family_counts: dict[str, int] = defaultdict(int)
    for row in artifacts:
        artifact_family_counts[str(row["family"])] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_scope": "version-controlled project files excluding generated registries; artifact control files only",
        "tracked_file_count": len(inventory),
        "python_node_count": len(dependencies["nodes"]),
        "python_parse_error_count": len(dependencies["parse_errors"]),
        "artifact_record_count": len(artifacts),
        "artifact_read_error_count": sum(bool(row["read_errors"]) for row in artifacts),
        "archive_candidate_count": len(archive_candidates),
        "archive_candidates_with_current_references": sum(
            int(row["current_imported_by_count"]) > 0 for row in archive_candidates
        ),
        "deferred_item_count": len(deferred["items"]),
        "important_experiment_count": len(experiments["experiments"]),
        "verified_result_count": result_validation["validated_result_count"],
        "result_evidence_mismatch_count": len(result_validation["mismatches"]),
        "historical_method_count": len(historical_methods["methods"]),
        "question_route_count": len(question_routes["routes"]),
        "ai_context_file_count": ai_context_validation["validated_file_count"],
        "ai_context_error_count": len(ai_context_validation["errors"]),
        "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "artifact_family_counts": dict(sorted(artifact_family_counts.items())),
        "locked_test_boundary": "sealed",
        "formal_performance_claim_ready": False,
    }
    return {
        f"{GENERATED_PREFIX}tracked_file_inventory.csv": _csv_text(inventory, inventory_fields),
        f"{GENERATED_PREFIX}python_dependency_map.json": _json_text(dependencies),
        f"{GENERATED_PREFIX}artifact_catalog.csv": _csv_text(artifacts, artifact_fields),
        f"{GENERATED_PREFIX}archive_candidate_registry.csv": _csv_text(
            archive_candidates, archive_fields
        ),
        f"{GENERATED_PREFIX}registry_summary.json": _json_text(summary),
    }


def write_or_check(repo_root: Path, *, check: bool) -> dict[str, Any]:
    outputs = build_outputs(repo_root)
    mismatches: list[str] = []
    for relative, content in outputs.items():
        target = repo_root / relative
        if check:
            if not target.is_file() or target.read_text(encoding="utf-8") != content:
                mismatches.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "check" if check else "write",
        "output_count": len(outputs),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = write_or_check(args.repo_root.resolve(), check=args.check)
    print(_json_text(result), end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
