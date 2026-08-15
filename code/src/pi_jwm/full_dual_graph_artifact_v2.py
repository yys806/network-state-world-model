"""Version-isolated, ledger-bound P2-B v2 artifact publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from pi_jwm.path_compat import load_exact_mapping, load_source_changes, resolve_repository_path

from .action_attempt_ledger_v1 import (
    candidate_digest,
    summarize_attempts,
    validate_attempt_records,
)
from .full_dual_graph_artifact_v1 import (
    build_full_collector_status_flags,
    validate_trajectory_frames,
)


ARTIFACT_CONTRACT_VERSION = "PIJWM-Full-Dual-Graph-Artifact-v2"
FAILURE_ARTIFACT_CONTRACT_VERSION = "PIJWM-Full-Dual-Graph-Failure-Artifact-v2"
SUCCESS_REQUIRED_FILES = (
    "collector_config.json",
    "vocabularies.json",
    "frames.jsonl",
    "action_attempts.jsonl",
    "coverage_report.json",
    "validation_report.json",
    "replay_report.json",
    "status_flags.json",
    "manifest.json",
)
FAILURE_REQUIRED_FILES = (
    "action_attempts.jsonl",
    "failure_report.json",
    "manifest.json",
)
_SUCCESS_PAYLOADS = set(SUCCESS_REQUIRED_FILES) - {"manifest.json"}
_FAILURE_PAYLOADS = set(FAILURE_REQUIRED_FILES) - {"manifest.json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: object) -> None:
    if not isinstance(rows, (list, tuple)):
        raise TypeError(f"{path.name} payload must be a sequence")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _source_hashes(source_paths: Sequence[Path]) -> dict[str, str]:
    sources = tuple(Path(path).absolute() for path in source_paths)
    if not sources:
        raise ValueError("source_paths must not be empty")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    common_root = Path(os.path.commonpath([str(path) for path in sources]))
    if common_root.is_file():
        common_root = common_root.parent
    result: dict[str, str] = {}
    for source in sorted(sources, key=lambda item: str(item)):
        key = source.relative_to(common_root).as_posix()
        pure = PurePosixPath(key)
        if pure.is_absolute() or ".." in pure.parts or key.startswith(".worktrees/"):
            raise ValueError(f"non-portable source key: {key}")
        if key in result:
            raise ValueError(f"duplicate source key: {key}")
        result[key] = _sha256(source)
    return result


def _identity_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row.get("episode_id"),
        row.get("trajectory_id"),
        row.get("frame_index"),
        row.get("candidate_ordinal"),
    )


def validate_bundle_alignment(
    frames: Sequence[Mapping[str, object]],
    attempts: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    errors = list(validate_attempt_records(attempts))
    if errors:
        return tuple(errors)
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        return ("frames must be a sequence",)
    if any(not isinstance(frame, Mapping) for frame in frames):
        return ("every frame must be an object",)

    if any(row.get("candidate_ordinal") != 0 for row in attempts):
        errors.append("current P2-B v2 permits only candidate_ordinal=0")
    rejected = [row for row in attempts if row.get("disposition") == "rejected"]
    if rejected:
        errors.append("a successful P2-B v2 bundle cannot contain rejected attempts")

    frame_map: dict[tuple[object, object, int], Mapping[str, object]] = {}
    for frame in frames:
        trajectory_id = frame.get("trajectory_id")
        frame_index = frame.get("frame_index")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            errors.append("frame trajectory_id must be non-empty")
            continue
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            errors.append("frame_index must be a nonnegative integer")
            continue
        key = (trajectory_id, frame_index, 0)
        if key in frame_map:
            errors.append(f"duplicate frame identity: {key}")
        frame_map[key] = frame
        if frame.get("fixture") is not False:
            errors.append(f"published training-shaped frame is not natural reference: {key}")
        if frame.get("quarantined") is not False:
            errors.append(f"published frame is quarantined: {key}")
        if frame.get("training_eligible") is not False:
            errors.append(f"published frame training_eligible must be false: {key}")

    reference_rows = [
        row
        for row in attempts
        if row.get("run_role") == "natural_reference"
        and row.get("disposition") == "accepted"
    ]
    reference_map: dict[tuple[object, object, int], Mapping[str, object]] = {}
    for row in reference_rows:
        key = (row.get("trajectory_id"), row.get("frame_index"), row.get("candidate_ordinal"))
        if key in reference_map:
            errors.append(f"duplicate natural-reference identity: {key}")
        reference_map[key] = row
    if set(frame_map) != set(reference_map):
        errors.append("accepted natural-reference attempts and frames are not one-to-one")
    for key in sorted(set(frame_map) & set(reference_map), key=str):
        try:
            observed_digest = candidate_digest(frame_map[key].get("action"))
        except Exception as exc:
            errors.append(f"frame candidate is not digestible for {key}: {exc}")
        else:
            if reference_map[key].get("candidate_digest") != observed_digest:
                errors.append(f"frame/attempt candidate digest mismatch: {key}")

    reference_by_episode = {_identity_key(row): row for row in reference_rows}
    replay_rows = [
        row
        for row in attempts
        if row.get("run_role") == "natural_replay"
        and row.get("disposition") == "accepted"
    ]
    replay_by_episode = {_identity_key(row): row for row in replay_rows}
    if len(reference_by_episode) != len(reference_rows):
        errors.append("duplicate natural-reference episode/frame identity")
    if len(replay_by_episode) != len(replay_rows):
        errors.append("duplicate natural-replay episode/frame identity")
    if set(reference_by_episode) != set(replay_by_episode):
        errors.append("natural reference/replay attempt matrices differ")
    for key in sorted(set(reference_by_episode) & set(replay_by_episode), key=str):
        reference = reference_by_episode[key]
        replay = replay_by_episode[key]
        if reference.get("candidate_digest") != replay.get("candidate_digest"):
            errors.append(f"natural reference/replay candidate digest mismatch: {key}")
        if reference.get("stage_trace") != replay.get("stage_trace"):
            errors.append(f"natural reference/replay stage structure mismatch: {key}")
        if reference.get("disposition") != replay.get("disposition"):
            errors.append(f"natural reference/replay disposition mismatch: {key}")
    return tuple(errors)


def validate_success_payloads(payloads: Mapping[str, object]) -> tuple[str, ...]:
    errors: list[str] = []
    if set(payloads) != _SUCCESS_PAYLOADS:
        return (f"success payload names differ: {sorted(set(payloads) ^ _SUCCESS_PAYLOADS)}",)
    frames = payloads.get("frames.jsonl")
    attempts = payloads.get("action_attempts.jsonl")
    vocabularies = payloads.get("vocabularies.json")
    if not isinstance(frames, list):
        errors.append("frames.jsonl must be an array before publication")
    if not isinstance(attempts, list):
        errors.append("action_attempts.jsonl must be an array before publication")
    if isinstance(frames, list) and isinstance(attempts, list):
        errors.extend(validate_bundle_alignment(frames, attempts))
        try:
            summary = summarize_attempts(attempts)
        except Exception as exc:
            errors.append(f"ledger summary recomputation failed: {exc}")
        else:
            if summary["binary_conservation_passed"] is not True:
                errors.append("ledger binary conservation failed")
    if not isinstance(vocabularies, Mapping):
        errors.append("vocabularies.json must be an object")
    elif isinstance(frames, list):
        grouped: dict[str, list[Mapping[str, object]]] = {}
        for frame in frames:
            if isinstance(frame, Mapping) and isinstance(frame.get("trajectory_id"), str):
                grouped.setdefault(str(frame["trajectory_id"]), []).append(frame)
        if set(grouped) != set(vocabularies):
            errors.append("frame/vocabulary trajectory matrices differ")
        else:
            for trajectory_id in sorted(grouped):
                trajectory_frames = sorted(
                    grouped[trajectory_id], key=lambda row: int(row.get("frame_index", -1))
                )
                errors.extend(
                    f"trajectory {trajectory_id}: {error}"
                    for error in validate_trajectory_frames(
                        trajectory_frames,
                        vocabulary=vocabularies[trajectory_id],
                        fixture=False,
                    )
                )
    for report_name in ("validation_report.json", "replay_report.json"):
        report = payloads.get(report_name)
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            errors.append(f"{report_name} did not pass")
    if payloads.get("status_flags.json") != build_full_collector_status_flags(passed=True):
        errors.append("status flags exceed or differ from conservative preflight scope")
    config = payloads.get("collector_config.json")
    if not isinstance(config, Mapping):
        errors.append("collector_config.json must be an object")
    else:
        for flag in ("formal_data_approved", "training_eligible"):
            if config.get(flag) is not False:
                errors.append(f"collector config {flag} must be false")
    serialized = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    for forbidden in ("legacy_13", "13维补齐", "13_dim_filled"):
        if forbidden in serialized:
            errors.append(f"forbidden fabricated-width claim found: {forbidden}")
    return tuple(errors)


def assert_publish_targets_absent(output_dir: Path) -> None:
    output = Path(output_dir)
    failed = output.with_name(f"{output.name}_failed")
    existing = [str(path) for path in (output, failed) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite P2-B v2 artifact target: {existing}")


def _publish(
    output_dir: Path,
    payloads: Mapping[str, object],
    *,
    required_files: Sequence[str],
    schema_version: str,
    source_paths: Sequence[Path],
    status: Mapping[str, object],
) -> None:
    output = Path(output_dir)
    source_hashes = _source_hashes(source_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        for name in sorted(set(required_files) - {"manifest.json"}):
            if name.endswith(".jsonl"):
                _write_jsonl(temporary / name, payloads[name])
            else:
                _write_json(temporary / name, payloads[name])
        artifact_hashes = {
            name: _sha256(temporary / name)
            for name in sorted(set(required_files) - {"manifest.json"})
        }
        manifest = {
            "schema_version": schema_version,
            "required_files": list(required_files),
            "artifact_hashes": artifact_hashes,
            "source_hashes": source_hashes,
            "status": dict(status),
        }
        _write_json(temporary / "manifest.json", manifest)
        for name, expected in artifact_hashes.items():
            if _sha256(temporary / name) != expected:
                raise RuntimeError(f"artifact hash changed before publication: {name}")
        if _source_hashes(source_paths) != source_hashes:
            raise RuntimeError("source hash changed before publication")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def publish_success_bundle(
    output_dir: Path,
    payloads: Mapping[str, object],
    source_paths: Sequence[Path],
) -> None:
    assert_publish_targets_absent(output_dir)
    errors = validate_success_payloads(payloads)
    if errors:
        raise ValueError(f"success payload invalid: {list(errors)}")
    _publish(
        Path(output_dir),
        payloads,
        required_files=SUCCESS_REQUIRED_FILES,
        schema_version=ARTIFACT_CONTRACT_VERSION,
        source_paths=source_paths,
        status=payloads["status_flags.json"],  # type: ignore[arg-type]
    )


def _validate_failure(
    attempts: object,
    failure_report: object,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(attempts, list):
        return ("failure attempts must be an array",)
    errors.extend(validate_attempt_records(attempts))
    if not isinstance(failure_report, Mapping):
        return (*errors, "failure_report.json must be an object")
    for flag in ("formal_data_approved", "training_eligible", "gpu_started", "locked_test_accessed"):
        if failure_report.get(flag) is not False:
            errors.append(f"failure report {flag} must be false")
    scope = failure_report.get("failure_scope")
    if scope == "attempt":
        rejected = [row for row in attempts if row.get("disposition") == "rejected"]
        if not rejected:
            errors.append("attempt failure report lacks a rejected terminal attempt")
        elif failure_report.get("attempt_id") not in {row.get("attempt_id") for row in rejected}:
            errors.append("failure report attempt_id is not a rejected ledger row")
        for field in ("run_role", "terminal_stage", "rejection_code", "quarantined"):
            matching = next(
                (row for row in rejected if row.get("attempt_id") == failure_report.get("attempt_id")),
                None,
            )
            if matching is not None and failure_report.get(field) != matching.get(field):
                errors.append(f"failure report {field} differs from ledger")
    elif scope == "run_level_observation_failure":
        if failure_report.get("attempt_id") is not None:
            errors.append("run-level observation failure cannot claim an attempt_id")
    else:
        errors.append("failure_scope must be attempt or run_level_observation_failure")
    for field in ("error_type", "error_detail"):
        if not isinstance(failure_report.get(field), str) or not str(failure_report.get(field)).strip():
            errors.append(f"failure report lacks {field}")
    return tuple(errors)


def publish_failure_bundle(
    output_dir: Path,
    attempts: list[Mapping[str, object]],
    failure_report: Mapping[str, object],
    source_paths: Sequence[Path],
) -> Path:
    output = Path(output_dir)
    assert_publish_targets_absent(output)
    failed = output.with_name(f"{output.name}_failed")
    errors = _validate_failure(attempts, failure_report)
    if errors:
        raise ValueError(f"failure payload invalid: {list(errors)}")
    payloads = {
        "action_attempts.jsonl": attempts,
        "failure_report.json": failure_report,
    }
    _publish(
        failed,
        payloads,
        required_files=FAILURE_REQUIRED_FILES,
        schema_version=FAILURE_ARTIFACT_CONTRACT_VERSION,
        source_paths=source_paths,
        status={
            flag: failure_report[flag]
            for flag in (
                "formal_data_approved",
                "training_eligible",
                "gpu_started",
                "locked_test_accessed",
            )
        },
    )
    return failed


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[object]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_manifest(
    output_dir: Path,
    *,
    required_files: Sequence[str],
    schema_version: str,
    source_paths: Sequence[Path],
    project_root: Path | None = None,
) -> tuple[dict[str, object] | None, list[str]]:
    output = Path(output_dir)
    errors: list[str] = []
    missing = [name for name in required_files if not (output / name).is_file()]
    if missing:
        return None, [f"missing artifact files: {missing}"]
    try:
        manifest = _load_json(output / "manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"manifest unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return None, ["manifest must be an object"]
    if manifest.get("schema_version") != schema_version:
        errors.append("manifest schema mismatch")
    if manifest.get("required_files") != list(required_files):
        errors.append("manifest required_files mismatch")
    hashes = manifest.get("artifact_hashes")
    expected_names = set(required_files) - {"manifest.json"}
    if not isinstance(hashes, Mapping) or set(hashes) != expected_names:
        errors.append("manifest artifact hash matrix mismatch")
    else:
        for name in sorted(expected_names):
            if hashes.get(name) != _sha256(output / name):
                errors.append(f"artifact hash mismatch: {name}")
    try:
        if project_root is None:
            expected_sources = _source_hashes(source_paths)
        else:
            mapping_path = Path(project_root) / "记录" / "迁移" / "2026-08-16-仓库目录迁移映射.json"
            exact = load_exact_mapping(mapping_path) if mapping_path.is_file() else None
            changes_path = Path(project_root) / "记录" / "迁移" / "2026-08-16-迁移源变更.json"
            source_changes = load_source_changes(changes_path) if changes_path.is_file() else {}
            recorded = manifest.get("source_hashes")
            if not isinstance(recorded, Mapping):
                raise ValueError("manifest source_hashes must be an object")
            expected_sources = {}
            for raw_key in sorted(str(key) for key in recorded):
                source = (
                    resolve_repository_path(Path(project_root), raw_key, exact_mapping=exact)
                    if exact is not None
                    else Path(project_root).joinpath(*PurePosixPath(raw_key).parts)
                )
                if not source.is_file():
                    raise FileNotFoundError(f"source is missing: {source}")
                observed_hash = _sha256(source)
                expected_sources[raw_key] = (
                    str(recorded[raw_key])
                    if source_changes.get(raw_key) == observed_hash
                    else observed_hash
                )
    except (OSError, ValueError) as exc:
        errors.append(f"source verification failure: {exc}")
    else:
        if manifest.get("source_hashes") != expected_sources:
            errors.append("source hash mismatch")
    return manifest, errors


def verify_success_bundle(
    output_dir: Path,
    source_paths: Sequence[Path],
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    _, errors = _verify_manifest(
        output,
        required_files=SUCCESS_REQUIRED_FILES,
        schema_version=ARTIFACT_CONTRACT_VERSION,
        source_paths=source_paths,
        project_root=project_root,
    )
    if not errors:
        try:
            payloads = {
                name: (
                    _load_jsonl(output / name)
                    if name.endswith(".jsonl")
                    else _load_json(output / name)
                )
                for name in sorted(_SUCCESS_PAYLOADS)
            }
            errors.extend(validate_success_payloads(payloads))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"success payload unreadable: {exc}")
    return {"passed": not errors, "errors": errors}


def verify_failure_bundle(
    failed_dir: Path,
    source_paths: Sequence[Path],
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    failed = Path(failed_dir)
    _, errors = _verify_manifest(
        failed,
        required_files=FAILURE_REQUIRED_FILES,
        schema_version=FAILURE_ARTIFACT_CONTRACT_VERSION,
        source_paths=source_paths,
        project_root=project_root,
    )
    if not errors:
        try:
            attempts = _load_jsonl(failed / "action_attempts.jsonl")
            report = _load_json(failed / "failure_report.json")
            errors.extend(_validate_failure(attempts, report))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"failure payload unreadable: {exc}")
    return {"passed": not errors, "errors": errors}
