"""CPU-only P2-C scale and distribution audit for the PI-JWM v4 preflight.

The audit is deliberately read-only.  It consumes the eight-file P2-B bundle,
reports observed support, and emits a *candidate* formal-data configuration.
It never promotes a preflight to a training dataset and never fills missing
information fields.
"""

from __future__ import annotations

import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA_VERSION = "PIJWM-P2C-Scale-Distribution-Audit-v1"
FORMAL_CONFIG_SCHEMA_VERSION = "PIJWM-P2C-Formal-Data-Config-Candidate-v1"
REQUIRED_ARTIFACT_FILES = (
    "collector_config.json",
    "coverage_report.json",
    "frames.jsonl",
    "manifest.json",
    "replay_report.json",
    "status_flags.json",
    "validation_report.json",
    "vocabularies.json",
)
E1_FIELDS = (
    "channel_attenuation_mean_db",
    "channel_attenuation_std_db",
    "prev_active_flow_count",
    "prev_effective_rate_per_s",
    "prev_served_data",
)
_EPISODE_RE = re.compile(r"^natural-seed-(?P<seed>\d+)-(?P<arm>orthogonal|interference_reuse)(?:-|$)")
_SPLITS = ("train", "validation", "locked_test")


class AuditContractError(ValueError):
    """Input bundle violates the machine-readable P2-C audit contract."""


def _require_mapping(value: object, *, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditContractError(f"{what} must be a JSON object")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), what=path.name)
    except json.JSONDecodeError as exc:
        raise AuditContractError(f"invalid JSON: {path.name}: {exc}") from exc


def _load_frames(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditContractError(f"invalid frames.jsonl line {line_number}: {exc}") from exc
            rows.append(_require_mapping(row, what=f"frames.jsonl line {line_number}"))
    if not rows:
        raise AuditContractError("frames.jsonl is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_manifest(
    bundle_dir: str | Path,
    *,
    project_root: str | Path | None,
) -> dict[str, Any]:
    """Independently verify the P2-B artifact and source closure."""

    root = Path(bundle_dir)
    manifest = _load_json(root / "manifest.json")
    required_files = manifest.get("required_files")
    artifact_hashes = manifest.get("artifact_hashes")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(required_files, list):
        raise AuditContractError("manifest required_files must be an array")
    artifact_hashes = _require_mapping(artifact_hashes, what="manifest artifact_hashes")
    source_hashes = _require_mapping(source_hashes, what="manifest source_hashes")
    missing_required = sorted(
        str(name) for name in required_files if not isinstance(name, str) or not (root / name).is_file()
    )
    artifact_mismatches = sorted(
        str(name)
        for name, expected in artifact_hashes.items()
        if not (root / str(name)).is_file() or _sha256(root / str(name)) != str(expected)
    )
    nonportable_source_keys: list[str] = []
    missing_sources: list[str] = []
    source_hash_mismatches: list[str] = []
    closure_checked = project_root is not None
    if project_root is not None:
        project = Path(project_root)
        for raw_key, expected in source_hashes.items():
            key = str(raw_key)
            pure = PurePosixPath(key)
            if pure.is_absolute() or ".." in pure.parts or key.startswith(".worktrees/"):
                nonportable_source_keys.append(key)
                continue
            source = project.joinpath(*pure.parts)
            if not source.is_file():
                missing_sources.append(key)
            elif _sha256(source) != str(expected):
                source_hash_mismatches.append(key)
    passed = bool(closure_checked) and not any(
        (missing_required, artifact_mismatches, nonportable_source_keys, missing_sources, source_hash_mismatches)
    )
    return {
        "passed": passed,
        "source_closure_checked": closure_checked,
        "required_file_count": len(required_files),
        "artifact_hash_count": len(artifact_hashes),
        "source_hash_count": len(source_hashes),
        "missing_required_files": missing_required,
        "artifact_hash_mismatches": artifact_mismatches,
        "nonportable_source_keys": sorted(nonportable_source_keys),
        "missing_sources": sorted(missing_sources),
        "source_hash_mismatches": sorted(source_hash_mismatches),
    }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _episode_identity(trajectory_id: object) -> tuple[int, str]:
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        raise AuditContractError("trajectory_id must be a non-empty string")
    match = _EPISODE_RE.match(trajectory_id)
    if match is None:
        raise AuditContractError(f"natural trajectory has no stable seed/arm identity: {trajectory_id}")
    return int(match.group("seed")), match.group("arm")


def audit_e1_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and summarize E1 rows without adding or imputing fields."""

    field_counts: dict[str, Counter[str]] = {
        name: Counter(valid=0, invalid=0, missing_reason_null=0, missing_reason_nonnull=0)
        for name in E1_FIELDS
    }
    missing_reasons: dict[str, Counter[str]] = {name: Counter() for name in E1_FIELDS}
    observed_names: set[str] = set()
    row_widths: Counter[str] = Counter()
    legacy_placeholder_count = 0
    for row in rows:
        fields = _require_mapping(row.get("fields"), what="E1 fields")
        names = set(str(name) for name in fields)
        observed_names.update(names)
        row_widths[str(len(names))] += 1
        unknown = names.difference(E1_FIELDS)
        if unknown:
            legacy_placeholder_count += sum(
                1 for name in unknown if "slot" in name.lower() or "dim" in name.lower() or name.startswith("e")
            )
            raise AuditContractError(f"E1 contains non-contract fields: {sorted(unknown)}")
        if names != set(E1_FIELDS):
            raise AuditContractError(
                f"E1 must contain exactly five named fields, got {sorted(names)}"
            )
        for name in E1_FIELDS:
            payload = _require_mapping(fields[name], what=f"E1 field {name}")
            valid_mask = payload.get("valid_mask")
            missing_reason = payload.get("missing_reason")
            if not isinstance(valid_mask, bool):
                raise AuditContractError(f"E1 field {name} valid_mask must be bool")
            if valid_mask and missing_reason is not None:
                raise AuditContractError(f"E1 field {name}: valid value has missing_reason")
            if not valid_mask and (not isinstance(missing_reason, str) or not missing_reason.strip()):
                raise AuditContractError(f"E1 field {name}: invalid value lacks missing_reason")
            if not valid_mask and payload.get("value") is not None:
                raise AuditContractError(f"E1 field {name}: invalid value must be null, not imputed")
            field_counts[name]["valid" if valid_mask else "invalid"] += 1
            field_counts[name]["missing_reason_null" if missing_reason is None else "missing_reason_nonnull"] += 1
            if missing_reason is not None:
                missing_reasons[name][missing_reason] += 1
    return {
        "field_names": list(E1_FIELDS),
        "row_count": len(rows),
        "row_widths": {width: int(count) for width, count in sorted(row_widths.items())},
        "legacy_placeholder_field_count": legacy_placeholder_count,
        "per_field": {
            name: {
                **{key: int(value) for key, value in field_counts[name].items()},
                "missing_reasons": _sorted_counter(missing_reasons[name]),
            }
            for name in E1_FIELDS
        },
    }


def _longest_dag_path(tasks: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> int:
    task_ids = {str(task.get("task_id")) for task in tasks if task.get("task_id") is not None}
    children: dict[str, list[str]] = defaultdict(list)
    indegree: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source_task_id"))
        target = str(edge.get("target_task_id"))
        if source not in task_ids or target not in task_ids:
            continue
        children[source].append(target)
        indegree[target] += 1
    roots = sorted(task_ids.difference(indegree))
    depth = {node: 0 for node in roots}
    queue = list(roots)
    for node in queue:
        for child in sorted(children.get(node, ())):
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(depth) != len(task_ids):
        raise AuditContractError("task dependency graph contains a cycle")
    return max(depth.values(), default=0)


def _snapshot_coverage(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    lifecycles: Counter[str] = Counter()
    dag_depths: list[int] = []
    node_widths: Counter[str] = Counter()
    edge_widths: Counter[str] = Counter()
    dag_widths: Counter[str] = Counter()
    for frame in frames:
        snapshot = _require_mapping(frame.get("decision_snapshot"), what="decision_snapshot")
        nodes = snapshot.get("nodes", ())
        edges = snapshot.get("physical_edges", ())
        dag_edges = snapshot.get("dag_edges", ())
        tasks = snapshot.get("tasks", ())
        if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(dag_edges, list) or not isinstance(tasks, list):
            raise AuditContractError("snapshot graph/task collections must be arrays")
        node_widths[str(len(nodes))] += 1
        edge_widths[str(len(edges))] += 1
        dag_widths[str(len(dag_edges))] += 1
        for node in nodes:
            node_types[str(_require_mapping(node, what="node").get("node_type"))] += 1
        for edge in edges:
            edge_types[str(_require_mapping(edge, what="physical edge").get("edge_type"))] += 1
        for task in tasks:
            lifecycles[str(_require_mapping(task, what="task").get("lifecycle"))] += 1
        dag_depths.append(_longest_dag_path(tasks, dag_edges))
    return {
        "node_type_counts": _sorted_counter(node_types),
        "physical_edge_type_counts": _sorted_counter(edge_types),
        "task_lifecycle_counts": _sorted_counter(lifecycles),
        "node_widths": {key: int(value) for key, value in sorted(node_widths.items())},
        "physical_edge_widths": {key: int(value) for key, value in sorted(edge_widths.items())},
        "dag_edge_widths": {key: int(value) for key, value in sorted(dag_widths.items())},
        "dag_depth": {
            "observed": bool(dag_depths),
            "min": min(dag_depths) if dag_depths else None,
            "max": max(dag_depths) if dag_depths else None,
            "values": sorted(set(dag_depths)),
        },
    }


def _action_coverage(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_local = 0
    selected_wired = 0
    multi_flow_frames = 0
    flow_count = hop_count = rb_count = 0
    rb_reuse_frames = 0
    same_transmitter_conflict_frames = 0
    decision_declines: Counter[str] = Counter()
    for frame in frames:
        action = _require_mapping(frame.get("action"), what="action")
        decisions = action.get("decisions", [])
        flows = action.get("flows", [])
        hops = action.get("hops", [])
        allocations = action.get("rb_allocations", [])
        if not all(isinstance(value, list) for value in (decisions, flows, hops, allocations)):
            raise AuditContractError("action collections must be arrays")
        flow_count += len(flows)
        hop_count += len(hops)
        rb_count += len(allocations)
        if len(flows) > 1:
            multi_flow_frames += 1
        hop_sources = {
            str(hop.get("hop_id")): str(hop.get("source_id"))
            for hop in hops
            if isinstance(hop, Mapping)
        }
        rb_sources: dict[int, set[str]] = defaultdict(set)
        rb_same_source: set[tuple[str, int]] = set()
        same_source = False
        for allocation in allocations:
            if not isinstance(allocation, Mapping):
                raise AuditContractError("RB allocation must be an object")
            rb = int(allocation.get("rb_index"))
            source = hop_sources.get(str(allocation.get("hop_id")))
            if source is None:
                raise AuditContractError("RB allocation references unknown hop")
            rb_sources[rb].add(source)
            key = (source, rb)
            if key in rb_same_source:
                same_source = True
            rb_same_source.add(key)
        if any(len(sources) > 1 for sources in rb_sources.values()):
            rb_reuse_frames += 1
        if same_source:
            same_transmitter_conflict_frames += 1
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise AuditContractError("decision must be an object")
            if decision.get("selected") is True and decision.get("flow_id") is None and decision.get("hop_id") is None:
                selected_local += 1
            if decision.get("selected") is True:
                flow_id = decision.get("flow_id")
                hop_id = decision.get("hop_id")
                if flow_id is not None and hop_id is not None:
                    hop = next((item for item in hops if item.get("hop_id") == hop_id), None)
                    if isinstance(hop, Mapping) and hop.get("transport") == "wired":
                        selected_wired += 1
            if decision.get("selected") is False:
                decision_declines[str(decision.get("reason"))] += 1
    return {
        "selected_local_decisions": selected_local,
        "selected_wired_hops": selected_wired,
        "multi_flow_frame_count": multi_flow_frames,
        "flow_count": flow_count,
        "hop_count": hop_count,
        "rb_allocation_count": rb_count,
        "rb_reuse_frame_count": rb_reuse_frames,
        "same_transmitter_rb_conflict_frame_count": same_transmitter_conflict_frames,
        "decision_decline_reasons": _sorted_counter(decision_declines),
    }


def _transfer_coverage(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outage = 0
    transfer_rows = 0
    failed_tasks = 0
    for frame in frames:
        transfers = frame.get("transfer_rows", [])
        if not isinstance(transfers, list):
            raise AuditContractError("transfer_rows must be an array")
        transfer_rows += len(transfers)
        outage += sum(1 for row in transfers if isinstance(row, Mapping) and row.get("outage") is True)
        snapshot = _require_mapping(frame.get("outcome_snapshot"), what="outcome_snapshot")
        tasks = snapshot.get("tasks", [])
        if isinstance(tasks, list):
            failed_tasks += sum(1 for task in tasks if isinstance(task, Mapping) and task.get("lifecycle") == "failed")
    return {
        "transfer_row_count": transfer_rows,
        "outage_row_count": outage,
        "failed_task_snapshot_count": failed_tasks,
        "outage_observed": bool(outage),
        "failure_observed": bool(failed_tasks),
    }


def _runtime_guard_evidence(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cpu_rule_versions: Counter[str] = Counter()
    lifecycle_guard_frames = 0
    dependency_declines = 0
    for frame in frames:
        cpu_rows = frame.get("cpu_rows", [])
        if not isinstance(cpu_rows, list):
            raise AuditContractError("cpu_rows must be an array")
        for row in cpu_rows:
            cpu_rule_versions[str(_require_mapping(row, what="CPU row").get("rule_version"))] += 1
        temporal_trace = frame.get("temporal_trace", [])
        if not isinstance(temporal_trace, list) or any(not isinstance(item, str) for item in temporal_trace):
            raise AuditContractError("temporal_trace must be an array of strings")
        if "airfogsim_lifecycle_alias_guard_installed" in temporal_trace:
            lifecycle_guard_frames += 1
        action = _require_mapping(frame.get("action"), what="action")
        decisions = action.get("decisions", [])
        if not isinstance(decisions, list):
            raise AuditContractError("action decisions must be an array")
        dependency_declines += sum(
            1
            for decision in decisions
            if isinstance(decision, Mapping)
            and decision.get("selected") is False
            and decision.get("reason") in {"dependency_not_satisfied", "dependency_failed"}
        )
    return {
        "cpu_rule_versions": _sorted_counter(cpu_rule_versions),
        "lifecycle_alias_guard_frame_count": lifecycle_guard_frames,
        "dependency_gate_decline_count": dependency_declines,
        "all_frames_have_lifecycle_alias_guard": lifecycle_guard_frames == len(frames),
    }


def _candidate_split_errors(seed_split: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    sets: dict[str, set[int]] = {}
    for split in _SPLITS:
        values = seed_split.get(split)
        if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            errors.append(f"{split}_seed_list_invalid")
            sets[split] = set()
        else:
            sets[split] = set(values)
    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            if sets[left].intersection(sets[right]):
                errors.append("seed_split_overlap")
    if not all(sets.values()):
        errors.append("seed_split_not_frozen")
    return sorted(set(errors))


def validate_candidate_formal_data_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if config.get("formal_data_approved") is not False:
        errors.append("formal_data_approval_must_remain_false")
    errors.extend(_candidate_split_errors(_require_mapping(config.get("seed_split"), what="seed_split")))
    if config.get("target_scale", {}).get("status") != "candidate_target_definition":
        errors.append("target_scale_status_invalid")
    return tuple(sorted(set(errors)))


def build_candidate_formal_data_config(report: Mapping[str, Any]) -> dict[str, Any]:
    natural = _require_mapping(_require_mapping(report.get("observed_facts"), what="observed_facts").get("natural"), what="natural facts")
    return {
        "schema_version": FORMAL_CONFIG_SCHEMA_VERSION,
        "status": "candidate_target_definition",
        "formal_data_approved": False,
        "source_scope": "P2-B CPU-only nontraining preflight; not a formal dataset",
        "target_scale": {
            "status": "candidate_target_definition",
            "natural_episode_count": None,
            "steps_per_episode": None,
            "basis": "scenario matrix and quota review required; no number inferred from six preflight episodes",
        },
        "scenario_factors": {
            "task_load": {"status": "not_frozen", "observed": None},
            "network_density": {"status": "not_frozen", "observed": None},
            "node_presence_and_mobility": {"status": "not_frozen", "observed": None},
            "failure_and_outage_rates": {"status": "not_frozen", "observed": None},
        },
        "seed_split": {
            "status": "not_frozen",
            "train": [],
            "validation": [],
            "locked_test": [],
            "preflight_observed_seeds": sorted(int(value) for value in natural.get("seed_values", [])),
        },
        "rejection_policy": {
            "status": "candidate_definition",
            "max_quarantined_trajectories": 0,
            "max_manifest_mismatches": 0,
            "action_rejection_rate_upper_bound": 0.0,
            "explicit_action_rejection_source_required": True,
        },
        "replay_policy": {
            "fresh_environment_per_reference_and_replay": True,
            "exact_mismatch_upper_bound": 0,
            "numeric_tolerance_must_be_recorded": True,
        },
        "formal_output_directory": {
            "status": "candidate_target_definition",
            "relative_path": "代码/artifacts/formal_data/pi_jwm_v4_formal_candidate_v1",
            "must_not_reuse_v3_trajectories": True,
        },
        "field_contract": {
            "e1_fields": list(E1_FIELDS),
            "legacy_13_or_18_slot_fill": "forbidden",
            "invalid_value_requires_missing_reason": True,
            "invalid_value_must_be_null": True,
        },
    }


def audit_bundle(
    bundle_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir)
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (root / name).is_file()]
    if missing:
        raise AuditContractError(f"bundle missing required files: {missing}")
    config = _load_json(root / "collector_config.json")
    coverage = _load_json(root / "coverage_report.json")
    validation = _load_json(root / "validation_report.json")
    status_flags = _load_json(root / "status_flags.json")
    manifest = _load_json(root / "manifest.json")
    replay = _load_json(root / "replay_report.json")
    frames = _load_frames(root / "frames.jsonl")
    manifest_verification = verify_input_manifest(root, project_root=project_root)
    if any(not isinstance(value, bool) for value in status_flags.values()):
        raise AuditContractError("every status flag must be boolean")

    natural = [frame for frame in frames if frame.get("fixture") is False]
    fixtures = [frame for frame in frames if frame.get("fixture") is True]
    if len(natural) + len(fixtures) != len(frames):
        raise AuditContractError("every frame must carry a boolean fixture flag")
    if any(not isinstance(frame.get("training_eligible"), bool) for frame in frames):
        raise AuditContractError("every frame must carry boolean training_eligible")

    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seed_arm_pairs: set[tuple[int, str]] = set()
    for frame in natural:
        trajectory_id = str(frame.get("trajectory_id", ""))
        seed, arm = _episode_identity(trajectory_id)
        seed_arm_pairs.add((seed, arm))
        by_trajectory[trajectory_id].append(frame)
        if not isinstance(frame.get("frame_index"), int) or isinstance(frame.get("frame_index"), bool):
            raise AuditContractError("frame_index must be an integer")
    expected_seeds = {int(value) for value in config.get("seeds", [])}
    expected_arms = {str(value) for value in config.get("natural_arms", [])}
    expected_pairs = {(seed, arm) for seed in expected_seeds for arm in expected_arms}
    blocking: list[str] = []
    if seed_arm_pairs != expected_pairs:
        blocking.append("natural_seed_arm_coverage_incomplete")
    configured_episode_count = config.get("natural_episode_count")
    if configured_episode_count != len(by_trajectory):
        blocking.append("natural_episode_count_mismatch")
    expected_steps = config.get("steps")
    episode_steps: list[int] = []
    for trajectory_id, episode_frames in sorted(by_trajectory.items()):
        indices = sorted(int(frame["frame_index"]) for frame in episode_frames)
        episode_steps.append(len(indices))
        if expected_steps is not None and len(indices) != int(expected_steps):
            blocking.append("episode_step_count_mismatch")
        if indices != list(range(len(indices))):
            blocking.append("episode_frame_sequence_invalid")
    if float(config.get("traffic_interval", 0)) != float(config.get("simulation_interval", 0)):
        blocking.append("traffic_simulation_interval_ratio_not_one")

    e1_rows = [row for frame in frames for row in frame.get("e1_rows", [])]
    e1_report = audit_e1_rows(e1_rows)

    natural_facts = {
        "episode_count": len(by_trajectory),
        "frame_count": len(natural),
        "episode_steps": sorted(set(episode_steps)),
        "seed_values": sorted({seed for seed, _ in seed_arm_pairs}),
        "seed_arm_pairs": [list(pair) for pair in sorted(seed_arm_pairs)],
        "all_training_eligible_false": all(frame["training_eligible"] is False for frame in natural),
        "all_quarantined_false": all(frame.get("quarantined") is False for frame in natural),
    }
    fixture_facts = {
        "frame_count": len(fixtures),
        "coverage_fixture_count": len(_require_mapping(coverage.get("fixtures"), what="coverage fixtures")),
        "training_eligible_true_count": sum(frame["training_eligible"] is True for frame in fixtures),
        "quarantined_frame_count": sum(frame.get("quarantined") is True for frame in fixtures),
    }
    if fixture_facts["training_eligible_true_count"]:
        blocking.append("fixture_training_eligible_true")
    if fixture_facts["quarantined_frame_count"]:
        blocking.append("fixture_quarantine_present")

    validation_errors = validation.get("errors", [])
    if not isinstance(validation_errors, list):
        raise AuditContractError("validation_report.errors must be an array")
    if validation_errors:
        blocking.append("validation_report_not_empty")
    explicit_rejection_count = validation.get("action_rejection_count")
    rejection_observed = isinstance(explicit_rejection_count, int) and not isinstance(explicit_rejection_count, bool)
    if not rejection_observed:
        blocking.append("action_rejection_rate_not_observed")
    if "seed_split" not in config:
        blocking.append("formal_split_not_frozen")
    if not manifest_verification["source_closure_checked"]:
        blocking.append("input_manifest_source_closure_not_checked")
    elif not manifest_verification["passed"]:
        blocking.append("input_manifest_source_closure_failed")
    blocking.append("formal_scale_not_frozen")
    blocking.append("scenario_matrix_not_frozen")

    replay_episodes = replay.get("episodes")
    if not isinstance(replay_episodes, Mapping):
        raise AuditContractError("replay_report.episodes must be a JSON object")
    replay_passed_count = sum(
        row.get("passed") is True
        for row in replay_episodes.values()
        if isinstance(row, Mapping)
    )
    if replay.get("passed") is not True or replay_passed_count != len(by_trajectory):
        blocking.append("replay_gate_failed")

    observed = {
        "natural": natural_facts,
        "fixture": fixture_facts,
        "snapshot_coverage": _snapshot_coverage(natural),
        "action_coverage": _action_coverage(natural),
        "transfer_coverage": _transfer_coverage(natural),
        "runtime_guards": _runtime_guard_evidence(natural),
        "replay": {
            "episode_count": len(replay_episodes),
            "passed_episode_count": replay_passed_count,
            "passed": replay.get("passed") is True,
            "fresh_environment_per_reference_and_replay": replay.get(
                "fresh_environment_per_reference_and_replay"
            )
            is True,
        },
        "status_flags": {str(key): bool(value) for key, value in status_flags.items()},
        "validation": {
            "passed": validation.get("passed") is True,
            "errors": validation_errors,
            "rejection_count": explicit_rejection_count if rejection_observed else None,
        },
        "manifest": {
            "required_file_count": len(manifest.get("required_files", [])) if isinstance(manifest.get("required_files"), list) else None,
            "artifact_hash_count": len(manifest.get("artifact_hashes", {})) if isinstance(manifest.get("artifact_hashes"), Mapping) else None,
            "source_hash_count": len(manifest.get("source_hashes", {})) if isinstance(manifest.get("source_hashes"), Mapping) else None,
            "independent_verification": manifest_verification,
        },
    }
    natural_episode_reports = coverage.get("natural_episodes")
    if not isinstance(natural_episode_reports, list):
        raise AuditContractError("coverage natural_episodes must be a JSON array")
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope": "CPU-only read-only audit of P2-B canonical preflight; no training data approval",
        "observed_facts": observed,
        "e1_field_validity": e1_report,
        "coverage_evidence": {
            "natural_and_fixture_reports_separate": coverage.get("natural_and_fixture_reports_separate") is True,
            "fixture_names": sorted(_require_mapping(coverage.get("fixtures"), what="coverage fixtures")),
            "natural_report_count": len(natural_episode_reports),
        },
        "rejection_quarantine": {
            "action_rejection_count": explicit_rejection_count if rejection_observed else None,
            "action_rejection_rate": None,
            "quarantined_frame_count": sum(frame.get("quarantined") is True for frame in frames),
            "upper_bound_candidate": 0.0,
            "status": "observed_quarantine_only_action_rejection_source_missing" if not rejection_observed else "observed",
        },
        "blocking_reasons": sorted(set(blocking)),
        "candidate_formal_data_config": None,
    }
    report["candidate_formal_data_config"] = build_candidate_formal_data_config(report)
    report["audit_status"] = "blocked" if report["blocking_reasons"] else "passed_for_candidate_review"
    return report
