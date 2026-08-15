"""Generate leakage-safe actual-rollout labels for PI-JWM v11 candidates.

The frozen PI-JWM world model supplies candidate-specific rollout predictions;
future truth is used only to form training/evaluation labels.  Matched-test
labels cannot be generated until a validation-selected configuration manifest
has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.evaluation.candidate_selection import sample_active_sse
from pi_jwm.v11_candidates import (
    RB_TOTAL_DIM,
    build_support_constrained_candidates,
    infer_stage_from_observable_actions,
    positive_value_quantiles,
)
from pi_jwm.v11_labeling import (
    build_candidate_feature_batch,
    build_observable_state_context,
    compute_rollout_outcome_metrics,
    save_candidate_interaction_cache,
    save_candidate_label_cache,
)
from pi_jwm.v11_interactions import (
    build_candidate_interaction_tokens,
    pool_candidate_interactions,
)
from pi_jwm.v11_crossfit import (
    build_crossfit_protocol_manifest,
    resolve_crossfit_execution,
    validate_crossfit_label_indices,
)
from pi_jwm.v11_rollout_value_calibrator import freeze_module
from pi_jwm.v11_selector import (
    DEFAULT_SELECTOR_SEEDS,
    CandidateBatch,
    CandidateOutcome,
    SelectorProtocol,
    audit_candidate_library,
    audit_selector_protocol,
    build_selector_split,
    canonical_sha256,
)

from compare_v11_base_policy_candidates import _fit_models
from compare_v11_rollout_reward_template_selector import make_split_payload
from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from diagnose_v11_rb_total_latent_identifiability import (
    collect_rollout_edge_context,
    make_targets,
    rows_from_context,
)
from diagnose_v11_scheduler_ranked_allocation import resolve_torch_device
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset
from run_v11_rb_total_value_head import (
    collect_edge_gradient_improvement,
    load_context_limited,
    make_critical_examples,
)


DEFAULT_OUTPUT_DIR = (
    CODE_ROOT / "artifacts" / "reports" / "pi_jwm_v11_selector_finalization_20260719" / "label_cache"
)
ALLOWED_SPLITS = (
    "train",
    "calibration",
    "validation",
    "background",
    "matched_test",
    "external_holdout",
)


def validate_requested_splits(
    splits: Sequence[str], frozen_manifest: Mapping[str, Any] | None
) -> None:
    unknown = sorted(set(str(value) for value in splits) - set(ALLOWED_SPLITS))
    if unknown:
        raise ValueError(f"unknown selector splits: {unknown}")
    locked_splits = sorted(set(splits) & {"matched_test", "external_holdout"})
    if locked_splits:
        if not frozen_manifest or not bool(frozen_manifest.get("configuration_frozen")):
            raise PermissionError(f"{locked_splits} require a frozen validation-selected manifest")
        digest = str(frozen_manifest.get("configuration_digest", ""))
        if len(digest) != 64:
            raise PermissionError("frozen manifest must contain a 64-character configuration digest")


def validate_helper_execution_args(
    helper_protocol: str,
    crossfit_fold: int | None,
    splits: Sequence[str],
) -> None:
    """Validate invocation-specific helper execution without touching data."""
    protocol = str(helper_protocol)
    requested = tuple(str(value) for value in splits)
    if protocol == "in_sample":
        if crossfit_fold is not None:
            raise ValueError("in-sample helper protocol does not accept a crossfit fold")
        return
    if protocol != "seed_crossfit_5fold":
        raise ValueError(f"unknown helper protocol: {protocol}")
    if "train" in requested:
        if requested != ("train",) or crossfit_fold is None:
            raise ValueError(
                "crossfit train generation requires one fold and only the train split"
            )
    elif crossfit_fold is not None:
        raise ValueError("crossfit evaluation must not specify a fold")


def limit_indices_seed_balanced(
    indices: np.ndarray,
    sample_seed: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Round-robin a smoke subset so early seeds cannot dominate it."""
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    if values.size == 0 or int(limit) <= 0 or values.size <= int(limit):
        return values.copy()
    buckets = {int(seed): values[seeds[values] == int(seed)].tolist() for seed in sorted(np.unique(seeds[values]))}
    selected = []
    offset = 0
    while len(selected) < int(limit):
        changed = False
        for seed in sorted(buckets):
            if offset < len(buckets[seed]) and len(selected) < int(limit):
                selected.append(int(buckets[seed][offset]))
                changed = True
        if not changed:
            break
        offset += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def scatter_edge_values(
    baseline: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Scatter edge-step predictions while retaining baseline elsewhere."""
    dense = np.asarray(baseline, dtype=np.float32).copy()
    coords = np.asarray(coordinates, dtype=np.int64)
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != flat_values.shape[0]:
        raise ValueError("coordinates must be [row,3] and match values")
    if coords.size:
        if (
            np.any(coords < 0)
            or np.any(coords[:, 0] >= dense.shape[0])
            or np.any(coords[:, 1] >= dense.shape[1])
            or np.any(coords[:, 2] >= dense.shape[2])
        ):
            raise ValueError("edge coordinates outside dense tensor")
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = flat_values
    return dense


def build_reproduction_command(args: argparse.Namespace) -> str:
    """Serialize every label-affecting CLI argument into a runnable command."""

    def path_arg(value: Path) -> str:
        return Path(value).as_posix()

    tokens = [
        "python",
        "code/scripts/run_v11_selector_candidate_labels.py",
        "--world-experiment-dir",
        path_arg(args.world_experiment_dir),
        "--world-checkpoint",
        path_arg(args.world_checkpoint),
        "--policy-checkpoint",
        path_arg(args.policy_checkpoint),
        "--output-dir",
        path_arg(args.output_dir),
        "--splits",
        *(str(value) for value in args.splits),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--helper-train-limit",
        str(args.helper_train_limit),
        "--split-sample-limit",
        str(args.split_sample_limit),
        "--policy-threshold",
        str(args.policy_threshold),
        "--value-scale",
        str(args.value_scale),
        "--new-policy-threshold",
        str(args.new_policy_threshold),
        "--new-value-scale",
        str(args.new_value_scale),
        "--gate-feature",
        str(args.gate_feature),
        "--gate-threshold",
        str(args.gate_threshold),
        "--value-codebook-size",
        str(args.value_codebook_size),
        "--min-effective-rb-total",
        str(args.min_effective_rb_total),
        "--activity-threshold",
        str(args.activity_threshold),
        "--rf-trees",
        str(args.rf_trees),
        "--seed",
        str(args.seed),
        "--stats-chunk-size",
        str(args.stats_chunk_size),
        "--cache-schema-version",
        str(getattr(args, "cache_schema_version", 5)),
        "--helper-protocol",
        str(getattr(args, "helper_protocol", "in_sample")),
    ]
    crossfit_fold = getattr(args, "crossfit_fold", None)
    if crossfit_fold is not None:
        tokens.extend(["--crossfit-fold", str(int(crossfit_fold))])
    if args.frozen_config_manifest is not None:
        tokens.extend(
            ["--frozen-config-manifest", path_arg(args.frozen_config_manifest)]
        )
    return shlex.join(tokens)


def _canonical_configuration(args: argparse.Namespace) -> dict[str, Any]:
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    configuration = {
        "framework": "PI-JWM",
        "candidate_protocol": "support_constrained_edge_step_repair_v2",
        "world_checkpoint": args.world_checkpoint.name,
        "world_checkpoint_sha256": sha256(args.world_checkpoint.resolve()),
        "policy_checkpoint": args.policy_checkpoint.name,
        "policy_checkpoint_sha256": sha256(args.policy_checkpoint.resolve()),
        "source_git_sha": git_sha,
        "selector_seed_spec": {name: list(values) for name, values in DEFAULT_SELECTOR_SEEDS.items()},
        "steps": [1, 2],
        "candidate_count": 32,
        "k": [8, 16, 32],
        "magnitude": ["value_head", "train_positive_q50", "train_positive_q75"],
        "patterns": ["persistent", "decayed"],
        "min_effective_rb_total": float(args.min_effective_rb_total),
        "activity_threshold": float(args.activity_threshold),
        "policy_operating_point": {
            "old_threshold": float(args.policy_threshold),
            "old_value_scale": float(args.value_scale),
            "new_threshold": float(args.new_policy_threshold),
            "new_value_scale": float(args.new_value_scale),
            "gate_feature": str(args.gate_feature),
            "gate_threshold": float(args.gate_threshold),
        },
        "helper_model": {"type": "random_forest", "trees": int(args.rf_trees), "seed": int(args.seed)},
        "helper_train_limit": int(args.helper_train_limit),
        "split_sample_limit": int(args.split_sample_limit),
        "cache_schema_version": int(args.cache_schema_version),
    }
    if str(getattr(args, "helper_protocol", "in_sample")) == "seed_crossfit_5fold":
        crossfit = build_crossfit_protocol_manifest(configuration)
        configuration["helper_generation_protocol"] = crossfit[
            "crossfit_protocol_payload"
        ]
    return configuration


def _load_frozen_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fit_helper_models(
    args: argparse.Namespace,
    train_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict,
    world_model,
    policy_model,
    action_scale: np.ndarray,
    value_vocab,
    world_config: dict,
    device: torch.device,
):
    train_base, train_dataset = make_adaptive_dataset(
        args,
        arrays,
        train_indices,
        stats,
        policy_model,
        action_scale,
        value_vocab,
        device,
        train_indices,
    )
    train_actions, train_truth = collect_raw_actions(train_dataset, stats)
    examples = make_critical_examples(train_actions, train_truth, steps=(1, 2))
    edge_improvement = collect_edge_gradient_improvement(
        world_model,
        train_base,
        train_actions,
        train_truth,
        stats,
        world_config,
        device,
        args.batch_size,
    )
    train_score, train_value = make_targets(examples, train_truth, edge_improvement)
    context = collect_rollout_edge_context(
        world_model, train_base, train_actions, stats, device, args.batch_size
    )
    features = rows_from_context(context, examples.coordinates)
    score_model, value_model = _fit_models(
        args, features, train_score, train_value, examples.baseline_values
    )
    quantiles = positive_value_quantiles(
        train_truth[..., RB_TOTAL_DIM],
        quantiles=(0.5, 0.75),
        min_value=float(args.min_effective_rb_total),
    )
    return score_model, value_model, train_score, quantiles


def _make_label_split(
    args: argparse.Namespace,
    split_name: str,
    split_indices: np.ndarray,
    train_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict,
    world_model,
    policy_model,
    action_scale: np.ndarray,
    value_vocab,
    world_config: dict,
    device: torch.device,
    score_model,
    value_model,
    train_score: np.ndarray,
    quantiles: Mapping[float, float],
    configuration_digest: str,
    protocol_metadata: Mapping[str, Any],
    sample_fold_id: np.ndarray | None = None,
) -> dict[str, Any]:
    payload = make_split_payload(
        args,
        split_name,
        arrays,
        split_indices,
        train_indices,
        stats,
        policy_model,
        action_scale,
        value_vocab,
        world_model,
        world_config,
        device,
        score_model,
        value_model,
        (1, 2),
    )
    baseline_rb = np.asarray(payload.baseline_actions[..., RB_TOTAL_DIM], dtype=np.float32)
    dense_score = scatter_edge_values(
        np.zeros_like(baseline_rb), payload.coordinates, payload.pred_score
    )
    dense_value = scatter_edge_values(
        baseline_rb, payload.coordinates, payload.pred_value
    )
    valid_edge = np.asarray(arrays["valid_edge_node"], dtype=bool)
    valid = np.broadcast_to(valid_edge[None, None, :], baseline_rb.shape).copy()
    support = valid & (
        (payload.baseline_actions[..., 1] > 0.0)
        | (payload.baseline_actions[..., RB_TOTAL_DIM] >= float(args.min_effective_rb_total))
    )
    stages = infer_stage_from_observable_actions(payload.baseline_actions)
    library = build_support_constrained_candidates(
        baseline_actions=payload.baseline_actions,
        selection_score=dense_score,
        value_head=dense_value,
        train_positive_quantiles=quantiles,
        support_mask=support,
        valid_element_mask=valid,
        stages=stages,
    )
    predictions_by_candidate = []
    sample_sse = []
    sample_count = None
    link_sse = []
    link_count = None
    activity_confusion = {name: [] for name in ("activity_tp", "activity_fp", "activity_fn", "activity_tn")}
    metric_rows = []
    baseline_rmse = float("nan")
    for candidate_index, (name, family) in enumerate(
        zip(library.candidate_names, library.action_families)
    ):
        predictions = evaluate_raw_actions(
            library.actions[:, candidate_index],
            payload.base_dataset,
            stats,
            world_model,
            world_config,
            device,
            args.batch_size,
        )
        row = active_rate_row(name, split_name, predictions, baseline_rmse)
        if candidate_index == 0:
            baseline_rmse = float(row["active_rate_rmse"])
            row["improvement_vs_baseline"] = 0.0
        sse, count = sample_active_sse(predictions)
        outcome_metrics = compute_rollout_outcome_metrics(
            predictions, activity_threshold=args.activity_threshold
        )
        if sample_count is None:
            sample_count = count
        elif not np.array_equal(sample_count, count):
            raise ValueError("candidate rollouts changed active target counts")
        if link_count is None:
            link_count = outcome_metrics["link_count"]
        elif not np.array_equal(link_count, outcome_metrics["link_count"]):
            raise ValueError("candidate rollouts changed link target counts")
        row.update(
            {
                "candidate_index": candidate_index,
                "action_family": family,
                "candidate_available_ratio": float(np.mean(library.candidate_mask[:, candidate_index])),
                "action_applied_ratio": float(np.mean(library.action_applied[:, candidate_index])),
                "result_kind": "diagnostic_only",
            }
        )
        metric_rows.append(row)
        predictions_by_candidate.append(predictions)
        sample_sse.append(sse)
        link_sse.append(outcome_metrics["link_sse"])
        for field in activity_confusion:
            activity_confusion[field].append(outcome_metrics[field])
    active_sse = np.stack(sample_sse, axis=1).astype(np.float32)
    default_index = library.candidate_names.index("ranked_allocation_baseline")
    features, feature_names = build_candidate_feature_batch(
        library.actions,
        predictions_by_candidate,
        library.action_families,
        default_index=default_index,
        current_link_features=arrays["x_link"][split_indices, -1],
        current_link_feature_names=tuple(str(value) for value in arrays["link_features"]),
    )
    state_context, state_context_names = build_observable_state_context(
        arrays["x_node"][split_indices],
        arrays["x_link"][split_indices],
        arrays["x_task"][split_indices],
        arrays["edge_a_hist"][split_indices],
        valid_edge_mask=arrays["valid_edge_node"],
        node_feature_names=tuple(str(value) for value in arrays["node_features"]),
        link_feature_names=tuple(str(value) for value in arrays["link_features"]),
        task_feature_names=tuple(str(value) for value in arrays["task_features"]),
        action_feature_names=tuple(str(value) for value in arrays["edge_action_features"]),
    )
    context = np.concatenate([features[:, default_index], state_context], axis=1).astype(np.float32)
    context_feature_names = tuple(f"default_{name}" for name in feature_names) + state_context_names
    protocol_audit = audit_selector_protocol(
        feature_names,
        {
            "train": set(DEFAULT_SELECTOR_SEEDS["train"]),
            split_name: set(int(seed) for seed in np.unique(arrays["sample_seed"][split_indices])),
        }
        if split_name != "train"
        else {"train": set(DEFAULT_SELECTOR_SEEDS["train"])},
    )
    if not protocol_audit["passed"]:
        raise ValueError(f"selector protocol audit failed: {protocol_audit}")
    batch = CandidateBatch(
        context=context,
        candidate_features=features,
        candidate_mask=library.candidate_mask,
        stage=stages,
        feature_names=feature_names,
        candidate_names=library.candidate_names,
        context_feature_names=context_feature_names,
    )
    outcome = CandidateOutcome(
        active_sse=active_sse,
        active_count=np.asarray(sample_count, dtype=np.int64),
        link_sse=np.stack(link_sse, axis=1).astype(np.float32),
        link_count=np.asarray(link_count, dtype=np.int64),
        activity_tp=np.stack(activity_confusion["activity_tp"], axis=1),
        activity_fp=np.stack(activity_confusion["activity_fp"], axis=1),
        activity_fn=np.stack(activity_confusion["activity_fn"], axis=1),
        activity_tn=np.stack(activity_confusion["activity_tn"], axis=1),
        action_applied=library.action_applied,
        action_applicable=library.applicability_mask,
        default_index=default_index,
        result_kind="diagnostic_only",
    )
    cache_path = args.output_dir / f"candidate_labels_{split_name}.npz"
    if int(args.cache_schema_version) == 6:
        action_feature_names = tuple(
            str(value) for value in arrays["edge_action_features"]
        )
        interactions = pool_candidate_interactions(
            build_candidate_interaction_tokens(
                library.actions,
                predictions_by_candidate,
                current_link_features=arrays["x_link"][split_indices, -1],
                action_feature_names=action_feature_names,
                current_link_feature_names=tuple(
                    str(value) for value in arrays["link_features"]
                ),
                default_index=default_index,
            ),
            action_feature_names=action_feature_names,
        )
        manifest = save_candidate_interaction_cache(
            cache_path,
            split_name=split_name,
            sample_ids=split_indices,
            sample_seed=arrays["sample_seed"][split_indices],
            batch=batch,
            outcome=outcome,
            interactions=interactions,
            action_feature_names=action_feature_names,
            configuration_digest=configuration_digest,
            protocol_metadata=dict(protocol_metadata),
            sample_fold_id=sample_fold_id,
        )
    else:
        manifest = save_candidate_label_cache(
            cache_path,
            split_name=split_name,
            sample_ids=split_indices,
            sample_seed=arrays["sample_seed"][split_indices],
            batch=batch,
            outcome=outcome,
            configuration_digest=configuration_digest,
            protocol_metadata=dict(protocol_metadata),
            sample_fold_id=sample_fold_id,
        )
    gate = audit_candidate_library(
        active_sse=active_sse,
        active_count=sample_count,
        action_applied=library.action_applied,
        candidate_mask=library.candidate_mask,
        applicability_mask=library.applicability_mask,
        identity_index=0,
    )
    _write_csv(args.output_dir / f"candidate_metrics_{split_name}.csv", metric_rows)
    split_summary = {
        "split_name": split_name,
        "sample_limit": int(args.split_sample_limit),
        "num_samples": int(len(split_indices)),
        "seed_values": sorted(int(value) for value in np.unique(arrays["sample_seed"][split_indices])),
        "candidate_gate": gate,
        "protocol_audit": protocol_audit,
        "cache_manifest": manifest,
        "interaction_audit": manifest.get("interaction"),
        "train_positive_quantiles": {str(key): float(value) for key, value in quantiles.items()},
    }
    (args.output_dir / f"summary_{split_name}.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-experiment-dir",
        type=Path,
        default=CODE_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline",
    )
    parser.add_argument(
        "--world-checkpoint",
        type=Path,
        default=CODE_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=CODE_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", choices=ALLOWED_SPLITS, default=["validation"])
    parser.add_argument("--frozen-config-manifest", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--helper-train-limit", type=int, default=256)
    parser.add_argument("--split-sample-limit", type=int, default=64)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument(
        "--gate-feature",
        choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"),
        default="step_rb_cpu_total",
    )
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--min-effective-rb-total", type=float, default=1.0)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--rf-trees", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--cache-schema-version", type=int, choices=(5, 6), default=5)
    parser.add_argument(
        "--helper-protocol",
        choices=("in_sample", "seed_crossfit_5fold"),
        default="in_sample",
    )
    parser.add_argument("--crossfit-fold", type=int, choices=range(5))
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    frozen_manifest = _load_frozen_manifest(args.frozen_config_manifest)
    validate_requested_splits(tuple(args.splits), frozen_manifest)
    helper_protocol = str(getattr(args, "helper_protocol", "in_sample"))
    crossfit_fold = getattr(args, "crossfit_fold", None)
    validate_helper_execution_args(helper_protocol, crossfit_fold, tuple(args.splits))
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = _canonical_configuration(args)
    configuration_digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if set(args.splits) & {"matched_test", "external_holdout"} and str(
        frozen_manifest["configuration_digest"]
    ) != configuration_digest:
        raise ValueError("frozen configuration digest does not match candidate-label configuration")

    device = resolve_torch_device(args.device)
    # Reuse the established loader with full streaming train statistics while
    # discarding its legacy split indices in favor of the selector protocol.
    args.max_train_samples = 1
    args.max_val_samples = 1
    args.max_test_samples = 1
    args.limit_after_stats = True
    args.streaming_stats = True
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, _ = load_context_limited(
        args, device
    )
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    split_indices = build_selector_split(arrays["sample_seed"])
    protocol = SelectorProtocol(arrays["sample_seed"])
    protocol.freeze_configuration(configuration)
    crossfit_manifest = None
    if helper_protocol == "seed_crossfit_5fold":
        payload = configuration["helper_generation_protocol"]
        crossfit_manifest = {
            "crossfit_protocol_payload": payload,
            "crossfit_protocol_digest": canonical_sha256(payload),
        }
        execution = resolve_crossfit_execution(
            arrays["sample_seed"], tuple(args.splits), crossfit_fold
        )
        raw_train_indices = execution.helper_train_indices
        requested_indices = execution.label_indices
        validate_crossfit_label_indices(
            execution,
            {name: protocol.indices(name) for name in requested_indices},
            arrays["sample_seed"],
        )
        helper_execution = {
            "mode": execution.mode,
            "fold_id": execution.fold_id,
            "held_out_seeds": list(execution.held_out_seeds),
            "helper_train_seeds": list(execution.helper_train_seeds),
        }
        (args.output_dir / "crossfit_protocol.json").write_text(
            json.dumps(crossfit_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        raw_train_indices = split_indices["train"]
        requested_indices = {
            split_name: protocol.indices(split_name) for split_name in args.splits
        }
        helper_execution = {
            "mode": "in_sample_helper",
            "fold_id": None,
            "held_out_seeds": [],
            "helper_train_seeds": list(DEFAULT_SELECTOR_SEEDS["train"]),
        }
    train_indices = limit_indices_seed_balanced(
        raw_train_indices, arrays["sample_seed"], int(args.helper_train_limit)
    )
    score_model, value_model, train_score, quantiles = _fit_helper_models(
        args,
        train_indices,
        arrays,
        stats,
        world_model,
        policy_model,
        action_scale,
        value_vocab,
        summary["config"],
        device,
    )
    split_summaries = {}
    for split_name in args.splits:
        indices = requested_indices[split_name]
        indices = limit_indices_seed_balanced(
            indices, arrays["sample_seed"], int(args.split_sample_limit)
        )
        protocol_metadata = dict(configuration)
        protocol_metadata["helper_execution"] = helper_execution
        if crossfit_manifest is not None:
            protocol_metadata["crossfit_protocol_digest"] = crossfit_manifest[
                "crossfit_protocol_digest"
            ]
        sample_fold_id = (
            np.full(len(indices), int(crossfit_fold), dtype=np.int16)
            if split_name == "train" and crossfit_fold is not None
            else None
        )
        split_summaries[split_name] = _make_label_split(
            args,
            split_name,
            indices,
            train_indices,
            arrays,
            stats,
            world_model,
            policy_model,
            action_scale,
            value_vocab,
            summary["config"],
            device,
            score_model,
            value_model,
            train_score,
            quantiles,
            configuration_digest,
            protocol_metadata,
            sample_fold_id=sample_fold_id,
        )
    result = {
        "framework": "PI-JWM",
        "mode": "v11_selector_candidate_actual_rollout_labels",
        "result_kind": "diagnostic_only",
        "configuration": configuration,
        "configuration_digest": configuration_digest,
        "configuration_frozen": False,
        "helper_protocol": helper_protocol,
        "helper_execution": helper_execution,
        "device": str(device),
        "helper_train_samples": int(len(train_indices)),
        "label_samples": {
            name: int(value["num_samples"]) for name, value in split_summaries.items()
        },
        "splits": split_summaries,
        "runtime_seconds": float(time.time() - started),
    }
    (args.output_dir / "candidate_label_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "reproduction_command.txt").write_text(
        build_reproduction_command(args) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
