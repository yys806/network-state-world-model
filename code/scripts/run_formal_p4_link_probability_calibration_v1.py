"""CPU-only probability-calibration audit for the frozen P4 link sentinel."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.airfogsim_window_dataset_v2 import _read_window_rows
from pi_jwm.formal_binary_calibration_v1 import (
    InverseTemperatureCalibration,
    binary_probability_metrics,
    choose_legacy_threshold,
    fit_inverse_temperature,
)
from pi_jwm.formal_world_model_metrics_v1 import _classification_records
from run_formal_dual_graph_cpu_smoke_v1 import _manifest, _sha256, _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import _prediction, _reload_learned_model


SCHEMA_VERSION = "PI-JWM-link-probability-calibration-v1"
RAW_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
REQUIRED_HORIZONS = (1, 5, 10, 20)
CANONICAL_TENSOR_MANIFEST_SHA256 = "d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781"


def _require_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("explicit calibration provenance is required")
    value = dict(provenance)
    if value.get("source_split") != "train-only":
        raise ValueError("pos_weight provenance must be train-only")
    for name in ("checkpoint_sha256", "tensor_manifest_sha256", "pos_weight_source_path", "pos_weight_source_sha256"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"provenance is missing {name}")
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, Mapping) or not all(isinstance(sample_ids.get(split), str) and sample_ids[split] for split in ("calibration", "validation")):
        raise ValueError("provenance must identify calibration and validation sample IDs")
    counts = value.get("mask_counts")
    if not isinstance(counts, Mapping) or any(not isinstance(counts.get(split), int) or counts[split] <= 0 for split in ("calibration", "validation")):
        raise ValueError("valid physical-edge aggregate activity masks must be non-empty")
    if value.get("locked_test_accessed") is not False:
        raise ValueError("locked_test access is prohibited")
    if value.get("gpu_execution") is not False:
        raise ValueError("this calibration audit is CPU-only")
    if value.get("formal_performance_claim_ready") is not False:
        raise ValueError("calibration audit may not publish a performance claim")
    return value


def _per_horizon(values: torch.Tensor | Sequence[torch.Tensor], name: str) -> list[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        if values.ndim < 2 or values.shape[1] < max(REQUIRED_HORIZONS):
            raise ValueError(f"{name} must retain at least 20 rollout horizons")
        return [values[:, horizon, ...].detach().cpu().reshape(-1) for horizon in range(values.shape[1])]
    if not isinstance(values, Sequence) or len(values) < max(REQUIRED_HORIZONS):
        raise ValueError(f"{name} must provide one valid vector per rollout horizon")
    result: list[torch.Tensor] = []
    for index, item in enumerate(values):
        if not isinstance(item, torch.Tensor):
            raise ValueError(f"{name}[{index}] must be a tensor")
        result.append(item.detach().cpu().reshape(-1))
    return result


def _split_vectors(logits: torch.Tensor | Sequence[torch.Tensor], labels: torch.Tensor | Sequence[torch.Tensor], split: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logit_rows = _per_horizon(logits, f"{split}_logits")
    label_rows = _per_horizon(labels, f"{split}_labels")
    if len(logit_rows) != len(label_rows):
        raise ValueError(f"{split} logits and labels have different horizon counts")
    for horizon, (logit, label) in enumerate(zip(logit_rows, label_rows)):
        if logit.numel() == 0 or label.numel() == 0 or logit.shape != label.shape:
            raise ValueError(f"{split} horizon {horizon + 1} has no aligned valid link inputs")
        if torch.is_complex(logit) or not torch.isfinite(logit).all().item():
            raise ValueError(f"{split} logits must be finite")
        if torch.is_complex(label) or not torch.isfinite(label).all().item() or not torch.logical_or(label == 0, label == 1).all().item():
            raise ValueError(f"{split} labels must be finite binary values")
    return logit_rows, label_rows


def _metrics_by_horizon(logits: list[torch.Tensor], labels: list[torch.Tensor], calibration: InverseTemperatureCalibration) -> dict[str, Any]:
    def values(logit: torch.Tensor, label: torch.Tensor) -> dict[str, Any]:
        probabilities = calibration.probabilities_from_weighted_logits(logit.double())
        return binary_probability_metrics(probabilities, label)

    aggregate_logits = torch.cat(logits)
    aggregate_labels = torch.cat(labels)
    result = {"overall": values(aggregate_logits, aggregate_labels)}
    for horizon in REQUIRED_HORIZONS:
        result[f"k={horizon}"] = values(logits[horizon - 1], labels[horizon - 1])
    return result


def _paired_probability_metrics(
    logits: list[torch.Tensor], labels: list[torch.Tensor], identity: InverseTemperatureCalibration,
    fitted: InverseTemperatureCalibration,
) -> dict[str, Any]:
    identity_metrics = _metrics_by_horizon(logits, labels, identity)
    fitted_metrics = _metrics_by_horizon(logits, labels, fitted)
    return {
        bucket: {"identity": identity_metrics[bucket], "fitted": fitted_metrics[bucket]}
        for bucket in identity_metrics
    }


def _classification(logits: list[torch.Tensor], labels: list[torch.Tensor], *, raw_threshold: float, calibration: InverseTemperatureCalibration) -> dict[str, Any]:
    raw = torch.sigmoid(torch.cat(logits))
    probabilities = calibration.probabilities_from_weighted_logits(torch.cat(logits).double())
    truth = torch.cat(labels).bool()

    def records(scores: torch.Tensor, decisions: torch.Tensor) -> dict[str, Any]:
        counts = {
            "tp": int((decisions & truth).sum().item()),
            "fp": int((decisions & ~truth).sum().item()),
            "fn": int((~decisions & truth).sum().item()),
            "valid": int(truth.numel()),
        }
        metric = _classification_records(
            "event.link_activity", counts, [scores.numpy()], [truth.numpy()],
            ["aggregate_link_activity", "aggregate_link_activity_mask", "physical_edge_endpoint_index"],
        )
        return {
            "counts": counts,
            "precision": metric["event.link_activity.precision"]["value"],
            "recall": metric["event.link_activity.recall"]["value"],
            "f1": metric["event.link_activity.f1"]["value"],
            "auprc": metric["event.link_activity.auprc"]["value"],
        }

    legacy_decisions = raw >= raw_threshold
    legacy = records(raw, legacy_decisions)
    fitted = records(probabilities, legacy_decisions)
    if legacy["counts"] != fitted["counts"] or any(legacy[name] != fitted[name] for name in ("precision", "recall", "f1", "auprc")):
        raise ValueError("probability calibration changed protected classification or ranking metrics")
    return {"legacy_raw": legacy, "fitted": fitted}


def _direct_probability_threshold_equivalence(
    logits: list[torch.Tensor], calibration: InverseTemperatureCalibration,
) -> dict[str, bool]:
    raw_logits = torch.cat(logits)
    raw_scores = torch.sigmoid(raw_logits)
    probabilities = calibration.probabilities_from_weighted_logits(raw_logits.double())
    mapped = [calibration.map_raw_threshold(threshold) for threshold in RAW_THRESHOLDS]
    if mapped != sorted(mapped) or len(set(mapped)) != len(mapped):
        raise ValueError("shared probability-threshold mapping is not strictly monotonic")
    return {
        str(threshold): bool(torch.equal(
            raw_scores >= threshold,
            probabilities >= calibration.map_raw_threshold(threshold),
        ))
        for threshold in RAW_THRESHOLDS
    }


def _legacy_raw_candidates(
    raw_logits: torch.Tensor, labels: torch.Tensor, calibration: InverseTemperatureCalibration,
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_scores = torch.sigmoid(raw_logits)
    probabilities = calibration.probabilities_from_weighted_logits(raw_logits.double())
    truth = labels.bool()
    result: list[dict[str, Any]] = []
    for selected in selection["candidates"]:
        threshold = float(selected["raw_threshold"])
        decision = raw_scores >= threshold
        counts = {
            "tp": int((decision & truth).sum().item()),
            "fp": int((decision & ~truth).sum().item()),
            "fn": int((~decision & truth).sum().item()),
            "valid": int(truth.numel()),
        }
        metric = _classification_records(
            "event.link_activity", counts, [raw_scores.detach().cpu().numpy()],
            [truth.detach().cpu().numpy()],
            ["aggregate_link_activity", "aggregate_link_activity_mask", "physical_edge_endpoint_index"],
        )
        result.append({
            "raw_threshold": threshold,
            "probability_threshold": calibration.map_raw_threshold(threshold),
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "f1": metric["event.link_activity.f1"]["value"] or 0.0,
            "direct_probability_threshold_decision_equivalent": bool(torch.equal(
                decision, probabilities >= calibration.map_raw_threshold(threshold),
            )),
            "formal_classification_decision_equivalent_to_legacy": True,
        })
    return result


def build_link_probability_calibration_audit(
    calibration_logits: torch.Tensor | Sequence[torch.Tensor],
    calibration_labels: torch.Tensor | Sequence[torch.Tensor],
    validation_logits: torch.Tensor | Sequence[torch.Tensor],
    validation_labels: torch.Tensor | Sequence[torch.Tensor],
    pos_weight: float = 50.0,
    legacy_selected_raw_threshold: float = 0.9,
    provenance: Mapping[str, Any] | None = None,
    method: str = "coupled_dual_gnn_residual",
) -> dict[str, Any]:
    """Fit T on calibration only and audit the fixed legacy threshold on both splits."""

    provenance_value = _require_provenance(provenance)
    if not math.isfinite(float(pos_weight)) or float(pos_weight) <= 0.0:
        raise ValueError("pos_weight must be finite and positive")
    if float(legacy_selected_raw_threshold) not in RAW_THRESHOLDS:
        raise ValueError("the selected raw threshold is outside the frozen candidate set")
    if method not in {"coupled_dual_gnn_residual", "link_activity_persistence_residual_v1"}:
        raise ValueError(f"unsupported probability-audit method: {method}")
    if method == "coupled_dual_gnn_residual" and float(legacy_selected_raw_threshold) != 0.9:
        raise ValueError("the frozen legacy sentinel raw threshold is 0.9")
    calibration_logit_rows, calibration_label_rows = _split_vectors(calibration_logits, calibration_labels, "calibration")
    validation_logit_rows, validation_label_rows = _split_vectors(validation_logits, validation_labels, "validation")
    actual_mask_counts = {
        "calibration": sum(int(row.numel()) for row in calibration_logit_rows),
        "validation": sum(int(row.numel()) for row in validation_logit_rows),
    }
    if dict(provenance_value["mask_counts"]) != actual_mask_counts:
        raise ValueError("provenance mask_counts drift from actual valid link inputs")
    all_calibration_logits_raw = torch.cat(calibration_logit_rows)
    all_calibration_logits = all_calibration_logits_raw.double()
    all_calibration_labels = torch.cat(calibration_label_rows).bool()
    fitted, fit_report = fit_inverse_temperature(all_calibration_logits, all_calibration_labels, pos_weight=float(pos_weight), fit_split="calibration")
    identity = InverseTemperatureCalibration(float(pos_weight), 0.0)
    selection = choose_legacy_threshold(
        torch.sigmoid(all_calibration_logits_raw), all_calibration_labels, fitted, RAW_THRESHOLDS
    )
    selected = float(selection["selected"]["raw_threshold"])
    if selected != float(legacy_selected_raw_threshold):
        raise ValueError("calibration threshold selection drifted from the frozen selected raw threshold")
    mapped = fitted.map_raw_threshold(selected)
    mapped_thresholds = {str(value): fitted.map_raw_threshold(value) for value in RAW_THRESHOLDS}
    legacy_candidates = _legacy_raw_candidates(
        all_calibration_logits_raw, all_calibration_labels, fitted, selection
    )
    validation_metrics = _paired_probability_metrics(
        validation_logit_rows, validation_label_rows, identity, fitted
    )
    for metric_name in ("nll", "brier", "ece"):
        fitted_value = float(validation_metrics["overall"]["fitted"][metric_name])
        identity_value = float(validation_metrics["overall"]["identity"][metric_name])
        if fitted_value > identity_value + 1e-12:
            raise ValueError(f"validation {metric_name} regressed after temperature fitting")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "calibration_method": "inverse_temperature_v1",
        "raw_logit_source": "link_activity_logits",
        "raw_weighted_score_definition": "sigmoid(raw_logit)",
        "pos_weight": float(pos_weight),
        "pos_weight_source_path": provenance_value["pos_weight_source_path"],
        "pos_weight_source_sha256": provenance_value["pos_weight_source_sha256"],
        "inverse_probability_definition": "sigmoid((raw_logit - log(pos_weight)) / temperature)",
        "temperature": fitted.temperature,
        "log_temperature": fitted.log_temperature,
        "temperature_fit_split": "calibration",
        "temperature_fit_objective": "unweighted_bernoulli_nll",
        "temperature_fit_report": fit_report,
        "threshold_selection_split": "calibration",
        "validation_used_for_fit_or_selection": False,
        "legacy_raw_thresholds": list(RAW_THRESHOLDS),
        "legacy_raw_candidates": legacy_candidates,
        "mapped_probability_thresholds": mapped_thresholds,
        "selected_legacy_raw_threshold": selected,
        "selected_probability_threshold": mapped,
        "decision_equivalence_to_legacy": True,
        "decision_equivalence_to_legacy_definition": "formal classification path explicitly reuses legacy raw decisions, preserving TP/FP/FN/precision/recall/F1; it does not claim bitwise equality for direct finite-precision probability-threshold comparison",
        "direct_probability_threshold_comparison_equivalence_by_split": {
            "calibration": _direct_probability_threshold_equivalence(calibration_logit_rows, fitted),
            "validation": _direct_probability_threshold_equivalence(validation_logit_rows, fitted),
        },
        "classification_decision_coordinate": "legacy_raw_weighted_score",
        "classification_decision_reuses_legacy_raw": True,
        "mapped_thresholds_strictly_monotonic": True,
        "probability_metrics": {
            "calibration": _paired_probability_metrics(calibration_logit_rows, calibration_label_rows, identity, fitted),
            "validation": validation_metrics,
        },
        "classification": {
            "calibration": _classification(calibration_logit_rows, calibration_label_rows, raw_threshold=selected, calibration=fitted),
            "validation": _classification(validation_logit_rows, validation_label_rows, raw_threshold=selected, calibration=fitted),
        },
        "provenance": provenance_value,
        "locked_test_accessed": False,
        "gpu_execution": False,
        "formal_performance_claim_ready": False,
        "sentinel_performance_gate": "no_go",
        "p4_status": "blocked",
    }
    return audit


def _verify_manifest_file(tensor_root: Path, manifest: Mapping[str, Any], relative: str) -> None:
    path = tensor_root / relative
    entry = manifest.get("files", {}).get(relative)
    if not path.is_file() or not isinstance(entry, Mapping):
        raise ValueError(f"frozen tensor manifest is missing required input: {relative}")
    if (
        int(entry.get("size_bytes", -1)) != path.stat().st_size
        or str(entry.get("sha256", "")).lower() != _sha256(path).lower()
    ):
        raise ValueError(f"frozen tensor input identity drift: {relative}")


def _verify_selected_tensor_inputs(
    tensor_root: Path, manifest: Mapping[str, Any], sample_ids: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    for relative in ("tensor_contract.json", "normalization_stats.json", "window_index.csv"):
        _verify_manifest_file(tensor_root, manifest, relative)
    rows = _read_window_rows(tensor_root, split=None)
    by_id = {str(row["sample_id"]): row for row in rows}
    selected: dict[str, list[dict[str, Any]]] = {}
    for split in ("calibration", "validation"):
        requested = sample_ids.get(split)
        if not isinstance(requested, list) or not requested:
            raise ValueError(f"frozen {split} sample IDs are required")
        selected[split] = []
        for sample_id in requested:
            row = by_id.get(str(sample_id))
            if row is None or str(row.get("split")) != split:
                raise ValueError(f"selected sample IDs are missing from the dataset: {sample_id}")
            relative = f"seed_{int(row['seed']):03d}/trajectory_tensors.npz"
            _verify_manifest_file(tensor_root, manifest, relative)
            selected[split].append(row)
    return selected


def _publish_calibration_audit_atomically(
    output_dir: Path, audit: Mapping[str, Any], *, before_rename: Any | None = None,
) -> None:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    parent = target.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    prefix = f".{target.name}.staging-"
    staging = parent / f"{prefix}{token}"
    try:
        staging.mkdir()
        audit_path = staging / "link_probability_calibration_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        manifest_files = {
            name: {"size_bytes": entry["bytes"], "sha256": entry["sha256"]}
            for name, entry in _manifest(staging)["files"].items()
        }
        manifest = {"schema_version": "PI-JWM-link-probability-calibration-manifest-v1", "files": manifest_files, "locked_test_accessed": False, "gpu_execution": False, "formal_performance_claim_ready": False}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        if before_rename is not None:
            before_rename()
        if target.exists():
            raise FileExistsError(f"output directory already exists: {target}")
        staging.rename(target)
    except BaseException:
        if staging.exists() and staging.resolve().parent == parent and staging.name == f"{prefix}{token}":
            shutil.rmtree(staging)
        raise


def _collect_split(model: torch.nn.Module, loader: DataLoader, *, method: str = "coupled_dual_gnn_residual") -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    logits_by_horizon: list[list[torch.Tensor]] = []
    labels_by_horizon: list[list[torch.Tensor]] = []
    mask_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = _prediction(method, model, batch, {})
            target = batch["target"]
            logits = prediction["link_activity_logits"]
            labels = target.get("aggregate_link_activity", target["link_activity"])
            aggregate_mask = target.get("aggregate_link_activity_mask")
            if aggregate_mask is None:
                raise ValueError("aggregate_link_activity_mask is required by the formal path")
            edge_valid = torch.all(batch["static"]["physical_edge_endpoint_index"] >= 0, dim=-1)
            valid = edge_valid[:, None, :].expand_as(labels) & aggregate_mask.bool()
            if not logits_by_horizon:
                logits_by_horizon = [[] for _ in range(logits.shape[1])]
                labels_by_horizon = [[] for _ in range(logits.shape[1])]
            if logits.shape != labels.shape or valid.shape != labels.shape:
                raise ValueError("formal link logits, labels, and masks are not aligned")
            for horizon in range(logits.shape[1]):
                logits_by_horizon[horizon].append(logits[:, horizon][valid[:, horizon]].cpu())
                labels_by_horizon[horizon].append(labels[:, horizon][valid[:, horizon]].cpu().bool())
                mask_count += int(valid[:, horizon].sum().item())
    return [torch.cat(row) if row else torch.empty((0,), dtype=torch.float64) for row in logits_by_horizon], [torch.cat(row) if row else torch.empty((0,), dtype=torch.bool) for row in labels_by_horizon], mask_count


def _resolve_frozen_raw_threshold(method: str, path: Path) -> float:
    """Resolve the calibration-selected raw threshold without consulting validation."""

    if method == "coupled_dual_gnn_residual":
        return 0.9
    if method != "link_activity_persistence_residual_v1":
        raise ValueError(f"unsupported probability-audit method: {method}")
    if not path.is_file():
        raise ValueError("new method requires a calibration threshold-selection artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "calibration":
        raise ValueError("threshold selection must use calibration split")
    selection = payload.get("events", {}).get("link_activity", {})
    selected = selection.get("selected")
    candidates = selection.get("candidates")
    if not isinstance(selected, Mapping) or not isinstance(candidates, list):
        raise ValueError("link activity threshold selection is incomplete")
    if selected.get("coordinate") != "raw_weighted_score":
        raise ValueError("link activity threshold must be in raw_weighted_score coordinate")
    threshold = float(selected.get("raw_threshold", float("nan")))
    if threshold not in RAW_THRESHOLDS:
        raise ValueError("selected raw threshold is outside the frozen candidate set")
    candidate_values = {
        float(item.get("raw_threshold", float("nan")))
        for item in candidates
        if isinstance(item, Mapping)
    }
    if candidate_values != set(RAW_THRESHOLDS) or len(candidates) != len(RAW_THRESHOLDS):
        raise ValueError("link activity threshold candidate set is incomplete or contains extras")
    if threshold not in candidate_values:
        raise ValueError("selected raw threshold is absent from the frozen candidate set")
    return threshold


def run_formal_p4_link_probability_calibration(
    *, run_root: Path, tensor_root: Path, method: str, output_dir: Path,
    _expected_tensor_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    run_root, tensor_root, output_dir = Path(run_root).resolve(), Path(tensor_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if method not in {"coupled_dual_gnn_residual", "link_activity_persistence_residual_v1"}:
        raise ValueError(f"unsupported probability-audit method: {method}")
    config_path, sample_ids_path, weights_path = (run_root / name for name in ("config.json", "sample_ids.json", "class_weights.json"))
    checkpoint_path = run_root / "checkpoints" / f"{method}__best.pt"
    tensor_manifest_path = tensor_root / "manifest.json"
    normalization_stats_path = tensor_root / "normalization_stats.json"
    threshold_selection_path = run_root / "metrics" / f"{method}__threshold_selection.json"
    required_paths = [config_path, sample_ids_path, weights_path, checkpoint_path, run_root / "manifest.json", tensor_manifest_path, normalization_stats_path, tensor_root / "tensor_contract.json"]
    if method == "link_activity_persistence_residual_v1":
        required_paths.append(threshold_selection_path)
    required = tuple(required_paths)
    if any(not path.is_file() for path in required):
        raise ValueError("frozen run or tensor inputs are incomplete")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sample_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    if config.get("locked_test_accessed") is not False or "locked_test" in sample_ids:
        raise ValueError("locked_test evidence is forbidden")
    if int(config.get("seed", -1)) != 20260831:
        raise ValueError("frozen sentinel seed must be 20260831")
    if config.get("deterministic_rule_layer") is not True or method not in config.get("learned_methods", []):
        raise ValueError("run config method or deterministic rule-layer identity drift")
    if weights.get("source_split") != "train" or float(weights.get("pos_weight", {}).get("link_activity", float("nan"))) != 50.0:
        raise ValueError("frozen train-only link_activity pos_weight=50.0 is required")
    if not all(isinstance(sample_ids.get(split), list) and sample_ids[split] for split in ("calibration", "validation")):
        raise ValueError("frozen calibration and validation sample IDs are required")
    run_manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_inputs = [config_path, sample_ids_path, weights_path, checkpoint_path]
    if method == "link_activity_persistence_residual_v1":
        manifest_inputs.append(threshold_selection_path)
    for path in manifest_inputs:
        relative = str(path.relative_to(run_root)).replace("\\", "/")
        expected = run_manifest.get("files", {}).get(relative, {}).get("sha256")
        if not isinstance(expected, str) or _sha256(path).lower() != expected.lower():
            raise ValueError(f"frozen run manifest identity drift: {relative}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != method or not isinstance(checkpoint.get("model_config"), Mapping):
        raise ValueError("checkpoint method or model_config drift")
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    model_config = dict(checkpoint["model_config"])
    if (
        model_config.get("mode") != "coupled_dual_gnn"
        or model_config.get("residual_state_prediction") is not True
        or model_config.get("deterministic_rule_layer") is not True
        or int(model_config.get("history_steps", -1)) != int(config.get("history_steps", -2))
        or int(model_config.get("horizon_steps", -1)) != int(config.get("horizon_steps", -2))
        or int(model_config.get("history_steps", -1)) != int(contract.get("history_steps", -3))
        or int(model_config.get("horizon_steps", -1)) != int(contract.get("horizon_steps", -3))
        or int(model_config.get("n_rb", -1)) != int(contract.get("n_rb", -4))
    ):
        raise ValueError("checkpoint model_config semantic identity drift")
    tensor_manifest = json.loads(tensor_manifest_path.read_text(encoding="utf-8"))
    if tensor_manifest.get("schema_version") != "PI-JWM-AirFogSim-formal-tensor-manifest-v1":
        raise ValueError("tensor manifest schema identity drift")
    if (
        _expected_tensor_manifest_sha256 is not None
        and _sha256(tensor_manifest_path).lower() != str(_expected_tensor_manifest_sha256).lower()
    ):
        raise ValueError("canonical tensor manifest identity drift")
    if str(config.get("dataset_manifest_sha256", "")).lower() != _sha256(tensor_manifest_path).lower():
        raise ValueError("frozen tensor manifest identity drift")
    _verify_selected_tensor_inputs(tensor_root, tensor_manifest, sample_ids)
    model = _reload_learned_model(method, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    stats = json.loads(normalization_stats_path.read_text(encoding="utf-8"))
    dataset = FormalAirFogSimWindowDataset(tensor_root, split="calibration", config=FormalWindowConfig(history_steps=int(contract["history_steps"]), horizon_steps=int(contract["horizon_steps"])), stats=stats, normalize=True)
    validation_dataset = FormalAirFogSimWindowDataset(tensor_root, split="validation", config=FormalWindowConfig(history_steps=int(contract["history_steps"]), horizon_steps=int(contract["horizon_steps"])), stats=stats, normalize=True)
    calibration_logits, calibration_labels, calibration_count = _collect_split(model, DataLoader(_subset_for_ids(dataset, sample_ids["calibration"]), batch_size=2, shuffle=False, num_workers=0), method=method)
    validation_logits, validation_labels, validation_count = _collect_split(model, DataLoader(_subset_for_ids(validation_dataset, sample_ids["validation"]), batch_size=2, shuffle=False, num_workers=0), method=method)
    provenance = {
        "source_split": "train-only",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "tensor_manifest_sha256": _sha256(tensor_manifest_path),
        "sample_ids": {"calibration": _sha256(sample_ids_path) + ":calibration", "validation": _sha256(sample_ids_path) + ":validation"},
        "pos_weight_source_path": str(weights_path),
        "pos_weight_source_sha256": _sha256(weights_path),
        "mask_counts": {"calibration": calibration_count, "validation": validation_count},
        "locked_test_accessed": False,
        "gpu_execution": False,
        "formal_performance_claim_ready": False,
        "checkpoint_path": str(checkpoint_path),
        "tensor_root": str(tensor_root),
        "run_root": str(run_root),
        "checkpoint_reload": {"strict": True, "missing_keys": 0, "unexpected_keys": 0},
    }
    selected_raw_threshold = _resolve_frozen_raw_threshold(method, threshold_selection_path)
    audit = build_link_probability_calibration_audit(calibration_logits, calibration_labels, validation_logits, validation_labels, pos_weight=float(weights["pos_weight"]["link_activity"]), legacy_selected_raw_threshold=selected_raw_threshold, provenance=provenance, method=method)
    audit["method"] = method
    _publish_calibration_audit_atomically(output_dir, audit)
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tensor-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = run_formal_p4_link_probability_calibration(run_root=args.run_root, tensor_root=args.tensor_root, method=args.method, output_dir=args.output_dir, _expected_tensor_manifest_sha256=CANONICAL_TENSOR_MANIFEST_SHA256)
    print(json.dumps({"schema_version": audit["schema_version"], "temperature": audit["temperature"], "formal_performance_claim_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
