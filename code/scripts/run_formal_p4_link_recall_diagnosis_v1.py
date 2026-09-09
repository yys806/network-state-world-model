"""CPU-only validation diagnosis for the frozen P4 low-recall sentinel."""

from __future__ import annotations

import argparse
import hashlib
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
from run_formal_dual_graph_cpu_smoke_v1 import _manifest, _sha256, _subset_for_ids
from run_formal_dual_graph_gpu_train_v1 import _prediction, _reload_learned_model
from run_formal_p4_link_probability_calibration_v1 import _verify_manifest_file


SCHEMA_VERSION = "PI-JWM-link-recall-diagnosis-v1"
CANONICAL_TENSOR_MANIFEST_SHA256 = "d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781"
FROZEN_SENTINEL_CHECKPOINT_SHA256 = "0d203fa46b89e3b4267f387371e4a8c38682810482d30af7c8058f2ea2ac141f"
FROZEN_SENTINEL_SAMPLE_IDS_SHA256 = "1f01dade9a0a673fd95bb3d5953a00617c09f124cdc019673765983ea7cde582"
RAW_THRESHOLD = 0.9
POS_WEIGHT = 50.0
HORIZONS = tuple(range(1, 21))


def _require_binary_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() <= 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor")
    if torch.is_complex(value) or not torch.isfinite(value).all().item() or not torch.logical_or(value == 0, value == 1).all().item():
        raise ValueError(f"{name} must contain only finite binary values")
    return value.detach().cpu().to(torch.int64)


def _require_inputs(
    logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], previous: Sequence[torch.Tensor], persistence: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    values = (logits, labels, previous, persistence)
    names = ("logits", "labels", "previous_link_activity", "persistence_predictions")
    if any(not isinstance(value, Sequence) or len(value) != 20 for value in values):
        raise ValueError("all diagnosis inputs must provide exactly 20 aligned horizons")
    checked: list[list[torch.Tensor]] = [[], [], [], []]
    for index in range(20):
        logit = logits[index]
        if not isinstance(logit, torch.Tensor) or logit.ndim != 1 or logit.numel() <= 0 or torch.is_complex(logit) or not torch.isfinite(logit).all().item():
            raise ValueError(f"logits h{index + 1} must be a finite non-empty 1D tensor")
        checked[0].append(logit.detach().cpu())
        for output, value, name in zip(checked[1:], values[1:], names[1:]):
            row = _require_binary_vector(value[index], f"{name} h{index + 1}") if name != "previous_link_activity" else value[index]
            if name == "previous_link_activity":
                if not isinstance(row, torch.Tensor) or row.ndim != 1 or row.numel() <= 0 or torch.is_complex(row) or not torch.isfinite(row).all().item() or not torch.logical_or(torch.logical_or(row == -1, row == 0), row == 1).all().item():
                    raise ValueError(f"previous_link_activity h{index + 1} must contain only -1, 0, or 1")
                row = row.detach().cpu().to(torch.int64)
            output.append(row)
        if any(row[index].shape != checked[0][index].shape for row in checked[1:]):
            raise ValueError(f"diagnosis inputs are not aligned at h{index + 1}")
    return checked[0], checked[1], checked[2], checked[3]


def _require_provenance(provenance: Mapping[str, Any] | None, mask_counts: Mapping[str, int]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("explicit diagnosis provenance is required")
    result = dict(provenance)
    for name in ("checkpoint_sha256", "tensor_manifest_sha256", "sample_ids_sha256"):
        if not isinstance(result.get(name), str) or len(result[name]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in result[name]):
            raise ValueError(f"provenance is missing {name}")
    if not isinstance(result.get("class_weights_path"), str) or not result["class_weights_path"]:
        raise ValueError("provenance is missing class_weights_path")
    class_weights_sha256 = result.get("class_weights_sha256")
    if not isinstance(class_weights_sha256, str) or len(class_weights_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in class_weights_sha256):
        raise ValueError("provenance is missing class_weights_sha256")
    if result.get("class_weights_source_split") != "train" or float(result.get("link_activity_pos_weight", float("nan"))) != POS_WEIGHT:
        raise ValueError("class_weights provenance must be train-only link_activity pos_weight=50")
    supplied = result.get("mask_counts")
    expected_keys = {f"h{index}" for index in HORIZONS} | {"total"}
    if not isinstance(supplied, Mapping) or set(supplied) != expected_keys or any(not isinstance(supplied[key], int) or supplied[key] <= 0 for key in expected_keys):
        raise ValueError("provenance mask_counts must contain positive h1..h20 and total counts")
    if dict(supplied) != dict(mask_counts):
        raise ValueError("provenance mask_counts drift from actual valid link inputs")
    reload = result.get("checkpoint_reload")
    if not isinstance(reload, Mapping) or reload.get("strict") is not True or reload.get("missing_keys") != 0 or reload.get("unexpected_keys") != 0:
        raise ValueError("strict checkpoint reload provenance is required")
    for name, required in (("locked_test_accessed", False), ("gpu_execution", False), ("formal_performance_claim_ready", False)):
        if result.get(name) is not required:
            raise ValueError(f"provenance {name} must be {required}")
    return result


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    return {f"q{int(level * 100)}": float(torch.quantile(values, level).item()) for level in (0.0, 0.1, 0.5, 0.9, 1.0)}


def _statistics(logits: torch.Tensor) -> dict[str, Any]:
    if logits.numel() == 0:
        return {"status": "not_computable", "reason": "empty_subset"}
    logits = logits.to(torch.float64)
    raw_score = torch.sigmoid(logits)
    corrected = torch.sigmoid(logits - math.log(POS_WEIGHT))
    return {
        "raw_logit": _quantiles(logits),
        "raw_weighted_score": _quantiles(raw_score),
        "weight_corrected_probability": _quantiles(corrected),
    }


def _bucket(logits: torch.Tensor, labels: torch.Tensor, persistence: torch.Tensor, subset: torch.Tensor) -> dict[str, Any]:
    if not subset.any().item():
        return {"status": "not_computable", "reason": "empty_subset"}
    logit = logits[subset]
    label = labels[subset].bool()
    candidate = torch.sigmoid(logit) >= RAW_THRESHOLD
    persistent = persistence[subset].bool()
    def counts(prediction: torch.Tensor) -> tuple[int, int, float]:
        tp = int((prediction & label).sum().item())
        fn = int((~prediction & label).sum().item())
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        return tp, fn, recall
    candidate_tp, candidate_fn, candidate_recall = counts(candidate)
    persistence_tp, persistence_fn, persistence_recall = counts(persistent)
    return {
        "status": "computed",
        "positive_count": int(label.sum().item()),
        "candidate": {
            "tp": candidate_tp, "fn": candidate_fn, "recall": candidate_recall,
            "statistics": {"tp": _statistics(logit[candidate & label]), "fn": _statistics(logit[~candidate & label])},
        },
        "persistence": {"tp": persistence_tp, "fn": persistence_fn, "recall": persistence_recall},
        "persistence_tp_candidate_fn": int((persistent & ~candidate & label).sum().item()),
    }


def build_link_recall_diagnosis(
    logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], previous_link_activity: Sequence[torch.Tensor],
    persistence_predictions: Sequence[torch.Tensor], provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize frozen-threshold validation recall without fitting or selecting anything."""
    logits, labels, previous, persistence = _require_inputs(logits, labels, previous_link_activity, persistence_predictions)
    mask_counts = {f"h{index + 1}": int(row.numel()) for index, row in enumerate(logits)}
    mask_counts["total"] = sum(mask_counts.values())
    provenance_value = _require_provenance(provenance, mask_counts)
    by_horizon: dict[str, Any] = {}
    for index, (logit, label, prior, baseline) in enumerate(zip(logits, labels, previous, persistence), start=1):
        positive = label == 1
        by_horizon[f"h{index}"] = {
            "all_positive": _bucket(logit, label, baseline, positive),
            "continued_active": _bucket(logit, label, baseline, (prior == 1) & positive),
            "newly_active": _bucket(logit, label, baseline, (prior == 0) & positive),
            "previous_unobserved": _bucket(logit, label, baseline, (prior == -1) & positive),
            "previous_unobserved_positive_count": int(((prior == -1) & positive).sum().item()),
        }
    all_logits, all_labels = torch.cat(logits), torch.cat(labels)
    all_persistence = torch.cat(persistence)
    overall = {
        "all_positive": _bucket(all_logits, all_labels, all_persistence, all_labels == 1),
        "continued_active": _bucket(all_logits, all_labels, all_persistence, (torch.cat(previous) == 1) & (all_labels == 1)),
        "newly_active": _bucket(all_logits, all_labels, all_persistence, (torch.cat(previous) == 0) & (all_labels == 1)),
        "previous_unobserved": _bucket(all_logits, all_labels, all_persistence, (torch.cat(previous) == -1) & (all_labels == 1)),
        "previous_unobserved_positive_count": int(((torch.cat(previous) == -1) & (all_labels == 1)).sum().item()),
    }
    candidate = torch.sigmoid(all_logits) >= RAW_THRESHOLD
    truth, persistent = all_labels.bool(), all_persistence.bool()
    def full_counts(prediction: torch.Tensor) -> dict[str, Any]:
        tp, fp, fn = int((prediction & truth).sum().item()), int((prediction & ~truth).sum().item()), int((~prediction & truth).sum().item())
        return {"tp": tp, "fp": fp, "fn": fn, "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0}
    overall["candidate"] = full_counts(candidate)
    overall["persistence"] = full_counts(persistent)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_split": "validation",
        "raw_threshold": RAW_THRESHOLD,
        "pos_weight": POS_WEIGHT,
        "temperature_fitted": False,
        "probability_interpretation": "diagnostic_weight_corrected_probability_estimate",
        "overall": overall,
        "by_horizon": by_horizon,
        "highlighted_horizons": {f"h{index}": by_horizon[f"h{index}"] for index in (1, 5, 10, 20)},
        "provenance": provenance_value,
        "locked_test_accessed": False,
        "gpu_execution": False,
        "formal_performance_claim_ready": False,
        "sentinel_performance_gate": "no_go",
        "p4_status": "blocked",
    }


def _previous_activity_and_mask(history_activity: torch.Tensor, history_mask: torch.Tensor, target_activity: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if history_activity.shape != history_mask.shape or target_activity.shape != target_mask.shape or history_activity.ndim != 3 or target_activity.ndim != 3 or target_activity.shape[1] != 20 or history_activity.shape[0] != target_activity.shape[0] or history_activity.shape[2] != target_activity.shape[2]:
        raise ValueError("history and target aggregate link activity tensors are not aligned")
    return torch.cat((history_activity[:, -1:], target_activity[:, :-1]), dim=1), torch.cat((history_mask[:, -1:], target_mask[:, :-1]), dim=1).bool()


def _collect_validation(model: torch.nn.Module, loader: DataLoader) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, int], dict[str, str]]:
    rows = [[] for _ in range(20)]
    mask_fingerprint_parts = [[] for _ in range(20)]
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            prediction = _prediction("coupled_dual_gnn_residual", model, batch, {})
            target, history = batch["target"], batch["history"]
            labels, mask = target["aggregate_link_activity"], target["aggregate_link_activity_mask"].bool()
            prior, prior_mask = _previous_activity_and_mask(history["aggregate_link_activity"], history["aggregate_link_activity_mask"].bool(), labels, mask)
            logits = prediction["link_activity_logits"]
            edge_valid = torch.all(batch["static"]["physical_edge_endpoint_index"] >= 0, dim=-1)[:, None, :].expand_as(labels)
            valid = edge_valid & mask
            if logits.shape != labels.shape or valid.shape != labels.shape:
                raise ValueError("formal validation logits, labels, and masks are not aligned")
            persistent = history["aggregate_link_activity"][:, -1:].expand_as(labels)
            for index in range(20):
                active = valid[:, index]
                # Preserve every valid position, not merely its count, for intervention identity checks.
                # Flatten the batch-by-edge mask so each individual valid position
                # contributes its actual bit rather than a nested-row truthiness.
                mask_bits = active.detach().cpu().reshape(-1).tolist()
                mask_fingerprint_parts[index].append(f"{batch_index}:{''.join('1' if item else '0' for item in mask_bits)}")
                rows[index].append((logits[:, index][active].cpu(), labels[:, index][active].cpu(), torch.where(prior_mask[:, index][active], prior[:, index][active], torch.full_like(prior[:, index][active], -1)).cpu(), persistent[:, index][active].cpu()))
    if any(not row for row in rows):
        raise ValueError("validation mask must retain positive count at each horizon")
    result = [[torch.cat([item[column] for item in row]) for row in rows] for column in range(4)]
    counts = {f"h{index + 1}": int(value.numel()) for index, value in enumerate(result[0])}
    counts["total"] = sum(counts.values())
    fingerprints = {f"h{index + 1}": hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest() for index, parts in enumerate(mask_fingerprint_parts)}
    return result[0], result[1], result[2], result[3], counts, fingerprints


def _publish_link_recall_diagnosis_atomically(output_dir: Path, diagnosis: Mapping[str, Any], *, before_rename: Any | None = None) -> None:
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    parent, prefix = target.parent.resolve(), f".{target.name}.staging-"
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f"{prefix}{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        (staging / "link_recall_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {"schema_version": "PI-JWM-link-recall-diagnosis-manifest-v1", "files": {name: {"size_bytes": entry["bytes"], "sha256": entry["sha256"]} for name, entry in _manifest(staging)["files"].items()}, "locked_test_accessed": False, "gpu_execution": False, "formal_performance_claim_ready": False}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        if before_rename is not None:
            before_rename()
        if target.exists():
            raise FileExistsError(f"output directory already exists: {target}")
        staging.rename(target)
    except BaseException:
        if staging.exists() and staging.resolve().parent == parent and staging.name.startswith(prefix):
            shutil.rmtree(staging)
        raise


def _prepare_frozen_link_validation(run_root: Path, tensor_root: Path, method: str, expected_manifest_sha: str | None = None, _expected_checkpoint_sha256: str | None = None, _expected_sample_ids_sha256: str | None = None) -> tuple[torch.nn.Module, DataLoader, dict[str, Any]]:
    """Strictly load the frozen non-locked sentinel and its fixed validation loader."""
    run_root, tensor_root = Path(run_root).resolve(), Path(tensor_root).resolve()
    if method != "coupled_dual_gnn_residual":
        raise ValueError("the frozen sentinel method is coupled_dual_gnn_residual")
    config_path, sample_ids_path, weights_path, checkpoint_path, tensor_manifest_path = run_root / "config.json", run_root / "sample_ids.json", run_root / "class_weights.json", run_root / "checkpoints" / f"{method}__best.pt", tensor_root / "manifest.json"
    if any(not path.is_file() for path in (config_path, sample_ids_path, weights_path, checkpoint_path, run_root / "manifest.json", tensor_manifest_path, tensor_root / "normalization_stats.json", tensor_root / "tensor_contract.json")):
        raise ValueError("frozen run or tensor inputs are incomplete")
    config, sample_ids, weights = (json.loads(path.read_text(encoding="utf-8")) for path in (config_path, sample_ids_path, weights_path))
    if config.get("locked_test_accessed") is not False or "locked_test" in sample_ids or int(config.get("seed", -1)) != 20260831:
        raise ValueError("locked_test evidence is forbidden or frozen sentinel seed drifted")
    if float(weights.get("pos_weight", {}).get("link_activity", float("nan"))) != POS_WEIGHT or weights.get("source_split") != "train":
        raise ValueError("frozen train-only link_activity pos_weight=50.0 is required")
    if config.get("deterministic_rule_layer") is not True or method not in config.get("learned_methods", []):
        raise ValueError("run config method or deterministic rule-layer identity drift")
    if not isinstance(sample_ids.get("validation"), list) or not sample_ids["validation"]:
        raise ValueError("frozen validation sample IDs are required")
    run_manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    for path in (config_path, sample_ids_path, weights_path, checkpoint_path):
        relative = path.relative_to(run_root).as_posix()
        if _sha256(path).lower() != str(run_manifest.get("files", {}).get(relative, {}).get("sha256", "")).lower():
            raise ValueError(f"frozen run manifest identity drift: {relative}")
    expected_checkpoint = _expected_checkpoint_sha256 or FROZEN_SENTINEL_CHECKPOINT_SHA256
    expected_sample_ids = _expected_sample_ids_sha256 or FROZEN_SENTINEL_SAMPLE_IDS_SHA256
    if _sha256(checkpoint_path).lower() != expected_checkpoint.lower():
        raise ValueError("frozen sentinel checkpoint SHA-256 drift")
    if _sha256(sample_ids_path).lower() != expected_sample_ids.lower():
        raise ValueError("frozen sentinel sample_ids SHA-256 drift")
    tensor_manifest = json.loads(tensor_manifest_path.read_text(encoding="utf-8"))
    expected = expected_manifest_sha or CANONICAL_TENSOR_MANIFEST_SHA256
    if tensor_manifest.get("schema_version") != "PI-JWM-AirFogSim-formal-tensor-manifest-v1" or _sha256(tensor_manifest_path).lower() != expected.lower() or str(config.get("dataset_manifest_sha256", "")).lower() != _sha256(tensor_manifest_path).lower():
        raise ValueError("canonical tensor manifest identity drift")
    for relative in ("tensor_contract.json", "normalization_stats.json", "window_index.csv"):
        _verify_manifest_file(tensor_root, tensor_manifest, relative)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != method or not isinstance(checkpoint.get("model_config"), Mapping):
        raise ValueError("checkpoint method or model_config drift")
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    model_config = checkpoint["model_config"]
    if (model_config.get("mode") != "coupled_dual_gnn" or model_config.get("residual_state_prediction") is not True or model_config.get("deterministic_rule_layer") is not True or int(model_config.get("history_steps", -1)) != int(contract.get("history_steps", -2)) or int(model_config.get("horizon_steps", -1)) != int(contract.get("horizon_steps", -2))):
        raise ValueError("checkpoint model_config semantic identity drift")
    model = _reload_learned_model(method, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
    dataset = FormalAirFogSimWindowDataset(tensor_root, split="validation", config=FormalWindowConfig(history_steps=int(contract["history_steps"]), horizon_steps=int(contract["horizon_steps"])), stats=stats, normalize=True)
    selected_validation = _subset_for_ids(dataset, sample_ids["validation"])
    selected_tensor_inputs: list[dict[str, Any]] = []
    for sample_id in sample_ids["validation"]:
        row = next((item for item in dataset.rows if str(item["sample_id"]) == str(sample_id)), None)
        if row is None or str(row.get("split")) != "validation":
            raise ValueError(f"selected validation sample IDs are missing from the dataset: {sample_id}")
        relative = f"seed_{int(row['seed']):03d}/trajectory_tensors.npz"
        _verify_manifest_file(tensor_root, tensor_manifest, relative)
        if not any(item["relative_path"] == relative for item in selected_tensor_inputs):
            entry = tensor_manifest["files"][relative]
            selected_tensor_inputs.append({"relative_path": relative, "size_bytes": int(entry["size_bytes"]), "sha256": str(entry["sha256"]).lower()})
    provenance = {"config_sha256": _sha256(config_path), "checkpoint_sha256": _sha256(checkpoint_path), "tensor_manifest_path": str(tensor_manifest_path), "tensor_manifest_sha256": _sha256(tensor_manifest_path), "sample_ids_path": str(sample_ids_path), "sample_ids_sha256": _sha256(sample_ids_path), "class_weights_path": str(weights_path), "class_weights_sha256": _sha256(weights_path), "selected_validation_tensor_inputs": selected_tensor_inputs, "class_weights_source_split": weights["source_split"], "link_activity_pos_weight": weights["pos_weight"]["link_activity"], "checkpoint_reload": {"strict": True, "missing_keys": 0, "unexpected_keys": 0}, "locked_test_accessed": False, "gpu_execution": False, "formal_performance_claim_ready": False, "checkpoint_path": str(checkpoint_path), "tensor_root": str(tensor_root), "run_root": str(run_root)}
    return model, DataLoader(selected_validation, batch_size=2, shuffle=False, num_workers=0), provenance


def run_formal_p4_link_recall_diagnosis(*, run_root: Path, tensor_root: Path, method: str, output_dir: Path, _expected_tensor_manifest_sha256: str | None = None, _expected_checkpoint_sha256: str | None = None, _expected_sample_ids_sha256: str | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    model, loader, provenance = _prepare_frozen_link_validation(run_root, tensor_root, method, _expected_tensor_manifest_sha256, _expected_checkpoint_sha256, _expected_sample_ids_sha256)
    logits, labels, previous, persistence, mask_counts, _mask_fingerprints = _collect_validation(model, loader)
    # Keep the published v1 report byte-for-byte schema-compatible while sharing preparation.
    legacy_keys = ("checkpoint_sha256", "tensor_manifest_sha256", "sample_ids_sha256", "class_weights_path", "class_weights_sha256", "class_weights_source_split", "link_activity_pos_weight", "checkpoint_reload", "locked_test_accessed", "gpu_execution", "formal_performance_claim_ready", "checkpoint_path", "tensor_root", "run_root")
    diagnosis = build_link_recall_diagnosis(logits, labels, previous, persistence, {**{key: provenance[key] for key in legacy_keys}, "mask_counts": mask_counts})
    if (diagnosis["overall"]["candidate"]["tp"], diagnosis["overall"]["candidate"]["fp"], diagnosis["overall"]["candidate"]["fn"]) != (1902, 557, 5855):
        raise ValueError("frozen sentinel candidate TP/FP/FN drifted from 1902/557/5855")
    _publish_link_recall_diagnosis_atomically(output_dir, diagnosis)
    return diagnosis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tensor-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_formal_p4_link_recall_diagnosis(run_root=args.run_root, tensor_root=args.tensor_root, method=args.method, output_dir=args.output_dir)
    print(json.dumps({"schema_version": report["schema_version"], "p4_status": report["p4_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
