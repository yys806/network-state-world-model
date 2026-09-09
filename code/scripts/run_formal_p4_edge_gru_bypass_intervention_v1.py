"""CPU-only causal intervention: retain the pre-call edge-GRU hidden state."""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_formal_dual_graph_cpu_smoke_v1 import _manifest, _sha256
from run_formal_p4_link_recall_diagnosis_v1 import (
    CANONICAL_TENSOR_MANIFEST_SHA256,
    FROZEN_SENTINEL_CHECKPOINT_SHA256,
    FROZEN_SENTINEL_SAMPLE_IDS_SHA256,
    _collect_validation,
    _prepare_frozen_link_validation,
    build_link_recall_diagnosis,
)


SCHEMA_VERSION = "PI-JWM-p4-edge-gru-bypass-intervention-v1"
FROZEN_BASELINE_DIAGNOSIS_SHA256 = "ee03931c27ddccf4b8967247db3f06dcd3db7f2efe546917ca5e6d7bf1556e4a"
BASELINE_TP_FP_FN = (1902, 557, 5855)
ORIGINAL_CRITICAL_SET_COUNT = 2791
RAW_THRESHOLD = 0.9


@contextlib.contextmanager
def edge_gru_bypass(model: torch.nn.Module) -> Iterator[dict[str, Any]]:
    """Temporarily make only model.edge_transition return its incoming hidden state."""
    transition = getattr(model, "edge_transition", None)
    if not isinstance(transition, torch.nn.GRUCell):
        raise ValueError("model.edge_transition must be torch.nn.GRUCell")
    audit: dict[str, Any] = {"module": "edge_transition", "hook_semantics": "forward_hook_returns_pre_call_hidden_state_args_1", "actual_hook_calls": 0, "hook_removed": False}
    preexisting_hook_ids = set(transition._forward_hooks)

    def bypass(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], _output: torch.Tensor) -> torch.Tensor:
        if len(args) < 2 or not isinstance(args[1], torch.Tensor):
            raise RuntimeError("edge_transition GRUCell hook did not receive a hidden-state argument")
        audit["actual_hook_calls"] += 1
        return args[1]

    handle = transition.register_forward_hook(bypass)
    try:
        yield audit
    finally:
        handle.remove()
        audit["hook_removed"] = True
        if set(transition._forward_hooks) != preexisting_hook_ids:
            raise RuntimeError("edge_transition forward hook removal left residual hooks")


def _require_same_input_sequence(
    baseline_logits: Sequence[torch.Tensor], intervention_logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], previous: Sequence[torch.Tensor], persistence: Sequence[torch.Tensor],
) -> None:
    values = (baseline_logits, intervention_logits, labels, previous, persistence)
    if any(not isinstance(value, Sequence) or len(value) != 20 for value in values):
        raise ValueError("baseline, intervention, labels, previous activity, and persistence must have 20 horizons")
    for index in range(20):
        if not all(isinstance(value[index], torch.Tensor) and value[index].ndim == 1 for value in values):
            raise ValueError(f"all intervention inputs must be 1D tensors at h{index + 1}")
        shape = baseline_logits[index].shape
        if any(value[index].shape != shape for value in values[1:]):
            raise ValueError(f"baseline and intervention inputs drift at h{index + 1}")


def _raw_decision(logits: torch.Tensor) -> torch.Tensor:
    """Keep legacy float32 thresholding in its original tensor dtype."""
    return torch.sigmoid(logits) >= RAW_THRESHOLD


def _evaluate_sufficiency(values: Mapping[str, float | int]) -> dict[str, Any]:
    gates = {
        "recovered_continued": int(values["recovered_continued"]) >= 1396,
        "intervention_continued_fn": int(values["intervention_continued_fn"]) <= 3336,
        "h1_continued_recall": float(values["h1_continued_recall"]) >= 0.20,
        "h20_continued_recall": float(values["h20_continued_recall"]) >= 0.20,
        "intervention_overall_f1": float(values["intervention_overall_f1"]) >= 0.4723570869,
        "intervention_overall_fp": int(values["intervention_overall_fp"]) <= 1114,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {"status": "sufficient_to_explain" if not failed else "not_sufficient", "gates": gates, "failed_gates": failed, "thresholds": {"recovered_continued_min": 1396, "intervention_continued_fn_max": 3336, "h1_continued_recall_min": 0.20, "h20_continued_recall_min": 0.20, "intervention_overall_f1_min": 0.4723570869, "intervention_overall_fp_max": 1114}}


def _critical_set(baseline_logits: Sequence[torch.Tensor], intervention_logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], previous: Sequence[torch.Tensor], persistence: Sequence[torch.Tensor]) -> dict[str, Any]:
    base = torch.cat([value.detach().cpu() for value in baseline_logits])
    intervention = torch.cat([value.detach().cpu() for value in intervention_logits])
    truth = torch.cat([value.detach().cpu().to(torch.int64) for value in labels]).bool()
    prior = torch.cat([value.detach().cpu().to(torch.int64) for value in previous])
    persistent = torch.cat([value.detach().cpu().to(torch.int64) for value in persistence]).bool()
    selected = (prior == 1) & persistent & truth & ~_raw_decision(base)
    if not selected.any().item():
        return {"status": "not_computable", "reason": "empty_original_critical_set"}
    selected_base, selected_intervention = base[selected], intervention[selected]
    recovered = _raw_decision(selected_intervention)
    def quantiles(value: torch.Tensor) -> dict[str, float]:
        value = value.to(torch.float64)
        return {f"q{int(level * 100)}": float(torch.quantile(value, level).item()) for level in (0.0, 0.1, 0.5, 0.9, 1.0)}
    count = int(selected.sum().item())
    if count != ORIGINAL_CRITICAL_SET_COUNT:
        raise ValueError("frozen original critical-set count drifted from 2791")
    return {"status": "computed", "definition": "continued_and_persistence_positive_and_baseline_candidate_negative", "baseline_candidate_negative_count": count, "intervention_recovered_count": int(recovered.sum().item()), "baseline_raw_logit": quantiles(selected_base), "intervention_raw_logit": quantiles(selected_intervention)}


def _add_precision_recall(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """Enrich only this wrapper's copy; do not change the frozen v1 artifact schema."""
    result = json.loads(json.dumps(diagnosis))
    for prediction in ("candidate", "persistence"):
        item = result["overall"][prediction]
        tp, fp, fn = int(item["tp"]), int(item["fp"]), int(item["fn"])
        item["precision"] = float(tp / (tp + fp)) if tp + fp else 0.0
        item["recall"] = float(tp / (tp + fn)) if tp + fn else 0.0
    return result


def _require_intervention_provenance(provenance: Mapping[str, Any]) -> None:
    required_strings = {
        "tensor_manifest_path": None,
        "sample_ids_path": None,
        "intervention": "edge_transition_returns_previous_hidden",
        "edge_transition_module": "FormalDualGraphWorldModel.edge_transition",
    }
    for name, expected in required_strings.items():
        value = provenance.get(name)
        if not isinstance(value, str) or not value or (expected is not None and value != expected):
            raise ValueError(f"intervention provenance {name} drifted")


def build_edge_gru_bypass_intervention_audit(
    baseline_logits: Sequence[torch.Tensor], intervention_logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], previous_link_activity: Sequence[torch.Tensor], persistence_predictions: Sequence[torch.Tensor], provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare the untouched frozen validation scores with one edge-GRU bypass."""
    _require_same_input_sequence(baseline_logits, intervention_logits, labels, previous_link_activity, persistence_predictions)
    if not isinstance(provenance, Mapping):
        raise ValueError("explicit provenance is required")
    _require_intervention_provenance(provenance)
    counts = {f"h{index + 1}": int(value.numel()) for index, value in enumerate(baseline_logits)}
    counts["total"] = sum(counts.values())
    base_provenance = {**dict(provenance), "mask_counts": counts}
    baseline = _add_precision_recall(build_link_recall_diagnosis(baseline_logits, labels, previous_link_activity, persistence_predictions, base_provenance))
    intervention = _add_precision_recall(build_link_recall_diagnosis(intervention_logits, labels, previous_link_activity, persistence_predictions, base_provenance))
    baseline_counts = baseline["overall"]["candidate"]
    if (baseline_counts["tp"], baseline_counts["fp"], baseline_counts["fn"]) != BASELINE_TP_FP_FN:
        raise ValueError("frozen sentinel candidate TP/FP/FN drifted from 1902/557/5855")
    critical = _critical_set(baseline_logits, intervention_logits, labels, previous_link_activity, persistence_predictions)
    recovered = int(critical.get("intervention_recovered_count", 0))
    continued = intervention["overall"]["continued_active"]
    h1, h20 = intervention["by_horizon"]["h1"]["continued_active"], intervention["by_horizon"]["h20"]["continued_active"]
    checks = _evaluate_sufficiency({
        "recovered_continued": recovered,
        "intervention_continued_fn": int(continued.get("candidate", {}).get("fn", 10 ** 9)),
        "h1_continued_recall": float(h1.get("candidate", {}).get("recall", 0.0)),
        "h20_continued_recall": float(h20.get("candidate", {}).get("recall", 0.0)),
        "intervention_overall_f1": float(intervention["overall"]["candidate"]["f1"]),
        "intervention_overall_fp": int(intervention["overall"]["candidate"]["fp"]),
    })
    status = "sufficient_to_explain_dominant_low_recall" if checks["status"] == "sufficient_to_explain" else "not_sufficient_to_explain_dominant_low_recall"
    return {"schema_version": SCHEMA_VERSION, "analysis_split": "validation", "raw_threshold": RAW_THRESHOLD, "baseline": baseline, "intervention": intervention, "original_critical_set": critical, "sufficiency": checks, "edge_gru_transition_intervention": status, "provenance": dict(provenance), "locked_test_accessed": False, "gpu_execution": False, "formal_performance_claim_ready": False, "p4_status": "blocked"}


def _verify_aligned_collections(baseline: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, int], dict[str, str]], intervention: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, int], dict[str, str]]) -> None:
    for name, left, right in zip(("labels", "previous_link_activity", "persistence_predictions"), baseline[1:4], intervention[1:4]):
        if any(not torch.equal(a, b) for a, b in zip(left, right)):
            raise ValueError(f"baseline and intervention {name} drifted")
    if baseline[4] != intervention[4]:
        raise ValueError("baseline and intervention mask counts drifted")
    if baseline[5] != intervention[5]:
        raise ValueError("baseline and intervention mask fingerprints drifted")


def _publish_edge_gru_bypass_intervention_atomically(output_dir: Path, audit: Mapping[str, Any], *, before_rename: Any | None = None) -> None:
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    parent, prefix = target.parent.resolve(), f".{target.name}.staging-"
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f"{prefix}{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        name = "edge_gru_bypass_intervention.json"
        (staging / name).write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {"schema_version": "PI-JWM-p4-edge-gru-bypass-intervention-manifest-v1", "files": {item: {"size_bytes": entry["bytes"], "sha256": entry["sha256"]} for item, entry in _manifest(staging)["files"].items()}, "locked_test_accessed": False, "gpu_execution": False, "formal_performance_claim_ready": False}
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


def run_formal_p4_edge_gru_bypass_intervention(*, run_root: Path, tensor_root: Path, method: str, baseline_diagnosis_path: Path, output_dir: Path, _expected_tensor_manifest_sha256: str | None = None, _expected_baseline_diagnosis_sha256: str | None = None, _expected_checkpoint_sha256: str | None = None, _expected_sample_ids_sha256: str | None = None) -> dict[str, Any]:
    """Run exactly one no-training CPU-only edge-transition bypass comparison."""
    output_dir, baseline_diagnosis_path = Path(output_dir).resolve(), Path(baseline_diagnosis_path).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    expected_baseline = _expected_baseline_diagnosis_sha256 or FROZEN_BASELINE_DIAGNOSIS_SHA256
    if not baseline_diagnosis_path.is_file() or _sha256(baseline_diagnosis_path).lower() != expected_baseline.lower():
        raise ValueError("frozen baseline diagnosis SHA-256 drift")
    frozen_baseline = json.loads(baseline_diagnosis_path.read_text(encoding="utf-8"))
    if tuple(frozen_baseline.get("overall", {}).get("candidate", {}).get(key) for key in ("tp", "fp", "fn")) != BASELINE_TP_FP_FN:
        raise ValueError("frozen baseline diagnosis candidate counts drifted")
    expected_checkpoint = _expected_checkpoint_sha256 or FROZEN_SENTINEL_CHECKPOINT_SHA256
    expected_sample_ids = _expected_sample_ids_sha256 or FROZEN_SENTINEL_SAMPLE_IDS_SHA256
    expected_tensor = _expected_tensor_manifest_sha256 or CANONICAL_TENSOR_MANIFEST_SHA256
    baseline_provenance = frozen_baseline.get("provenance")
    if not isinstance(baseline_provenance, Mapping) or baseline_provenance.get("checkpoint_sha256") != expected_checkpoint or baseline_provenance.get("sample_ids_sha256") != expected_sample_ids or baseline_provenance.get("tensor_manifest_sha256") != expected_tensor:
        raise ValueError("frozen baseline diagnosis provenance identity drift")
    model, loader, provenance = _prepare_frozen_link_validation(run_root, tensor_root, method, _expected_tensor_manifest_sha256, _expected_checkpoint_sha256, _expected_sample_ids_sha256)
    baseline = _collect_validation(model, loader)
    with edge_gru_bypass(model) as hook:
        intervention = _collect_validation(model, loader)
    _verify_aligned_collections(baseline, intervention)
    expected_calls = 20 * len(loader)
    if hook["actual_hook_calls"] != expected_calls:
        raise ValueError(f"edge GRU hook call count drifted: expected {expected_calls}, got {hook['actual_hook_calls']}")
    if hook["hook_removed"] is not True or model.edge_transition._forward_hooks:
        raise RuntimeError("edge GRU hook was not fully removed")
    report = build_edge_gru_bypass_intervention_audit(*baseline[:1], *intervention[:1], *baseline[1:4], {**provenance, "mask_counts": baseline[4], "mask_fingerprints": baseline[5], "intervention_script_sha256": _sha256(Path(__file__)), "baseline_diagnosis_path": str(baseline_diagnosis_path), "baseline_diagnosis_sha256": _sha256(baseline_diagnosis_path), "intervention": "edge_transition_returns_previous_hidden", "edge_transition_module": "FormalDualGraphWorldModel.edge_transition", "hook_semantics": hook["hook_semantics"], "actual_hook_calls": hook["actual_hook_calls"], "expected_hook_calls": expected_calls, "hook_removed": hook["hook_removed"], "input_sha256": {"config": provenance.get("config_sha256"), "checkpoint": provenance.get("checkpoint_sha256"), "tensor_manifest": provenance.get("tensor_manifest_sha256"), "sample_ids": provenance.get("sample_ids_sha256"), "class_weights": provenance.get("class_weights_sha256")}, "locked_test_accessed": False, "gpu_execution": False, "formal_performance_claim_ready": False})
    _publish_edge_gru_bypass_intervention_atomically(output_dir, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tensor-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline-diagnosis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_formal_p4_edge_gru_bypass_intervention(run_root=args.run_root, tensor_root=args.tensor_root, method=args.method, baseline_diagnosis_path=args.baseline_diagnosis, output_dir=args.output_dir)
    print(report["edge_gru_transition_intervention"])


if __name__ == "__main__":
    main()
