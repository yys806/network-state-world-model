"""CPU-only P4 intervention for direct physical-edge rule feedback."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import run_formal_p4_link_activity_bias_latent_diagnosis_v2 as bias_latent_v2
from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig


REQUIRED_SEED = 20260831
REQUIRED_HORIZON = 20
REQUIRED_SPLIT = "validation"
FROZEN_V2_REPORT_SHA256 = "4fbbd9f41bd4e300200e4e276ea5f3401f63c21b4d58e8962641b50723df0783"
LOGIT_THRESHOLD = bias_latent_v2.probability_to_logit(bias_latent_v2.FROZEN_PROBABILITY_THRESHOLD)


def validate_intervention_contract(contract: Mapping[str, Any]) -> str:
    """Reject inputs outside the one pre-registered P4 diagnostic slice."""
    if int(contract.get("seed", -1)) != REQUIRED_SEED:
        raise ValueError(f"direct-feedback intervention requires seed={REQUIRED_SEED}")
    if int(contract.get("horizon_steps", -1)) != REQUIRED_HORIZON:
        raise ValueError(f"direct-feedback intervention requires horizon_steps={REQUIRED_HORIZON}")
    if contract.get("split") != REQUIRED_SPLIT:
        raise ValueError(f"direct-feedback intervention requires split={REQUIRED_SPLIT!r}")
    if contract.get("locked_test_accessed") is not False:
        raise ValueError("direct-feedback intervention requires locked_test_accessed=false")
    return REQUIRED_SPLIT


def suppress_direct_physical_edge_rule_feedback(model: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Temporarily zero only the direct physical-edge rule-feedback projection."""
    state_feedback = getattr(model, "state_feedback", None)
    if state_feedback is None or "physical_edge" not in state_feedback:
        raise ValueError("model does not expose the direct physical_edge rule-feedback projection")

    def _zero_output(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(output)

    return state_feedback["physical_edge"].register_forward_hook(_zero_output)


def _zero_bias_fp(horizon: Mapping[str, Any]) -> int:
    return int(horizon["zero_bias_counterfactual_at_frozen_threshold"]["count"])


def classify_direct_path_sufficiency(
    baseline_h20: Mapping[str, Any], intervention_h20: Mapping[str, Any]
) -> str:
    """Apply the pre-registered, conservative h20 sufficiency rule."""
    fp_reduced_to_tenth = _zero_bias_fp(intervention_h20) <= 0.10 * _zero_bias_fp(baseline_h20)
    q99_below_threshold = float(intervention_h20["pre_bias_logit"]["q99"]) < LOGIT_THRESHOLD
    return "sufficient_to_explain" if fp_reduced_to_tenth and q99_below_threshold else "not_sufficient_to_explain"


def sufficiency_decision_boundary(decision: str) -> dict[str, Any]:
    """Return the non-causal interpretation boundary attached to either decision."""
    if decision == "not_sufficient_to_explain":
        interpretation = (
            "not_sufficient_to_explain means the direct path alone is insufficient to explain "
            "the observed h20 tail; it does not mean no contribution, does not authorize "
            "attribution to other paths, and does not authorize repair."
        )
    elif decision == "sufficient_to_explain":
        interpretation = (
            "sufficient_to_explain applies only to this pre-registered direct-path intervention; "
            "it does not authorize attribution to other paths or repair."
        )
    else:
        raise ValueError(f"unknown sufficiency decision: {decision!r}")
    return {
        "decision": decision,
        "interpretation": interpretation,
        "authorizes_other_path_attribution": False,
        "authorizes_repair": False,
    }


def assert_v2_baseline_alignment(
    observed_horizons: list[Mapping[str, Any]], expected_horizons: list[Mapping[str, Any]]
) -> None:
    """Ensure the local baseline replay exactly matches the immutable v2 report."""
    if len(observed_horizons) != REQUIRED_HORIZON or len(expected_horizons) != REQUIRED_HORIZON:
        raise AssertionError("baseline alignment requires exactly 20 horizons")
    for index, (observed, expected) in enumerate(zip(observed_horizons, expected_horizons), start=1):
        observed_negative_count = int(observed["negative_count"])
        expected_negative_count = int(expected["negative_count"])
        observed_fp = int(observed["raw_false_positive_at_frozen_threshold"]["count"])
        expected_fp = int(expected["raw_false_positive_at_frozen_threshold"]["count"])
        if observed_negative_count != expected_negative_count:
            raise AssertionError(
                f"v2 baseline negative_count mismatch at h{index}: "
                f"{observed_negative_count} != {expected_negative_count}"
            )
        if observed_fp != expected_fp:
            raise AssertionError(
                f"v2 baseline raw FP mismatch at h{index}: {observed_fp} != {expected_fp}"
            )


def assert_v2_input_provenance(
    observed_provenance: Mapping[str, str], expected_provenance: Mapping[str, Any]
) -> None:
    """Reject a replay unless its v2-recorded inputs are byte-identical."""
    for field in (
        "config_sha256",
        "sample_ids_sha256",
        "best_checkpoint_sha256",
        "tensor_manifest_sha256",
    ):
        if observed_provenance.get(field) != expected_provenance.get(field):
            raise AssertionError(
                f"v2 input provenance mismatch for {field}: "
                f"{observed_provenance.get(field)!r} != {expected_provenance.get(field)!r}"
            )


def _assert_numeric_equal(left: Any, right: Any, *, path: str) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AssertionError(f"h1 baseline/intervention keys differ at {path}")
        for key in left:
            _assert_numeric_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if left is None or right is None:
        if left != right:
            raise AssertionError(f"h1 baseline/intervention mismatch at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(f"h1 baseline/intervention mismatch at {path}: {left!r} != {right!r}")
        return
    if left != right:
        raise AssertionError(f"h1 baseline/intervention mismatch at {path}: {left!r} != {right!r}")


def assert_h1_no_feedback_alignment(
    baseline_horizons: list[Mapping[str, Any]], intervention_horizons: list[Mapping[str, Any]]
) -> None:
    """The first rollout step precedes rule feedback, so it must remain unchanged."""
    _assert_numeric_equal(baseline_horizons[0], intervention_horizons[0], path="h1")


def _sha256(path: Path) -> str:
    return bias_latent_v2._sha256(path)


def load_verified_seed31_validation_baseline(baseline_report_path: Path) -> dict[str, Any]:
    """Load the approved v2 report and select its unique seed-31 validation baseline."""
    actual_sha256 = _sha256(baseline_report_path)
    if actual_sha256 != FROZEN_V2_REPORT_SHA256:
        raise ValueError(
            f"v2 baseline SHA-256 mismatch: {actual_sha256}; expected {FROZEN_V2_REPORT_SHA256}"
        )
    report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if report.get("execution_policy", {}).get("locked_test_accessed") is not False:
        raise ValueError("v2 baseline is not certified non-locked")
    matches = [run for run in report.get("runs", []) if int(run.get("seed", -1)) == REQUIRED_SEED]
    if len(matches) != 1:
        raise ValueError(f"v2 baseline must contain exactly one seed={REQUIRED_SEED} run")
    validation = matches[0].get("splits", {}).get(REQUIRED_SPLIT)
    if not isinstance(validation, dict) or len(validation.get("horizons", [])) != REQUIRED_HORIZON:
        raise ValueError("v2 baseline does not contain the required 20-horizon validation slice")
    return {"report_sha256": actual_sha256, "run": matches[0], "validation": validation}


def _horizon_delta(baseline: Mapping[str, Any], intervention: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "horizon": int(baseline["horizon"]),
        "zero_bias_false_positive_count": _zero_bias_fp(intervention) - _zero_bias_fp(baseline),
        "pre_bias_logit": {
            "q95": float(intervention["pre_bias_logit"]["q95"]) - float(baseline["pre_bias_logit"]["q95"]),
            "q99": float(intervention["pre_bias_logit"]["q99"]) - float(baseline["pre_bias_logit"]["q99"]),
        },
    }


def diagnose_intervention(run_dir: Path, baseline_report_path: Path, output_path: Path, *, batch_size: int = 16) -> dict[str, Any]:
    """Replay the one approved seed/split once before and once during the temporary hook."""
    if batch_size <= 1:
        raise ValueError("batch_size must be greater than one")
    baseline_v2 = load_verified_seed31_validation_baseline(baseline_report_path)
    config_path = run_dir / "config.json"
    sample_ids_path = run_dir / "sample_ids.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_intervention_contract(
        {
            "seed": config.get("seed"),
            "horizon_steps": config.get("horizon_steps"),
            "split": REQUIRED_SPLIT,
            "locked_test_accessed": config.get("locked_test_accessed"),
        }
    )
    sample_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if set(sample_ids) != {"train", "validation", "calibration"} or REQUIRED_SPLIT not in sample_ids:
        raise ValueError("run sample_ids.json must retain the expected train/validation/calibration contract")

    tensor_root = Path(config["tensor_root"])
    tensor_manifest_path = tensor_root / "manifest.json"
    if not tensor_manifest_path.is_file():
        tensor_manifest_path = tensor_root / "tensor_contract.json"
    stats = json.loads((tensor_root / "normalization_stats.json").read_text(encoding="utf-8"))
    checkpoint_path = bias_latent_v2._best_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    observed_provenance = {
        "config_sha256": _sha256(config_path),
        "sample_ids_sha256": _sha256(sample_ids_path),
        "best_checkpoint_sha256": _sha256(checkpoint_path),
        "tensor_manifest_sha256": _sha256(tensor_manifest_path),
    }
    assert_v2_input_provenance(observed_provenance, baseline_v2["run"]["provenance"])
    model = bias_latent_v2._model_from_run(config, stats, tensor_root)
    model.load_state_dict(checkpoint["model_state_dict"])
    bias = float(model.link_activity_head.bias.detach().cpu().item())
    contract = FormalWindowConfig(history_steps=int(config["history_steps"]), horizon_steps=int(config["horizon_steps"]))
    dataset = FormalAirFogSimWindowDataset(tensor_root, split=REQUIRED_SPLIT, config=contract, stats=stats, normalize=True)

    baseline_horizons = bias_latent_v2._collect_horizons(
        model, dataset, sample_ids[REQUIRED_SPLIT], batch_size=batch_size, bias=bias
    )
    assert_v2_baseline_alignment(baseline_horizons, baseline_v2["validation"]["horizons"])

    hook_handle = suppress_direct_physical_edge_rule_feedback(model)
    try:
        intervention_horizons = bias_latent_v2._collect_horizons(
            model, dataset, sample_ids[REQUIRED_SPLIT], batch_size=batch_size, bias=bias
        )
    finally:
        hook_handle.remove()
    assert_h1_no_feedback_alignment(baseline_horizons, intervention_horizons)

    deltas = [_horizon_delta(baseline, intervention) for baseline, intervention in zip(baseline_horizons, intervention_horizons)]
    h20_sufficiency = classify_direct_path_sufficiency(baseline_horizons[-1], intervention_horizons[-1])
    report = {
        "schema_version": "PI-JWM-formal-p4-direct-edge-rule-feedback-intervention-v1",
        "execution_policy": {
            "device": "cpu",
            "training_started": False,
            "gpu_started": False,
            "locked_test_accessed": False,
        },
        "intervention_definition": {
            "target_code_path": "FormalDualGraphWorldModel.forward direct physical-edge rule feedback",
            "target_expression": "state_feedback['physical_edge'](rule_feedback['physical_edge'])",
            "mechanism": "temporary forward hook returning torch.zeros_like(output)",
            "not_intervened": ["node feedback", "flow feedback", "task feedback", "edge GRU", "physical message", "cross-flow coupling", "link head bias", "checkpoint"],
        },
        "input_provenance": {
            "v2_baseline_report": str(baseline_report_path),
            "v2_baseline_report_sha256": baseline_v2["report_sha256"],
            **observed_provenance,
            "tensor_root": str(tensor_root),
        },
        "run": run_dir.name,
        "seed": REQUIRED_SEED,
        "split": REQUIRED_SPLIT,
        "link_activity_head_bias": bias,
        "baseline": {"horizons": baseline_horizons, "highlight_horizons": {str(h): baseline_horizons[h - 1] for h in (1, 5, 20)}},
        "intervention": {"horizons": intervention_horizons, "highlight_horizons": {str(h): intervention_horizons[h - 1] for h in (1, 5, 20)}},
        "intervention_minus_baseline": {"horizons": deltas, "highlight_horizons": {str(h): deltas[h - 1] for h in (1, 5, 20)}},
        "sufficiency": {
            "h20_zero_bias_fp_fraction_of_baseline": (
                float(_zero_bias_fp(intervention_horizons[-1]) / _zero_bias_fp(baseline_horizons[-1]))
                if _zero_bias_fp(baseline_horizons[-1])
                else None
            ),
            "h20_intervention_pre_bias_q99": intervention_horizons[-1]["pre_bias_logit"]["q99"],
            "required_pre_bias_q99_below": LOGIT_THRESHOLD,
            "decision": h20_sufficiency,
            "decision_boundary": sufficiency_decision_boundary(h20_sufficiency),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--v2-baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            diagnose_intervention(args.run_dir, args.v2_baseline_report, args.output, batch_size=args.batch_size),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
