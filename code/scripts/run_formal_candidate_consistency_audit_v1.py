"""Audit theory-code consistency for the selected non-locked aggregate candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_candidate_consistency_audit_v1 import evaluate_consistency_claims
from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig


RULE_MARKERS = (
    "apply_deterministic",
    "U_det",
    "work_conservation",
    "rule_update",
    "deterministic_rule",
    "DeterministicRuleLayer",
)


def protocol_config_view(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only the independent-seed identity from a run configuration."""

    return {key: value for key, value in config.items() if key != "seed"}


def assess_rule_layer_input_contract(observed: dict[str, bool]) -> dict[str, Any]:
    """Report whether the current model boundary can support deterministic rules."""

    missing = [name for name, present in observed.items() if not bool(present)]
    return {
        "status": "ready" if not missing else "blocked",
        "requirements": dict(observed),
        "missing_requirements": missing,
    }


def inspect_model_source(source: str) -> dict[str, bool]:
    """Extract conservative mechanism observations from the model source text."""

    return {
        "residual_anchor_present": all(
            marker in source
            for marker in (
                "residual_bases",
                'history[f"{name}_state"][:, -1]',
                "residual_state_scale * mean",
            )
        ),
        "task_history_conditioning_present": all(
            marker in source
            for marker in (
                'history["task_state"]',
                "self.task_encoder",
                "self.task_history",
            )
        ),
        "future_action_conditioning_present": all(
            marker in source
            for marker in (
                'future_action["task_action"]',
                "range(self.config.horizon_steps)",
            )
        ),
        "deterministic_rule_update_present": any(marker in source for marker in RULE_MARKERS),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _checkpoint_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint has no model_config: {path}")
    return dict(config)


def _all_equal(values: list[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def run_audit(*, run_dirs: list[str | Path], tensor_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_paths = [Path(value) for value in run_dirs]
    tensor_root = Path(tensor_root)
    output_dir = Path(output_dir)
    if not run_paths:
        raise ValueError("at least one run directory is required")
    if "locked_test" in str(tensor_root).lower() or any("locked_test" in str(path).lower() for path in run_paths):
        raise ValueError("locked_test path is forbidden")

    configs = [_read_json(path / "config.json") for path in run_paths]
    summaries = [_read_json(path / "run_summary.json") for path in run_paths]
    checkpoint_configs = [_checkpoint_config(path) for path in run_paths]
    model_source_path = SRC_ROOT / "pi_jwm" / "formal_dual_graph_world_model_v1.py"
    model_source = model_source_path.read_text(encoding="utf-8")
    rule_source_path = SRC_ROOT / "pi_jwm" / "formal_deterministic_rule_layer_v1.py"
    rule_source = rule_source_path.read_text(encoding="utf-8")
    window_source_path = SRC_ROOT / "pi_jwm" / "formal_airfogsim_window_v1.py"
    window_source = window_source_path.read_text(encoding="utf-8")
    source_observations = inspect_model_source(model_source)
    contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _read_json(tensor_root / "normalization_stats.json")
    sample = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(
            history_steps=int(contract["history_steps"]),
            horizon_steps=int(contract["horizon_steps"]),
        ),
        stats=stats,
        normalize=True,
    )[0]
    future_keys = set(sample["future_action"])
    batch_keys = set(sample["history"]) | set(sample["static"])
    rule_layer_input_contract = assess_rule_layer_input_contract(
        {
            "stats_passed_into_model": "rule_layer_stats" in model_source,
            "future_source_endpoint_mapping": "task_action_source_node_index" in future_keys,
            "flow_task_mapping": "flow_task_index" in sample["static"],
            "slot_duration": "slot_seconds" in sample["static"],
            "future_service_outcome": bool(
                {"future_rate", "future_delivered_data", "future_cpu_allocation"} & future_keys
            ) or all(marker in model_source for marker in ("service_outcome", "edge_service_head", "flow_service_head")),
            "deterministic_target_masks": any(
                key.startswith("deterministic_") for key in batch_keys
            ),
        }
    )
    per_rb_source_paths = [
        SRC_ROOT / "pi_jwm" / "formal_dual_graph_world_model_v1.py",
        SRC_ROOT / "pi_jwm" / "formal_world_model_loss_v1.py",
        SRC_ROOT / "pi_jwm" / "formal_world_model_metrics_v1.py",
    ]
    per_rb_outputs_consumed = any(
        "link_activity_by_rb" in path.read_text(encoding="utf-8")
        or "link_rate_by_rb" in path.read_text(encoding="utf-8")
        for path in per_rb_source_paths
    )
    aggregate_boundary_matches = all(
        config.get("model_versions", {}).get("coupled_dual_gnn_residual") == "formal_v1"
        and config.get("residual_state_scale") == 0.5
        and config.get("zero_init_residual_state_heads") is True
        and config.get("locked_test_accessed") is False
        for config in configs
    ) and not per_rb_outputs_consumed
    residual_config_matches = aggregate_boundary_matches and all(
        checkpoint.get("residual_state_prediction") is True
        and checkpoint.get("zero_init_residual_state_heads") is True
        and checkpoint.get("residual_state_scale") == 0.5
        for checkpoint in checkpoint_configs
    )
    observed = {
        "aggregate_boundary_matches": aggregate_boundary_matches,
        "residual_config_matches": residual_config_matches,
        "task_history_conditioning_matches": source_observations["task_history_conditioning_present"],
        "future_action_conditioning_matches": source_observations["future_action_conditioning_present"],
        "deterministic_rule_update_implemented": source_observations["deterministic_rule_update_present"],
        "rule_layer_input_contract_ready": rule_layer_input_contract["status"] == "ready",
        "recursive_rule_output_applied": all(
            marker in model_source
            for marker in (
                "previous_states=recursive_states",
                "recursive_states = {name: value for name, value in learned_means.items()}",
                'outputs.setdefault(f"{name}_state_mean", []).append(learned_means[name])',
            )
        ),
        "cpu_inner_rule_work_conserving": all(
            marker in rule_source
            for marker in (
                "_capped_equal_share_tensor",
                "remaining_cpu / slot",
                "cpu_allocation * slot",
            )
        ) and all(
            forbidden not in rule_source
            for forbidden in ("action[..., 6]", ".item()", "for batch_index", "for task_index")
        ),
        "no_future_state_endpoint_leakage": (
            contract.get("action_source_endpoint_field") == "task_action_source_node_index"
            and "task_action_source_node_index" in future_keys
            and 'arrays["task_node_index"][label_start:label_end]' not in window_source
        ),
        "candidate_checkpoint_rule_layer_enabled": all(
            bool(checkpoint.get("deterministic_rule_layer", False))
            for checkpoint in checkpoint_configs
        ),
        "per_rb_outputs_consumed": per_rb_outputs_consumed,
    }
    report = evaluate_consistency_claims(observed)
    training_started = any(
        bool(summary.get("training_run_complete", False)) for summary in summaries
    )
    gpu_training_started = any(
        bool(summary.get("gpu_execution", False)) for summary in summaries
    )
    report.update(
        {
            "audit_completed": True,
            "formal_performance_claim_ready": False,
            "candidate": {
                "method": "coupled_dual_gnn_residual",
                "learning_rate": 3e-4,
                "seed_count": len(run_paths),
                "seeds": [int(config["seed"]) for config in configs],
                "protocol_configs_identical": _all_equal(
                    [protocol_config_view(config) for config in configs]
                ),
                "checkpoint_configs_identical": _all_equal(checkpoint_configs),
            },
            "observed": observed,
            "source_observations": source_observations,
            "rule_layer_input_contract": rule_layer_input_contract,
            "evidence": {
                "model_source": str(model_source_path.resolve()),
                "rule_layer_source": str(rule_source_path.resolve()),
                "window_source": str(window_source_path.resolve()),
                "loss_source": str((SRC_ROOT / "pi_jwm" / "formal_world_model_loss_v1.py").resolve()),
                "metrics_source": str((SRC_ROOT / "pi_jwm" / "formal_world_model_metrics_v1.py").resolve()),
                "tensor_root": str(tensor_root.resolve()),
                "run_dirs": [str(path.resolve()) for path in run_paths],
                "theory_boundary": "PI-JWM uses learned service outcomes plus per-step deterministic conservation and lifecycle rules before each recursive explicit-state emission; the current candidate remains aggregate-baseline only.",
            },
            "execution_policy": {
                "training_started": training_started,
                "cpu_training_completed": training_started and not gpu_training_started,
                "gpu_started": gpu_training_started,
                "locked_test_accessed": False,
                "locked_test_materialized": (tensor_root / "locked_test").exists(),
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_consistency_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        run_dirs=args.run_dir,
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["audit_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
