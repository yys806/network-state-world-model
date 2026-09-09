"""Audit the complete RSSM candidate on the formal non-locked P4 path."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from pi_jwm.formal_world_model_loss_v1 import FormalLossWeights, formal_world_model_loss
from run_formal_dual_graph_gpu_train_v1 import _reload_learned_model


LATENT_DYNAMICS = "complete_rssm_prior_posterior_v1"
METHOD_SPECS = {
    "complete_rssm_dual_graph_v1": {
        "node_x_residual_loss_contract": "none",
        "node_x_residual_loss_weight": 0.0,
    },
    "complete_rssm_node_x_safe_dual_graph_v1": {
        "node_x_residual_loss_contract": "node_x_residual_non_degradation_v1",
        "node_x_residual_loss_weight": 1.0,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_mismatches(run_dir: Path) -> list[str]:
    manifest = _read_json(run_dir / "manifest.json")
    mismatches = []
    for relative, expected in manifest.get("files", {}).items():
        path = run_dir / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["bytes"])
            or _sha256(path) != str(expected["sha256"])
        ):
            mismatches.append(relative)
    return mismatches


def _tensor_outputs_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_tensors = {key: value for key, value in left.items() if isinstance(value, torch.Tensor)}
    right_tensors = {key: value for key, value in right.items() if isinstance(value, torch.Tensor)}
    return left_tensors.keys() == right_tensors.keys() and all(
        torch.equal(value, right_tensors[key]) for key, value in left_tensors.items()
    )


def _gradient_sum(model: torch.nn.Module, prefix: str) -> float:
    return float(
        sum(
            parameter.grad.detach().abs().sum()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        )
    )


def run_audit(
    *,
    method: str = "complete_rssm_dual_graph_v1",
    run_dir: str | Path,
    tensor_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    tensor_root = Path(tensor_root)
    output_dir = Path(output_dir)
    if "locked_test" in str(run_dir).lower() or "locked_test" in str(tensor_root).lower():
        raise ValueError("locked_test path is forbidden")
    if method not in METHOD_SPECS:
        raise ValueError(f"unsupported complete RSSM method: {method}")
    method_spec = METHOD_SPECS[method]
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "run_summary.json")
    class_weights = _read_json(run_dir / "class_weights.json")["pos_weight"]
    tensor_contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _read_json(tensor_root / "normalization_stats.json")
    checkpoint_path = run_dir / "checkpoints" / f"{method}__best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config is missing")
    model = _reload_learned_model(method, model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(
            history_steps=int(tensor_contract["history_steps"]),
            horizon_steps=int(tensor_contract["horizon_steps"]),
        ),
        stats=stats,
        normalize=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    target_changed = copy.deepcopy(batch)
    valid_node = target_changed["target"]["node_present"].unsqueeze(-1)
    target_changed["target"]["node_state"] = torch.where(
        valid_node,
        target_changed["target"]["node_state"] + 7.0,
        target_changed["target"]["node_state"],
    )
    model.eval()
    with torch.no_grad():
        original_eval = model(batch)
        changed_eval = model(target_changed)
    deployment_target_invariant = _tensor_outputs_equal(original_eval, changed_eval)

    action_changed = copy.deepcopy(batch)
    action_changed["future_action"]["task_action_present"][:, :, 0] = True
    action_changed["future_action"]["task_action"][:, :, 0, 0] += 3.0
    with torch.no_grad():
        changed_action_eval = model(action_changed)
    action_conditioned_prior = not torch.equal(
        original_eval["rssm_rollout_prior_mean"],
        changed_action_eval["rssm_rollout_prior_mean"],
    )
    action_conditioned_event = not torch.equal(
        original_eval["link_activity_logits"],
        changed_action_eval["link_activity_logits"],
    )

    model.train()
    torch.manual_seed(20260905)
    original_train = model(batch)
    torch.manual_seed(20260905)
    changed_train = model(target_changed)
    training_prior_target_invariant = torch.equal(
        original_train["rssm_rollout_prior_mean"],
        changed_train["rssm_rollout_prior_mean"],
    ) and torch.equal(
        original_train["node_state_mean"], changed_train["node_state_mean"]
    )
    posterior_teacher_target_sensitive = not torch.equal(
        original_train["training_node_state_mean"],
        changed_train["training_node_state_mean"],
    )
    posterior_teacher_event_target_sensitive = not torch.equal(
        original_train["training_link_activity_logits"],
        changed_train["training_link_activity_logits"],
    )
    loss_weights = FormalLossWeights(**config["loss_weights"])
    loss, components = formal_world_model_loss(
        original_train,
        batch["target"],
        batch["static"],
        weights=loss_weights,
        class_weights=class_weights,
        normalization_stats=stats,
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradient_sums = {
        "transition": _gradient_sum(model, "rssm_transition"),
        "prior": _gradient_sum(model, "rssm_prior"),
        "posterior": _gradient_sum(model, "rssm_posterior"),
        "teacher_decoder": _gradient_sum(model, "rssm_teacher_state_heads"),
    }
    safety_checks: dict[str, bool] = {}
    safety_evidence: dict[str, float] = {}
    if method_spec["node_x_residual_loss_contract"] != "none":
        torch.manual_seed(20260906)
        safety_prediction = model(batch)
        safety_target = copy.deepcopy(batch["target"])
        correction = safety_prediction["rssm_node_state_correction"]
        base_x = (
            safety_prediction["node_state_mean"][..., 0] - correction[..., 0]
        ).detach()
        safety_target["node_state"][..., 0] = base_x
        safety_weights = FormalLossWeights(
            state_nll=0.0,
            state_mae=0.0,
            presence=0.0,
            sparse_event=0.0,
            lifecycle=0.0,
            dag=0.0,
            active_rate_mae=0.0,
            rb_occupancy_mae=0.0,
            task_delay_mae=0.0,
            task_deadline_mae=0.0,
            uav_energy_nll=0.0,
            uav_energy_mae=0.0,
            node_x_residual_non_degradation=1.0,
        )
        safety_loss, _ = formal_world_model_loss(
            safety_prediction,
            safety_target,
            batch["static"],
            weights=safety_weights,
            class_weights=class_weights,
            normalization_stats=stats,
        )
        model.zero_grad(set_to_none=True)
        safety_loss.backward()
        correction_gradient = _gradient_sum(model, "rssm_prior_state_heads.node")
        base_gradient = _gradient_sum(model, "base.")
        safety_evidence = {
            "isolated_loss": float(safety_loss.detach()),
            "correction_gradient_sum": correction_gradient,
            "base_gradient_sum": base_gradient,
        }
        safety_checks = {
            "node_x_loss_contract_identity": model_config.get(
                "node_x_residual_loss_contract"
            )
            == method_spec["node_x_residual_loss_contract"],
            "node_x_loss_weight_identity": float(
                config["loss_weights"].get("node_x_residual_non_degradation", 0.0)
            )
            == float(method_spec["node_x_residual_loss_weight"]),
            "node_x_loss_component_present": "node_x_residual_non_degradation"
            in components,
            "node_x_loss_isolated_finite_positive": bool(
                torch.isfinite(safety_loss) and float(safety_loss.detach()) > 0.0
            ),
            "node_x_loss_correction_gradient_nonzero": correction_gradient > 0.0,
            "node_x_loss_base_gradient_zero": base_gradient == 0.0,
        }
    manifest_mismatches = _manifest_mismatches(run_dir)
    checks = {
        "training_run_complete": summary.get("training_run_complete") is True,
        "cpu_execution_only": summary.get("gpu_execution") is False,
        "locked_test_sealed": summary.get("locked_test_accessed") is False,
        "formal_claim_not_ready": summary.get("formal_performance_claim_ready") is False,
        "method_identity": checkpoint.get("method") == method,
        "model_version_identity": checkpoint.get("model_version") == "formal_complete_rssm_v1_1",
        "latent_dynamics_identity": checkpoint.get("latent_dynamics") == LATENT_DYNAMICS,
        "posterior_teacher_identity": model_config.get("training_posterior_teacher") is True,
        "prior_only_identity": model_config.get("deployment_prior_only") is True,
        "residual_initialization_identity": model_config.get(
            "rssm_residual_head_initialization"
        )
        == "zero_when_requested_v1",
        "strict_checkpoint_reload": True,
        "manifest_integrity": not manifest_mismatches,
        "deployment_target_invariant": deployment_target_invariant,
        "training_prior_target_invariant": training_prior_target_invariant,
        "posterior_teacher_target_sensitive": posterior_teacher_target_sensitive,
        "posterior_teacher_event_target_sensitive": posterior_teacher_event_target_sensitive,
        "action_conditioned_prior": action_conditioned_prior,
        "action_conditioned_event": action_conditioned_event,
        "finite_loss": bool(torch.isfinite(loss)),
        "rssm_losses_present": all(
            key in components
            for key in ("rssm_kl", "rssm_teacher_reconstruction", "rssm_overshooting")
        ),
        "rssm_gradients_nonzero": all(value > 0.0 for value in gradient_sums.values()),
        **safety_checks,
    }
    ready = all(checks.values())
    safety_candidate = method_spec["node_x_residual_loss_contract"] != "none"
    report = {
        "schema_version": "PI-JWM-formal-complete-rssm-consistency-audit-v1",
        "status": (
            "ready_for_protocol_freeze"
            if ready and safety_candidate
            else "ready_for_gpu_sentinel"
            if ready
            else "blocked"
        ),
        "checks": checks,
        "gradient_sums": gradient_sums,
        "node_x_safety_evidence": safety_evidence,
        "loss_components": {
            key: float(components[key])
            for key in ("rssm_kl", "rssm_teacher_reconstruction", "rssm_overshooting")
        },
        "manifest_mismatches": manifest_mismatches,
        "evidence": {
            "run_dir": str(run_dir.resolve()),
            "tensor_root": str(tensor_root.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "tensor_manifest_sha256": _sha256(tensor_root / "manifest.json"),
        },
        "launch_gates": {
            "single_seed_gpu_sentinel_allowed": ready and not safety_candidate,
            "protocol_freeze_allowed": ready,
            "followup_seeds_allowed": False,
            "locked_test_allowed": False,
            "formal_performance_claim_allowed": False,
        },
        "result_boundary": "CPU consistency evidence only; no performance improvement claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "complete_rssm_consistency_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_SPECS),
        default="complete_rssm_dual_graph_v1",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        method=args.method,
        run_dir=args.run_dir,
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"].startswith("ready_for_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
