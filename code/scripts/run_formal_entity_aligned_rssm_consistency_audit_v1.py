"""Audit the entity-aligned RSSM on real non-locked P4 tensors."""

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
from pi_jwm.formal_entity_aligned_rssm_world_model_v1 import FormalEntityAlignedRSSMWorldModel
from pi_jwm.formal_world_model_loss_v1 import FormalLossWeights, formal_world_model_loss
from run_formal_dual_graph_gpu_train_v1 import (
    _reload_learned_model,
    _set_entity_training_stage,
)


METHOD = "entity_aligned_dual_graph_rssm_v1"
MODEL_VERSION = "formal_entity_aligned_rssm_v1"
LATENT_DYNAMICS = "entity_aligned_complete_rssm_prior_posterior_v1"


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


def _manifest_mismatches(root: Path) -> list[str]:
    manifest = _read_json(root / "manifest.json")
    mismatches = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        expected_size = expected.get("bytes", expected.get("size_bytes"))
        if (
            not path.is_file()
            or path.stat().st_size != int(expected_size)
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
    *, run_dir: str | Path, tensor_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    run_dir, tensor_root, output_dir = Path(run_dir), Path(tensor_root), Path(output_dir)
    if any("locked_test" in str(path).lower() for path in (run_dir, tensor_root, output_dir)):
        raise ValueError("locked_test path is forbidden")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "run_summary.json")
    class_weights = _read_json(run_dir / "class_weights.json")["pos_weight"]
    contract = _read_json(tensor_root / "tensor_contract.json")
    stats = _read_json(tensor_root / "normalization_stats.json")
    checkpoint_path = run_dir / "checkpoints" / f"{METHOD}__best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    model = _reload_learned_model(METHOD, model_config)
    if not isinstance(model, FormalEntityAlignedRSSMWorldModel):
        raise TypeError("checkpoint did not reload the entity-aligned RSSM")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    dataset = FormalAirFogSimWindowDataset(
        tensor_root,
        split="validation",
        config=FormalWindowConfig(
            history_steps=int(contract["history_steps"]),
            horizon_steps=int(contract["horizon_steps"]),
        ),
        stats=stats,
        normalize=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    changed_target = copy.deepcopy(batch)
    changed_target["target"]["node_state"][:, :, 0, 0] += 7.0

    model.eval()
    with torch.no_grad():
        original_eval = model(batch)
        changed_eval = model(changed_target)
    target_invariant = _tensor_outputs_equal(original_eval, changed_eval)

    changed_action = copy.deepcopy(batch)
    present = changed_action["future_action"]["task_action_present"][0]
    target_index = changed_action["future_action"]["task_action_node_index"][0]
    candidates = torch.nonzero(present & torch.any(target_index >= 0, dim=-1))
    if candidates.numel() == 0:
        raise ValueError("validation sample has no valid action endpoint for sensitivity audit")
    action_step, action_task = (int(value) for value in candidates[0])
    changed_action["future_action"]["task_action"][
        0, action_step, action_task, 0
    ] += 3.0
    with torch.no_grad():
        action_eval = model(changed_action)
    action_checks = {
        name: not torch.equal(
            original_eval[f"rssm_{name}_rollout_prior_mean"],
            action_eval[f"rssm_{name}_rollout_prior_mean"],
        )
        for name in ("node", "physical_edge", "flow", "task")
    }

    model.train()
    torch.manual_seed(20260906)
    original_train = model(batch)
    torch.manual_seed(20260906)
    changed_train = model(changed_target)
    prior_target_invariant = torch.equal(
        original_train["rssm_rollout_prior_mean"],
        changed_train["rssm_rollout_prior_mean"],
    ) and torch.equal(original_train["node_state_mean"], changed_train["node_state_mean"])
    posterior_delta = (
        original_train["rssm_node_posterior_mean"]
        - changed_train["rssm_node_posterior_mean"]
    ).abs().sum(dim=-1)
    posterior_target_sensitive = bool(torch.all(posterior_delta[:, :, 0] > 0))
    posterior_entity_local = bool(torch.all(posterior_delta[:, :, 1:] == 0))

    loss_weights = FormalLossWeights(**config["loss_weights"])
    _set_entity_training_stage(model, "rssm")
    model.zero_grad(set_to_none=True)
    loss, components = formal_world_model_loss(
        original_train,
        batch["target"],
        batch["static"],
        weights=loss_weights,
        class_weights=class_weights,
        normalization_stats=stats,
    )
    loss.backward()
    gradient_sums = {
        "base": _gradient_sum(model, "base."),
        "transition": _gradient_sum(model, "transitions."),
        "prior": _gradient_sum(model, "priors."),
        "posterior": _gradient_sum(model, "posteriors."),
        "prior_decoder": _gradient_sum(model, "prior_state_heads."),
        "teacher_decoder": _gradient_sum(model, "teacher_state_heads."),
    }
    masks = model._entity_masks(batch)
    masked_corrections_zero = all(
        bool(
            torch.all(
                original_eval[f"rssm_{name}_state_correction"]
                * (~masks[name])[:, None, :, None]
                == 0
            )
        )
        for name in ("node", "physical_edge", "flow", "task")
    )
    finite_horizons = all(
        bool(torch.isfinite(original_eval[f"{name}_state_mean"]).all())
        for name in ("node", "physical_edge", "flow", "task")
    )
    motion_stats = stats["features"]["node_motion_state"]
    run_manifest_mismatches = _manifest_mismatches(run_dir)
    tensor_manifest_mismatches = _manifest_mismatches(tensor_root)
    checks = {
        "training_run_complete": summary.get("training_run_complete") is True,
        "cpu_execution_only": summary.get("gpu_execution") is False,
        "locked_test_sealed": summary.get("locked_test_accessed") is False,
        "formal_claim_not_ready": summary.get("formal_performance_claim_ready") is False,
        "method_identity": checkpoint.get("method") == METHOD,
        "model_version_identity": checkpoint.get("model_version") == MODEL_VERSION,
        "latent_dynamics_identity": checkpoint.get("latent_dynamics") == LATENT_DYNAMICS,
        "entity_layout_identity": model_config.get("entity_latent_layout") == "node_physical_edge_flow_task_v1",
        "motion_contract_identity": model_config.get("node_motion_contract") == "causal_backward_difference_v1",
        "posterior_teacher_identity": model_config.get("training_posterior_teacher") is True,
        "prior_only_identity": model_config.get("deployment_prior_only") is True,
        "staged_base_identity": checkpoint.get("base_frozen_during_rssm") is True,
        "strict_checkpoint_reload": True,
        "run_manifest_integrity": not run_manifest_mismatches,
        "tensor_manifest_integrity": not tensor_manifest_mismatches,
        "causal_motion_nonzero": int(motion_stats["count"]) > 0,
        "deployment_target_invariant": target_invariant,
        "training_prior_target_invariant": prior_target_invariant,
        "posterior_target_sensitive": posterior_target_sensitive,
        "posterior_entity_local": posterior_entity_local,
        "action_conditioned_node": action_checks["node"],
        "action_conditioned_task": action_checks["task"],
        "finite_h1_h5_h10_h20": finite_horizons and original_eval["node_state_mean"].shape[1] >= 20,
        "masked_entity_corrections_zero": masked_corrections_zero,
        "deterministic_rule_feedback_present": "deterministic_rule_mask" in original_eval,
        "finite_loss": bool(torch.isfinite(loss)),
        "rssm_losses_present": all(
            name in components
            for name in ("rssm_kl", "rssm_teacher_reconstruction", "rssm_overshooting")
        ),
        "rssm_gradients_nonzero": all(
            gradient_sums[name] > 0
            for name in ("transition", "prior", "posterior", "prior_decoder", "teacher_decoder")
        ),
        "base_gradient_zero_when_frozen": gradient_sums["base"] == 0,
    }
    ready = all(checks.values())
    report = {
        "schema_version": "PI-JWM-formal-entity-aligned-rssm-consistency-audit-v1",
        "status": "ready_for_gpu_execution_sentinel" if ready else "blocked",
        "checks": checks,
        "action_sensitivity": action_checks,
        "action_mutation": {"step": action_step, "task": action_task, "feature": "offload"},
        "gradient_sums": gradient_sums,
        "loss_components": {
            key: float(components[key])
            for key in ("rssm_kl", "rssm_teacher_reconstruction", "rssm_overshooting")
        },
        "mismatches": {
            "run_manifest": run_manifest_mismatches,
            "tensor_manifest": tensor_manifest_mismatches,
        },
        "evidence": {
            "run_dir": str(run_dir.resolve()),
            "tensor_root": str(tensor_root.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "run_manifest_sha256": _sha256(run_dir / "manifest.json"),
            "tensor_manifest_sha256": _sha256(tensor_root / "manifest.json"),
        },
        "launch_gates": {
            "gpu_execution_sentinel_allowed": ready,
            "formal_full_training_allowed": False,
            "followup_seeds_allowed": False,
            "locked_test_allowed": False,
        },
        "result_boundary": "CPU consistency evidence only; no P4 performance claim.",
        "locked_test_accessed": False,
        "formal_performance_claim_ready": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "entity_aligned_rssm_consistency_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "PI-JWM-formal-entity-aligned-rssm-audit-manifest-v1",
        "files": {
            report_path.name: {
                "size_bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        },
        "locked_test_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        run_dir=args.run_dir,
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready_for_gpu_execution_sentinel" else 1


if __name__ == "__main__":
    raise SystemExit(main())
