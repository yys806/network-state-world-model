"""Auditable CPU gates for R6 learning-policy candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .r6_learning_policy import MaskedActorCritic
from .r6_learning_policy_contract import (
    ActionSpec,
    NONLOCKED_SPLITS,
    PolicyIdentity,
    PolicyState,
)
from .r6_learning_policy_training import (
    PolicyTrainingBatch,
    actor_critic_cpu_step,
    ppo_cpu_step,
)
from .r3_preflight_data import load_r3_window, make_explicit_batch
from .r4_gpu_screening import build_validation_windows
from .r5_checkpoint import load_r5_checkpoint
from .r5_protocol import load_r5_protocol


R6_LEARNING_POLICY_PREFLIGHT_SCHEMA = "PIJWM-R6-Learning-Policy-CPU-Preflight-v1"


def validate_nonlocked_splits(splits: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in splits)
    if not normalized:
        raise ValueError("at least one nonlocked split is required")
    if "locked_test" in normalized:
        raise ValueError("locked_test is sealed until R9")
    unknown = sorted(set(normalized).difference(NONLOCKED_SPLITS))
    if unknown:
        raise ValueError(f"unsupported policy splits: {unknown}")
    return normalized


def _action_rows(policy_id: str, state: PolicyState, decision: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_index, identity in enumerate(state.identities):
        rows.append(
            {
                "policy_id": policy_id,
                "scenario_id": identity.scenario_id,
                "seed": identity.seed,
                "slot": identity.slot,
                "split": identity.split,
                "offload_index": int(decision.projected.action.offload_index[batch_index]),
                "rb_index": int(decision.projected.action.rb_index[batch_index]),
                "cpu_total": float(decision.projected.action.cpu_allocation[batch_index].sum()),
                "projection_record_count": len(decision.projected.records),
                "fallback_count": int(decision.projected.fallback_count),
                "legal": True,
            }
        )
    return rows


def _hard_constraint_violations(state: PolicyState, decision: Any) -> int:
    action = decision.projected.action
    violations = int((action.cpu_allocation < -1e-7).any().item())
    for batch_index in range(state.batch_size):
        for node_index in range(state.cpu_capacity.shape[1]):
            selected = state.cpu_task_mask[batch_index] & (
                state.cpu_task_node_index[batch_index] == node_index
            )
            used = action.cpu_allocation[batch_index, selected].sum()
            if used > state.cpu_capacity[batch_index, node_index] + 1e-7:
                violations += 1
    return violations


def run_policy_cpu_smoke(
    state: PolicyState,
    *,
    hidden_dim: int = 32,
    seed: int = 20260808,
) -> dict[str, Any]:
    """Run one finite Actor-Critic and PPO update on an already detached state."""

    if state.latent.requires_grad or state.latent.grad_fn is not None:
        raise ValueError("world-model latent must be detached before policy training")
    spec = ActionSpec(
        offload_count=int(state.offload_mask.shape[1]),
        rb_count=int(state.rb_mask.shape[1]),
        cpu_task_count=int(state.cpu_task_mask.shape[1]),
        offload_noop_index=0,
        rb_noop_index=0,
    )
    torch.manual_seed(int(seed))
    actor_policy = MaskedActorCritic(
        explicit_dim=int(state.explicit.shape[1]),
        latent_dim=int(state.latent.shape[1]),
        hidden_dim=int(hidden_dim),
        spec=spec,
    )
    ppo_policy = MaskedActorCritic(
        explicit_dim=int(state.explicit.shape[1]),
        latent_dim=int(state.latent.shape[1]),
        hidden_dim=int(hidden_dim),
        spec=spec,
    )
    ppo_policy.load_state_dict(actor_policy.state_dict(), strict=True)

    actor_decision = actor_policy.act(state, deterministic=False, seed=int(seed) + 1)
    actor_old = actor_policy.evaluate(state, actor_decision.proposed).log_prob.detach().clone()
    advantage = torch.ones(state.batch_size, dtype=state.explicit.dtype)
    returns = torch.zeros(state.batch_size, dtype=state.explicit.dtype)
    actor_batch = PolicyTrainingBatch(
        state=state,
        action=actor_decision.proposed,
        advantage=advantage,
        returns=returns,
        old_log_prob=actor_old,
    )
    actor_report = actor_critic_cpu_step(
        policy=actor_policy,
        batch=actor_batch,
        optimizer=torch.optim.Adam(actor_policy.parameters(), lr=1e-3),
    )

    ppo_decision = ppo_policy.act(state, deterministic=False, seed=int(seed) + 2)
    ppo_old = ppo_policy.evaluate(state, ppo_decision.proposed).log_prob.detach().clone()
    ppo_batch = PolicyTrainingBatch(
        state=state,
        action=ppo_decision.proposed,
        advantage=advantage,
        returns=returns,
        old_log_prob=ppo_old,
    )
    ppo_report = ppo_cpu_step(
        policy=ppo_policy,
        batch=ppo_batch,
        optimizer=torch.optim.Adam(ppo_policy.parameters(), lr=1e-3),
        clip_epsilon=0.2,
    )
    violation_count = _hard_constraint_violations(state, actor_decision)
    violation_count += _hard_constraint_violations(state, ppo_decision)
    reports = {
        actor_report.objective_id: asdict(actor_report),
        ppo_report.objective_id: asdict(ppo_report),
    }
    action_rows = _action_rows("actor_critic", state, actor_decision)
    action_rows.extend(_action_rows("ppo_clipped", state, ppo_decision))
    ready = (
        violation_count == 0
        and actor_report.policy_parameter_changed
        and ppo_report.policy_parameter_changed
    )
    return {
        "schema_version": R6_LEARNING_POLICY_PREFLIGHT_SCHEMA,
        "r6_learning_policy_cpu_ready": ready,
        "r6_gpu_strategy_training_ready": False,
        "final_method_frozen": False,
        "locked_test_accessed": False,
        "gpu_started": False,
        "world_model_updated": False,
        "hard_constraint_violation_count": violation_count,
        "batch_size": state.batch_size,
        "active_cpu_task_count": int(state.cpu_task_mask.sum().item()),
        "training_target_semantics": "constant numerical smoke only; no policy performance claim",
        "training_reports": reports,
        "action_rows": action_rows,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], default_fields: Sequence[str]) -> None:
    fields = list(default_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _model_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_parameter_sha256(model: torch.nn.Module) -> str:
    """Public read-only hash used by later R6 readiness gates."""

    return _model_parameter_sha256(model)


def _masked_latest_mean(batch: Any, value_key: str, present_key: str) -> torch.Tensor:
    value = batch.history[value_key][:, -1]
    present = batch.history[present_key][:, -1].bool()
    weight = present.unsqueeze(-1).to(value.dtype)
    denominator = weight.sum(dim=1).clamp_min(1.0)
    return (value * weight).sum(dim=1) / denominator


def policy_explicit_from_batch(batch: Any) -> torch.Tensor:
    """Build the frozen 9+7+8 dimensional explicit policy summary."""

    explicit = torch.cat(
        (
            _masked_latest_mean(batch, "physical_node_state", "physical_node_present"),
            _masked_latest_mean(batch, "information_node_state", "information_node_present"),
            _masked_latest_mean(batch, "task_state", "task_present"),
        ),
        dim=-1,
    )
    if explicit.ndim != 2 or explicit.shape[1] != 24:
        raise ValueError("frozen explicit policy summary must be [batch,24]")
    if not torch.isfinite(explicit).all():
        raise ValueError("frozen explicit policy summary must be finite")
    return explicit.detach().clone()


def policy_latent_from_belief(belief: Any) -> torch.Tensor:
    """Expose the actual frozen belief state without assuming one backend type."""

    if hasattr(belief, "deterministic") and hasattr(belief, "stochastic"):
        deterministic = belief.deterministic
        stochastic = belief.stochastic
        if deterministic.shape != stochastic.shape:
            raise ValueError("RSSM deterministic and stochastic states differ in shape")
        latent = torch.cat((deterministic, stochastic), dim=-1)
    elif hasattr(belief, "joint"):
        latent = belief.joint
    else:
        raise ValueError("world-model belief has no supported policy latent state")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
        raise ValueError("policy latent must be a [batch, feature] tensor")
    if not torch.isfinite(latent).all():
        raise ValueError("policy latent must be finite")
    return latent.detach().clone()


def _raw_cpu_contract(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    history = dict(payload["history"])
    static = dict(payload["static"])
    physical = np.asarray(history["physical_node_state"])[-1]
    information_present = np.asarray(history["information_node_present"])[-1].astype(bool)
    cip = np.asarray(static["cip_agent_node_index"]).astype(np.int64)
    capacity = np.zeros(len(cip), dtype=np.float32)
    for information_index, physical_index in enumerate(cip):
        if (
            information_index < len(information_present)
            and information_present[information_index]
            and 0 <= physical_index < physical.shape[0]
        ):
            capacity[information_index] = max(float(physical[physical_index, 5]), 0.0)

    task_present = np.asarray(history["task_present"])[-1].astype(bool)
    task_roles = np.asarray(history["task_information_node_index"])[-1].astype(np.int64)
    task_node = np.full(task_present.shape, -1, dtype=np.int64)
    for task_index in range(len(task_present)):
        if not task_present[task_index]:
            continue
        for role in (2, 1, 0):
            candidate = int(task_roles[task_index, role])
            if 0 <= candidate < len(capacity) and capacity[candidate] > 0.0:
                task_node[task_index] = candidate
                break
    task_mask = task_present & (task_node >= 0)
    return capacity, np.stack((task_mask.astype(np.int64), task_node), axis=0)


def load_real_frozen_policy_state(
    *,
    dataset_root: str | Path,
    evaluation_root: str | Path,
    r5_training_root: str | Path,
    r5_analysis_root: str | Path,
    r6_paired_root: str | Path,
    training_seed: int = 20260803,
) -> tuple[PolicyState, dict[str, Any], dict[str, str]]:
    """Create one real CPU-only policy state from a frozen B checkpoint and nonlocked window."""

    dataset = Path(dataset_root).resolve()
    evaluation = Path(evaluation_root).resolve()
    r5_training = Path(r5_training_root).resolve()
    r5_analysis = Path(r5_analysis_root).resolve()
    r6_paired = Path(r6_paired_root).resolve()
    validate_nonlocked_splits(("validation",))
    required = {
        "dataset_protocol": dataset / "protocol.json",
        "evaluation_protocol": evaluation / "fair_experiment_protocol.json",
        "r5_provenance": r5_training / "input_provenance.json",
        "r5_candidate_freeze": r5_analysis / "candidate_freeze.json",
        "r6_paired_summary": r6_paired / "summary.json",
        "r6_paired_manifest": r6_paired / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R6 learning-policy input is missing: {missing}")
    paired_summary = _read_json(required["r6_paired_summary"])
    if paired_summary.get("locked_test_accessed") is not False:
        raise ValueError("R6 paired input accessed locked_test")
    if paired_summary.get("paired_closed_loop_ready") is not True:
        raise ValueError("R6 paired closed-loop input is not ready")
    candidate_freeze = _read_json(required["r5_candidate_freeze"])
    if candidate_freeze.get("primary_working_candidate") != "B":
        raise ValueError("R5.1 primary working candidate must remain B")
    provenance = _read_json(required["r5_provenance"])
    if provenance.get("locked_test_accessed") is not False:
        raise ValueError("R5 input provenance accessed locked_test")
    protocol = load_r5_protocol(evaluation)
    checkpoint = r5_training / "combinations" / "B" / f"seed_{int(training_seed)}" / "best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen B checkpoint is missing: {checkpoint}")
    loaded = load_r5_checkpoint(
        checkpoint,
        expected_bindings=dict(provenance["bindings"]),
        expected_protocol=protocol,
    )
    model = loaded.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_before = _model_parameter_sha256(model)
    normalization = _read_json(evaluation / "evaluation_normalization_stats.json")
    windows = build_validation_windows(
        dataset,
        split="validation",
        horizons=(1,),
        seed=int(training_seed),
    )
    selected_payload: dict[str, Any] | None = None
    selected_window: Any = None
    raw_capacity: np.ndarray | None = None
    raw_task_contract: np.ndarray | None = None
    for window in windows:
        payload = load_r3_window(window)
        capacity, task_contract = _raw_cpu_contract(payload)
        if int(task_contract[0].sum()) > 0:
            selected_payload = payload
            selected_window = window
            raw_capacity = capacity
            raw_task_contract = task_contract
            break
    if selected_payload is None or raw_capacity is None or raw_task_contract is None:
        raise ValueError("no validation window contains an active CPU task with a valid node capacity")
    batch = make_explicit_batch(selected_payload, normalization, device="cpu")
    torch.manual_seed(int(training_seed))
    with torch.no_grad():
        belief = model.infer_belief(batch)
        latent = policy_latent_from_belief(belief)
        explicit = policy_explicit_from_batch(batch)
    model_after = _model_parameter_sha256(model)
    if model_before != model_after:
        raise ValueError("frozen world-model parameters changed during state generation")
    task_mask = torch.from_numpy(raw_task_contract[0].astype(bool)).unsqueeze(0)
    task_node = torch.from_numpy(raw_task_contract[1]).long().unsqueeze(0)
    identity = PolicyIdentity(
        scenario_id=str(selected_window.trajectory_id),
        seed=int(selected_window.environment_seed),
        slot=int(selected_window.history_end - 1),
        split=str(selected_window.split),
        protocol_fingerprint=_sha256(required["r6_paired_manifest"]),
    )
    state = PolicyState.create(
        explicit=explicit,
        latent=latent,
        offload_mask=torch.ones((1, 1), dtype=torch.bool),
        rb_mask=torch.ones((1, 1), dtype=torch.bool),
        cpu_task_mask=task_mask,
        cpu_capacity=torch.from_numpy(raw_capacity).unsqueeze(0),
        cpu_task_node_index=task_node,
        identities=(identity,),
    )
    bindings = {name: _sha256(path) for name, path in required.items()}
    bindings["frozen_b_checkpoint"] = _sha256(checkpoint)
    audit = {
        "state_source": "real_nonlocked_validation_window_and_frozen_B_checkpoint",
        "trajectory_id": identity.scenario_id,
        "environment_seed": identity.seed,
        "slot": identity.slot,
        "split": identity.split,
        "explicit_dim": int(state.explicit.shape[1]),
        "latent_dim": int(state.latent.shape[1]),
        "cpu_task_slots": int(state.cpu_task_mask.shape[1]),
        "active_cpu_task_count": int(state.cpu_task_mask.sum().item()),
        "cpu_node_slots": int(state.cpu_capacity.shape[1]),
        "active_cpu_node_count": int((state.cpu_capacity > 0).sum().item()),
        "offload_scope": "safe_noop_only_in_CPU_gate",
        "rb_scope": "safe_noop_only_in_CPU_gate",
        "world_model_candidate": "B",
        "world_model_training_seed": int(training_seed),
        "world_model_parameter_sha256_before": model_before,
        "world_model_parameter_sha256_after": model_after,
        "world_model_updated": False,
        "future_target_used_by_policy": False,
        "locked_test_accessed": False,
    }
    return state, audit, bindings


def write_preflight_bundle(
    output_dir: str | Path,
    *,
    summary: Mapping[str, Any],
    bindings: Mapping[str, str],
    state_audit: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> Path:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"preflight output already exists: {output}")
    output.mkdir(parents=True)
    serializable_summary = dict(summary)
    serializable_summary.pop("action_rows", None)
    _write_json(output / "summary.json", serializable_summary)
    _write_json(output / "bindings.json", dict(bindings))
    _write_json(output / "state_audit.json", dict(state_audit))
    _write_json(output / "training_smoke.json", dict(summary.get("training_reports", {})))
    _write_csv(output / "action_audit.csv", action_rows, ("policy_id", "legal"))
    _write_csv(output / "failures.csv", failures, ("stage", "error_type", "error_message"))
    (output / "README.md").write_text(
        "# PI-JWM R6 learning-policy CPU preflight\n\n"
        "This bundle validates interfaces, masks, projection, finite gradients and frozen-world-model isolation only.\n",
        encoding="utf-8",
    )
    managed = sorted(path for path in output.iterdir() if path.is_file())
    files = {
        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in managed
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": R6_LEARNING_POLICY_PREFLIGHT_SCHEMA,
            "manifest_entry_count": len(files),
            "locked_test_accessed": False,
            "files": files,
        },
    )
    return output
