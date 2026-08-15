"""Formal R6 GPU training runner for PI-JWM joint candidate policies.

The runner is deliberately thin: it reuses the audited R6 AirFogSim adapter,
frozen Graph-RSSM checkpoint, reward protocol, candidate contract and policy
objectives.  It supports a non-formal 2,000-step smoke and the frozen 18-run
matrix.  Locked-test is rejected unconditionally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))
if str(CODE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "scripts"))

import run_r6_joint_policy_gpu_readiness as preflight  # noqa: E402
from pi_jwm.airfogsim_teacher_tensor_v3 import contract_from_dict  # noqa: E402
from pi_jwm.r3_preflight_data import make_explicit_batch  # noqa: E402
from pi_jwm.r6_gpu_training_protocol import (  # noqa: E402
    CheckpointMetric,
    ValidationCheckpointGate,
    build_default_gpu_training_protocol,
)
from pi_jwm.r6_joint_policy import (  # noqa: E402
    CandidateMaskedActorCritic,
    JointPolicyState,
    JointPolicyTrainingBatch,
    joint_actor_critic_step,
    joint_ppo_step,
)
from pi_jwm.r6_learning_policy_contract import PolicyIdentity  # noqa: E402
from pi_jwm.r6_learning_policy_preflight import (  # noqa: E402
    policy_explicit_from_batch,
    policy_latent_from_belief,
)
from pi_jwm.r6_online_observation import (  # noqa: E402
    OnlineDualGraphHistory,
    build_online_teacher_arrays,
    make_online_inference_payload,
)
from pi_jwm.r6_rollout import JointRollout, JointTransition, compute_gae  # noqa: E402


DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_gpu_training_v1"
DEFAULT_SMOKE_OUTPUT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_gpu_training_smoke_v1"


def _args() -> argparse.Namespace:
    protocol = build_default_gpu_training_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-run-id", default="actor_critic__explicit_latent__seed_20260803")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-output-dir", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-environment-steps", type=int, default=protocol.max_environment_steps)
    parser.add_argument("--rollout-length", type=int, default=protocol.rollout_length)
    parser.add_argument("--minibatch-size", type=int, default=protocol.minibatch_size)
    parser.add_argument("--evaluation-interval", type=int, default=protocol.evaluation_interval)
    parser.add_argument("--validation-trajectory-limit", type=int, default=12)
    parser.add_argument(
        "--validation-step-limit",
        type=int,
        default=protocol.validation_step_limit,
    )
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--sumo-port", type=int, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _move_state(state: JointPolicyState, device: torch.device) -> JointPolicyState:
    return JointPolicyState.create(
        explicit=state.explicit.to(device),
        latent=state.latent.to(device),
        candidate_descriptors=state.candidate_descriptors.to(device),
        candidate_mask=state.candidate_mask.to(device),
        identities=state.identities,
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _ensure_sumolib_lane_compatibility() -> str:
    """Expose the new read-only permission accessor on SUMO 1.12.

    AirFogSim calls ``Lane.getPermissions``.  SUMO 1.12 stores the same
    permission tuple as ``Lane._allowed`` and exposes ``Lane.allows`` but does
    not yet provide that accessor.  The shim is therefore semantic-preserving
    and is only installed when the public method is absent.
    """

    from sumolib.net.lane import Lane

    if hasattr(Lane, "getPermissions"):
        return "native_getPermissions"

    def get_permissions(self: Any) -> tuple[str, ...]:
        value = getattr(self, "_allowed", ())
        return tuple(value or ())

    setattr(Lane, "getPermissions", get_permissions)
    return "compat_getPermissions_from_allowed"


def _formal_specs(split: str) -> list[Any]:
    specs = [row for row in preflight.build_formal_trajectory_specs() if row.split == split]
    if not specs:
        raise ValueError(f"no specs for split {split}")
    if any(row.split == "locked_test" for row in specs):
        raise PermissionError("locked_test is sealed")
    return specs


class _OnlineAirFogSimRecorder:
    """Capture the post-action AirFogSim state used by the next decision."""

    def __init__(self, trajectory_id: str, *, history_steps: int = 8) -> None:
        import airfogsim_cross_graph_evidence_closure as evidence_runtime
        import airfogsim_strict_dual_graph_preflight as strict_preflight
        from airfogsim.scheduler import TaskScheduler
        from export_dataset_demo import (
            active_link_counts,
            channel_csi_mean,
            channel_rate_sum,
            node_rows,
        )

        self.history = OnlineDualGraphHistory(
            trajectory_id,
            max_frames=int(history_steps),
        )
        self.evidence_runtime = evidence_runtime
        self.strict_preflight = strict_preflight
        self.TaskScheduler = TaskScheduler
        self.active_link_counts = active_link_counts
        self.channel_csi_mean = channel_csi_mean
        self.channel_rate_sum = channel_rate_sum
        self.node_rows = node_rows
        self.total_frames = 0
        self._transfer_cursor = 0
        self._offload_cursor = 0
        self._return_cursor = 0
        self._rb_cursor = 0
        self._cpu_cursor = 0
        self.capture_counts: Counter[str] = Counter()

    def active_links(self, env: Any) -> Any:
        return self.active_link_counts(env)

    def capture_after_step(self, env: Any, algorithm: Any, active: Any) -> None:
        trajectory_id = self.history.trajectory_id
        current_time = float(env.simulation_time)
        nodes = [
            self.strict_preflight._node_record(row, trajectory_id)
            for row in self.node_rows(env)
        ]
        edges = [
            self.strict_preflight._physical_edge_record(row, trajectory_id)
            for row in self.evidence_runtime.all_directed_link_rows(
                env,
                active,
                rate_reader=self.channel_rate_sum,
                csi_reader=self.channel_csi_mean,
            )
        ]
        tasks = [
            self.strict_preflight._task_record(task, trajectory_id, current_time)
            for task in self.strict_preflight.iter_airfogsim_tasks(env.task_manager)
        ]
        dags = self.strict_preflight.normalize_airfogsim_dags(
            self.TaskScheduler.getAllTaskDAGs(env),
            trajectory_id=trajectory_id,
            step=self.total_frames + 1,
            time_value=current_time,
        )
        transfer_rows = list(env.pi_jwm_transfer_events[self._transfer_cursor :])
        offload_rows = list(algorithm.offload_rows[self._offload_cursor :])
        return_rows = list(algorithm.return_rows[self._return_cursor :])
        rb_rows = list(algorithm.rb_rows[self._rb_cursor :])
        cpu_rows = list(algorithm.cpu_rows[self._cpu_cursor :])
        self._transfer_cursor = len(env.pi_jwm_transfer_events)
        self._offload_cursor = len(algorithm.offload_rows)
        self._return_cursor = len(algorithm.return_rows)
        self._rb_cursor = len(algorithm.rb_rows)
        self._cpu_cursor = len(algorithm.cpu_rows)
        self.capture_counts.update(
            {
                "frames": 1,
                "transfer_events": len(transfer_rows),
                "offload_actions": len(offload_rows),
                "return_actions": len(return_rows),
                "rb_actions": len(rb_rows),
                "cpu_actions": len(cpu_rows),
            }
        )
        self.history.append_frame(
            physical_nodes=nodes,
            physical_edges=edges,
            task_records=tasks,
            dag_edges=dags,
            transfer_events=transfer_rows,
            offload_actions=offload_rows,
            return_actions=return_rows,
            rb_actions=rb_rows,
            cpu_actions=cpu_rows,
        )
        self.total_frames += 1


def _warm_online_history(
    env: Any,
    algorithm: Any,
    recorder: _OnlineAirFogSimRecorder,
    *,
    history_steps: int = 8,
) -> None:
    while recorder.total_frames < int(history_steps) and not env.isDone():
        algorithm.scheduleStep(env)
        active = recorder.active_links(env)
        env.step()
        recorder.capture_after_step(env, algorithm, active)
    if recorder.total_frames != int(history_steps):
        raise RuntimeError("AirFogSim ended before the online history was warm")


def _state_from_online_history(
    *,
    prepared: Any,
    recorder: _OnlineAirFogSimRecorder,
    spec: Any,
    model: Any,
    normalization: dict[str, Any],
    tensor_contract: Any,
    protocol_fingerprint: str,
) -> JointPolicyState:
    source_graph = recorder.history.build_source_graph()
    arrays, _ = build_online_teacher_arrays(source_graph, contract=tensor_contract)
    payload = make_online_inference_payload(
        arrays,
        trajectory_id=spec.trajectory_id,
        environment_seed=int(spec.seed),
        split=spec.split,
        history_steps=8,
    )
    batch = make_explicit_batch(payload, normalization, device="cpu")
    if batch.metadata["state_source"] != "online_airfogsim_strict_dual_graph":
        raise RuntimeError("R6 policy state is not sourced from live AirFogSim")
    with torch.no_grad():
        belief = model.infer_belief(batch)
        explicit = policy_explicit_from_batch(batch)
        latent = policy_latent_from_belief(belief)
    descriptors, mask = prepared.candidates.padded_descriptors(max_candidates=6)
    identity = PolicyIdentity(
        scenario_id=spec.trajectory_id,
        seed=int(spec.seed),
        slot=int(recorder.total_frames - 1),
        split=spec.split,
        protocol_fingerprint=protocol_fingerprint,
    )
    return JointPolicyState.create(
        explicit=explicit,
        latent=latent,
        candidate_descriptors=torch.from_numpy(descriptors).unsqueeze(0),
        candidate_mask=torch.from_numpy(mask).unsqueeze(0),
        identities=(identity,),
    )


def _collect_trajectory(
    *, spec: Any, model: Any, policy: CandidateMaskedActorCritic,
    reward_protocol: Any, dataset_root: Path, evaluation_root: Path,
    protocol_fingerprint: str, device: torch.device, max_steps: int,
    stochastic: bool, action_seed: int, tensor_contract: Any,
    normalization: dict[str, Any],
) -> tuple[list[JointPolicyState], list[JointTransition], int, dict[str, int]]:
    tensor_path = dataset_root / f"seed_{int(spec.seed):03d}" / "trajectory_tensors.npz"
    with np.load(tensor_path, allow_pickle=False) as arrays:
        times = np.asarray(arrays["time"], dtype=np.float64)
    env, algorithm = preflight._make_formal_env(spec, max_time=float(times[-1]) + 0.1)
    states: list[JointPolicyState] = []
    transitions: list[JointTransition] = []
    used = 0
    try:
        recorder = _OnlineAirFogSimRecorder(spec.trajectory_id, history_steps=8)
        _warm_online_history(env, algorithm, recorder, history_steps=8)
        current_prepared = preflight.prepare_joint_action_step(
            env, algorithm, scenario_id=spec.trajectory_id, seed=int(spec.seed),
            slot=int(recorder.total_frames - 1), split=spec.split, max_candidates=6,
        )
        current_state_cpu = _state_from_online_history(
            prepared=current_prepared, recorder=recorder, spec=spec, model=model,
            normalization=normalization, tensor_contract=tensor_contract,
            protocol_fingerprint=protocol_fingerprint,
        )
        for slot in range(7, len(times) - 1):
            if used >= int(max_steps) or env.isDone():
                break
            state = _move_state(current_state_cpu, device)
            decision = policy.act(
                state, deterministic=not stochastic,
                seed=int(action_seed) + int(slot) + int(spec.seed),
            )
            candidate_index = int(decision.candidate_index[0].item())
            candidate = current_prepared.candidates.candidates[candidate_index]
            throughput_before = float(env.channel.get("data_size", 0.0))
            energy_before = preflight._remaining_energy(env)
            record = preflight.apply_prepared_candidate(
                env, algorithm, current_prepared, candidate_index=candidate_index,
            )
            active = recorder.active_links(env)
            env.step()
            recorder.capture_after_step(env, algorithm, active)
            facts = preflight._feedback_facts(
                env, algorithm, throughput_before=throughput_before,
                energy_before=energy_before,
                hard_violation_count=int(record.hard_violation_count),
            )
            reward = reward_protocol.score(facts)
            if not reward.valid:
                raise ValueError(f"invalid reward at {spec.trajectory_id}:{slot}: {reward.invalid_reason}")
            terminated = bool(env.isDone())
            next_value = 0.0
            if not terminated and slot + 1 < len(times):
                next_prepared = preflight.prepare_joint_action_step(
                    env, algorithm, scenario_id=spec.trajectory_id, seed=int(spec.seed),
                    slot=int(recorder.total_frames - 1), split=spec.split, max_candidates=6,
                )
                next_state_cpu = _state_from_online_history(
                    prepared=next_prepared, recorder=recorder, spec=spec, model=model,
                    normalization=normalization, tensor_contract=tensor_contract,
                    protocol_fingerprint=protocol_fingerprint,
                )
                next_state = _move_state(next_state_cpu, device)
                with torch.no_grad():
                    next_value = float(policy(next_state).value[0].item())
            else:
                next_state_cpu, next_prepared = current_state_cpu, current_prepared
            states.append(state)
            transitions.append(
                JointTransition.create(
                    identity=state.identities[0], candidate_id=candidate.candidate_id,
                    candidate_index=candidate_index,
                    old_log_prob=float(decision.log_prob[0].detach().item()),
                    value=float(decision.value[0].detach().item()),
                    next_value=next_value, reward=reward,
                    terminated=terminated, truncated=False,
                )
            )
            used += 1
            current_state_cpu, current_prepared = next_state_cpu, next_prepared
            if terminated:
                break
    finally:
        env.close()
    if not transitions:
        raise RuntimeError(f"no transitions collected for {spec.trajectory_id}")
    return states, transitions, used, dict(recorder.capture_counts)


def _batch_from_trajectory(
    states: list[JointPolicyState], transitions: list[JointTransition], device: torch.device,
) -> JointPolicyTrainingBatch:
    rollout = JointRollout.create(transitions)
    gae = compute_gae(rollout, gamma=0.99, gae_lambda=0.95)
    state = JointPolicyState.create(
        explicit=torch.cat([row.explicit for row in states], dim=0),
        latent=torch.cat([row.latent for row in states], dim=0),
        candidate_descriptors=torch.cat([row.candidate_descriptors for row in states], dim=0),
        candidate_mask=torch.cat([row.candidate_mask for row in states], dim=0),
        identities=tuple(identity for row in states for identity in row.identities),
    )
    return JointPolicyTrainingBatch(
        state=state,
        candidate_index=torch.tensor([row.candidate_index for row in transitions], device=device),
        advantage=torch.as_tensor(gae.advantage, dtype=torch.float32, device=device),
        returns=torch.as_tensor(gae.returns, dtype=torch.float32, device=device),
        old_log_prob=torch.tensor([row.old_log_prob for row in transitions], dtype=torch.float32, device=device),
    )


def _build_policy(first_state: JointPolicyState, *, mode: str, hidden_dim: int, device: torch.device) -> CandidateMaskedActorCritic:
    policy = CandidateMaskedActorCritic(
        int(first_state.explicit.shape[1]), int(first_state.latent.shape[1]),
        int(first_state.candidate_descriptors.shape[2]), hidden_dim=int(hidden_dim),
        state_mode=mode,
    )
    return policy.to(device)


def _explicit_state_sha256(state: JointPolicyState) -> str:
    value = state.explicit.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _verify_checkpoint_reload(
    checkpoint: Path,
    *,
    reference: CandidateMaskedActorCritic,
    first_state: JointPolicyState,
    mode: str,
    hidden_dim: int,
    device: torch.device,
) -> bool:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded = _build_policy(first_state, mode=mode, hidden_dim=hidden_dim, device=device)
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    return all(
        torch.equal(reference.state_dict()[key], reloaded.state_dict()[key])
        for key in reference.state_dict()
    )


def _evaluate_policy(
    *, policy: CandidateMaskedActorCritic, model: Any, reward_protocol: Any,
    dataset_root: Path, evaluation_root: Path, protocol_fingerprint: str,
    device: torch.device, trajectory_limit: int, action_seed: int,
    tensor_contract: Any, normalization: dict[str, Any], validation_step_limit: int,
) -> dict[str, float | int]:
    validation_specs = _formal_specs("validation")[: int(trajectory_limit)]
    reward_total = 0.0
    step_count = 0
    success_count = 0
    failure_count = 0
    delay_sum = 0.0
    hard_violations = 0
    policy.eval()
    with torch.no_grad():
        for spec in validation_specs:
            _, transitions, used, _ = _collect_trajectory(
                spec=spec, model=model, policy=policy, reward_protocol=reward_protocol,
                dataset_root=dataset_root, evaluation_root=evaluation_root,
                protocol_fingerprint=protocol_fingerprint, device=device,
                max_steps=int(validation_step_limit), stochastic=False,
                action_seed=action_seed, tensor_contract=tensor_contract,
                normalization=normalization,
            )
            step_count += int(used)
            for transition in transitions:
                facts = dict(transition.reward.raw_facts)
                reward_total += float(transition.reward.total_reward)
                success_count += int(facts["on_time_completion_count"])
                failure_count += int(facts["failure_count"])
                delay_sum += float(facts["completed_delay_sum"])
                hard_violations += int(facts["hard_violation_count"])
    policy.train()
    service_count = success_count + failure_count
    return {
        "validation_return": reward_total / max(step_count, 1),
        "on_time_completion_rate": success_count / max(service_count, 1),
        "mean_latency": delay_sum / max(success_count, 1),
        "hard_violation_count": hard_violations,
        "validation_step_count": step_count,
        "validation_trajectory_count": len(validation_specs),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _save_resume_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _run_one(
    *, run: Any, args: argparse.Namespace, model: Any, reward_protocol: Any,
    dataset_root: Path, evaluation_root: Path, protocol_fingerprint: str,
    output_dir: Path, device: torch.device, smoke: bool = False,
) -> dict[str, Any]:
    _seed_everything(run.seed)
    specs = _formal_specs("train")
    tensor_contract = contract_from_dict(_read_json(dataset_root / "tensor_contract.json"))
    normalization = _read_json(evaluation_root / "evaluation_normalization_stats.json")
    first_spec = specs[0]
    first_tensor = dataset_root / f"seed_{int(first_spec.seed):03d}" / "trajectory_tensors.npz"
    with np.load(first_tensor, allow_pickle=False) as arrays:
        first_times = np.asarray(arrays["time"], dtype=np.float64)
    first_env, first_algorithm = preflight._make_formal_env(
        first_spec, max_time=float(first_times[-1]) + 0.1
    )
    try:
        first_recorder = _OnlineAirFogSimRecorder(first_spec.trajectory_id, history_steps=8)
        _warm_online_history(first_env, first_algorithm, first_recorder, history_steps=8)
        first_prepared = preflight.prepare_joint_action_step(
            first_env, first_algorithm, scenario_id=first_spec.trajectory_id,
            seed=int(first_spec.seed), slot=7, split=first_spec.split, max_candidates=6,
        )
        first_state = _state_from_online_history(
            prepared=first_prepared, recorder=first_recorder, spec=first_spec,
            model=model, normalization=normalization, tensor_contract=tensor_contract,
            protocol_fingerprint=protocol_fingerprint,
        )
    finally:
        first_env.close()
    policy = _build_policy(first_state, mode=run.state_mode, hidden_dim=args.hidden_dim, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(run.learning_rate))
    gate = ValidationCheckpointGate(patience=5)
    run_root = output_dir / run.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(asdict(run), indent=2, sort_keys=True), encoding="utf-8")
    resume_checkpoint = run_root / "resume_checkpoint.pt"
    total_steps = 0
    update_count = 0
    reports: list[dict[str, Any]] = []
    validation_reports: list[dict[str, Any]] = []
    pending_states: list[JointPolicyState] = []
    pending_transitions: list[JointTransition] = []
    candidate_selection_counts: Counter[str] = Counter()
    explicit_state_hashes: set[str] = set()
    hard_violation_count = 0
    online_capture_counts: Counter[str] = Counter()
    prior_distinct_explicit_state_count = 0
    spec_index = 0
    target_steps = int(args.max_environment_steps)
    if smoke:
        target_steps = min(target_steps, 2000)
    resumed_from_environment_step = 0
    elapsed_before_resume = 0.0
    if resume_checkpoint.is_file():
        resume = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if str(resume.get("run_id")) != run.run_id:
            raise ValueError("resume checkpoint run_id mismatch")
        if resume.get("state_source") != "online_airfogsim_strict_dual_graph":
            raise ValueError("resume checkpoint does not use online AirFogSim state")
        policy.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        total_steps = int(resume["environment_steps"])
        resumed_from_environment_step = total_steps
        update_count = int(resume["update_count"])
        spec_index = int(resume["spec_index"])
        next_evaluation = int(resume["next_evaluation"])
        reports = list(resume.get("reports", []))
        validation_reports = list(resume.get("validation_reports", []))
        candidate_selection_counts.update(resume.get("candidate_selection_counts", {}))
        hard_violation_count = int(resume.get("hard_violation_count", 0))
        online_capture_counts.update(resume.get("online_capture_counts", {}))
        prior_distinct_explicit_state_count = int(
            resume.get("distinct_explicit_state_count", 0)
        )
        elapsed_before_resume = float(resume.get("elapsed_seconds", 0.0))
        gate.no_improvement_count = int(resume.get("gate_no_improvement_count", 0))
        gate.last_environment_step = int(resume.get("gate_last_environment_step", 0))
        best = resume.get("gate_best")
        gate.best = None if best is None else CheckpointMetric(**best)
    start = time.time()
    if resumed_from_environment_step == 0:
        next_evaluation = int(args.evaluation_interval)
    stop_reason = "max_environment_steps"
    progress_path = run_root / "progress.jsonl"
    while total_steps < target_steps:
        spec = specs[spec_index % len(specs)]
        spec_index += 1
        states, transitions, used, capture_counts = _collect_trajectory(
            spec=spec, model=model, policy=policy, reward_protocol=reward_protocol,
            dataset_root=dataset_root, evaluation_root=evaluation_root,
            protocol_fingerprint=protocol_fingerprint, device=device,
            max_steps=min(int(args.rollout_length), target_steps - total_steps),
            stochastic=True, action_seed=run.seed + total_steps,
            tensor_contract=tensor_contract, normalization=normalization,
        )
        pending_states.extend(states)
        pending_transitions.extend(transitions)
        explicit_state_hashes.update(_explicit_state_sha256(state) for state in states)
        candidate_selection_counts.update(row.candidate_id for row in transitions)
        hard_violation_count += sum(
            int(row.reward.raw_facts["hard_violation_count"])
            for row in transitions
        )
        online_capture_counts.update(capture_counts)
        total_steps += int(used)
        if (
            len(pending_transitions) >= int(args.minibatch_size)
            or (total_steps >= target_steps and bool(pending_transitions))
        ):
            batch = _batch_from_trajectory(pending_states, pending_transitions, device)
            if run.method_id == "actor_critic":
                report = joint_actor_critic_step(
                    policy=policy, batch=batch, optimizer=optimizer,
                    value_coef=run.value_coef, entropy_coef=run.entropy_coef,
                    max_grad_norm=run.max_grad_norm,
                )
            else:
                for _ in range(int(run.ppo_epochs)):
                    report = joint_ppo_step(
                        policy=policy, batch=batch, optimizer=optimizer,
                        clip_epsilon=run.clip_epsilon, value_coef=run.value_coef,
                        entropy_coef=run.entropy_coef, max_grad_norm=run.max_grad_norm,
                    )
            reports.append({"environment_step": total_steps, **asdict(report)})
            update_count += 1
            pending_states.clear()
            pending_transitions.clear()
        _append_jsonl(
            progress_path,
            {
                "environment_step": total_steps,
                "update_count": update_count,
                "trajectory_id": spec.trajectory_id,
                "elapsed_seconds": time.time() - start,
            },
        )
        should_stop_after_checkpoint = False
        if not smoke and total_steps >= next_evaluation:
            metrics = _evaluate_policy(
                policy=policy, model=model, reward_protocol=reward_protocol,
                dataset_root=dataset_root, evaluation_root=evaluation_root,
                protocol_fingerprint=protocol_fingerprint, device=device,
                trajectory_limit=args.validation_trajectory_limit,
                action_seed=run.seed + total_steps,
                tensor_contract=tensor_contract, normalization=normalization,
                validation_step_limit=args.validation_step_limit,
            )
            metric = CheckpointMetric(
                environment_step=total_steps,
                validation_return=float(metrics["validation_return"]),
                on_time_completion_rate=float(metrics["on_time_completion_rate"]),
                mean_latency=float(metrics["mean_latency"]),
                hard_violation_count=int(metrics["hard_violation_count"]),
            )
            update = gate.update(metric)
            validation_row = {
                "environment_step": total_steps,
                **metrics,
                **asdict(update),
            }
            validation_reports.append(validation_row)
            _append_jsonl(run_root / "validation.jsonl", validation_row)
            if update.improved:
                torch.save(
                    {
                        "state_dict": policy.state_dict(),
                        "run": asdict(run),
                        "environment_steps": total_steps,
                        "validation": validation_row,
                    },
                    run_root / "best_checkpoint.pt",
                )
            next_evaluation += int(args.evaluation_interval)
            if update.should_stop:
                stop_reason = "validation_patience"
                should_stop_after_checkpoint = True
        if pending_states or pending_transitions:
            raise RuntimeError(
                "resume checkpoint requires rollout_length >= minibatch_size"
            )
        _save_resume_checkpoint(
            resume_checkpoint,
            {
                "run_id": run.run_id,
                "state_source": "online_airfogsim_strict_dual_graph",
                "state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "environment_steps": total_steps,
                "update_count": update_count,
                "spec_index": spec_index,
                "next_evaluation": next_evaluation,
                "reports": reports,
                "validation_reports": validation_reports,
                "candidate_selection_counts": dict(candidate_selection_counts),
                "hard_violation_count": hard_violation_count,
                "online_capture_counts": dict(online_capture_counts),
                "distinct_explicit_state_count": (
                    prior_distinct_explicit_state_count + len(explicit_state_hashes)
                ),
                "gate_no_improvement_count": gate.no_improvement_count,
                "gate_last_environment_step": gate.last_environment_step,
                "gate_best": None if gate.best is None else asdict(gate.best),
                "elapsed_seconds": elapsed_before_resume + time.time() - start,
            },
        )
        if should_stop_after_checkpoint:
            break
        if smoke and total_steps >= target_steps:
            break
    checkpoint = run_root / "last_checkpoint.pt"
    torch.save({"state_dict": policy.state_dict(), "run": asdict(run), "environment_steps": total_steps}, checkpoint)
    checkpoint_reload_verified = _verify_checkpoint_reload(
        checkpoint,
        reference=policy,
        first_state=first_state,
        mode=run.state_mode,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    if not checkpoint_reload_verified:
        raise RuntimeError("R6 policy checkpoint reload differs from the trained policy")
    summary = {
        "run_id": run.run_id, "formal": not smoke, "status": "complete",
        "smoke": bool(smoke), "environment_steps": total_steps,
        "update_count": update_count, "checkpoint": str(checkpoint),
        "reports": reports, "validation_reports": validation_reports,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_before_resume + time.time() - start,
        "locked_test_accessed": False, "world_model_updated": False,
        "device": str(device),
        "state_source": "online_airfogsim_strict_dual_graph",
        "candidate_selection_counts": dict(sorted(candidate_selection_counts.items())),
        "nondefault_selection_count": int(
            sum(value for key, value in candidate_selection_counts.items() if key != "airfogsim_default")
        ),
        "distinct_explicit_state_count": (
            prior_distinct_explicit_state_count + len(explicit_state_hashes)
        ),
        "hard_violation_count": int(hard_violation_count),
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "resumed_from_environment_step": resumed_from_environment_step,
        "online_capture_counts": dict(sorted(online_capture_counts.items())),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = _args()
    if args.sumo_port is not None:
        if not 1024 <= int(args.sumo_port) <= 65535:
            raise ValueError("SUMO port must be in [1024, 65535]")
        os.environ["PIJWM_SUMO_PORT"] = str(int(args.sumo_port))
    protocol = build_default_gpu_training_protocol()
    sumolib_compatibility = _ensure_sumolib_lane_compatibility()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R6 GPU training")
    device = torch.device(args.device)
    if args.smoke:
        matches = [row for row in protocol.formal_runs() if row.run_id == args.smoke_run_id]
        if len(matches) != 1:
            raise ValueError(f"unknown smoke run id: {args.smoke_run_id}")
        runs = matches
        output_dir = args.smoke_output_dir
    else:
        runs = protocol.formal_runs()
        output_dir = args.output_dir
    if args.run_id is not None:
        runs = tuple(run for run in runs if run.run_id == str(args.run_id))
        if len(runs) != 1:
            raise ValueError(f"unknown or ambiguous formal run_id: {args.run_id}")
    if args.seed_limit is not None:
        runs = tuple(run for run in runs if run.seed in {20260803, 20260804, 20260805} and run.seed <= int(args.seed_limit))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, bindings = preflight._load_world_model(
        argparse.Namespace(
            r5_training_root=preflight.DEFAULT_R5_TRAINING_ROOT,
            r5_analysis_root=preflight.DEFAULT_R5_ANALYSIS_ROOT,
            r6_paired_root=preflight.DEFAULT_R6_PAIRED_ROOT,
            dataset_root=preflight.DEFAULT_DATASET_ROOT,
            evaluation_root=preflight.DEFAULT_EVALUATION_ROOT,
            reward_root=preflight.DEFAULT_REWARD_ROOT,
            world_model_seed=20260803,
        )
    )
    reward_protocol = preflight.ServiceFirstRewardProtocol(
        preflight.RewardScale.from_mapping(preflight._read_json(preflight.DEFAULT_REWARD_ROOT / "reward_scale.json"))
    )
    world_model_before = preflight.model_parameter_sha256(model)
    old_cwd = Path.cwd()
    try:
        os.chdir(preflight.AIRFOGSIM_EXAMPLES)
        rows = []
        for run in runs:
            try:
                rows.append(_run_one(
                    run=run, args=args, model=model, reward_protocol=reward_protocol,
                    dataset_root=preflight.DEFAULT_DATASET_ROOT,
                    evaluation_root=preflight.DEFAULT_EVALUATION_ROOT,
                    protocol_fingerprint=bindings["r6_paired_manifest"],
                    output_dir=output_dir, device=device, smoke=bool(args.smoke),
                ))
            except Exception as exc:
                if args.smoke:
                    raise
                rows.append({"run_id": run.run_id, "formal": True, "status": "failed", "error": repr(exc), "locked_test_accessed": False})
        records_path = (
            output_dir / "run_records.json"
            if args.run_id is None
            else output_dir / str(args.run_id) / "run_record.json"
        )
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        world_model_after = preflight.model_parameter_sha256(model)
        if world_model_after != world_model_before:
            raise RuntimeError("frozen world model changed during R6 training")
        print(json.dumps({"output_dir": str(output_dir.resolve()), "sumolib_compatibility": sumolib_compatibility, "world_model_unchanged": True, "records": rows}, ensure_ascii=False))
    finally:
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
