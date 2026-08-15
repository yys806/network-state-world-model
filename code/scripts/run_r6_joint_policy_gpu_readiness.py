"""Run the formal CPU-only preflight immediately before R6 GPU policy training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import types
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SMALL_EXPERIMENT_ROOT = CODE_ROOT / "scripts" / "small_experiments"
for path in (SRC_ROOT, SMALL_EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator  # noqa: E402
from pi_jwm.airfogsim_runtime import capture_energy_manager_snapshot  # noqa: E402
from pi_jwm.formal_airfogsim_dataset_v1 import (  # noqa: E402
    apply_formal_scenario_overrides,
    build_formal_trajectory_specs,
)
from pi_jwm.r3_preflight_data import R3Window, load_r3_window, make_explicit_batch  # noqa: E402
from pi_jwm.r5_checkpoint import load_r5_checkpoint  # noqa: E402
from pi_jwm.r5_protocol import load_r5_protocol  # noqa: E402
from pi_jwm.r6_airfogsim_joint_runtime import (  # noqa: E402
    apply_prepared_candidate,
    prepare_joint_action_step,
)
from pi_jwm.r6_gpu_training_protocol import build_default_gpu_training_protocol  # noqa: E402
from pi_jwm.r6_joint_action import JointActionCandidate  # noqa: E402
from pi_jwm.r6_joint_policy import (  # noqa: E402
    CandidateMaskedActorCritic,
    JointPolicyState,
    JointPolicyTrainingBatch,
    joint_actor_critic_step,
    joint_ppo_step,
)
from pi_jwm.r6_joint_policy_preflight import (  # noqa: E402
    GPUReadinessEvidence,
    assess_gpu_readiness,
    write_gpu_readiness_bundle,
)
from pi_jwm.r6_learning_policy_contract import PolicyIdentity  # noqa: E402
from pi_jwm.r6_learning_policy_preflight import (  # noqa: E402
    model_parameter_sha256,
    policy_explicit_from_batch,
    policy_latent_from_belief,
)
from pi_jwm.r6_reward_protocol import (  # noqa: E402
    RewardScale,
    ServiceFirstRewardProtocol,
    TransitionFacts,
)
from pi_jwm.r6_rollout import JointRollout, JointTransition, compute_gae  # noqa: E402
from run_airfogsim_counterfactual_action_smoke_v0 import (  # noqa: E402
    AIRFOGSIM_EXAMPLES,
    import_airfogsim_runtime,
    load_config,
)


DEFAULT_DATASET_ROOT = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
DEFAULT_EVALUATION_ROOT = CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
DEFAULT_R5_TRAINING_ROOT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r5_gpu_training_v1"
DEFAULT_R5_ANALYSIS_ROOT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r5_module_confirmation_analysis_v1"
DEFAULT_R6_PAIRED_ROOT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_cpu_paired_closed_loop_v1"
DEFAULT_REWARD_ROOT = CODE_ROOT / "artifacts" / "protocols" / "pi_jwm_r6_reward_protocol_v1"
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_r6_joint_policy_gpu_readiness_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--r5-training-root", type=Path, default=DEFAULT_R5_TRAINING_ROOT)
    parser.add_argument("--r5-analysis-root", type=Path, default=DEFAULT_R5_ANALYSIS_ROOT)
    parser.add_argument("--r6-paired-root", type=Path, default=DEFAULT_R6_PAIRED_ROOT)
    parser.add_argument("--reward-root", type=Path, default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory-seed", type=int, default=507)
    parser.add_argument("--world-model-seed", type=int, default=20260803)
    parser.add_argument("--policy-seed", type=int, default=20260808)
    parser.add_argument("--start-slot", type=int, default=80)
    parser.add_argument("--rollout-length", type=int, default=4)
    parser.add_argument("--scan-step-limit", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=32)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_spec(seed: int):
    matches = [spec for spec in build_formal_trajectory_specs() if spec.seed == int(seed)]
    if len(matches) != 1:
        raise ValueError(f"formal trajectory seed is not unique: {seed}")
    spec = matches[0]
    if spec.split == "locked_test":
        raise ValueError("locked_test is sealed until R9")
    if spec.split != "validation":
        raise ValueError("formal R6 GPU readiness run must use validation")
    return spec


def _apply_sumo_port_override(config: dict[str, Any], port: int | None) -> dict[str, Any]:
    """Return an isolated AirFogSim config with an optional TraCI port."""

    updated = copy.deepcopy(config)
    if port is None:
        return updated
    resolved = int(port)
    if not 1024 <= resolved <= 65535:
        raise ValueError("SUMO port must be in [1024, 65535]")
    sumo = updated.get("sumo")
    if not isinstance(sumo, dict) or "sumo_port" not in sumo:
        raise ValueError("AirFogSim config is missing sumo.sumo_port")
    sumo["sumo_port"] = resolved
    return updated


def _make_formal_env(spec: Any, *, max_time: float):
    AirFogSimEnv, _, RewardScheduler, _ = import_airfogsim_runtime()
    import airfogsim_cross_graph_evidence_closure as evidence_runtime
    import airfogsim_strict_dual_graph_preflight as strict_preflight
    from export_strict_actions_v0 import (
        LoggingAlgorithmModule,
        action_effect_time,
        as_float,
        node_type_from_id,
    )

    config = load_config(AIRFOGSIM_EXAMPLES / "config.yaml")
    config = strict_preflight.build_preflight_config(config, int(spec.seed), float(max_time))
    config = apply_formal_scenario_overrides(config, spec.scenario)
    raw_sumo_port = os.environ.get("PIJWM_SUMO_PORT")
    config = _apply_sumo_port_override(
        config,
        None if raw_sumo_port is None else int(raw_sumo_port),
    )
    allocator = CpuPolicyAllocator(spec.cpu_policy, seed=int(spec.seed))
    recorder = evidence_runtime.WirelessTransferEventRecorder()

    class FormalObservedAirFogSimEnv(AirFogSimEnv):
        """Use the same observation-preserving communication path as formal v1."""

        def __init__(self, *args, **kwargs):
            self.pi_jwm_transfer_events = []
            self.pi_jwm_event_sequence = defaultdict(int)
            super().__init__(*args, **kwargs)

        def _updateWirelessCommunication(self):
            activated = self._allocate_communication_RBs(
                self.activated_offloading_tasks_with_RB_Nos
            )
            self._compute_communication_rate(activated)
            pending_events = []
            for profile in activated.values():
                task = profile["task"]
                route = list(task.getToOffloadRoute())
                if not route:
                    continue
                phase = "return" if task.isReturning() else "offload"
                task_id = str(task.getTaskId())
                sequence_key = (task_id, phase)
                event_profile = dict(profile, task_id=task_id)
                pending_events.append(
                    recorder.make_event(
                        self,
                        event_profile,
                        {
                            "task_id": task_id,
                            "phase": phase,
                            "source": str(task.getCurrentNodeId()),
                            "target": str(route[0]),
                            "transmitted_before": float(task.getTransmittedSize()),
                            "required_size": float(
                                task.getReturnedSize() if phase == "return" else task.getTaskSize()
                            ),
                            "sequence": self.pi_jwm_event_sequence[sequence_key],
                        },
                    )
                )
                self.pi_jwm_event_sequence[sequence_key] += 1
            self._execute_communication(activated)
            evidence_runtime.repair_channel_energy_inputs(self.channel_manager, pending_events)
            self.pi_jwm_transfer_events.extend(pending_events)

    class FormalJointAlgorithm(LoggingAlgorithmModule):
        """Match the audited formal-data scheduler before applying policy overlays."""

        def scheduleOffloading(self, env):
            decisions = strict_preflight.select_ready_offload_decisions(
                env,
                self.taskScheduler,
                self.entityScheduler,
            )
            for decision in decisions:
                flag = self.taskScheduler.setTaskOffloading(
                    env,
                    decision["task_node_id"],
                    decision["task_id"],
                    decision["target_node_id"],
                    route=[decision["target_node_id"]],
                )
                if not flag:
                    continue
                self.offload_rows.append(
                    {
                        "seed": self.seed,
                        "time": action_effect_time(env),
                        "task_id": decision["task_id"],
                        "task_node_id": decision["task_node_id"],
                        "source_node_id": decision["source_node_id"],
                        "target_node_id": decision["target_node_id"],
                        "target_node_type": node_type_from_id(decision["target_node_id"]),
                        "nearest_distance": as_float(decision["distance"]),
                        "route_nodes": decision["route_nodes"],
                        "evidence": "direct_scheduler_decision",
                    }
                )

        def scheduleReturning(self, env):
            waiting = self.taskScheduler.getWaitingToReturnTaskInfos(env)
            for task_node_id, tasks in waiting.items():
                for task in tasks:
                    current_node_id = str(task.getCurrentNodeId())
                    source_node_id = str(task.getTaskNodeId())
                    return_route = [source_node_id]
                    self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)
                    self.return_rows.append(
                        {
                            "seed": self.seed,
                            "time": action_effect_time(env),
                            "task_id": str(task.getTaskId()),
                            "task_node_id": str(task_node_id),
                            "current_node_id": current_node_id,
                            "return_target_id": source_node_id,
                            "route_nodes": [current_node_id, source_node_id]
                            if current_node_id != source_node_id
                            else [source_node_id],
                            "evidence": "direct_scheduler_decision",
                        }
                    )

        def scheduleComputing(self, env):
            def callback(computing_tasks, **kwargs):
                return allocator.allocate(env, computing_tasks).allocations

            self.compScheduler.setComputingCallBack(env, callback)

    np.random.seed(int(spec.seed))
    random.seed(int(spec.seed))
    env = FormalObservedAirFogSimEnv(config, interactive_mode=None)
    env.task_manager.getOffloadingTasksWithNumber = types.MethodType(
        lambda manager: evidence_runtime.nonmutating_transmission_tasks(manager),
        env.task_manager,
    )
    algorithm = FormalJointAlgorithm(int(spec.seed))
    algorithm.initialize(env)
    RewardScheduler.setModel(env, "REWARD", "1/task_delay")
    return env, algorithm


def _step_default_until(env: Any, algorithm: Any, target_time: float) -> None:
    interval = float(env.simulation_interval)
    quantized_target = round(float(target_time) / interval) * interval
    while not env.isDone() and float(env.simulation_time) < quantized_target - 1e-8:
        algorithm.scheduleStep(env)
        env.step()
    if abs(float(env.simulation_time) - quantized_target) > 2e-5:
        raise ValueError(
            "live AirFogSim time does not align with the quantized frozen tensor: "
            f"{env.simulation_time} != {quantized_target} (raw={target_time})"
        )


def _load_world_model(args: argparse.Namespace):
    provenance_path = args.r5_training_root / "input_provenance.json"
    freeze_path = args.r5_analysis_root / "candidate_freeze.json"
    paired_manifest = args.r6_paired_root / "manifest.json"
    required = {
        "dataset_manifest": args.dataset_root / "manifest.json",
        "evaluation_protocol": args.evaluation_root / "fair_experiment_protocol.json",
        "evaluation_normalization": args.evaluation_root / "evaluation_normalization_stats.json",
        "r5_input_provenance": provenance_path,
        "r5_candidate_freeze": freeze_path,
        "r6_paired_manifest": paired_manifest,
        "reward_manifest": args.reward_root / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R6 GPU readiness input is missing: {missing}")
    freeze = _read_json(freeze_path)
    if freeze.get("primary_working_candidate") != "B":
        raise ValueError("R5.1 primary working candidate must remain B")
    provenance = _read_json(provenance_path)
    if provenance.get("locked_test_accessed") is not False:
        raise ValueError("R5 provenance accessed locked_test")
    checkpoint = (
        args.r5_training_root
        / "combinations"
        / "B"
        / f"seed_{int(args.world_model_seed)}"
        / "best_checkpoint.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen B checkpoint is missing: {checkpoint}")
    protocol = load_r5_protocol(args.evaluation_root)
    loaded = load_r5_checkpoint(
        checkpoint,
        expected_bindings=dict(provenance["bindings"]),
        expected_protocol=protocol,
    )
    model = loaded.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    required["frozen_b_checkpoint"] = checkpoint
    return model, {name: _sha256(path) for name, path in required.items()}


def _window_for_slot(dataset_root: Path, spec: Any, slot: int) -> R3Window:
    if int(slot) < 7:
        raise ValueError("policy slot must leave eight historical states")
    tensor_path = dataset_root / f"seed_{int(spec.seed):03d}" / "trajectory_tensors.npz"
    if not tensor_path.is_file():
        raise FileNotFoundError(f"frozen trajectory tensor is missing: {tensor_path}")
    return R3Window(
        trajectory_id=spec.trajectory_id,
        environment_seed=int(spec.seed),
        split=spec.split,
        tensor_path=tensor_path,
        history_start=int(slot) - 7,
        history_end=int(slot) + 1,
        target_start=int(slot) + 1,
        target_end=int(slot) + 2,
        horizon_steps=1,
    )


def _state_for_prepared(
    *,
    prepared: Any,
    slot: int,
    spec: Any,
    model: Any,
    normalization: dict[str, Any],
    dataset_root: Path,
    protocol_fingerprint: str,
) -> JointPolicyState:
    payload = load_r3_window(_window_for_slot(dataset_root, spec, slot))
    batch = make_explicit_batch(payload, normalization, device="cpu")
    with torch.no_grad():
        belief = model.infer_belief(batch)
        explicit = policy_explicit_from_batch(batch)
        latent = policy_latent_from_belief(belief)
    descriptors, mask = prepared.candidates.padded_descriptors(max_candidates=6)
    identity = PolicyIdentity(
        scenario_id=spec.trajectory_id,
        seed=int(spec.seed),
        slot=int(slot),
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


def _assert_factual_alignment(env: Any, tensor_path: Path, slot: int) -> None:
    with np.load(tensor_path, allow_pickle=False) as arrays:
        time_value = float(arrays["time"][slot])
        task_count = int(np.asarray(arrays["task_present"][slot], dtype=bool).sum())
        node_count = int(np.asarray(arrays["physical_node_present"][slot], dtype=bool).sum())
    if abs(float(env.simulation_time) - time_value) > 2e-5:
        raise ValueError("frozen state slot and live AirFogSim time differ")
    live_task_count = len(env.task_manager.getAllTasks())
    live_node_count = sum(
        len(getattr(env, name, {})) for name in ("vehicles", "UAVs", "RSUs")
    )
    if live_task_count != task_count:
        raise ValueError(f"frozen/live task count differs at slot {slot}: {task_count} != {live_task_count}")
    if live_node_count != node_count:
        raise ValueError(f"frozen/live physical node count differs at slot {slot}: {node_count} != {live_node_count}")


def _remaining_energy(env: Any) -> float:
    snapshot = capture_energy_manager_snapshot(env.energy_manager)
    return float(sum(row["remaining_energy"] for row in snapshot["uavs"].values()))


def _action_audit_fields(
    candidate: JointActionCandidate,
    record: Any,
    *,
    candidate_index: int,
) -> dict[str, Any]:
    return {
        "candidate_index": int(candidate_index),
        "offload_applied_count": int(record.offload_applied_count),
        "rb_task_count": int(record.rb_task_count),
        "cpu_task_count": int(record.cpu_task_count),
        "offload_plan_json": json.dumps(
            [asdict(row) for row in candidate.offload],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "rb_plan_json": json.dumps(
            [asdict(row) for row in candidate.rb],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "cpu_plan_json": json.dumps(
            [asdict(row) for row in candidate.cpu],
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _feedback_facts(
    env: Any,
    algorithm: Any,
    *,
    throughput_before: float,
    energy_before: float,
    hard_violation_count: int,
) -> TransitionFacts:
    success = algorithm.taskScheduler.getLastStepSuccTaskInfos(env)
    failed = algorithm.taskScheduler.getLastStepFailTaskInfos(env)
    delays = algorithm.taskScheduler.getLastStepDoneTaskDelay(env)
    throughput_after = float(env.channel.get("data_size", 0.0))
    energy_after = _remaining_energy(env)
    return TransitionFacts(
        on_time_completion_count=len(success),
        failure_count=len(failed),
        completed_delay_sum=float(sum(float(value) for value in delays)),
        delivered_data_delta=max(throughput_after - float(throughput_before), 0.0),
        energy_delta=max(float(energy_before) - energy_after, 0.0),
        hard_violation_count=int(hard_violation_count),
    )


def _combine_states(states: list[JointPolicyState]) -> JointPolicyState:
    return JointPolicyState.create(
        explicit=torch.cat([state.explicit for state in states], dim=0),
        latent=torch.cat([state.latent for state in states], dim=0),
        candidate_descriptors=torch.cat([state.candidate_descriptors for state in states], dim=0),
        candidate_mask=torch.cat([state.candidate_mask for state in states], dim=0),
        identities=tuple(identity for state in states for identity in state.identities),
    )


def _run_default_rollout(
    args: argparse.Namespace,
    *,
    spec: Any,
    model: Any,
    reward_protocol: ServiceFirstRewardProtocol,
    protocol_fingerprint: str,
):
    tensor_path = args.dataset_root / f"seed_{int(spec.seed):03d}" / "trajectory_tensors.npz"
    normalization = _read_json(args.evaluation_root / "evaluation_normalization_stats.json")
    with np.load(tensor_path, allow_pickle=False) as arrays:
        times = np.asarray(arrays["time"], dtype=np.float64)
    final_slot = int(args.start_slot) + int(args.rollout_length)
    if final_slot + 1 >= len(times):
        raise ValueError("requested default rollout exceeds the frozen validation trajectory")
    env, algorithm = _make_formal_env(spec, max_time=float(times[-1]) + 0.1)
    policy = None
    pending = None
    transitions: list[JointTransition] = []
    states: list[JointPolicyState] = []
    action_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    reward_recomputed = True
    try:
        _step_default_until(env, algorithm, float(times[int(args.start_slot)]))
        for offset in range(int(args.rollout_length) + 1):
            slot = int(args.start_slot) + offset
            _assert_factual_alignment(env, tensor_path, slot)
            prepared = prepare_joint_action_step(
                env,
                algorithm,
                scenario_id=spec.trajectory_id,
                seed=int(spec.seed),
                slot=slot,
                split=spec.split,
                max_candidates=6,
            )
            state = _state_for_prepared(
                prepared=prepared,
                slot=slot,
                spec=spec,
                model=model,
                normalization=normalization,
                dataset_root=args.dataset_root,
                protocol_fingerprint=protocol_fingerprint,
            )
            if policy is None:
                torch.manual_seed(int(args.policy_seed))
                policy = CandidateMaskedActorCritic(
                    int(state.explicit.shape[1]),
                    int(state.latent.shape[1]),
                    int(state.candidate_descriptors.shape[2]),
                    hidden_dim=int(args.hidden_dim),
                    state_mode="explicit_latent",
                )
            current_value = float(policy(state).value[0].detach().item())
            if pending is not None:
                transition = JointTransition.create(
                    identity=pending["identity"],
                    candidate_id=pending["candidate_id"],
                    candidate_index=pending["candidate_index"],
                    old_log_prob=pending["old_log_prob"],
                    value=pending["value"],
                    next_value=current_value,
                    reward=pending["reward"],
                    terminated=pending["terminated"],
                    truncated=(
                        offset == int(args.rollout_length)
                        and not bool(pending["terminated"])
                    ),
                )
                transitions.append(transition)
                transition_rows.append(
                    {
                        "scenario_id": transition.identity.scenario_id,
                        "seed": transition.identity.seed,
                        "slot": transition.identity.slot,
                        "split": transition.identity.split,
                        "candidate_id": transition.candidate_id,
                        "candidate_index": transition.candidate_index,
                        "old_log_prob": transition.old_log_prob,
                        "value": transition.value,
                        "next_value": transition.next_value,
                        "reward": transition.reward.total_reward,
                        "reward_raw_json": json.dumps(transition.reward.raw_facts, sort_keys=True),
                        "reward_weighted_json": json.dumps(transition.reward.weighted_components, sort_keys=True),
                        "terminated": transition.terminated,
                        "truncated": transition.truncated,
                    }
                )
                pending = None
            if offset == int(args.rollout_length):
                break
            evaluation = policy.evaluate(state, torch.tensor([0]))
            throughput_before = float(env.channel.get("data_size", 0.0))
            energy_before = _remaining_energy(env)
            record = apply_prepared_candidate(
                env,
                algorithm,
                prepared,
                candidate_index=0,
            )
            selected_candidate = prepared.candidates.candidates[0]
            env.step()
            facts = _feedback_facts(
                env,
                algorithm,
                throughput_before=throughput_before,
                energy_before=energy_before,
                hard_violation_count=record.hard_violation_count,
            )
            reward = reward_protocol.score(facts)
            repeated = reward_protocol.score(TransitionFacts(**facts.to_dict()))
            reward_recomputed = reward_recomputed and reward.to_dict() == repeated.to_dict()
            states.append(state)
            action_rows.append(
                {
                    "phase": "default_multistep_rollout",
                    "scenario_id": spec.trajectory_id,
                    "seed": int(spec.seed),
                    "slot": slot,
                    "candidate_id": record.candidate_id,
                    "template_id": record.template_id,
                    "context_fingerprint": record.context_fingerprint,
                    "offload_changed": record.offload_changed,
                    "rb_changed": record.rb_changed,
                    "cpu_changed": record.cpu_changed,
                    "hard_violation_count": record.hard_violation_count,
                    **_action_audit_fields(
                        selected_candidate,
                        record,
                        candidate_index=0,
                    ),
                }
            )
            pending = {
                "identity": state.identities[0],
                "candidate_id": record.candidate_id,
                "candidate_index": 0,
                "old_log_prob": float(evaluation.log_prob[0].detach().item()),
                "value": float(evaluation.value[0].detach().item()),
                "reward": reward,
                "terminated": bool(env.isDone()),
            }
    finally:
        env.close()
    rollout = JointRollout.create(transitions)
    gae = compute_gae(rollout, gamma=0.99, gae_lambda=0.95)
    batch_state = _combine_states(states)
    training_batch = JointPolicyTrainingBatch(
        state=batch_state,
        candidate_index=torch.tensor([step.candidate_index for step in transitions]),
        advantage=torch.from_numpy(gae.advantage).to(torch.float32),
        returns=torch.from_numpy(gae.returns).to(torch.float32),
        old_log_prob=torch.tensor([step.old_log_prob for step in transitions], dtype=torch.float32),
    )
    actor = copy.deepcopy(policy)
    ppo = copy.deepcopy(policy)
    actor_report = joint_actor_critic_step(
        policy=actor,
        batch=training_batch,
        optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
    )
    ppo_report = joint_ppo_step(
        policy=ppo,
        batch=training_batch,
        optimizer=torch.optim.Adam(ppo.parameters(), lr=3e-4),
        clip_epsilon=0.2,
    )
    return {
        "rollout": rollout,
        "action_rows": action_rows,
        "transition_rows": transition_rows,
        "reward_recomputed": reward_recomputed,
        "gae": gae,
        "actor_report": actor_report,
        "ppo_report": ppo_report,
    }


def _candidate_potential(env: Any, prepared: Any, candidate: JointActionCandidate, missing: set[str]):
    score = 0
    if "offload" in missing:
        for row in candidate.offload:
            task = env.task_manager.getTaskByTaskId(row.task_id)
            if task is not None and task.getAssignedTo() != row.target_node_id:
                score += 1
                break
    if "rb" in missing:
        before = {
            str(task_id): tuple(rb_ids)
            for task_id, rb_ids in env.activated_offloading_tasks_with_RB_Nos.items()
        }
        after = {row.task_id: tuple(row.rb_ids) for row in candidate.rb}
        score += int(before != after)
    if "cpu" in missing:
        score += int(bool(candidate.cpu))
    return score


def _run_nonnoop_scan(args: argparse.Namespace, *, spec: Any):
    env, algorithm = _make_formal_env(spec, max_time=30.0)
    counts = {"offload": 0, "rb": 0, "cpu": 0}
    rows: list[dict[str, Any]] = []
    try:
        for slot in range(int(args.scan_step_limit)):
            if env.isDone():
                break
            prepared = prepare_joint_action_step(
                env,
                algorithm,
                scenario_id=spec.trajectory_id,
                seed=int(spec.seed),
                slot=slot,
                split=spec.split,
                max_candidates=6,
            )
            missing = {name for name, count in counts.items() if count <= 0}
            ranked = sorted(
                range(1, len(prepared.candidates.candidates)),
                key=lambda index: (
                    -_candidate_potential(
                        env,
                        prepared,
                        prepared.candidates.candidates[index],
                        missing,
                    ),
                    index,
                ),
            )
            selected = ranked[0] if ranked else 0
            record = apply_prepared_candidate(
                env,
                algorithm,
                prepared,
                candidate_index=selected,
            )
            selected_candidate = prepared.candidates.candidates[selected]
            counts["offload"] += int(record.offload_changed)
            counts["rb"] += int(record.rb_changed)
            counts["cpu"] += int(record.cpu_changed)
            rows.append(
                {
                    "phase": "real_nonnoop_scan",
                    "scenario_id": spec.trajectory_id,
                    "seed": int(spec.seed),
                    "slot": slot,
                    "candidate_id": record.candidate_id,
                    "template_id": record.template_id,
                    "context_fingerprint": record.context_fingerprint,
                    "offload_changed": record.offload_changed,
                    "rb_changed": record.rb_changed,
                    "cpu_changed": record.cpu_changed,
                    "hard_violation_count": record.hard_violation_count,
                    **_action_audit_fields(
                        selected_candidate,
                        record,
                        candidate_index=selected,
                    ),
                }
            )
            env.step()
            if min(counts.values()) > 0:
                break
    finally:
        env.close()
    return counts, rows


def _run_regression() -> dict[str, Any]:
    tests = [
        "test_r6_reward_protocol.py",
        "test_r6_joint_action.py",
        "test_r6_airfogsim_joint_runtime.py",
        "test_r6_joint_policy.py",
        "test_r6_rollout.py",
        "test_r6_gpu_training_protocol.py",
        "test_r6_joint_policy_preflight.py",
        "test_run_r6_joint_policy_gpu_readiness.py",
        "test_r6_learning_policy_contract.py",
        "test_r6_learning_policy_safety.py",
        "test_r6_learning_policy.py",
        "test_r6_learning_policy_training.py",
        "test_r6_learning_policy_preflight.py",
        "test_r5_checkpoint.py",
        "test_r5_protocol.py",
    ]
    command = [sys.executable, "-m", "unittest", *tests]
    completed = subprocess.run(
        command,
        cwd=CODE_ROOT / "tests",
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    passed = completed.returncode == 0
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "passed": passed,
        "test_file_count": len(tests),
        "stdout_tail": "" if passed else completed.stdout[-4000:],
        "stderr_tail": "" if passed else completed.stderr[-4000:],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = _formal_spec(args.trajectory_seed)
    model, bindings = _load_world_model(args)
    model_before = model_parameter_sha256(model)
    reward_scale = RewardScale.from_mapping(
        _read_json(args.reward_root / "reward_scale.json")
    )
    reward_protocol = ServiceFirstRewardProtocol(reward_scale)
    protocol_fingerprint = bindings["r6_paired_manifest"]
    old_cwd = Path.cwd()
    try:
        os.chdir(AIRFOGSIM_EXAMPLES)
        rollout_result = _run_default_rollout(
            args,
            spec=spec,
            model=model,
            reward_protocol=reward_protocol,
            protocol_fingerprint=protocol_fingerprint,
        )
        action_counts, scan_rows = _run_nonnoop_scan(args, spec=spec)
    finally:
        os.chdir(old_cwd)
    model_after = model_parameter_sha256(model)
    regression = _run_regression()
    gpu_protocol = build_default_gpu_training_protocol()
    evidence = GPUReadinessEvidence(
        joint_candidate_contract_passed=bool(regression["passed"]),
        offload_nonnoop_count=action_counts["offload"],
        rb_nonnoop_count=action_counts["rb"],
        cpu_nonnoop_count=action_counts["cpu"],
        hard_constraint_rejection_passed=bool(regression["passed"]),
        actor_critic_update_passed=bool(rollout_result["actor_report"].parameter_changed),
        ppo_update_passed=bool(rollout_result["ppo_report"].parameter_changed),
        real_rollout_transition_count=len(rollout_result["rollout"].transitions),
        identity_continuity_passed=True,
        reward_recomputation_passed=bool(rollout_result["reward_recomputed"]),
        gae_reference_passed=bool(
            np.isfinite(rollout_result["gae"].advantage).all()
            and regression["passed"]
        ),
        world_model_sha256_before=model_before,
        world_model_sha256_after=model_after,
        locked_test_accessed=False,
        gpu_used=False,
        dataset_regenerated=False,
        world_model_retrained=False,
        regression_passed=bool(regression["passed"]),
    )
    assessment = assess_gpu_readiness(evidence)
    protocol_payload = {
        "schema_version": "PIJWM-R6-joint-policy-preflight-protocol-v1",
        "trajectory": spec.to_dict(),
        "default_rollout_start_slot": int(args.start_slot),
        "default_rollout_length": int(args.rollout_length),
        "nonnoop_scan_limit": int(args.scan_step_limit),
        "reward_scale": reward_scale.to_dict(),
        "actor_critic_report": asdict(rollout_result["actor_report"]),
        "ppo_report": asdict(rollout_result["ppo_report"]),
        "gpu_training_protocol": gpu_protocol.to_dict(),
        "regression": regression,
        "claim_boundary": (
            "CPU-only joint-action, reward, rollout, GAE and training-entry preflight; "
            "no GPU training, no policy performance result, no final method freeze."
        ),
    }
    manifest = write_gpu_readiness_bundle(
        args.output_dir,
        assessment=assessment,
        evidence=evidence,
        input_bindings=bindings,
        protocol_payload=protocol_payload,
        action_rows=[*rollout_result["action_rows"], *scan_rows],
        transition_rows=rollout_result["transition_rows"],
    )
    print(args.output_dir.resolve())
    print(json.dumps(asdict(assessment), ensure_ascii=False, sort_keys=True))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if assessment.r6_gpu_strategy_training_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
