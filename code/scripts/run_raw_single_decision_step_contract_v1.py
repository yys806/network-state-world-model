"""Build machine-readable evidence for the Step 2 raw-step contract."""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from pi_jwm.raw_single_decision_step_contract_v1 import (ACTION_FIELD_SPECS, ActionBundle, CommAction, CompAction, DecisionSnapshot, EntityState, OutcomeSnapshot, RouteAction, SingleDecisionStep, TaskState, UavMobilityAction, apply_action_bundle_to_airfogsim, load_airfogsim_scheduler_classes_from_source, validate_single_decision_step)
class _TaskManager:
    def offloadTask(self, task_node_id, task_id, target_node_id, current_time, route): return True
class _ChannelManager: n_RB = 4
class _RuntimeTask:
    def __init__(self, task_id): self.task_id, self.computed = task_id, 0.0
    def getTaskId(self): return self.task_id
class _MinimalEnv:
    def __init__(self):
        self.simulation_time, self.simulation_interval = 1.0, 0.1
        self.task_manager, self.channel_manager = _TaskManager(), _ChannelManager()
        self.activated_offloading_tasks_with_RB_Nos, self.task_return_routes = {}, {}
        self.uav_mobility_patterns, self.alloc_cpu_callback = {}, None
        self.uav_position, self.compute_task = [0.0, 0.0, 10.0], _RuntimeTask("task-comp")
    def step(self):
        p = self.uav_mobility_patterns["uav-0"]; d = p["speed"] * self.simulation_interval
        self.uav_position[0] += d * math.cos(p["angle"]) * math.cos(p["phi"])
        self.uav_position[1] += d * math.sin(p["angle"]) * math.cos(p["phi"])
        self.uav_position[2] += d * math.sin(p["phi"])
        self.compute_task.computed += self.alloc_cpu_callback({"rsu-0": [self.compute_task]})["task-comp"] * self.simulation_interval
        self.simulation_time += self.simulation_interval
def _decision(frame_index=4, time_s=1.0):
    return DecisionSnapshot("step2-minimal-trajectory", frame_index, time_s, 0.1,
        (EntityState("uav-0", "uav", True, (0.0, 0.0, 10.0)), EntityState("vehicle-0", "vehicle", True, (5.0, 0.0, 0.0)), EntityState("rsu-0", "rsu", True, (20.0, 0.0, 0.0))),
        (TaskState("task-route", "vehicle-0", "vehicle-0", "waiting_to_offload", (), None, 0.0, 10.0, 2.0, 0.0), TaskState("task-comp", "vehicle-0", "rsu-0", "computing", (), None, 0.0, 0.0, 5.0, 0.0)), 4, {"rsu-0": 3.0}, {"entities": "decision", "tasks": "decision", "channel": "decision", "resources": "decision"})
def _action(): return ActionBundle("step2-minimal-trajectory", 4, 1.0, (RouteAction("task-route", "vehicle-0", "offload", "rsu-0", ("rsu-0",)),), (CommAction("task-route", (0, 1)),), (CompAction("task-comp", "rsu-0", 2.0),), (UavMobilityAction("uav-0", 0.0, 0.0, 10.0),))
def _scan_historical_action_attempts():
    root = ROOT / "code" / "artifacts" / "formal_data" / "pi_jwm_v4_formal_candidate_v6_rb_v1_unlocked_20260821"
    for path in sorted(root.glob("trajectories/*/action_attempts.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line); kinds = [c["setter_kind"] for c in row.get("setter_calls", [])]
            if {"cpu_callback", "offload", "rb"}.issubset(kinds) and row.get("env_step_completed"):
                return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trajectory_id": row.get("trajectory_id"), "frame_index": row.get("frame_index"), "setter_kinds": kinds, "env_step_completed": True, "scope": "historical unlocked three-family evidence; no UAV mobility setter recorded"}
    raise RuntimeError("no historical accepted route/comm/comp action attempt found")
def main():
    output = ROOT / "code" / "artifacts" / "protocols" / "pi_jwm_raw_single_decision_step_contract_v1_20260918"; output.mkdir(parents=True, exist_ok=True)
    schedulers = load_airfogsim_scheduler_classes_from_source(ROOT / "code" / "reference" / "AirFogSim"); env = _MinimalEnv()
    receipt = apply_action_bundle_to_airfogsim(env, _action(), task_scheduler=schedulers.task, communication_scheduler=schedulers.communication, computation_scheduler=schedulers.computation, traffic_scheduler=schedulers.traffic)
    decision, after = _decision(), _decision(5, 1.1); entities = list(after.entities); entities[0] = EntityState("uav-0", "uav", True, tuple(env.uav_position))
    after = DecisionSnapshot(after.trajectory_id, after.frame_index, after.decision_time_s, after.slot_duration_s, tuple(entities), after.tasks, after.n_rb, after.node_cpu_capacity_per_s, after.source_phases)
    outcome = OutcomeSnapshot(1.1, after.entities, after.tasks, {"task-route": 0.5}, {"task-comp": 0.2}); validate_single_decision_step(SingleDecisionStep(decision, _action(), receipt, outcome, after))
    hashes = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in schedulers.source_files.items()}
    (output / "contract.json").write_text(json.dumps({"contract_version": "PIJWM-Raw-Single-Decision-Step-v1", "action_space": ["route", "comm", "comp", "mobility"], "vehicle_motion_boundary": "SUMO advances vehicles; PI-JWM mobility action targets UAV only", "action_field_specs": ACTION_FIELD_SPECS, "scheduler_source_files": schedulers.source_files, "scheduler_source_sha256": hashes, "source_load_notes": schedulers.source_load_notes}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "minimum_closure.json").write_text(json.dumps({"passed": True, "decision_time_s": 1.0, "outcome_time_s": 1.1, "next_decision_time_s": 1.1, "setter_kinds": [r.setter_kind for r in receipt.setter_calls], "uav_position_after": env.uav_position, "served_cpu_work": env.compute_task.computed, "scope": "minimal environment using repository AirFogSim scheduler source; not full AirFogSim scenario acceptance"}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "historical_real_evidence.json").write_text(json.dumps(_scan_historical_action_attempts(), indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "validation_summary.json").write_text(json.dumps({"step": "STEP 2", "passed": True, "minimum_four_family_closure": True, "historical_real_three_family_evidence": True, "full_real_four_family_airfogsim_trace": False, "training_started": False, "gpu_used": False, "locked_test_accessed": False}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0
if __name__ == "__main__": raise SystemExit(main())
