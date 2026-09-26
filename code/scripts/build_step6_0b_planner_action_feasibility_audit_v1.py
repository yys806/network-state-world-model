"""Read-only source audit receipts for STEP 6.0B; no simulator run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AIR = ROOT / "code/reference/AirFogSim"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_0b_planner_action_feasibility_audit_v1_20260926"
FILES = {
    "cpu_scheduler": "airfogsim/scheduler/computation_sched.py",
    "cpu_manager": "airfogsim/manager/task_manager.py",
    "task": "airfogsim/entities/task.py",
    "fog_node": "airfogsim/entities/abstract/fog_node.py",
    "algorithm": "airfogsim/airfogsim_algorithm.py",
    "env": "airfogsim/airfogsim_env.py",
    "traffic_scheduler": "airfogsim/scheduler/traffic_sched.py",
    "traffic_manager": "airfogsim/manager/traffic_manager.py",
    "scenario_config": "examples/config.yaml",
}
PI_FILES = {
    "collector": "code/scripts/collect_step5_5_formal_raw_v1.py",
    "raw_adapter": "code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py",
    "sample_tensor": "code/src/pi_jwm/step4_2a_graph_input_extension_v1.py",
    "graph": "code/src/pi_jwm/step4_3a_typed_dual_graph_builder_v1.py",
    "world_model": "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
    "candidate": "code/src/pi_jwm/step6_0a_candidate_generation_v1.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(path: Path, snippet: str) -> int:
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if snippet in text:
            return number
    raise ValueError(f"source anchor absent: {path}: {snippet}")


def anchor(base: Path, rel: str, snippet: str) -> dict[str, object]:
    path = base / rel
    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "line": line(path, snippet), "symbol_or_expression": snippet, "sha256": sha(path)}


def git_root(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip().replace("\\", "/") if result.returncode == 0 else None


def payloads() -> dict[str, dict]:
    if not AIR.is_dir():
        raise FileNotFoundError("LOCAL_AIRFOGSIM_SOURCE_UNAVAILABLE")
    nested_git = (AIR / ".git").exists() or git_root(AIR) != str(ROOT).replace("\\", "/")
    if nested_git:
        raise ValueError("AirFogSim Git identity changed; re-audit required")
    air_hashes = {key: sha(AIR / path) for key, path in FILES.items()}
    pi_hashes = {key: sha(ROOT / path) for key, path in PI_FILES.items()}
    local_fingerprint = hashlib.sha256(json.dumps(air_hashes, sort_keys=True).encode()).hexdigest()
    common = {
        "schema": "PIJWM_STEP_06_0B_SOURCE_AUDIT_V1",
        "pi_jwm_source_sha": "2a9c06aa7737d112722b0d1f53d239ce828f9107",
        "airfogsim_source_sha": None,
        "airfogsim_git_identity_reason": "LOCAL_DIRECTORY_HAS_NO_INDEPENDENT_GIT_METADATA; git -C resolves PI-JWM parent",
        "airfogsim_worktree_clean_or_dirty": "UNDETERMINABLE_WITHOUT_GIT_METADATA",
        "airfogsim_local_relevant_source_sha256": local_fingerprint,
        "airfogsim_relevant_file_sha256": air_hashes,
        "pi_jwm_relevant_file_sha256": pi_hashes,
        "airfogsim_official_upstream_comparison": {
            "repository": "https://github.com/ZhiweiWei-NAMI/AirFogSim",
            "candidate_commit_not_local_identity": "76c0edb4ac6c7b5e09cc146db9928c1bc64bbb60",
            "sampled_blob_matches": "3/5; local task_manager.py and airfogsim_env.py differ",
            "sampled_upstream_git_blob_sha": {
                "airfogsim/manager/traffic_manager.py": "381af01c7fe05adb2cc339414abf8e845f9d8a5c",
                "airfogsim/manager/task_manager.py": "0e9c6f2a3437ac953bbc5b549bb4a5b76298c793",
                "airfogsim/airfogsim_env.py": "44dbbdceb2cbcceeac4992e68ababe92c9ba1d15",
                "examples/config.yaml": "a857d1a00020babf22ced79fbae27a0f019db9b1",
                "README.md": "cb4f0090befb498d09fe1d2115dc19cdc823de26",
            },
        },
        "runtime_probe_performed": False,
        "step5_6b_remote_contacted": False,
        "gpu_used": False,
        "checkpoint_consumed": False,
        "world_model_rollout_performed": False,
        "planner_objective_computed": False,
        "locked_test_accessed": False,
    }
    cpu_sources = {
        "static_capacity_config": anchor(AIR, FILES["scenario_config"], "cpu: 3 # CPU capacity of UAVs"),
        "fog_profile_getter": anchor(AIR, "airfogsim/entities/abstract/fog_node.py", "def getFogProfile"),
        "callback_api": anchor(AIR, FILES["cpu_scheduler"], "def setComputingCallBack"),
        "callback_install": anchor(AIR, FILES["cpu_scheduler"], "env.alloc_cpu_callback = callback"),
        "simulator_execute": anchor(AIR, FILES["env"], "self.task_manager.computeTasks(self.alloc_cpu_callback"),
        "allocation_lookup": anchor(AIR, FILES["cpu_manager"], "allocated_cpu = allocated_cpus.get(task_id, 0)"),
        "service": anchor(AIR, FILES["task"], "self._computed_size += allocated_cpu * simulation_interval"),
        "pi_static_raw": anchor(ROOT, PI_FILES["raw_adapter"], '"node_cpu_capacity_observation_rows": cpu_observations'),
        "pi_static_sample": anchor(ROOT, PI_FILES["sample_tensor"], 'sample["static"]["agent_static_capability"] = static_capability'),
        "pi_static_tensor": anchor(ROOT, PI_FILES["sample_tensor"], 'output["agent_cpu_capacity_raw"][batch_index, slot]'),
        "pi_static_graph": anchor(ROOT, PI_FILES["graph"], '"cpu_capacity": _array(tensor["agent_cpu_capacity_raw"]'),
        "pi_comp_action": anchor(ROOT, PI_FILES["collector"], 'step23._install_cpu_callback(env, allocations)'),
        "pi_rule_service": anchor(ROOT, PI_FILES["world_model"], 'cpu_service = action["comp_values"][..., 0].clamp_min(0) * self.config.slot_duration_s'),
    }
    cpu = {**common, "cpu_verdict": "STATIC_CAPACITY_ONLY", "source_symbols": cpu_sources,
        "static_capacity": {"source": "FogNode.getFogProfile()['cpu'] from scenario fog_profile",
                            "unit": "CPU-work-unit/s", "decision_time": "observed when key exists; masked otherwise"},
        "comp_action": "task_id,node_id,allocated_cpu_per_s -> PI-JWM callback -> ComputationScheduler.setComputingCallBack -> env._updateComputation -> TaskManager.computeTasks",
        "actual_service": "Task.compute adds allocated_cpu * simulation_interval, capped by task remaining work",
        "dynamic_available_cpu": {"direct_field": None, "causal_derivation": None,
                                  "reason": "no reservation/available state; callback is chosen at execution and is not a decision-time remaining-resource field"},
        "oversubscription_behavior": "ALLOW_OVERSUBSCRIPTION",
        "oversubscription_scope": "Simulator accepts callback per-task allocations without per-node capacity sum check; task-work cap still applies. PI-JWM collector separately scales/validates static capacity.",
        "raw_field": "decision.node_cpu_capacity_observation_rows[].capacity_per_s (static)",
        "sample_field": "static.agent_static_capability[].cpu_capacity_per_s (static)",
        "tensor_field": "agent_cpu_capacity_raw/agent_cpu_capacity/agent_cpu_capacity_mask (static)",
        "graph_field": "agent_nodes.cpu_capacity/cpu_capacity_mask (static)",
        "world_model_field": "task_work_remaining and Comp action comp_values; no dynamic available CPU",
        "constraint_closure": "dynamic_available_cpu remains UNKNOWN"}
    uav_sources = {
        "action_api": anchor(AIR, FILES["traffic_scheduler"], "def setUAVMobilityPatterns"),
        "manager_setter": anchor(AIR, FILES["traffic_manager"], "def _updateUAVMobilityPatternById"),
        "position_update": anchor(AIR, FILES["traffic_manager"], "new_position = (org_position[0] + speed * np.cos(angle)"),
        "initial_altitude": anchor(AIR, FILES["traffic_manager"], 'self._UAV_z_range = config_traffic.get'),
        "speed_config": anchor(AIR, FILES["scenario_config"], "UAV_speed_range: [10,30]"),
        "altitude_config": anchor(AIR, FILES["scenario_config"], "UAV_z_range: [100, 200]"),
        "behavior_policy": anchor(ROOT, PI_FILES["collector"], '"speed": (5.0, 8.0, 10.0, 12.0, 15.0)[profile]'),
    }
    uav = {**common, "mobility_verdicts": {
        "azimuth": "DATASET_SUPPORT_BOUND_ONLY", "elevation": "DATASET_SUPPORT_BOUND_ONLY",
        "speed": "CONFIG_BOUND_FOUND", "spatial_geofence": "CONFIG_BOUND_FOUND"},
        "source_symbols": uav_sources,
        "simulator_hard_bounds": {key: None for key in ("azimuth", "elevation", "speed", "altitude_z", "xy_map", "acceleration")},
        "simulator_out_of_bound_behavior": "ALLOW; no numeric reject/clamp/wrap in setter or stepSimulation; XY grid indexing drops out-of-grid membership, not motion",
        "units": {"angle": "radian azimuth from +x toward +y", "phi": "radian elevation from xy plane",
                  "speed": "distance/s by executed formula; config comment says distance/timeslot and conflicts with code", "step": "traffic_interval=0.1 s in current example"},
        "config_sources": {"example_UAV_speed_range": [10, 30], "example_UAV_z_initialization_range": [100, 200],
                           "example_nonfly_zone_coordinates": "present, checked by target helper but not direct setter/stepSimulation", "enforcement": "generation/initialization only"},
        "formal_dataset_behavior_support": {"kind": "DATASET_BEHAVIOR_SUPPORT_ONLY", "accepted_trajectories": 60,
            "verified_raw_file_hash_mismatches": 0, "rows": 11520,
            "azimuth_rad": {"min": -1.25, "max": 2.0500000000000007},
            "elevation_rad": {"min": 0.0, "max": 0.0},
            "speed_mps": {"min": 0.0, "max": 15.0, "unique_count": 6},
            "source_manifest": "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923/collection_summary.json",
            "source_manifest_sha256": sha(ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923/collection_summary.json")},
        "constraint_closure": "mobility_numeric_bounds remains UNKNOWN; no simulator hard numeric bound"}
    semantics = {**common, "source_symbols": {
        "simulator_position": uav_sources["position_update"],
        "simulator_acceleration": anchor(AIR, FILES["traffic_manager"], "acceleration = (last_speed - speed) / self._traffic_interval"),
        "world_model_position": anchor(ROOT, PI_FILES["world_model"], "delta = torch.stack((speed * torch.cos(azimuth)"),
        "world_model_acceleration": anchor(ROOT, PI_FILES["world_model"], 'nxt["acceleration"] = (nxt["speed"] - old_speed) / self.config.slot_duration_s'),
        "canonical_raw": anchor(ROOT, "code/src/pi_jwm/raw_trajectory_causal_contract_v1.py", "current_speed - float(previous_speed)"),
    }, "position_equations_match": True, "angle_unit_match": True, "speed_execution_unit_match": True,
        "slot_duration_match_in_current_example": True,
        "acceleration_difference": "Simulator raw reports (old-new)/dt; PI-JWM canonical and World Model use (new-old)/dt. Existing Raw explicitly separates simulator raw from canonical, so no silent training-label swap.",
        "world_model_numeric_clamp": False, "simulator_numeric_clamp": False,
        "status": "DOCUMENTED_CANONICAL_TRANSFORM; no World Model edit"}
    closure = {**common, "cpu_verdict": cpu["cpu_verdict"], "mobility_verdicts": uav["mobility_verdicts"],
        "candidate_constraint_code_modified": False,
        "unknown_constraints_remaining": ["dynamic_available_cpu", "mobility_numeric_bounds"],
        "source_facts_audited": True, "source_git_provenance_complete": False,
        "acceptance_status": "SOURCE_FACTS_COMPLETE_GIT_PROVENANCE_LIMITED",
        "no_world_model_candidate_rollout": True}
    overview = {**common, "cpu_verdict": cpu["cpu_verdict"], "mobility_verdicts": uav["mobility_verdicts"],
        "source_files": {**{key: "code/reference/AirFogSim/" + value for key, value in FILES.items()}, **PI_FILES},
        "source_symbols": {"cpu": cpu_sources, "mobility": uav_sources},
        "config_sources": uav["config_sources"], "candidate_constraint_code_modified": False,
        "unknown_constraints_remaining": closure["unknown_constraints_remaining"],
        "audit_evidence_kind": "LOCAL_SOURCE_AND_FORMAL_DATA_BEHAVIOR_SUPPORT_ONLY"}
    return {
        "planner_action_feasibility_source_audit.json": overview,
        "cpu_feasibility_source_receipt.json": cpu,
        "uav_mobility_bound_source_receipt.json": uav,
        "world_model_vs_simulator_mobility_semantics_receipt.json": semantics,
        "constraint_closure_receipt.json": closure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = payloads()
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        encoded = json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"receipt mismatch: {name}")
        else:
            path.write_text(encoded, encoding="utf-8", newline="\n")
    print(json.dumps({"passed": True, "mode": "check" if args.check else "write",
                      "receipt_count": len(records), "cpu_verdict": records["cpu_feasibility_source_receipt.json"]["cpu_verdict"],
                      "airfogsim_git_sha_available": False}, sort_keys=True))


if __name__ == "__main__":
    main()
