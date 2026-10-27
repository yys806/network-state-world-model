"""R2-05B CPU smoke: real V2U4Real frame -> mock perception -> app ingress.

The script intentionally stops at AirFogSim's application task record.  It
does not call an offload setter, allocate RBs, calculate a delivery amount, or
claim wireless success.  Exit codes are 0 for all requested software checks,
2 for a documented dependency/communication block, and 1 for an unexpected
failure.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/scripts"))

from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, sha256
from pi_jwm.v2u4real_perception_interface_v1 import (
    MockPerception,
    read_v2u4real_frame,
)
from pi_jwm.v2u4real_airfogsim_adapter_v1 import (
    PerceptionPayloadRecord,
    submit_payload_to_env,
    task_record_dict,
)


SCENE = "2025-07-18-12-53_2"
FRAME_KEY = "1752814590.2"
DATASET_ROOT = ROOT / "code/artifacts/data/v2u4real_r2_01/selected_samples"
CONFIG_PATH = ROOT / "code/scripts/configs/v2u4real_airfogsim_interface_v1.json"
DEFAULT_OUT = ROOT / "code/artifacts/audit/v2u4real_r2_05b_20261027"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _input_hash_rows(bundles: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for agent in ("1", "2"):
        metadata = bundles[agent].file_metadata
        for field in ("pcd_path", "yaml_path"):
            path = DATASET_ROOT / metadata[field]
            rows.append({
                "agent": agent,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256_before": sha256(path),
                "bytes_before": path.stat().st_size,
            })
    return rows


def _input_hash_after(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        path = ROOT / str(row["path"])
        after = sha256(path) if path.is_file() else None
        output.append({**row, "sha256_after": after, "unchanged": after == row["sha256_before"]})
    return output


def read_and_mock() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    bundles = {
        agent: read_v2u4real_frame(DATASET_ROOT, SCENE, FRAME_KEY, agent)
        for agent in ("1", "2")
    }
    before = _input_hash_rows(bundles)
    detector = MockPerception()
    results = {agent: detector.predict(bundle.frame) for agent, bundle in bundles.items()}
    repeated = {agent: detector.predict(bundle.frame) for agent, bundle in bundles.items()}
    if any(results[agent].to_dict() != repeated[agent].to_dict() for agent in results):
        raise InterfaceBlocked("MockPerception is not deterministic for the fixed input")
    payloads = {agent: PerceptionPayloadRecord.from_result(result) for agent, result in results.items()}
    repeated_payloads = {agent: PerceptionPayloadRecord.from_result(result) for agent, result in repeated.items()}
    if payloads != repeated_payloads:
        raise InterfaceBlocked("payload serialization is not deterministic")
    data_receipt = {
        "status": "PASS",
        "dataset": "VJiaLi/V2U4Real",
        "dataset_version": "36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a",
        "split": "train",
        "scene": SCENE,
        "frame_key": FRAME_KEY,
        "frame_key_semantics": "identifier_only_not_physical_timestamp",
        "sensor_agents": {agent: bundles[agent].frame.sensor_id for agent in bundles},
        "frames": {
            agent: bundles[agent].frame.contract_dict()
            for agent in bundles
        },
        "raw_annotation_ids": {
            agent: [str(key) for key in bundles[agent].annotations]
            for agent in bundles
        },
        "physical_alignment_status": "NOT_ESTABLISHED",
        "labels_written": False,
    }
    perception_receipt = {
        "status": "PASS",
        "interface_version": "PIJWM-V2U4Real-Perception-Interface-v1",
        "mock_source": MockPerception.source,
        "prediction_semantics": "deterministic point-cloud envelope fixture; not ground truth; no accuracy reported",
        "results": {agent: results[agent].to_dict() for agent in results},
        "payloads": {agent: payloads[agent].to_dict() for agent in payloads},
        "repeat_equal": True,
    }
    return data_receipt, perception_receipt, before


def build_label_merge_receipt() -> dict[str, object]:
    """Reuse the frozen R2-04C official-source audit for this frame in memory."""
    try:
        import audit_v2u4real_r2_04c_label_merge_policy_v1 as merge_audit
    except ModuleNotFoundError as exc:
        return {
            "status": "BLOCKED_DEPENDENCY",
            "reason": "The frozen R2-04C audit helper could not be imported; no label merge PASS was inferred",
            "missing_module": str(exc),
        }

    official, _ = merge_audit.official_functions()
    _, references = merge_audit.load_r2_04b_reference()
    row = merge_audit.audit_frame(FRAME_KEY, references[FRAME_KEY]["inputs"], official)
    provenance = {
        object_id: {
            "final_source": record["final_source"],
            "vehicle_input_present": record["vehicle_input"] is not None,
            "uav_input_present": record["uav_input"] is not None,
            "overwritten_vehicle": record["overwritten_vehicle"] is not None,
            "interpretation": record["interpretation"],
        }
        for object_id, record in row["final_objects"].items()
    }
    return {
        "status": "PASS",
        "policy": "implementation_faithful_policy_B",
        "policy_boundary": "in-memory annotation audit only; never a prediction and never written as labels",
        "frame_key": FRAME_KEY,
        "vehicle_ids": row["vehicle_ids"],
        "uav_ids": row["uav_ids"],
        "shared_ids": row["shared_ids"],
        "merged_ids": row["merged_ids"],
        "counts": row["counts"],
        "set_checks": row["set_checks"],
        "object_provenance": provenance,
        "raw_annotations_preserved_in_audit": row["raw_annotations_preserved_in_audit"],
        "labels_used_by_mock_perception": False,
        "source_evidence": {
            "repository": merge_audit.OFFICIAL_REPOSITORY,
            "commit": merge_audit.OFFICIAL_COMMIT,
            "path": "code/reference/V2U4Real/opencood/utils/uav_to_single.py",
            "sha256": merge_audit.OFFICIAL_SHA,
        },
    }


def _airfogsim_once(config: dict[str, object], out: Path, seed: int) -> dict[str, object]:
    """Run one real AirFogSimEnv application-ingress smoke."""
    out.mkdir(parents=True, exist_ok=True)
    os.environ["useCUPY"] = "False"
    sys.path.insert(0, str(ROOT / "code/reference/AirFogSim"))
    from run_v2u4real_airfogsim_interface_v1 import make_absolute_replay_manager, prepare_config, snapshots
    from airfogsim import AirFogSimEnv
    from airfogsim.manager.traffic_manager import TrafficManager

    random.seed(seed)
    np.random.seed(seed)
    prepared = prepare_config(config, out)
    env = AirFogSimEnv(deepcopy(prepared["airfogsim"] if "airfogsim" in prepared else prepared), interactive_mode=None)
    try:
        replay_cls = make_absolute_replay_manager(TrafficManager)
        replay = replay_cls(deepcopy(prepared["traffic"]), None, prepared["sumo"]["sumo_net"])
        replay.bind_snapshots(snapshots(prepared["traffic"]["traffic_interval"]))
        env.traffic_manager = replay
        payloads = {}
        for agent, source_node in (("1", "vehicle_0"), ("2", "UAV_0")):
            bundle = read_v2u4real_frame(DATASET_ROOT, SCENE, FRAME_KEY, agent)
            result = MockPerception().predict(bundle.frame)
            payload = PerceptionPayloadRecord.from_result(result)
            task, receipt = submit_payload_to_env(
                env, payload, source_node_id=source_node, target_node_id="RSU_0", deadline=1.0,
            )
            payloads[agent] = {
                "receipt": receipt.to_dict(),
                "task_record_before_step": task_record_dict(task),
                "waiting_count_before_step": len(env.task_manager.getWaitingToOffloadTasksByNodeId(source_node)),
            }
        channel_names = ("channel", "V2I_channel", "V2U_channel", "U2I_channel")
        before_channel = {name: deepcopy(getattr(env, name)) for name in channel_names}
        env.alloc_cpu_callback = lambda *_args, **_kwargs: {}
        env.step()
        after_channel = {name: deepcopy(getattr(env, name)) for name in channel_names}
        for agent, source_node in (("1", "vehicle_0"), ("2", "UAV_0")):
            payloads[agent]["waiting_count_after_step"] = len(env.task_manager.getWaitingToOffloadTasksByNodeId(source_node))
        channel_data_unchanged = all(
            before_channel[name].get("data_size") == after_channel[name].get("data_size")
            for name in channel_names
        )
        if not channel_data_unchanged:
            raise InterfaceBlocked("application-only smoke changed AirFogSim delivered-byte counters")
        return {
            "status": "PASS",
            "environment": "AirFogSimEnv",
            "device": "CPU",
            "env_step_count": 1,
            "simulation_time_after_step": env.simulation_time,
            "payload_tasks": payloads,
            "channel_counters_before": before_channel,
            "channel_counters_after": after_channel,
            "channel_data_size_unchanged": channel_data_unchanged,
            "channel_time_may_advance": True,
            "wireless_delivery_status": "NOT_ATTEMPTED_BLOCKED",
            "software_ingress_status": "PASS",
            "simulation_task_record_status": "PASS",
            "wireless_block_reason": "R1 feature-packet service and communication semantics are not verified; no offload/RB/rate call was made",
        }
    finally:
        env.close()


def run(output: Path = DEFAULT_OUT, label_merge_receipt: Path | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": "PIJWM-V2U4Real-R2-05B-interface-v1",
        "step": "R2-05B",
        "device": "CPU",
        "gpu_used": False,
        "locked_test_accessed": False,
        "training_run": False,
        "formal_detection_evaluation": False,
        "wireless_delivery_claim": False,
        "scene": SCENE,
        "frame_key": FRAME_KEY,
        "data_ingestion": {"status": "NOT_RUN"},
        "label_merge": {"status": "NOT_RUN"},
        "perception_interface": {"status": "NOT_RUN"},
        "airfogsim_software_ingress": {"status": "NOT_RUN"},
        "simulation_task_record": {"status": "NOT_RUN"},
        "wireless_delivery": {"status": "BLOCKED_NOT_ATTEMPTED", "reason": "R1 communication semantic contract remains blocked"},
    }
    hashes_before: list[dict[str, object]] = []
    try:
        data_receipt, perception_receipt, hashes_before = read_and_mock()
        result["data_ingestion"] = data_receipt
        result["perception_interface"] = perception_receipt
        if label_merge_receipt is not None:
            source = Path(label_merge_receipt)
            loaded = json.loads(source.read_text(encoding="utf-8"))
            if loaded.get("status") != "PASS":
                raise InterfaceBlocked("external label merge receipt is not PASS")
            result["label_merge"] = {
                **loaded,
                "reused_receipt": str(source.resolve()),
                "reuse_semantics": "read-only R2-04C evidence reuse; no new label merge decision",
            }
        else:
            result["label_merge"] = build_label_merge_receipt()
        write_json(output / "data_ingestion.json", data_receipt)
        write_json(output / "label_merge.json", result["label_merge"])
        write_json(output / "perception_results.json", perception_receipt)
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        try:
            first = _airfogsim_once(config, output / "airfogsim_run_1", 20261027)
            second = _airfogsim_once(config, output / "airfogsim_run_2", 20261027)
            normalized_first = deepcopy(first)
            normalized_second = deepcopy(second)
            # Task IDs and records should be identical in fresh deterministic runs.
            if normalized_first != normalized_second:
                raise InterfaceBlocked("AirFogSim application smoke is not reproducible")
            result["airfogsim_software_ingress"] = {"status": "PASS", "repeat_equal": True, "evidence": ["airfogsim_run_1", "airfogsim_run_2"]}
            result["simulation_task_record"] = {
                "status": "PASS",
                "application_layer": "AirFogSim TaskManager waiting-to-offload queue",
                "env_step_count": 1,
                "wireless_delivery_status": "BLOCKED_NOT_ATTEMPTED",
            }
            write_json(output / "airfogsim_run_1.json", first)
            write_json(output / "airfogsim_run_2.json", second)
        except ModuleNotFoundError as exc:
            result["airfogsim_software_ingress"] = {
                "status": "BLOCKED_DEPENDENCY",
                "missing_module": str(exc),
                "reason": "The vendored AirFogSim runtime could not be imported; no fake application PASS was recorded",
            }
            result["simulation_task_record"] = {"status": "BLOCKED_DEPENDENCY", "reason": "AirFogSim environment was not available"}
        result["input_hashes_before_after"] = _input_hash_after(hashes_before)
        if not all(row["unchanged"] for row in result["input_hashes_before_after"]):
            raise InterfaceBlocked("input file changed during R2-05B")
        result["data_ingestion"]["input_integrity_status"] = "PASS"
        label_status = result["label_merge"].get("status")
        airfog_status = result["airfogsim_software_ingress"].get("status")
        if label_status != "PASS":
            result["overall_status"] = "PARTIAL_PASS_LABEL_MERGE_BLOCKED"
        elif airfog_status == "PASS":
            result["overall_status"] = "PASS_SOFTWARE_CHAIN_WITH_WIRELESS_BLOCK"
        else:
            result["overall_status"] = "PARTIAL_PASS_AIRFOGSIM_BLOCKED"
    except (OSError, ValueError, TypeError, KeyError, InterfaceBlocked) as exc:
        result["overall_status"] = "BLOCKED_UNEXPECTED_OR_CONTRACT_FAILURE"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result["input_hashes_before_after"] = _input_hash_after(hashes_before) if hashes_before else []
    except Exception as exc:
        result["overall_status"] = "BLOCKED_UNEXPECTED_OR_CONTRACT_FAILURE"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result["input_hashes_before_after"] = _input_hash_after(hashes_before) if hashes_before else []
    finally:
        result.setdefault("input_hashes_before_after", _input_hash_after(hashes_before) if hashes_before else [])
        write_json(output / "acceptance.json", result)
    print(json.dumps({key: result.get(key) for key in ("overall_status", "data_ingestion", "label_merge", "perception_interface", "airfogsim_software_ingress", "simulation_task_record", "wireless_delivery")}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label-merge-receipt", type=Path, default=None,
                        help="reuse a previously PASSed R2-04C read-only receipt when the runtime lacks its optional dependency")
    args = parser.parse_args()
    result = run(args.output, args.label_merge_receipt)
    return 0 if result.get("overall_status") == "PASS_SOFTWARE_CHAIN_WITH_WIRELESS_BLOCK" else 2 if result.get("overall_status") in {"PARTIAL_PASS_AIRFOGSIM_BLOCKED", "PARTIAL_PASS_LABEL_MERGE_BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
