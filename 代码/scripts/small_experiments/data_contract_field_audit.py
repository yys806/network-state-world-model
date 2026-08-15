from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
AIRFOGSIM_ARTIFACT_ROOT = CODE_ROOT / "artifacts" / "experiments" / "airfogsim_v0"
DEFAULT_RAW_ROOT = AIRFOGSIM_ARTIFACT_ROOT / "reports" / "multiseed_raw_active_heavy_v2_60seed_20260619"
DEFAULT_ACTION_ROOT = AIRFOGSIM_ARTIFACT_ROOT / "reports" / "strict_action_active_heavy_v2_60seed_20260619"
DEFAULT_WORLD_MODEL_NPZ = AIRFOGSIM_ARTIFACT_ROOT / "datasets" / "world_model_dataset_active_heavy_v2_60seed_20260619" / "world_model_dataset_v0_samples.npz"
DEFAULT_EDGE_ACTION_SUMMARY = AIRFOGSIM_ARTIFACT_ROOT / "datasets" / "edge_action_active_heavy_v2_60seed_20260619" / "edge_action_v0_summary.json"
DEFAULT_CANDIDATE_DATA_ROOT = CODE_ROOT / "artifacts" / "literature"
DEFAULT_OUTPUT_DIR = CODE_ROOT / "artifacts" / "small_experiments" / "exp01_data_contract_field_audit"

RAW_FILES = ("node_states.csv", "link_states.csv", "task_states.csv")
ACTION_FILES = (
    "offload_actions.csv",
    "return_actions.csv",
    "rb_actions.csv",
    "cpu_actions.csv",
    "uav_mobility_actions.csv",
)
CANDIDATE_DATA_EXTENSIONS = {".h5", ".hdf5", ".mat", ".pcap", ".parquet", ".npy", ".npz", ".csv"}


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def scan_csv_runs(raw_root: Path) -> dict:
    raw_root = Path(raw_root)
    seed_dirs = sorted(path for path in raw_root.glob("seed_*") if path.is_dir())
    files = {
        filename: {
            "columns": [],
            "rows": 0,
            "nonempty_seeds": 0,
            "missing_seeds": 0,
            "blank_counts": Counter(),
        }
        for filename in RAW_FILES
    }
    schema_sets: dict[str, set[tuple[str, ...]]] = {filename: set() for filename in RAW_FILES}
    dag_probabilities: list[float] = []
    profile_cache: dict[str, float | None] = {}
    node_types: Counter[str] = Counter()
    link_types: Counter[str] = Counter()
    lifecycle_states: Counter[str] = Counter()
    node_times_by_seed: dict[str, set[float]] = defaultdict(set)

    for seed_dir in seed_dirs:
        for filename in RAW_FILES:
            path = seed_dir / filename
            if not path.exists():
                files[filename]["missing_seeds"] += 1
                continue
            row_count = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                schema_sets[filename].add(tuple(columns))
                for row in reader:
                    row_count += 1
                    for key, value in row.items():
                        if value == "":
                            files[filename]["blank_counts"][key] += 1
                    if filename == "node_states.csv":
                        node_types[row.get("node_type", "")] += 1
                        try:
                            node_times_by_seed[seed_dir.name].add(float(row.get("time", "nan")))
                        except ValueError:
                            pass
                        profile_text = row.get("task_profile", "")
                        if profile_text not in profile_cache:
                            try:
                                profile = json.loads(profile_text) if profile_text else {}
                                probability = profile.get("dag_edge_prob")
                                profile_cache[profile_text] = float(probability) if probability is not None else None
                            except (TypeError, ValueError, json.JSONDecodeError):
                                profile_cache[profile_text] = None
                        probability = profile_cache[profile_text]
                        if probability is not None:
                            dag_probabilities.append(probability)
                    elif filename == "link_states.csv":
                        link_types[row.get("link_type", "")] += 1
                    elif filename == "task_states.csv":
                        lifecycle_states[row.get("lifecycle_state", "")] += 1
            files[filename]["rows"] += row_count
            files[filename]["nonempty_seeds"] += int(row_count > 0)

    sampling_steps: list[float] = []
    for times in node_times_by_seed.values():
        ordered = sorted(times)
        sampling_steps.extend(round(b - a, 9) for a, b in zip(ordered, ordered[1:]) if b > a)
    for filename in RAW_FILES:
        schemas = schema_sets[filename]
        files[filename]["columns"] = list(next(iter(schemas))) if len(schemas) == 1 else []
        files[filename]["schema_variants"] = [list(schema) for schema in sorted(schemas)]
        files[filename]["blank_counts"] = dict(files[filename]["blank_counts"])

    trajectory_keys = {"trajectory_id", "run_id"}
    identifier_sets = [trajectory_keys.intersection(files[name]["columns"]) for name in RAW_FILES]
    shared_identifiers = set.intersection(*identifier_sets) if identifier_sets else set()

    return {
        "root": str(raw_root.resolve()),
        "exists": raw_root.exists(),
        "seed_count": len(seed_dirs),
        "seed_names": [path.name for path in seed_dirs],
        "files": files,
        "schemas_consistent": bool(seed_dirs) and all(len(schema_sets[name]) == 1 for name in RAW_FILES),
        "all_required_files_present": bool(seed_dirs) and all(files[name]["missing_seeds"] == 0 for name in RAW_FILES),
        "all_required_files_nonempty": bool(seed_dirs) and all(files[name]["nonempty_seeds"] == len(seed_dirs) for name in RAW_FILES),
        "shared_trajectory_identifiers": sorted(shared_identifiers),
        "dag_edge_probability_min": min(dag_probabilities) if dag_probabilities else None,
        "dag_edge_probability_max": max(dag_probabilities) if dag_probabilities else None,
        "dag_nonzero_configured": any(value > 0 for value in dag_probabilities),
        "sampling_interval_mode": Counter(sampling_steps).most_common(1)[0][0] if sampling_steps else None,
        "node_types": dict(node_types),
        "link_types": dict(link_types),
        "lifecycle_states": dict(lifecycle_states),
    }


def scan_action_runs(action_root: Path) -> dict:
    action_root = Path(action_root)
    seed_dirs = sorted(path for path in action_root.glob("seed_*") if path.is_dir())
    files = {}
    for filename in ACTION_FILES:
        schemas: set[tuple[str, ...]] = set()
        rows = 0
        nonempty_seeds = 0
        missing_seeds = 0
        for seed_dir in seed_dirs:
            path = seed_dir / filename
            if not path.exists():
                missing_seeds += 1
                continue
            row_count = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                schemas.add(tuple(reader.fieldnames or []))
                for _ in reader:
                    row_count += 1
            rows += row_count
            nonempty_seeds += int(row_count > 0)
        files[filename] = {
            "columns": list(next(iter(schemas))) if len(schemas) == 1 else [],
            "schema_variants": [list(schema) for schema in sorted(schemas)],
            "rows": rows,
            "nonempty_seeds": nonempty_seeds,
            "missing_seeds": missing_seeds,
        }
    core_files = ("offload_actions.csv", "rb_actions.csv", "cpu_actions.csv")
    trajectory_keys = {"trajectory_id", "run_id"}
    core_identifier_sets = [trajectory_keys.intersection(files[name]["columns"]) for name in core_files]
    shared_identifiers = set.intersection(*core_identifier_sets) if core_identifier_sets else set()
    core_nonempty_all_seeds = bool(seed_dirs) and all(files[name]["nonempty_seeds"] == len(seed_dirs) for name in core_files)
    return {
        "root": str(action_root.resolve()),
        "exists": action_root.exists(),
        "seed_count": len(seed_dirs),
        "files": files,
        "schemas_consistent": bool(seed_dirs) and all(len(info["schema_variants"]) == 1 for info in files.values()),
        "all_required_files_present": bool(seed_dirs) and all(info["missing_seeds"] == 0 for info in files.values()),
        "core_action_files_nonempty_for_all_seeds": core_nonempty_all_seeds,
        "shared_trajectory_identifiers": sorted(shared_identifiers),
        "shared_trajectory_identifier_present": bool(shared_identifiers),
    }


def _npy_shape_from_zip(archive: zipfile.ZipFile, member: str) -> tuple[list[int], str]:
    with archive.open(member, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
    return list(shape), str(dtype)


def scan_npz_metadata(npz_path: Path) -> dict:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return {"path": str(npz_path.resolve()), "exists": False, "keys": [], "arrays": {}, "feature_names": {}}
    arrays = {}
    with zipfile.ZipFile(npz_path, "r") as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            key = Path(member).stem
            shape, dtype = _npy_shape_from_zip(archive, member)
            arrays[key] = {"shape": shape, "dtype": dtype}
    feature_names = {}
    feature_keys = [key for key in arrays if key.endswith("features")]
    if feature_keys:
        with np.load(npz_path, allow_pickle=False) as payload:
            for key in feature_keys:
                feature_names[key] = [str(value) for value in payload[key].tolist()]
    return {
        "path": str(npz_path.resolve()),
        "exists": True,
        "size_bytes": npz_path.stat().st_size,
        "keys": sorted(arrays),
        "arrays": arrays,
        "feature_names": feature_names,
    }


def scan_candidate_data(candidate_root: Path) -> dict:
    candidate_root = Path(candidate_root)
    if not candidate_root.exists():
        return {"root": str(candidate_root.resolve()), "exists": False, "data_files": [], "verified_data_files": [], "paper_files": []}
    data_files = []
    paper_files = []
    for path in candidate_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in CANDIDATE_DATA_EXTENSIONS:
            data_files.append(str(path.resolve()))
        elif suffix == ".pdf":
            paper_files.append(str(path.resolve()))
    return {
        "root": str(candidate_root.resolve()),
        "exists": True,
        "data_files": sorted(data_files),
        "verified_data_files": [],
        "paper_files": sorted(paper_files),
    }


def scan_action_projection(summary_path: Path) -> dict:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return {"path": str(summary_path.resolve()), "exists": False, "rates": {}, "minimum_core_match_rate": 0.0}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("match_summary", [])
    rates = {}
    for family in ("offload", "rb", "cpu", "return"):
        total = sum(float(row.get(f"{family}_total", 0)) for row in rows)
        matched = sum(float(row.get(f"{family}_matched", 0)) for row in rows)
        rates[family] = matched / total if total > 0 else 0.0
    core_rates = [rates[name] for name in ("offload", "rb", "cpu")]
    return {
        "path": str(summary_path.resolve()),
        "exists": True,
        "rates": rates,
        "minimum_core_match_rate": min(core_rates) if core_rates else 0.0,
        "complete_core_projection": bool(core_rates) and min(core_rates) >= 0.999,
    }


def _columns(inventory: dict, layer: str, filename: str) -> set[str]:
    return set(inventory.get(layer, {}).get("files", {}).get(filename, {}).get("columns", []))


def build_field_contract(inventory: dict) -> list[dict]:
    rows: list[dict] = []

    def add(
        field_id: str,
        category: str,
        definition: str,
        status: str,
        source_layer: str,
        source_fields: str,
        derivation: str = "",
        quality_flag: str = "complete",
        required_for_v1: bool = False,
        evidence: str = "",
    ) -> None:
        rows.append(
            {
                "field_id": field_id,
                "category": category,
                "definition": definition,
                "status": status,
                "source_layer": source_layer,
                "source_fields": source_fields,
                "derivation": derivation,
                "quality_flag": quality_flag,
                "required_for_v1": required_for_v1,
                "evidence": evidence,
            }
        )

    def direct_raw(field_id: str, category: str, definition: str, filename: str, fields: Iterable[str], **kwargs) -> None:
        required = set(fields)
        present = required.issubset(_columns(inventory, "raw", filename))
        add(
            field_id,
            category,
            definition,
            "direct" if present else "missing",
            f"raw/{filename}" if present else "none",
            ";".join(fields) if present else "",
            evidence=f"schema fields={sorted(required)}",
            **kwargs,
        )

    def direct_action(field_id: str, category: str, definition: str, filename: str, fields: Iterable[str], **kwargs) -> None:
        required = set(fields)
        info = inventory.get("actions", {}).get("files", {}).get(filename, {})
        present = required.issubset(set(info.get("columns", [])))
        quality = kwargs.pop("quality_flag", "complete")
        if present and int(info.get("rows", 0)) == 0:
            quality = "empty_observation"
        add(
            field_id,
            category,
            definition,
            "direct" if present else "missing",
            f"strict_actions/{filename}" if present else "none",
            ";".join(fields) if present else "",
            quality_flag=quality,
            evidence=f"rows={info.get('rows', 0)}",
            **kwargs,
        )

    def raw_has(filename: str, fields: Iterable[str]) -> bool:
        return set(fields).issubset(_columns(inventory, "raw", filename))

    def derived_raw(
        field_id: str,
        category: str,
        definition: str,
        filename: str,
        fields: Iterable[str],
        derivation: str,
        **kwargs,
    ) -> None:
        present = raw_has(filename, fields)
        add(
            field_id,
            category,
            definition,
            "derivable" if present else "missing",
            f"raw/{filename}" if present else "none",
            ";".join(fields) if present else "",
            derivation if present else "",
            evidence=f"required fields={sorted(set(fields))}",
            **kwargs,
        )

    direct_raw("timestamp", "metadata", "仿真时间戳", "node_states.csv", ["time"], required_for_v1=True)
    direct_raw("seed", "metadata", "随机种子", "node_states.csv", ["seed"])
    add("sampling_interval", "metadata", "采样间隔", "derivable" if inventory.get("raw", {}).get("sampling_interval_mode") else "missing", "raw/node_states.csv", "time", "相邻唯一时间戳之差的众数", required_for_v1=True, evidence=f"mode={inventory.get('raw', {}).get('sampling_interval_mode')}")
    add("run_configuration", "metadata", "可复现实验配置快照或哈希", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)
    add("units_and_radio_metadata", "metadata", "单位、频段、带宽、天线和协议配置", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)
    add("license_and_source_version", "metadata", "许可证和数据/仿真器版本", "missing", "none", "", quality_flag="not_exported")

    direct_raw("node_identity", "physical_node", "物理节点唯一标识与类型", "node_states.csv", ["node_id", "node_type"], required_for_v1=True)
    direct_raw("node_position", "physical_node", "三维位置", "node_states.csv", ["node_id", "time", "x", "y", "z"], required_for_v1=True)
    direct_raw("node_speed", "physical_node", "速度标量", "node_states.csv", ["speed"])
    derived_raw("velocity_vector", "physical_node", "三维速度向量", "node_states.csv", ["node_id", "time", "x", "y", "z"], "相邻位置差除以时间差", required_for_v1=True)
    direct_raw("node_acceleration", "physical_node", "加速度标量", "node_states.csv", ["acceleration"])
    direct_raw("cpu_capacity", "resource", "节点CPU容量", "node_states.csv", ["node_id", "cpu"], quality_flag="ambiguous_zero_fill", required_for_v1=True)
    direct_raw("storage_capacity", "resource", "节点存储容量", "node_states.csv", ["node_id", "storage"], quality_flag="ambiguous_zero_fill")
    add("node_energy", "resource", "节点能量状态或累计能耗", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)

    direct_raw("physical_edge_endpoints", "physical_edge", "物理链路发送端与接收端", "link_states.csv", ["tx_id", "rx_id", "link_type"], quality_flag="partial_link_families", required_for_v1=True)
    direct_raw("link_distance", "physical_edge", "链路距离", "link_states.csv", ["distance"])
    direct_raw("link_rate", "physical_edge", "所有RB速率之和", "link_states.csv", ["rate_sum"], required_for_v1=True)
    direct_raw("csi_mean", "physical_edge", "AirFogSim CSI均值", "link_states.csv", ["csi_mean"], quality_flag="aggregate_only", required_for_v1=True)
    add("full_csi_or_sinr", "physical_edge", "完整CSI或SINR/RSRP/RSRQ", "missing", "none", "", quality_flag="not_exported")
    direct_raw("link_active_task_count", "physical_edge", "链路活动任务数", "link_states.csv", ["active_task_count"])
    direct_raw("allocated_rb_count", "resource", "链路已分配RB数量", "link_states.csv", ["allocated_rb_count"])
    add("total_rb_capacity", "resource", "每时刻可用RB总容量", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)

    direct_raw("task_identity", "information_node", "任务唯一标识", "task_states.csv", ["task_id"], required_for_v1=True)
    direct_raw("task_source_node", "information_node", "任务产生节点", "task_states.csv", ["task_node_id"], required_for_v1=True)
    direct_raw("task_current_node", "information_node", "任务当前所在节点", "task_states.csv", ["current_node_id"], required_for_v1=True)
    direct_raw("task_execution_target", "information_node", "任务被分配的执行节点", "task_states.csv", ["assigned_to"], required_for_v1=True)
    direct_raw("task_input_size", "information_node", "任务输入数据量", "task_states.csv", ["task_size"], required_for_v1=True)
    direct_raw("task_compute_demand", "information_node", "任务计算需求", "task_states.csv", ["task_cpu"], required_for_v1=True)
    add("task_return_size", "information_node", "任务结果回传数据量", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)
    direct_raw("task_deadline", "information_node", "任务截止期", "task_states.csv", ["deadline"], required_for_v1=True)
    direct_raw("task_priority", "information_node", "任务优先级", "task_states.csv", ["priority"])
    direct_raw("task_arrival_time", "information_node", "任务到达时间", "task_states.csv", ["arrival_time"], required_for_v1=True)
    direct_raw("task_transmitted_size", "information_node", "累计已传输量", "task_states.csv", ["transmitted_size"])
    direct_raw("task_computed_size", "information_node", "累计已计算量", "task_states.csv", ["computed_size"])
    direct_raw("task_lifecycle", "information_node", "任务生命周期状态", "task_states.csv", ["lifecycle_state"], required_for_v1=True)
    direct_raw("task_failure_reason", "metric", "任务失败原因", "task_states.csv", ["failure_reason"])
    derived_raw("task_remaining_input", "information_node", "剩余待传输入量", "task_states.csv", ["task_size", "transmitted_size"], "max(task_size-transmitted_size,0)", required_for_v1=True)
    derived_raw("task_remaining_compute", "information_node", "剩余计算量", "task_states.csv", ["task_cpu", "computed_size"], "max(task_cpu-computed_size,0)", required_for_v1=True)
    add("task_dag_nodes", "information_graph", "严格信息图任务节点", "direct" if "task_id" in _columns(inventory, "raw", "task_states.csv") else "missing", "raw/task_states.csv", "task_id", quality_flag="independent_tasks_only", required_for_v1=True)
    dag_quality = "zero_configured" if inventory.get("raw", {}).get("dag_edge_probability_max") == 0 else "not_exported"
    add("task_dag_edges", "information_graph", "任务父子依赖边", "missing", "none", "", quality_flag=dag_quality, required_for_v1=True, evidence=f"dag_edge_probability_max={inventory.get('raw', {}).get('dag_edge_probability_max')}")
    add("dependency_data_volume", "information_graph", "每条依赖边传递的数据量", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)

    direct_action("offload_action_task_level", "action", "逐任务卸载源与目标", "offload_actions.csv", ["time", "task_id", "source_node_id", "target_node_id"], required_for_v1=True)
    direct_action("rb_action_task_level", "action", "逐任务具体RB集合", "rb_actions.csv", ["time", "task_id", "rb_count", "rb_indices"], required_for_v1=True)
    direct_action("cpu_action_task_level", "action", "逐任务CPU分配", "cpu_actions.csv", ["time", "task_id", "assigned_to", "allocated_cpu"], required_for_v1=True)
    direct_action("return_action_task_level", "action", "逐任务返回下一跳", "return_actions.csv", ["time", "task_id", "current_node_id", "return_target_id"])
    direct_action("uav_mobility_action", "action", "UAV移动角度、俯仰与速度命令", "uav_mobility_actions.csv", ["time", "uav_id", "angle", "phi", "speed"])
    add("action_feasibility_mask", "action", "动作可行域或非法动作掩码", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)
    projection_rate = float(inventory.get("action_projection", {}).get("minimum_core_match_rate", 0.0))
    raw_ids = set(inventory.get("raw", {}).get("shared_trajectory_identifiers", []))
    action_ids = set(inventory.get("actions", {}).get("shared_trajectory_identifiers", []))
    paired = bool(raw_ids.intersection(action_ids))
    projection_complete = projection_rate >= 0.999 and paired
    add("aligned_core_action_sequence", "action", "与状态轨迹严格配对的核心动作序列", "direct" if projection_rate > 0 else "missing", "processed/edge_action", "edge_a_hist;edge_a_future", quality_flag="complete" if projection_complete else "partial", required_for_v1=True, evidence=f"minimum_core_match_rate={projection_rate:.6f};shared_trajectory_id={paired}")

    direct_raw("mn_source_mapping", "cross_graph", "任务到源物理节点映射", "task_states.csv", ["task_id", "task_node_id"], required_for_v1=True)
    direct_raw("mn_host_mapping", "cross_graph", "任务到当前宿主节点映射", "task_states.csv", ["task_id", "current_node_id"], required_for_v1=True)
    direct_raw("mn_exec_mapping", "cross_graph", "任务到执行节点映射", "task_states.csv", ["task_id", "assigned_to"], quality_flag="conditional_by_lifecycle", required_for_v1=True)
    add("mn_return_mapping", "cross_graph", "任务返回目的映射", "direct" if "return_target_id" in _columns(inventory, "actions", "return_actions.csv") else "missing", "strict_actions/return_actions.csv", "task_id;return_target_id", quality_flag="next_hop_only", required_for_v1=True)
    add("me_input_path", "cross_graph", "任务输入在物理图上的完整路径", "missing", "none", "", quality_flag="endpoints_only", required_for_v1=True)
    add("me_output_path", "cross_graph", "任务结果返回的完整路径", "missing", "none", "", quality_flag="next_hop_only", required_for_v1=True)
    add("ep_dependency_path", "cross_graph", "信息依赖边对应的物理传输路径", "missing", "none", "", quality_flag=dag_quality, required_for_v1=True)

    derived_raw("metric_completion_rate", "metric", "任务完成率", "task_states.csv", ["task_id", "lifecycle_state"], "最终完成任务数/终止任务数", required_for_v1=True)
    derived_raw("metric_task_latency", "metric", "任务完成时延", "task_states.csv", ["time", "task_id", "arrival_time", "lifecycle_state"], "首次进入finished的时间-arrival_time", quality_flag="derived_first_observation", required_for_v1=True)
    derived_raw("metric_deadline_violation", "metric", "超期率", "task_states.csv", ["deadline", "failure_reason", "lifecycle_state", "time"], "超期失败或完成时延超过deadline", required_for_v1=True)
    direct_raw("metric_throughput", "metric", "链路吞吐率", "link_states.csv", ["rate_sum"], required_for_v1=True)
    add("metric_resource_utilization", "metric", "RB和CPU利用率", "missing", "none", "", quality_flag="missing_capacity_or_pairing", required_for_v1=True)
    add("metric_energy", "metric", "通信与计算能耗", "missing", "none", "", quality_flag="not_exported", required_for_v1=True)
    derived_raw("metric_fairness", "metric", "任务或节点服务公平性", "task_states.csv", ["task_node_id", "task_id", "lifecycle_state"], "按节点完成量计算Jain指数")
    add("missing_value_mask", "target", "缺失值与真实零值区分掩码", "missing", "none", "", quality_flag="zeros_used_for_missing", required_for_v1=True)
    verified_real_data = inventory.get("candidates", {}).get("verified_data_files", [])
    unverified_real_data = inventory.get("candidates", {}).get("data_files", [])
    add("real_measurement_holdout", "external_data", "本地可读取且完成语义核验的真实无线测量留出集", "direct" if verified_real_data else "missing", "candidate_data" if verified_real_data else "none", "verified local files" if verified_real_data else "", quality_flag="complete" if verified_real_data else ("unverified_local_files" if unverified_real_data else "papers_only"))

    identifiers = [row["field_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Field contract contains duplicate field_id values.")
    return rows


def evaluate_readiness(contract: list[dict], inventory: dict) -> dict:
    allowed_statuses = {"direct", "derivable", "missing"}
    all_classified = bool(contract) and all(row.get("status") in allowed_statuses for row in contract)
    blocking_quality = {
        "partial",
        "zero_configured",
        "ambiguous_zero_fill",
        "partial_link_families",
        "endpoints_only",
        "next_hop_only",
        "empty_observation",
        "zeros_used_for_missing",
    }
    blocking_fields = [
        row["field_id"]
        for row in contract
        if row.get("required_for_v1")
        and (row.get("status") == "missing" or row.get("quality_flag") in blocking_quality)
    ]
    actions = inventory.get("actions", {})
    seed_count = int(actions.get("seed_count", 0))
    core_files = ("offload_actions.csv", "rb_actions.csv", "cpu_actions.csv")
    core_nonempty = bool(actions.get("core_action_files_nonempty_for_all_seeds"))
    if "core_action_files_nonempty_for_all_seeds" not in actions:
        core_nonempty = bool(seed_count) and all(
            int(actions.get("files", {}).get(name, {}).get("nonempty_seeds", 0)) == seed_count for name in core_files
        )
    action_files_ready = bool(actions.get("all_required_files_present")) and core_nonempty
    checks = [
        {"name": "all_requirements_classified", "passed": all_classified},
        {"name": "raw_schemas_consistent", "passed": bool(inventory.get("raw", {}).get("schemas_consistent"))},
        {"name": "raw_files_present_and_nonempty", "passed": bool(inventory.get("raw", {}).get("all_required_files_present")) and bool(inventory.get("raw", {}).get("all_required_files_nonempty"))},
        {"name": "strict_action_schemas_consistent", "passed": bool(inventory.get("actions", {}).get("schemas_consistent"))},
        {"name": "strict_action_files_present_and_core_nonempty", "passed": action_files_ready},
        {"name": "processed_npz_readable", "passed": bool(inventory.get("processed", {}).get("exists"))},
        {"name": "candidate_real_data_assessed", "passed": "data_files" in inventory.get("candidates", {})},
    ]
    audit_completed = all(check["passed"] for check in checks)
    return {
        "audit_completed": audit_completed,
        "formal_training_ready": audit_completed and not blocking_fields,
        "checks": checks,
        "blocking_fields": blocking_fields,
        "counts": {
            "requirements": len(contract),
            "direct": sum(row["status"] == "direct" for row in contract),
            "derivable": sum(row["status"] == "derivable" for row in contract),
            "missing": sum(row["status"] == "missing" for row in contract),
            "blocking": len(blocking_fields),
        },
        "evidence_scope": "retained_airfogsim_exports_and_local_assets_only",
        "airfogsim_is_data_source_not_framework": True,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coverage_rows(contract: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], int] = Counter((row["category"], row["status"]) for row in contract)
    return [
        {"category": category, "status": status, "count": count}
        for (category, status), count in sorted(grouped.items())
    ]


def _report_text(inventory: dict, contract: list[dict], validation: dict) -> str:
    counts = validation["counts"]
    dag_max = inventory["raw"].get("dag_edge_probability_max")
    projection = inventory["action_projection"]
    rates = projection.get("rates", {})
    missing = [row for row in contract if row["status"] == "missing"]
    lines = [
        "# 小实验01：数据契约与字段审计报告",
        "",
        "## 结论",
        "",
        f"- 审计执行：{'完成' if validation['audit_completed'] else '未完成'}",
        f"- PI-JWM v1正式训练数据就绪：{'是' if validation['formal_training_ready'] else '否'}",
        f"- 契约字段：{counts['requirements']}项；直接提供{counts['direct']}项，可推导{counts['derivable']}项，缺失{counts['missing']}项。",
        f"- 当前DAG边概率最大值：{dag_max}；当前CSV未导出任务依赖边。",
        f"- 核心动作投影最低匹配率：{projection.get('minimum_core_match_rate', 0.0):.4f}。",
        "- 审计完成不等于数据已满足正式训练条件。",
        "- AirFogSim是数据源，不是PI-JWM框架。",
        "- 候选真实数据当前只有文献证据，不计入本地字段覆盖。" if not inventory["candidates"].get("data_files") else "- 已发现本地候选真实数据文件，但其语义仍需单独核验。",
        "",
        "## 已确认的数据",
        "",
        f"- 原始状态：{inventory['raw']['seed_count']}个seed；节点{inventory['raw']['files']['node_states.csv']['rows']}行，链路{inventory['raw']['files']['link_states.csv']['rows']}行，任务{inventory['raw']['files']['task_states.csv']['rows']}行。",
        f"- 严格动作：卸载{inventory['actions']['files']['offload_actions.csv']['rows']}行，RB{inventory['actions']['files']['rb_actions.csv']['rows']}行，CPU{inventory['actions']['files']['cpu_actions.csv']['rows']}行，返回{inventory['actions']['files']['return_actions.csv']['rows']}行，UAV移动{inventory['actions']['files']['uav_mobility_actions.csv']['rows']}行。",
        f"- 动作投影：offload={rates.get('offload', 0.0):.4f}，RB={rates.get('rb', 0.0):.4f}，CPU={rates.get('cpu', 0.0):.4f}，return={rates.get('return', 0.0):.4f}。",
        "",
        "## 关键阻断项",
        "",
    ]
    for field_id in validation["blocking_fields"]:
        row = next(item for item in contract if item["field_id"] == field_id)
        lines.append(f"- `{field_id}`：{row['definition']}（{row['status']}；{row['quality_flag']}）")
    lines.extend(["", "## 全部缺失字段", ""])
    for row in missing:
        lines.append(f"- `{row['field_id']}`：{row['definition']}。")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "本实验只审计项目中保留的AirFogSim导出、严格动作日志、加工NPZ和本地数据资产。仿真器内部存在接口不代表字段已写入当前数据；论文描述某个数据集也不代表本地已经取得并验证其字段。实验没有修改schema、没有补造标签、没有训练模型。",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    raw_root: Path,
    action_root: Path,
    world_model_npz: Path,
    edge_action_summary: Path,
    candidate_data_root: Path,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "raw": scan_csv_runs(raw_root),
        "actions": scan_action_runs(action_root),
        "processed": scan_npz_metadata(world_model_npz),
        "action_projection": scan_action_projection(edge_action_summary),
        "candidates": scan_candidate_data(candidate_data_root),
    }
    contract = build_field_contract(inventory)
    validation = evaluate_readiness(contract, inventory)

    (output_dir / "source_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        output_dir / "field_contract.csv",
        contract,
        ["field_id", "category", "definition", "status", "source_layer", "source_fields", "derivation", "quality_flag", "required_for_v1", "evidence"],
    )
    _write_csv(output_dir / "coverage_summary.csv", _coverage_rows(contract), ["category", "status", "count"])
    (output_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(_report_text(inventory, contract, validation), encoding="utf-8")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current PI-JWM data fields without modifying source data.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--action-root", type=Path, default=DEFAULT_ACTION_ROOT)
    parser.add_argument("--world-model-npz", type=Path, default=DEFAULT_WORLD_MODEL_NPZ)
    parser.add_argument("--edge-action-summary", type=Path, default=DEFAULT_EDGE_ACTION_SUMMARY)
    parser.add_argument("--candidate-data-root", type=Path, default=DEFAULT_CANDIDATE_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        raw_root=args.raw_root,
        action_root=args.action_root,
        world_model_npz=args.world_model_npz,
        edge_action_summary=args.edge_action_summary,
        candidate_data_root=args.candidate_data_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
