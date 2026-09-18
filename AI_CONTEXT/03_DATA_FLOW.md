# 数据流与张量合同

> 2026-09-18 Step 1 结论：下列 v5 数据流是旧协议下可追溯资产，尚未满足新 `00–06` 的 Agent/Communication/Task-Agent 关系和 Route/Comm/Comp/UAV 四类动作合同。稳定 ID/index、mask、split 和 train-only normalization 原则可复用；新 schema 尚未冻结。

## Step 2.1 real raw flow

One non-locked AirFogSim trajectory verified `Decision_t -> Route/Comm/Comp/UAV Mobility -> env.step() -> Outcome_t -> Decision_{t+1}`. Vehicle motion is advanced by SUMO; UAV mobility is the planner action. Entity observation `heading` is vehicle degree from SUMO and UAV rad from AirFogSim; UAV action uses `azimuth_rad`. Evidence: `code/artifacts/protocols/pi_jwm_raw_single_decision_step_real_airfogsim_v2_20260919/`.

## 总流程

```text
AirFogSim 参考仿真运行
→ formal trajectory / bundle
→ tensorization 与非锁定 split
→ 因果运动字段派生
→ FormalAirFogSimWindowDataset
→ history + static + future_action + target
→ 双图底座与实体 RSSM
→ prior 20 步输出
→ FormalMetricAccumulator
→ checkpoint gate、manifest 与独立 acceptance
```

Source of truth：数据读取与访问守卫见 `code/src/pi_jwm/formal_airfogsim_dataset_v1.py`、`formal_airfogsim_window_v1.py`；当前 tensor 事实见其 manifest 和 validation report。

## 旧协议正式 tensor（Historical / Archived）

- 路径：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`
- history/horizon：8/20。
- train/validation/calibration：9828/3276/1638 windows。
- 只包含 non-locked split；validation report 记录 `locked_test_not_materialized=true`、`locked_test_accessed=false`。
- normalization 来源固定为 train。
- 因果运动合同：`causal_backward_difference_v1`，由历史位置向后差分派生，不暴露未来运动。
- 构建入口：`code/scripts/build_formal_causal_motion_tensor_v1.py::build_causal_motion_tensor()`。

## Window 输出结构

`code/src/pi_jwm/formal_airfogsim_window_v1.py::FormalAirFogSimWindowDataset.__getitem__()` 返回：

- `history`：过去实体状态、presence/mask、DAG、聚合链路标签和六维 `node_motion_state/mask`。
- `static`：节点类型、端点索引、flow/task/DAG 有效性和跨图映射。
- `future_action`：`task_action`、presence、目标节点和显式源节点。
- `target`：未来 20 步状态、事件、DAG、链路活动/速率/RB 等评价标签。
- `metadata`：sample、split、合同等追溯信息。

## 核心对象含义

- node state 7维：位置、运动、计算能力与存储等；具体名称和单位来自 `code/src/pi_jwm/airfogsim_tensor_v2.py` 与 metric registry。
- physical edge state 5维：distance、csi_mean、rate_sum、active_task_count、allocated_rb_count。
- flow state 5维：total_data、remaining_data、delivered_cumulative、delivered_this_slot、age。
- task state 8维：数据量、工作量、时间和任务进度相关字段。
- 信息 agent 当前没有独立观测张量，使用所附着物理节点的编码表示。

## 训练数据流

DataLoader → 模型 prior/teacher 输出 → `formal_world_model_loss()`。base 阶段训练确定性底座；RSSM 阶段冻结 base 并训练实体级随机动力学。类别权重由 train 数据计算，validation/calibration 不反向进入训练统计。

## 评价数据流

`code/src/pi_jwm/formal_world_model_metrics_v1.py::FormalMetricAccumulator` 按 horizon、mask、单位和分母累计连续状态、二分类事件、不确定性和系统指标。`code/src/pi_jwm/formal_p4_gate_v1.py::evaluate_p4_checkpoint_gates()` 将当前 checkpoint 与冻结基线比较。正式数字最终必须以 acceptance JSON 重算结果为准。

## 安全边界

- `require_split_access()` 默认拒绝 locked split；正式 runner 还拒绝路径中出现 `locked_test`。
- `locked_test_accessed=false`。
- `formal_performance_claim_ready=false`。
- `step2_started=false`；没有生成新定义 tensor。

Unverified：未在当前 tensor manifest、loader 或模型实际读路径出现的字段，不得推断为当前模型输入。

## 2026-09-18 STEP 2 状态

单决策步 `Decision_t → Action_t → Execution_t → Outcome_t → Decision_{t+1}` 合同已冻结。四类动作明确为 Route、Comm、Comp、UAV Mobility；车辆运动由 SUMO 推进，不属于 PI-JWM planner action。合同和验证证据见 `docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`、`docs/implementation_records/STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md`。当前只完成最小 setter 闭环和历史三类真实 ledger 证据，完整真实四类轨迹尚未验收。
