# 数据流与张量合同

> 2026-09-18 Step 1 结论：下列 v5 数据流是旧协议下可追溯资产，尚未满足新 `00–06` 的 Agent/Communication/Task-Agent 关系和 Route/Comm/Comp/UAV 四类动作合同。稳定 ID/index、mask、split 和 train-only normalization 原则可复用；新 schema 尚未冻结。

## Step 2.2 real raw flow

One non-locked AirFogSim trajectory verified six repeated `Decision_t -> Route/Comm/Comp/UAV Mobility -> env.step() -> Outcome_t` transitions. Every next Decision is independently recollected at the next loop start. Route/Comm/Comp distinguish explicit empty/no-op from a missing field. Vehicle motion is advanced by SUMO; UAV mobility is the planner action. Evidence: `code/artifacts/protocols/pi_jwm_raw_multi_decision_step_real_airfogsim_v2_20260919/`.

## Step 2.4 frozen causal raw flow

The current Raw layer admits only tasks with `arrival_time_s <= decision_time_s` into `O_t`, History and the input-side Entity Index. Later AirFogSim schedules remain internal metadata. Decision rows include real CSI and masked CPU capacity observations; Outcome maps aggregate same-slot real transfer events and Task computed-size deltas. Raw simulator acceleration and canonical backward-difference acceleration are different fields; missing history produces `null + mask=false`. Evidence: `code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/`. Dataset/Tensor mapping is not started.

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
- 通信 Outcome 拆为 `wireless_delivered_data_by_task`、`wired_delivered_data_by_task` 和两部分按 task 聚合的 `delivered_data_by_task`。wired source 是真实 `WiredNetworkManager.step` 返回的 slot transmitted bytes；`{}` + observed mask 表示已观测但无服务，`null` + missing mask/reason 表示不可恢复。
- Raw Trajectory Layer / 01 与当前最小 Dataset/Tensor / 02 已冻结；STEP 4.2A 以独立版本链生成 graph input extension，但尚未生成 graph object。
- Step 3.1F 最小样本：History 为 `[t-H+1,t]` 的 `H=2` observation，并在过去 `tau<t` 行保留对齐的已执行四类 Action 和 Outcome；当前帧不含 `A_t/Y_t`。Future Action/Target 为 `[t,t+L-1]` 的 `L=2` 帧。input physical/task index 来自整个 History 的因果可观对象 union，flow index 来自过去 Outcome transfer event；Future Action 先按 anchor visibility 拒绝不可见 object，再引用同一 static union index；future-only endpoint 仍不进入 input Static。
- STEP 3.3F CPU collation 入口为 `code/src/pi_jwm/step3_3_model_input_tensor_v1.py`；12 个 Step 3.2 sample 生成 fixed-shape arrays。Past Outcome 为独立 H-1 轴；Target 保留 future entity/task/flow/service；stable index、固定 vocab、padding/mask 和 target namespace 均有 semantic receipt。该 tensor 尚未被双图或模型读取。
- STEP 4.1 mapping 已核实：position、heading/elevation、wireless CSI、CPU static capacity 和若干 Task current fields 在 Raw 有来源但未进入当前 input tensor；wired relation 可由 `environment.wired_edges` / `hasLink` 取得但尚未逐 Decision 物化，无 CSI 时以 type + feature mask 表示；可选 wired numeric state 不属于 03 minimum。完整 stateful Flow 在当前 Raw 不足。过去 `past_outcome_flow_service` 仍只是一条 hop service outcome，不能改名成 current Flow total/rem。
- STEP 4.2A 当前链路：原 Raw + decision-before-action wired amendment → Sample v5（position、typed Comm、static CPU、Task current、Src/Host/Exec/Ret）→ trajectory split/train-only normalization → Tensor v3。旧 Step 3 arrays 保留；Future Target 只用于隔离校验，不进入 History input。STEP 4.2B source audit 进一步确认 stable stateful Flow、Physical topology 和 graph object 仍不存在。
- STEP 4.2B 机器证据：`transmitted_size` 是当前 stage/hop progress，完成 hop 后 reset；Input/Return remaining 与 stable identity 未证明，DAG 不生成 DepData Flow。综合 verdict=`FLOW_CONTRACT_NOT_YET_SUPPORTED`。
- STEP 4.2B-PATCH：Flow verdict 仅由 Flow-specific evidence 计算；dynamic available CPU、storage、wired queue/load/utilization 是独立 `other_information_graph_gaps`。simulator-issued Flow ID unavailable 是事实，logical `task_id + input/return` identity 是否采用留待研究者决定；DepData 无真实 transfer process 也不允许由 DAG 虚构。
- STEP 4.2C-C（含 PATCH）：`code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py` 消费 C-B Raw 的 `logical_flow_rows/carrying_rows`，建立独立 History `logical_flow` union 与 target namespace。Logical Flow 保存 FlowID/Epoch/destination/total/E2E/presence/status；Carrying 保存 holder/hop/route/progress；completed/superseded known row 与 padding 分开。仅 `total_data/e2e_delivered/e2e_remaining/hop_progress/hop_remaining` 在 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train` 上标准化，stats 的 `mask_policy` 精确保留该 Step 3.2 policy，单位为 AirFogSim data-unit；categorical/reference/Epoch/RouteRevision/presence 不标准化。History/target Logical/Carrying 四组 tensor semantic equality 与 target carrying future ground-truth namespace 均有机器检查。Tensor artifact 是 CPU/non-locked development evidence，不是 formal Dataset。
- STEP 4.3A：`step4_3a_typed_dual_graph_builder_v1.py` 只读取冻结 Tensor 的 `history[-1]`。位置/运动经显式 development topology config 形成 Physical blocks；其余 current tensor namespaces 分别进入独立 Information blocks；Align 使用同一 entity index，GeoComm 使用 wireless Comm endpoints 的有效 Physical representation。target namespace counterfactual 不改变当前图。
- STEP 4.3B：`step4_3b_dual_graph_encoder_v1.py` 读取完整 History 的 Physical/Agent/Task/Logical Flow/Carrying raw fields 与 fixed train-only stats，再结合 STEP 4.3A current relation/connectivity，输出 aligned `Z_t^{PI,L_g}`。Future Target/Future Action 不进入 read path；graph layers 只做同一时刻表示细化，不执行状态转移。
- STEP 4.4：`step4_4_structured_rssm_world_model_v1.py` 将 `Z_t^{PI,L_g}` 初始化为五类 aligned h 与 Vehicle/Comm z，按 stable slot 路由 Route/Comm/Comp/Mobility；每步执行 dynamics graph interaction → prior unknown dynamics → known stochastic wireless outage → deterministic Physical/Comm/Flow/CPU/Task rules → predicted graph rebuild → state feedback。未来步不重跑 History encoder、不读 Future Target。当前仅有未训练 CPU 两步 rollout evidence。
- STEP 4.2A-PATCH mask rule：observer 的 supported wireless directed relation 与 CSI feature observability 是两个层次。endpoint present 且 structural row 存在时，CSI `observed_mask=false` 仍保留 relation validity；CSI tensor mask=false、placeholder=0。relation presence=false 时 CSI mask 不得为 true。wired valid/no-CSI 使用同一分层原则。
- STEP 4.2C-B Raw Flow rule：`target_node_id` 是 current action/carrying-hop target，不是可靠的 end-to-end destination。Input logical destination 从已建立 offload route terminal 捕获；Return 从 Decision task `return_destination_id` 捕获。Flow row 保存 source/capture phase；同一 `(TaskID, FlowType, Epoch)` destination 固定。真实两跳只在最终 hop 增加 E2E delivered，普通 hop advancement 只推进 hop index。
- Remaining Task source：`_extract_tasks()` 已读取 return size/deadline/priority，但 frozen Raw decision row 未透传；它们不是 simulator-source missing。`task_delay` 已由当前 `elapsed_time_s=max(decision_time-arrival_time,0)` 因果表达。

Unverified：未在当前 tensor manifest、loader 或模型实际读路径出现的字段，不得推断为当前模型输入。

## 2026-09-19 Raw 状态

单步、多步、因果字段与 return route 已完成真实验收。Raw 层冻结不等于 Dataset/Tensor 已实现。
