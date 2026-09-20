# PI-JWM Physical / Information Object–Field–Relation Mapping V1

状态：STEP 4.1 映射已冻结；这是数据事实到图语义的合同，不是 graph builder、GNN 或模型实现。

## 1. 结论

当前 stable entity/task identity、速度、规范加速度、任务大小、lifecycle 和 DAG 可以直接复用。Raw 还有位置、运动方向、无线 CSI、CPU capacity、任务 CPU 需求与当前进度，但 v4 Sample / v2 Tensor 没有暴露。wired relation 的端点、方向与存在性可由 `environment.wired_edges` / `WiredNetworkManager.hasLink` 可靠取得，但尚未逐 Decision 物化并进入 Sample/Tensor；当前 Raw 仍缺具有稳定 ID、类型、端点、总量和剩余量的 current stateful Flow。

因此：**不得在当前合同上直接实现完整新图 builder**。下一步若获授权，应先做 Data Contract Additive Extension；不能用旧 `physical_edge_state` 或过去 hop service 填补缺口。

## 2. 身份与对象

| 对象 | 身份来源 | 当前结论 |
|---|---|---|
| Physical Node | `static.input_entity_index.physical` | vehicle/UAV/RSU 冻结为 Physical + Info；edge/cloud 是否有独立空间意义待研究者决定 |
| Information Agent | 与对应现实实体共享 stable entity slot，但有独立 presence/feature mask | 只保存通信、计算、信息处理资源；不复制 position/speed |
| Task Node | `static.input_entity_index.task` | 保存 Demand、Progress、Time、lifecycle；端点身份改为 typed relation |
| Flow | 需要新的 stable stateful Flow index | 当前 event-derived flow index 只标识过去 hop event，不满足定义 03 |

同一实体对齐为：`real entity ID -> physical index/presence + agent index/presence`。共享身份 slot 不等于共享连续特征。当前不存在已证明“只有 Physical、没有 Info”的实体类型；edge/cloud 当前按 Info Agent 可用，Physical membership 待决定。

## 3. Physical Graph

Physical Node 只描述现实实体的空间与运动状态。

| 字段 | Raw | Sample/Tensor | 分类 | 归属 |
|---|---|---|---|---|
| `position_m` | 有 | 无 | `RAW_AVAILABLE_BUT_NOT_EXPOSED` | Physical Node dynamic |
| `speed_mps` | 有 | 有 | `AVAILABLE_NOW` | Physical Node dynamic |
| heading/elevation | 有，但 heading unit 随实体语义变化 | 无 | `RAW_AVAILABLE_BUT_NOT_EXPOSED` | canonical motion direction |
| canonical acceleration | 有，因果后向差分及 mask | 有 | `AVAILABLE_NOW` | Physical Node dynamic |
| relative position/distance/motion | 可由当前端点物理状态推导 | 无 | `DERIVABLE_CAUSALLY` | Physical relation |

Physical relation 的硬约束：由物理空间状态构造，不由通信、任务或 RB 活动构造。radius、kNN 或 radius+kNN 尚未决定。

## 4. Information Agent

| 字段 | 真实来源 | 当前结论 |
|---|---|---|
| entity type | Raw → Static → Tensor category | `AVAILABLE_NOW`，静态/type 信息 |
| CPU capacity | Raw `node_cpu_capacity_observation_rows`，来自 `entity.getFogProfile()['cpu']` | `RAW_AVAILABLE_BUT_NOT_EXPOSED`；配置的静态能力上限，不是动态可用量 |
| available CPU | 无可靠当前字段 | `RAW_INSUFFICIENT`；不能把 capacity 当 availability |
| storage | 无可靠当前字段 | `RAW_INSUFFICIENT`；旧 zero-fill 不可复用 |
| service/load | 可能可从 Task/Flow 聚合 | `RESEARCHER_DECISION_REQUIRED`；未证明不可推导前不重复存储 |

Position/speed 属于 Physical Node，不能作为 Agent resource continuous feature。CPU 四类语义必须分开：capacity 是 Agent 静态 capability；allocation 是 `A_t^Comp`；actual served CPU 是 Outcome；available CPU 才是动态 Agent resource，当前无可靠 decision-time source。

## 5. Task Node 与 Task–Agent Relation

| Task 语义 | 当前事实 | 分类 |
|---|---|---|
| data size | Raw/Sample/Tensor 都有 | `AVAILABLE_NOW` |
| return size、priority、deadline | simulator/旧 observer 有候选 getter，但冻结 Raw 行没有 | `RAW_INSUFFICIENT` |
| computation demand | Raw `task_cpu_work` 有，Sample/Tensor 无 | `RAW_AVAILABLE_BUT_NOT_EXPOSED` |
| transmission/computing progress | Raw 有，当前 History input 未暴露 | `RAW_AVAILABLE_BUT_NOT_EXPOSED` |
| elapsed time | decision time - arrival time | `DERIVABLE_CAUSALLY` |
| lifecycle | Raw/Sample/Tensor 都有 | `AVAILABLE_NOW` |

四类关系方向均为 `Task -> Agent`：

- Src：`task_node_id`；
- Host：当前 decision 的 `current_node_id`；
- Exec：只有 lifecycle 已表明正在计算时，才由 `current_node_id` 建立；
- Ret：非空 `return_destination_id`。

四者都必须同时检查 Task/Agent presence 和 stable index。`A_t^Route` 新选的目标属于 action，不能提前成为动作前 Host/Exec 关系。

## 6. Comm、Flow 与 DAG

Comm 是 directed `Agent -> Agent` relation。Raw `channel_rows` 给出无线 source/target、类型、每 RB attenuation、observed mask；当前 Sample/Tensor 只保留端点，未保留 numeric CSI。wired 最小 relation 由 source Agent、target Agent、direction、`relation_type=wired` 和当前 presence 构成；这些字段可从 `environment.wired_edges` / `WiredNetworkManager.hasLink` 可靠物化，但当前冻结合同尚未逐 Decision 暴露。wired relation 不要求伪造 CSI：`csi_value=null`、`csi_feature_mask=false`，由 relation type 区分。wired latency、带宽能力及其他 wired-specific numeric state 不是当前定义 03 minimum；其中配置的 capacity/propagation delay 是静态 link capability，真实动态 queue/load/utilization 仍无冻结 Raw 来源。实际 throughput、delivered data、RB allocation、past service 都不是动作前 Comm state。

Flow 是 directed stateful multiedge，并关联 Task。最低字段为 stable Flow ID、Input/Return/DepData 类型、端点、presence、total data、remaining data。当前 Raw 的 task lifecycle/route/progress 只能提供部分候选线索；缺少完整稳定 Flow 状态，特别是 return size、dependency-data flow 和明确 stage total/rem。`slot_transfer_events` / `past_outcome_flow_service` 只表示过去某个 hop 的 service，不可冒充 current Flow state。

DAG 可直接复用：`j -> k` 表示 Task k depends on Task j；端点使用 input task namespace；DAG 不等于 dependency-data Flow，当前最小 DAG 无连续 feature。

最终 Information Graph：Nodes = Agent + Task；Relations = Comm + Src/Host/Exec/Ret + Flow + DAG。Comm 与 Flow 即使端点相同也保持不同 relation type，并允许同一 Agent 对之间有多个 Flow。

## 7. Required Additive Data Extension

| 项目 | 分类 | 下一合同必须做什么 |
|---|---|---|
| Physical position | `RAW_AVAILABLE_BUT_NOT_EXPOSED` | 暴露 position、单位和 mask |
| Spatial relations | `DERIVABLE_CAUSALLY` | 暴露位置/运动后再推导；本 Step 不定拓扑策略 |
| Wireless CSI | `RAW_AVAILABLE_BUT_NOT_EXPOSED` | 暴露 directed per-RB CSI/type/mask |
| Wired relation | `RAW_AVAILABLE_BUT_NOT_EXPOSED` | 从拓扑/`hasLink` 逐 Decision 物化端点、方向、type、presence；CSI 保持 null + mask=false |
| Wired optional dynamic numeric state | `RAW_INSUFFICIENT`，且 `minimum_for_03=false` | 只有未来明确选择并存在真实来源时才采集；不阻塞最小 Information Graph |
| Agent CPU capacity | `RAW_AVAILABLE_BUT_NOT_EXPOSED` | 作为静态 capability 暴露 capacity 与 observed mask |
| Agent available CPU | `RAW_INSUFFICIENT`，且 `minimum_for_03=false` | 需要独立、真实的 decision-time 剩余资源来源；不能从 capacity/allocation/service 推断 |
| Task demand/progress/time | 混合 | 暴露 Raw 已有字段；补采 return size/priority/deadline |
| Typed Task–Agent relations | `DERIVABLE_CAUSALLY` | lifecycle-conditioned materialization |
| Current stateful Flow | `RAW_INSUFFICIENT` | 补 stable ID/type/endpoints/presence/total/remaining |
| edge/cloud Physical membership | `RESEARCHER_DECISION_REQUIRED` | 先判断坐标是否具有真实空间建模意义 |

完整机器表见 `required_additive_data_extensions.json`。

## 8. Forbidden / Wrong Placement

| 当前事实 | 禁止放置 | 正确角色 |
|---|---|---|
| CSI | Physical Edge | Comm current state |
| rate/service | Physical Edge | Execution Outcome |
| RB allocation | Physical Edge | Comm Action |
| CPU allocation | Agent state | Comp Action |
| actual CPU service | Comp Action | Execution Outcome |
| CPU capacity | dynamic available resource | Agent static capability |
| Src/Host/Exec/Ret | Task continuous feature | typed Task–Agent relation |
| past hop service | current Flow remaining | past execution outcome |
| DAG | dependency-data Flow | Task dependency relation |
| position/speed | Agent resource feature | Physical Node |
| Future Action result | current graph state | target/outcome |

## 9. 旧实现复用边界

- 可复用：stable ID/index/presence/mask、通用 `masked_index_mean` 思路、DAG 的 parent→child 方向。
- 需重构：旧 Agent/Flow message path、旧 node vector、Task endpoint 字段表示。
- 禁止按原语义复用：`EDGE_FEATURES=(distance,csi_mean,rate_sum,active_task_count,allocated_rb_count)` 作为 Physical Edge；`couple_flow_bearer` 的旧 Flow↔physical/channel edge 绑定；当前未由 Raw 证明的 legacy `FLOW_FEATURES`。
- `formal_dual_graph_world_model_v1` 仅是历史实现证据，本 Step 不复用或修改模型。

## 10. 机器证据

Artifact：`code/artifacts/protocols/pi_jwm_step4_1_pi_graph_mapping_v1_20260919/`。

`mapping_schema.json` 是总合同；四张独立 matrix JSON 和 `validation_report.json` 由 builder 生成；`manifest.json` 绑定真实 Raw、Step 3.3 tensor schema/manifest、定义 03 SHA 和本 Step 源码/测试 hash。机器 receipt 对所有 required checks 取逻辑 AND。

范围固定为：`graph_builder_implemented=false`、`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。

## 11. STEP 4.2A resolution overlay（2026-09-20）

本文件冻结的是 Step 4.1 当时的 mapping 与 gap，不改写历史分类。后续 Step 4.2A 已用独立版本链把 position、wireless per-RB CSI、wired typed relation、CPU static capability、已有 Task demand/progress/elapsed 以及 Src/Host/Exec/Ret 贯穿到 Sample/Tensor。机器化 resolution overlay 见 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/step4_1_gap_resolution.json`。

仍未解决且不得补造的 minimum gap 是 stable stateful Flow。STEP 4.2A-PATCH 进一步核实 return size/priority/deadline 有 `_extract_tasks()` getter/TaskSnapshot 来源，但冻结 Raw 尚未透传，因此分类为 `SIMULATOR_OBSERVER_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED`；不是 simulator 无来源，也未在该 Patch 输入化。dynamic available CPU、storage 和 wired queue/load/utilization 继续缺少可靠 Raw 来源。Physical topology 与 graph builder 未实现。
