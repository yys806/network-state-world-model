# PI-JWM STEP 4.2A Existing-Source Graph Input Additive Extension V1

状态：已实现并通过 development machine receipt；这是 Raw → Sample → Dataset/Preprocessing → Tensor 的 additive data contract，不是 Graph Builder。

## 1. 版本链

| 层 | 新版本 | 与旧合同关系 |
|---|---|---|
| Raw amendment | `PI-JWM-Raw-Graph-Input-Additive-Amendment-v1-step4.2A` | 保留原 Decision 字段，只增加 `wired_relation_rows` |
| Model-ready Sample | `PI-JWM-Model-Ready-Sample-Contract-v5-step4.2A` | 复用 Step 3.1F H=2/L=2、History union index 与 Future Action 规则 |
| Dataset / Preprocessing | `PI-JWM-Step-4.2A-Dataset-Preprocessing-v1` | 复用 trajectory split 与 train-only fit |
| Tensor | `PI-JWM-Model-Input-Tensor-Collation-v3-step4.2A` | 保留 Step 3.3 v2 arrays，additive 增加明确 graph-input arrays |

旧 Step 2/3 artifacts 不覆盖，继续作为原合同历史证据。

## 2. Raw amendment

- `entities[].position_m`、`channel_rows`、`node_cpu_capacity_observation_rows` 和 Task current fields 继续使用原冻结来源。
- 每个 Decision 新增 `wired_relation_rows`。双向配置按 `WiredNetworkManager` 的 directed `_links` / `hasLink(source,target)` 语义物化为两个方向。
- wired 行包含 source、target、direction、`relation_type=wired`、presence/validity、observed mask、`csi_value=null`、`csi_feature_mask=false` 和来源说明。
- 来源是动作执行前的 recorded topology；不读取 wired service outcome，不用实际吞吐反推 relation。

## 3. Model-ready Sample

- Physical History entity 增加 `position_m=[x,y,z]`、逐坐标 mask、单位 `m`。
- 每个 History frame 增加独立 `communication_relations`：wireless 保存 directed endpoint、type、presence/validity、RB identity、per-RB attenuation/mask 和 missing reason；wired 保存有效 typed relation，但没有 CSI。
- Communication relation level 与 CSI feature level 分离：observer 先建立支持的 V/U/I directed structural relation，再尝试读取 CSI。endpoint 当前存在时，即使 CSI `observed_mask=false`，relation 仍保持 presence/validity=true；CSI 使用 `mask=false`、numeric placeholder=0 和非空 missing reason。`communication_mask_semantics=relation_validity_independent_of_feature_observability_v2`。
- Static 增加 `agent_static_capability`，CPU capacity 只保存一次，不带 H 轴；保留 observed mask、missing reason、单位与 `getFogProfile()['cpu']` 来源。
- History Task 增加 `task_cpu_work`、`computed_cpu_work`、`transmitted_size`、`elapsed_time_s`；`arrival_time_s` 只作为 causal derivation source。`elapsed_time_s=decision_time-arrival_time`，不读取未来完成时间。
- 每帧增加 Task→Agent 的 Src/Host/Exec/Ret typed relation。Exec 只在当前 lifecycle=`computing` 时建立；所有关系仅读取 Decision 字段，不读取 Future Route Action。
- Future Target 保存同名 position/progress 事实用于隔离验证；它们不进入 History tensor。

## 4. Preprocessing

连续字段只用 `dev_train` 中 presence/validity=true、feature mask=true、非 null 值拟合；validation 只复用统计量。当前统计覆盖 position x/y/z、wireless attenuation、CPU static capacity、task CPU demand、computed/transmitted progress 和 elapsed time。

每项保存 count、mean、std、zero-variance handling、source field、mask policy、unit 和 policy。静态 CPU capability 同样遵守 train-only fit；static 不等于可绕过 split isolation。

## 5. Tensor

- `entity_position[_raw] / entity_position_mask`：`[B,H,E,3]`。
- typed Comm：source/target/type/presence/validity `[B,H,C]`；CSI/RB arrays `[B,H,C,N_RB]`。wireless 或 wired relation validity 可以为 true，同时 CSI mask 全 false；mask=false 的 CSI numeric placeholder 固定为 0 且无真实数值语义。relation presence=false 时 CSI mask 必须全 false。
- `agent_cpu_capacity[_raw] / mask`：`[B,E]`，没有 H 轴。
- `task_history_extended_*`：`[B,H,T,4]`，feature order 为 CPU demand、computed progress、transmitted progress、elapsed time。
- typed Task–Agent relation：task index、agent index、Src/Host/Exec/Ret type、validity `[B,H,K]`。

所有 endpoint 必须满足 stable ID ↔ sample numeric index ↔ tensor slot 一致。没有构造 Physical Edge、邻域、graph object 或 GNN edge list。

## 6. 仍然阻塞

以下字段仍未进入当前 Sample/Tensor，但来源类别不同：

- `return_size`、`priority`、`deadline`：`SIMULATOR_OBSERVER_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED`。`_extract_tasks()` 已分别调用 `getReturnedSize()`、`getTaskPriority()`、`getTaskDeadline()` 并写入 `TaskSnapshot`，但冻结 Raw decision row 尚未透传，本 Patch 不新增字段。
- `task_delay`：observer 已提供；Step 4.2A 已用同一因果式 `max(decision_time-arrival_time,0)` 作为 `elapsed_time_s` 输入化。
- stable stateful Flow total/rem/type/endpoints、dynamic available CPU、storage、wired queue/load/utilization：继续 `RAW_INSUFFICIENT`。

旧 `LogicalFlow` / `CarryingHop` 名称不足以证明定义 03 Flow。仍需证明 stable identity、Input/Return/DepData type、logical endpoints、Task association、total/rem、multi-hop、route revision identity 与动作前因果性；`past_outcome_flow_service` 禁止替代 current Flow state。radius/kNN/hybrid、edge/cloud Physical membership、motion encoder 和 final feature subset 仍为 `RESEARCHER_DECISION_REQUIRED`。

因此 graph readiness 仍为 false；本合同不能作为 Graph Builder 已完成的证据。

## 7. 机器证据

Artifact：`code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。

`validation_report.json` 对 required checks 取逻辑 AND，并实际运行 wireless missing-CSI counterfactual；`step4_1_gap_resolution.json` 记录 Step 4.1 历史 gap 到当前 availability/source classification 的 overlay；`manifest.json` 保存输入/source/artifact hash、版本、array shape/dtype、unit、resolved/blocked 字段和范围。

范围固定为：`graph_builder=false`、`physical_topology=false`、`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。
