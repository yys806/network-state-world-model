# PI-JWM STEP 4.2C-C Stateful Flow Sample/Tensor Contract V1

状态：**COMPLETE / FROZEN（含 STEP 4.2C-C-PATCH；仅 Raw Flow → Model-ready Sample → CPU Tensor additive extension）**

本合同消费已经冻结的 STEP 4.2C-B Raw `logical_flow_rows` 与 `carrying_rows`。它只做对齐、稳定索引、presence/mask、允许的连续字段标准化和固定形状整理；不重新解释 AirFogSim，也不创建 Physical/Information graph。

## 1. 因果边界与来源

- History 仍是 STEP 3.1F 的 `O_{t-H+1:t}` 及过去 Action/Outcome；Flow History 行只来自对应 Decision 的 `O_t.logical_flow_rows`、`O_t.carrying_rows`。
- target Flow 行来自 `Decision_{t+1}` 的冻结 Raw 行，使用独立 target namespace；future action、future outcome、future schedule、target tensor 或 rollout prediction 不回填 History。
- STEP 4.2C-B Raw 是 Flow state source of truth。Sample/Tensor 不重算 logical destination、Epoch、E2E remaining、holder、RouteRevision 或 completion。
- Flow 为 logical end-to-end business Flow，`FlowID=(TaskID, FlowType, Epoch)`；Hop 仅为 Carrying State。`target_node_id` 不被提升为 logical destination。

## 2. Sample namespace 与状态

新增独立 namespace：

- `static.input_entity_index.logical_flow`：History causal FlowID union 的稳定 index；Flow 后来出现、完成或 superseded 后仍不换 slot。
- `static.target_index.logical_flow`：target-side Flow union；`target_only_objects.logical_flow` 明确记录只在 target 出现的 Flow/Epoch。

每个 Logical Flow row 保存：`flow_id`、`flow_index`、`task_id/task_index`、`flow_type`、`epoch`、logical source/destination 及其 index、`total_data`、`e2e_delivered`、`e2e_remaining`、`presence`、`status`、三个 feature mask 和 destination provenance。Carrying row 独立保存 Flow reference、RouteRevision、route node references/mask、current holder/hop index、hop source/destination、hop progress/remaining、active。

已知但当前 inactive 的 `COMPLETED`/`SUPERSEDED` Flow 保留稳定 slot，`presence=false`；不存在的 padding 使用 `known=false`，两者不可混同。

## 3. Tensor contract

实现入口：`code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py`。

- Sample schema：`PI-JWM-Model-Ready-Sample-Contract-v6-step4.2C-C`
- Tensor schema：`PI-JWM-Model-Input-Tensor-Collation-v5-step4.2C-C-PATCH`
- Raw source：`PI-JWM-Step-4.2C-B-Causal-Flow-Ledger-Raw-v1`
- Logical namespace 与旧 Step 3 hop-service `flow` namespace 分离；旧 `past_outcome_flow_service` 仅保留为兼容字段，不被宣称为 Logical Flow state。
- Tensor 保存 logical categorical/index/state arrays、presence/feature masks、History/target carrying holder/hop/route arrays，以及 target-side logical arrays。target carrying 是 future ground-truth/deterministic-transition state，不是 learned prediction head。容量是显式 development bound；超限必须抛出 `capacity overflow`，不得静默截断。

只有以下连续字段做 train-only、presence-aware normalization：`total_data`、`e2e_delivered`、`e2e_remaining`、`hop_progress`、`hop_remaining`。统计样本必须同时满足 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`；stats 的 `mask_policy` 精确记录为 `presence=true AND feature_mask=true AND value!=null; train split only`。单位为 AirFogSim data-unit；`count/mean/std/zero-variance/source_field/mask_policy/unit` 写入 stats。Flow index、Task/entity reference、type、status、Epoch、presence、RouteRevision 不标准化。

## 4. 机器验收与证据边界

机器 receipt 必须分别通过 History Logical、History Carrying、target Logical、target Carrying 四个 Raw→Sample 与 Sample→Tensor semantic equality 子检查，并检查 target namespace/future-Epoch isolation、presence/masked-zero placeholders、History/target endpoint/task bounds、route masks、presence-aware normalization policy、stable FlowID/index、multi-hop single Flow identity、target carrying ground-truth boundary、overflow rejection、Input/Return category、DepData runtime zero、deterministic rebuild、serialize/load 和 scope checks；顶层 `passed` 是这些 required checks 与 scope 的实际 AND，并覆盖 receipt/semantic tamper negative。语义 metadata 同时保存 Flow/Task/节点/route ID provenance，防止只改字符串而保留同一 numeric index 的篡改漏检。

真实证据位于：

- `code/artifacts/protocols/pi_jwm_step4_2c_b_real_flow_trace_source_v1_20260920/`：真实 Input/Return Flow source；
- `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/`：真实两 hop Input evidence；
- `code/artifacts/protocols/pi_jwm_step4_2c_c_real_multihop_cross_slot_v1_20260920/`：CPU、non-locked、低 wired capacity 的真实跨 Decision carrying-state trace；
- `code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/`：最终 Sample/Tensor receipt、stats、NPZ、manifest。

Return multi-hop、same-destination partial-hop reroute、destination-change runtime 和 formal capacity 仍不是本 Step 的结论；旧 direct trace 的 wired structural relation 仅标注为 contract fixture。`formal_dataset=false`、`training=false`、`gpu=false`、`locked_test=false`、`graph_builder=false`。

## 5. 停止边界

本 Step 不实现 Graph Builder、Physical topology、Information graph materialization、GNN/Encoder、World Model/RSSM、Loss、Planner、Training 或 GPU。下一步只能由研究者另行审阅并授权 Definition 03 Graph Builder Contract。
