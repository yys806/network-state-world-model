# PI-JWM STEP 4.2C-B Causal Flow Ledger & Raw Additive Contract V1

状态：Ledger + Raw additive Flow contract 及 logical-destination provenance patch 已验收；Sample/Tensor 与 Graph Builder 未开始。

## 冻结语义

- Flow 是 logical end-to-end business payload；Hop 只是 carrying segment，Flow 不等于 Hop、Route 或 Communication edge。
- `FlowID=(TaskID, FlowType, Epoch)`，FlowType 固定为 `Input/Return/DepData`。FlowID 不依赖 Hop 或 RouteRevision；Flow Index 按首次出现稳定分配。
- same-destination reroute 保持 FlowID/Epoch/E2E remaining，仅增加 RouteRevision；logical destination change 只在 clean hop boundary 创建新 Epoch，旧 Epoch 标记 `SUPERSEDED`，新 total 等于旧 remaining。
- partial active hop 上 destination change 返回 `NOT_AT_CLEAN_HOP_BOUNDARY`。
- DepData 仅保留 vocabulary，当前 runtime instances=0；DAG 禁止生成假 DepData Flow。
- `target_node_id` 只表示当前动作或 carrying hop 的目标，不可直接解释为 logical end-to-end destination。
- Input logical destination 在动作成立后取已建立 offload route 的 terminal；AirFogSim `TaskManager.offloadTask` 强制 route terminal 等于 assigned execution target，完成普通 hop 时只删除 remaining route 的首元素。
- Return logical destination 独立取 Decision task row 的 `return_destination_id`；该字段来自 `Task.getToReturnNodeId()`，不随 carrying hop 推进改变。
- 每个 Flow row 必须保存 `logical_destination_source` 和 `logical_destination_capture_phase`；同一 `(TaskID, FlowType, Epoch)` 的 logical destination 不可变化。

## Flow 与 Carrying state

Flow row 固定保存 flow/task/type/epoch、logical source/destination、total、E2E delivered/remaining、presence/status、Flow Index、lineage、mask 和 provenance。`delivered + remaining = total`；同 Epoch delivered 单调不减、remaining 单调不增。完成或 superseded 的当前 row 保留稳定身份但 `presence=false`。

Carrying row 独立保存 FlowID、RouteRevision、route、current holder、current hop index/source/destination、hop progress/remaining、active、mask/missing reason。hop endpoints 不替代 logical endpoints。普通 hop advancement 只增加 `current_hop_index`，不得增加 RouteRevision 或 Epoch。

## 因果时序

顺序固定为：capture `O_t` → 暴露当前 Ledger → 记录 `A_t` → 执行 AirFogSim → 收集真实 `Y_t` → 用 `Y_t` 更新 Ledger → 核对真实 Task holder → 形成 `O_{t+1}`。`O_t` 不读取 `Y_t`、future action/outcome、target tensor、rollout prediction 或 future task schedule。

legacy `flow_completed` 只在 additive overlay 中映射为 `stage_or_hop_completed`，禁止直接决定 logical completion。Logical completion 只由 `e2e_remaining==0` 决定。

## 真实证据与限制

non-locked AirFogSim trace 实际观察到 Input 和 Return 创建/传输/完成。独立真实 Input 两跳 trace 已证明 `UAV_0→RSU_0→cloudServer_4` 的两次服务属于同一 FlowID、同一 Epoch，logical destination 始终为 `cloudServer_4`；中间 hop 不推进 E2E，最终 hop 才推进 E2E，避免 double-count。Return multi-hop、same-destination reroute、destination-change Epoch、local execution no-flow 使用 `SYNTHETIC_CONTRACT_FIXTURE` 验证，未冒充真实场景。

same-destination partial-hop reroute 是否被 simulator 真实支持仍是 unresolved runtime evidence；本 Patch 没有新增 clean-boundary 研究规则。

旧无线 Return hook 的 delivered amount 可大于 observer `return_size`；Ledger 依冻结规则 `min(total, delivered+delta)` 封顶并保持守恒。这是 Real Trace Observation，不是 simulator 语义修改。

Artifact：`code/artifacts/protocols/pi_jwm_step4_2c_b_causal_flow_ledger_raw_v1_20260920/`。
