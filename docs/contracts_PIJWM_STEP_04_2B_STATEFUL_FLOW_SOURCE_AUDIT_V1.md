# PI-JWM STEP 4.2B Stateful Flow Source Audit V1

状态：审计完成；`FLOW_CONTRACT_NOT_YET_SUPPORTED`；Graph Builder NOT STARTED。

## 目的与边界

本合同只审计定义 03 的 Flow 最小字段是否有真实、动作前可追溯的来源。它不新增 Raw 字段、不构造 Flow Tensor、不实现 Graph Builder、模型、Loss、Planner、训练或 GPU。审计 receipt 是 observation-only 机器证据，不是正式 Dataset 结论。

## 定义 03 最小字段

每条 stateful directed Flow 需要稳定 identity、Input/Return/DepData type、关联 Task、source Agent、destination Agent、presence、total data 和 current remaining data。Task DAG 只表达依赖门控，不能自动变成 DepData Flow；过去 slot 的 service outcome 不能替代当前 Flow state。

## 当前源码结论

- Input：`task_size` 可作为任务级总量；`getTransmittedSize()` 只表示当前传输阶段/当前 hop 的累计量，完成 hop 后归零。因此无法跨 multi-hop 因果重建端到端 remaining，也没有 simulator-issued stable Flow ID。逻辑端到端候选和 hop-local 候选都只能 `PARTIALLY_CONSTRUCTIBLE`。
- Return：`getReturnedSize()` 是 return total requirement；return 阶段复用同一 `transmitted_size` 并在 hop 完成时归零。完整 return Flow identity/remain 仍未证明，结论为 `PARTIALLY_CONSTRUCTIBLE`。
- DepData：`TaskManager._task_dependencies` 只用于 parent completion gating；当前审计源码没有 dependency payload、独立 Flow ID 或 dependency transfer event，结论为 `NOT_YET_SUPPORTED`。
- Route revision：`changeOffloadTo()` 替换未完成 route suffix，但源码没有独立 Flow/revision identity。frame builder 的 revision 只是一侧动作约定，不能当成 simulator provenance。
- 其他缺口：dynamic available CPU、storage、wired queue/load/utilization 和 dependency payload 仍 `RAW_INSUFFICIENT`；return size/deadline/priority 已有 observer getter，但 frozen Raw 尚未透传。

## 机器证据

Artifact：`code/artifacts/protocols/pi_jwm_step4_2b_stateful_flow_source_audit_v1_20260920/`。

`stateful_flow_source_audit.json` 保存 Input/Return/DepData、Task Progress ↔ Flow Progress、Flow ↔ Hop、route revision、remaining source matrix、scope 和源码 SHA-256。`manifest.json` 保存 artifact/source hash。综合 verdict 为 `FLOW_CONTRACT_NOT_YET_SUPPORTED`，因此依据停止门不得进入 Graph Builder。

范围：`graph_builder_started=false`、`formal_dataset=false`、`training=false`、`gpu=false`、`locked_test=false`。
