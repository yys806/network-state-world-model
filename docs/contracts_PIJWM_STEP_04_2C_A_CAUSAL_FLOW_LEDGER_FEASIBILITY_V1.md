# PI-JWM STEP 4.2C-A Causal Flow Ledger Feasibility V1

状态：原始审计为 `CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE`；4.2C-A-PATCH 经 existing-event replay 修正为 `CAUSAL_FLOW_LEDGER_FEASIBLE`。本 Patch 只验证 Real Event → Causal Ledger 的可因果派生性，不实现长期 Flow Ledger。

## 目的与边界

审计定义 03 所需的 Input、Return、DepData 事件是否能在动作前由真实 simulator event/state 因果追溯。`past_outcome_flow_service` 不是 current Flow state；未来 action、future outcome、target tensor 和 rollout prediction 不得进入审计。范围固定为 `formal_dataset=false`、`training=false`、`gpu=false`、`locked_test=false`。

## 当前机器结论

- Input/Return 的 task、phase、hop、端点和 delivered event 可追溯；在 logical destination 已写入 Flow state 的前提下，E2E delivered/remaining 可由真实事件因果维护。
- `Task._transmitted_size` 仍是当前 stage/hop 进度，hop 完成后 reset；它不作为 logical remaining，而是由真实 event replay 派生 E2E remaining。
- `event.target_id == logical_destination` 区分 intermediate service 与 final delivery；`current_node_id` + hop completion 维护当前 simulator task payload holder。
- same-destination reroute 可保持 Flow ID/e2e remaining、增加 route revision 并从 current holder 开始新 carrying hop；destination change 的 epoch/remaining inheritance 仍是研究者决定。
- DAG 是 dependency gating，不是 DepData transfer。当前无 dependency payload/transfer event；是否保留空类型或扩展 simulator 属于研究者决定，不单独阻塞 Input/Return。
- 多 hop 计数只把 final-destination delivery 算入端到端 delivered，不能把各 hop service 直接相加。

机器 verdict 由 `ledger_specific_required_evidence` 实际计算，不是硬编码。Patch verdict 为 `CAUSAL_FLOW_LEDGER_FEASIBLE`。`flow_completed` 仅解释为 stage/hop completion，禁止作为 logical Flow completion。

Artifact：`code/artifacts/protocols/pi_jwm_step4_2c_a_causal_flow_ledger_feasibility_patch_v1_20260920/`。Step 2.4 真实 artifact 提供 2 条 real transfer events，但无线 event 缺 phase；严格 Input/Return replay 同时保存为 schema-equivalent fixture evidence。
