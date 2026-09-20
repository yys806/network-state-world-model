# PI-JWM STEP 4.2C-A Causal Flow Ledger Feasibility V1

状态：`CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE`；本 Step 只审计 Real Event → Causal Ledger 的可行性，不实现长期 Flow Ledger。

## 目的与边界

审计定义 03 所需的 Input、Return、DepData 事件是否能在动作前由真实 simulator event/state 因果追溯。`past_outcome_flow_service` 不是 current Flow state；未来 action、future outcome、target tensor 和 rollout prediction 不得进入审计。范围固定为 `formal_dataset=false`、`training=false`、`gpu=false`、`locked_test=false`。

## 当前机器结论

- Input/Return 的 task、phase、hop、端点和 delivered event 可部分追溯。
- `Task._transmitted_size` 是当前 stage/hop 进度，hop 完成后 reset；它不能证明跨 hop logical remaining。
- final logical destination delivery、跨 hop remaining 和 reroute 时 payload holder 没有可靠现成 source，需要新增 observer/event hook。
- DAG 是 dependency gating，不是 DepData transfer。当前无 dependency payload/transfer event；是否保留空类型或扩展 simulator 属于研究者决定，不单独阻塞 Input/Return。
- 多 hop 计数只把 final-destination delivery 算入端到端 delivered，不能把各 hop service 直接相加。

机器 verdict 由 `ledger_specific_required_evidence` 实际计算，不是硬编码。当前 verdict 为 `CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE`，receipt 的 `passed=false` 表示完整 ledger feasibility 尚未满足，而不是测试失败。

Artifact：`code/artifacts/protocols/pi_jwm_step4_2c_a_causal_flow_ledger_feasibility_v1_20260920/`。
