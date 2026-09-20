# STEP 4.2C-A — PI-JWM Causal Flow Ledger Feasibility Audit

## Step Goal

审计真实 Event 是否足以支持定义 03 的因果 Flow Ledger；不实现 Ledger、Raw additive extension、Sample/Tensor、Graph Builder 或模型链。

## Definition Basis

定义 03；STEP 4.2A/4.2B 已冻结合同；AirFogSim `Task`、通信更新路径、`WiredNetworkManager`、observer 和 collector 源码。

## Changes

新增 `step4_2c_a_causal_flow_ledger_feasibility_v1.py`、构建脚本、focused tests、合同和 observation-only artifact。记录 Input/Return/DepData、reroute、transition/source matrix、multi-hop no-double-count invariant、future-action counterfactual 和源码 SHA-256/symbol provenance。

## Results

原始 audit 将 cross-hop state 误作 simulator direct source，结论为 partial。4.2C-A-PATCH 通过纯 replay 函数纠正：在 Flow identity 已确定 logical destination/total 后，真实 event 可因果维护 E2E delivered/remaining；`target_id==logical_destination` 识别 final delivery；Task current location + hop completion 维护 holder；same-destination reroute 可保持 Flow epoch。机器 verdict 更新为 `CAUSAL_FLOW_LEDGER_FEASIBLE`。destination change 的 epoch inheritance 仍是 researcher decision；DepData 仍不由 DAG 虚构。

## Validation

本轮执行 4.2C-A-PATCH focused replay、Input/Return no-double-count、holder transition、same-destination boundary、`flow_completed` 禁止作为 logical completion、verdict tamper、4.2C-A/4.2B/Raw 回归、artifact rebuild/hash、compileall、knowledge index write/check、`git diff --check`；结果写入 Patch artifact receipt。

## Expected vs Actual

预期是只完成 feasibility audit。实际未修改 simulator、Raw、Sample、Tensor、Graph Builder、World Model、Loss、Planner、training、GPU 或 locked_test。

## Known Issues / Decision Boundary

真正 blocker 是跨 multi-hop 动作前 current remaining、final destination delivery 和 reroute payload holder 的可靠 source。下一步只能由研究者审阅并授权最小 event/source contract；不自动进入长期 Ledger 或 Graph Builder。

## Git / Next Step

本记录随本 Step commit。唯一建议：研究者审阅缺失 event/hook 的语义边界后，再决定是否授权 Raw additive source extension。

## STEP 4.2C-A-PATCH — Existing Event → Causal Ledger Derivability

研究者已冻结 logical Flow `A→C`，A→B/B→C 为 carrying hops。本 Patch 纠正了把 derived E2E state 误判为 simulator direct field 的问题：`apply_real_transfer_event_to_audit_ledger` 是临时纯 replay 函数，不是生产 Ledger。它只使用 task/phase/source/target/delivered event、logical destination/total 和 current holder；中间 hop 不减少 E2E remaining，final target 才累加，且保持 `delivered + remaining = total` 与单 epoch monotonic。

Input/Return 都有严格 schema-equivalent replay；Task current location + explicit stage/hop completion 维护 simulator task payload holder。same-destination route change 可保持 Flow ID/remaining 并增加 route revision；destination change 的 epoch/remaining inheritance 保留为 researcher decision。真实 Step 2.4 artifact 有 2 条 transfer events，但无线 event 缺 phase，因此 artifact 明确标记 `REAL_TRACE_EVIDENCE_PARTIAL`，不把它夸大为完整 Input/Return real trace。

Patch verdict：`CAUSAL_FLOW_LEDGER_FEASIBLE`。`flow_completed` 明确禁止作为 logical Flow completion source。长期 Ledger、Raw extension、Sample/Tensor、Graph Builder、模型和训练仍未启动。
