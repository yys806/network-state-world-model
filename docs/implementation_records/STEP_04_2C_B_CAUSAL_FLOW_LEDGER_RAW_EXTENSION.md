# STEP 4.2C-B — Stateful Flow Ledger Contract & Raw Additive Extension

## Step Goal

按研究者冻结的 logical Flow/Epoch/RouteRevision 语义，实现 PI-JWM Causal Flow Ledger 与独立版本 Raw amendment；截止 Raw，不进入 Sample/Tensor 或 Graph Builder。

## Definition Basis

定义 03；STEP 4.1、4.2A、4.2B、4.2C-A-PATCH；研究者本 Step 明确冻结的八项 Flow 决策；AirFogSim Task/transfer event/TaskSnapshot 真实接口。

## Initial State

基线 `6997a757e11e71b20417e6c45d4597bc91dc7416` 只有 feasibility replay，没有长期 Ledger、Epoch lineage、Raw current Flow rows 或真实 Input/Return Raw acceptance。

## Files Involved / Changes

- 新增 `step4_2c_b_causal_flow_ledger_raw_v1.py`：FlowID、Flow/Carrying state、Input/Return lifecycle、Outcome update、holder guard、same-destination reroute、clean-boundary Epoch、receipt。
- 新增真实 trace runner，只复用 Step 2.3 runner并透传 observer 已有 `return_size`，不修改 AirFogSim。
- 新增 Raw additive builder、focused tests、contract、artifact/manifest；旧 Raw schema/artifact 不覆盖。
- 同步 Tracker、AI_CONTEXT、authority/process records、knowledge index。

## Reuse

复用现有 TaskSnapshot、真实 slot transfer events、Task current node、Route action execution、Raw Decision/Action/Outcome 顺序和 Step 2.3/2.4 non-locked runner。legacy `flow_completed` 只作 hop/stage overlay。

## Validation

执行 focused 4.2C-B、真实 Flow lifecycle、Input/Return、多跳不重复计数、holder、local execution、reroute/Epoch、lineage、presence、DepData zero、future counterfactual、legacy completion prohibition、4.2C-A/4.2B/4.2A/4.1/Raw regressions、deterministic rebuild/hash、JSON reload、compileall、knowledge index write/check 和 diff check。

## Results / Expected vs Actual

Implementation Fact：Ledger 与 Raw additive state 已实现，machine receipt required checks 使用真实 AND。

Real Trace Observation：真实 non-locked trace 覆盖直接 Input/Return 完成；另一个真实 trace 覆盖 Input 两跳。旧无线 Return hook amount 可超过 `return_size`，Ledger 依 frozen min rule 封顶。

Derived Ledger State：E2E progress、current holder、same-destination RouteRevision、destination-change Epoch lineage 由真实 state/event + 历史 Ledger 确定性维护。

Researcher Decision：Flow/Hop 分离、FlowID 三元组、same-destination 不换 Epoch、destination change 新 Epoch 且限 clean boundary、DepData zero-instance。

## Known Issues / Remaining Gap

真实 trace 尚未覆盖 Return multi-hop、same-destination reroute、destination-change Epoch 和 local execution；这些只有 contract fixture evidence。Sample/Tensor Flow extension、Graph Builder、模型和训练未开始。

## Git / Next Step

本记录随本 Step commit。若最终验收通过，唯一建议为 `STEP 4.2C-C — Flow Sample/Tensor Additive Extension`，不得直接进入 Graph Builder。
