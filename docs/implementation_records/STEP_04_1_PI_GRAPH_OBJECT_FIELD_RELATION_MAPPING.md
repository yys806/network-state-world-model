# STEP 4.1 — Physical / Information Object–Field–Relation Mapping Freeze

状态：COMPLETE / FROZEN（待本次 Git 回执）；只冻结映射，不代表新双图已实现。

## Step Goal

把当前 Raw / model-ready sample / tensor 的真实数据事实逐项映射到定义 03 的 Physical / Information 对象、字段和关系；区分可直接复用、需重解释、Raw 有但未暴露、Raw 不足、因果可推导、不应作为特征和需研究者决定。

## Definition Basis

- 只读：`D:\shen\OB\科研\PIJWM\03物理-信息双图建模.md`
- SHA-256：`6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e`
- 当前工程依据：Step 2 frozen Raw、Step 3.1F sample v4、Step 3.3 tensor v2、旧 graph/tensor/model 源码。

## Initial State

01 与 02 已按当前最小合同冻结。Step 3.3 Tensor 已保留 stable index、mask、Past Outcome、Target 和四类动作，但 Gap Table 明确未暴露 position、CSI numeric、CPU capacity；当前 flow tensor 表示过去 hop service，不是 current stateful Flow。

## Files Involved

- `code/src/pi_jwm/step4_1_pi_graph_mapping_v1.py`
- `code/scripts/build_step4_1_pi_graph_mapping_v1.py`
- `code/tests/test_step4_1_pi_graph_mapping_v1.py`
- `docs/contracts_PIJWM_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING_V1.md`
- `code/artifacts/protocols/pi_jwm_step4_1_pi_graph_mapping_v1_20260919/`
- Tracker、authority/process records、必要 AI_CONTEXT、知识索引。

## Changes

- 冻结 Physical Node、Agent、Task 及 Physical/Comm/Task-Agent/Flow/DAG relation 语义。
- vehicle/UAV/RSU = Physical + Info；edge/cloud 的 Physical membership 因坐标空间意义未证明而保留 `RESEARCHER_DECISION_REQUIRED`，Info Agent 不受影响。
- 冻结 Physical 与 Agent 共用现实实体 stable slot、但 presence 与 feature semantic 分离；Task 使用 task namespace；current Flow 需要新的 stable stateful index。
- 对 26 个字段形成 Raw/Sample/Tensor source、Static/Dynamic、Direct/Derived、mask 和 availability 记录。
- 对 8 类 relation 冻结方向、端点 namespace、当前状态来源及 validity。
- 形成 Current Data → Graph Role、Required Additive Data Extension、Forbidden Placement、Old Implementation Reuse/Conflict 四张机器表。
- validator 从真实 Raw 和 tensor schema 计算 acceptance；negative fixture 篡改 CSI placement 或把 CPU capacity 误标为动态 resource 后，receipt 必须失败。
- 生成 mapping schema、四张独立 JSON 表、validation report 与 hash/provenance manifest。

### STEP 4.1-PATCH — Minimum Gap Semantic Correction

- 把 wired 最小 relation 与 wired 可选 numeric state 分开：端点、方向、`relation_type=wired`、presence 可由 `environment.wired_edges` / `WiredNetworkManager.hasLink` 取得，分类为 `RAW_AVAILABLE_BUT_NOT_EXPOSED`；无 CSI 时使用 `null + feature_mask=false`。
- wired latency、带宽能力及其他动态 numeric state 不属于定义 03 minimum，保持 optional；真实动态 queue/load/utilization 当前为 `RAW_INSUFFICIENT`，不作为 graph readiness blocker。
- 核实 `capacity_per_s` 来自 `entity.getFogProfile()['cpu']`。配置注释将其定义为 CPU capacity，当前源码没有运行期 `setFogProfile` 调用，真实决策帧中每节点观测值保持不变，因此归入 `information_agent.static_capability`。
- 明确四类 CPU 语义：capacity = 静态 Agent capability；allocation = `A_t^Comp`；actual service = Outcome；available CPU = 动态 Agent resource，但当前没有可靠来源。
- 未改动其他 object/field/relation mapping，也未实现 graph builder。

## Reuse

直接复用 Step 02/03 的 stable ID/index/presence/mask、DAG 方向和通用 masked-index 思路。旧 `EDGE_FEATURES` 混合 distance/CSI/rate/task/RB，旧 Flow↔physical edge coupling 混合语义，禁止按原含义复用。未修改任何旧模型、图算子或 frozen 02 contract。

## Validation

- TDD red：新增测试在模块不存在时以 `ModuleNotFoundError` 失败。
- focused：`python -m unittest discover -s code/tests -p 'test_step4_1_pi_graph_mapping_v1.py' -v` → 7/7 passed（含 wired type+mask 与 CPU 四类语义、negative tamper）。
- artifact builder：`python code/scripts/build_step4_1_pi_graph_mapping_v1.py` → `passed=true`，生成 7 个 JSON 文件；required checks 包含 wired relation source/type/mask、optional numeric non-minimum、CPU static capability 与 CPU 四类语义分离。
- Step 3.3 regression：8/8 passed；Step 3.1F regression：12/12 passed。
- deterministic artifact rebuild：两次 builder 后 7 个 JSON SHA-256 全部一致；`mapping_schema.json=62aa4ef0...f0b8ea37`，`manifest.json=f72e5742...3f1d8e92`。
- `python -m compileall -q code/src code/scripts code/tests`：通过。
- knowledge index write / `--check`：5 个输出，`mismatches=[]`、`passed=true`。
- `git diff --check`：通过；仅显示现有工作区行尾转换 warning，无 whitespace error。

## Results

字段归属和旧实现冲突已经机器化。当前 Tensor 不足以支撑定义 03 最小图；wired relation 有可靠 simulator/Raw topology 来源但尚未逐 Decision 物化和输入化，stable stateful Flow 的 Raw 仍不足。因此 readiness 保持 `DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1`；缺少 wired 可选 numeric state 本身不再构成最小图 blocker。

## Expected vs Actual

预期是冻结映射并暴露缺口，实际符合。没有为了让图看似可构造而复用过去 hop service、执行结果或旧 mixed physical edge。没有实现 graph builder、GNN、encoder、message passing、GRU、coupling、World Model、Loss、Planner 或训练。

## Known Issues

- edge/cloud 是否进入 Physical Graph 需要研究者确认其 simulator coordinate 是否具有独立空间意义。
- Physical topology 的 radius/kNN/radius+kNN 未决定，符合本 Step 范围。
- Frozen Raw 尚未逐 Decision 物化 wired relation，且缺完整 Flow state、return size/priority/deadline；v4 Sample/v2 Tensor 还未暴露 position、CSI、CPU static capacity 和当前完整 Task state。
- Dynamic available CPU 与 wired queue/load/utilization 仍无可靠来源，但它们不是本 Patch 新增的定义 03 minimum blocker。
- 这些 gap 阻止直接进入 graph builder，但不是本 Step 的实现失败。

## Git

Commit / push 状态以最终回执为准。

## Next Step

唯一建议：研究者审阅本映射和 gap 后，单独授权最小 Data Contract Additive Extension；不要自动进入 graph builder。
