# STEP 1 — New Definition → Current Implementation Audit

日期：2026-09-18
状态：`AUDITED / WAITING_FOR_RESEARCHER_REVIEW`
范围：只审计新定义与当前实现，不进入 Step 2

## 1. Step Goal

把研究者授权的 `D:\shen\OB\科研\PIJWM` 中最新 `00–06` 目标定义，同 `D:\shen\PKU\PIJWM` 当前可核验的源码、配置、测试和机器产物逐项对照。每项只使用以下实现判断：

- `DIRECT_REUSE`
- `MINOR_MODIFICATION`
- `STRUCTURAL_CHANGE`
- `MISSING`
- `RESEARCHER_DECISION_REQUIRED`
- `HISTORICAL_ONLY`

本 Step 的输出是实施边界、可复用机制、关键冲突和唯一建议的 Step 2。它不实现新模型、不重建数据、不训练、不访问 `locked_test`，也不把审计结果写成新方法验收。

## 2. Definition Basis

七个目标定义文件在只读目录中读取，文件指纹由 `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/initial_snapshot.json` 保存：

| 定义文件 | 主要审计章节 | SHA-256 | 行数 |
| --- | --- | --- | ---: |
| `00课题研究问题与总体链路.md` | 一、课题研究问题；三、物理—信息联合建模；四至六、总体离线/在线闭环 | `ac651586cadd2638c7cb8f68e80144c5bf087b62b2e1923c6d6625c5de2b82f0` | 528 |
| `01仿真系统与原始轨迹.md` | 一、运行对象；二、单步运行；三、轨迹采集与身份/时间对齐；四、覆盖边界 | `01478d0c61065adccc50cd58747cb28cdf554a5b963e07ef154b8afcbad43ac1` | 366 |
| `02数据集构建与模型输入.md` | 一、样本窗口；二、对象/index/mask/DAG；三、模型读取字段；四、因果安全与 split | `702a12e0c723d73ce91da1e8b7c3e781e688733efa0ef34df09c352b17977b6e` | 455 |
| `03物理-信息双图建模.md` | 2、语义划分与跨图关系；3、物理图；4、异构信息图；5、消息传递和跨图耦合 | `6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e` | 812 |
| `04世界模型预测边界与当前模型.md` | 二、学习/规则/派生边界；三、单步动态、动态图与多步闭环；四、当前实现差异 | `ef49cd0802a163886ae324879c01e9fbbd3f3f2dc1ec4079e4022b2c4f7a46c5` | 510 |
| `05模型训练、Loss与评价.md` | 一、未知动态 loss；二、posterior teacher/prior rollout；三、预测和系统评价 | `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9` | 260 |
| `06策略器与候选动作规划.md` | 二、动作生成与合法性；三、逐候选 rollout/目标/选择；五、反馈与重规划 | `f20294bd8708076ae7583be679a8182a0990ecf6777e05df378ae4c2855431f1` | 331 |

这些文件是目标定义，不自动证明当前实现。当前实现事实以本仓库 source/config/test/artifact 为准。

## 3. Initial State

- Repository：`https://github.com/yys806/network-state-world-model`
- Branch：`main`
- Base commit：`829276241a0da72d3a5393946086daba40b0a0fe`
- 初始已有 dirty 文件：`task_plan.md`、`progress.md`、`findings.md`。它们在本 Step 前已有 2026-09-16 AirFogSim 分享包记录，本 Step 保留其内容；开始时保存为 `.git/step01-original-*.md` 和 `.git/step01-preexisting.patch`。
- 当前旧实现入口包括 `formal_dual_graph_world_model_v1.py`、`formal_entity_aligned_rssm_world_model_v1.py`、`airfogsim_tensor_v2.py`、`formal_airfogsim_window_v1.py`、`formal_world_model_loss_v1.py`、`formal_world_model_metrics_v1.py`、`formal_candidate_rollout_planner_v1.py` 及其 runner/audit。
- 旧证据仍有两个 P4 entity-RSSM 单 seed unlocked 验收，但它们属于旧 tensor、旧图语义、旧 loss 和旧训练合同；不能自动成为新 `00–06` 的性能证据。

## 4. Files Involved

### 4.1 本 Step 修改或新增

- `AGENTS.md`
- `docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- `docs/implementation_records/README.md`
- `docs/implementation_records/STEP_01_DATA_GRAPH_AUDIT.md`
- `docs/implementation_records/STEP_01_AUDIT.md`
- `记录/本地计划表.md`
- `记录/PIJWM主文档.md`
- `记录/8.12之后推进.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `AI_CONTEXT/00_PROJECT_STATE.md` 至 `AI_CONTEXT/08_CHANGELOG.md`
- `docs/PROJECT_INDEX.md`
- `docs/ARCHITECTURE.md`
- `docs/RESEARCH_STATUS.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/RESULTS_INDEX.md`
- `docs/CHANGELOG.md`
- `docs/registries/document_authority.json`
- `docs/registries/question_routes.json`
- `docs/registries/generated/artifact_catalog.csv`
- `docs/registries/generated/registry_summary.json`
- `docs/registries/generated/tracked_file_inventory.csv`

### 4.2 读取但未修改的主要实现证据

- 轨迹与采集：`code/src/pi_jwm/full_dual_graph_collector_contract_v1.py`、`airfogsim_full_dual_graph_collector_v1.py`、`airfogsim_full_dual_graph_collector_v2.py`、`formal_airfogsim_collector_adapter_v2.py`、`action_attempt_ledger_v1.py`
- 数据与张量：`code/src/pi_jwm/formal_airfogsim_dataset_v1.py`、`formal_airfogsim_window_v1.py`、`airfogsim_tensor_v2.py`
- 图、规则和模型：`code/src/pi_jwm/formal_graph_ops_v1.py`、`formal_dual_graph_world_model_v1.py`、`formal_deterministic_rule_layer_v1.py`、`formal_entity_aligned_rssm_world_model_v1.py`
- 训练、loss、评价和 planner：`code/src/pi_jwm/formal_world_model_loss_v1.py`、`formal_world_model_metrics_v1.py`、`formal_candidate_rollout_planner_v1.py`、`code/scripts/run_formal_dual_graph_gpu_train_v1.py`
- 既有 synthetic CPU 合同测试：`code/tests/test_formal_entity_aligned_rssm_world_model_v1.py`、`test_formal_deterministic_rule_layer_v1.py`、`test_formal_candidate_rollout_planner_v1.py`、`test_formal_world_model_loss_v1.py`、`test_formal_world_model_metrics_v1.py`
- 当前非锁定 tensor 与旧验收入口：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`、相关 P4 acceptance/audit 目录

### 4.3 本 Step 机器化审计辅助证据

- `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/initial_snapshot.json`
- `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/cpu_existing_tests.json`
- `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/cpu_existing_tests.log`

## 5. Changes

1. 在 `AGENTS.md` 中把 `00–06` 设为当前目标定义链，把旧 P0–P10/P4/P6 标为 Historical / Archived，固定科研笔记只读边界、Step 停止门、低成本训练前检查和完成汇报格式。
2. 建立长期 Tracker 和实施记录目录，并把 `Step 1` 限定为审计，不把章节编号误当成后续实现授权。
3. 建立本报告和数据/双图附件，逐项标记当前实现的复用类别、缺口、证据和唯一 Step 2。
4. 在权威计划、进展、理论边界和 `AI_CONTEXT` 中同步当前 Step 1 状态；历史 P4/P6 数字保留原协议边界。
5. 在导航索引和文档注册表中增加 Tracker、Step 记录与新定义审计查询入口。

没有修改 `code/src/pi_jwm/`、`code/scripts/`、`code/tests/` 或 `code/artifacts/` 中的模型、数据、loss、planner、checkpoint 和实验产物。

## 6. Reuse Assessment

| 主题 | 判定 | 当前可复用部分 | 新定义下的主要缺口 |
| --- | --- | --- | --- |
| AirFogSim trajectory；Decision → Execution → Outcome | `MINOR_MODIFICATION` | 三阶段采集顺序、动作尝试账本、稳定 ID、时间网格、失败隔离和 outcome-only 结果边界 | 新字段可获得性、UAV 指令、逐字段可见时刻和原始轨迹正式合同未冻结 |
| Dataset、split、presence/mask、padding、DAG | `MINOR_MODIFICATION` | trajectory 级 split、train-only normalization、稳定 index、presence/mask 分离、端点合法性和 DAG 方向检查 | 新实体/关系/动作 schema 尚未进入正式 manifest；旧 `tensor ready` 不等于新合同 ready |
| Physical / Information 双图 | `STRUCTURAL_CHANGE` | 图算子、附着/承载映射和有向消息传播的部分代码 | 当前 `physical_edge` 混合空间、CSI、rate、任务数和 RB；Agent、Communication relation、Task–Agent typed relation 未形成独立完整合同 |
| entity alignment | `DIRECT_REUSE` | 稳定 ID → 固定 index、端点引用、mask 和实体局部检查 | 需为新 Agent/Comm/Flow/Task 对象重新建立映射和关系类型 |
| Route / Comm / Comp / UAV Mobility action | `MINOR_MODIFICATION` / `STRUCTURAL_CHANGE` / `MISSING` | 旧 offload、RB、return、CPU 规则和事件记录可作来源；Route 端点机制部分可复用 | 完整多跳 Route、逐 RB Comm、CPU action、UAV Mobility action 的采集、张量、执行和后果闭合尚未证明 |
| action routing、单步学习—规则反馈、动态图重构 | `STRUCTURAL_CHANGE` / `MISSING` | 旧 `forward` 有动作分发和规则层；静态端点/承载映射可作基线 | 目标要求每一步学习动态 → 规则更新 → 重构双图 → 下一步；当前 RSSM 修正是在整段规则推演后叠加，未形成完整闭环 |
| RSSM 与 latent layout | `MINOR_MODIFICATION` / `STRUCTURAL_CHANGE` | prior/posterior、KL、prior-only 评价、实体对齐实现思路 | 当前 node、physical_edge、flow、task 都有随机状态；新定义主要只给 Physical / Communication 未知动态设置 stochastic state，Flow/Task 不应自动沿用旧 layout |
| learned / deterministic boundary、loss、overshooting、training | `STRUCTURAL_CHANGE` | masked loss、KL、teacher、overshooting 工具和两阶段训练脚手架 | 当前学习了大量可由规则/派生得到的状态和事件；旧 overshooting 是切片比较，未证明多起点多距离 rollout consistency；训练阶段和 checkpoint 选择合同不同 |
| prediction evaluation | `MINOR_MODIFICATION` | MAE/RMSE/NLL、coverage、F1/AUPRC、逐步指标和 manifest 复核工具 | 评价对象要改为未知动态预测、规则守恒和新四类动作后果；旧 P4 gate 不可直接复用 |
| candidate generation、legality、objective、fallback | `MISSING` / `RESEARCHER_DECISION_REQUIRED` | planner 中可保留逐候选调用骨架和首动作切片 | 缺正式 proposal、领域合法性、安全 fallback、warm start、风险定义、objective 权重和硬约束；这些涉及研究者决策 |
| candidate rollout、execute-first-action、real feedback、replanning | `MINOR_MODIFICATION` / `MISSING` | 共同 belief 下逐候选调用和返回首动作的原型接口 | 新动作与 RSSM 未接通，尚无真实环境执行反馈、历史更新和至少两次决策闭环验收 |
| 旧 tensor、checkpoint、P4/P6 结果 | `HISTORICAL_ONLY` | 可作 provenance、回归和差异解释 | 新语义、输入布局、动作、loss 和门不同；不能 strict-load 新模型或外推新性能 |

## 7. Validation

本 Step 的实际验证：

```text
python -m unittest test_formal_entity_aligned_rssm_world_model_v1 \
  test_formal_deterministic_rule_layer_v1 \
  test_formal_candidate_rollout_planner_v1 \
  test_formal_world_model_loss_v1 \
  test_formal_world_model_metrics_v1 -v
```

- 返回码：`0`
- 结果：`Ran 49 tests in 1.232s`、`OK`
- 范围：既有 synthetic CPU contract；不是新 `00–06` 验收
- 边界：`gpu_started=false`、`locked_test_accessed=false`

另有只读快照：定义文件大小/行数/SHA、初始 dirty 状态、源码指纹、GPU/locked-test/Step 2 状态见 `initial_snapshot.json`；测试逐项输出见 `cpu_existing_tests.log`。

本阶段不执行 GPU、数据重建、模型训练、planner 运行或 `locked_test` 读取。

## 8. Results

- Step 1 审计框架已经建立，当前新定义实现状态为“审计完成、实现尚未开始”。
- 可直接保留的是时间因果、稳定 ID/index、presence/mask/padding、DAG 方向、train-only split 和部分采集/规则工具。
- 需要结构性修改的是严格 Physical / Information 双图、Agent/Comm/Task–Agent 关系、四类动作合同、RSSM latent 范围、单步规则反馈、动态图重构、训练/loss/overshooting 和完整 planner 闭环。
- 通信状态充分性、外生到达/离开、proposal 训练方式、planner objective/risk/fallback、实验预算和新门槛仍属于 `RESEARCHER_DECISION_REQUIRED` 或未冻结边界。
- 旧 P4 两个 seed、旧 tensor、旧 checkpoint 和旧 planner audit 保留为 Historical / Archived evidence；它们没有被删除或重新解释。

## 9. Expected vs Actual

| 预期 | 实际 | 判断 |
| --- | --- | --- |
| 完成 `00–06` 与当前实现的逐项审计 | Tracker、主报告和数据/双图附件已建立，覆盖轨迹、数据、双图、RSSM、动作、训练、loss、评价和 planner | 符合预期 |
| 不修改模型、数据、planner 或启动训练 | 未修改 source/config/test/artifact；未启动 GPU；未访问 `locked_test` | 符合预期 |
| 保留旧证据并标为历史 | 旧计划、旧结果和 checkpoint 未移动/删除；新入口明确 Historical / Archived | 符合预期 |
| 得到新定义已实现或已验收的结论 | 发现多个结构性缺口；新定义不能宣称已实现或已通过 | 符合预期，且按规则阻止扩展 |

## 10. Known Issues / Researcher Decisions

1. `physical_edge` 的通信状态是否足以由规则层推出真实 service，还是必须引入 effective service/residual，不能由 Codex 自行决定。
2. task arrival、entity departure、background load 等外生事件的未来观测/生成边界尚未冻结。
3. CPU action、逐 RB Comm action、完整多跳 Route action、UAV Mobility action 的字段、单位、执行接口和支持范围尚未形成合同。
4. planner 的 objective 权重、风险形式、未来硬约束、fallback、warm start 和 proposal 训练目标尚未冻结。
5. 旧 `记录/本地计划表.md`、`记录/8.12之后推进.md` 和 AI_CONTEXT 中的 P4 状态仍保留历史证据；本次同步增加新的 Step 1 当前入口，后续引用必须先看新入口。

这些问题阻止受影响的科研实现扩展；本 Step 不替研究者作决定。

## 11. Git

- Initial branch/commit：`main` / `829276241a0da72d3a5393946086daba40b0a0fe`
- Step 1 主提交：`16dc7d0`（`docs(pijwm): establish step 1 definition audit`）
- 推送分支：`origin/main`
- 主提交已推送；回执提交的精确 hash 由最终回复给出，因为提交不能在自身内容中预先记录自己的 hash。

## 12. Next Step

唯一建议的 Step 2：**冻结一个决策步的原始轨迹字段与 Route / Comm / Comp / UAV 四类动作映射合同**。只建立 `Decision → Action → Execution → Outcome → 下一 Decision` 的字段、单位、可见时刻、缺失语义、实体 ID 和真实采集/执行入口，并用已有非 locked 片段或合成样例做低成本断言；不重构模型、不生成大数据、不训练。通信状态充分性和外生事件仍留给研究者决定。研究者审阅并明确授权前，不执行该 Step。

## Git Receipt

- 主提交：`16dc7d0`（`docs(pijwm): establish step 1 definition audit`）
- 主提交 push：成功，`origin/main` 从 `8292762` 前进到 `16dc7d0`
- 回执提交：本小节所在提交；精确 hash 见最终完成汇报
- 远端分支：`origin/main`；回执提交推送后再次核对远端 HEAD
