# 项目结构与知识入口变更记录

## 2026-09-19：STEP 3.3F Tensor Semantic Completeness

- Model-ready sample 升级 v4，因果 Static/History/Target 保留真实 entity type；Raw schema 与时间边界不变。
- Tensor 升级 v2：增加 Past Outcome H-1 轴、完整 Target entity/task/flow/service、固定 category vocab 和 Comp `allocated_cpu_per_s`。
- machine receipt 对 required semantic checks 取 AND；STEP 3.3 与定义 02 按当前最小数据合同冻结。

## 2026-09-19：STEP 3.3 Model Input Tensor / Collation Contract

- 新增 CPU NumPy fixed-shape JSON sample collation、builder、focused tests 和 machine-readable schema/manifest artifact。
- 保持 STEP 3.1F History-union index、anchor visibility、presence/mask 与 target-only namespace；容量超限拒绝并提供 NPZ round-trip。
- 未进入双图、World Model、Loss、Planner、训练、GPU 或 `locked_test`。

## 2026-09-19：Step 2.4 通信 Outcome 语义最终冻结

- 真实 wired service 接入 `WiredNetworkManager.step` 返回值，Raw Contract 拆分 wireless/wired/total delivered data。
- 固定 `{}` observed no-service 与 `null + mask=false + missing_reason` unavailable 的区别；真实 6 slot / 7 Decision / 14 checks 证据已生成。
- Raw Trajectory Layer / 定义 01 COMPLETE / FROZEN；Dataset/Tensor、模型和训练未开始。

## 2026-09-19：Step 2.1 真实 AirFogSim 验收

- 完成 Step 2.3 Raw Contract 因果完整性收尾：future-task 隔离、真实 CSI/CPU/slot outcome、return route 与双 acceleration 字段全部验收，Raw Trajectory Layer / 01 冻结。
- 完成 Step 2.2 真实 6 步 Raw Trajectory 和独立下一 Decision 验收；Route/Comm/Comp 覆盖真实动作与显式 no-op。
- 版本控制 Step 2.1 v3 和 Step 2.2 的 JSON/manifest，补齐 GitHub 机器证据。
- 新增真实单轨迹四类动作接线与 Outcome/next Decision 证据，修正 vehicle degree/UAV rad heading 合同，纠正 Step 2 专项测试数量为 6。

## 2026-09-18：新定义 Step 1 实施审计

- 将研究者只读 `00–06` 设为当前目标定义链，旧 P4/P6/P0–P10 工作流和结果逻辑归档，原文件与证据不移动、不删除。
- 新增 `docs/PIJWM_IMPLEMENTATION_TRACKER.md`、`docs/implementation_records/README.md`、Step 1 主报告和数据/双图附件。
- 审计确认时间因果、稳定索引、mask/split 等可复用，同时记录严格双图、四类动作、RSSM 边界、逐步规则反馈和完整 planner 闭环的结构性缺口。
- 同步权威计划、AI_CONTEXT、项目/架构/科研/实验/结果索引与机器文档路由；模型、数据、loss、planner、checkpoint 和实验产物未修改。
- 验证使用既有 synthetic CPU 合同 49 项；它们不构成新定义验收。未启动 GPU、未访问 `locked_test`、未进入 Step 2。

## 2026-09-08：第一阶段索引建立

- 新增 `PROJECT_INDEX.md`、`ARCHITECTURE.md`、`RESEARCH_STATUS.md`、`EXPERIMENT_INDEX.md`、`RESULTS_INDEX.md`。
- 新增 `PROJECT_RESTRUCTURE_PLAN.md`，明确训练同步期间的保护边界、分阶段迁移和回滚要求。
- 只读盘点了当前代码、记录、文献、会议材料和机器证据；未移动、删除、重命名或覆盖任何训练/同步产物。
- 首个单 seed 正式验收、当前运行中的 seed、P6 和 `locked_test` 边界均按机器产物和最新过程记录登记。
- 发现并记录当前 PPT 文件与旧 PPT 验收 JSON 的页数和 SHA-256 不一致；未擅自重新验收或改写 PPT。

后续每次索引更新应说明：变更原因、受影响入口、是否触及研究定义、是否触及训练/同步保护区、验证结果和回滚位置。

## 2026-09-09：第二个正式 seed 验收登记

- 登记 seed `20260830` 正式 run 和独立验收入口，并把状态从“运行中”更新为“单 seed通过后暂停”。
- 未改变研究定义、模型、数据、训练配置或阈值；正式产物和验收报告均为新增证据。
- 验证结果为 manifest `79/79`、strict reload `0/0`、9项数值门全部通过；`locked_test`未访问。
- 当前仍需 seed `20260832` 和三 seed审计才能关闭 P4；没有开放 P6。

## 2026-09-09：项目知识与工程结构重构第二阶段

- 新增用户学习路径、AI 检索指南、代码状态索引、已知冲突和归档候选门。
- 在 `code/src/pi_jwm/`、`code/scripts/` 和 `code/tests/` 增加目录级导航，明确当前正式、支撑、原型和历史兼容边界。
- 新增文件、Python 依赖、artifact、实验、结果、文档权威和延后任务注册表，以及可重复生成/`--check` 的索引脚本。
- 自动映射覆盖 828 个项目文件、604 个 Python 节点和 802 个 artifact 一级目录；12 个 BOM 解析误报修复后 Python 解析错误为 0，14 个历史 manifest 权限错误被如实保留。
- 识别 211 个历史 Python 候选；94 个仍有反向引用、93 个仍有直接测试、0 个被当前正式节点引用。基于全量测试基线和 provenance 风险，本轮采用逻辑归档，没有移动、删除或覆盖任何历史代码与 artifact。
- seed `20260832` 和远端同步已登记为延后且需用户授权；没有启动 GPU、同步或访问 `locked_test`。

## 2026-09-09：长期协作与快速问答闭环

- 将用户科研决策权、AI 独立质疑义务、通俗解释顺序和证据边界写入永久治理与协作指南。
- 将重要实验注册表升级为统一字段合同；新增 6 类历史方法语义注册和 5 类常见问题路由。
- 正式结果注册表现会自动对照原始 acceptance JSON 的 SHA-256、seed、epoch、状态、9 项指标和 `locked_test` 边界；当前 2 项正式结果全部一致。
- 新增只读查询工具，可按自然语言、精确旧实验目录名或精确代码文件名定位入口；查询始终提示回到原始证据验证。
- 最终映射覆盖 834 个项目文件、606 个 Python 节点和 802 个 artifact 一级目录；Python 解析错误 0，历史 artifact 控制文件读取错误 14 个原样保留。
- 验收：统一 `--check`、项目知识/结构 27 项、正式 P4 210 项、compileall 和 diff 检查通过；全量 1636 项为 0 failure/21 个已登记环境或历史错误。
- 本轮未修改研究方法、训练合同或原始产物，未启动 GPU、seed `20260832`、远端同步或 `locked_test`。

## 2026-09-10：ChatGPT AI_CONTEXT 与三方协作闭环

- 新增 `AI_CONTEXT/00_PROJECT_STATE.md` 至 `08_CHANGELOG.md`，分别覆盖当前状态、研究背景、真实架构、数据流、模块地图、实验、研究者决策、已知问题和重要变化。
- `AGENTS.md` 按用户明确授权增加 Research Engineer、ChatGPT Web、研究者三方边界，以及 Context Consistency Check、Git commit/push、私人笔记禁区和冲突处理规则。
- 文档权威注册表新增 ChatGPT 首入口；问题路由新增 `ROUTE-CHATGPT-ONBOARDING`；知识索引生成器会验证九个上下文文件及关键证据边界。
- 当前生成快照覆盖 844 个项目文件、607 个 Python 节点和 802 个 artifact 目录；九个 AI_CONTEXT 文件和 6 类问题路由有效，当前 artifact 控制入口读取错误为 0。
- 验证：AI_CONTEXT 6/6、项目知识/结构 28/28、正式 P4 210/210、compileall 和 diff 检查通过；全量 1643 项为 0 assertion failure/17 个已登记环境或历史错误。
- 未修改模型、loss、metrics、tensor、协议、checkpoint、实验结果或 `code/artifacts/`。

- 2026-09-18: Added Step 2 raw trajectory/four-action contract, minimum closure validation, and evidence artifacts.
- 2026-09-19: Added Step 3.1 model-ready sample/tensor contract, minimal real sample artifact and focused validation; formal dataset/model/training remain unopened.
- 2026-09-19: Corrected Step 3.1R History alignment, fixed typed indices/presence, real DAG capture and relation endpoints; regenerated non-locked evidence without entering Step 3.2.
- 2026-09-19: Finalized Step 3.1F History with past action/outcome, causal History-union indices, aligned historical flow/relation/DAG rows, and an observation-only audit of future action references; Step 3.2 remains unauthorized.
- 2026-09-19: Applied the bounded Step 3.1F-PATCH: Future Action now uses anchor visibility only for admissibility and the shared History-union input namespace for numeric indices; added ID↔index validation, corrected policy provenance, and tracked the audit JSON in Git.
