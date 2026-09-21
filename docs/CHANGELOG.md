# 项目结构与知识入口变更记录

## 2026-09-21：STEP 5.0 Definition 05 Decision Freeze / Context Sync

- 新增 Definition 05 loss/training/evaluation 决策合同与 STEP 5.0 实施审计记录，冻结 10 项研究者决策；与旧只读 Definition 05 冲突的 NLL/Event/Residual/overshooting 条款由更晚的明确决策取代。
- 审计当前 STEP 4.4、Dataset/Tensor 与历史 loss/metrics/RSSM/training runner，逐项标注 `DIRECT_REUSE`、`MINOR_MODIFICATION`、`STRUCTURAL_CHANGE`、`MISSING` 或 `HISTORICAL_ONLY`。
- 当前训练前阻塞为未来逐 RB CSI target 缺失，以及未来 Motion position 尚未归一化并张量化。未实现 STEP 5.1，未训练、未用 GPU、未访问 `locked_test`、未生成正式 Dataset。

## 2026-09-21：STEP 4.4 Structured RSSM World Model Contract

- PATCH3：冻结 current-side 缺少 `Task.return_size`，所以 real adapter 显式保留 Return requirement unknown，unknown/known-required-no-slot 均阻止虚假 final completion且 side-state 可区分。DAG release 只看有效前驱并要求全部完成；terminal Flow 使用冻结 status vocabulary 同步 COMPLETED，partial/intermediate 保持非完成。focused 30/30、正式 receipt 92/92、6 文件独立重建 hash/size equality 通过；STEP 4.4 COMPLETE / FROZEN。

- 研究者将 wireless outage 冻结为独立 known stochastic service event，并关闭 learned outage/rate/service residual heads。
- 新增 structured h/z、diagonal-Gaussian prior/posterior、四类局部 Action routing、独立 dynamics graph interaction、vehicle/CSI learned heads、deterministic rules、dynamic graph rebuild 与 recursive prior rollout。
- wired capacity 从真实 source config 读取；Flow Carrying-derived membership/count 与真实 `WiredNetworkManager` equality 通过。focused 16/16、machine receipt 71/71。
- artifact 明确为 untrained CPU development evidence；无 Loss/optimizer/Training/GPU/Planner/locked-test/formal Dataset/performance claim。
- PATCH2：existing Return Flow 只按 current-support `(task_index, flow_type_index=Return)` 绑定，Input/其他 Task Return 不可替代；`future_return_birth_supported=false`，缺少 required slot 时阻止 final completion并输出 side-state。focused 26/26、machine receipt 91/91。

## 2026-09-20：STEP 4.3B-PATCH Cross-Processor Formula & Z_PI Structural Interface Closure

- P2A/P2C value processors now consume the complete Definition 03 joint contexts, with gates and values kept as separate processors.
- `Z_t^{PI,L_g}` output preserves all eleven STEP 4.3A structural blocks; semantic equality, complete digest and serialize/load checks include structural side information.
- Comm CSI width is derived from tensor-contract `n_comm_rb` with explicit mismatch rejection. Acceptance artifact reports 48 required and 29 negative/counterfactual checks, all passing; evidence remains untrained CPU development only.

## 2026-09-20：STEP 4.3B Definition 03 Dual-Graph Encoder Contract

- 新增 mask-explicit type-specific MLP、Physical/Agent/Task/Flow object-wise GRU、五类 directed relation processors、family-wise masked mean、独立 node update 与 P2A/P2C。
- Logical Flow 与 Carrying 独立编码后 fuse，只形成一个 Flow relation latent；Future Target/Action、delivered、Epoch 和 numeric ID 不进入 learned inputs。
- 新增 additive Physical-relation train-only stats、input audit、37 项 actual-AND receipt、counterfactual/permutation/round-trip/CPU backward 证据。
- artifact 明确为 `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`；无 World Model、prediction、loss、planner、training、GPU 或 locked_test。

## 2026-09-20：STEP 4.3A Definition 03 Typed Dual-Graph Builder Contract

- 新增冻结 Tensor `history[-1]` 到 11 个 typed Physical/Information/cross-domain blocks 的确定性 builder；Flow 保持 logical multiedge，Carrying 仅为 side state。
- Physical topology 只读位置/运动与显式 development config；Comm validity 与 CSI mask 分离，Align/GeoComm 不引入旧 Task/Flow↔Physical shortcuts。
- acceptance 对 24 项 required checks 实际 AND，并保存 20 项 negative/counterfactual、deterministic digest、round-trip 与 source hashes；未实现 Encoder、GNN、World Model、Loss、Planner 或训练。

## 2026-09-20：STEP 4.2C-C-PATCH Presence-aware Normalization & Full Flow Semantic Coverage

- Flow normalization stats 固定为 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`，并补 presence=false completed/superseded repeated-lineage negative fixture。
- Sample/Tensor receipt 新增 History/target Logical/Carrying 四组全字段 semantic equality、ID/provenance tamper、target carrying future-ground-truth namespace、target namespace/Epoch、placeholder/bounds/route-mask 与 normalization policy 的实际 required checks。
- 更新 contract/record/Tracker/AI_CONTEXT、重建 artifact/manifest；23/23 focused、deterministic rebuild、serialize/load 和 scope 通过；未触及 Graph Builder、模型、训练、GPU 或 `locked_test`。

## 2026-09-20：STEP 4.2B Stateful Flow Source Audit

- 新增 Definition 03 Flow source audit helper、builder、focused tests 和 provenance artifact。
- 机器化记录 Input/Return/DepData、Task/Flow progress、Flow/Hop、route revision 和 remaining source gap；综合 verdict 为 `FLOW_CONTRACT_NOT_YET_SUPPORTED`。
- 同步 Tracker、authority records、AI_CONTEXT；未实现 Graph Builder、模型或训练。

## 2026-09-20：STEP 4.2B-PATCH Flow Readiness Receipt

- Flow verdict 改为由 Flow-specific evidence 实际计算，并加入 verdict tamper negative test。
- dynamic CPU/storage/wired queue 等与 Flow readiness 解耦；stable Flow ID 与 DepData 改为 implementation fact + researcher decision boundary。
- provenance manifest 增加 source symbol、semantic claim 和 symbol-level anchor。

## 2026-09-20：STEP 4.2A-PATCH Comm Mask Semantics & Gap Reclassification

- 解耦 wireless structural relation presence/validity 与 CSI observed/mask；missing CSI 保留 relation，Sample/Tensor 使用 masked zero placeholder 和 missing reason。
- 强化 Sample/Tensor validator 与 machine counterfactual，覆盖无线/有线 no-CSI、absent relation、tampered validity 和 masked placeholder。
- return size/priority/deadline 改为 simulator observer available but frozen Raw not exposed；stateful Flow 继续 Raw-insufficient，未实现 Graph Builder。

## 2026-09-20：STEP 4.2A Existing-Source Graph Input Additive Extension

- 新增独立 Raw amendment / Sample v5 / preprocessing v1 / Tensor v3 链路，把 position、wireless CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 输入化。
- 保持 History union、Future Action、trajectory split、train-only normalization 和 mask/index 合同；旧 Step 2/3 artifacts 不覆盖。
- 新增 focused tests、validation receipt、gap-resolution overlay 与 hash/provenance manifest；stable stateful Flow 继续 blocked，未实现 graph builder、模型或训练。

## 2026-09-19：STEP 4.1-PATCH Minimum Gap Semantic Correction

- wired 最小 relation 改为 simulator/Raw topology 已有但尚未逐 Decision 暴露；以 relation type + CSI mask 表示，无需伪造 CSI。
- wired 可选动态 numeric state 标记为非 03 minimum；CPU capacity 改为 Agent static capability，并与 Comp allocation、actual service、dynamic available CPU 分离。
- 更新 validator negative tamper、mapping artifact、原 Step 4.1 记录、Tracker、authority records 与 AI_CONTEXT；仍未实现 graph builder 或训练。

## 2026-09-19：STEP 4.1 PI Graph Object–Field–Relation Mapping

- 新增机器 mapping schema、四张独立 matrix、validator/focused tests 和 hash/provenance artifact。
- 冻结严格 Physical/Information 字段归属、Task-Agent 四类关系、Flow/DAG 边界与 forbidden placement；未实现图构建或模型。
- 同步 Tracker、authority records、AI_CONTEXT 和项目索引；scope 保持 training/GPU/locked_test/formal_dataset 全 false。

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
## 2026-09-20：STEP 4.2C-A Causal Flow Ledger Feasibility Audit

- 新增 observation-only Real Event → Causal Ledger feasibility helper、builder、focused tests、合同、实施记录和源码 SHA-256/symbol provenance artifact。
- 机器 verdict=`CAUSAL_FLOW_LEDGER_PARTIALLY_FEASIBLE`：Input/Return hop events 可部分追溯；跨 hop remaining、final delivery、reroute payload ownership 仍需 hook；DepData 不由 DAG 虚构。
- 未修改 Raw/Sample/Tensor、simulator、Graph Builder、World Model、Loss、Planner、training、GPU 或 locked_test。
## 2026-09-20：STEP 4.2C-A-PATCH Existing Event → Causal Ledger Derivability

- 新增 audit-only pure replay：按 logical destination 过滤 final delivery，因果维护 E2E delivered/remaining，并验证 Input/Return multi-hop、holder transition 和 same-destination reroute。
- 明确 `transfer_row.flow_completed` 不是 logical Flow completion；真实 Step 2.4 trace 作为部分 real evidence，严格 phase replay 作为 schema-equivalent fixture。
- 机器 verdict 更新为 `CAUSAL_FLOW_LEDGER_FEASIBLE`；destination-change epoch、DepData、长期 Ledger、Raw extension 和 Graph Builder 仍未授权。
## 2026-09-20：STEP 4.2C-B Causal Flow Ledger & Raw Additive Extension

- 实现稳定 FlowID/FlowIndex、Flow/Carrying 分离、Input/Return lifecycle、E2E progress、holder、RouteRevision、clean-boundary Epoch lineage 和 DepData zero-instance guard。
- 新增独立 Raw amendment：`O_t` 只含此前已发生事件更新后的 Ledger，当前 `Y_t` 只进入 `O_{t+1}`；legacy `flow_completed` 只映射为 hop/stage completion。
- 真实 non-locked trace 覆盖 Input/Return，独立真实 trace 覆盖 Input 两跳；未真实覆盖场景明确保留为 contract fixture，不进入 Sample/Tensor 或 Graph Builder。

## 2026-09-20：STEP 4.2C-B-PATCH Logical Destination Provenance

- 修正 Raw amendment 的 next-hop-as-destination bug：Input logical destination 使用 established offload route terminal，Return 使用 Decision `return_destination_id`；旧 `target_node_id` 语义不改。
- Flow row 增加 destination source/capture phase，并新增 within-Epoch destination continuity 与普通 hop 不增 Epoch/RouteRevision 的机器约束。
- 真实两跳 Input 按单 FlowID/Epoch、固定 destination、distinct hops 和 final-hop-only E2E 验收；fake multi-hop 与语义篡改负例会令顶层 acceptance 失败。
- 未修改 simulator、Sample/Tensor、Graph Builder、模型或训练；GPU/locked_test 未使用。

## 2026-09-20：STEP 4.2C-C Stateful Flow Sample/Tensor Additive Extension

- 新增 C-B Raw → 独立 `logical_flow` History-union/target Sample/Tensor additive extension、builder、focused tests、合同/实施记录和机器 artifact。
- Flow 与 Carrying state、known inactive 与 padding、Input/Return/DepData vocabulary、train-only numeric normalization、explicit development capacity、round-trip 和 receipt tamper 均已机器化；真实低 wired capacity cross-slot trace 补齐多个 Decision 的 carrying evidence。
- 12/12 focused、4.2B/4.2A/3.3 regressions、deterministic rebuild/hash、serialize/load 和 scope checks 通过；不进入 Graph Builder、模型、Loss、Planner、训练、GPU、locked_test 或 formal Dataset。
- 下一步仅建议研究者另行授权 Definition 03 Graph Builder Contract。
# 2026-09-21：STEP 4.4 Communication Service Audit Gate

- Added a source-hashed dependency matrix and computed three-way sufficiency verdict before Structured RSSM implementation.
- Found nominal pre-outage wireless rate rule-recoverable, but actual rate depends on a random per-RB outage realization unavailable at Decision time. Wired capacity/active-flow competition are separate additive gaps.
- Verdict is `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`; no residual target, World Model, training, GPU, planner, or locked-test work was started.
# 2026-09-21：STEP 5.1A Future Motion / CSI Target Contract

- 新增 additive Future Target sample/tensor contract：Vehicle delta Motion `[delta_x, delta_y, delta_z, next_speed]`、future outcome per-RB CSI、current support/RB identity alignment、component masks、wired/missing/unsupported side metadata。
- 复用 STEP 4.3B frozen train-only normalization stats，完成 normalized↔raw bridge、NPZ serialize/load、tamper checks、deterministic rebuild 和真实 non-locked development receipt；12 samples，receipt `passed=true`。
- 明确边界：不是正式 Dataset、Loss/Posterior/Metric、训练、GPU 或 `locked_test` 证据；当前模型尚未读取 Future Target。
# 2026-09-21：STEP 5.1A-PATCH Multi-Horizon Motion / Stable Slot Alignment

- 修正 horizon 2+ Motion：从固定 History anchor 的累计位移改为相邻 future frame 的 local one-step displacement，并新增缺失前一 future position component 的 mask 回归。
- Motion tensor 现在严格按 current physical input slots；future row/target-index permutation、future-only Vehicle 和 disappearing Vehicle 不再改变槽位。
- CSI 增加 current model relation slot、relation identity、endpoint input slots、type 与 RB identity 的机器验收；12-sample non-locked receipt 重建通过。
