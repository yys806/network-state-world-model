# AI_CONTEXT 重要变更

## 2026-09-22 STEP 5.1D

- 新增 unified Tensor full-package roundtrip，补齐 sample IDs 与 base Step 3.3 validation provenance。
- 从同一 12-sample bundle 重建 4.3A/4.3B/4.4，并完成 paired recursive prior/posterior/decoder、Loss/KL/raw-unit metric/gradient receipt；当前 5.1D-PATCH receipt 47/47 checks 与 deterministic rebuild 通过。
- 仍不训练、不用 GPU、不访问 formal dataset/locked_test，不进入 STEP 5.2。

## 2026-09-21 STEP 5.1A-PATCH

- 修正 multi-horizon Motion：horizon 2+ 从 anchor-to-future cumulative delta 改为 adjacent-frame local one-step delta，缺失上一 future position component 时下一步对应 mask=false。
- Motion tensor 改为 current `input_entity_index["physical"]` 固定槽位；future permutation/birth/disappearance/target-index 不再改变 supervision slots。
- CSI 增加 current model relation slot、identity、endpoint input slots、type 与 RB 的机器绑定；patched 12-sample non-locked receipt `passed=true`。

## 2026-09-21 STEP 5.1A

- 新增 Future Motion/CSI additive target contract、frozen train-only normalization、stable current-support/RB alignment、raw-unit bridge、NPZ round-trip 和 tamper validation。
- 真实 non-locked development artifact 含 12 samples，receipt `passed=true`；没有训练、GPU、formal Dataset 或 `locked_test`。

只记录影响项目结构、模型实现、实验流程或 AI 上下文恢复的重要变化。微小代码编辑不在此逐条登记。

## 2026-09-20

- 2026-09-21：STEP 4.4 audit 先得到 `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`；研究者随后选择独立 known stochastic outage event 并关闭 residual。Structured RSSM、四类 Action 路由、vehicle/CSI dynamics、规则 transition、动态图与两步 prior rollout 完成；STEP 4.4-PATCH 补齐 carrying/hop、Flow 完成同步、4.3A topology reuse、端点/lifecycle/DAG 规则与 state-changing recursive counterfactual，87/87 machine checks 通过。证据仍是 untrained CPU development，不含 Loss/Training/Planner/GPU/locked-test。
- 2026-09-21：STEP 4.4-PATCH2 将 current-support Existing Return 按 `(task_index, Return)` typed identity 接入 final completion gate；固定 `future_return_birth_supported=false`，缺失 required support 时以 side-state 阻止 final completion。focused 26/26、receipt 91/91；Definition 05 仅记录后续 mask/exclude/classify 要求，未实现 Loss。
- 2026-09-21：STEP 4.4-PATCH3 将 no Return slot 从错误的 no-return 改为显式 unknown，real adapter computation-finished 反事实不会 final-complete；DAG release 忽略 invalid edge并要求全部有效前驱完成；terminal Flow 使用冻结 status vocabulary 同步 COMPLETED。focused 30/30、正式 receipt 92/92、6 文件独立重建 hash/size equality 均通过，STEP 4.4 正式 COMPLETE / FROZEN。

- 完成 STEP 4.3B-PATCH：P2A/P2C value processor 回到 Definition 03 联合上下文公式；`Z_t^{PI,L_g}` 保留 STEP 4.3A 全部 11 个 structural blocks，并加入 endpoint/index/presence/validity equality、完整 digest、round-trip 与 structural tamper 机器检查；Comm CSI width 改为读取 tensor contract `n_comm_rb`。仍为未训练 CPU development evidence。

- 完成 Step 2.4 通信 Outcome 语义最终验收：真实 wired manager service 与 wireless event 分拆，total 聚合和 empty/missing 语义冻结；6 slot、7 Decision、14 checks 通过。
- 完成 Step 2.3 Raw 因果最终验收：隔离 future task、补齐真实 Decision/Outcome 字段、执行 return route、冻结 raw/canonical acceleration；Raw Trajectory Layer / 01 完成并冻结。
- 完成真实 AirFogSim Step 2.2 六步 Raw Trajectory：下一 Decision 独立采集，Route/Comm/Comp 显式 no-op，跨步 ID/lifecycle 对齐。
- 将 Step 2.1 v3 与 Step 2.2 JSON/manifest 纳入 Git，补齐机器证据 GitHub 可追溯性。
- 完成真实 AirFogSim Step 2.1 单轨迹四类动作与 Outcome/next Decision 对齐；修正实体 heading 单位，并记录仿真器 acceleration 观察事实。

## 2026-09-19

- active workflow 切换为研究者最新只读 `00–06` 定义；旧 P4/P6/P0–P10 逻辑归档，原证据和原验收边界保留。
- 新增 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`，完成 Step 1 定义—实现审计。
- AI_CONTEXT 现在区分目标定义、被审计的当前代码、旧协议实验和新定义 `NOT_STARTED` 状态。
- 49 项旧 synthetic CPU contract 测试通过，仅证明旧实现可执行；未修改模型、数据、loss、planner、checkpoint，未启动 GPU，未访问 `locked_test`，未执行 Step 2。

## 2026-09-10

- 初次建立 `AI_CONTEXT/00_PROJECT_STATE.md` 至 `08_CHANGELOG.md`。
- 将研究状态、真实架构、数据流、模块导航、实验、研究者决策、已知问题分开，防止摘要层混淆事实与科研解释。
- 将 `AI_CONTEXT/` 接入项目文档权威、自然语言问题路由和知识索引 `--check`。
- `AGENTS.md` 增加三方角色、Context Consistency Check、Git commit/push、私人笔记禁区和冲突处理规则。
- 科研语义未改变；没有修改模型、loss、metric、协议、tensor、checkpoint 或实验产物。
- 验证：AI_CONTEXT 6/6、项目知识/结构 28/28、正式 P4 210/210 通过；全量 1643 项为 0 assertion failure/17 个已登记环境或历史错误。
- 本次完整重构以本文件所在的 Conventional Commit 推送到 GitHub `main`；精确提交以分支 `HEAD` 为准。

## 2026-09-09

- 建立 `docs/` 人类导航和 `docs/registries/` 机器注册表。
- 两个实体级双图 RSSM 正式 seed 分别通过单 seed non-locked 验收；第三 seed 延后。

## 维护规则

每个有效任务完成前执行 Context Consistency Check，只更新真正受影响的文件：current state、research、architecture、data flow、module map、experiments、decisions、known issues、changelog。实现事实变化必须与相应 AI_CONTEXT 更新进入同一任务。

Source of truth：本 changelog 只说明发生了什么；实现和实验真假仍由源码/config/experiment/audit 决定。

## 2026-09-20 STEP 4.2C-B-PATCH

- 修正 Raw amendment 将 current hop `target_node_id` 误作 logical destination 的问题；Input 改用已建立 route terminal，Return 改用 `return_destination_id`，并保存 source/capture-phase provenance。
- 新增同 Epoch destination continuity 与真实 multi-hop single-Flow validator；真实两跳证据通过 FlowID/Epoch/destination/route-revision/E2E no-double-count 检查。
- 顶层 acceptance 对 19 项 required checks 与 scope 取实际 AND；negative tamper/fake-multihop 测试通过。未进入 Sample/Tensor、Graph Builder、模型、训练、GPU 或 locked_test。

Unverified：未在 Git diff 和相应证据中出现的变化不得仅凭本文件推断。

- 2026-09-18: Step 2 froze raw single-step and four action-family contract; added source adapter, focused tests, evidence script, and protocol artifacts.
- 2026-09-19: Added Step 3.1 model-ready sample/tensor contract, minimal real sample artifact, source-hash manifest, focused tests, and navigation records; formal dataset and training remain unopened.
- 2026-09-19: Step 3.1R corrected History alignment, fixed-index presence/padding, DAG Raw capture, typed target namespaces, relation endpoints, and unresolved action reference handling; real v2 Raw and minimal sample evidence regenerated.
- 2026-09-19: Step 3.1F finalized History with past action/outcome, History-union physical/task/flow indices, history relation/DAG/flow alignment, late-entry/disappearance fixtures, and observation-only future-action reference audit; Step 3.2 remains unopened.
- 2026-09-19: Step 3.1F-PATCH separated anchor visibility from unified History-union Future Action indices, added ID↔index validation and disappearing-object regression coverage, corrected the machine policy, and committed the observation-only audit JSON provenance; Step 3.2 remains unopened.
- 2026-09-19: STEP 3.2 completed a minimal non-locked Raw-to-Dataset batch/split/preprocessing validation with three independent development trajectories, train-only masked statistics, deterministic rebuild and round-trip; formal Dataset/training remain unopened.
- 2026-09-19: STEP 3.2-PATCH finalized Dataset isolation evidence with real Raw provenance/lineage/time-grid checks, a separate 12-window development future-reference audit, physical normalization units, and computed validation-report checks; formal Dataset/training/GPU/locked_test remain unopened.
- 2026-09-19: STEP 3.2-PATCH-RECEIPT corrected validation `passed` to require all checks and explicit scope booleans, added negative failure propagation coverage, and reused the frozen sample schema constant; STEP 3.2 is now COMPLETE / FROZEN.

## 2026-09-20 STEP 4.2C-C-PATCH Presence-aware Normalization & Full Flow Semantic Coverage

- `_iter_numeric()` 与 stats policy 固定为 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`，并新增 presence=false completed/superseded 重复 lineage negative fixture。
- History Logical、History Carrying、target Logical、target Carrying 四组 Sample→Tensor 全字段语义检查已接入 receipt；target carrying namespace 明确为 future ground-truth/deterministic-transition state，不是 learned prediction head；ID/provenance tamper 会失败。
- 23/23 focused tests、receipt/build、deterministic rebuild、serialize/load、scope checks 通过；`graph_builder/information_graph/physical_topology/training/gpu/locked_test/formal_dataset=false`。

## 2026-09-20 STEP 4.2C-C Stateful Flow Sample/Tensor Additive Extension

- 新增 `step4_2c_c_flow_sample_tensor_v1.py`、builder、真实低 wired capacity cross-slot runner、focused tests、合同/实施记录和最终 artifact。
- C-B Raw logical Flow/Carrying state 进入独立 History-union/target `logical_flow` namespace；Flow/Carrying、known inactive/padding、Input/Return、DepData=0、train-only numeric normalization、capacity overflow 和 round-trip 均有机器证据。
- 基线阶段的 focused tests、4.2B/4.2A/3.3 回归、真实 trace、deterministic rebuild/hash、serialize/load、tamper 通过；Patch 进一步将 focused suite 扩展为 23/23。
- 该 Step 当时不宣称 formal Dataset、Return multi-hop/reroute runtime、formal capacity、Graph Builder、模型或性能，并建议另行授权 Graph Builder；此后 STEP 4.3A 已完成 builder，原有数据证据边界不变。
## 2026-09-20 STEP 4.3A Typed Dual-Graph Builder

- 新增 11 个 typed graph blocks、显式 development-only Physical topology config、Align/GeoComm 与 current-frame target isolation。
- receipt 对 24 项 required checks 实际 AND，20 项 negative/counterfactual 全通过；无 Encoder、GNN、World Model、Loss、Planner、训练、GPU 或 locked_test。

## 2026-09-20 STEP 4.3B Dual-Graph Encoder

- 新增 type-specific mask-explicit encoders、四类 object-wise presence-gated GRU、五类 directed relation processors、relation-wise masked mean、P2A/P2C 和 residual+LayerNorm updates。
- Flow 采用 Logical/Carrying 独立分支后 fusion，只输出一个 Flow relation latent；delivered/Epoch/index 不作为 learned numeric input。
- 输出冻结为 aligned `Z_t^{PI,L_g}`；artifact 是未训练 CPU development evidence，不进入 World Model/dynamics/loss/planner/training。
- 2026-09-21：STEP 5.0 冻结 Definition 05 十项研究决定并审计旧实现。新 active contract 固定 deterministic mean decoder、Motion/CSI mask-MSE、family KL、overshooting OFF、family-only posterior teacher、joint training 与 prior-only validation。发现 future position 尚未 normalized/tensorized、future CSI target 缺失；Loss/Training 仍未实现，无 GPU/locked-test。
# 2026-09-22 STEP 5.1B-PATCH

- 修正原 5.1B 的跨 horizon target aggregation、mask 不可见、独立 PriorPredictor、zero-h/identity loss、batch-level free bits、raw-unit metric 和硬编码 receipt 问题。
- 当前 CPU receipt 真实调用 STEP 4.4 `initialize_latent → one_step → phy_prior/comm_prior → vehicle_decoder/csi_decoder`，并验证逐 horizon isolation、mask evidence、逐维 free bits、gradient、raw-unit metric 与 target isolation；仍未进入 training/optimizer/GPU/locked-test。

# 2026-09-22 STEP 5.1C-PATCH

- 完成 unified development bundle 的 Physical/Communication slot-wise identity proof、端点真实 ID 反查、10/74 capacity exact alignment、no-prefix/no-crop 机器证明和 tamper negative。
- 重新拟合 `unified_flow_train_normalization_stats.json`：仅使用 8 个 `dev_train` samples 的 History，4 个 validation samples 排除，Future Target 不参与；旧 5-sample stats 仅保留为历史 provenance。
- 截至 STEP 5.1C-PATCH 当时，4.3A/4.3B/4.4 unified rebuild 和 paired 5.1B acceptance 尚未执行；该历史边界已由后续 STEP 5.1D-PATCH 闭合，STEP 5.2 仍需单独授权。
- 2026-09-22：STEP 5.1D-PATCH 完成 Evidence / Context / Action-Coverage Closure。receipt 改为 executed/forbidden scope，跟踪 compact GitHub evidence；机器比较 unified stats source IDs、4.2A frozen batch 恢复的 fit source 与逐窗口 normalized samples；新增 runtime prior-target isolation 与 posterior sensitivity negative test。真实覆盖为 Mobility=48、Comm=1、Route=0、Comp=0，Route/Comp 仅显式 no-op；47/47 checks 通过。仍为 untrained CPU/non-locked evidence。
