# PI-JWM Current State Snapshot

更新时间：2026-09-21

这是 ChatGPT 网页端进入仓库后的第一读取入口。它只提供当前快照和继续查证的路径，不替代源码、配置、checkpoint、metrics、manifest 或 audit。

## Source of truth

事实优先级固定为：当前源码/config/experiment → `AI_CONTEXT/` → 普通项目文档 → 历史聊天或推断。任何正式科研结论都必须回到第一层验证。

## 当前状态

- 项目：PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）。AirFogSim 只是参考仿真器和数据生成工具。
- 当前 active workflow：研究者最新只读 `00–06` 定义链；工程执行入口为 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`。
- 当前 Step：`STEP 5.1A-PATCH — Multi-Horizon Motion Semantics & Stable Slot Alignment Fix` 已完成并冻结 target contract。初版 horizon 2+ Motion 错用 History anchor 且 Motion rows 依赖 future entity order；现已改为 local one-step displacement，并按 current physical input slots 固定对齐。CSI 已显式绑定 current model relation slot/identity/endpoint/type/RB。Loss/Posterior/Metric 仍为 NOT STARTED。STEP 4.4-PATCH3 继续 COMPLETE / FROZEN。
- `STEP 3.2-PATCH` 已补齐 Dataset isolation provenance、time-grid、development future-reference audit 与 normalization units；仍是 observation-only/non-locked evidence，不是正式 Dataset。
- `STEP 3.2-PATCH-RECEIPT` 已修正顶层 acceptance AND、显式 scope checks 和 frozen sample schema reuse；STEP 3.2 现正式 COMPLETE / FROZEN。
- STEP 4.4-PATCH3 已在源码中显式区分 Return requirement 的 known/unknown：冻结 current-side 不含 `Task.return_size`，所以 no slot 是 unknown，不是 no-return；unknown 或 known-required/no-slot 都不能错误 final-complete，但 side-state 可区分两者。DAG 只按有效前驱动态释放，terminal Flow completion 同步 remaining/presence/carrying/status。仍是 untrained CPU development evidence，不代表预测精度或训练结果。
- Definition 05 v1 已决定：deterministic mean decoder；Motion/CSI family-wise mask-MSE；Phy/Comm analytic KL + warm-up/free-bits；overshooting OFF；Future Target 仅进入 family-specific training posterior；joint training；prior-only validation；checkpoint=`argmin L_Val`；逐 horizon raw-unit Motion/CSI MAE/RMSE。STEP 5.1A 已补齐 target contract，但 Loss/Posterior/Metric/Training 尚未实现，禁止开始训练。
- 新定义实现状态：Raw、最小 Dataset/Tensor、Typed Graph Builder、Dual-Graph Encoder 与 Structured RSSM World Model Contract 已验收；Definition 05 的科研决策已冻结，但 Loss/Posterior/Metric/Training 实现仍未开始。Physical topology、Encoder/World Model 参数仍是 development-only。
- 审计结论：时间因果、稳定 ID/index、mask/split、typed graph 与 Definition 03 encoder 已落地；`Z_t^{PI,L_g}→xi_t^Lat`、目标 RSSM 边界、逐步规则反馈和完整 planner 闭环仍需后续授权与实现。
- 当前运行：没有正式 GPU 训练或远端同步任务；旧 `seed=20260832` 仍不得自动启动。
- `locked_test_accessed=false`；`formal_performance_claim_ready=false`；本 Step `training=false`、`gpu=false`。

## 当前最重要问题

当前实现不能按模块名称或旧测试外推性能。STEP 4.4 已把 `Z_t^{PI,L_g}` 接入当前定义的结构化 latent 和未训练 prior rollout，但尚无 Loss、优化、训练、校准或预测精度证据；planner 真实反馈也未实现。

## 单一科研下一步

唯一下一动作建议是研究者审阅后另行授权 **STEP 5.1B — Definition 05 Loss / Posterior / Metric Implementation**。不得自动执行，不启动 GPU，不访问 `locked_test`。

## 当前 Git

- Branch：`main`。
- Step 1 base commit：`829276241a0da72d3a5393946086daba40b0a0fe`。
- Latest commit：以 GitHub `main` 的 `HEAD` 为准；本文件不能稳定硬编码包含自身的 commit hash。读取时运行 `git rev-parse HEAD` 或查看 GitHub 分支头。

## 关键入口

- 当前方法与边界：`AI_CONTEXT/02_ARCHITECTURE.md`
- 新定义实现总表：`docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- Step 1 详细记录：`docs/implementation_records/STEP_01_AUDIT.md`
- 数据和张量：`AI_CONTEXT/03_DATA_FLOW.md`
- 问题到源码：`AI_CONTEXT/04_MODULE_MAP.md`
- 当前/历史实验：`AI_CONTEXT/05_EXPERIMENTS.md`
- 已确认决策：`AI_CONTEXT/06_DECISIONS.md`
- 冲突和阻塞：`AI_CONTEXT/07_KNOWN_ISSUES.md`
- 机器注册表：`docs/registries/`
- 原始状态权威：`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`

## 最近重要变化

- 2026-09-20：STEP 4.2A-PATCH 解耦 wireless structural relation validity 与 CSI observability；missing CSI 保留 relation 并使用 mask=false/zero placeholder。return size/priority/deadline 改为 observer available but frozen Raw not exposed；stateful Flow 仍未解决。
- 2026-09-20：STEP 4.2B source audit 证明 `transmitted_size` 为 hop-local stage progress，不能推出 end-to-end Flow remaining；DAG 只提供 gating，DepData transfer 未找到。综合 verdict=`FLOW_CONTRACT_NOT_YET_SUPPORTED`。
- 2026-09-20：STEP 4.2B-PATCH 将 verdict 改为 Flow-specific evidence 的实际计算；resource gaps 与 Flow readiness 解耦，stable ID/DepData 改为 implementation fact + researcher decision boundary；provenance 增加 symbol anchors。
- 2026-09-20：STEP 4.2C-A-PATCH 通过 audit-only replay 修正 verdict：logical destination 过滤 final delivery，E2E remaining 与 holder 可因果派生，same-destination reroute 可保持 Flow epoch；`flow_completed` 禁止作为 logical completion；机器 verdict=`CAUSAL_FLOW_LEDGER_FEASIBLE`。destination change/DepData 仍需研究者决定。
- 2026-09-20：STEP 4.2C-B 实现 FlowID/Epoch/RouteRevision、Flow/Carrying 分离、Input/Return lifecycle、clean-boundary destination change、lineage 与 Raw additive state；真实 non-locked trace覆盖 Input/Return，DepData runtime=0。随后 STEP 4.2C-C 完成 Sample/Tensor additive extension；在该 Step 当时 Graph Builder 尚未开始，之后已由 STEP 4.3A 完成。
- 2026-09-20：STEP 4.2C-B-PATCH 修正 logical destination provenance：Input 使用已成立 route terminal，Return 使用 `return_destination_id`；真实 `UAV_0→RSU_0→cloudServer_4` 两跳通过单 FlowID/Epoch、固定 destination、无重复 E2E 计数验收。
- 2026-09-20：STEP 4.2C-C-PATCH 将 normalization stats 收紧为 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`，并补齐 History/target Logical/Carrying 四组全字段 semantic equality、ID/provenance tamper 检查与 target carrying future-ground-truth namespace；23/23 focused、跨时隙真实 trace、deterministic/round-trip/receipt negative checks 通过。在该 Step 当时 Graph Builder 尚未开始；之后 STEP 4.3A/4.3B/4.4 已依次完成 Builder/Encoder/World Model Contract，训练仍未开始。

- 2026-09-19：STEP 4.1-PATCH 修正最小 gap 语义：wired relation 是 Raw/simulator 有来源但未暴露，无 CSI 时用 type + mask；wired 可选 numeric state 不阻塞 03 minimum；CPU capacity 是静态 capability，并与 allocation/service/available CPU 分离。在该 Step 当时 graph builder 保持关闭，之后已由 STEP 4.3A 完成。

- 2026-09-19：完成 Step 2.4 wireless/wired/total communication Outcome 语义最终验收；wired 服务来自真实 `WiredNetworkManager.step`，空 map 与 missing 分开，冻结 Raw Trajectory Layer / 01。
- 2026-09-19：完成 Step 2.2 真实 AirFogSim 6 步轨迹与独立下一 Decision 验收；补齐 Step 2.1/2.2 机器证据 Git 追溯。
- 2026-09-09：第二个正式 seed `20260830` 完成并通过单 seed 验收；第三 seed 暂停。
- 2026-09-09：建立项目文件、依赖、artifact、实验、结果、历史方法和问答路由索引。
- 2026-09-10：新增面向 ChatGPT 网页端的 `AI_CONTEXT/`，并将三方协作和同步规则写入 `AGENTS.md`。
- 2026-09-19：STEP 3.1 冻结最小 Model-ready Sample & Tensor Contract；`H=2/L=2` 真实样本、四类 action、input/target index 隔离和 mask 语义通过机器检查。正式 batch/split builder 未开始。
- 2026-09-19：STEP 3.1R 修正 History `[t-H+1,t]`、固定 index/presence、真实 DAG 接线、typed target namespaces 和 relation endpoints；v2 Raw 与最小样本证据已重建。
- 2026-09-19：STEP 3.1F 及其最小 PATCH 已冻结；History 保留 past Action/Outcome，input index 使用 History causal union，Future Action 先做 anchor visibility 检查再引用同一 static index，future-reference 观察审计 JSON 已纳入 provenance/Git。
- 2026-09-19：STEP 3.2 完成最小 non-locked batch/split/preprocessing validation；3 trajectories、12 windows、train-only mask-aware stats、deterministic rebuild 和 round-trip 已有 artifact/test 证据，不代表正式 Dataset。

Unverified：当前没有“最终 PI-JWM 方法已冻结”或“正式性能声明已开放”的证据。

## Freeze chain（2026-09-20 current）

- Raw Trajectory / 01、当前最小 Dataset/Tensor / 02、STEP 4.1 mapping、STEP 4.2A existing-source input extension、STEP 4.2C-B Raw Flow、STEP 4.2C-C Flow Sample/Tensor、STEP 4.3A Typed Dual-Graph Builder 与 STEP 4.3B Dual-Graph Encoder 均已冻结。
- Causal boundary: future task schedule is internal metadata only; canonical acceleration is backward speed difference with an explicit missing-history mask.
- Current boundary: Physical topology 的 `radius_knn/radius=1000m/k=2` 仅是 deterministic development config，`research_frozen=false`；Return multi-hop、same-destination reroute runtime 与 formal capacities 仍未冻结。
- Boundary: Graph Encoder 与 STEP 4.4 World Model Contract 已 COMPLETE / FROZEN；Definition 05 decisions FROZEN，但 implementation NOT STARTED。`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。
