# PI-JWM Current State Snapshot

更新时间：2026-09-20

这是 ChatGPT 网页端进入仓库后的第一读取入口。它只提供当前快照和继续查证的路径，不替代源码、配置、checkpoint、metrics、manifest 或 audit。

## Source of truth

事实优先级固定为：当前源码/config/experiment → `AI_CONTEXT/` → 普通项目文档 → 历史聊天或推断。任何正式科研结论都必须回到第一层验证。

## 当前状态

- 项目：PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）。AirFogSim 只是参考仿真器和数据生成工具。
- 当前 active workflow：研究者最新只读 `00–06` 定义链；工程执行入口为 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`。
- 当前 Step：`STEP 4.2B` source audit 已完成，但 stateful Flow contract 尚未支持。定义 03 的 graph builder、GNN 与模型仍未开始。机器证据位于 `code/artifacts/protocols/pi_jwm_step4_2b_stateful_flow_source_audit_v1_20260920/`。
- `STEP 3.2-PATCH` 已补齐 Dataset isolation provenance、time-grid、development future-reference audit 与 normalization units；仍是 observation-only/non-locked evidence，不是正式 Dataset。
- `STEP 3.2-PATCH-RECEIPT` 已修正顶层 acceptance AND、显式 scope checks 和 frozen sample schema reuse；STEP 3.2 现正式 COMPLETE / FROZEN。
- 新定义实现状态：Raw 因果可观测、四类动作、slot Outcome 已验收；STEP 3.2/3.3 完成最小 Dataset/Tensor 合同；STEP 4.2A 已把 position/CSI/wired relation/CPU static capability/部分 Task 状态和 typed Task-Agent relation 输入化。Wireless structural relation 不因 CSI missing 而失效。source audit 确认 `transmitted_size` 是 hop-local 且完成 hop 后 reset；Input/Return stable Flow remaining、DepData transfer 和动态资源 Raw 仍不足。双图、World Model、Loss、Planner 仍未开始。
- 审计结论：时间因果、稳定 ID/index、mask/split 和部分规则/评价工具可复用；严格双图、四类动作、目标 RSSM 边界、逐步规则反馈和完整 planner 闭环需要结构性修改或新增实现。
- 当前运行：没有正式 GPU 训练或远端同步任务；旧 `seed=20260832` 仍不得自动启动。
- `locked_test_accessed=false`；`formal_performance_claim_ready=false`；本 Step `training=false`、`gpu=false`。

## 当前最重要问题

当前实现不能按模块名称或旧测试推断为符合新定义。STEP 4.1 已明确旧 `physical_edge` 混合 CSI/rate/task/RB，且 current Tensor/Raw 尚不能支持最小新图；四类动作张量已冻结，但尚未接入新 graph/model。旧随机状态范围、逐步规则反馈、动态图和 planner 真实反馈仍未实现。

## 单一科研下一步

STEP 4.2B source audit 已完成；唯一下一步建议是研究者审阅后授权最小 Raw additive source contract。不要自动实现 Graph Builder，不启动 GPU，不访问 `locked_test`。

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

- 2026-09-19：STEP 4.1-PATCH 修正最小 gap 语义：wired relation 是 Raw/simulator 有来源但未暴露，无 CSI 时用 type + mask；wired 可选 numeric state 不阻塞 03 minimum；CPU capacity 是静态 capability，并与 allocation/service/available CPU 分离。graph builder 保持关闭。

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

## 2026-09-19 Current freeze chain

- Raw Trajectory / 01 与当前最小 Dataset/Tensor / 02 均已冻结；STEP 4.1 mapping 已冻结，STEP 4.2A existing-source input extension 已完成。
- Causal boundary: future task schedule is internal metadata only; canonical acceleration is backward speed difference with an explicit missing-history mask.
- Current blocker: remaining Raw source audit 与 stateful-Flow contract gap；可选 wired numeric state 不是 minimum blocker；graph builder 未授权、未开始。
- Boundary: no graph/model/planner/training, no GPU, no `locked_test`.
