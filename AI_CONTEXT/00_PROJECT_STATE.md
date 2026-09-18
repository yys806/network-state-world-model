# PI-JWM Current State Snapshot

更新时间：2026-09-19

这是 ChatGPT 网页端进入仓库后的第一读取入口。它只提供当前快照和继续查证的路径，不替代源码、配置、checkpoint、metrics、manifest 或 audit。

## Source of truth

事实优先级固定为：当前源码/config/experiment → `AI_CONTEXT/` → 普通项目文档 → 历史聊天或推断。任何正式科研结论都必须回到第一层验证。

## 当前状态

- 项目：PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）。AirFogSim 只是参考仿真器和数据生成工具。
- 当前 active workflow：研究者最新只读 `00–06` 定义链；工程执行入口为 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`。
- 当前 Step：`STEP 2.1` 已完成真实 AirFogSim 非 locked 单轨迹四类动作与 Outcome/next Decision 对齐验收；证据位于 `code/artifacts/protocols/pi_jwm_raw_single_decision_step_real_airfogsim_v2_20260919/`。
- 新定义实现状态：Raw single-step interface verified only. Dataset/Tensor、双图、World Model、Loss、Planner 仍未开始。
- 审计结论：时间因果、稳定 ID/index、mask/split 和部分规则/评价工具可复用；严格双图、四类动作、目标 RSSM 边界、逐步规则反馈和完整 planner 闭环需要结构性修改或新增实现。
- 当前运行：没有正式 GPU 训练或远端同步任务；旧 `seed=20260832` 仍不得自动启动。
- `locked_test_accessed=false`；`formal_performance_claim_ready=false`；本 Step 未训练、无 GPU。

## 当前最重要问题

当前实现不能按模块名称或旧测试推断为符合新定义。关键差异包括：通信状态混在旧 `physical_edge`、独立 Agent/Communication/Task-Agent 合同缺失、四类动作未闭合、旧随机状态范围与新定义不同、RSSM 修正未形成逐步规则反馈和动态图重构、planner 没有真实反馈闭环。

## 单一科研下一步

研究者先审阅 Step 1；若明确授权，再执行唯一建议的 Step 2：冻结一个决策步的原始轨迹字段与 Route/Comm/Comp/UAV 四类动作映射合同。未授权时不进入 Step 2，不启动 GPU，不访问 `locked_test`。

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

- 2026-09-19：完成 Step 2.1 真实 AirFogSim 单轨迹四类动作与 Outcome/next Decision 对齐；修正 vehicle degree/UAV rad heading 语义。
- 2026-09-09：第二个正式 seed `20260830` 完成并通过单 seed 验收；第三 seed 暂停。
- 2026-09-09：建立项目文件、依赖、artifact、实验、结果、历史方法和问答路由索引。
- 2026-09-10：新增面向 ChatGPT 网页端的 `AI_CONTEXT/`，并将三方协作和同步规则写入 `AGENTS.md`。

Unverified：当前没有“最终 PI-JWM 方法已冻结”或“正式性能声明已开放”的证据。

## 2026-09-18 STEP 2

- Current gate: Step 2 contract frozen; minimum four-family scheduler closure passed.
- Action boundary: Route, Comm, Comp, and UAV-only Mobility. SUMO advances vehicles.
- Evidence: `code/artifacts/protocols/pi_jwm_raw_single_decision_step_contract_v1_20260918/`.
- Limitation: full real AirFogSim four-family trajectory has not been accepted; no GPU, training, or locked_test.
- Next single action: researcher review before Step 3 real single-trajectory wiring.
