# PI-JWM Current State Snapshot

更新时间：2026-09-10

这是 ChatGPT 网页端进入仓库后的第一读取入口。它只提供当前快照和继续查证的路径，不替代源码、配置、checkpoint、metrics、manifest 或 audit。

## Source of truth

事实优先级固定为：当前源码/config/experiment → `AI_CONTEXT/` → 普通项目文档 → 历史聊天或推断。任何正式科研结论都必须回到第一层验证。

## 当前状态

- 项目：PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）。AirFogSim 只是参考仿真器和数据生成工具。
- 当前粗粒度阶段：P4 世界模型非锁定数据验收，尚未关闭。
- 当前候选模型：`entity_aligned_dual_graph_rssm_v1`，即实体对齐的双图 RSSM。RSSM 是带确定性记忆和随机状态的时序模型。
- 已完成正式 seed：`20260831`、`20260830`，两者分别通过冻结的单 seed 数值门。
- 未完成正式 seed：`20260832`，状态为 deferred；`authorization_required=true`、`auto_start=false`。
- 当前运行：没有正式 GPU 训练或远端同步任务。
- P6 候选动作 rollout 规划器：只有 CPU 机制原型，正式阶段未开放。
- `locked_test_accessed=false`；`formal_performance_claim_ready=false`。
- 当前知识快照：844 个项目文件、607 个 Python 节点、802 个 artifact 目录；9 个 AI_CONTEXT 文件检查通过，artifact 控制入口读取错误为 0。

## 当前最重要问题

两个 seed 的通过不能代替预注册的三 seed 证据。缺少 `20260832` 及三 seed 独立汇总审计，因此不能关闭 P4、开放 P6 或发布正式性能结论。

## 单一科研下一步

等待研究者明确决定是否按冻结协议运行 `seed=20260832`。未授权时不得自动启动 GPU、远端同步或 `locked_test`。

## 当前 Git

- Branch：`main`。
- Snapshot base commit：`e382d79ae0048dc16ede8304eb1a8cbff5f1d25e`。
- Latest commit：以 GitHub `main` 的 `HEAD` 为准；本文件不能稳定硬编码包含自身的 commit hash。读取时运行 `git rev-parse HEAD` 或查看 GitHub 分支头。

## 关键入口

- 当前方法与边界：`AI_CONTEXT/02_ARCHITECTURE.md`
- 数据和张量：`AI_CONTEXT/03_DATA_FLOW.md`
- 问题到源码：`AI_CONTEXT/04_MODULE_MAP.md`
- 当前/历史实验：`AI_CONTEXT/05_EXPERIMENTS.md`
- 已确认决策：`AI_CONTEXT/06_DECISIONS.md`
- 冲突和阻塞：`AI_CONTEXT/07_KNOWN_ISSUES.md`
- 机器注册表：`docs/registries/`
- 原始状态权威：`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`

## 最近重要变化

- 2026-09-09：第二个正式 seed `20260830` 完成并通过单 seed 验收；第三 seed 暂停。
- 2026-09-09：建立项目文件、依赖、artifact、实验、结果、历史方法和问答路由索引。
- 2026-09-10：新增面向 ChatGPT 网页端的 `AI_CONTEXT/`，并将三方协作和同步规则写入 `AGENTS.md`。

Unverified：当前没有“最终 PI-JWM 方法已冻结”或“正式性能声明已开放”的证据。
