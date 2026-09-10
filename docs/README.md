# PI-JWM 杂项文档

本目录承接不属于论文、文献、组会或项目事实记录的材料：

- `templates/`：通用LaTeX、IEEE和工具模板；
- `project-notes/`：项目说明、边界说明和导航性资料；
- `archive/`：为保持可回滚性保留的旧导航原字节；
- `migration/`：如需长期维护的目录治理说明。

正式文献PDF只放在根级 `literature/`，组会资料只放在 `meeting/`。

## 项目知识入口

ChatGPT 网页端的首入口是根目录 [`../AI_CONTEXT/00_PROJECT_STATE.md`](../AI_CONTEXT/00_PROJECT_STATE.md)。`AI_CONTEXT/` 用少量上下文恢复全貌并指向真实源码；本目录继续承担详细的人类导航和机器注册表说明。

- [`PROJECT_INDEX.md`](PROJECT_INDEX.md)：项目地图和最短检索路径；
- [`LEARNING_PATH.md`](LEARNING_PATH.md)：用户逐步重新掌握研究问题、数据、模型和实验的路径；
- [`COLLABORATION_GUIDE.md`](COLLABORATION_GUIDE.md)：用户与 AI 的职责、独立判断、解释顺序和证据表达规则；
- [`RESTRUCTURE_ACCEPTANCE.md`](RESTRUCTURE_ACCEPTANCE.md)：原始重构目标到当前实现、测试和剩余安全限制的逐项验收；
- [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md)：AI 回答项目问题时的固定检索顺序；
- [`CODE_INDEX.md`](CODE_INDEX.md)：当前正式、支撑、原型和历史兼容代码的边界；
- [`ARCHITECTURE.md`](ARCHITECTURE.md)：当前代码和数据架构；
- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：当前科研问题、证据、阻塞和边界；
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)：实验目的、入口、结果和状态；
- [`RESULTS_INDEX.md`](RESULTS_INDEX.md)：结果到实验、配置、代码和数据的追踪；
- [`KNOWN_CONFLICTS.md`](KNOWN_CONFLICTS.md)：尚未解决的理论、测试和文档冲突；
- [`ARCHIVE_CANDIDATES.md`](ARCHIVE_CANDIDATES.md)：逻辑归档范围和物理迁移停止门；
- [`registries/`](registries/)：文件、依赖、实验、结果、历史方法、问题路由和延后任务的机器注册表；
- [`PROJECT_RESTRUCTURE_PLAN.md`](PROJECT_RESTRUCTURE_PLAN.md)：本次重构的保护边界和实施阶段。

这些文件只负责导航和状态摘要，精确判断必须回到代码、配置、原始产物和机器可读验收文件。
