# PI-JWM 项目重构目标验收表

本表把用户提出的长期科研协作要求逐项连接到当前实现。验收标准优先采用用户确认的实际目标：以后协作更顺畅，提问时更容易找到正确文件和证据；目录外观整齐不是单独验收标准。

## 1. 目标覆盖

| 用户目标 | 当前实现 | 可验证证据 | 状态 |
| --- | --- | --- | --- |
| 用户逐步重新掌握项目 | 分层学习路径，按研究问题、数据、模型、实验、决策逐步进入 | `LEARNING_PATH.md` | 已建立 |
| 用户掌握最终科研决策 | 永久规则明确研究问题、假设、方法和最终解释由用户决定 | `AGENTS.md`、`COLLABORATION_GUIDE.md` | 已固化 |
| AI 独立判断而非迎合 | 固定检查前提、反例、混杂因素、简单解释、可证伪实验 | 同上 | 已固化 |
| 通俗解释复杂内容 | 固定“问题—直觉—必要性—机制—数学—代码—实验”顺序 | 同上 | 已固化 |
| 避免科研空话 | 结论必须说明对象、条件、机制、指标、基线和边界 | 同上 | 已固化 |
| 当前代码容易定位 | 当前、支撑、原型、历史生命周期分层，并生成依赖和测试映射 | `CODE_INDEX.md`、`python_dependency_map.json` | 已建立 |
| 历史内容可追踪 | 重要旧方法记录动机、结果、弃用原因、替代关系和证据；其余 artifact 可按精确名称查询 | `historical_method_registry.json`、`artifact_catalog.csv` | 已建立语义入口 |
| AI 不再默认全仓扫描 | 固定索引优先顺序和只读问题查询命令 | `RETRIEVAL_GUIDE.md`、`query_project_knowledge_v1.py` | 已建立 |
| 稳定项目知识入口 | 项目、架构、科研状态、代码、实验、结果、变更和协作入口 | `PROJECT_INDEX.md`、`docs/README.md` | 已建立 |
| ChatGPT 新对话快速恢复 | 九个精简且可追溯的 AI_CONTEXT 文件，首入口为当前状态快照 | `AI_CONTEXT/00_PROJECT_STATE.md` 至 `08_CHANGELOG.md` | 已建立 |
| 三方长期协作 | 研究者决策、ChatGPT 理论讨论、Codex 工程执行与 Git 同步分工 | `AGENTS.md`、`COLLABORATION_GUIDE.md`、`AI_CONTEXT/06_DECISIONS.md` | 已固化 |
| 重要实验完整登记 | 统一 v2 schema；未知或不适用字段必须 `null + field_notes` | `experiment_registry.json`、契约测试 | 已建立 |
| 结果双向追踪 | 结果连接实验、配置、代码身份、数据、checkpoint 和 audit | `results_registry.json`、`experiment_registry.json` | 已建立 |
| 防止数字和文档漂移 | `--check` 自动比较正式结果和原始 acceptance 的哈希、seed、epoch、9 项指标与边界 | `build_project_knowledge_index_v1.py` | 已建立 |
| 当前科研状态清楚 | 当前问题、方法、证据、未知、阻塞和下一动作集中呈现 | `RESEARCH_STATUS.md` | 已建立 |
| 持续维护而非一次生成 | 永久规则要求重要变化同步更新索引并运行生成和检查 | `AGENTS.md`、`CHANGELOG.md` | 已固化 |
| 索引导航、原始证据验证 | 查询输出始终标记 `verification_required=true` | 查询工具与测试 | 已固化 |
| 第三个 seed 和同步接口预留 | 复用既有入口，显式禁止自动启动 | `deferred_work.json` | 已建立 |

## 2. 可观察的问答验收

以下问题已纳入自动测试：

- “现在使用的模型和训练入口在哪里？”
- “我们现在模型是啥？”
- “之前的完整 RSSM 为什么不用了？”
- “seed 20260831 的指标结果从哪里来？”
- “第三个 seed 什么时候运行？”
- 给出一个具体旧实验目录名时，能否定位到它？
- 给出一个具体代码文件名时，能否定位到它？

查询结果只负责指出入口。回答科研结论时仍必须打开返回的代码、配置、metrics、manifest 或 audit。

## 3. 当前安全限制

- 211 个历史 Python 候选已逻辑归档，但没有物理移动。94 个仍被历史脚本或测试引用，强行迁移会损害旧实验复现。
- 全量测试仍受 AirFogSim `traci` 环境、历史 fixture 和历史 artifact 权限问题影响；这不应通过放宽合同或删除测试来掩盖。
- 2026-09-09 受限环境曾有 14 个历史 artifact 控制文件不可读；2026-09-10 当前权限下重建 catalog 后读取错误为 0。权限变化不改变实验状态。
- seed `20260832`、远端同步、P6、正式性能声明和 `locked_test` 均保持关闭。

这些限制不会阻止当前目标——快速定位与证据核实——但会阻止声称“所有历史文件都已安全物理迁移”或“全仓回归全部通过”。

## 4. 维护验收

2026-09-10 当前验收结果：生成器与 `--check` 通过，AI_CONTEXT 6/6、项目知识/结构 28/28、正式 P4 210/210、compileall 和 `git diff --check` 通过；全量 1643 项为 0 failure/17 个已登记环境或历史错误。正式结果证据核对为 2 项通过、0 项不一致。

重要信息变化后执行：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
$env:PYTHONUTF8='1'
python .\code\scripts\build_project_knowledge_index_v1.py
python .\code\scripts\build_project_knowledge_index_v1.py --check
python -m unittest discover -s .\code\tests -p 'test_*project_knowledge*.py'
```

任何 schema 缺失、证据路径丢失、正式数字变化、audit 哈希变化、封存边界改变或问答路由失效都会使对应检查失败。
