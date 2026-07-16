# PI-JWM v11 Selector 定型执行计划

## 目标

在冻结 PI-JWM world model 的前提下，完成候选生成、actual-rollout 标签、benefit/risk selector、不确定性回退、独立测试闭环；以 matched test seeds 18–19 active-rate RMSE `<200` 作为 A 级突破标准。

## 固定边界

- 不使用 test seeds 18–19 选方法、超参数或 defer 阈值。
- 不把 AirFogSim future/counterfactual outcome、future truth 或 seed id 用作测试时特征。
- 候选库 validation sample-oracle 未达到 `<190` 前不训练主 selector。
- world model 保持冻结；本轮仍命名为 `v11 candidate`，只有达到验收条件才冻结 selector 结构。
- 当前工作区直接实施，不创建 worktree；先独立验收并提交上周未提交工作。

## 阶段

1. **上周成果冻结与基线复核** — `complete`
   - 审计当前未提交能耗/reward/horizon 改动。
   - 运行相关单测、脚本测试与全量测试。
   - 核对报告和研究文档口径后提交独立 commit。
   - 复现/校验 235.423、210.975、213.160874 三个基线来源。
2. **协议与防泄漏基础设施** — `complete`
   - 固定 selector split、sample-time 映射、test lock 和特征协议。
   - 测试先行实现 `CandidateBatch`、`CandidateOutcome`、`SelectorDecision` 与审计接口。
3. **候选库与 actual-rollout 标签** — `in_progress`
   - 实现支持约束的 edge-step value repair、阶段耦合候选和动作投影。
   - 完成 64/256 sample CPU smoke。
   - 生成 train/calibration/validation 标签并执行 oracle gate（软件链完成，正式 GPU 标签待服务器恢复）。
4. **Selector 训练与 calibration** — `in_progress`
   - 训练 RF/GB、pointwise benefit 和 CandidateSetBenefitRanker。
   - calibration 确定 defer；validation 锁定唯一结构与 ensemble。
   - 完成 feature-group ablation。
5. **一次性测试与外部验证** — `pending`
   - 配置冻结后仅打开一次 seeds 18–19。
   - 生成/评估 external holdout seeds 60–69。
   - 逐 seed/stage/family/coupling 审计。
6. **结果冻结与交付** — `pending`
   - 生成 CSV、图、summary、命令、checkpoint 与 SHA-256 manifest。
   - 更新本地计划表和研究进展文档。
   - 全量验证、代码审查、最终 commit。

## 错误记录

| 时间 | 错误 | 尝试 | 处理 |
|---|---|---:|---|
| 2026-07-16 | 并行执行多个 `conda run` 时争用同一个 `__conda_tmp` 临时文件，命令内部失败但外层返回码为 0 | 1 | 后续测试改用 `D:\miniconda\envs\airfogsim\python.exe` 直接运行，并检查测试输出而非只信任 conda 外层返回码 |
| 2026-07-16 | `airfogsim` 环境缺少 Torch，混合排序诊断测试 4 项导入失败；其余 24 项通过 | 1 | 按职责拆分：主项目 Python 运行模型/排序测试，AirFogSim 环境只运行参考模拟器 smoke 与能耗接口测试 |
| 2026-07-16 | 从仓库根目录以包名运行脚本测试时，74 项均因 `代码/scripts` 未进入 `sys.path` 而导入失败 | 1 | 按 AGENTS.md 约定切换到 `代码/scripts` 后运行，74 项全部通过；不是实现缺陷 |
| 2026-07-16 | Python `-c` 参数中直接包含中文相对路径时被终端编码为乱码，导致文件查找失败 | 1 | 将工作目录切到 `代码/` 后只向 Python 传 ASCII 相对路径；后续数据脚本采用 Path 参数/配置文件，不把中文路径硬编码进 shell 字符串 |
| 2026-07-16 | 4-sample integration smoke 恰好没有 active target，candidate gate 抛异常 | 1 | 新增失败测试后把零 active 组改为保留结果并返回 `passed=false, failure_reason=no_active_targets`；不再静默删除或中止缓存 |
| 2026-07-16 | schema-v4 的 8-sample calibration 恰好没有 active target，基础 selector 阈值评估把 `None` 强转为 float | 1 | 新增失败测试；无 active 的 smoke calibration 明确记为 unscored 并强制保守 defer，正式协议仍要求有效 calibration 标签 |
| 2026-07-16 | `yuyaoshen_VM` SSH 连接 `192.168.89.132:22` 超时 | 1 | 停止重复连接；本地完成代码、测试和 smoke，正式 GPU full-label/selector 训练等待服务器或 VPN 恢复 |
| 2026-07-16 | 首次 RTX 4090 full validation gate 的 action-applied 为 99.9841%，其余 oracle/coverage/identity 三项均过线 | 1 | 定位为 25 个候选在构造时就与 baseline 完全相同却被预标 applicable；新增失败测试，applicability 现在同时要求 raw candidate 确实请求变化，投影/执行后 no-op 仍按失败计数；不放宽 100% gate |
