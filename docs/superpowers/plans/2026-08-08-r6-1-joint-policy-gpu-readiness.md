# PI-JWM R6.1 联合策略 GPU 训练前实现计划

> 执行边界：本计划在当前 `codex/formal-airfogsim-dataset-v1` 分支内完成，因为 R1—R6 依赖资产仍包含尚未提交的正式文件；新建隔离 worktree 会丢失这些依赖。只增量修改本任务文件，不清理或回退用户现有改动。

## 目标

在 CPU 上完成 R6 正式联合动作、奖励、transition/GAE、候选策略和真实 AirFogSim 预检，写出可审计的 GPU 训练协议与 go/no-go。最终只允许进入 GPU 训练准备状态，不启动 GPU。

## Task 1：冻结机器可读协议与 train-only reward 尺度

**新增文件**

- `代码/tests/test_r6_reward_protocol.py`
- `代码/src/pi_jwm/r6_reward_protocol.py`
- `代码/scripts/run_r6_reward_scale_freeze.py`

**步骤**

1. 先写失败测试：拒绝 locked-test/validation/calibration 参与尺度估计、分位数必须为正、硬约束使 transition invalid、分量重算与总 reward 一致、次级总贡献不反转一次主事件。
2. 运行：

   `python -m unittest test_r6_reward_protocol.py`

   记录预期的模块缺失失败。
3. 实现 `TransitionFacts`、`RewardScale`、`RewardBreakdown`、`ServiceFirstRewardProtocol` 和只读系统目标 NPZ 尺度计算。
4. 再运行定向测试直到通过。
5. 从 `代码/artifacts/reports/airfogsim_formal_system_targets_v1` 的 36 条 train 轨迹生成版本化 `reward_scale.json`，并保存输入清单哈希和有效样本数。

## Task 2：实现完整联合动作候选、验证和 AirFogSim 适配

**新增文件**

- `代码/tests/test_r6_joint_action.py`
- `代码/tests/test_r6_airfogsim_joint_runtime.py`
- `代码/src/pi_jwm/r6_joint_action.py`
- `代码/src/pi_jwm/r6_airfogsim_joint_runtime.py`

**修改文件**

- `代码/scripts/run_pi_jwm_energy_reward_diagnostic.py`（仅改为复用新公共执行辅助，保持旧命令语义）

**步骤**

1. 先写纯契约失败测试：默认候选唯一、候选排序确定、task/node/RB绑定完整、DAG/邻接/RB/CPU/阶段故意破坏时 fail-fast、padding mask 与 fallback 正确。
2. 写 duck-typed AirFogSim 适配失败测试：严格执行“捕获→默认调度→卸载→RB→CPU→验证→step”顺序，默认 RB 为未选任务保留，CPU 按节点投影。
3. 运行两组测试并记录模块缺失失败。
4. 实现 `JointActionCandidate`、`JointActionCandidateSet`、`JointActionContext`、确定性五模板生成器、描述向量和协议哈希。
5. 实现只调用已核对公共接口的 runtime adapter；不修改 AirFogSim 内核。
6. 把旧诊断 runner 的相同动作执行辅助切到公共实现，防止两套语义漂移；运行旧相关测试。

## Task 3：实现候选集 Actor-Critic/PPO 与严格 transition/GAE

**新增文件**

- `代码/tests/test_r6_joint_policy.py`
- `代码/tests/test_r6_rollout.py`
- `代码/src/pi_jwm/r6_joint_policy.py`
- `代码/src/pi_jwm/r6_rollout.py`

**步骤**

1. 先写失败测试：候选 mask 后概率和为 1、全空 mask 回退候选0、非法候选不能 evaluate、explicit/latent/joint 三状态模式维度一致、Actor-Critic/PPO 有限反向。
2. 先写 GAE 手算测试：真实终止不 bootstrap、时间截断允许 bootstrap、不得跨 trajectory/seed/split 拼接、身份 slot 必须连续、reward ledger 可重算。
3. 实现共享候选 scorer、价值头和 masked categorical policy；复用已有 PPO clip/value/entropy 约定，不改旧 CPU-only policy。
4. 实现 `JointTransition`、`JointRollout`、`compute_gae()`、训练批 collator 和审计序列化。
5. 运行新增测试与已有 R6 learning-policy 测试。

## Task 4：真实非锁定 AirFogSim 多步闭环 CPU 预检

**新增文件**

- `代码/tests/test_r6_joint_policy_preflight.py`
- `代码/src/pi_jwm/r6_joint_policy_preflight.py`
- `代码/scripts/run_r6_joint_policy_gpu_readiness.py`

**步骤**

1. 先写失败测试：预检必须绑定 R1/R2/R5.1/R6 哈希、拒绝 locked-test、拒绝未来字段、必须出现卸载/RB/CPU 非 no-op 执行证据、世界模型哈希前后不变、报告必须声明 CPU-only。
2. 实现 preflight orchestrator：加载冻结 B checkpoint 和 validation 轨迹身份，使用实时显式观测与冻结/仅 prior 更新的隐式 belief；每一步生成同一完整候选集、选择、执行、读取事实、重建 reward 并写 transition。
3. 在 `airfogsim` conda 环境选择 `load_high__density_dense__r07`/seed 507 的非锁定窗口运行最小多步闭环；如某动作族在单段窗口无可用实体，继续同一 validation 轨迹到满足证据，不伪造动作。
4. 构造真实 GAE batch，分别完成 Actor-Critic 和 PPO 一次 CPU 数值更新；验证世界模型参数 SHA-256 未变化。
5. 生成 `代码/artifacts/preflight/pi_jwm_r6_joint_policy_gpu_readiness_v1/`，含协议、尺度、candidate/action/transition/reward ledger、GAE batch 摘要、失败记录、summary 和 manifest。

## Task 5：冻结 GPU 矩阵、停止门和审计器

**新增文件**

- `代码/tests/test_r6_gpu_training_protocol.py`
- `代码/src/pi_jwm/r6_gpu_training_protocol.py`

**步骤**

1. 先写失败测试：方法/状态/seed 矩阵必须完整 18 项，预算完全相同，validation 选模，calibration 只定阈值，locked-test 禁止，失败运行不可删除，smoke 不得混入正式结果。
2. 实现 dataclass/JSON 协议、运行 ID、checkpoint tie-break、early-stop 状态机和 manifest validator。
3. 把冻结协议写入预检 bundle；本阶段不实现或启动 GPU worker。

## Task 6：回归、独立验收和文档同步

**修改文件**

- `本地计划表.md`
- `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\PIJWM推进.md`
- `task_plan.md`
- `findings.md`
- `progress.md`

**新增文件**

- `文档/研究进展/2026-08-08-PI-JWM-R6联合策略GPU训练前预检结果.md`

**步骤**

1. 运行全部新增测试。
2. 运行已有 R6、R5/R4 关键回归及项目指定命令：

   `python -m unittest test_dual_graph_features.py test_v4_ablation_active_rate.py`

3. 在 `airfogsim` 环境复跑真实 preflight，独立脚本重算所有输出 SHA-256、检查 CSV/JSON 行数和汇总值。
4. 只在九项门全部通过时写 `r6_gpu_strategy_training_ready=true`；否则写具体 no-go，不冒充完成。
5. 同步单一计划表和权威主文档，明确已完成、未运行、结果边界和下一条 GPU 命令（只记录，不执行）。

