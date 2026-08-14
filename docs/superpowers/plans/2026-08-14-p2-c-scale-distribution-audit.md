# P2-C 规模与分布审计实施计划

> 本计划只针对已通过的 P2-B CPU 非训练 canonical preflight 做审计；不生成正式训练数据、不启动 GPU、不访问 locked test。

## 目标与不可违反边界

- 以 `代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v1/` 的八个机器产物为输入，独立核验当前自然轨迹规模、场景/节点/无线/有线边、任务生命周期、DAG、flow/hop、RB 复用、失败/outage、E1 五维 Mask/MissingReason、fixture 分离、拒收/隔离、split 与 manifest 闭包。
- P2-B 的 6 条自然 episode、10 个覆盖夹具和 120 帧只作为“已观测事实”，不升级为正式数据集完成或训练资格。
- 没有产物支持的维度、分布或目标数不得通过常量、代理量、旧 v3 轨迹或 Mask 伪造；应报告为 `not_observed`、`candidate_target_definition` 或阻断项。
- 任何审计失败都阻止正式 v4 数据生成；不修改 AirFogSim 第三方源码，不清理主工作树。

## 文件与接口

- 新增纯 CPU 审计库：`代码/src/pi_jwm/p2c_scale_distribution_audit_v1.py`。
- 新增测试：`代码/tests/test_p2c_scale_distribution_audit_v1.py`。
- 新增短脚本：`代码/scripts/run_p2c_scale_distribution_audit_v1.py`，只读 canonical bundle 并写入显式指定的 audit 输出目录。
- 机器输出（均置于 `代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v1/`，不作为训练数据）：
  - `p2c_scale_distribution_audit_v1.json`：观测统计、阻断项、拒收/隔离和证据引用；
  - `p2c_formal_data_config_candidate_v1.json`：候选规模/分层/split/拒收上限定义，明确 `formal_data_approved=false`；
  - `manifest.json`：审计输入与代码/测试依赖哈希。

## 执行任务（测试先行）

1. **基线读取与 schema inventory**（只读）
   - 读取 P2-B 设计、collector/artifact 代码和八个产物；用独立脚本复算 episode/frame/fixture/字段统计。
   - 验收：所有统计均能定位到具体 JSON 字段或代码路径；若字段不存在，报告缺失而不猜测。

2. **先写失败测试：审计契约**
   - 覆盖自然/fixture 分离、配置与 episode 笛卡尔积、每 episode 步数、split 不重叠且包含 rejected/quarantine 限制、E1 恰为五字段且逐字段 valid_mask/missing_reason 计数、禁止 13/18 占位、双图/任务/DAG/flow/hop/RB/失败覆盖统计和 manifest source closure。
   - 构造最小临时 bundle，证明缺失字段、fixture 混入自然统计、split 重叠或伪造 13 维时审计失败。
   - 先运行测试并确认 RED，再写实现。

3. **实现确定性审计库**
   - 只读取 JSON/JSONL；不调用 AirFogSim、不改变输入、不运行训练。
   - 输出 `observed_facts`、`coverage`、`e1_field_validity`、`split_policy`、`rejection_policy`、`evidence_gates` 和稳定排序的 `blocking_reasons`。
   - 对自然分布和 fixture 覆盖分别计数；对任务 lifecycle、DAG depth、multi-flow、wireless/wired/local、RB reuse/outage 只报告真实观测及 `not_observed`。
   - 对当前 P2-B 只生成候选 formal config；候选值必须带来源/状态，未完成目标不能把 `formal_data_approved` 置真。

4. **实现 CLI 与 artifact manifest**
   - CLI 默认拒绝将输出写回输入 bundle；要求显式 `--bundle` 和 `--output-dir`。
   - 审计 manifest 记录输入八文件哈希及审计库、脚本、测试、P2-B 设计依赖；不包含绝对路径或临时 worktree 键。

5. **验证与交接**
   - 运行新测试、脚本 `--verify-only`/审计写出、独立 JSON 解析与哈希复算，再运行与本变更相关的既有 focused tests。
   - 不启动 GPU、不访问 locked test；只有在 P2-C 阻断项清零且用户确认后，才进入正式 v4 数据生成设计的下一串行门。

## 验收标准

- 新增测试全部通过，且能在最小伪造 bundle 上拒绝理论—实现不一致。
- 对 P2-B canonical bundle 生成机器可读审计；自然统计为 6 episode/120 frame，fixture 不进入自然分布，E1 只含五个命名字段且无 13 维补齐证据，当前状态仍 `training_eligible=false`、`formal_data_approved=false`。
- 报告明确当前缺口（如真实正式 split、规模目标和自然稀有事件覆盖）以及下一步阻断条件；不将候选配置或 preflight 写成最终数据集/方法。
- 审计输出可由 manifest 独立重算，输入和源依赖哈希无不匹配。
