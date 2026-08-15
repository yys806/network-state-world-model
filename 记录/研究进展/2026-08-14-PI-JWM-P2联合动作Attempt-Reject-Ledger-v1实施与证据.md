# PI-JWM P2联合动作Attempt/Reject Ledger v1实施与证据

日期：2026-08-14
范围：CPU-only P2-B v2 preflight与P2-C v2候选审计
状态：P2-B v2候选已形成并复验；P2-C v2预文档闭包已通过，最终candidate须在本文件提交后生成

## 1. 方法边界

本阶段实现的是采集执行链的审计基础设施，不是世界模型、策略器或基于世界模型候选rollout的规划器。一个attempt严格定义为一个时隙提交给执行链的一份完整联合动作候选；当前P2-B每个frame只允许`candidate_ordinal=0`。

P2-B v1代码和历史artifact保持不动。v2通过实例级scheduler代理、`env.step`包装和observer包装，记录真实setter、step和outcome调用；所有包装均在`finally`恢复。任何rejected attempt都会阻断成功bundle并产生独立`_failed`证据，不允许换候选重试，也不允许向旧bundle回填记录。

## 2. 代码与测试证据

实现路径：

- `代码/src/pi_jwm/action_attempt_ledger_v1.py`：attempt身份、状态机、setter明细、二元守恒与分角色汇总；
- `代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v2.py`：v1 executor外的实例级真实runtime观察；
- `代码/src/pi_jwm/full_dual_graph_artifact_v2.py`：九文件成功bundle与三文件失败bundle；
- `代码/scripts/run_p2_full_dual_graph_collector_preflight_v2.py`：reference/replay/bootstrap/fixture编排；
- `代码/src/pi_jwm/p2c_scale_distribution_audit_v2.py`与对应runner：从ledger独立重算拒绝率。

每个生产切片均先运行缺模块或缺接口的RED测试，再做最小GREEN。候选前，使用`airfogsim` Python运行24个P1/P2测试模块，实际展开203项，结果为203/203通过；P2-B v1和P2-C v1的`--verify-only`均通过；P2-B v1 manifest中的AirFogSim依赖重新计算为83/83匹配；编译与`git diff --check`通过。没有修改AirFogSim第三方源码。

## 3. P2-B v2真实候选

候选目录：

```text
代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814
```

冻结请求为3 seeds × 2 resource arms × 20 natural-reference frames。以下数字不是从请求常量推断，而是重新读取`action_attempts.jsonl`和`frames.jsonl`所得：

| run role | attempts | accepted | rejected | quarantined | mutation=confirmed |
|---|---:|---:|---:|---:|---:|
| natural_reference | 120 | 120 | 0 | 0 | 120 |
| natural_replay | 120 | 120 | 0 | 0 | 120 |
| bootstrap | 10 | 10 | 0 | 0 | 10 |
| fixture | 10 | 10 | 0 | 0 | 10 |

独立检查还得到：

- 总attempt为260，发布frame为120；
- natural-reference attempt与frame的`trajectory_id/frame_index`键差异为0；
- reference/replay对应键为120，candidate digest差异为0；
- 非零`candidate_ordinal`为0；
- `env_step_called=false`和`env_step_completed=false`均为0；
- setter明细总记录627次；
- `training_eligible=true`为0；
- 成功bundle为九个受管文件，artifact hash为8项，source hash为126项；
- runner发布后验证和独立`--verify-only`均为`passed=true`、0错误。

因此该候选支持“natural-reference联合动作拒绝率已由完整ledger观测为0/120=0.0”的候选审计输入，但不支持“正式数据完成”或“训练可启动”的表述。

## 4. P2-C v2预文档闭包结果

预文档候选目录为`代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v2_pre_document_closure_20260814/`。独立`--verify-only`与从P2-B v2输入重新计算均通过，且没有生成正式数据。机器审计记录如下：

- natural-reference为6个episode、120个发布frame、120/120 accepted、0 rejected、0 quarantined；自然参考拒绝率为`0/120=0.0`。
- natural-replay为6个episode、120次attempt、120/120 accepted，并与reference逐键对齐；bootstrap为10/10，fixture为10/10，四类角色合计260 attempts。
- 5个E1字段每行宽度固定为5，共5760行；字段有效计数分别为4392、4392、5472、5472、5472；无legacy 13/18槽占位值。
- 观测覆盖包括6个自然episode、3个seed（0/1/2）、两个resource arm、20步/episode、82条通信flow/hop、82次RB分配、9个RB复用frame和25行outage；自然数据仍全部`training_eligible=false`。
- manifest绑定9个受管artifact、8个artifact hash和126个source hash，独立复算无不匹配；replay 6/6通过；CPU规则版本120帧均为`PIJWM-CPU-Inner-Rule-v1`。

即使预文档审计通过，正式数据仍被以下三个冻结门阻断：

```text
scenario_matrix_not_frozen
formal_scale_not_frozen
formal_split_not_frozen
```

当前固定为false或未实施：`formal_data_approved`、`training_eligible`、GPU训练、locked-test访问、正式轨迹生成、世界模型候选rollout规划器完成、最终方法冻结。

本次提交只绑定上述可复核计数和阻断，不把预检候选升级为正式数据或训练资格。最终candidate目录将在本文件提交后以新目录生成，再进行字节和manifest比对。

## 5. 对已有实验的影响

P2-B v1的120个frame继续作为历史结构、五维E1、任务/RB/runtime接口preflight证据，但不能用于拒绝率声明。v2没有修改旧代码、旧artifact或模型checkpoint，因此不使已有checkpoint自动失效；它修正的是采集审计证据边界。任何后续正式数据和训练必须使用最终冻结的数据合同重新判断有效性，不能把本CPU候选直接称为正式数据集。
