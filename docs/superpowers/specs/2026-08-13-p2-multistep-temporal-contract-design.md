# PI-JWM P2 多步时序与跨步词表契约设计

**日期：** 2026-08-13
**状态：** 方案已获用户确认，待书面复核后实施
**范围：** 修订 P2 单步信道采样时序证据，并完成小规模 CPU 非训练多步轨迹 smoke；不生成正式 v4 数据集，不训练模型，不访问 locked test，不冻结最终方法。

## 1. 已证实冲突与影响边界

当前单步 runner 在 `AirFogSimEnv.step()` 内部执行 `_updateTraffics()`、`updateFastFading()` 和 `computeRate()` 之后，才由 `_event_from_profile()` 读取 `attenuation_db`。但正式单步 bundle 又把这个值写入 `pre_link.channel_attenuation_*` 和 `pre_rb_optional.channel_attenuation_db`，并称为当前动作前观测。

这两个口径不能同时为真。必须撤回现有 bundle 对“action-pre attenuation 已验证”的证明力。

以下事实不受该冲突影响：候选动作通过真实 scheduler setter 写入；AirFogSim 真实执行通信、CPU 和能耗更新；两个候选的卸载目标差异真实进入对应计算集合；CPU 守恒和能耗方程来自实际事件与状态。旧 bundle 不删除，归档为带已知时序缺陷的历史证据。

## 2. 方案比较与冻结选择

### 方案 A：动作前观测与动作后结果分离（冻结方案）

在调用任何 offload/RB setter 前，从当前 `channel_manager` 状态读取 decision-time CSI。动作写入并执行 `env.step()` 后，再读取本槽更新快衰落和速率得到的 outcome-side 信道矩阵。两者使用不同字段、不同采样阶段和不同用途。

优点是保留 v4 对无未来泄漏的定义，不改变 AirFogSim 的原生执行顺序。限制是 decision-time CSI 的准确含义为“决策时最新可用的 channel-manager 状态”，不是尚未发生的本槽交通/快衰落结果。

### 方案 B：把全部衰减改成上一槽结果

实现更简单，但会改变 E1/E3 已冻结定义，并使当前可观测信道与历史结果混在一起。暂不采用。

### 方案 C：拆分 `env.step()`，先更新交通/快衰落再决策

这会改变 AirFogSim 原生 MDP 时序和与既有轨迹的可比性。除非未来重新定义环境交互协议并全面重建证据，否则禁止采用。

## 3. 冻结时序

每个轨迹帧严格按以下顺序形成：

1. `decision_snapshot_started`：记录环境时间、节点/任务状态、Python/NumPy RNG 状态哈希。
2. `decision_time_csi_read`：在任何候选动作 setter 前，按稳定端点和 RB 身份读取最新可用 CSI 衰减。
3. `action_validated`：验证 offload、RB COO、词表索引、重复和越界。
4. `action_setters_called`：依次写入 offload 和 RB 动作。
5. `env_step_started`：只调用一次真实 `env.step()`。
6. AirFogSim 原生执行交通更新、无线通信、有线通信、计算、存储、能耗和时间推进。
7. `outcome_channel_captured`：在 `updateFastFading()` 与 `computeRate()` 之后记录当前槽 attenuation/rate/SINR/interference/outage direct event；这些值只进入当前结果或下一帧历史，不进入同槽决策输入。
8. `frame_validated`：验证 link/RB、任务、CPU、能耗和 Mask/缺失原因。
9. `history_committed`：只有本帧全部通过后，当前结果才成为下一帧 E1/E2 历史源。

必须为每次读数记录 `capture_phase`、`simulation_time`、端点、RB 身份和来源方法。测试须证明 `decision_time_csi_read` 发生在第一个 setter 前，不能只比较两个浮点数组是否不同。

## 4. 字段语义修订

- `pre_link.channel_attenuation_mean_db/std_db`：由 setter 前同一组 decision-time per-RB CSI 直接聚合，`valid=true` 仅在端点、RB 和直接来源完整时成立。
- `pre_rb_optional.channel_attenuation_db`：同一个 setter 前 CSI 快照，不允许使用 transfer event 中的本槽更新值替代。
- transfer event 中的 `attenuation_db`：重命名为机器语义明确的 outcome-side channel snapshot，保留 direct runtime 来源；它可作为监督诊断或下一帧历史候选，禁止声明为同槽 action-pre。
- `pre_link.prev_active_flow_count/effective_rate_per_s/served_data`：只从上一成功提交帧的完整结果 ledger 回填。
- 第一帧没有上一帧，三项历史值为零、`valid=false`、`missing_reason=no_history`。
- 已成功执行且完整观测到“无激活传输”的上一帧，下一帧三项历史值为零、`valid=true`、`missing_reason=none`。合法零值不能伪装成缺失。
- 若上一帧执行或校验失败，不得部分提交历史；整条轨迹失败，不能把失败帧写成 `source_absent` 后继续。

## 5. 跨步身份词表

新建纯契约层维护 append-only 词表：

```text
node_id -> node_index
information_edge_id -> information_edge_index
flow_id/task_id -> flow_index
```

冻结规则：

- 首次出现按输入记录的稳定排序加入；后续新身份只追加，已有索引永不重排或复用。
- 节点暂时消失只改变 presence，不删除其身份；重新出现时必须复用原索引。
- 信息边身份至少绑定 `source_id`、`target_id` 和明确的 channel/edge class；同一 ID 的端点或类别变化立即失败。
- flow 身份绑定原始 `task_id`；任务完成后保留身份，不能把索引分配给新任务。
- action COO、当前 outcome 和下一帧历史必须通过同一词表索引对齐。
- 单元测试覆盖新增、消失、重现、身份冲突、重复输入和非法悬空端点。

本 smoke 的 `information_edge_vocabulary` 只证明已观测/已执行通信边的跨步身份，不证明正式 v4 全拓扑 `structure.edge_present`、完整物理图、完整信息图或 CEP 已实现。

## 6. 固定真实轨迹 fixture

CPU smoke 使用预先冻结的单条三帧轨迹，不根据运行结果临时选择动作：

- 使用非 locked 的 seed 0 和现有 P2 preflight 配置；暖机到第一个可卸载任务。
- 目标固定为决策时距离排序的第一个无线远端节点；排序只依赖动作前节点状态和稳定 ID。
- 第 0 帧写入一次远端 offload，并分配全部合法 RB，以覆盖真实通信、任务迁移和候选后 CPU 路径。
- 第 1、2 帧不写入新的 offload/RB 动作，但仍各执行一个真实 `env.step()`。
- 第 1 帧必须把第 0 帧 direct outcome 回填为 E1；第 2 帧必须把第 1 帧已完整观测的零通信结果回填为 `valid=true` 的零值。

该 fixture 只用于验证时序、历史回填、合法零值和索引稳定。它不是训练样本、不是性能评价，也不证明多任务、多流、动态拓扑或多 seed 覆盖。

## 7. 组件边界

### 7.1 纯契约模块

新增 `代码/src/pi_jwm/multistep_collector_contract_v1.py`，只负责词表、帧记录、上一帧回填和纯验证，不导入 AirFogSim。

### 7.2 AirFogSim 适配层

在 PI-JWM 源码中新增或扩展适配器，负责 setter 前 CSI 读取、现有单步真实执行边界复用、帧阶段事件和结果 ledger。不得修改 `代码/reference/AirFogSim/`。

### 7.3 Runner 与产物

新增 `代码/scripts/run_p2_multistep_collector_preflight_v1.py`，生成版本化 bundle。单步 runner 同时修订并重放；旧正式目录先移动到带时间和缺陷标签的归档目录，不覆盖、不删除。

多步 bundle 至少包含：

```text
trajectory_frames.json
vocabularies.json
temporal_trace.json
history_alignment_audit.json
resource_bundle.json
validation_report.json
summary.json
manifest.json
```

## 8. 原子失败与证据发布

- 所有 payload 先在内存验证，再写同父目录临时候选目录。
- 候选目录中的文件哈希、源码/设计/测试输入哈希和状态标志全部复核后，才原子提升为 canonical 目录。
- canonical 已存在时默认拒绝覆盖；只有显式归档且归档复核成功后才能发布新版本。
- 任一步失败不得留下带 `passed=true` 的 canonical 目录；失败证据写入独立 rejected 路径或仅报告错误。
- verify-only 必须重新计算所有哈希与关键时序门，不能只读取 `validation_report.json` 中的布尔值。

## 9. 测试与验收

实施采用 TDD，至少覆盖以下门：

1. 旧 event attenuation 直接冒充 action-pre 时测试失败。
2. setter 前 CSI 读取顺序由调用 trace 证明。
3. outcome CSI 不能进入同帧 pre 字段。
4. 第一帧 E1 为 `no_history`。
5. 第二帧 E1 与第一帧 direct outcome 在 edge identity、数值和单位上逐项一致。
6. 第三帧合法零历史为 `valid=true/missing_reason=none`。
7. node/edge/flow 已有索引跨三帧不变；消失重现和追加单元测试通过。
8. 当前 action、outcome、下一帧 history 使用同一 edge/flow 索引。
9. 三次真实 `env.step()` 与原生模块顺序可审计，CPU/能耗守恒继续成立。
10. 任一帧失败时 history 不提交，canonical bundle 不发布。
11. manifest 的 artifact、代码、测试、设计和 AirFogSim 源文件哈希全部独立复算一致。
12. `gpu_started=false`、`locked_test_accessed=false`、`training_eligible=false`、`v4_collector_implemented=false`、`v4_dataset_complete=false`、`candidate_rollout_planner_complete=false`、`final_method_frozen=false`。

## 10. 对已有实验的影响

- P2 单步真实动作执行、通信、CPU、能耗和单因子候选对照仍可作为局部证据。
- 现有 P2 单步 bundle 对 action-pre attenuation 的证明失效；修订后必须重放，旧 bundle 仅作缺陷归档。
- P1 registry 的目标定义不因本次修订失效，但 P1 只证明字段契约，不证明真实采集完成。
- 旧 R1/R2、R3-R6 数据和训练没有 setter 前 v4 decision-time CSI 与严格跨步 E1，因此不能自动升级为 v4 结果，也不能用于继续 100k。
- 本 smoke 通过后只解除“多步时序/词表接口未验证”的局部阻塞；正式数据生成仍需完整双图、多任务/多流/动态拓扑、多 seed、失败恢复和覆盖率门。

## 11. 暂不开展

- 不启动 GPU 或任何长训练。
- 不访问 locked test。
- 不补造 13 维字段，不用 outcome 值改名伪装 action-pre。
- 不实现或宣称世界模型候选 rollout 规划器。
- 不把三帧 fixture 作为分布覆盖、模型效果或最终方法证据。
