# PI-JWM Model-ready Sample & Tensor Contract v4 (STEP 3.3F)

状态：STEP 3.3F 完成 JSON sample → CPU tensor 的语义完整性收尾；不是正式大规模数据集，也未决定 03/04 的最终 feature selection。

依据：研究者只读定义 `D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`，以及已冻结的 Step 2 Raw Contract。研究笔记只读，本仓库保存实现映射和证据，不复制私有笔记。

## 样本时间合同

对当前决策 frame `t`、History 长度 `H`、预测长度 `L`：

```text
D_t = (History[t-H+1:t], Static, FutureAction[t:t+L-1], Target[t:t+L-1], Metadata)
```

其中 History 的最后一帧是 `O_t`，所以当前动作 `A_t` 不进入 History。`FutureAction[t+k]` 与同一步执行后的 `Target[t+k]` 一一对应。窗口只能在同一 `trajectory_id`、同一连续时间网格内构造；不能跨 reset 或跨 trajectory 拼接。

最小真实验收使用 `H=2, L=2, anchor frame=2`：History frame `1,2`，Future Action/Target frame `2,3`。History 最后一帧必须是当前 `O_t`，Future Action 第一帧必须是当前 `A_t`。

History 的完整形式为：

```text
H_{t-H+1:t} = (O_{t-H+1:t}, A_{t-H+1:t-1}, Y_{t-H+1:t-1})
```

每个历史行都有对齐的 `observation` 。对于过去帧 `τ<t`，额外保存该帧已执行的四类 `action` 和执行后已知的 `outcome`；当前帧 `t` 只保留 `O_t`，不保留 `A_t/Y_t`。

## Index、Presence 和 Mask

输入 index 由整个 History `O_{t-H+1:t}` 中已因果可观测到的 physical/task 对象并集建立，并在整个 History 窗口固定；机器合同 policy 为 `history_causal_observable_object_union`。Future Action 的机器 policy 为 `anchor_visibility_then_history_union_input_index`：先按 anchor visibility 拒绝不可见引用，再返回统一 History-union 数值 index。过去存在而后离开的对象也保留 index，后续帧使用 `presence=false`。较晚进入的对象在更早帧使用 `presence=false`。缺失 feature 和真实值为 0 仍分别处理，padding 位置不参与归一化统计。Flow index 同样由历史 Outcome 中已出现的传输 event 建立。

Static 另保存 `input_entity_type_by_index`，类型只来自截至 anchor 的 History/Raw 可见事实，并与 physical input index 一一对应。canonical entity type code 固定为 `<PAD>, unknown, vehicle, uav, rsu, edge, cloud`，不会因 development sample 的子集或顺序改变。

未来新对象不提前加入输入 index。Target 单独保留 `target_index.physical/task/flow` 三个 namespace 和 `target_only_objects`，因此新出现 task/flow/entity 可以在 Target 中表示，同时不进入当前 History 或 input-side index。

## 四类 Future Action

`Route / Comm / Comp / UAV Mobility` 均有正式 action family。训练模式从 Raw trajectory 的真实 action 读取；在线模式应提供同字段、同对象索引和同 no-op/missing 语义的候选动作。动作条目保存 task/entity/RB/UAV 的稳定引用；车辆运动保持 `SUMO external`，不进入 Mob tensor。

`empty=true, missing=false` 表示字段已观测且没有动作，`missing=true, empty=false` 表示字段没有可靠来源。二者不能互换。

Action 中的 task/entity/UAV reference 必须先通过 anchor-time visibility 检查，再解析到同一套 History-union input index。不可解析 reference 立即拒绝并标记 `RESEARCHER_DECISION_REQUIRED`；禁止写入 `-1` 后继续通过 validation。若 future-only task 在未来动作中出现，不能据此把该 task 提前加入 input index。

## Raw Field → Dataset Role

| Raw field | Dataset role | 处理边界 |
| --- | --- | --- |
| decision entities/tasks | Model Condition / History | 取整个 History 中因果可观对象并集；按稳定 ID 建固定 index |
| `channel_rows` | Model Condition / History candidate | 保留真实 RB/channel 来源和 mask；最终模型是否使用由后续 03/04 决定 |
| `node_cpu_capacity_per_s` | Model Condition / History candidate | missing 保持 null + mask + reason，禁止填 0 |
| `raw_simulator_acceleration_mps2` | Metadata / Audit | 不作为 canonical model feature |
| `canonical_acceleration_mps2` | Model Condition / History candidate | 只依赖当前与历史，缺历史显式 mask |
| `future_task_schedule` | Metadata / Audit | internal metadata only，不进输入 index |
| `route/comm/comp/mobility` action | Future Action | 四类动作保持相同 tensor 语义；训练是真实动作，在线是候选动作 |
| `wireless_delivered_data_by_task` | Target / Supervision | transport hop service，不是 end-to-end progress |
| `wired_delivered_data_by_task` | Target / Supervision | transport hop service，不是 end-to-end progress |
| `delivered_data_by_task` | Target / Supervision | wireless+wired 累计 transport volume；禁止命名为 task end-to-end progress |
| task `transmitted_size`, lifecycle | Target / Supervision | task/data actual progress 与通信 service 分开 |
| transfer event source/target/task | Static + Target relation | 真实 hop source/destination；flow ID 由 task/transport/hop 组成 |
| DAG dependency rows | Static / Target relation | 来自真实 observer `_extract_dag_edges`；仅将两端均为 anchor 可见 task 的 rows 映射到 input task index，future-only endpoint 保留 raw count 并过滤出模型输入 |
| trajectory/seed/split/frame/time/contract version | Metadata | traceability/audit，默认不作为模型输入 |

## Flow/Data 和 DAG 边界

本 Step 的 Flow/Data 是真实 transfer event 形成的 hop-level service record：`flow_id = task + transport + source -> destination`。它保存 source、destination、task mapping、transport、service volume 和 presence；同一 payload 的 wireless 与 wired 两跳分别记录，service volume 可相加为累计 transport service，但不等于 task 的 end-to-end payload progress。

当前 Raw artifact 的 DAG rows 来自 `airfogsim_full_dual_graph_observer_v1._extract_dag_edges`。Raw capture 先按 anchor 可见 task 分区：可见边进入 `decision.dag_edges`，含未来端点的边仅进入 `internal_metadata.future_dag_edges`。Dataset Static 只映射 `decision.dag_edges`，并保留可见边数和 internal future 边数作审计，未来端点身份不进入模型输入。

Static relation endpoints 来自 anchor Decision 的真实 `channel_rows.source_id/target_id`，映射到同一个 stable physical index，并保存 validity mask。本步只冻结端点引用，不决定这些 relation 在定义 03 中的图类型。

History 中的过去 Outcome 也保留对齐的 relation endpoints、DAG rows 和 flow rows，均引用同一套 History union index。这些记录只来自 `τ<t` 的真实 Outcome，不使用未来 Target。

## Future Action 引用观察审计

`code/scripts/audit_step3_1f_future_action_references_v1.py` 扫描当前可用的非 locked Raw 轨迹窗口，统计构造窗口数、未来 `A_{t+1:t+L-1}` 对 anchor `O_t` 不可见 task/entity 的引用数、action family 和对象类型分布，并给出 affected window rate。它是 observation-only；不丢弃 window，也不自行决定未来任务到达的表示方案。

## Split 和 preprocessing

固定顺序：`Trajectory-level Split -> Window Construction -> Fit preprocessing on Train only -> Apply to Validation/Test`。统计量只使用 valid/masked-in 值；padding、missing 和无效 feature 不参与统计。保留原始单位、mask 和 normalization metadata。

## 机器合同和边界

STEP 3.3F tensor 明确区分 Observation、Past Action、Past Outcome、Future Action 和 Target：Past Outcome 使用 `[B,H-1,...]`，Target 使用独立 namespace。Past Outcome tensor 保留 entity/task/flow、通信 service、served CPU work（有来源时）、relation 和 DAG；Target tensor 保留 future entity speed、task lifecycle/progress、flow endpoints/transport/service，以及 wireless/wired/total service。通信 hop service 不与 task end-to-end progress 合并。

类别 vocabulary 为稳定合同，不从当前 batch 动态生成：padding code=`0`、unknown code=`1`；lifecycle、entity type、route kind 与 transport 的合法 code 固定。Comp Action 正式字段为 `allocated_cpu_per_s`。

机器定义在 `code/src/pi_jwm/model_ready_sample_contract_v1.py`，最小真实构造入口为 `code/scripts/build_step3_1_minimal_real_model_ready_sample_v1.py`。最小证据使用非 locked Step 2.4 最终通信 Raw artifact，`H=2, L=2`，不训练、不用 GPU、不访问 `locked_test`。

旧 `airfogsim_tensor_v2`、formal window loader 和旧 `flow_state` 只复用通用 index/mask/normalization 机制；它们不自动成为新 02 Flow/Data 语义。
