# PI-JWM Model-ready Sample & Tensor Contract v1

状态：STEP 3.1 候选实现并完成最小真实样本验收；不是正式大规模数据集，也不是模型输入已冻结为最终研究模型。

依据：研究者只读定义 `D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`，以及已冻结的 Step 2 Raw Contract。研究笔记只读，本仓库保存实现映射和证据，不复制私有笔记。

## 样本时间合同

对当前决策 frame `t`、History 长度 `H`、预测长度 `L`：

```text
D_t = (History[t-H:t], Static, FutureAction[t:t+L], Target[t:t+L], Metadata)
```

其中 History 的最后一帧是 `O_t`，所以当前动作 `A_t` 不进入 History。`FutureAction[t+k]` 与同一步执行后的 `Target[t+k]` 一一对应。窗口只能在同一 `trajectory_id`、同一连续时间网格内构造；不能跨 reset 或跨 trajectory 拼接。

最小真实验收使用 `H=2, L=2, anchor frame=2`：History frame `0,1`，Future Action/Target frame `2,3`。

## Index、Presence 和 Mask

输入 index 只由 anchor 前最后一个 Decision 可见对象建立。当前对象 ID 稳定映射到 sample 内 index；对象不存在使用 `presence=false`，对象存在但字段不可得使用 `feature_mask=false`，真实值为 0 仍是 `presence=true, feature_mask=true, value=0`。padding 位置不参与归一化统计。

未来新对象不提前加入输入 index。Target 单独保留 target-side index 和 `target_only_objects`，因此新出现 task/flow/entity 可以在 Target 中表示，同时不进入当前 History 或 input-side index。

## 四类 Future Action

`Route / Comm / Comp / UAV Mobility` 均有正式 action family。训练模式从 Raw trajectory 的真实 action 读取；在线模式应提供同字段、同对象索引和同 no-op/missing 语义的候选动作。动作条目保存 task/entity/RB/UAV 的稳定引用；车辆运动保持 `SUMO external`，不进入 Mob tensor。

`empty=true, missing=false` 表示字段已观测且没有动作，`missing=true, empty=false` 表示字段没有可靠来源。二者不能互换。

## Raw Field → Dataset Role

| Raw field | Dataset role | 处理边界 |
| --- | --- | --- |
| decision entities/tasks | Model Condition / History | 只取决策时可见对象；按稳定 ID 建 index |
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
| DAG dependency rows | Static / Target relation | 当前 Raw observer 未捕获可靠 dependency rows，显式 missing，不从旧 tensor 猜 |
| trajectory/seed/split/frame/time/contract version | Metadata | traceability/audit，默认不作为模型输入 |

## Flow/Data 和 DAG 边界

本 Step 的 Flow/Data 是真实 transfer event 形成的 hop-level service record：`flow_id = task + transport + source -> destination`。它保存 source、destination、task mapping、transport、service volume 和 presence；同一 payload 的 wireless 与 wired 两跳分别记录，service volume 可相加为累计 transport service，但不等于 task 的 end-to-end payload progress。

当前 Step 2 Raw artifact 没有独立、可靠的 DAG dependency rows。样本因此输出 `dag_relations.observed_mask=false` 和原因 `RAW_DAG_DEPENDENCY_SOURCE_NOT_CAPTURED`。这是已记录 Gap，不是空 DAG 结论。

## Split 和 preprocessing

固定顺序：`Trajectory-level Split -> Window Construction -> Fit preprocessing on Train only -> Apply to Validation/Test`。统计量只使用 valid/masked-in 值；padding、missing 和无效 feature 不参与统计。保留原始单位、mask 和 normalization metadata。

## 机器合同和边界

机器定义在 `code/src/pi_jwm/model_ready_sample_contract_v1.py`，最小真实构造入口为 `code/scripts/build_step3_1_minimal_real_model_ready_sample_v1.py`。最小证据使用非 locked Step 2.4 最终通信 Raw artifact，`H=2, L=2`，不训练、不用 GPU、不访问 `locked_test`。

旧 `airfogsim_tensor_v2`、formal window loader 和旧 `flow_state` 只复用通用 index/mask/normalization 机制；它们不自动成为新 02 Flow/Data 语义。
