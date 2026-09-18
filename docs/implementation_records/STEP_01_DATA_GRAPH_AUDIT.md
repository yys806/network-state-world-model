# Step 01 数据与双图实施审计

## 审计范围、证据等级与限制

- 日期：2026-09-18。
- 范围：仅将 `D:\shen\OB\科研\PIJWM` 中 `01仿真系统与原始轨迹.md`、`02数据集构建与模型输入.md`、`03物理-信息双图建模.md` 的定义，同本仓库现有 source/config/test/artifact 对照；不作研究决策、不改模型或数据。
- 外部定义笔记在本审计中只读。仓库原有 `task_plan.md`、`progress.md`、`findings.md` 已在审计开始前为 dirty；本 Step 保留原内容并在末尾追加了 Step 1 过程记录。
- 本 Step 运行了 49 项既有 synthetic CPU contract 测试；它们不构成新定义验收。未执行训练、数据生成、GPU 或 `locked_test` 访问。下文其他“已通过”如无本次命令说明，只转述既有 machine-readable artifact。
- 判定含义：`DIRECT_REUSE` 可按原接口直接使用；`MINOR_MODIFICATION` 不改变数据语义的局部补齐；`STRUCTURAL_CHANGE` 需变更输入/图/模型合同；`MISSING` 当前无可用实现证据；`RESEARCHER_DECISION_REQUIRED` 定义留给研究者确定；`HISTORICAL_ONLY` 只能追溯，不能作为本轮新定义的实现证据。

## 当前可核验数据基线

| 项目 | 当前事实与证据 | 边界 |
|---|---|---|
| 轨迹划分 | `formal_airfogsim_dataset_v1.py:18-23,74-114,137-184` 定义 6 场景、60 trajectory、36/12/6/6 train/validation/calibration/locked_test；`require_split_access` 在 `:187-194` 拒绝未授权 locked_test 标签访问。 | 这是旧 formal-v1 协议，不等同于新定义的数据方案。 |
| 当前可用 tensor | `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/dataset_summary.json`：54 条 unlocked trajectory、H=8、L=20、14,742 windows、train-only normalization、locked_test_trajectory_count=0。`validation_report.json` 记录 `formal_tensor_ready=true`、`formal_training_ready=false`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`。 | 当前 tensor 是旧合同的可追溯资产，不能据此声称满足 03 的严格双图定义。 |
| 既有 CPU 测试证据 | `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/cpu_existing_tests.json` 记录 5 个既有 synthetic CPU test module、49 项测试返回码 0，GPU 未启动、locked_test 未访问。 | 该 artifact 自己标注为“not new-definition acceptance”。 |
| 历史结构自述 | 新定义 `03:18-24` 明确：GitHub 当前通信链路被当作 `physical_edge`、两侧输入语义未真正分离，Task/Flow/DAG/Comm 未统一为信息世界；这是本审计的首要冲突证据。 | `HISTORICAL_ONLY`，它说明旧实现，不证明新结构已实现。 |

## 逐项定义—实现对照

| 定义要求（文件:行） | 当前源码、测试、artifact | 判定 | 可复用部分 / 缺口 / 对检查点的影响 |
|---|---|---|---|
| `01:42-53,236-252`：每帧严格 `Decision -> Execution -> Outcome`；A 是实际提交并执行的动作，Outcome 不能倒灌为同帧 decision input。 | `full_dual_graph_collector_contract_v1.py:18-22` 定义三 phase；`airfogsim_full_dual_graph_collector_v1.py:351-370` 先抓 decision 并以 DECISION 来源验证动作；`:520-534` 在真实 `env.step()` 内抓 execution、步后抓 outcome；`:551-578` 将传输结果标为 outcome-only。`formal_airfogsim_collector_adapter_v2.py:45-88,150-168` 分别保存三个 snapshot、RB outcome observation。相关测试：`test_formal_airfogsim_collector_adapter_v2.py:183-306`。 | `DIRECT_REUSE` | 顺序、因果来源与 outcome 隔离可保留。新协议仍须明确每个新字段属于哪个 phase，并做真实 runtime replay 验收；当前 synthetic 测试并非新定义验收。 |
| `01:253-285,307-324`：trajectory 内 node/task/flow/relation ID 稳定；记录 `(trajectory_id, frame/time, phase)`；失败/不完整不得静默拼接；采集策略、场景、seed、决策周期可追溯且覆盖可行动作。 | `full_dual_graph_collector_contract_v1.py:95-143` 有 flow/hop/RB/action identity；`action_attempt_ledger_v1.py` 与 `airfogsim_full_dual_graph_collector_v2.py:198-270` 记录动作尝试与 reject/quarantine；`formal_airfogsim_dataset_v1.py:40-53,74-114` 有 trajectory/seed/split/cpu policy/scenario；`airfogsim_tensor_v2.py:145-163` 要求统一时间网格。 | `MINOR_MODIFICATION` | ID、time grid、quarantine、seed/split 元数据可复用。新定义的 collection policy、完整 generation config、`Decision/Execution/Outcome` 的原始持久化 schema 及跨 phase 对齐尚未形成单一正式原始轨迹 artifact；需补合同和实轨迹检查。 |
| `02:16-25,30-90,265-328`：样本为 `History + Future Action -> Future Target`；History 截止当前 decision；future action 是训练时已执行、在线时候选的条件；每个 future action 与同槽真实 outcome 对齐，不跨 trajectory/reset。 | `formal_airfogsim_window_v1.py` 与 `formal_system_window_v1.py` 是现有窗口构造；`airfogsim_tensor_v2.py:409-450` 将动作按 observed action time 写入 `task_action`；`formal_airfogsim_collector_adapter_v2.py:108-162` 区分 action 与 outcome transfer/RB observation。当前 v5 tensor 的 H=8/L=20 见上表 artifact。 | `DIRECT_REUSE` | 因果窗口、时间对齐、训练/在线 action 语义可复用。必须重新核对新 action schema 是否覆盖新研究动作；既有 `task_action` 不是新 action space 的自动证明。 |
| `02:122-180,181-202`：Object ID 到稳定 Entity/Task/Flow Index；presence、feature mask、padding 分开；关系端点引用同一 index，DAG 为有向关系。 | `airfogsim_tensor_v2.py:170-193` 稳定排序 vocab；`:289-321` 建立固定容量 arrays、presence/mask、endpoint index；`:334-408` 写 node/task state、mask、`task_node_index`；`:455-514` 写 flow endpoint/type/task、bearer、DAG index；`:552-610` 检查 padding 为零、索引合法、mask shape。测试：`test_airfogsim_tensor_v2.py`、`test_formal_airfogsim_graph_v1.py:160-188`、`test_formal_directed_graph_ops_v2.py:8-143`。 | `DIRECT_REUSE` | 稳定索引、presence/mask、关系端点、DAG 方向均可复用。注意 `agent_node_index` 当前只是附着映射，未形成独立 agent state/index contract；见后续严格双图缺口。 |
| `02:393-455`：normalization 只由 train 拟合，trajectory 级 split 隔离；metadata 不做模型特征；锁定测试不在模型冻结前访问。 | `formal_airfogsim_dataset_v1.py:137-194`；v5 `dataset_summary.json`/`validation_report.json` 记录 train-only normalization、locked split 未 materialize。 | `DIRECT_REUSE` | 数据隔离、安全边界可复用。新双图新增 feature/relation 的 normalization、mask 统计和完整 trajectory split 需纳入同一新 manifest；现有“tensor ready”不能替代。 |
| `01:86-100`：当前采集可控维度是执行位置/路由、RB、CPU；它不是最终研究动作空间。 | 既有 tensor 真实记录的是四种**事件/控制记录**：offload、RB、return、CPU，`airfogsim_tensor_v2.py:25,411-450` 与 `formal_airfogsim_graph_v1.py:61-119`。其中 offload/return/RB 在 base action 5 维，CPU 在 formal extension 3 维。`formal_rule_tensor_contract_audit_v1.py:48-86` 对 offload/RB/return/CPU 的 source endpoint 完整性审计。 | `DIRECT_REUSE`（记录事实）；`RESEARCHER_DECISION_REQUIRED`（最终 action space） | 可复用事件帐本、作用 task、source/target endpoint、RB 数量、CPU allocation。**不得把旧四个事件误称为新四类 `Route/Comm/Comp/Mob`：当前没有 mobility action 记录；offload 与 return 是两个 route-related 事件，RB 是 communication-resource 事件，CPU 是 computation-resource 事件。** 新四类研究动作如需 Route/Comm/Comp/Mob 需要研究者定义与数据合同重构。 |
| `03:88-149`：Physical Graph 只表达实体空间存在、位置/运动与由空间邻域决定的边；不以通信、task/flow/RB 活动决定 physical edge。 | 当前 `NODE_FEATURES` 为 x/y/z/speed/acceleration/**cpu/storage**，`EDGE_FEATURES` 为 distance/**csi_mean/rate_sum/active_task_count/allocated_rb_count**，`airfogsim_tensor_v2.py:11-36`；实际写入见 `formal_airfogsim_collector_adapter_v2.py:310-407` 和 `airfogsim_tensor_v2.py:334-353`。 | `STRUCTURAL_CHANGE` | 位置、运动、实体 ID、空间端点、distance、node presence 可复用。CPU/storage 以及 CSI/rate/activity/RB 当前混入 physical tensor，直接违反严格语义；需拆分物理 feature contract、重建 physical spatial-neighborhood 规则与新 tensor/normalization/check。 |
| `03:151-353`：Information Graph 节点为独立 Agent + Task；边为 Comm、Task–Agent(Src/Host/Exec/Ret)、Flow、DAG。Comm 是通信条件，Flow 是业务数据过程，两者不可合并；Flow 为可并行有向多重 relation。 | `airfogsim_tensor_v2.py:299-320,395-514` 有 flow/task/DAG、task endpoint 及 flow endpoint，但没有独立 `agent_state`、Agent presence、Comm relation/state、Task-Agent typed relation/validity。`formal_airfogsim_collector_adapter_v2.py:176-206` 产物把 DAG 放 `information_edges`，且 `ep_relations=[]`；其 `physical_edge_snapshots` 承载 CSI/rate/RB。`formal_airfogsim_graph_v1.py:32-44` 还禁止 dependency-data Flow。 | `STRUCTURAL_CHANGE` | 复用 Task、Flow（含 task_input/result_return/dependency_data enum）、DAG、task 的 source/host/exec/ret endpoint 原始事实。缺独立 Agent/Comm/Task-Agent 图合同、typed validity、并行 relation vocabulary，且 DAG DepData 当前被禁止；新定义是否保留/如何采集 DepData 属 `RESEARCHER_DECISION_REQUIRED`。旧 Flow↔physical-edge bearer 只可作历史迁移线索。 |
| `03:357-450,690-750`：类型专属编码；按稳定历史对齐；不同 relation family；Comm/Flow/Task-Agent 要区分有向正反计算，DAG 默认仅 parent-to-child。 | 当前 `formal_dual_graph_world_model_v1.py:24-28,299-359,466-529` 只含 node、physical_edge、flow、task 四 component；历史 GRU 与 DAG 单向存在。`formal_graph_ops_v1.py:27-101` 对 physical/information relation 做两端对称聚合，且 information=agent+flow；`formal_directed_graph_ops_v2.py` 有通用 directed utility，但未证明接入 formal-v1 current graph contract。 | `STRUCTURAL_CHANGE` | 可复用时间对齐、GRU、mask、DAG parent→child，以及 directed helper 的低层工具。缺 relation-family processors、Comm/Task-Agent 独立编码、明确 reverse computational path、typed state update；禁止把“有 helper/test”称为新双图已接入。 |
| `03:541-689`：基础跨图只允许同实体 `Phy -> Agent` 的 gated P2A，以及 physical endpoint context 到 Comm relation 的 gated P2C；不默认 Info->Phy、Task↔Physical 或 Flow↔Physical edge。 | 当前 `formal_dual_graph_world_model_v1.py:488-519` 调 `couple_agent_physical` 和 `couple_flow_bearer`，并把 task 直接散射到 node/agent；`formal_graph_ops_v1.py:71-101` 的测试证实 agent-node 双向、flow-bearer 双向。 | `STRUCTURAL_CHANGE` | `agent_node_index` 可作为同实体 ID 锚点直接复用；已有 coupling 可作旧 baseline。缺单向 gated P2A、P2C（Comm relation 尚不存在）、路径 mask；现有双向 agent/physical、flow/physical、task/physical 不能直接复用为新基础结构。P2C 是否保留是文档明确需 ablation 的 `RESEARCHER_DECISION_REQUIRED`，不是工程默认。 |
| `03:785-806`：H1-H5 为待验证假设；特别要求 P2A vs P2A+P2C、单向 vs 双向等保持 data/history/action/prediction boundary/capacity/protocol 一致。 | 当前仅存在旧双向 coupling 与 historical tests；新定义 `03:18-24` 已将旧结构标为冲突。没有针对 strict split、P2A、P2C、方向、relation family 的新实验 manifest/acceptance artifact。 | `MISSING` | 旧 tensor 可作为迁移源或 historical baseline；新研究比较前须先冻结语义合同、数据重建/兼容映射、模型接线与 CPU contract，再由研究者批准 ablation/训练。 |

## 通信充分性与四种旧动作事件的专门核查

1. **通信当前是否有采集事实？** 有。decision snapshot 中以 `channel_rows` 读取 CSI，`formal_airfogsim_collector_adapter_v2.py:300-403` 将其聚合为 `csi_mean`；执行/结果阶段的逐 RB rate 与 delivered data 从 `transfer_rows` 进入 `source_rb_observations`，`:150-162`，并由 `formal_airfogsim_graph_v1.py:205-245` 构建每 RB target。这足以复用为“原始通信观测/结果来源”。
2. **通信是否已满足新 Comm relation？** 否。当前 CSI、rate、active_task_count、allocated_rb_count 都在 `physical_edge_state`；没有以 `Agent -> Agent` 独立 relation、Comm type、validity、独立状态或 P2C 输入保存。因此“有通信原始事实”不能升级为“新信息图 Comm 已实现”。
3. **旧四种动作记录的精确映射。** `offload` 记录 task 的 source/target 与 route；`return` 记录 current/return target 与 route；`rb` 记录 task 当前 hop 的 RB 分配；`cpu` 记录 task 的 CPU allocation/fraction。它们均来自实际 setter/callback，采集路径分别见 `airfogsim_full_dual_graph_collector_v1.py:404-467`、`formal_airfogsim_graph_v1.py:61-119`。这些是旧数据合同下的四个动作事件，并不是新研究动作分类的已批准定义。
4. **路由充分性。** offload/return 的 route list 在 collector action record 保留，但 base tensor 只存 source/target index，未保存完整 multi-hop route sequence；Flow 的 `flow_bearer_mask` 以执行 path 连接到旧 physical edge（`airfogsim_tensor_v2.py:499-507`）。若新 Route action 要以整条候选路由条件化未来，现有 tensor 不足，至少是 `STRUCTURAL_CHANGE`。
5. **移动充分性。** 节点位置/速度/加速度在 state 中，未发现 mobility command/action tensor。把状态运动误称为 mobility action 会破坏 Decision/Outcome 边界；若新定义需要 Mobility action，判定 `MISSING`，其控制接口、可行性、采集策略须 `RESEARCHER_DECISION_REQUIRED`。

## 新定义实施前的兼容性检查点

1. 冻结研究者批准的 action ontology，明确旧四事件如何映射，或明确不映射；尤其确认是否真的引入 Mobility action。
2. 冻结 strict Physical / Information schema：Physical 仅空间实体/空间边；Information 的 Agent、Task、Comm、Task-Agent、Flow、DAG 各自 ID、state、mask、relation type、direction 和 endpoint。
3. 制定旧 formal tensor 到新 schema 的逐字段映射表；无法无损映射的字段应标记需要重新采集，不能以改名通过。
4. 对新的 raw trajectory 逐帧验证 Decision/Execution/Outcome、实际执行动作、任务/flow/agent/comm 对齐及 outcome 不泄漏；对 route、RB、CPU、可能的 mobility 分别检查可行性和来源。
5. 在新 tensor builder 中验证：split/normalization/locked_test 边界、Agent/Comm/Task-Agent typed relation、Flow 与 Comm 不混同、DAG 与 DepData 的研究者决定、padding/presence/mask、P2A/P2C 所需的合法 endpoint/mask。
6. 只有完成以上 source—tensor—model-read-path 一致性审计后，才讨论 P2A、P2C、单/双向及 message-passing 的受控实验。当前 artifact 不支持进入该实验结论。

## 结论

当前仓库可直接继承的是**真实时序采集的因果顺序、稳定身份与 tensor 对齐、presence/mask、旧四动作事件及其执行证据、task/flow/DAG 原始事实、数据隔离边界**。新定义要求的严格物理—信息语义分离、独立 Agent/Comm/Task-Agent 异构信息图、P2A/P2C 单向 gated 跨图通路，以及把新动作类别正式化，均尚未接入当前合同；这些不能以旧 formal tensor、旧双图模型或既有 synthetic tests 宣称完成。
