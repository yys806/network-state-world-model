# STEP 6.4N — Dynamic Return 实验合同 v2

状态：研究者已授权的 opt-in CPU 实验合同；不是正式 Planner 版本。2026-10-12。

## 授权与版本隔离

配置 `STAGE_AWARE_RETURN_TTI_V2` + `DYNAMIC_RETURN_V2` 只在隔离 harness 生效。旧 AirFogSim 源文件、v1 Scorer/Domain/Grammar/Live bridge 不修改。TTI 阈值、计算精确完成判定、deadline、神经参数、normalization、五元组和字典序冻结。动态 burden 为 `OBJECTIVE_V2_SEMANTIC_CHANGE`，不可与历史 v1 分数直接比较。

## 真实时序与接口

| 时点 | 真实 runtime | v2 责任 |
|---|---|---|
| decision 前 | collector 回放旧因果前缀，capture Raw/History | 所有分支 seed/prefix/root/RNG 相同；尚无新 Flow |
| callback setter | TaskScheduler 写 Return buffer；CommunicationScheduler 可预写 TaskID→RB | 对整份动作先检查 waiting bucket、FP64 exact completion、需求、实体/无线边、单跳、TRAIN RB 模式；通过后先 Route 再 RB；异常禁止 step/retry |
| env.step task | generate/check → consume Return route → checkTasks | Return仅将 checkTasks 的 TTI getter 改为阶段活动时钟；Input仅正活动刷新原有Input clock；Return 首次消费初始化；重复 Route 不重置 |
| env.step wireless | allocation 从 returning/offloading bucket 取任务，读取已写 RB → rate/outage → transfer | 不生成边、不改变 CSI、服务或随机数；只有正传输活动刷新阶段 clock；保留原 deadline/其他业务分支 |
| next observation | real capture Task、Flow、transfer | Ledger 使用显式 Route/真实 transfer 更新；future outcome 仅作对照，不回灌预测 |

源码锚点：`task_manager.py::checkTasks/setTaskReturnRouteAndStartReturn`、`task.py::startToReturn/transmit_to_Node`、`communication_sched.py::setCommunicationWithRB`、`airfogsim_env.py::_updateTask/_allocate_communication_RBs/_execute_communication`。实际运行导入路径与源码哈希落盘，checkTasks中的TTI getter调用点须唯一。6.4M/R1用于对照，不宣称已由其旧hash自动验收新实现。

## 支持域、动态对象与因果边界

只允许 anchor Raw 明确 waiting_to_return、FP64 已完成、大小>0、宿主≠目的地、没有 Return Flow 的单跳 Return。其他 future birth、Task_15 computing/FP32 rounding、Input 优化、多跳、reroute、目的地变化、缺字段、重复 ID、slot overflow 全部拒绝。Input/Return identity/slot/mask/E2E 守恒底层接口复用，Input 优化未解禁。

共同 pre-Route state/graph/current-posterior latent → candidate action admission → 官方 Ledger Return birth → append Flow slot/整数 index/graph rebuild/zero hidden → 冻结 one_step(mean prior, expectation service) → H2–H4 重绑当前 Flow/Comm。Route 在 birth adapter 消费后不再次提交 learned route effect，避免双重执行。新 Flow 在当步 learned dynamics 前可见；旧对象 tensor/latent 不覆写。ZERO_COLD_START 为本轮基线，仍可能分布外，不声称预测可靠或无需重训。

## DynamicScorerV2

共同 anchor Task cohort、五目标 `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)`、H≤4、RB/CPU/Mob denominator 和字典序保持。anchor 正确 FP64 waiting Return 需求成为候选无关 Tx 基准；NOOP 保留全部需求，birth 没有奖励，只有 single-hop E2E useful progress 降低它。

每个 Task：anchor Input/其他已有 Tx 量与新增已知 Return 量互斥去重，`Tx0=sum(anchor active Flow remaining)+unmaterialized known Return size`；`Tx_h=sum(existing Flow remaining)+known Return remaining`（未 birth 时等于原需求）。Comp 保留 `work_h/work0`。burden 仍是适用 Tx/Comp 归一项均值，再按共同 cohort/horizon 均值。Return Flow identity/index/type/size/守恒必须验证，不能由 presence=false 推断已传完。真正未确认需求的 future birth 保持边界拒绝。

预测 TTI 用显式 v2 clock/阶段/正 E2E progress 推进，机制可确定超限候选 HARD_REJECT_TTI，与 deadline 指标分离。不得使用真实未来 failure 来拒绝预测候选。候选集合 common H_eff 取支持最小值；H_eff<4 不能冒充完整 H4。Empty/all rejected 返回明确不可评分。Route effort 不加入；Comm/Comp/Mob 复用旧 effort 公式和共同 denominator。

## 模块影响矩阵与依赖顺序

| 层 | 新 opt-in 接口 | v1 回归 |
|---|---|---|
| AirFogSim | instance-scoped stage-aware TTI adapter；源码唯一getter调用点检查、temporary getter、context restore | 源文件/类方法无改变，退出恢复 |
| Ledger | 官方 create_return_flow/apply_transfer_event | 不改旧 Ledger |
| Candidate/bridge | DynamicCandidateDomainV2隔离声明域 + 四动作族整体校验/提交；复用Comp节点资源和Mob映射检查，当前returning独立资格检查 | v1 NOOP/eligibility不改；v2 learned tensor以transmitting表示原始returning |
| state/graph/latent | action-trigger append + ZERO_COLD_START + rebuild | frozen weights/encoder 不改 |
| scorer | DynamicScorerV2，版本化 Tx burden、TTI hard rejection | 旧 scorer 不改 |

Gate 1：四模式（legacy Route、v2 Route无RB、v2 Route+RB、v2 NOOP），同一 Task_5 causal root，最多每分支4 steps；先验证执行。Gate 2：同一捕获 root 两候选/各H4，失败保存逐步回执。Gate 3：共同分母/需求/排名复算。最多2真实root；若需要替代按 trajectory ID SHA 排序、frame升序、当前 waiting Return exact completion/单跳无线资格筛选，先落候选清单再看未来结果。

## 实施计划与停止门

1. 本合同与测试先于 v2 实现；先 native TaskManager 单测、setter 原子前置验证和回归。
2. Gate 1 bounded runtime 采集四模式，阶段立即落盘；各分支超时180s，SUMO finally close，无自动重试。
3. Gate 1 PASS 后从捕获 prefix 按 live path prepare_anchor；单次两个候选各H4（8 WM transitions）；各次超时300s。修复重验保留全部失败namespace及累计计数，不增加root/动作搜索/挑样。
4. Dynamic Scorer 独立审计/误差/覆盖/参数 SHA；不扩大实验凑 PASS。
5. Context/知识索引/tests/diff/commit/push 收口。正式 Planner/S-CEM、GPU、训练、locked_test 与 6.4O 关闭。

## 实际实现限制与验证门

共同anchor先移除尾部未占用的Flow填充，不移动任何已知Flow/hidden，随后按官方Ledger连续index追加；旧Shadow的padding末端slot不直接作为正式业务index。所有候选共享这一步处理。birth的新Flow在当前one_step之前可见，新hidden为零；旧posterior与神经参数不变。

当前真实harness只执行单Task Return Route/RB以及Comp NOOP/Mob HOLD。v2模型harness接受typed候选，但不是正式CandidateDomain/S-CEM；四族联合校验、当前已有returning实时bridge、Input无活动计时及TTI终止返回已有CPU负例验收；真实四分支经联合入口执行，非空Comp/Mob联合资源有界测试，不冒充多任务正式搜索性能，未支持路径不能因此获得正式资格。实验配置manifest用于显式记录选定版本，默认正式v1不读取它。

FP64 Raw需求为共同baseline；模型FP32剩余量以 `Raw需求 × (模型剩余/模型总量)` 转换为同单位比例，不让birth的dtype舍入产生奖励。负担降低须有v2 causal useful-progress事件和服务前Comm relation/RB绑定；完成后的relation可能清除，不能冒充之前没有服务。

无线服务参数、有效Return交付、旧collector事件分别保存。旧collector事件在Task_5已证实错误地按Input task_size裁剪；不作为v2真实Return交付量。不改旧数据、模型checkpoint或normalization。

CPU机制通过不建立预测可靠性；Task_5首隙饱和完成不能验证未完成传输。正式Route集成仍BLOCKED，RETRAIN_REQUIRED=NOT_ESTABLISHED。完整目标支持与正式搜索必须重新授权，6.4O和训练均不自动启动。

## 联合入口补验（2026-10-12）

`step6_4n_joint_bridge_v2.py` 中隔离声明域提供NOOP/Route无RB/显式TRAIN RB block三类控制，规则只验证调用者声明，不自动选择动作。全族检查完成后Route→RB→Comp→Mob提交，任一setter失败隔离环境并禁止step/retry；当前已有returning的只读桶检查不用原生有副作用getter。Input保持既有阶段起点，仅正传输刷新活动clock。预测TTI致命状态返回明确reason并停止neural递归，Scorer硬拒绝而非DDL失败，不填充假H4状态。真实联合入口复用先前仅含因果输入的common_anchor，并要求当前Raw全量fingerprint一致；不是通用live root构建或正式搜索绑定。
