# PI-JWM 场景与可扩展接口合同 v1

日期：2026-10-27。状态：**已确定规则已落实；数值、数据映射和科研细节仍待研究者确认。**

本合同只定义任务3的配置、记录格式和一致性检查，不是任务4的传输实现，也不是正式实验准入。PI-JWM是研究框架；AirFogSim只是参考仿真与服务数据工具。

## 1. 依据及优先级

1. 本轮研究者STEP R2-08指令，特别是100 ms、区域独立到达和截止时间公式。
2. 最新《新场景下训练前任务书.md》任务3及《1028组会.md》末尾新场景补充。只读文档路径和SHA-256见本轮source_identity.json，不复制私人笔记全文。
3. R2-05B、R2-06、R2-07的源码、测试及机器回执；旧审计结论只继承，不重新冒充本轮物理证据。

1028前半部分仍有“摄像头、UAV备选中继”等旧讲稿；末尾补充、最新任务书和本轮指令已明确“车辆与UAV感知源、LiDAR第一阶段”。这里按明确的新授权落实，不把旧讲稿混合成新决策。AGENTS中2026-10-23“未授权新场景实施”的历史快照不覆盖本轮有界授权；AGENTS保持不动。

## 2. 三层配置及准入

| 层 | 含义 | 允许范围 |
|---|---|---|
| 冻结规则 | 研究者已明确给定的范围、时间关系、因果约束 | 配置Schema用const/enum拒绝越界 |
| formal_pending | 尚未确认的参数全部为null，工程假设列表为空 | 可以校验结构，不能执行场景 |
| synthetic_contract_test | 每个非null测试参数都带ENGINEERING_FIXTURE_ONLY说明 | 只用于本轮合成合同测试，不能进入科研训练/性能实验 |

配置Schema：code/scripts/configs/pi_jwm_scenario_v1.schema.json。

正式待配置文件：pi_jwm_scenario_v1.pending.json；独立测试夹具：pi_jwm_scenario_v1.synthetic_test.json，均在同目录。任何缺失字段、未知字段、非有限数、负值、概率越界、非法单位、不完整执行配置都应拒绝。validate_scene_config(require_complete=True)只接纳显式合成测试；没有正式运行开关、隐式回退或自动训练入口。后续冻结正式参数需要研究者授权并显式演进合同，不能改profile标签绕过。

## 3. 已确定并冻结的范围

- 车辆与UAV均为感知信息源，第一阶段LiDAR协同3D车辆检测。UAV采用预设轨迹，不优化飞行。
- 一个逻辑RSU集成边缘感知服务器，逻辑上服务全部区域。逻辑可寻址不证明实际无线覆盖、正速率或包已送达。
- 计算资源、任务依赖图DAG、路由Route和感知处理流程固定。DAG指处理环节的先后关系；Route指数据经过的网络路径。
- 当前动作仅选择已准备的特征包、分配资源块RB。RB是可分配的无线频率资源单元。动作时间可在观测周期内部推进，内部间隔尚未冻结。
- RGB、多RSU、动态计算、DAG和Route优化均有显式扩展开关，v1全部关闭。字段保留来源ID、RSU ID、包/块ID、处理/模型引用，使后续版本可迁移；不实现扩展功能。
- 不改世界模型、checkpoint、loss、传播模型或正式训练协议。

## 4. 时间、任务生成和因果合同

### 4.1 时钟与到达

时间统一为从显式仿真原点起算的秒，外部观测边界t_k = k × 0.1 s，k为非负整数。测试包含k=0；正式场景是否从零时刻产生首个任务属于同刻事件/初始状态待决项。该网格不是V2U4Real物理采样时钟，文件名或名义10 Hz不能证明真实对应。

在每个边界，各区域独立作一次Bernoulli抽样，即以该区域配置概率p_r决定本次是否到达。p_r=0必不生成，p_r=1必生成。sample_region_tasks按(版本、显式seed、区域ID、tick)构造独立随机流，排序后输出；区域顺序和其他区域概率变化不影响该区域的抽样。可复现范围包括保存的Python版本；不是对真实到达分布的统计证明。同一tick重复调用是重新计算同一结果，不是再次入队；任务4负责一次性事件登记。

### 4.2 必须分离的时间

| 字段 | 含义与约束 |
|---|---|
| target_observation_time_s | 本任务要求观测的时刻，位于100 ms网格 |
| generated_at_s | 实际生成任务的时刻，不早于目标观测时刻；允许软件处理滞后 |
| available_at_s（观测） | 本端实际获得观测记录的时刻；missing时为null |
| preparation_started_at_s | 编码/包准备开始，不早于观测可用；可早于晚到的任务记录 |
| ready_at_s | 特征包准备完成，只有到此时刻后才允许作为待传输包选择 |
| absolute_deadline_s | target_observation_time_s + allowed_total_latency_s，与生成延迟无关 |
| decision/effective/information_cutoff_s | 决策、动作生效和可用信息截止时刻，cutoff ≤ decision ≤ effective |
| output_at_s | 感知检测输出时间 |
| actual_completion_time_s | 外部提供的实际完成回执时间，不早于检测输出；未完成任务为null |

100 ms不是统一截止时间，也不是唯一调度周期。任务可以跨越多个观测周期；生成新一批任务不会清空旧记录、改ID或自动使旧任务超时。本轮测试检查调用前后记录不变，**没有实现跨周期队列、累计、完成或超时引擎**。截止边界相等时的事件排序、晚到结果处置和评价需要任务4/5另行明确，不在此偷偷定案。

准备耗时来自显式参数configured_synthetic或测量回执measured_encoder；后者必须有测量证据引用。配置40 ms不等于编码器真实耗时。真实采集数据身份与仿真处理耗时分别记来源。

### 4.3 周期内部调度

validate_schedule_progress只检查传入的动作序列：时间不得倒退，相邻同刻动作必须消费不同且已提供的有限事件；事件不可重复使用。事件仅允许观测可用、服务反馈、定时器，不能靠预测制造真实事件。该有限事件方案仅工程夹具，正式同刻顺序仍待决；本轮没有调度循环。

在线事实引用仅允许当前已获得的ObservationRecord或ServiceResult。未来观测、未来服务反馈、标签及离线评价不能作为在线事实。模型预测独立列出预测ID、发出时刻、目标时刻和模型引用：目标可以在未来，发出时刻必须不晚于cutoff。本轮不运行世界模型；字段通过不能证明预测可靠或真实运行时无泄漏。

## 5. 六类版本化记录

接口Schema：code/scripts/configs/pi_jwm_scenario_interfaces_v1.schema.json；实现：code/src/pi_jwm/scenario_contract_v1.py。用严格JSON记录复用现有DetectionResult/载荷/容量接口，版本号PIJWM-Scenario-Interfaces-v1；未知字段拒绝，所有时间和字节字段显式命名。

共同来源provenance含kind、source_ref、time_basis、physical_sync_status、evidence_ref。kind区分官方train样本、合成夹具、仿真状态、模型预测、Mock占位输出。物理证据状态不能由合成或Mock改成VERIFIED；非空证据引用也只是声明，校验器不负责认证其真实性。真实数据引用限定v2u4real://train/，不访问其他划分。

| 记录 | 责任及关键字段 |
|---|---|
| ObservationRecord | observation/source/sensor/frame身份、目标观测与可用时刻、位置米、轨迹来源及引用、available/missing。位置坐标解释必须由trajectory_ref的显式合同提供；测试仅合成世界坐标，不声称数据集坐标与仿真地图已对齐 |
| PerceptionTask | task/region身份，生成/目标观测/绝对截止，预算、所属包ID、外部状态和完成时间。状态是输入记录，不产生状态转移 |
| FeaturePacket | 包/任务/来源/观测ID，整数byte_count，准备开始与ready时刻，准备来源，整包完整性，粒度/块ID/布局合同，载荷角色及引用 |
| SchedulingAction | 选择包、每包来源/RSU/RB分配、cutoff/决策/生效时刻、事实/预测引用和触发事件 |
| ServiceResult | action/packet/link_service身份，实际或假设区间，有效RB与逐RB速率，Mbps或bit/s，速率采样/区间平均语义，理论容量，独立delivered_bytes及交付证据 |
| PerceptionResult | 实际使用包ID，输出/完成时刻、DetectionResult、评价合同与数值来源、每包上游完整输入回执；区分合成、旧Mock、真实检测输出 |

validate_record_graph检查ID唯一、任务包归属、来源/sensor/轨迹一致、准备与信息可用、RB范围及时间因果。source_complete与fixed_spatial_block均保留标识；不生成特征。PerceptionResult要求每个使用包都有外部提供的完整输入回执，长度和时间匹配；真实输出不能使用合成回执或旧检测JSON冒充的编码特征。

input_receipts仅是**上游已经提供**的完整输入声明，含packet_id、received_byte_count、available_at_s、integrity_status、evidence_ref和evidence_kind。本轮不产生该回执、不求跨时隙和，不检查真实网络日志真假。引用本图内capacity_diagnostic不能充当交付证据。合成测试手工提供并明确标识这些回执，不能据此宣称实现传输/检测。

## 6. 速率、数据量与资源守恒

数据量统一byte，1 byte = 8 bit，1 Mbps = 10^6 bit/s。对有效RB速率求和：

    theoretical_capacity_bytes = Σ effective_RB_rate × unit_factor × interval_duration_s / 8

unit_factor对Mbps为10^6，对bit/s为1。空分配/零速率容量为零；保留小数容量，不擅自截断、累计或当整数交付。浮点容量比较使用相对1e-9和绝对1e-9容差，不给业务状态增加时延宽限。

- capacity_diagnostic/predicted_service的delivered_bytes及delivery_evidence_ref必须为null。
- delivery_receipt须另有交付证据、整数交付字节、反馈不早于区间结束；假定瞬时速率恒定不能单独证明交付。
- 速率语义分别为瞬时样本假定恒定、测得区间平均或测试常数。只有后两者可用来校验传入的交付记录字段，但不因此认证真实物理吞吐。
- 工程夹具使用exclusive_fixture：同动作不重复分RB，重叠服务区间也不重复使用RB；link_service_id不得按包复制。正式跨链路复用/干扰合同仍待研究者确认，不能把每链路容量逐包复制或相加。
- R2-07兼容函数要求调用者显式提供sample_time_s；若旧回执也带此字段必须一致。输出保留瞬时容量、BLOCKED物理状态和delivered_bytes=null。
- R2-05B兼容函数保留payload ID/字节数，但标为legacy_detection_result_json，只覆盖已审计Mock来源，不等于真实LiDAR编码特征。
- AirFogSim原生trans_data数值是Mbit、task_size单位未闭合；本合同的byte不自动绑定原生Task._task_size，不改vendor，不声称解决了原生累计。

## 7. 特征包粒度候选比较（未决）

| 方案 | 调度与竞争 | 检测器对缺失特征的要求 | 任务价值估计与限制 |
|---|---|---|---|
| 每源一个完整包 | 动作少、每包头部开销较低，但包大、较易跨时隙；同一源不可拆分竞争 | 必须定义某源整包未到时的单源/多源输入掩码、融合和结果更新 | 较易以来源组合消融测价值，但掩盖源内空间差异；价值仍可能因来源互补而非可加 |
| 每源多个固定空间块 | 更细粒度竞争，可优先关键区域；包数、标头、重组及竞争增多 | 必须定义坐标/网格、边界覆盖与重复、缺块掩码、融合网络是否支持任意块集合 | 需验证空间互补、重复目标及非可加价值；不能把块梯度直接当AP收益，须独立消融验证 |

source_complete只是测试选择，不是正式粒度决定。fixed_spatial_block需要显式布局合同和block_id；测试只检查标识，不实现真实切块/重组/缺块检测。整包完整性针对单个可调度包，不意味着整源所有块必须齐全；检测器如何处理缺块另待决定。

## 8. 真实样本与多节点网络候选（未决）

| 候选 | 可以声称 | 不能声称/缺口 |
|---|---|---|
| A 一真实车+一真实UAV，其余节点仅网络合成 | 真实训练样本用于已支持的模块输入；额外节点明确只有合成轨迹/服务状态 | 没有这些节点真实特征，不能评价完整多车感知收益；任务质量要另定数据支撑范围 |
| B 对固定来源显式重放/复用 | 工程上可构造竞争，必须记录原来源、重放索引、复用组与时间映射假设 | 不等于独立同时采集的多车视角；相关性、重复点云和标签复用会混淆协同收益，需要单独协议 |
| C 未来仿真传感数据 | 若有地图、传感器和标注模型，可以独立设计合成感知数据集 | 尚未实现，需独立几何/标签/域差验证；不能将生成数据标成V2U4Real真实采集 |

本轮未选映射方案、未复制/重放/生成真实特征。V2U4Real只提供已审计官方train范围的车—UAV样本；R2-06固定train发布无RGB，真实物理时间和配准仍未建立。AirFogSim轨迹、无线状态与真实帧分别记来源，不将文件名解释成仿真秒。

## 9. 待研究者确认清单

| ID/项目 | 需要确认的内容 |
|---|---|
| regions_and_bounds | 区域数量、边界、重叠/边界归属及权重 |
| real_synthetic_source_mapping | 车/UAV数量、真实样本—网络节点映射、轨迹/位置坐标和时间来源 |
| feature_packet_granularity | 每源完整包或空间块、编码格式、完整性与检测器缺失输入规则 |
| per_region_arrival_probability | 各区域概率；现已确定独立抽样，具体数值未定 |
| allowed_total_latency | 从目标观测起算的截止预算，是否按区域/任务变化 |
| feature_preparation_delay | 参数或测量依据，不宣称测试值是真实编码耗时 |
| internal_scheduling_interval | 内部间隔/事件策略与仿真服务采样对齐 |
| rsu_position_and_physical_coverage | RSU位置、天线/高度语义和真实无线覆盖验证 |
| rb_budget_and_sharing | RB预算、跨链路复用/干扰及同链路多包容量分摊 |
| same_time_event_order | 零时刻、观测/任务/准备/反馈/决策/截止同刻顺序；原生步内错位的处理范围 |

正式pending配置全部保留null，不因为测试有数值就视为已决。夹具中的2车/1UAV、2矩形区、p=0.7/0.4、RSU[300,300,25]米、350 ms预算、40 ms准备、30 ms内部间隔、50RB、独占资源、有限事件及整源包，均仅工程测试值。

## 10. 验收与任务4边界

本轮CPU命令：python code/scripts/audit_pi_jwm_scenario_contract_r2_08.py --output <新的本仓库audit目录>。脚本拒绝覆盖已有result.json。最终回执在code/artifacts/audit/pi_jwm_r2_08_scenario_contract_20261027/final_acceptance_v2/result.json，含源码/依据SHA、环境、合成记录和保护检查。旧的首次回执保留，不替换历史结果。

任务4可直接复用六类字段、配置/图校验、独立到达抽样、旧payload ID和byte转换、有效RB容量接口。但多时隙累计、包完成、任务请求/特征/计算/检测分层状态、超时和结果更新均NOT_RUN。确定性服务下的后续工程实现仍需另行授权和显式参数；物理AirFogSim服务仍受SINR域、中断、V2I高度、快衰落及步内时序BLOCKED限制。

合同测试通过不能解释成真实同步、可靠覆盖、实际无线交付、感知精度、世界模型可靠性或完整闭环通过。本轮不读任何原始数据、validation/test/locked_test，不用GPU，不训练，不自动启动任务4。
