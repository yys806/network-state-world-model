# STEP R2-08 — 场景合同冻结与可扩展接口设计

日期：2026-10-27。**结论：已确定的场景/时间规则已落实，合同专项测试通过；未决科研参数未冻结，任务4完整生命周期未实现。**

## 1. Step Goal / 本次目标

完成最新版训练前任务书任务3：形成可校验的场景配置和版本化接口，供任务4后续复用；不重复R2-05B/06/07完整审计，不作传播模型选择，不启动训练或下一任务。

## 2. Definition Basis / 依据与冲突

只读依据为D:/shen/OB/科研/PKU/PIJWM/PIJWM/新场景下训练前任务书.md、同根组会笔记/1028组会.md及本轮研究者明确指令。执行时SHA分别为：

- 任务书：9df6fe58b9fb563f87bded2744cfe2243f67b7f0c57d0dd1e7a24bdbdefafa40。
- 1028：5399a4b4519b7b5b1b6427a057e1837e622208a614fb9ca33c14163906fa3144。

已读取AGENTS、PROJECT_MEMORY_AND_SOURCES、COLLABORATION_GUIDE、权威计划/主文档/推进/证据分层，AI_CONTEXT及指定R2-05B/06/07报告、相关源码/回执。资料最新指纹见final_acceptance_v2/source_identity.json。

Documented Intent：1028前半旧讲稿以图像/车辆为主，UAV是备选中继；末尾补充、最新任务书和本轮指令已明确车/UAV感知源与LiDAR第一阶段。Actual Implementation：已有Mock/入队/单时隙容量接口，尚无完整感知通信闭环。Evidence：上述文件与R2-05B/06/07。Affected Files：本轮合同、配置、隔离校验模块和上下文新增段落。Conflict：旧讲稿与当前有界授权存在范围差异，旧能力不能当作新方法完成。Status：已按研究者新指令落实明确范围；数值和物理模型仍Awaiting Researcher Decision。不修改AGENTS历史文字。

## 3. Initial State / 初始Git及证据

- main，HEAD f63ea82f0a04734f03fc441bbd401973aa5db914；相对本地缓存origin/main有9个未推送提交，未fetch，不能由此断言远端最新状态。
- 起始40项修改/删除/未跟踪文件，详见本轮baseline.json；包括已有AI_CONTEXT/00、CHANGELOG、PROJECT_INDEX改动以及起始已删除task_plan/progress/findings。本轮不恢复、覆盖或清理。
- R2-05B应用入队已通过，但不是交付；R2-06固定train发布无RGB，真实时间/物理配准未建立；R2-07动态合成轨迹、步后观测、有效RB和单位通过，原生内部时序及传播模型仍阻塞。
- 既有知识索引五项生成文件不同步。本轮只运行--check，不覆盖式重建。

本轮目录：code/artifacts/audit/pi_jwm_r2_08_scenario_contract_20261027/。根result.json保留第一次33项测试的阶段回执；**最终有效验收入口为final_acceptance_v2/result.json（35项）**，不能混用初次源码指纹和最终测试。

## 4. Files Involved / Changes / Reuse

| 文件/组件 | 实际变化及边界 |
|---|---|
| docs/contracts/PIJWM_SCENARIO_CONTRACT_V1.md | 冻结规则、三层配置、六类记录、时间/服务守恒、包粒度和多节点映射候选、待决清单 |
| code/scripts/configs/pi_jwm_scenario_v1.schema.json | 冻结范围const、待决null入口、来源/数值/单位结构 |
| 同目录pi_jwm_scenario_v1.pending.json | 全部未决参数null，不能执行，未引入测试默认值 |
| 同目录pi_jwm_scenario_v1.synthetic_test.json | 显式工程夹具，每个测试参数附ENGINEERING_FIXTURE_ONLY说明 |
| 同目录pi_jwm_scenario_interfaces_v1.schema.json | 六类严格JSON记录，未知字段拒绝；统一秒/字节，服务速率明确单位 |
| code/src/pi_jwm/scenario_contract_v1.py | 纯配置/记录/图校验，区域独立抽样及旧接口转换；无队列或生命周期状态推进 |
| code/tests/test_scenario_contract_v1.py | 25项合成专项测试，包含跨周期保留和多种拒绝反例 |
| code/scripts/audit_pi_jwm_scenario_contract_r2_08.py | CPU验收、合成记录、运行环境、源码/依据指纹和保护检查；拒绝覆盖已有result |
| AI_CONTEXT、权威记录、导航、注册表 | 增量记录本轮规则/证据/停止门，不替换旧结论，不更新无关生成索引 |

复用R2-05B DetectionResult/PerceptionPayloadRecord及R2-07 slot_service。未改这些旧文件、第三方AirFogSim、核心世界模型、checkpoint、标签、时间戳或训练协议。未安装依赖；使用现有jsonschema，仅本地Schema引用，无网络解析。

## 5. Results / 场景与时间合同

已落实：车辆/UAV均为信息源，LiDAR协同3D感知，预设UAV轨迹，单逻辑RSU，固定计算/DAG/Route/处理流程，动作只包选择/RB；扩展均关闭。逻辑服务全部区域不证明真实覆盖。

100 ms只约束观测与独立区域到达。每个任务有区域/任务ID、生成时间、目标观测时间、绝对截止、包ID和状态；截止从目标观测起算。测试目标0.1、生成0.12、ready0.14、deadline0.45秒，各自不同。准备耗时来自参数或测量声明，未假装实测。0.2/0.3及0.5秒新观测调用后，旧任务仍保留原状态/ID/截止；0.5已越过截止也不自动超时，因为本轮没有生命周期引擎。

调度序列必须时间前进或消费一个不同的已知有限事件；不允许同刻无事件无限重复。未来实际观测/反馈和标签不能在线引用；已在cutoff前发出的预测可指向未来，预测和实际反馈分开。模型没有运行。

六类记录可以供任务4复用：ObservationRecord、PerceptionTask、FeaturePacket、SchedulingAction、ServiceResult、PerceptionResult。容量诊断delivered_bytes必须null；瞬时容量不能当交付。检测输入要求外部完整输入回执，本轮只校验其ID/长度/时间，不生产该回执、不累计字节、不证明真实输入。测试常数速率和手工回执均标合成来源。

旧payload转换保留ID/byte_count但标记为检测JSON，不是编码特征；R2-07转换要求显式采样时刻，保留瞬时容量假定与物理BLOCKED。原生task_size/Mbit和payload byte尚未绑定；本轮没有冒称解决。

## 6. Validation / 实际命令、输出与限制

运行时设置PYTHONUTF8=1、PYTHONDONTWRITEBYTECODE=1，使用D:/miniconda/python.exe，CPU，无仿真器实例化。

| 命令/检查 | 实际结果 | 能证明与不能证明 |
|---|---|---|
| 初次限定unittest入口 | PASS，33 tests，exit0 | 23新测试+R2-07八项+R2-05B两项；日志attempt1_tests.log，不包含后续补强反例 |
| python code/scripts/audit_pi_jwm_scenario_contract_r2_08.py | PASS_CONTRACT_ONLY，33 tests，exit0 | 第一次机器回执保留，随后补足vendor忽略源码保护和两个反例 |
| 同命令加--output code/artifacts/audit/pi_jwm_r2_08_scenario_contract_20261027/final_acceptance | PASS_CONTRACT_ONLY，35 tests，exit0 | 25新测试+8+2必要回归，不重读真实帧，不重跑完整R1/RGB/五帧坐标/标签合并 |
| 3份新Python内存compile | PASS | 不写pycache，不重新触发历史缓存ACL问题 |
| formal_pending结构/执行分离 | PASS | 结构有效，require_complete执行被BLOCKED_AS_REQUIRED；不是完整科研参数冻结 |
| 源码和历史保护 | PASS，86份 | 含13份既有vendor指纹与73份历史回执，前后SHA一致；不读取原始数据 |
| python code/scripts/build_project_knowledge_index_v1.py --check | BLOCKED，exit1 | tracked_file_inventory、python_dependency_map、artifact_catalog、archive_candidate_registry、registry_summary五项不同步；没有重建或绕过 |
| 上下文/注册/暂存一致性 | 见context_consistency_check.json与commit_scope.json | 核对本轮新增段落、Schema、引用/来源指纹及隔离文件，不是物理验收 |

详细日志、实际命令、哈希和运行环境由本轮目录保存。定向读取中有一次辅助打印遗漏UTF-8导致UnicodeDecodeError，后续显式UTF-8重读成功；不属于测试通过，也未改变文件。索引保留的SyntaxWarning来自现有源码扫描，不修改无关旧代码。

## 7. Expected vs Actual / 是否符合预期

**部分符合。** 任务3要求的已确定规则、版本化配置/记录、未决入口和一致性测试已完成；不能冻结用户尚未决定的参数，也不能解除真实时间、传播模型、正式数据或知识索引门。合同通过不等于PI-JWM完整新方法、真实无线交付或真实检测精度通过。

### 当前待决参数及工程假设

必须由研究者确认：区域数量/边界、真实/合成样本与多节点映射、车辆/UAV数量、RSU位置/真实覆盖、特征包粒度/缺失特征规则、区域概率、总时延预算、准备耗时、内部调度间隔、RB共享/干扰以及零时刻/同刻事件顺序。

独立夹具仅用2车+1UAV、两个矩形区、0.7/0.4到达概率、RSU[300,300,25]米、350 ms总预算、40 ms准备、30 ms调度间隔、50 RB、整源包、全局独占RB和有限事件。它们**不是正式科研决定**。两种粒度及三种多节点映射候选的利弊见正式合同§7–9，本轮未选择真实映射或空间切块。

## 8. Known Issues / 继承阻塞与任务4准入

| 项目 | 本轮状态 | 含义 |
|---|---|---|
| 配置与六类接口、时间/来源/单位拒绝 | PASS | 软件合同可复用 |
| 正式场景参数/多节点映射/特征粒度 | BLOCKED | 等待研究者选择，不从工程夹具推断 |
| V2U4Real真实时间、物理配准、正式标签/评价与检测器 | BLOCKED（继承） | 本轮没有新真实数据证据或感知性能结论 |
| AirFogSim步内错位、SINR域/中断、高度、快衰落和覆盖 | BLOCKED（继承） | 外部时间合同不修复原生内部顺序，不修改物理模型 |
| 原生Mbit服务与payload byte绑定 | BLOCKED | 容量换算已明确，真实任务进度尚未接入 |
| 多时隙累计/完成/超时/请求和计算分层 | NOT_RUN | 属任务4，不因合同里有状态字段而认为已实现 |
| GPU、训练、正式性能、非train访问 | NOT_RUN | 本轮完全不访问数据集，不自动启动 |
| 知识索引及推送 | BLOCKED | 既有五项不同步；保留本地提交，不push |

任务4已经有记录/配置校验、任务抽样、准备门、旧载荷身份和有效RB单时隙单位接口，**具备设计确定性CPU工程闭环的接口基础**。仍需研究者审阅本合同、明确所需参数和新授权，才能实现累计/完成/超时；接真实AirFogSim物理服务另受传播/时序阻塞。不能把任务4尚未开展解释成失败，也不能把它描述为已通过。

## 9. Context Consistency / 项目记录

本轮以明确R2-08段落更新AI_CONTEXT 00–08及相关计划、主文档、推进、模块/架构/数据流、变更、实施/实验/结果导航。06_DECISIONS只记录研究者指令已给定的规则，不写测试参数为决定。新增实验/结果/检索/合同权威条目；任务4独立停止门明确authorization_required=true、auto_start=false。起始删除的根过程文件保持不变，由本地计划表代记计划、进度、发现。

## 10. Git

只提交本轮文件及共享脏文档的R2-08新增前缀；暂存内容按“本轮前缀+基线HEAD版本”构造，不夹带原有未提交修改。原40项内容/删除/未跟踪状态逐项保全；本轮audit文件使用局部字节属性，不规范化历史证据。提交SHA和message见最终Git回报/日志；回执以执行基线HEAD与源文件SHA标识，避免提交内容自引用哈希。

知识索引门未通过，独立本地提交后停止，不push，不强推，不绕hook；本轮未fetch远端。此前9个未推送提交完整保留。

## 11. Next Step / 唯一建议

请研究者审阅本合同§9并确认任务4最小确定性CPU场景所需的参数与包粒度；在确认和另行授权前停止，不开始任务4或正式训练。
