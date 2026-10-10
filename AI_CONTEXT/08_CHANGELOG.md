## 2026-10-10 STEP 6.4J — GPU Pilot 停止门修复与 r27 冻结

正式 GPU 分支采用全局一次性停止状态机，首个 B512 决策须完成 CUDA/FP32/batch16/执行身份、512 次独立 WM 转移、scorer 有限性、winner 或冻结 fallback A、真实 env.step 与 fresh observation 验收；失败立即 BLOCKED 且不启动第二 episode。每次真实 transition 调用前先同步 journal 与 attempted-search 记录，失败尝试计入 16 次总预算。600 秒单次规划使用硬中断，不返回 early-best；10200 秒资源门、预算溢出、重复动作/root、NaN/Inf 和异常均受停止门约束。最终结果从两条 episode 的决策 receipt、journal 与尝试记录独立重建：两条各 8 步为 COMPLETED/0；首个资格 PASS 后合法资源终止或协议允许的 fallback C 为 PARTIAL/2；资格/身份/预算/执行异常为 BLOCKED/1。r27 execution_config_id=`c019c075e35f10560f2a059ac8abf94809c8ef5d18b2f2be82585dec76ba3cf2`。r25 同核心真实 CPU 回归为 4 次 env.step、双 episode/fresh roots PASS；GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行。
## 2026-10-10 STEP 6.4J — GPU Pilot 停止门修复与 r26 冻结

正式 GPU 分支已改为全局一次性停止状态机：首个 B512 CUDA 决策必须完成身份、FP32/batch16、512 次独立 WM 转移、有限 scorer、winner/fallback A、真实 env.step 和 fresh observation；任一失败立即 BLOCKED，第二条 episode 不启动。搜索尝试在 Planner 调用前落盘并计入预算；600 秒单次规划、10200 秒实例资源门、重复 receipt/root、NaN/Inf、反馈/History 不一致均失败停止。最终状态由原始 receipt、journal、attempted-search 和 env.step 记录独立重建：COMPLETED=两条各8步；PARTIAL=资格通过后合法资源停止或冻结 fallback C；BLOCKED=资格/身份/预算/执行异常。r26 protocol 已重新冻结，execution_config_id=`bba80a237795e1fd6e03a6c0047005ae182d60b1178c3468c4f235bc5aced922`。r25 同一生产核心 CPU 回归仍为两 episode、4 次真实 env.step、独立审计 PASS；GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行。
<!-- STEP6.4J CURRENT -->
## 2026-10-10 STEP 6.4J — r23 最终 CPU 一致性与部署验收

r23 已重新冻结源码与协议：`execution_config_id=103bca16ef006869fe4da1af5f3402481c5a549ca6166bcd5e15ba9d6906fbfc`。真实 AirFogSim engineering run 两条冻结 episode 各连续 2 次 `env.step`，合计 4 次；每条 2 个不同 fresh root；4 份 decision receipt 均 EXECUTED，History 动作逐字段与实际 indexed action 对齐，真实 post-step outcome 与 capture 对齐，独立审计 PASS。生产轨迹实际有 2 次合法 Comp 写回；Comm 在该冻结样本中为空，非空 Comm+Comp 的实体/任务/RB/CPU 映射由同一生产 helper 的 focused contract test 单独 PASS，未伪造运行数据。r23 protocol/manifest 已准备强制纳入 Git，部署 verifier 对协议、原始输入、checkpoint、normalization 和源码 SHA 全部 PASS。GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行；完成提交同步后 `READY_FOR_GPU_LAUNCH=true`。

<!-- STEP6.4J CURRENT -->
## 2026-10-09 STEP 6.4J — CPU integration closure

Fixed live runner blockers: explicit `wired_edges` validation, real `uav_id` History reconstruction, and real `required_returned_size` capture. r19 CPU evidence passed with 4 real `env.step` calls; r20 protocol identity was frozen afterward. READY_FOR_GPU_LAUNCH=true; GPU and `locked_test` remain unused.

## 2026-10-09 STEP 6.4J — S-CEM B512 最小真实 GPU Pilot 协议冻结

研究者已授权2条dev_validation源轨迹、每条最多8个决策、search seed6311，最多16次B512；前2轨迹按trajectory_id UTF8 SHA256排序，初始frame取该轨迹静态manifest最早frame，不依据搜索结果。CPU协议与manifest已冻结，Pilot执行身份见 `code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009/00_protocol.json`，execution_config_id=`31667b891be356017220e205e19a33b1e3aeaf7df8fcc8f11af147061303ddf8`。

配置：S-CEM K4/rho0.2、B512、H4、batch16、RTX3080Ti CUDA FP32、Route NOOP、fallback A一次失败即C、同步暂停仿真。CPU manifest/SHA/测试通过；GPU尚未启动。首个真实CUDA决策是资格门并计入16次，不额外搜索；单次>600秒或实例累计达到10200秒停止新规划，身份/源码/checkpoint/预算/scorer/NaN/重复任务异常立即停Pilot。locked_test=false，正式闭环性能仍NOT_STARTED。下一步是远端精确提交、硬件与依赖只读检查，通过后启动首个批准episode。
<!-- END STEP6.4J CURRENT -->

<!-- STEP6.4I CURRENT -->
## 2026-10-09 STEP 6.4I — Pure-search正式闭环CPU准备

STEP_6_4I=PASS（仅CPU正式闭环准备）；SEARCH_METHOD=S-CEM，K=4/rho=0.2；FINAL_CLOSED_LOOP_BUDGET=512，PLANNER_V1_CLOSED_LOOP_B_WM=512，H=4。研究者本轮明确接受6.4H所见Effort质量损失换取计算节约；B1024保留参考，不声称等价/最优，不冻结hybrid预算。Route=EXPLICIT_NOOP_ONLY；fallback=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1（一次A失败即C，无重试）；同步暂停仿真、非实时。

独立CPU真实S-CEM机制：固定TRAIN anchor0003，engineering-only B64/batch1，两次winner第一步真实执行0.6→0.7→0.8s；每次实际64 unique transitions、13个不同可评分H4。第二root由新真实观测/History/后验重新构建，未复用预测root，模型与TRAIN normalization不变。固定TRAIN anchor0041注入NO_SCOREABLE_H4，canonical fallback A经同桥真实执行4.5→4.6s一次；不是搜索性能结论。74项focused CPU tests通过，覆盖空域、缺字段、bridge/setter/step异常、部分写入、重复决策保护、Return支持与Future Target poison。

CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_PROTOCOL_REVIEW；真实CPU链READY，通用episode组件已测试，live CUDA provider=IMPLEMENTED_NOT_GPU_VERIFIED。正式episode来源/数量/seed/长度、指标分母、timeout及baseline仍待研究者批准；GPU_LAUNCH_AUTHORIZED=false，READY_FOR_GPU_LAUNCH=false。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；Stage B=NOT_STARTED/DEFERRED；GPU=NOT_USED；locked_test=false。

证据：docs/implementation_records/STEP_06_4I_PURE_SEARCH_CLOSED_LOOP_PREPARATION.md；docs/contracts/PIJWM_STEP_06_4I_FORMAL_CLOSED_LOOP_PROTOCOL_DRAFT_V1.md；code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2/15_acceptance.json。唯一下一动作：研究者审阅Protocol Draft，批准最小GPU pilot与首个live CUDA资格检查。本轮停止。以下较早内容均为历史时点，旧pending不覆盖本轮明确接受预算决定。

本次新增必要执行/失败记录/评价草案接口，完成S-CEM真实反馈CPU闭合与73tests；Context Consistency Check覆盖00–08，当前source/config/artifact为Source of truth。
<!-- END STEP6.4I CURRENT -->

<!-- STEP6.4H ACCEPTED CURRENT -->
## 2026-10-09 STEP 6.4H — S-CEM B512 预算资格完成

STEP_6_4H=PASS；SEARCH_METHOD=S-CEM（K=4,rho=0.2，仅Planner v1 pure-search骨架）。新B512 320/320与修复域6.4G A的S-CEM B1024父320只读严格配对，本地原始结果独立重算、ZIP/逐文件SHA与持久D:备份PASS。两预算均160/320=50% H4可评分、集合完全相同，32/64困难起点保留；前三项N_DDL/A_DDL/J_Delay首差劣化0。B512对B1024为11胜/233平/76负，六类0/0/11/76/73/160，64-anchor整组抽样95% CI=[-0.290625,-0.11875]。差异仅在J_Effort首差：B512更好11、更差76；J_Burden首差0，双方可评分全等73。资格通过不表示整体目标等价；完整字典序比较显示Effort损失。

S_CEM_B512_QUALIFICATION=PASS；RECOMMENDED_CLOSED_LOOP_B_WM=512；FINAL_CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING。B512实际转移163840（名义相同）；不同可评分候选30209 vs 72587。单次内部搜索mean/median为74.548/51.416s vs175.249/126.359s，节省57.46%/59.31%。矩阵总墙钟42613.116s（11h50m12s），含内部搜索23855.506s、加载/setup10899.799s及其余核验/持久化开销，不能用总墙钟冒充单次搜索耗时。

future Return-birth fixed-support限制保留：B512/B1024边界49845/162239，grammar dead-end19882/46621，scorer exception/inconsistency均0。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_RESEARCHER_BUDGET_DECISION；已有闭环机制与fallback不改变，但FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED、HYBRID_PLANNER=NOT_FROZEN。6.4G Phase B=NOT_STARTED_BY_CONDITIONAL_STOP，原三方法Stage B=NOT_STARTED/DEFERRED；locked_test=false。GPU=RUNNER_STOPPED_NO_FURTHER_USE：原3080Ti CUDA FP32 batch16单runner自然结束，全部备份通过后已告知研究者可以关机，实例电源状态未由本任务确认。唯一下一动作：研究者审阅Effort损失与时间节约，决定最终pure-search闭环预算；不自动启动闭环或其他实验。

证据：code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2/{acceptance.json,local_backup_acceptance.json,qualification.json,summary.json,paired.json,objective_components.json,runtime_tradeoff.json,matrix_runtime.json,inventory.json,archive.json}；以下6.4G及更早内容为历史状态。
<!-- END STEP6.4H ACCEPTED CURRENT -->

<!-- STEP6.4G CURRENT -->
## 2026-10-08 STEP 6.4G — 修复域 Phase A 验收与条件硬停止

STEP_6_4G=CONDITIONAL_STOP_AFTER_PHASE_A；Phase T=PASS：新 TRAIN 768/768，S-CEM=(K4,rho0.2)，MH-CEM=(K3,rho0.1)。Phase A=PASS：修复域新 Validation 960/960，三方法各160/320 H4可评分；名义与实际一步转移均983040，scorer exception/inconsistency=0，独立统计、原始结果、ZIP与本地D:SHA验收PASS。冻结选法规则得 SEARCH_METHOD_REQUALIFIED=S-CEM，仅为 Planner v1 pure-search backbone，不是最终 hybrid planner。S-CEM对HRS 110胜/198平/12负，anchor-cluster 95% CI=[0.215625,0.403125]；MH-CEM对HRS 108/199/13，CI=[0.203125,0.39375]；MH-CEM对S-CEM 33/249/38，CI=[-0.065625,0.03125]。因此两个CEM都通过对HRS优势门，MH未通过对S优势门，按原规则选择S；不能据此声称S已被证明显著优于MH。

研究者预注册条件明确要求：新A若未选MH，STOP且不得运行Phase B。故 Phase B=NOT_STARTED_BY_CONDITIONAL_STOP，旧完整Stage B=NOT_STARTED/DEFERRED；新S-CEM闭环预算=RESEARCHER_DECISION_PENDING。旧MH-CEM B512只属historical-under-pre-6.4F-domain的working决定，不移植给新S。FORMAL_SEARCH_REQUALIFICATION=PENDING_RESEARCHER_BUDGET_DECISION；CLOSED_LOOP_PRE_FORMAL_READINESS=BLOCKED_BY_BUDGET_REQUALIFICATION；READY_FOR_FORMAL_CLOSED_LOOP_PROTOCOL=false。FINAL_FALLBACK_POLICY=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1（一次canonical A失败即C，无重试）。32/64 anchors在三方法五seed均不可评分；future Return-birth fixed-support限制保留。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN，locked_test=false。唯一下一动作：研究者审阅新S-CEM选法结果并决定其闭环预算验证方案；本Step停止，不启动B或其他实验。

权威证据：code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004/A/{acceptance.json,local_backup_acceptance.json,selected_method.json,paired.json,summary.json,inventory.json,archive.json}；新结果标签=repaired-domain formal evidence。
<!-- END STEP6.4G CURRENT -->

以下是历史记录；旧调参/选法/预算结果只属于historical-under-pre-6.4F-domain。

---

以下是历史记录；旧调参/选法/预算结果只属于historical-under-pre-6.4F-domain。

---

以下是历史记录；旧调参/选法/预算结果只属于historical-under-pre-6.4F-domain。

---

## 2026-10-04 STEP 6.4F（当前）

STEP_6_4F=PASS；Comm当前资格修复与真实一步机制通过。共同谓词：Task present、offloading/transmitting、未完成，再与唯一合法活跃无线Flow绑定相交。Flow对象不删除，Flow存在不等于可执行。TRAIN4416/Validation1104完整静态前后审计与输入SHA一致；32/64搜索anchors中28/57的根候选域改变。真实TRAIN anchor0041：Task_24（offloading），UAV_1→vehicle_7，RB0/start0/width1，setter与真实无线传输事件一致，4.5→4.6s一次决策步；候选A独立episode、模拟NO_SCOREABLE_H4、同一bridge、无Comm的canonical合法动作亦4.5→4.6s。Future Target poison不改变当前输入/动作。FORMAL_SEARCH_EVIDENCE_REUSE=TARGETED_REQUALIFICATION_REQUIRED：7个Validation起点有覆盖全部H1–H4分支的规则不变量证明，另外57个旧raw无完整候选/预测轨迹，不足以证明winner/ranking不变。最小建议重新验证57×5×3方法B1024=855及57×5 MH B512=285，共1140cases；仅建议，未授权、未执行。旧TRAIN调参28/32根域受影响，不声称历史调参验证新域；K4/rho0.1按本轮研究者指令保持，不自动retune。B512_EVIDENCE_REQUALIFICATION_REQUIRED=true；PLANNER_V1_CLOSED_LOOP_B_WM=512作为研究者working-budget保留。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_FALLBACK_DECISION仅表示执行机制就绪，不解除搜索证据重新验证门。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING；SEARCH_METHOD=MH-CEM pure-search骨架，非最终hybrid。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED / DEFERRED；GPU=NOT_USED；locked_test=false。唯一下一动作是研究者审阅旧证据重新验证范围及fallback决定，不自动开跑。

证据：`docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/09_final_acceptance_receipt.json`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/07_formal_search_evidence_reuse_assessment.json`。历史6.4E BLOCKED及旧搜索accepted状态仅在旧定义内保留。

以下为历史状态，旧PASS不代表新资格定义下已经重新验证。

---

## 2026-10-04 STEP 6.4E 预算决定与真实执行阻塞（当前）

STEP_6_4E=BLOCKED，CLOSED_LOOP_PRE_FORMAL_READINESS=BLOCKED。研究者正式冻结 PLANNER_V1_CLOSED_LOOP_B_WM=512，仅 MH-CEM K4/rho0.1 pure-search；B1024保留高预算参考，不声称等价/最优，6.4D后两项目标质量损失仍记录。统一fallback接口CPU合同就绪，33 focused CPU tests通过。真实TRAIN固定anchor0041、t=4.5s的非空Comm Task_1/RB0被执行前安全门拒绝：Domain仍绑定已failed的Task_1/Task_6，无本轮setter或env.step，第二场景未启动，无换动作/样本或科学失败重试。候选A可确定性准备合法当前动作但本轮真实执行未取得；候选B缺独立current behavior offer provider；C无动作终止及空域/缺字段/bridge/setter负路径通过。当前Flow支持与真实task执行条件不一致，Status=Awaiting Researcher Decision；不改Domain/Objective/Return。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN；原Stage B=NOT_STARTED、DEFERRED / NOT_REQUIRED_FOR_CURRENT_MAINLINE；GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并决定一致性修复与复验范围；不自动开跑。

证据：`code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/10_pre_formal_readiness_receipt.json`；`docs/implementation_records/STEP_06_4E_BUDGET_COMM_FALLBACK_CLOSURE.md`。以下为历史状态，较早预算待决定或机制PASS不覆盖本轮阻塞。

---

## 2026-10-04 STEP 6.4D 全量B512最终验收（当前）

STEP_6_4D_FULL_COHORT_B512=PASS。完整64 anchors×5 seeds，B512覆盖320/320（48原6.4C结果按SHA引用、272新增），原Stage A B1024父结果320/320身份与SHA通过且未重跑。新增名义/实际独立一步转移139264，完整B512实际163840，B1024参考327680。两预算均160/320可评分H4（50%），六类配对0/0/30/109/21/160合计320，B512对B1024为30胜/181平/109负；64anchor×5seed cluster bootstrap95% CI=[-0.3375,-0.15625]。PRIMARY_OBJECTIVE_DEGRADATION_COUNT=0，前三项N_DDL/A_DDL/J_Delay在双方可评分160对中均相同；首差J_Burden为B512更好1/B1024更好8，J_Effort为29/101，全等21。独特可评分候选15668/33505，B512减少53.24%；内部搜索mean57.641/119.746秒、median43.564/90.506秒，分别节省51.863%/51.866%。两预算困难anchor均32，交集32、差集0；future Return-birth固定支持限制保留，scorer exception/inconsistency=0。新增272总墙钟24692.314秒（6h51m32s），加载/准备开销单列。640raw独立核验、本地D:持久备份、新raw ZIP/SHA归档PASS；旧raw未复制/修改。GPU搜索已硬停止，研究者已在本聊天确认手动关机（Codex未取得UI关机核验）。SEARCH_METHOD仍MH-CEM纯搜索骨架，非最终hybrid冻结。证据支持将B512作为节省计算的预算候选，但完整词典序Objective有损失，不能声称等价、最优或真实闭环性能。CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。唯一下一动作研究者审阅完整预算取舍并决定预算，不自动执行后续实验。

证据：`code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003/17_independent_acceptance_receipt.json`、`19_archive_acceptance_receipt.json`、`20_formal_runtime_receipt.json`、`21_instance_shutdown_receipt.json`。执行协议commit9b7f50e58d410e012d9b27ed04661e8c88191f2a；分析/文档收口属于后续独立commit，不覆盖历史运行source。

以下均为历史gate；较早RUNNING/未启动表述不是当前状态。

---

## STEP 6.4D GPU运行（当前）

STEP_6_4D_FULL_COHORT_B512=RUNNING（尚未最终验收）。协议提交9b7f50e58d410e012d9b27ed04661e8c88191f2a已push，服务器fetch/ff到该精确commit，tracked clean及source/config/checkpoint/320父结果/48复用SHA核验PASS后才启动。执行身份1c3cd986dbda4559e94fa37013b2d70e96adf85911c44e88cc22cfe3f99fd843；RTX3080Ti CUDA FP32 batch16。正式start2026-10-03T13:38:52.692909+00:00；新增272 B512，旧48按SHA引用，B1024不重跑。进度以09_runtime_status及raw为准；已启用只读SSH/SFTP身份监控、逐case SHA核验与本地D:原子备份，无中途质量结论。监控脚本属于独立部署工具，不进入冻结科学source closure，不改科学运行源码。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false；Return-birth固定支持限制保留。唯一下一动作继续监控至272新case完成，再独立验收与Git收口；不启动B256/环境动作/其他实验。

证据：`code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003/08_launch_acceptance_receipt.json`。以下为历史gate，当前状态以本段和runtime/raw为准。

---

## STEP 6.4D protocol-freeze gate（当前）

STEP 6.4D 仅完整64 anchors×5 seeds的MH-CEM K4/rho0.1、B512扩样。320份B1024父结果身份/归档SHA通过；6.4C既有48份B512全部冻结源码字节、算法/运行身份和参数摘要一致，REUSE_VALID=true，按SHA引用、不复制不重跑。只新增272 cases，名义139264转移。Protocol/manifest/config已冻结，CPU预检PASS；正式GPU尚未启动，覆盖48/320，不是最终PASS。GPU3080Ti只读可达空闲；须先commit+push协议，服务器fetch并fast-forward精确协议commit、tracked clean/source/config/checkpoint全PASS才启动。首差分量N_DDL/A_DDL/J_Delay/J_Burden/J_Effort/all_equal及主目标损失预注册；64 anchor clusters×5seed bootstrap10000/seed6316。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false；Return-birth固定支持限制保留。唯一下一动作提交并同步协议后执行剩余B512，不运行B256/B1024/环境动作。

证据：`docs/implementation_records/STEP_06_4D_FULL_COHORT_MH_B512.md`；`code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003/07_cpu_preflight_receipt.json`。以下为历史记录。

---

## 2026-10-03 STEP 6.4C 预算校准验收（当前）

STEP_6_4C_BUDGET_CALIBRATION=PASS。静态预注册16 anchors×3 seeds，MH-CEM K4/rho0.1，B256/B512新增96/96，48份配对B1024身份SHA通过且未重跑；新GPU名义/实际独立一步转移36864。三预算各30/48 H4可评分（62.5%）。256vs1024=1胜/19平/28负；512vs1024=7/22/19；256vs512=1/21/26；每组六类合计48。逐case去重可评分候选1263/2850/6060；内部搜索median20.133/42.342/88.436秒，mean31.140/62.955/134.232秒，低预算平均耗时减少76.802%/53.100%。总墙钟7800.804秒（2h10m），其中内部搜索4516.527秒，加载准备等3284.277秒另列。独立raw验收、144文件inventory及ZIP/SHA PASS，原raw目录保留在D:。6 anchors×3 seeds三预算均无可评分H4；future Return-birth fixed-support limitation保留；scorer exception=0。前三项最佳objective逐对一致，排序差异在J_Burden/J_Effort（额外分项只作exploratory diagnostic）。建议另预注册64×5 MH-only扩样，属于NEW RESEARCH PROPOSAL，不自动运行。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN，Stage B=NOT_STARTED，locked_test=false。GPU搜索和监控已结束。唯一下一动作研究者审阅并决定预算或是否另授权扩样。

证据：`code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003/24_independent_raw_acceptance_receipt.json`、`26_archive_acceptance_receipt.json`、`27_tradeoff_and_return_boundary_observation.json`。以下为历史gate，其中RUNNING不是当前状态。

---

## 2026-10-03 STEP 6.4C 已正式启动（当前）

STEP_6_4C_BUDGET_CALIBRATION=RUNNING（未最终验收）。仅静态分层16 anchors×seeds6311/6312/6313，MH-CEM K4/rho0.1，B256→B512共96 cases；48份B1024参考SHA身份PASS，未重跑。启动提交c3c1f34e4b41b690333c77e8333265a01c69b474，执行身份0973bbf43d3531175b1c5a2bba62bd360833acddebda277511d6af13df37204a；RTX3080Ti CUDA FP32 batch16。旧07从未运行，07b绑定Git LF规范字节，tracked clean/source identity gate同时PASS。逐case原子写入并在本地D:按SHA备份；当前进度以09_runtime_status.json及原始结果为准，部分结果不作预算结论。CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；SEARCH_METHOD仍MH纯搜索骨架；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。唯一下一动作继续监控与备份至96完成并独立验收；不启动其他实验。

启动证据：`code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003/23_launch_acceptance_receipt.json`。以下为历史gate，不能当作当前状态。

---

## 2026-10-03 STEP 6.4C 预算校准预注册（当前）

STEP 6.4C 已获授权并完成预算校准预注册和 CPU 预检。只用运行前静态分层：7层配额3/3/2/2/2/2/2，SHA256(sample_id)排序选16 anchors，seeds6311/6312/6313，MH-CEM K4/rho0.1，新增B256→B512共96 cases；48份B1024父级raw身份与归档SHA通过，未重跑。3080Ti可达且checkpoint正确，远端仍是旧StageA代码，必须同步本轮已提交源码并重新核验身份后才能启动。当前0/96，不是校准结果PASS。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。

证据：`docs/implementation_records/STEP_06_4C_MH_BUDGET_CALIBRATION.md`；`code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003/08_cpu_preflight_receipt.json`。

---

## 2026-10-03 STEP 6.4B 最小闭环真实反馈验收（当前）

STEP_6_4B=PASS，TWO_CYCLE_REAL_FEEDBACK_SMOKE=PASS，CLOSED_LOOP_MECHANISM_READINESS=PASS。固定TRAIN fixture、MH-CEM K4/rho0.1、CPU FP32 batch1、B64：仅执行首轮winner第一动作，仿真0.6→0.7秒，真实新观测重建state/graph/current posterior并完成第二次规划；下一轮root不是上一轮预测。两轮各64独特转移（合计128）；另保留一次动作执行前回执异常的64次尝试，实际总计192。live tensor/deadline、旧搜索不变性、防未来泄漏及参数/归一化不变全部PASS。SEARCH_METHOD=MH-CEM仍仅pure-search backbone；READY_FOR_BUDGET_CALIBRATION=true只是机制就绪，不是运行授权。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN；future Return-birth fixed-support limitation保持。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并单独授权预算校准协议，不自动运行。

Context Consistency Check：当前状态/研究边界/架构/数据流/模块/实验/研究者决定/已知限制/变更均已同步。历史6.4A BLOCKED不代表当前最小机制状态；正式Stage A/TRAIN provenance未覆盖。新独立CPU执行身份4264d1b679173a1e518ee8276715892ca1760b28d1c2ef08f56ea28d32098bc0。

记录：`docs/implementation_records/STEP_06_4B_MINIMAL_CLOSED_LOOP_TWO_CYCLE_SMOKE.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/13_final_acceptance.json`；合同：`docs/contracts/PIJWM_STEP_06_4B_LIVE_CLOSED_LOOP_SMOKE_V1.md`。以下为历史记录，其中“当前”仅表示记录时点。

---

## 2026-10-03 STEP 6.4A 闭环前就绪审计（当前）

STEP 6.4A审计完成（审计证据PASS），CLOSED_LOOP_READINESS=BLOCKED。SEARCH_METHOD=MH-CEM仍只为pure-search backbone，非最终hybrid。局部链路7 IMPLEMENTED/3 PARTIAL/5 MISSING；缺live history-only输入/belief/sidecar、winner动作序列/首动作及真实ID执行桥、研究者fallback和latency决定、两轮真实反馈MH证据。MH B1024搜索median90.506/P90 204.916/P95 284.152/max421.701秒；仿真slot0.1秒不是墙钟deadline，DECISION_LATENCY_REQUIREMENT=UNRESOLVED，尚无在线执行证据。Stage B用途由研究者冻结为online-budget trade-off，不能重选算法；A约30.46–41.85h、B NEW PROPOSAL约10.26–14.06h、C NEW PROPOSAL约1.54–2.11h，仅成本情景。4090_MIGRATION_NOT_YET_JUSTIFIED。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作研究者确定fallback及decision latency模式，再授权最小接口集成；不自动运行实验。

新增01–07机器审计、用途合同、闭环readiness record与CPU tests；同步authority/process/Context/navigation/registry。

记录：`docs/implementation_records/STEP_06_4A_CLOSED_LOOP_READINESS_STAGE_B_PURPOSE_FREEZE.md`；机器审计：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/06_go_no_go_receipt.json`。下方为历史状态，不能用Stage A PASS替代闭环readiness。

---

## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

新增CPU独立验收脚本、拒绝门测试、正式统计/runtime/inventory/archive receipts、Stage A结果registry数值核验；同步权威/Context/index。focused34/34、index7/7、compileall通过；最终knowledge-index/diff/Context以收口收据16为准。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-02 17:23 Stage A 过半运行快照（非最终验收）

515/960（53.6%）完成结果已独立验身份并逐文件SHA备份至本机D:，实际独特一步转移527,360；覆盖35锚点，315个已完成case找到可评分H4。未发现重复/错误执行身份、source drift、NaN/Inf或scorer exception。GPU仍为RTX3080Ti/CUDA/FP32/batch16，原runner持续运行，约21.9小时、23.53 cases/hour，预计还需约19小时，仅资源规划估算。Stage A=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；Return-birth限制保留，无新增阻塞。唯一下一动作继续监控/备份至960完成，再独立统计验收并停止。部分有序结果不得用于提前选择方法或宣称优势。

证据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/snapshots/health_20261002T092315Z.json`。下方为较早快照。

## 2026-10-01 STEP 6.3D FORMAL VALIDATION STAGE A（当前运行）

研究者已单独授权并启动 Formal Validation Stage A（仅 B1024）：64锚点×5 seeds×3方法=960 cases，名义预算983,040。2026-10-01北京时间19:26在RTX3080Ti/CUDA/FP32/batch16启动，启动前源码/768 TRAIN/771文件SHA/64锚点/checkpoint/执行身份全部PASS，Validation起点0。首批2个case已通过身份检查并持久备份；这是运行快照，不是最终效果结论。STEP_6_3D_VALIDATION_STAGE_A=RUNNING，VALIDATION_COMPARISON=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；future Return-birth限制保留。唯一下一动作是监控并备份Stage A，960完成后独立验收并停止。选择范围仅Planner v1 structured-search backbone/pure-search baseline，不是最终hybrid planner冻结。

记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；启动前机器收据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/01_launch_preflight_receipt.json`；动态监控与本地备份在该运行目录。下方均为较早阶段快照，关于未授权/Validation0的表述不能代表当前运行。

## 2026-10-01 STEP 6.3D PRE-VALIDATION BLOCKER CLOSURE（当前）

PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。

入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md`；机器验收：`code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。下方为较早快照，原B1/B2阻塞已关闭。

## 2026-10-01 STEP 6.3D PRE-VALIDATION AUDIT（当前）

STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。

记录：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md`；机器验收：`code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。下方均为较早记录。

# AI_CONTEXT 重要变更

## 2026-10-01 STEP 6.3D FORMAL TRAIN TUNING CLOSURE

只读下载并 SHA 验证 RTX 3080 Ti 的 768 raw TRAIN solves、日志和两份选参收据；独立重算两种 CEM 的四组排名，生成逐组/逐锚点诊断、接受收据和本机 ZIP/tar 备份。正式状态为 TRAIN 选参 PASS、两种配置 `(4,0.1)`，Validation/方法选择未开始，future Return birth 边界及 `locked_test=false` 保持。同步 00/05/07/08；01 研究背景、02/03/04 架构/数据流/模块实现、06 研究者决定无变化。

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH

修正 3/5 个零 cohort TRAIN/Validation 锚点并精确重放 8 个 deadline sidecar；新增 TRAIN-only 32×64 H4 诊断与 CPU batch 1/4/8/16 一步/四步等价、吞吐及 SHA 收据。Patch PASS，H4 readiness 待研究者基于 16/16 分布判断，正式搜索比较未启动。更新 00/02/03/04/05/07/08；研究决定 06 未改。

## 2026-09-29 STEP 6.3D 进行中

研究者授权 HRS/S-CEM/MH-CEM 固定预算 H4 方法比较。新增分层锚点清单、只读因果 deadline 重放、共享五层提议分布与 CPU H4 搜索/调参/成对 bootstrap 代码；96/96 目标侧状态就绪，9/9 focused tests 与 synthetic exact oracle 通过。正式 TRAIN/Validation 矩阵尚未完成，方法未选；单锚点 CPU 探针不可作为方法性能结论。无 GPU、`locked_test`、训练或闭环。

## 2026-09-29 STEP 6.3B/6.3C-PATCH

研究者将 Comm 全 eligible Task 覆盖改为当前已有无线 Flow 唯一绑定的 Task 中按 TRAIN 结构条件数量选子集。代码、合同、TRAIN 支持目录、4416 H1 raw/projected/排除行审计和 4416/1104 静态域收据同步；余下结构/Comp/joint 失败保留。H1–H4 非空 Comm 机制路径继续与冻结模型顺序推演等价。无搜索器、排名、GPU、训练、`locked_test` 或闭环。

## 2026-09-29 STEP 6.3C

新增 `CandidateDomain`、`SearchNode`、one-step frozen rollout primitive、lazy/symbolic cardinality and budget/cache receipts. Full TRAIN/Validation static audits and recursive equivalence smoke passed. `STEP_6_3C=PASS`; no optimizer or closed-loop execution.

## 2026-09-29 STEP 6.3B

新增 Formal TRAIN 结构支持目录、search-independent 候选语法/准入器、focused CPU tests 与合同；修正 6.0A 同 Task 多 Comm row 拒绝。研究者冻结 shared-profile Mob、已见联合结构正式准入及时间诊断标签；Objective/Route no-op/checkpoint 不变。无优化器或闭环执行。

## 2026-09-28 STEP 6.3A

Completed CPU-only Formal TRAIN support evidence and security preflight. Corrected
the provisional Comp blocker by reconstructing the causal base with the existing
CPU rule and confirming global alpha `{0.5,0.75,1.0}`. Measured family/joint and
formal-index H1–H4 support; independent factorization is `NOT_SUPPORTED`. No
candidate method or optimizer was selected or run. Full unittest discovery
failed historical cases and invoked synthetic CPU trainer tests outside the
audit scope; no Formal training/checkpoint/dataset write occurred. See the
STEP 6.3A record for the exact verification limits.

## 2026-09-28 STEP 6.2B-PATCH

研究者冻结 Planner v1 Route `EXPLICIT_NOOP_ONLY`，6.0C gate 统一拒绝非空 Route；底层 Route/多跳与 learned encoder 保留。新 CPU receipts 验证负索引缺席编译、有效 Route 自由度为 NONE、Comm 分母 50、冻结 checkpoint 无 Route H4 集成；`STEP_6_2B=PASS` 限于 scorer 合同。原 6.2B blocked 观察保留为历史。无候选方法、闭环、baseline、GPU、`locked_test` 或训练。

## 2026-09-28 STEP 6.2B（历史验收阻塞）

新增五项 scorer、严格字典序、CPU 合同与冻结 checkpoint 单 anchor 机制证据；修正 Comm effort RB 分母。bounded Route audit 发现 pending Flow admission 与同路径 Host 更新的合同冲突，`STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`。没有新的研究者 Objective 决定，未开始候选方法、baseline、GPU、locked_test 或闭环。

## 2026-09-28 STEP 6.2A-CLOSURE

接受 no-retrain salvage，冻结 Planner v1 single-hop Route support，更新 STEP 6.2B scorer implementation readiness；保留 multi-hop code capability 与零 Formal coverage 边界。patched full validation、scorer、ranking、baseline、GPU、locked_test、closed loop均未执行。

## 2026-09-28 STEP 6.1

- 新增冻结 `best.pt` 的 CPU 候选递归 rollout：相同当前 belief、四类当前因果动作探针、H4 每步按预测 state 编译、逐 horizon 指纹/潜变量/decoder 诊断、随机配对复现与串行/批量计时。状态限定为 `TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT=PASS` 的机制证据；MPC objective、候选方法选择和闭环仍未开始。

## 2026-09-28 STEP 5.6C

- 本地 CPU 核验正式 run 5520/5520、五次完整 prior-only validation、严格最终 `argmin L_Val=0.0743133878`；best/latest 都是 step 5520 且 428 个模型张量相同。CPU bounded H1–H4 推理、SHA manifest 和机器回执通过。`STEP 5.6B=COMPLETE`、`FORMAL_BEST_CHECKPOINT=FROZEN`；仍无 baseline、locked_test、Planner rollout 或性能声明。

## 2026-09-26 STEP 6.0C

- 研究者冻结 Planner v1 静态每时隙 CPU 预算与六档 UAV 核心域；新增独立域模块、显式 HOLD fallback、控制侧 H4 验证及原 adapter 11 tensor 等价测试。保留 6.0B 仿真器事实；无训练/模型候选 rollout/GPU/`locked_test`。

## 2026-09-26 STEP 6.0B

- 只读审计本地 AirFogSim CPU/UAV 实际执行源码、当前配置及 Formal Raw 行为支持。CPU verdict=`STATIC_CAPACITY_ONLY`；UAV 无 simulator hard numeric bound，两个候选 UNKNOWN 保留。记录本地 AirFogSim 缺独立 Git 元数据的来源身份限制。未改 Candidate/World Model/训练代码，未联系 5.6B。

## 2026-09-26 STEP 6.0A

- 增加 CPU 静态统一 Candidate 合同、三态约束、当前固定支持、四动作正式 adapter wrapper、Search/Learned/Hybrid 接口、暖启动及去重池；合成 fixture 做 11 字段精确等价。5.6B 远端独立、未联系；无 World Model 候选 rollout、GPU、proposal training 或性能结论。

## 2026-09-25 STEP 5.6B 中途过程图

- 活跃正式 run 的 2646-step 日志只读快照产出三组过程图及源日志 SHA 凭证；两次完整验证分别在 1104/2208 步。训练未改，尚无最终性能结论；当前状态以远端 heartbeat 为准。

## 2026-09-24 STEP 5.6A-CONFIG-FREEZE

- 将研究者批准的 Formal Training Config v1 落入独立源、JSON 和 freeze receipt：seed 5601、batch 8、552 steps/epoch、10 epochs/5520 steps、Stage 1 552、curriculum 起点 0/1104/2208、KL warmup 1104、validation interval 1104、checkpoint interval 552、patience 3、FP32。
- 修正 validation `available_sample_count` 的 batch/merge bookkeeping：真实 1104 validation windows 的 H1–H4 corrected counts 均为 1104；官方 numerator/count、`L_Val` 和旧 GPU receipt 均未修改。无 GPU rerun。
- 当前状态：`FORMAL_TRAINING_CONFIG=FROZEN`、`FORMAL_TRAINING_READINESS=READY_TO_START`，但 `formal_training=false`、`gpu_training_verified=false`、`locked_test_accessed=false`。

## 2026-09-24 STEP 5.6A

- 正式数据 RTX 4090 H4 CUDA smoke 与全部 1104 validation 窗口的 prior-only GPU 遍历通过；完整唯一性、分 family/horizon raw metrics 和数值容差 checkpoint 重载记录在 STEP 5.6A 机器凭证中。
- 正式训练数值配置待研究者决定；`FORMAL_TRAINING_READINESS=BLOCKED_BY_CONFIG_DECISION`。没有正式训练、locked_test、baseline、Planner 或性能结论。

## 2026-09-23 STEP 5.5-PATCH

- 增加正式 60-shard lazy/batch Trainer 消费路径与 train-only encoder 统计；将原 1+1 runtime mini smoke 从 full-shard 证据中分离。
- Future Return birth 改由真实 target typed Flow 与 current support 比较；全量审计 8828 次 unsupported/fixed-support，旧字段零计数失效。生命周期修复 213 次经 Raw 和同对象 fixture 审计。
- CPU only；没有 GPU、正式训练、locked_test、baseline、Planner 或性能声明。

## 2026-09-23 STEP 5.5

- 建成并接受 Formal Dataset v1：60×96 real transitions、H2/L4、48/12 trajectory split、5520 windows、五类 hashed package 与 train-only normalization。
- 四动作真实 coverage、deterministic rebuild、negative hash fixtures 和 CPU H=4 trainer/checkpoint smoke 通过。
- readiness 更新为 Training Stack PASS、Formal Dataset READY、GPU Codepath PREPARED、Formal Training BLOCKED；未执行 GPU/formal training/locked test。

## 2026-09-22 STEP 5.3-PATCH

- 修正 5.3 Phase A 真正 pre/post、Phase C 复用 5.2 的 H=1→H=2 + KL warm-up、raw-unit metric bridge 与独立 fresh-run reproducibility。
- receipt 现在分离 `LEARNING_SIGNAL_GO` 与 `TINY_OVERFIT_NO_GO`；CSI scale audit 与分阶段 gradient/leakage/resume evidence 已生成。
- 未进入 STEP 5.4、GPU、正式训练或 locked_test。

## 2026-09-22 STEP 5.3D

- 完成固定 `[0,1]`、CPU 200-step CSI scale/optimization diagnosis；actual/expected raw bridge MSE exact match。
- Baseline 未通过 stronger tiny-overfit；mean-bias 仅诊断对照达到 H1/H2 normalized CSI MSE `0.1405/0.1633`。
- 记录为 raw-output initialization/conditioning bottleneck observation；不自动采用任何正式初始化或 decoder bridge 方案，等待研究者决定。

## 2026-09-22 STEP 5.3E

- 正式接入 raw CSI decoder train-only mean bias initialization；checkpoint 保存/校验初始化 contract 与 normalization provenance。
- 固定 CPU tiny subset `[0,1]` 的 H1/H2 Motion/CSI stronger gate 通过；receipt=`FORMALIZATION_PASS`、`TINY_OVERFIT_GO`。不开放 GPU、formal training、locked_test 或性能声明。

## 2026-09-22 STEP 5.3

- 新增 bounded CPU tiny-data preflight runner、focused tests 和 machine-readable evidence。
- 固定 1-sample/2-sample dev_train subset，完成 Stage 1 双 family、prior H1/H2、2-sample 1→2、module learning-signal、KL/normalization diagnostics、validation prior-only 与 deterministic checkpoint-resume。
- 预注册 0.5% development gate 全部通过，STEP 5.3=`GO`；仍未启动 GPU、full training、formal Dataset、Planner、baseline 或 locked_test。

## 2026-09-22 STEP 5.2

- 新增 `step5_2_training_loop_v1.py`、CPU smoke script、focused tests 和实现记录；Stage 1 posterior teacher、Stage 2 prior-only recursive rollout、`1→2→4` curriculum、KL schedule、joint optimizer audit、prior-only validation、`L_Val` selector、checkpoint/resume 已接入。
- 8/4 unified non-locked development smoke 的 5.2-PATCH 26/26 receipt checks 和 focused 12/12 通过；这是 training-loop implementation evidence，不是 tiny-data overfit、full training、GPU 或性能结果。5.3 preflight 后续已有独立记录。
- 更新当前状态、实验、已知问题、模块地图、tracker 与 Definition 05 contract；Route/Comp non-empty coverage=0 继续作为 future formal training/data gate，未进入 STEP 5.3。

## 2026-09-22 STEP 5.1D

- 新增 unified Tensor full-package roundtrip，补齐 sample IDs 与 base Step 3.3 validation provenance。
- 从同一 12-sample bundle 重建 4.3A/4.3B/4.4，并完成 paired recursive prior/posterior/decoder、Loss/KL/raw-unit metric/gradient receipt；当前 5.1D-PATCH receipt 47/47 checks 与 deterministic rebuild 通过。
- 当时仍不训练、不用 GPU、不访问 formal dataset/locked_test；该边界随后由研究者授权的 STEP 5.2 CPU development loop 更新。

## 2026-09-21 STEP 5.1A-PATCH

- 修正 multi-horizon Motion：horizon 2+ 从 anchor-to-future cumulative delta 改为 adjacent-frame local one-step delta，缺失上一 future position component 时下一步对应 mask=false。
- Motion tensor 改为 current `input_entity_index["physical"]` 固定槽位；future permutation/birth/disappearance/target-index 不再改变 supervision slots。
- CSI 增加 current model relation slot、identity、endpoint input slots、type 与 RB 的机器绑定；patched 12-sample non-locked receipt `passed=true`。

## 2026-09-21 STEP 5.1A

- 新增 Future Motion/CSI additive target contract、frozen train-only normalization、stable current-support/RB alignment、raw-unit bridge、NPZ round-trip 和 tamper validation。
- 真实 non-locked development artifact 含 12 samples，receipt `passed=true`；没有训练、GPU、formal Dataset 或 `locked_test`。

只记录影响项目结构、模型实现、实验流程或 AI 上下文恢复的重要变化。微小代码编辑不在此逐条登记。

## 2026-09-20

- 2026-09-21：STEP 4.4 audit 先得到 `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`；研究者随后选择独立 known stochastic outage event 并关闭 residual。Structured RSSM、四类 Action 路由、vehicle/CSI dynamics、规则 transition、动态图与两步 prior rollout 完成；STEP 4.4-PATCH 补齐 carrying/hop、Flow 完成同步、4.3A topology reuse、端点/lifecycle/DAG 规则与 state-changing recursive counterfactual，87/87 machine checks 通过。证据仍是 untrained CPU development，不含 Loss/Training/Planner/GPU/locked-test。
- 2026-09-21：STEP 4.4-PATCH2 将 current-support Existing Return 按 `(task_index, Return)` typed identity 接入 final completion gate；固定 `future_return_birth_supported=false`，缺失 required support 时以 side-state 阻止 final completion。focused 26/26、receipt 91/91；Definition 05 仅记录后续 mask/exclude/classify 要求，未实现 Loss。
- 2026-09-21：STEP 4.4-PATCH3 将 no Return slot 从错误的 no-return 改为显式 unknown，real adapter computation-finished 反事实不会 final-complete；DAG release 忽略 invalid edge并要求全部有效前驱完成；terminal Flow 使用冻结 status vocabulary 同步 COMPLETED。focused 30/30、正式 receipt 92/92、6 文件独立重建 hash/size equality 均通过，STEP 4.4 正式 COMPLETE / FROZEN。

- 完成 STEP 4.3B-PATCH：P2A/P2C value processor 回到 Definition 03 联合上下文公式；`Z_t^{PI,L_g}` 保留 STEP 4.3A 全部 11 个 structural blocks，并加入 endpoint/index/presence/validity equality、完整 digest、round-trip 与 structural tamper 机器检查；Comm CSI width 改为读取 tensor contract `n_comm_rb`。仍为未训练 CPU development evidence。

- 完成 Step 2.4 通信 Outcome 语义最终验收：真实 wired manager service 与 wireless event 分拆，total 聚合和 empty/missing 语义冻结；6 slot、7 Decision、14 checks 通过。
- 完成 Step 2.3 Raw 因果最终验收：隔离 future task、补齐真实 Decision/Outcome 字段、执行 return route、冻结 raw/canonical acceleration；Raw Trajectory Layer / 01 完成并冻结。
- 完成真实 AirFogSim Step 2.2 六步 Raw Trajectory：下一 Decision 独立采集，Route/Comm/Comp 显式 no-op，跨步 ID/lifecycle 对齐。
- 将 Step 2.1 v3 与 Step 2.2 JSON/manifest 纳入 Git，补齐机器证据 GitHub 可追溯性。
- 完成真实 AirFogSim Step 2.1 单轨迹四类动作与 Outcome/next Decision 对齐；修正实体 heading 单位，并记录仿真器 acceleration 观察事实。

## 2026-09-19

- active workflow 切换为研究者最新只读 `00–06` 定义；旧 P4/P6/P0–P10 逻辑归档，原证据和原验收边界保留。
- 新增 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`，完成 Step 1 定义—实现审计。
- AI_CONTEXT 现在区分目标定义、被审计的当前代码、旧协议实验和新定义 `NOT_STARTED` 状态。
- 49 项旧 synthetic CPU contract 测试通过，仅证明旧实现可执行；未修改模型、数据、loss、planner、checkpoint，未启动 GPU，未访问 `locked_test`，未执行 Step 2。

## 2026-09-10

- 初次建立 `AI_CONTEXT/00_PROJECT_STATE.md` 至 `08_CHANGELOG.md`。
- 将研究状态、真实架构、数据流、模块导航、实验、研究者决策、已知问题分开，防止摘要层混淆事实与科研解释。
- 将 `AI_CONTEXT/` 接入项目文档权威、自然语言问题路由和知识索引 `--check`。
- `AGENTS.md` 增加三方角色、Context Consistency Check、Git commit/push、私人笔记禁区和冲突处理规则。
- 科研语义未改变；没有修改模型、loss、metric、协议、tensor、checkpoint 或实验产物。
- 验证：AI_CONTEXT 6/6、项目知识/结构 28/28、正式 P4 210/210 通过；全量 1643 项为 0 assertion failure/17 个已登记环境或历史错误。
- 本次完整重构以本文件所在的 Conventional Commit 推送到 GitHub `main`；精确提交以分支 `HEAD` 为准。

## 2026-09-09

- 建立 `docs/` 人类导航和 `docs/registries/` 机器注册表。
- 两个实体级双图 RSSM 正式 seed 分别通过单 seed non-locked 验收；第三 seed 延后。

## 维护规则

每个有效任务完成前执行 Context Consistency Check，只更新真正受影响的文件：current state、research、architecture、data flow、module map、experiments、decisions、known issues、changelog。实现事实变化必须与相应 AI_CONTEXT 更新进入同一任务。

Source of truth：本 changelog 只说明发生了什么；实现和实验真假仍由源码/config/experiment/audit 决定。

## 2026-09-20 STEP 4.2C-B-PATCH

- 修正 Raw amendment 将 current hop `target_node_id` 误作 logical destination 的问题；Input 改用已建立 route terminal，Return 改用 `return_destination_id`，并保存 source/capture-phase provenance。
- 新增同 Epoch destination continuity 与真实 multi-hop single-Flow validator；真实两跳证据通过 FlowID/Epoch/destination/route-revision/E2E no-double-count 检查。
- 顶层 acceptance 对 19 项 required checks 与 scope 取实际 AND；negative tamper/fake-multihop 测试通过。未进入 Sample/Tensor、Graph Builder、模型、训练、GPU 或 locked_test。

Unverified：未在 Git diff 和相应证据中出现的变化不得仅凭本文件推断。

- 2026-09-18: Step 2 froze raw single-step and four action-family contract; added source adapter, focused tests, evidence script, and protocol artifacts.
- 2026-09-19: Added Step 3.1 model-ready sample/tensor contract, minimal real sample artifact, source-hash manifest, focused tests, and navigation records; formal dataset and training remain unopened.
- 2026-09-19: Step 3.1R corrected History alignment, fixed-index presence/padding, DAG Raw capture, typed target namespaces, relation endpoints, and unresolved action reference handling; real v2 Raw and minimal sample evidence regenerated.
- 2026-09-19: Step 3.1F finalized History with past action/outcome, History-union physical/task/flow indices, history relation/DAG/flow alignment, late-entry/disappearance fixtures, and observation-only future-action reference audit; Step 3.2 remains unopened.
- 2026-09-19: Step 3.1F-PATCH separated anchor visibility from unified History-union Future Action indices, added ID↔index validation and disappearing-object regression coverage, corrected the machine policy, and committed the observation-only audit JSON provenance; Step 3.2 remains unopened.
- 2026-09-19: STEP 3.2 completed a minimal non-locked Raw-to-Dataset batch/split/preprocessing validation with three independent development trajectories, train-only masked statistics, deterministic rebuild and round-trip; formal Dataset/training remain unopened.
- 2026-09-19: STEP 3.2-PATCH finalized Dataset isolation evidence with real Raw provenance/lineage/time-grid checks, a separate 12-window development future-reference audit, physical normalization units, and computed validation-report checks; formal Dataset/training/GPU/locked_test remain unopened.
- 2026-09-19: STEP 3.2-PATCH-RECEIPT corrected validation `passed` to require all checks and explicit scope booleans, added negative failure propagation coverage, and reused the frozen sample schema constant; STEP 3.2 is now COMPLETE / FROZEN.

## 2026-09-20 STEP 4.2C-C-PATCH Presence-aware Normalization & Full Flow Semantic Coverage

- `_iter_numeric()` 与 stats policy 固定为 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`，并新增 presence=false completed/superseded 重复 lineage negative fixture。
- History Logical、History Carrying、target Logical、target Carrying 四组 Sample→Tensor 全字段语义检查已接入 receipt；target carrying namespace 明确为 future ground-truth/deterministic-transition state，不是 learned prediction head；ID/provenance tamper 会失败。
- 23/23 focused tests、receipt/build、deterministic rebuild、serialize/load、scope checks 通过；`graph_builder/information_graph/physical_topology/training/gpu/locked_test/formal_dataset=false`。

## 2026-09-20 STEP 4.2C-C Stateful Flow Sample/Tensor Additive Extension

- 新增 `step4_2c_c_flow_sample_tensor_v1.py`、builder、真实低 wired capacity cross-slot runner、focused tests、合同/实施记录和最终 artifact。
- C-B Raw logical Flow/Carrying state 进入独立 History-union/target `logical_flow` namespace；Flow/Carrying、known inactive/padding、Input/Return、DepData=0、train-only numeric normalization、capacity overflow 和 round-trip 均有机器证据。
- 基线阶段的 focused tests、4.2B/4.2A/3.3 回归、真实 trace、deterministic rebuild/hash、serialize/load、tamper 通过；Patch 进一步将 focused suite 扩展为 23/23。
- 该 Step 当时不宣称 formal Dataset、Return multi-hop/reroute runtime、formal capacity、Graph Builder、模型或性能，并建议另行授权 Graph Builder；此后 STEP 4.3A 已完成 builder，原有数据证据边界不变。
## 2026-09-20 STEP 4.3A Typed Dual-Graph Builder

- 新增 11 个 typed graph blocks、显式 development-only Physical topology config、Align/GeoComm 与 current-frame target isolation。
- receipt 对 24 项 required checks 实际 AND，20 项 negative/counterfactual 全通过；无 Encoder、GNN、World Model、Loss、Planner、训练、GPU 或 locked_test。

## 2026-09-20 STEP 4.3B Dual-Graph Encoder

- 新增 type-specific mask-explicit encoders、四类 object-wise presence-gated GRU、五类 directed relation processors、relation-wise masked mean、P2A/P2C 和 residual+LayerNorm updates。
- Flow 采用 Logical/Carrying 独立分支后 fusion，只输出一个 Flow relation latent；delivered/Epoch/index 不作为 learned numeric input。
- 输出冻结为 aligned `Z_t^{PI,L_g}`；artifact 是未训练 CPU development evidence，不进入 World Model/dynamics/loss/planner/training。
- 2026-09-21：STEP 5.0 冻结 Definition 05 十项研究决定并审计旧实现。新 active contract 固定 deterministic mean decoder、Motion/CSI mask-MSE、family KL、overshooting OFF、family-only posterior teacher、joint training 与 prior-only validation。发现 future position 尚未 normalized/tensorized、future CSI target 缺失；Loss/Training 仍未实现，无 GPU/locked-test。
# 2026-09-22 STEP 5.1B-PATCH

- 修正原 5.1B 的跨 horizon target aggregation、mask 不可见、独立 PriorPredictor、zero-h/identity loss、batch-level free bits、raw-unit metric 和硬编码 receipt 问题。
- 当前 CPU receipt 真实调用 STEP 4.4 `initialize_latent → one_step → phy_prior/comm_prior → vehicle_decoder/csi_decoder`，并验证逐 horizon isolation、mask evidence、逐维 free bits、gradient、raw-unit metric 与 target isolation；仍未进入 training/optimizer/GPU/locked-test。

# 2026-09-22 STEP 5.2

# 2026-09-22 STEP 5.2-PATCH

- 修正 Stage 2/Validation 初始 latent：使用 current-observation posterior，未来递推仍为 prior-only；current posterior 纳入 optimizer audit。
- 增加 validation future-posterior/target-encoder runtime guards，并验证调用次数为 0。
- Validation 改为跨完整 validation set 的 per-horizon Motion/CSI numerator/count 聚合，补 unequal-mask fixture。
- Checkpoint load 增加 schema、data identity、normalization provenance、architecture-critical config 兼容性拒绝；receipt 26/26。

- 新增配置化 CPU training loop：Stage 1 posterior-assisted warm-up、Stage 2 prior-dominant recursive curriculum、KL warm-up/free bits、joint optimizer parameter audit、prior-only validation、`L_Val` checkpoint selector 与 checkpoint/resume。
- 8/4 unified non-locked development smoke 完成两步 optimizer update；20/20 required receipt checks、focused 9/9 和相关 current regressions 通过。Receipt 与 compact audit 位于 `code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/`；`.pt` checkpoint 保持 local-only。
- 保持边界：`full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test=false`、`performance_claim=false`；Route/Comp non-empty coverage=0，5.3 仅完成 bounded CPU preflight。

# 2026-09-22 STEP 5.1C-PATCH

- 完成 unified development bundle 的 Physical/Communication slot-wise identity proof、端点真实 ID 反查、10/74 capacity exact alignment、no-prefix/no-crop 机器证明和 tamper negative。
- 重新拟合 `unified_flow_train_normalization_stats.json`：仅使用 8 个 `dev_train` samples 的 History，4 个 validation samples 排除，Future Target 不参与；旧 5-sample stats 仅保留为历史 provenance。
- 截至 STEP 5.1C-PATCH 当时，4.3A/4.3B/4.4 unified rebuild 和 paired 5.1B acceptance 尚未执行；该历史边界已由后续 STEP 5.1D-PATCH 闭合，STEP 5.2 仍需单独授权。
- 2026-09-22：STEP 5.1D-PATCH 完成 Evidence / Context / Action-Coverage Closure。receipt 改为 executed/forbidden scope，跟踪 compact GitHub evidence；机器比较 unified stats source IDs、4.2A frozen batch 恢复的 fit source 与逐窗口 normalized samples；新增 runtime prior-target isolation 与 posterior sensitivity negative test。真实覆盖为 Mobility=48、Comm=1、Route=0、Comp=0，Route/Comp 仅显式 no-op；47/47 checks 通过。仍为 untrained CPU/non-locked evidence。
## 2026-09-22 STEP 5.4

- Added manifest-driven formal interface and CPU-only readiness receipts; recorded formal data/action/horizon/topology/config/checkpoint blockers.

## 2026-09-23 STEP 5.6A — 中途记录（由 2026-09-24 收口记录更新）

- 已接入真实 Formal Dataset v1 的 CUDA smoke 路径：H4 forward/backward、optimizer step、跨 trajectory batch、checkpoint reload 与错误 dataset/config identity 拒绝均已通过；batch 1/2/4/8 均完成工程探测。
- 全量 1104-window prior-only GPU validation 正在远端按 validation trajectory 分片执行；在合并 receipt 生成前不宣称 FULL_1104_GPU_VALIDATION=PASS。
- Formal training 未启动；训练数值配置仍标记为 `AWAITING_RESEARCHER_DECISION`，不把 development default 当作正式决定。locked_test、baseline、Planner、performance claim 均保持关闭。
# 2026-09-24 STEP 5.6B 预启动

新增复用 FormalTrainingInterface/FullFormalTrainer 的正式 runner、全量 prior-only validation、逐步日志、atomic heartbeat 和身份约束 checkpoint/resume。此变更先提交并同步精确 source 后才允许启动；运行证据另由远端 run manifest 和 heartbeat 给出。
# 2026-09-28 STEP 6.2A

Added a source-semantics audit runner, focused test, Planner objective target contract, baseline system metric interface, machine-readable receipts, and implementation record. Corrected the provenance matrix to distinguish actual pipeline exposure and removed unsupported `CAUSALLY_DERIVABLE`. Verdict: `PASS_WITH_READINESS_BLOCKERS`; STEP 6.2B remains blocked. No scoring, candidate selection, baseline, GPU, locked test, or closed-loop execution.
# 2026-09-28 STEP 6.2A-PATCH

重新裁决当前 Flow/E2E 事实，新增 Planner-only Task/Route side-state 与单 validation anchor deadline 因果重放，冻结 Useful/Network Service 双吞吐及 Comm/Comp/Mob effort，移除 Route/Priority blocker。机器发现 4.2C-C/4.4 路线语义冲突，保持 6.2B BLOCKED；无模型、Dataset、checkpoint、GPU、locked_test 改动。
2026-09-30：完成 3080 Ti Formal 数据迁移与身份验收、GPU FP32 等价、batch16 配置、正式 CUDA runner/resume 闭合及 bounded TRAIN smoke；正式调参/Validation 比较/locked_test 未运行。详见 Step 实施记录和机器收据。
