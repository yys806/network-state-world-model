<!-- STEP6.4J CURRENT -->
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

本轮只有固定TRAIN CPU机制测试：两个S-CEM planning/真实step及独立注入fallback一步；无正式B512新搜索、无多episode性能统计。Draft建议pilot2×8、正式24×64；未经授权。
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

机制配置CPU batch1不同于Stage A CUDA batch16，禁止性能比较。fixture在首次B64前冻结；首次搜到16可评分候选，回执frozenset序列化/停止控制异常发生在setter之前，episode停止。修复纯工程诊断后同fixture/seed/B64恢复，首轮离散结果完全一致；两轮各18完整/16 distinct scoreable H4。真实首动作含Comp和Mob，Comm为空；非空Comm仅有合成因果CPU桥测试。

记录：`docs/implementation_records/STEP_06_4B_MINIMAL_CLOSED_LOOP_TWO_CYCLE_SMOKE.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/13_final_acceptance.json`；合同：`docs/contracts/PIJWM_STEP_06_4B_LIVE_CLOSED_LOOP_SMOKE_V1.md`。以下为历史记录，其中“当前”仅表示记录时点。

---

## 2026-10-03 STEP 6.4A 闭环前就绪审计（当前）

STEP 6.4A审计完成（审计证据PASS），CLOSED_LOOP_READINESS=BLOCKED。SEARCH_METHOD=MH-CEM仍只为pure-search backbone，非最终hybrid。局部链路7 IMPLEMENTED/3 PARTIAL/5 MISSING；缺live history-only输入/belief/sidecar、winner动作序列/首动作及真实ID执行桥、研究者fallback和latency决定、两轮真实反馈MH证据。MH B1024搜索median90.506/P90 204.916/P95 284.152/max421.701秒；仿真slot0.1秒不是墙钟deadline，DECISION_LATENCY_REQUIREMENT=UNRESOLVED，尚无在线执行证据。Stage B用途由研究者冻结为online-budget trade-off，不能重选算法；A约30.46–41.85h、B NEW PROPOSAL约10.26–14.06h、C NEW PROPOSAL约1.54–2.11h，仅成本情景。4090_MIGRATION_NOT_YET_JUSTIFIED。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作研究者确定fallback及decision latency模式，再授权最小接口集成；不自动运行实验。

本Step没有新搜索或仿真运行。只从960历史raw验SHA读取MH320 timer；CPU grammar单fixture证明全NOOP在eligible compute被拒，合法alpha/HOLD可admitted，非全状态安全证明。

记录：`docs/implementation_records/STEP_06_4A_CLOSED_LOOP_READINESS_STAGE_B_PURPOSE_FREEZE.md`；机器审计：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/06_go_no_go_receipt.json`。下方为历史状态，不能用Stage A PASS替代闭环readiness。

---

## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

每方法320 cases：HRS/S/MH完整H4尝试86437/111093/111943；不同可评分候选26877/32497/33505；不可评分完成57565/55785/55999。S-HRS=135/168/17，CI=[0.2625,0.475]；MH-HRS=145/166/9，CI=[0.3125,0.5375]；MH-S=78/198/44，CI=[0.03125,0.184375]。64锚点cluster内保留5 seeds，10000次重采样、seed6316。六类及每锚点证据在10/13；diagnostic-only simultaneous sensitivity不改变选法。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-02 17:23 Stage A 过半运行快照（非最终验收）

515/960（53.6%）完成结果已独立验身份并逐文件SHA备份至本机D:，实际独特一步转移527,360；覆盖35锚点，315个已完成case找到可评分H4。未发现重复/错误执行身份、source drift、NaN/Inf或scorer exception。GPU仍为RTX3080Ti/CUDA/FP32/batch16，原runner持续运行，约21.9小时、23.53 cases/hour，预计还需约19小时，仅资源规划估算。Stage A=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；Return-birth限制保留，无新增阻塞。唯一下一动作继续监控/备份至960完成，再独立统计验收并停止。部分有序结果不得用于提前选择方法或宣称优势。

证据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/snapshots/health_20261002T092315Z.json`。下方为较早快照。

## 2026-10-02 Stage A 运行监控快照（非最终验收）

08:25北京时间已独立检查并逐文件SHA备份288/960 cases（30%），实际独特一步转移294,912；覆盖20个锚点，已有150个case找到可评分H4，因此未观察到全矩阵系统性零H4。没有重复/错误执行身份、NaN/Inf或评分器异常。约12.98小时、22.26 cases/hour，预计剩余约30.1小时，仅运行规划估算。源码与RTX3080Ti/CUDA/FP32/batch16身份保持冻结；Stage A仍RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false。无新增阻塞；唯一下一动作继续监控和持久备份至960完成，再独立验收并停止。

机器证据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/snapshots/health_20261002T002545Z.json`；raw/inventory保留本机D:，远端原始结果保留。该快照不可用于提前选择方法或宣称性能优势。

## 2026-10-01 STEP 6.3D FORMAL VALIDATION STAGE A（当前运行）

研究者已单独授权并启动 Formal Validation Stage A（仅 B1024）：64锚点×5 seeds×3方法=960 cases，名义预算983,040。2026-10-01北京时间19:26在RTX3080Ti/CUDA/FP32/batch16启动，启动前源码/768 TRAIN/771文件SHA/64锚点/checkpoint/执行身份全部PASS，Validation起点0。首批2个case已通过身份检查并持久备份；这是运行快照，不是最终效果结论。STEP_6_3D_VALIDATION_STAGE_A=RUNNING，VALIDATION_COMPARISON=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；future Return-birth限制保留。唯一下一动作是监控并备份Stage A，960完成后独立验收并停止。选择范围仅Planner v1 structured-search backbone/pure-search baseline，不是最终hybrid planner冻结。

记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；启动前机器收据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/01_launch_preflight_receipt.json`；动态监控与本地备份在该运行目录。下方均为较早阶段快照，关于未授权/Validation0的表述不能代表当前运行。

## 2026-10-01 STEP 6.3D PRE-VALIDATION BLOCKER CLOSURE（当前）

PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。

入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md`；机器验收：`code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。下方为较早快照，原B1/B2阻塞已关闭。

## 2026-10-01 STEP 6.3D PRE-VALIDATION AUDIT（当前）

STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。

记录：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md`；机器验收：`code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。下方均为较早记录。

# 当前与历史实验

## 2026-10-01 STEP 6.3D 正式 TRAIN 选参验收

3080 Ti、FP32 batch16，32 锚点×3 seeds×两种 CEM×四组参数 = 768/768 正式 TRAIN solves，名义/实际独特一步转移均为 393,216。各组 48/96 cases 找到可评分 H4；原始结果独立重算后 S-CEM/MH-CEM 均选 `(K=4,rho=0.1)`。总计 90,263 条完整 H4 路径，31,667 个去重后可评分候选和 57,797 次不可评分完成尝试；16 个锚点在所有配置中均不可评分。正式耗时 21.38 小时。逐组、逐锚点/种子和 SHA 归档见 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/train_tuning_diagnostic_summary.json`、`train_tuning_closure_acceptance.json`、`train_tuning_local_archive_manifest.json`。这些仅是 TRAIN 选参观察；Validation=0，HRS 对照/方法选择/闭环/`locked_test` 均未运行。

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH（TRAIN 诊断）

修正后 TRAIN 32 的 HRS seed6391/B_WM64 固定诊断共 2048 次 unique World Model 一步转移；16/32 锚点有至少一个 H4 可评分候选，16/32 没有。406 条完整 H4 中 131 可评分、275 不可评分；275 条均有 future Return birth 支持边界。240 个语法死路分支。Validation 64 仅修正静态 Objective 资格及 deadline sidecar，不做方法搜索。CPU batch 1/4/8/16 一步与 H4 等价，吞吐分别约 0.87/0.89/0.96/0.92 unique transitions/s。详见 20–25 机器收据；这些是前置诊断，不是 HRS/CEM 对比或性能证据。

## 2026-09-29 STEP 6.3D（进行中，非方法比较结论）

32 个 Formal TRAIN、64 个 Formal Validation 非空锚点按各 split 的候选数对数四分位、Comp-base 有无和 sample-id hash 固定；Validation 六个静态空域单列。38 条原始轨迹只读重放对齐 96/96 因果 deadline sidecar，96/96 冻结目标侧状态就绪。Synthetic exact oracle 的 16 个 H4 序列上，HRS/S-CEM/MH-CEM 均能恢复最优；9/9 focused tests 与 compileall 通过。冻结模型 CPU 的首个 TRAIN 锚点 B_WM=512 当前收据：512 次独特转移、130 条完整 H4、0 条可评分、547.008 秒，均在 H1 触发既有 future Return birth 支持边界；558.291 秒的原收据保留为诊断字段增加前对照。它只是一条配置和一个锚点的诊断，不构成 TRAIN 调参、Validation 比较或最终方法选择。完整矩阵尚未完成；无 GPU、训练、闭环或 `locked_test`。

## 2026-09-29 STEP 6.3B/6.3C-PATCH

CPU-only 4416 个 Formal TRAIN H1 raw→Planner-v1 投影审计：728 条同决策 offload Route 依赖 Comm 行逐条排除，任务数量条件拒绝为 0；结构准入 4045/4416，剩余 1 个投影后未见 Comm 结构、349/15 个 Comp 类别和 6 个未见 joint 结构。TRAIN/Validation 静态候选域空数为 14/4416、6/1104；候选数增长只是规则变化，非性能改善。冻结 checkpoint 单非锁定 Validation anchor 的非空 Comm H1–H4 机制路径仍与原顺序推演等价，`B_WM=4`。结构准入后的动作语义字段对照完成：3920/4045 通过，125 个 Comp amount 差异超过绝对容差 `1e-7`，绝对差中位数 `1.4475e-7`、最大 `2.9793e-7`；容差和策略均未修改。无优化器、排名、baseline、闭环、GPU、训练或 `locked_test`。

## 2026-09-29 STEP 6.3C

Historical before-patch CPU-only static CandidateDomain audit: TRAIN 4416 anchors, Validation 1104 anchors (descriptive only). TRAIN empty domains: 1935; exact candidate cardinality median 6, P90 205701120, max 313949952. These counts have been superseded by the 6.3B/6.3C-PATCH Task selection boundary above. No optimizer, ranking, baseline, closed loop, GPU, training or locked_test.

## 2026-09-29 STEP 6.3B

CPU-only Formal TRAIN support-catalog and bounded synthetic grammar/admission tests. No World Model candidate ranking, frozen-checkpoint rollout, GPU, training, locked test, baseline or closed-loop run. Validation scan is descriptive after TRAIN catalog freeze. `STEP_6_3B=PASS` means grammar/labels/admission only; no performance claim.

## 2026-09-28 STEP 6.3A support audit

CPU-only support audit over Formal TRAIN 48 trajectories/4416 windows and
validation 12 trajectories/1104 windows. TRAIN support: Comm widths 1/2/3,
RB IDs 0–49; Mob HOLD plus five profiles with same-profile two-UAV joint
observations; Comp reconstructs from the causal CPU base rule with global alpha
`{0.5,0.75,1.0}`. Joint family factorization is `NOT_SUPPORTED`. Formal-index
rolling H1–H4 sequences were audited. No Future Target, checkpoint or World
Model rollout was used. Verdict: `PASS` for support evidence only; no candidate
method was selected.

Verification boundary: `compileall` passed and CROSS_LAYER gate passed 115/115.
Full repository unittest discovery was not clean (1993 run, 34 errors, 1
historical receipt mismatch); Windows GBK output and missing historical local
artifacts caused errors. The broad suite also ran synthetic CPU trainer tests,
an execution deviation outside the audit-only scope. No Formal training,
checkpoint or dataset write occurred; the rewritten historical cross-layer
receipt was restored.

## 2026-09-28 STEP 6.2B-PATCH CPU 机制验收

同一非锁定 Formal Validation `anchor-0001` 的两个 Route 空动作候选，共用冻结 best.pt、mean prior、expected service，H1–H4 rollout/scorer 有限；Route 编译为缺席哨兵，参数摘要与 checkpoint SHA 不变。Comm 分母复核为 50 个全局 RB ID，而非 242 条关系行。8 份新收据在 `code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928/`。`STEP_6_2B=PASS` 是 scorer 合同结果，不是候选优劣、闭环或性能结果。无 GPU、`locked_test`、训练、baseline。

## 2026-09-28 STEP 6.2B CPU scorer probe（历史阻塞）

一个非锁定 Formal Validation `anchor-0001` 上的冻结 `best.pt` strict-load、mean-prior/expected-service H1–H4 rollout 与 scorer 均有限，参数 digest 和 checkpoint SHA 未变。这是机制证据，不是候选优劣或性能结果。13 份机器收据位于 `code/artifacts/protocols/pi_jwm_step6_2b_objective_scorer_v1_20260928/`；`STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`，因 pending Route 固定支持与同路径 Host 语义待裁决。GPU、locked_test、baseline、闭环未执行。

## 2026-09-28 STEP 6.2A-CLOSURE

`NO_RETRAIN_ACCEPTED=true`; existing formal checkpoint retained. Planner v1 has enabled single-hop Route only. At this closure, readiness was `READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`; scorer had not yet started. The later 6.2B blocker is recorded above. Formal multi-hop train/validation coverage is zero; patched full validation did not run. Receipts: `code/artifacts/protocols/pi_jwm_step6_2a_closure_single_hop_v1_20260928/`.

## 2026-09-28 STEP 6.1 Formal Validation CPU mechanism diagnostic

冻结 seed 5601 `best.pt`、一个确定性 Formal Validation anchor、四种 `CAUSAL_DOMAIN_PROBE`，分别与 H4 RULE_FALLBACK 对照；H1–H4 action-conditioned/recursive/finite、三种固定随机种子配对复现和 K=1/2/4/8 CPU 实现计时通过。原始数值见 `action_family_response.json`、`recursive_feedback_audit.json`、`stochastic_common_seed_diagnostic.json`、`cpu_batch_runtime_diagnostic.json`。这不是新训练、预测质量评估、性能比较、MPC 优化或 closed-loop 实验。

## 2026-09-28 STEP 5.6C Formal Validation Observation

正式 seed 5601 训练 5520/5520 步完成；五次完整 1104-window prior-only validation 的 `L_Val` 依次为 `0.1766124568, 0.0801012691, 0.0775463209, 0.0764608792, 0.0743133878`。最终步是严格 `argmin L_Val`，best/latest 都是 step 5520 且 428 个模型张量逐项一致。完整 H1–H4 `L_Pred`、Motion/CSI raw MAE/RMSE、CPU 单样本推理和文件 SHA 见 `code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928/`。训练源 Git SHA=`6e15ec2`；Dataset/Config SHA 分别为 `6392a08b...`/`a806c320...`。这是单 seed、同一 validation split 的 **Formal Validation Observation**，不是 locked-test、baseline、泛化或闭环系统性能证据。

## 2026-09-24 STEP 5.6A-CONFIG-FREEZE

研究者批准的 Formal Training Config v1 已冻结并写入 `code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/`。这是训练协议配置证据，不是训练结果；`formal_training=false`、`gpu_training_verified=false`、`locked_test_accessed=false`。旧 GPU validation receipt 的 availability bookkeeping 已由真实 validation target masks 在 CPU 重算为 H1–H4 各 1104，原始 numerator/count 和 `L_Val` 不变。

## 2026-09-24 STEP 5.6A GPU smoke / validation（运行验收完成）

正式数据 H4、RTX 4090 few-step CUDA smoke 已通过 batch 1/2/4/8 的有限 loss/gradient、真实参数更新、跨轨迹 batch、checkpoint 重载与错误 Dataset/config 身份拒绝。完整 1104-window prior-only GPU validation 通过四组互斥轨迹的 sample identity 合并：12 条 validation 轨迹、无重复/遗漏、无参数更新或未来 posterior teacher；`L_Val=0.829751`，四个 horizon 的 Motion/CSI raw MAE/RMSE 已写入机器凭证。四组串行计算/遍历耗时合计 7103.09 秒、数据加载合计 72.15 秒。此处属于未训练的运行能力验收，不是预测性能、收敛或泛化结果；formal training、baseline、Planner、locked_test、performance claim 均为 false。机器证据入口：`code/artifacts/audit/pi_jwm_step5_6a_20260923/`。

## 2026-09-23 STEP 5.5-PATCH CPU full-shard acceptance

这是数据路径验收，不是正式训练或性能实验。正式 package 的 4416 train、1104 validation 全部可索引；跨不同 trajectory 的 batch、H=4 CPU optimizer step、prior-only validation batch、checkpoint reload 和错误 dataset identity 拒绝见 `code/artifacts/audit/pi_jwm_step5_5_patch_20260923/`。旧 `runtime/` 1+1 smoke 仍为独立 mini 证据。全量 future Return 结构审计识别 8828 次 unsupported/fixed-support 事件；旧字段计数 0/0/0 已被取代。GPU、formal training、locked_test、baseline、Planner 和 performance claim 均为 false。

## 2026-09-23 STEP 5.5 Formal Dataset v1 acceptance

- 60/60 真实 causal trajectories，0 rejected/replacement；H=2/L=4、每条 96 transitions、48/12 trajectory split、4416/1104/5520 windows。
- 四动作 train intervention rate：Route 38.60%、Comm 42.78%、Comp 38.52%、Mobility 40.08%；每族干预覆盖 48 条 train 与 12 条 validation trajectory。
- Dataset acceptance 25/25、deterministic rebuild 5/5、CPU FormalTrainingInterface **runtime 1+1 mini** smoke 13/13 均为 true；其 H=4 optimizer/validation/checkpoint path 不证明 5520-window full-shard consumption，后者由 STEP 5.5-PATCH 单独验证。
- 证据入口：`code/artifacts/manifests/pi_jwm_step5_5_formal_dataset_v1_20260923/`。这是 Dataset/CPU interface acceptance，不是 formal training 或 performance run。

## 2026-09-21 STEP 5.1A-PATCH target-contract validation

- 这不是训练实验：脚本从真实 STEP 4.2A non-locked development trajectory 和 STEP 4.3B frozen train-only stats 构造 12 个 Future Target samples/tensor。
- 19/19 focused contract tests 与 102/102 related regression 通过；receipt 的 local H1/H2 Motion、current physical/model comm slots、sample/tensor、deterministic rebuild、real trajectory、formal_dataset/training/gpu/locked_test scope checks全部为 true，顶层 `passed=true`。artifact digest 为 `dc6c5b0b0d957f0e1e57ee09c19d2b54e7b2ecd0bb631a9612a8200078ab579a`。
- 证据覆盖 local one-step Motion/next-speed、相邻帧 component masks、future entity/order/birth/disappearance、未来 outcome per-RB CSI、History CSI 隔离、current model slot identity、wired/missing masks、future-only isolation、unsupported side metadata、normalization round-trip、serialization 和 tamper rejection。它不是正式 Dataset、Loss/Posterior/Metric、训练或性能结论。

本文件只记录客观状态，不自动解释科研意义。Source of truth：完整字段见 `docs/registries/experiment_registry.json`，正式数字见 `results_registry.json` 和对应 acceptance JSON。

> 2026-09-19：新的 active workflow 已冻结 Raw Trajectory Layer / 01。Step 2.4 是非 locked 真实通信接口验收，不是训练实验；下列 P4/P6 实验保留为旧定义下的 Historical / Archived evidence。未启动 GPU，未访问 `locked_test`。

Step 2.4 机器证据：`code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，真实 6 slot / 7 independently recaptured Decisions / 14 checks；wireless、wired、total transmitted progress 与 task lifecycle 对齐。

## 旧 P4 正式实验（Historical / Archived）

| ID | Seed | 数据/协议 | 状态 | 客观边界 |
| --- | ---: | --- | --- | --- |
| `P4-EARSSM-SEED-20260831` | 20260831 | v5 causal-motion h20 / frozen v2 | 单 seed 通过，best epoch 39 | unlocked 单 seed，不是跨 seed 结论 |
| `P4-EARSSM-SEED-20260830` | 20260830 | 同上 | 单 seed 通过，best epoch 40 | unlocked 单 seed，不是跨 seed 结论 |
| `P4-EARSSM-SEED-20260832` | 20260832 | 同上 | deferred，未运行 | 没有结果；需要用户明确授权 |

共同训练入口：`code/scripts/run_formal_p4_entity_rssm_gpu_v1.py`。共同 tensor：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v5_causal_motion_h20_unlocked_20260906/`。冻结协议：`code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

## 正式验收证据

- seed 20260831：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`
- seed 20260830：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`
- 两份报告均为 `status=passed`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- 完整 9 项指标保存在 `docs/registries/results_registry.json`，并由索引生成器自动与 acceptance 核对。

## 前置门

- `P4-EARSSM-CPU-CONSISTENCY`：机制、梯度和 strict reload 证据通过；不证明性能。
- `P4-EARSSM-GPU-BATCH-PROBE`：选择 batch=8，未执行 optimizer step；不证明性能。
- `P4-FIRST-PRINCIPLES-20260906`：历史 global RSSM 表达限制的只读诊断；动机证据，不是新方法性能。

## 重要历史方法

以下均不是当前方法，详情与原始路径见 `docs/registries/historical_method_registry.json`：

- aggregate dual-graph residual baseline：通信/资源诊断有部分通过，node 位置和逐 RB/实体级边界未闭合。
- physical-edge feedback GRU：冻结 sentinel 的 validation link-F1 门失败。
- link persistence residual：单独修链路头不足以闭合完整 P4。
- global complete RSSM：补齐 prior/posterior/KL 语义，但全局池化/广播不能区分实体，且 sentinel node-x 门失败。
- node-x non-degradation loss：修正幅度下降但共享 base 变差，sentinel No-Go。
- v11 selector/ranking：历史决策诊断，不是逐候选世界模型 rollout planner。

## 旧 P6 实验边界

`P6-CANDIDATE-ROLLOUT-AUDIT-20260826` 状态为 `blocked_prototype_only`。CPU 原型存在，但 P6 未开放，不得运行正式 planner GPU 或给出规划收益结论。

Unverified：第三 seed 结果、三 seed 均值/方差、locked test、最终泛化和正式 planner 收益均不存在。

## 2026-09-22 STEP 5.1B-PATCH Posterior / Loss / KL / Metric

修正原 5.1B 的跨 horizon target aggregation、mask evidence 缺失、独立 prior、zero-h/identity loss、batch-level free bits、raw-unit metric 和硬编码 receipt。当前 `TargetEncoder` 输出保留 `[B,L,S,D]`，future teacher 按 Physical/Communication 分离；receipt 真实调用 STEP 4.4 current latent/dynamics/priors/decoders，并验证 temporal isolation、mask evidence、per-dim free bits、gradient、raw-unit metric、prior target isolation。

截至 STEP 5.1B-PATCH 的历史快照：Focused 5/5，STEP 4.4 regression 30/30，12-sample non-locked development receipt `passed=true`。随后 5.1C/5.1D 已统一 support 为 10/74 并完成 paired integration；5.2 仅增加 CPU development optimizer smoke，当前仍明确 `full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test_accessed=false`、`performance_claim=false`。

## 当前新定义实验状态

## 2026-09-22 STEP 5.3 CPU Training Preflight

### STEP 5.3-PATCH closure

- 修正 raw metric bridge：decoder prediction 保持 raw units，只对 normalized target 逆变换一次；Motion 按 x/y/z/speed 分量报告，CSI 按 dB 报告。
- Phase A 使用真正无更新 pre 与训练后同路径比较；Phase C 真实复用 5.2 `train_step` 的 H=1→H=2 与 beta 0→1 KL warm-up。
- 独立 fresh run 与 checkpoint resume 均 deterministic；CSI scale audit 显示 prediction 初始约 0 dB、target 约 98–99 dB，未发现 normalization bridge bug。
- Verdict 分离为 `LEARNING_SIGNAL_GO` 与 `TINY_OVERFIT_NO_GO`；不得写成 tiny-data overfit 或 GPU readiness。

## 2026-09-22 STEP 5.3E CSI Train-Mean Bias Formalization

- 研究者决定 v1 使用 raw CSI decoder + train-only CSI mean bias initialization；正式接入 `Step52Trainer`，不再依赖诊断脚本手工 mutation。
- 固定 `[0,1]` dev_train、200 CPU steps、Stage 1→Stage 2、1→2 curriculum、beta 0→1 warm-up；H1/H2 Motion 与 CSI 均满足 relative drop ≥50% 且 final normalized MSE ≤1。
- receipt=`FORMALIZATION_PASS`、`TINY_OVERFIT_GO`；该结果仅为 development tiny-data capacity/optimization evidence，不是 formal training、泛化或性能声明。

### STEP 5.3D CSI Scale / Optimization Diagnosis

- 固定 `[0,1]` dev_train subset，CPU 200-step baseline 与 mean-bias diagnostic 对照均复用 5.2 的 Stage 1→Stage 2、1→2 curriculum、KL warm-up/free-bits 和 0.5/0.5 loss。
- H1/H2 actual CSI normalized MSE 与 raw bridge expected MSE 完全一致；baseline 最终约 `117.66/119.74`，mean-bias diagnostic 最终约 `0.141/0.163`。
- Observation 支持 raw-output initialization/conditioning bottleneck；这不是 researcher decision。正式 raw-head bias 或 normalized-output bridge 均未采用。

- 固定 `dev_train` tiny subset：Phase A/B 使用 sample index `0`（`step2.4-real-communication-seed0::anchor-0001`），Phase C 使用 index `0,1` 两个真实 samples；4 个 `dev_validation` 只作 prior-only diagnostic。
- 预注册 development-only gate 为 family/prior relative loss drop `>=0.5%`，A/B/C bounded steps 为 `30/30/40`；receipt 所有 required checks 为 true，结果 `GO`。
- Phase A Stage 1 Motion `0.0005155009→0.0000088083`、CSI `324.4449→320.2018`；Phase B prior H1/H2 `L_Pred` 相对下降 `1.20%/1.19%`；Phase C 两样本 H1/H2 Motion `78.57%/84.69%`、CSI `1.68%/1.69%`。
- 11 个 trainable module groups 均有 finite gradient 并发生更新；zero-gradient step 数、KL raw/adjusted、free-bits、normalized distribution、raw-unit Motion/CSI MAE/RMSE 均写入 artifact。Resume 后续轨迹与 uninterrupted run 完全一致。
- 这是 CPU development capacity/optimization preflight，不是正式训练、泛化或性能结论；full training/GPU/formal Dataset/locked_test/baseline/Planner 仍未开始。

## 2026-09-22 STEP 5.2 Training Loop / Curriculum / Joint Training

- 这是 CPU development implementation smoke，不是正式训练实验。`Step52Trainer` 接入 8 个 `dev_train` 与 4 个 `dev_validation` unified samples；Stage 1 使用 family-specific posterior teacher，Stage 2 使用 prior-only recursive rollout，当前 `L=2` 的配置化 curriculum 为 `1→2`。
- receipt `code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/acceptance_receipt.json` 的 26/26 required checks 为 true：KL warm-up/free bits、joint optimizer groups、known-rule 参数排除、两步 CPU optimizer update、prior-only validation、`argmin L_Val` selector、checkpoint/reload/resume、causal leakage negative checks 和 reproducibility 均通过；5.2-PATCH 还闭合了 current-observation posterior 与 identity-safe resume。
- smoke diagnostic `L_Val=168.31609344482422` 不是性能结果；没有 tiny-data overfit、full training、GPU、formal Dataset、baseline、Planner 或 locked-test。Route/Comp non-empty coverage=0，Comm=1，Mobility=48，Route/Comp 仍是 future formal training/data coverage gate。

- `STEP 1` 只有只读实现审计和 49 项旧 synthetic CPU contract 回归；它们不是新定义性能实验。
- 新定义正式 Dataset、full training、性能、planner 实验均为 `NOT_STARTED`；5.1B/5.1D primitives 与 5.2 CPU development loop 不是正式性能实验。
- STEP 4.2C-C 是非训练的 CPU/non-locked contract validation：真实 direct Input/Return、真实两-hop和低 wired capacity cross-slot trace进入 additive Flow Sample/Tensor artifact；`passed=true`，不构成 formal Dataset、模型或性能实验。
- STEP 4.3A 是非训练的 CPU/non-locked representation validation：复用冻结的五个 development Tensor samples，生成 typed graph artifact；24 项 required checks 与 20 项 negative/counterfactual 均通过。这不是图编码器实验、正式 Dataset 或性能结论。
- STEP 4.3B 是 `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`：复用同一批 frozen development inputs，在 CPU 上验证结构接线、History 因果、mask、方向、P2A/P2C、置换等变、序列化、确定性与 backward。5.2 之后该 encoder 可在 CPU development loop 中参与 joint optimizer smoke，但没有正式训练或性能结论。
- 下一实验步骤尚未授权；Step 2 建议仅冻结一步轨迹和四类动作合同，不训练。
- STEP 3.1 原 10 项/4 tests 记录已由 STEP 3.1R 修正证据取代，不再作为当前合同验收。
- STEP 3.1F 不是训练实验：最小样本通过 24 项合同 checks、12 项 focused tests、round-trip；History 为 `O_1+A_1+Y_1+O_2`，Action/Target `[2,3]`，History union index、Future Action ID↔index 对齐、history relation/DAG/flow 对齐已验收。未来 reference audit 扫描 4 个非 locked Raw artifact、18 个窗口，0 个 unresolved reference；locked/training/gpu 均为 false。
# 2026-09-19 STEP 3.2 validation

- `code/artifacts/protocols/pi_jwm_step3_2_raw_to_dataset_batch_v1_20260919/` is an observation-only development bundle: 3 independent trajectories, 12 causal windows, `dev_train=8`, `dev_validation=4`.

- STEP 3.2-PATCH finalized isolation evidence in the same bundle: provenance records real seed/source SHA/config lineage/time ranges/slot duration and 3.1F contract version; time-grid and execution timing checks run before windowing; future-reference audit is a separate Git-tracked observation artifact with 12 candidate/12 constructed/0 unresolved windows; normalization units are `m/s`, `m/s^2`, and `AirFogSim data-unit`.
- STEP 3.2-PATCH-RECEIPT finalized the machine receipt: top-level `passed` is the AND of computed required checks plus explicit non-locked scope checks; negative fixture verified failure propagation; sample contract provenance reuses the frozen schema constant. STEP 3.2 is COMPLETE / FROZEN.
- The split is trajectory-level; normalization is fit only on train valid masked values for speed, canonical acceleration and task size. The bundle is not a formal Dataset and has no training/GPU/locked-test evidence.
# 2026-09-21 STEP 4.4 Communication Service Source Audit（历史前置门）

- Evidence class: `SOURCE_AUDIT_ONLY_NO_WORLD_MODEL_IMPLEMENTATION`.
- Verdict: `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`.

# 2026-09-21 STEP 4.4 Structured RSSM acceptance

- Researcher resolved the audit gate by selecting an independent known stochastic outage event and closing learned service residual.
- PATCH3 focused tests: 30/30. Formal machine receipt: 92/92 required checks (50 original + 21 service + 21 structural), including the canonical real-adapter unknown Return path, typed Existing Return binding, Future-Target support isolation, valid/invalid/multiple-predecessor DAG changes, and partial/intermediate/terminal Flow status transitions. Six formal artifact files match an independent rebuild by SHA-256 and size; evidence remains untrained CPU development only.
- Evidence class: `UNTRAINED_DEVELOPMENT_WORLD_MODEL_EVIDENCE`. No prediction/calibration/planning/performance claim; `training=false`, `gpu=false`, `locked_test=false`, `formal_dataset=false`.
- This is source/contract evidence only, not training, prediction accuracy, stochastic quality, or performance evidence.
# 2026-09-21 STEP 5.0 Definition 05 decision/audit

- Documentation and source audit only; no forward/backward, optimizer, training, GPU, or metric result was produced.
- Historical pre-5.1A audit: STEP 4.2A sample future position was raw-only and frozen tensor lacked future CSI target; the additive 5.1A target contract has since closed this prerequisite.
- Historical P4 checkpoints/results remain Historical Reference only and cannot be compared with the new method without matching dataset/split/history/horizon/target/normalization/metrics/seed policy.
- Scope: `implementation=false`, `training=false`, `gpu=false`, `formal_dataset=false`, `locked_test=false`, `performance_claim=false`.
# STEP 5.1D-PATCH acceptance（2026-09-22）

receipt 为 47/47 checks true，12 samples，capacity 10/74，prior/posterior/decoder calls 24/24/24，gradient probe finite/non-zero，unified stats source IDs、4.2A frozen batch recovery、exact upstream train lineage 与 runtime prior-target isolation 通过，deterministic rebuild identical；仅为 non-locked CPU development integration evidence。真实 action coverage 为 Mobility=48、Comm=1、Route=0、Comp=0，Route/Comp 仅 explicit no-op。
# 2026-09-22 STEP 5.2-PATCH

- CPU-only semantic closure artifact：`code/artifacts/protocols/pi_jwm_step5_2_training_loop_v1_20260922/acceptance_receipt.json`，26/26 required checks passed。
- Validation audit now records current-observation posterior calls=4, future posterior teacher calls=0, Future Target Encoder calls=0, and per-horizon Motion/CSI numerator/count plus `L_Mot/L_CSI/L_Pred/L_Val`.
- Checkpoint audit includes compatible reload and rejection of wrong data identity / normalization provenance. This remains development smoke evidence, not convergence or performance evidence.
## 2026-09-22 STEP 5.4 GPU Training Readiness

- CPU-only readiness artifact: `code/artifacts/protocols/pi_jwm_step5_4_gpu_training_readiness_v1_20260922/`；manifest-driven interface、action coverage、config/checkpoint schema 已生成；未执行 CUDA 或 formal training。
# 2026-09-25 STEP 5.6B 中途过程图

远端正式 run 的 2646-step 只读日志快照生成训练损失、两次完整验证损失及 Motion/CSI 分 horizon raw MAE/RMSE 图；`docs/figures/step5_6b_live_progress_20260925/` 保存 PNG 和源日志/图 SHA 凭证。训练源码 Git SHA 为 `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`，本地后续文档提交不改变该 run 身份。已完成验证的 `L_Val` 为 0.1766124568、0.0801012691；这只是中途诊断，不是最终性能结论。训练仍运行，未访问 locked_test。

# 2026-09-24 STEP 5.6B 预启动（历史）

研究者已授权正式 GPU 训练启动，runner 与独立 Go/No-Go 正在预启动验收。此源码快照尚无正式训练结果或性能结论；训练一旦开始，run ID、Git SHA、Dataset/Config SHA 和进度以远端 run manifest/heartbeat 为准，不以旧 5.6A smoke receipt 代替。
# 2026-09-28 STEP 6.2A — Planner Objective Source & Semantics Audit

Status: `PASS_WITH_READINESS_BLOCKERS`; `STEP_6_2B_READINESS=BLOCKED`. CPU source/provenance audit and objective/baseline interface records only. No checkpoint load, baseline run, candidate ranking, MPC, GPU, locked test, or closed loop. Focused test: 1/1 passed before final regression closure. Receipts are under `code/artifacts/protocols/pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928/`.
# 2026-09-28 STEP 6.2A-PATCH diagnostic

CPU-only 单条 Formal Validation deterministic replay 与当前 4.4 规则负例：deadline sidecar/Task 槽位对齐通过；两跳目的节点数组的中间跳服务后 holder/跳序号未推进，故 `STEP_6_2B_READINESS=BLOCKED`。这是机制诊断，不是模型效果或闭环结果。没有 GPU、`locked_test`、baseline、排序或训练。
2026-09-30：`STEP_6_3D_3080TI_MIGRATION_QUALIFICATION=PASS`，仅为迁移、FP32 GPU 等价、batch 探针和 bounded TRAIN smoke。batch 8/16/32/64 中位分别 10.3593/10.9154/12.4562/12.6699 unique transitions/s；正式选 batch16，因为 CEM K4/B256 在 32/64 没有完整 H4。完整矩阵与 Validation 方法比较未运行，未选搜索方法、未访问 `locked_test`。
