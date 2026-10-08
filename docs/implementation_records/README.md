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

## 2026-10-04 STEP 6.4F（已完成当前授权范围）

STEP_6_4F=PASS；Comm当前资格修复与真实一步机制通过。共同谓词：Task present、offloading/transmitting、未完成，再与唯一合法活跃无线Flow绑定相交。Flow对象不删除，Flow存在不等于可执行。TRAIN4416/Validation1104完整静态前后审计与输入SHA一致；32/64搜索anchors中28/57的根候选域改变。真实TRAIN anchor0041：Task_24（offloading），UAV_1→vehicle_7，RB0/start0/width1，setter与真实无线传输事件一致，4.5→4.6s一次决策步；候选A独立episode、模拟NO_SCOREABLE_H4、同一bridge、无Comm的canonical合法动作亦4.5→4.6s。Future Target poison不改变当前输入/动作。FORMAL_SEARCH_EVIDENCE_REUSE=TARGETED_REQUALIFICATION_REQUIRED：7个Validation起点有覆盖全部H1–H4分支的规则不变量证明，另外57个旧raw无完整候选/预测轨迹，不足以证明winner/ranking不变。最小建议重新验证57×5×3方法B1024=855及57×5 MH B512=285，共1140cases；仅建议，未授权、未执行。旧TRAIN调参28/32根域受影响，不声称历史调参验证新域；K4/rho0.1按本轮研究者指令保持，不自动retune。B512_EVIDENCE_REQUALIFICATION_REQUIRED=true；PLANNER_V1_CLOSED_LOOP_B_WM=512作为研究者working-budget保留。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_FALLBACK_DECISION仅表示执行机制就绪，不解除搜索证据重新验证门。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING；SEARCH_METHOD=MH-CEM pure-search骨架，非最终hybrid。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED / DEFERRED；GPU=NOT_USED；locked_test=false。唯一下一动作是研究者审阅旧证据重新验证范围及fallback决定，不自动开跑。

证据：`docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/09_final_acceptance_receipt.json`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/07_formal_search_evidence_reuse_assessment.json`。历史6.4E BLOCKED及旧搜索accepted状态仅在旧定义内保留。

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

## 2026-10-03 STEP 6.4C 预算校准预注册（当前）

STEP 6.4C 已获授权并完成预算校准预注册和 CPU 预检。只用运行前静态分层：7层配额3/3/2/2/2/2/2，SHA256(sample_id)排序选16 anchors，seeds6311/6312/6313，MH-CEM K4/rho0.1，新增B256→B512共96 cases；48份B1024父级raw身份与归档SHA通过，未重跑。3080Ti可达且checkpoint正确，远端仍是旧StageA代码，必须同步本轮已提交源码并重新核验身份后才能启动。当前0/96，不是校准结果PASS。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。

证据：`docs/implementation_records/STEP_06_4C_MH_BUDGET_CALIBRATION.md`；`code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003/08_cpu_preflight_receipt.json`。

---

## 2026-10-03 STEP 6.4B 最小闭环真实反馈验收（当前）

STEP_6_4B=PASS，TWO_CYCLE_REAL_FEEDBACK_SMOKE=PASS，CLOSED_LOOP_MECHANISM_READINESS=PASS。固定TRAIN fixture、MH-CEM K4/rho0.1、CPU FP32 batch1、B64：仅执行首轮winner第一动作，仿真0.6→0.7秒，真实新观测重建state/graph/current posterior并完成第二次规划；下一轮root不是上一轮预测。两轮各64独特转移（合计128）；另保留一次动作执行前回执异常的64次尝试，实际总计192。live tensor/deadline、旧搜索不变性、防未来泄漏及参数/归一化不变全部PASS。SEARCH_METHOD=MH-CEM仍仅pure-search backbone；READY_FOR_BUDGET_CALIBRATION=true只是机制就绪，不是运行授权。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN；future Return-birth fixed-support limitation保持。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并单独授权预算校准协议，不自动运行。

记录：`docs/implementation_records/STEP_06_4B_MINIMAL_CLOSED_LOOP_TWO_CYCLE_SMOKE.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/13_final_acceptance.json`；合同：`docs/contracts/PIJWM_STEP_06_4B_LIVE_CLOSED_LOOP_SMOKE_V1.md`。以下为历史记录，其中“当前”仅表示记录时点。

---

## 2026-10-03 STEP 6.4A 闭环前就绪审计（当前）

STEP 6.4A审计完成（审计证据PASS），CLOSED_LOOP_READINESS=BLOCKED。SEARCH_METHOD=MH-CEM仍只为pure-search backbone，非最终hybrid。局部链路7 IMPLEMENTED/3 PARTIAL/5 MISSING；缺live history-only输入/belief/sidecar、winner动作序列/首动作及真实ID执行桥、研究者fallback和latency决定、两轮真实反馈MH证据。MH B1024搜索median90.506/P90 204.916/P95 284.152/max421.701秒；仿真slot0.1秒不是墙钟deadline，DECISION_LATENCY_REQUIREMENT=UNRESOLVED，尚无在线执行证据。Stage B用途由研究者冻结为online-budget trade-off，不能重选算法；A约30.46–41.85h、B NEW PROPOSAL约10.26–14.06h、C NEW PROPOSAL约1.54–2.11h，仅成本情景。4090_MIGRATION_NOT_YET_JUSTIFIED。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作研究者确定fallback及decision latency模式，再授权最小接口集成；不自动运行实验。



记录：`docs/implementation_records/STEP_06_4A_CLOSED_LOOP_READINESS_STAGE_B_PURPOSE_FREEZE.md`；机器审计：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/06_go_no_go_receipt.json`。下方为历史状态，不能用Stage A PASS替代闭环readiness。

---

## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-01 STEP 6.3D FORMAL VALIDATION STAGE A（当前运行）

研究者已单独授权并启动 Formal Validation Stage A（仅 B1024）：64锚点×5 seeds×3方法=960 cases，名义预算983,040。2026-10-01北京时间19:26在RTX3080Ti/CUDA/FP32/batch16启动，启动前源码/768 TRAIN/771文件SHA/64锚点/checkpoint/执行身份全部PASS，Validation起点0。首批2个case已通过身份检查并持久备份；这是运行快照，不是最终效果结论。STEP_6_3D_VALIDATION_STAGE_A=RUNNING，VALIDATION_COMPARISON=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；future Return-birth限制保留。唯一下一动作是监控并备份Stage A，960完成后独立验收并停止。选择范围仅Planner v1 structured-search backbone/pure-search baseline，不是最终hybrid planner冻结。

记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；启动前机器收据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/01_launch_preflight_receipt.json`；动态监控与本地备份在该运行目录。下方均为较早阶段快照，关于未授权/Validation0的表述不能代表当前运行。

# PI-JWM 实施记录

> 2026-10-01 当前：PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。

> 2026-10-01 当前：STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。

**当前 STEP 6.3B/6.3C-PATCH：** [Comm Task Selection & TRAIN Projected H1 Audit](STEP_06_3BC_PATCH_COMM_TASK_SELECTION_TRAIN_SELF_REPLAY_AUDIT.md) 记录研究者新任务选择边界、原始/投影/排除行、全量候选域与残余。完成本补丁后停止；STEP 6.3D 需单独授权。

**当前 STEP 6.3B：** [Structured Candidate Grammar and Support Policy](STEP_06_3B_STRUCTURED_CANDIDATE_GRAMMAR_AND_SUPPORT_POLICY.md) 记录 TRAIN-only 证据、研究者准入决定、CPU 语法/validator 和明确未开始的 optimizer/闭环。

**当前 STEP 6.2B-PATCH：** [Route no-op closure](STEP_06_2B_PATCH_ROUTE_NOOP_CLOSURE.md) 记录研究者新 Route 域、冻结 checkpoint CPU 验收及 `STEP_6_2B=PASS` 的严格范围；此前 [STEP 6.2B](STEP_06_2B_PLANNER_OBJECTIVE_SCORER_AND_COMPARATOR.md) 的 BLOCKED 观察保留历史。

当前工作以研究者 2026-09-18 授权的只读 `00–06` 定义为目标，以仓库 source/config/test/artifact 为实现事实。总体状态见 [Implementation Tracker](../PIJWM_IMPLEMENTATION_TRACKER.md)。

## 记录目录

| Step | 记录 | 范围 | 状态 |
| --- | --- | --- | --- |
| STEP 6.1 | [STEP_06_1_TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT_PREFLIGHT.md](STEP_06_1_TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT_PREFLIGHT.md) | 冻结 best checkpoint、同一当前 belief、四族合法动作、H4 递归 prior-only 推演、随机配对和 CPU 批量诊断 | 机制预检 PASS；无 objective/winner/闭环性能声明 |
| STEP 5.1A | [STEP_05_1A_MOTION_CSI_TARGET_CONTRACT.md](STEP_05_1A_MOTION_CSI_TARGET_CONTRACT.md) | Future Motion/CSI target、mask、stable support alignment、normalization 与 raw-rule bridge | COMPLETE；non-locked development evidence，不是正式 Dataset |
| STEP 5.2 | [STEP_05_2_TRAINING_LOOP_CURRICULUM_JOINT_TRAINING.md](STEP_05_2_TRAINING_LOOP_CURRICULUM_JOINT_TRAINING.md) | CPU training loop、posterior warm-up、prior recursive curriculum、KL schedule、validation、checkpoint/resume | COMPLETE；CPU development smoke，不是 formal training/performance |
| STEP 5.3 | [STEP_05_3_CPU_TRAINING_PREFLIGHT_TINY_OVERFIT_GO_NO_GO.md](STEP_05_3_CPU_TRAINING_PREFLIGHT_TINY_OVERFIT_GO_NO_GO.md) | 固定 dev_train tiny subset、Stage 1/2 capacity、H1/H2 recursive learning、learning-signal、resume 与 Go/No-Go | GO；development-only CPU preflight，不是 formal training/performance |
| STEP 5.3E | [STEP_05_3E_CSI_TRAIN_MEAN_BIAS_FORMALIZATION_TINY_OVERFIT_ACCEPTANCE.md](STEP_05_3E_CSI_TRAIN_MEAN_BIAS_FORMALIZATION_TINY_OVERFIT_ACCEPTANCE.md) | raw CSI decoder train-only mean bias formalization、checkpoint identity、固定 tiny-overfit acceptance | FORMALIZATION_PASS / TINY_OVERFIT_GO；CPU development evidence，不是 formal training/performance |
| STEP 5.4 | [STEP_05_4_GPU_TRAINING_READINESS_FORMAL_TRAINING_PREPARATION.md](STEP_05_4_GPU_TRAINING_READINESS_FORMAL_TRAINING_PREPARATION.md) | manifest-driven formal interface、formal data/action coverage、config/checkpoint/device readiness | BLOCKED；formal Dataset 与研究者参数未决，GPU 未执行 |
| STEP 5.5 | [STEP_05_5_FORMAL_DATASET_V1_BUILD_ACCEPTANCE.md](STEP_05_5_FORMAL_DATASET_V1_BUILD_ACCEPTANCE.md) | 60 条真实 trajectory、H2/L4 五类 package、四动作 coverage、deterministic rebuild 与 CPU trainer smoke | COMPLETE；FORMAL_DATASET_READY，formal training 仍等待 5.6A 配置冻结/GPU smoke |
| STEP 5.6A | [STEP_05_6A_GPU_SMOKE_FORMAL_CONFIG_EVIDENCE.md](STEP_05_6A_GPU_SMOKE_FORMAL_CONFIG_EVIDENCE.md) / [STEP_05_6A_CONFIG_FREEZE.md](STEP_05_6A_CONFIG_FREEZE.md) | Formal Dataset H4 CUDA smoke、全量 prior-only validation、trajectory-aware sampler、Formal Training Config v1 与 availability bookkeeping | GPU smoke/full validation PASS；config FROZEN；bookkeeping correction CPU-only；formal training 未开始 |
| STEP 5.6B | [STEP_05_6B_FORMAL_GPU_TRAINING_LAUNCH.md](STEP_05_6B_FORMAL_GPU_TRAINING_LAUNCH.md) | 冻结配置正式 runner、独立 Go/No-Go、日志/心跳/checkpoint 与 detached GPU launch | source 提交时为预启动；实时状态以远端 run manifest/heartbeat 为准 |
| STEP 5.6C | [STEP_05_6C_FORMAL_TRAINING_FINAL_ACCEPTANCE.md](STEP_05_6C_FORMAL_TRAINING_FINAL_ACCEPTANCE.md) | 本地 CPU 5520 步/五次 validation 验收、最终 best checkpoint 身份冻结与单样本 H4 推理 | COMPLETE；Formal Validation Observation，无 baseline/locked_test/Planner rollout |
| STEP 1 | [STEP_01_AUDIT.md](STEP_01_AUDIT.md) | 新定义与现有实现审计、治理与导航同步 | 见记录中的验证和 Git 状态 |
| STEP 1 数据/双图附件 | [STEP_01_DATA_GRAPH_AUDIT.md](STEP_01_DATA_GRAPH_AUDIT.md) | 01–03 定义、时间、张量、实体和动作映射 | 支撑证据，不是下一 Step |
| STEP 2 | [STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md](STEP_02_RAW_TRAJECTORY_ACTION_CONTRACT.md) | 单决策步 Raw Trajectory 与四类动作合同 | COMPLETE |
| STEP 2.1 | [STEP_02_1_REAL_AIRFOGSIM_SINGLE_STEP.md](STEP_02_1_REAL_AIRFOGSIM_SINGLE_STEP.md) | 真实 AirFogSim 单步接线与对齐 | COMPLETE |
| STEP 2.2 | [STEP_02_2_REAL_AIRFOGSIM_MULTI_STEP.md](STEP_02_2_REAL_AIRFOGSIM_MULTI_STEP.md) | 真实 AirFogSim 6 步 Raw Trajectory | COMPLETE |
| STEP 2.3 | [STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md](STEP_02_3_RAW_CONTRACT_CAUSAL_COMPLETENESS.md) | Raw 因果可观测、真实字段、return route 与 acceleration 最终冻结 | COMPLETE；Raw Trajectory Layer / 01 FROZEN |
| STEP 2.4 | [STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md](STEP_02_4_COMMUNICATION_OUTCOME_SEMANTICS.md) | wireless/wired Communication Outcome 拆分、total 与 empty/missing 语义 | COMPLETE；Raw Trajectory Layer / 01 FROZEN |
| STEP 3.1 | [STEP_03_1_MODEL_READY_SAMPLE_TENSOR_CONTRACT.md](STEP_03_1_MODEL_READY_SAMPLE_TENSOR_CONTRACT.md) | Model-ready sample、时间窗口、index/mask、四类 action 与最小真实 tensor contract | COMPLETE；经 STEP 3.1R 修正；正式大数据集未开始 |
| STEP 3.1R | [STEP_03_1R_MODEL_READY_SAMPLE_CONTRACT_CORRECTION.md](STEP_03_1R_MODEL_READY_SAMPLE_CONTRACT_CORRECTION.md) | 修正 History、固定 index/presence、真实 DAG、typed target index、relation 和 Action reference | COMPLETE；STEP 3.2 未开始 |
| STEP 3.1F | [STEP_03_1F_MODEL_READY_HISTORY_CONTRACT_FINALIZATION.md](STEP_03_1F_MODEL_READY_HISTORY_CONTRACT_FINALIZATION.md) | 收尾 History 中 past Action/Outcome、History union index、历史 relation/DAG/Flow 对齐、Future Action namespace 修正和 future-reference 观察审计 | COMPLETE；含 3.1F-PATCH；STEP 3.2 未开始 |
| STEP 3.2 | [STEP_03_2_RAW_TO_DATASET_BATCH_SPLIT_PREPROCESSING.md](STEP_03_2_RAW_TO_DATASET_BATCH_SPLIT_PREPROCESSING.md) | Raw-to-Dataset 批量、trajectory-level split、causal windows、train-only preprocessing、batch/load 验证 | COMPLETE；non-locked validation bundle；正式 Dataset/训练未开始 |
| STEP 3.3 | [STEP_03_3_MODEL_INPUT_TENSOR_COLLATION_CONTRACT.md](STEP_03_3_MODEL_INPUT_TENSOR_COLLATION_CONTRACT.md) | fixed-shape CPU tensor/collation、Past Outcome、Target、四类 action、stable vocab/mask | COMPLETE / FROZEN；双图/模型未开始 |
| STEP 4.1 | [STEP_04_1_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING.md](STEP_04_1_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING.md) | Physical / Information 对象—字段—关系映射、数据缺口和旧实现冲突 | COMPLETE / FROZEN mapping；graph builder 未开始 |
| STEP 4.2A | [STEP_04_2A_EXISTING_SOURCE_GRAPH_INPUT_ADDITIVE_EXTENSION.md](STEP_04_2A_EXISTING_SOURCE_GRAPH_INPUT_ADDITIVE_EXTENSION.md) | 已有来源的 graph minimum inputs 贯穿 Raw amendment、Sample、preprocessing 与 Tensor | COMPLETE；graph builder 未开始 |
| STEP 4.4 | [STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL.md](STEP_04_4_STRUCTURED_RSSM_WORLD_MODEL.md) | Structured RSSM、known stochastic outage、deterministic transition 与 dynamic graph rollout | COMPLETE / FROZEN；PATCH3 92/92；untrained CPU model evidence；5.1B/5.2 consumes it |
| STEP 5.0 | [STEP_05_0_DEFINITION_05_DECISION_FREEZE.md](STEP_05_0_DEFINITION_05_DECISION_FREEZE.md) | Definition 05 十项研究决定冻结、旧 Loss/Training/Evaluation 定向复用审计 | DECISION FROZEN；5.1B/5.2 implementation follows；无 full training/GPU/locked-test |

不预建貌似已经执行的后续 Step 文件。`00–06` 是研究定义章节，不是可以自动执行的七个工程 Step。

## 每步记录要求

每个 Step/Substep 记录必须包含：Step Goal、Definition Basis、Initial State、Files Involved、Changes、Reuse、Validation、Results、Expected vs Actual、Known Issues、Git、Next Step。

- Definition Basis 写清文件、节号和目标；源文件哈希保存在对应 audit 中。笔记只读，工程映射留在本仓库，不整库复制笔记。
- 实现判断只用 `DIRECT_REUSE / MINOR_MODIFICATION / STRUCTURAL_CHANGE / MISSING / RESEARCHER_DECISION_REQUIRED / HISTORICAL_ONLY`。执行进度（如 NOT_STARTED）另列，不能与复用判断混用。
- 验证写实际命令、退出码、测试数量及日志路径；旧测试通过不等于新方法通过。
- 不移动或改写历史数据、实验、协议和 checkpoint；逻辑归档为 Historical / Archived。
- 每一步完成：验证 → Tracker/记录/AI_CONTEXT → diff → commit → push → 固定格式汇报 → 停止。不得自动进入建议下一步。
- Git hash 无法写入包含自身的同一 commit。正文用唯一 commit message 定位主提交，后续 Git 回执记录真实 hash 与 push 结果；回执提交自身用 Git log 定位。

## 成本与边界

正式训练之前先完成新合同下的 schema/tensor、单测、tiny forward/backward、finite gradient、tiny-data overfit、短 smoke、动作敏感性、rollout sanity 和 learning signal 验证，再向研究者提出训练请求。记录目录不是自动训练队列。`locked_test` 保持封存。
**当前 STEP 6.3D-3080TI-MIGRATION-QUALIFICATION：** [迁移与 CUDA runner 验收](STEP_06_3D_3080TI_MIGRATION_QUALIFICATION.md) 记录正式数据/冻结 checkpoint SHA、CPU/GPU 等价、batch16 选择、formal runner/resume 和 bounded TRAIN smoke；正式调参/Validation 比较尚未运行。
