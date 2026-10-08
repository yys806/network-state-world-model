<!-- STEP6.4H CPU CURRENT -->
## 2026-10-08 STEP 6.4H — S-CEM 预算资格：CPU 前置门

当前授权为独立 S-CEM K4/rho0.2 的 B512 vs 已有 B1024；方法不重新选择。B1024 的320份修复域原始结果与ZIP内文件逐份SHA、source/config/checkpoint身份核验PASS。新B512=0/320，名义预算163840；没有启动GPU或搜索。协议、64×5清单、独立身份、原子保存/严格恢复、六类配对、目标首差及资格门已实现；CPU检查通过后 READY_FOR_GPU_LAUNCH=true，等待研究者开启服务器并续跑。

SEARCH_METHOD=S-CEM；S_CEM_B512_QUALIFICATION=NOT_RUN；RECOMMENDED_CLOSED_LOOP_B_WM=PENDING_EVIDENCE；FINAL_CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING。6.4G Phase B=NOT_STARTED_BY_CONDITIONAL_STOP，原Stage B=NOT_STARTED/DEFERRED，历史停止门不变。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；GPU=NOT_USED；locked_test=false。正式闭环仍等待预算证据与研究者决定。服务器容量、设备和无旧runner检查须在后续启动前实测，当前未探测服务器。

证据：docs/implementation_records/STEP_06_4H_S_CEM_BUDGET_QUALIFICATION.md；code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2/。以下6.4G及更早章节保留为此前状态，不是当前执行指令。
<!-- END STEP6.4H CPU CURRENT -->

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

以下为历史记录，旧结果仅在旧资格定义下成立。

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

# PI-JWM 当前科研状态

> 2026-10-01 当前：PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。

> 2026-10-01 当前：STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。

> 2026-10-01 STEP 6.3D FORMAL TRAIN TUNING CLOSURE：768/768 TRAIN 搜索调参已在 RTX 3080 Ti 完成并逐项验收、SHA 归档；S-CEM/MH-CEM 各冻结 `(K=4,rho=0.1)`。每组 48/96 cases 有可评分 H4，16/32 锚点无可评分结果，future Return birth 固定支持限制保持。`VALIDATION_COMPARISON=NOT_STARTED`、`SEARCH_METHOD=NOT_SELECTED`、`locked_test=false`。正式 TRAIN 耗时 21.38 小时；方法效果必须等后续 Validation，不从 TRAIN 选参推出。

> 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH：`PASS` 只验收修正后 TRAIN 32 / Validation 64 锚点的静态 Objective 资格和 CPU 批量执行路径。TRAIN-only 固定 HRS seed6391/B_WM64 的 32-anchor 诊断中 16 个锚点找到可评分 H4、16 个没有；275 条不可评分完整 H4 均触发 future Return birth 支持边界。`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`，正式 TRAIN tuning/Validation 比较未运行，方法未选。证据见 20–25 收据和 Patch 实施记录；无 GPU、`locked_test`、训练或闭环。

> 2026-09-29 STEP 6.3B/6.3C-PATCH：研究者已冻结 Comm 子集选择和 TRAIN 通信结构条件数量支持；原“所有 eligible Task 都有行”的旧限制不再适用。4416 TRAIN H1 投影中任务数量拒绝为 0，728 条依赖同决策 offload Route 的历史 Comm 行已排除并逐条记录；其他 Comp/联合结构残余没有放宽。静态域 TRAIN 14/4416、Validation 6/1104 为空。证据只涉及 CPU 候选准入与机制；优化器、排名、baseline、闭环、GPU 和 `locked_test` 均未进入。

> 2026-09-29 STEP 6.3B：候选语法与支持政策 `PASS`（CPU 合同）。正式池 Route 空、Comm 当前关系绑定、Comp 当前因果 CPU base、Mob 共享 profile，联合结构须为 TRAIN 已见签名；时间观察性只作标签。仍未选候选优化器、未排名、未进入闭环或性能验证。见 STEP 6.3B 合同与收据。

> 2026-09-28 STEP 6.3A：Formal TRAIN action-support 证据审计 `PASS`；Comp causal template 可复现，独立 family factorization 不受支持。候选算法仅有证据建议、未由研究者冻结；无 optimizer、ranking、闭环或 baseline。

> 2026-09-28 STEP 6.2B-PATCH（当前）：研究者冻结 Planner v1 Route `EXPLICIT_NOOP_ONLY`，非空 Route 准入拒绝；`STEP_6_2B=PASS`、`MPC_OBJECTIVE=FROZEN_AND_IMPLEMENTED` 只表示五项 scorer/字典序的 CPU 合同验收。候选方法待决、闭环未就绪；无 Route 优化、性能、baseline、GPU 或 locked-test 结论。下方 6.2B BLOCKED 是历史观察。

> 2026-09-28 STEP 6.2B：冻结 Objective 五项 scorer 与严格字典序已实现并通过一条 validation anchor 的 CPU checkpoint 机制验证，但 pending Route 固定支持和同路径 Host 更新与现有合同存在冲突，`STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`。不把 scorer 接通视为 Planner 方法或性能验收；candidate method、closed loop、baseline、GPU、locked_test 均未开始。

> 2026-09-28 STEP 6.1：`TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT=PASS` 限于一个 Formal Validation anchor/冻结 seed 5601 的 CPU 机制验收，四类当前因果动作的 H4 路径、对应潜变量和规则响应有机器收据。此状态不证明反事实预测准确或 Planner 控制效果；`MPC_OBJECTIVE=NOT_STARTED`、`CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`、`CLOSED_LOOP=NOT_STARTED`，baseline/locked-test/performance claim 均无。

> 2026-09-28 STEP 5.6C：正式 seed 5601 训练已完整结束并通过本地 CPU 验收，`STEP 5.6B=COMPLETE`、`FORMAL_BEST_CHECKPOINT=FROZEN`。五次完整 validation 的最终严格最低 `L_Val=0.07431338784170399`，best/latest 均为 step 5520 且模型张量逐项相同；这是 Formal Validation Observation。之前 6.0A–C 段落对 5.6B“未查询”的说法是各自历史时点，不是当前状态。仍无 baseline、locked-test、模型候选 Planner rollout 或闭环性能结果。

> 2026-09-26 STEP 6.0C：Planner v1 operational action domain 已按研究者决定冻结并做 CPU 合同测试；Search/Learned/Hybrid、目标函数、模型候选 rollout、性能、闭环与安全性仍未决定或未执行。动态可用 CPU 的 simulator source 仍不存在；本域改用静态预算。5.6B 远端状态本 Step 未查询。

> 2026-09-26 STEP 6.0B：CPU verdict=`STATIC_CAPACITY_ONLY`；UAV 配置和行为数据未构成 simulator hard bound。两项约束保持 UNKNOWN，未开始模型依赖候选推演。AirFogSim 本地源码可读，但缺独立 Git metadata，精确本地 Git SHA 不能确认；5.6B 远端未查询。

> 2026-09-26 最新增量：5.6B 正式训练由研究者先前授权在远端独立运行，本 Step 未联系远端，当前进度和 best checkpoint 未核实。6.0A 只通过 CPU 静态候选合同与合成 fixture 验收；最终候选方法、World Model 候选 rollout 和性能仍待后续授权。下文旧的“未正式训练/GPU”属于当时快照。

> STEP 5.5-PATCH 已补齐 Formal Dataset 60 trajectory shard 的全量索引与按需 CPU batch 消费；原 1+1 runtime smoke 不再承担 full-shard 证明。future Return birth 全量检测为 8828 次按窗口-未来步计数的 unsupported/fixed-support 事件，原零计数已失效。未正式训练，未用 GPU，未访问 `locked_test`。

> 本文是导航性状态摘要。当前结果必须回到原始 checkpoint、metrics、manifest 和 audit 验证。
> 截至 2026-09-23，STEP 5.5 Formal Dataset v1 已接受：60 条真实 trajectory、H2/L4、48/12 split、5520 windows、五类 package 与 CPU H=4 interface smoke 均通过。GPU 未使用，`locked_test` 未访问，formal training 未开始。

## 0. 当前实施状态

- Step 2.4 已用真实 non-locked AirFogSim 完成通信 Outcome 最终验收；Step 3.2/3.3 已冻结最小 Dataset/Tensor 合同。
- STEP 4.1 已冻结映射；STEP 4.2A/4.2C-B/4.2C-C 已闭合 graph inputs 与 Flow Raw→Tensor；STEP 4.3A 已物化 typed graph；STEP 4.3B 已实现 History temporal encoding、typed message passing 与 P2A/P2C，输出 `Z_t^{PI,L_g}`。
- 现有代码已在 STEP 4.4 contract 中接入四类动作路由、`Z_t^{PI,L_g}→xi_t^Lat`、结构化 RSSM 边界、逐步规则反馈与预测态动态图重建；5.1B/5.1D 已接入 Loss/KL/Metric，5.2 已接入 CPU training loop；真实重规划与性能验证仍未开始。
- STEP 5.0 已冻结 deterministic mean decoder、Motion/CSI mask-normalized MSE、family-specific posterior teacher、分族 KL、overshooting OFF、prior-dominant curriculum、component mask、无规则状态 loss、joint training 与 prior-only validation/evaluation。
- STEP 5.1A-PATCH 已把 Motion 改为 local one-step delta 并固定到 current physical input slots，同时把 CSI 绑定到 current model relation slot/identity/endpoint/type/RB；5.1B/5.1D 已完成 posterior/loss/metric 与 paired integration，5.2 已完成 CPU training-loop smoke。下一独立门是 STEP 5.3；这不是正式训练或性能证据。

## 1. 当前研究问题

在车联网、无人机、路侧单元和边缘计算组成的动态系统中，如何在严格区分物理连接与业务数据流的前提下，学习动作条件的多步联合状态演化，并最终支持基于预测后果的候选动作选择和滚动重规划。

## 2. 旧协议方法边界（Historical / Archived）

被审计的旧协议候选方法是 `entity_aligned_dual_graph_rssm_v1`：

- 双图底座分别表示物理网络和信息网络；
- 跨图消息只沿真实附着和承载关系传播；
- 节点、物理边、数据流和任务拥有实体级随机动力学；
- 运动状态使用因果历史差分；
- 训练时使用 posterior teacher，正式预测使用 prior-only；
- 先训练 deterministic base，再冻结 base 训练 RSSM；
- 通过 P4 非锁定数据门后，才讨论 P6 规划。

该方法及其两个 seed 结果仅保留原协议含义，不是新定义候选已经确定或实现。

## 3. 已有证据

### 3.1 方法机制和执行证据

- 第一性原理审计发现旧 global RSSM 的实体表达、链路排序和运动输入问题。
- 实体级 RSSM CPU 一致性审计通过了因果运动、动作敏感性、prior target invariant、posterior 局部性、梯度和 strict reload 等机制检查。
- GPU batch probe 选择 batch size 8；execution sentinel 只证明 CUDA 路径可执行，不承担正式性能结论。
- 冻结协议位于 `code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

### 3.2 seed 20260831

正式单 seed 验收已通过，详情见 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`。

| 指标 | 数值 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| validation link-F1 delta | `+0.44415` | `>= -0.05` | 通过 |
| calibration link-F1 delta | `+0.86229` | `>= +0.05` | 通过 |
| node-x overall ratio | `0.75475` | `<= 1.25` | 通过 |
| node-x h5/h10/h20 | `0.76995/0.75071/0.75495` | 各 `<= 1` | 通过 |
| throughput/RB/task delay | `0.93555/0.47363/0.01287` | 各 `<= 1` | 通过 |

该结果只覆盖一个 seed 的 unlocked 数据，不能推出跨 seed 泛化，也不能打开 locked test。

### 3.3 seed 20260830

正式单 seed 验收已通过，详情见 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`。

| 指标 | 数值 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| validation link-F1 delta | `+0.45710` | `>= -0.05` | 通过 |
| calibration link-F1 delta | `+0.88509` | `>= +0.05` | 通过 |
| node-x overall ratio | `0.75086` | `<= 1.25` | 通过 |
| node-x h5/h10/h20 | `0.76033/0.74939/0.74960` | 各 `<= 1` | 通过 |
| throughput/RB/task delay | `0.93831/0.47523/0.01175` | 各 `<= 1` | 通过 |

最佳 RSSM checkpoint 为 epoch 40。正式 manifest 共 79 项，零缺失、零哈希差异；strict reload 为 `0/0`，机器重算最大差异为 `4.15555434507553E-09`。

## 4. 当前运行状态

远端训练进程已退出，GPU 已释放。当前没有获准继续运行的训练；旧 `20260832` 不得自动启动，新 Step 2 也未获授权。

## 5. 当前阻塞和未知

- 新定义关键模块尚未实现，不能用旧 P4/P6 状态替代。
- 通信状态充分性、外生事件、proposal 训练、planner objective/risk/fallback 和新实验门仍需冻结。
- 旧 P4 缺第三 seed 和三 seed审计，但它已经不在 active execution queue。
- `locked_test` 保持封存。
- `formal_performance_claim_ready=false`。
- 两个已完成 seed 均通过单 seed 门，但三 seed 的均值、方差、稳定性和正式泛化结论仍未知。
- 最终采用纯候选搜索、学习策略还是混合策略未知。

## 6. 历史失败的科研价值

历史 global complete RSSM、node-x-safe、edge feedback、persistence residual、概率校准等实验不能当作当前方法，但它们解释了为什么当前方案需要实体级 latent、因果运动输入、逐边修正和分阶段训练。失败结果应保留并通过 `EXPERIMENT_INDEX.md` 和原始 audit 追踪。

## 7. 当前科研决策权

- 用户决定研究问题、核心假设、方法取舍、实验目的和最终科研解释。
- AI负责实现、测试、运行、审计、整理和提出有证据的质疑。
- 当理论、代码、数据、运行配置和结果不一致时，先报告冲突，停止扩展受影响实验，不用改名或模糊表述掩盖。

## 8. 单一下一动作

研究者检查已冻结 Raw 层；若明确授权，再进入 Step 3 Dataset/Tensor Contract。在此之前不启动训练、不访问 `locked_test`。

## 2026-09-19 STEP 3.1F 状态

最小 Model-ready Sample & Tensor Contract 已冻结：History `[1,2]` 中明确包含 `O_1+A_1+Y_1+O_2`，Future Action/Target `[2,3]`，input index 来自 History union，Future Action 先做 anchor visibility 检查后引用同一 static index，固定 presence/mask、typed target index、真实 DAG 因果分区、history relation/flow 和严格 Action reference 均已验收。机器 policy 为 `history_causal_observable_object_union`；真实非 locked Raw、样本和合同测试通过；已扫描 4 个 Raw artifact 的 18 个窗口且当前无 unresolved future reference，audit JSON 的 SHA-256 已进入 manifest provenance。该扫描不代表正式 dataset 可用率；正式 batch/split、模型、训练和 `locked_test` 仍未开始。STEP 3.2 未授权。

## 9. 项目重构状态

- 人类可读的项目、代码、架构、实验、结果、冲突、归档、学习和长期协作入口已经建立。
- 机器注册表覆盖 848 个项目文件、607 个 Python 节点和 803 个 artifact 一级目录；Python 解析错误为 0，当前 artifact 控制入口读取错误为 0。
- `AI_CONTEXT/` 九个 ChatGPT 上下文文件已通过结构和证据边界检查；问题路由增至 7 类。
- 7 个重要实验使用统一完整字段，2 个正式结果已与原始 acceptance JSON 自动逐项核对，证据不一致数为 0。
- 6 类重要历史方法记录了尝试原因、实际结果、弃用原因、替代关系和原始证据；5 类常见问题已有只读路由。
- 查询工具既能按自然语言定位当前方法、历史尝试、结果来源和延后任务，也能按精确文件名或 artifact 目录名检索；输出只负责导航，正式结论仍回到原始证据核实。
- 第三个 seed 和远端同步已登记为 `deferred`、`authorization_required=true`、`auto_start=false`。
- 历史 Python 代码先做逻辑归档；因为 94 个历史节点仍被历史脚本/测试引用且全量测试基线仍有环境与 fixture 错误，本轮不做破坏 provenance 的物理移动。
## STEP 6.2A-PATCH (2026-09-28)

Objective source reconciliation and a single non-locked Formal Validation deadline sidecar passed. E2E Flow state exists. Route action and cross-hop model route semantics remain inconsistent with 4.2C-C; `STEP_6_2B_READINESS=BLOCKED`. No scorer, candidate ranking, baseline, GPU, locked_test or closed-loop result exists from this Patch.
## 2026-09-28 STEP 6.2A-CLOSURE

研究者接受现有 Formal best checkpoint、不重训；Planner v1 Route 限定为 Formal Dataset single-hop support。当前只达到 `READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`，不代表 Planner、ranking、closed loop、baseline 或 performance ready。Formal multi-hop coverage 为零，multi-hop code 保留作未来扩展/消融。
2026-09-30：`STEP_6_3D_3080TI_MIGRATION_QUALIFICATION=PASS`，仅表示正式数据迁移、GPU FP32 等价、runner/resume 和 bounded TRAIN smoke 就绪。正式 TRAIN tuning、Validation 方法比较、locked_test 未运行，搜索方法未选；future Return birth 固定支持限制保留。
