<!-- STEP6.4G CURRENT -->
## 2026-10-05 STEP 6.4G — Phase T 已验收

STEP_6_4G=IN_PROGRESS；Phase T=PASS，768/768全部新结果独立验身份/预算/选参与本地D:归档通过；名义/实际unique一步转移393216，正式矩阵32.5955小时。修复域新冻结 S-CEM=(K4,rho0.2)，MH-CEM=(K3,rho0.1)，两方法四配置均48/96 H4可评分；这只属于TRAIN选参观察，不能据此选择搜索方法。scorer exception/inconsistency=0。Phase A=NOT_STARTED（新960 B1024）；Phase B=NOT_STARTED（只有新A选MH才320 B512）。SEARCH_METHOD_REQUALIFIED=PENDING，FORMAL_SEARCH_REQUALIFICATION=PENDING，修复域最终budget待新证据；旧MH/K4rho0.1/B512只在historical-under-pre-6.4F-domain/working决定范围保留，不复用任何旧raw。FINAL_FALLBACK_POLICY=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1；只尝试一次canonical A失败则C，不重试。future Return-birth fixed-support限制保留。CLOSED_LOOP_PRE_FORMAL_READINESS=PENDING_SEARCH_REQUALIFICATION，READY_FOR_FORMAL_CLOSED_LOOP_PROTOCOL=false；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN，旧Stage B=NOT_STARTED / DEFERRED，locked_test=false。唯一下一动作：T阶段Git门完成后，同步服务器至精确新commit并启动PhaseA；科学源码SHA不变。

证据：`code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004/T/acceptance.json`、`T/local_backup_acceptance.json`、`T/selected_configs.json`、`T/inventory.json`、`T/archive.json`；新证据=repaired-domain formal evidence。

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

## 2026-10-01 STEP 6.3D FORMAL TRAIN TUNING CLOSURE（当前）

`STEP_6_3D_FORMAL_TRAIN_TUNING_CLOSURE=PASS`：RTX 3080 Ti 上正式 TRAIN-only 搜索调参 768/768 份原始结果已逐份验身份、SHA 归档并独立重算选参。名义和实际 World Model 一步转移均为 393,216；S-CEM、MH-CEM 分别冻结 `(K=4,rho=0.1)`。所有八组参数各 48/96 次找到可评分 H4；16/32 锚点始终未找到。future Return birth 固定支持限制保留，`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`。这只是 TRAIN 选参，不是方法优劣或系统性能结论。`VALIDATION_COMPARISON=NOT_STARTED`、`SEARCH_METHOD=NOT_SELECTED`、`locked_test=false`。原始结果、日志、收据和本机归档 SHA 见 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/`；说明见 `docs/implementation_records/STEP_06_3D_FORMAL_TRAIN_TUNING_CLOSURE.md`。下方段落均为较早快照。

## 2026-09-30 STEP 6.3D-GPU-EXECUTION 当前状态

`STEP_6_3D_GPU_EXECUTION=PASS`：RTX 4090 24GB 上冻结 checkpoint 的 FP32 CPU/GPU 单步与 H4 离散等价通过，测试 batch 1/4/8/16/32/64/128/256；batch 128/256 正式吞吐探测 OOM。推荐不分桶 batch=32，稳定中位数约 12.71 unique transitions/s、峰值约 2.03 GiB，约 13.19x CPU batch8；完整 2,113,536 transitions 理论约 46.2 小时。未运行正式 TRAIN tuning、Validation comparison、方法选择、训练、闭环或 `locked_test`。收据见 `code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930/`。

# PI-JWM Current State Snapshot

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH 当前关口

`STEP_6_3D_PREFLIGHT_PATCH=PASS` 只表示样本输入与 CPU 批量执行路径就绪。旧 32 TRAIN / 64 Validation 锚点中 3/5 个零 Objective cohort 已按原分层/哈希静态替换，修正后 32/64 全部候选域非空、cohort>0；8 个新 deadline sidecar 精确对齐。固定 TRAIN-only HRS seed6391/B_WM64 诊断：16/32 锚点找到可评分 H4，16/32 未找到，275 条不可评分完整 H4 均有 future Return birth 支持边界。CPU batch 1/4/8/16 一步与 H4 串行等价。`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`：16/16 分布不满足“多数可评分”，研究者已确认 future Return birth 是固定支持边界限制；正式 6.3D 可以继续，H4 scoreable success rate 单独报告。正式 TRAIN tuning/Validation 方法比较未运行，方法未选；GPU 执行路径已闭合，未访问 `locked_test`、训练或闭环。详见 `docs/implementation_records/STEP_06_3D_PREFLIGHT_PATCH_OBJECTIVE_H4_BATCH.md` 和 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/25_preflight_patch_acceptance.json`；下方 6.3D 段落是补丁前状态。

## 2026-09-29 STEP 6.3D 进行中

研究者已冻结 HRS、S-CEM、MH-CEM 的同域同预算 H4 方法比较；本机仅使用 CPU。TRAIN 32、Validation 64 个非空 anchor 的确定性清单与 96/96 对齐的因果 deadline sidecar 已建立；三种方法共享结构化 proposal / CandidateDomain / 一步 World Model 转移 / 6.2B 字典序 Objective 的代码与 9/9 CPU 定向测试已通过。真实模型探针和首条 B_WM=512 TRAIN solve 均未取得该单锚点的 H4 可评分候选。完整 TRAIN tuning 与 Validation 三预算比较尚未完成，`STEP_6_3D=IN_PROGRESS`、`SEARCH_METHOD=NOT_SELECTED`。正式矩阵总预算约 211 万次独特一步转移，本机探针推算数周串行 CPU；此推算仅为资源安排。未用 GPU、`locked_test`、训练或闭环。详见 STEP 6.3D 实施记录与源码；旧条目为此前 Step 的历史状态。

## 2026-09-29 STEP 6.3B/6.3C-PATCH 当前状态

研究者已明确批准 Comm 在当前已有无线 Flow 唯一可绑定 Task 中选择子集，并按 TRAIN 对应通信结构已见的选中任务数准入；原全部覆盖要求已取消。4416 TRAIN H1 投影把 728 条依赖同决策 offload Route 的历史 Comm 行逐条排除，任务数量拒绝为 0；结构准入 4045/4416，语义字段 3920 通过，125 个 Comp 数值差异和未见结构残余保留。TRAIN/Validation 静态空域为 14/4416、6/1104。验收、回归、索引和 Git 收口已完成，等待研究者审阅。没有运行 GPU、训练、baseline、闭环或 `locked_test`；不进入 6.3D。

## 2026-09-29 STEP 6.3C — 当前状态

`STEP_6_3C=PASS`。已实现唯一的 search-independent `CandidateDomain`、lazy/symbolic canonical candidate counting、共享 `SearchNode` 与 `B_WM` budget/cache contract，以及复用 STEP 6.1 frozen World Model 的 one-step primitive。Formal TRAIN 4416 个 anchor 的静态审计完成；1935 个空域按冻结 TRAIN joint-support/causal Comp 规则记录为无正式候选，不做自动 horizon backoff。Validation 1104 个 anchor 仅作 descriptive audit，未改变 TRAIN catalog 或 policy。固定非锁定 validation anchor 的 H1-H4 interleaved smoke 与既有 6.1 预绑定 rollout 指纹和数值完全一致，4 次 transition、无 cache hit、无 dead-end。该 PASS 只表示候选域/搜索协议/可行性证据闭合，不表示任何 optimizer、候选质量、H4 可行率或闭环性能；未执行 Random/CEM/Beam/Learned Proposal、ranking、baseline、GPU、training、locked_test。

## 2026-09-29 STEP 6.3B — 当前状态

`STEP_6_3B=PASS`（仅 CPU 候选语法/支持标签/准入合同）。研究者已冻结 Planner v1 的结构化候选语法与支持准入：Route 显式空；Comm 按当前无线 Flow/关系绑定、允许同 Task 多 row、每 row 为 TRAIN 观察到的循环连续 RB block；Comp 在当前 computing/Exec 关系上逐时隙重建 CPU base，并用全局 alpha `{0.5,0.75,1.0}`；Mob 正式池只允许所有在场 UAV 共享一个 profile。每一步的 Comm–Comp–Mob 粗粒度联合结构必须在 Formal TRAIN 251 种观察签名中；未见联合组合留作未来消融。时间序列观察性只作标签，不作硬门槛。`H_sup` 需实际 rollout 后由既有 scorer 判定。见 STEP 6.3B 合同、实施记录和机器收据。此处的语法闭合不等于已选优化器、候选质量或闭环性能；GPU、训练、`locked_test`、baseline 均未执行。

## 2026-09-28 STEP 6.3A — 当前状态

`STEP_6_3A=PASS`：CPU-only support audit measured Formal TRAIN Comm/Comp/Mob,
their joint/temporal structure and bounded search-space illustrations. Comp
base allocation is causally reproducible with the existing CPU inner rule and
observed global alpha `{0.5,0.75,1.0}`. Comm widths are 1/2/3 over RB IDs 0–49;
two-UAV mobility was observed as shared-profile joint actions. Independent
family factorization is `NOT_SUPPORTED`. Evidence-based recommendation is to
study support-constrained structured/conditional search; no method is frozen
or implemented. Validation was descriptive only. No candidate ranking,
optimizer, World Model rollout, GPU, Formal model training, baseline, closed loop or
`locked_test` was used. See the STEP 6.3A implementation record and receipts.

Verification note: the full repository unittest command was not clean (1993
tests: 34 errors and one historical receipt mismatch). It also executed
synthetic CPU trainer tests outside this Step's audit-only scope; no Formal
training, checkpoint/data write, GPU or `locked_test` occurred. The exact scope
deviation is recorded in the STEP 6.3A implementation record.

## 2026-09-28 STEP 6.2B-PATCH — 当前状态

研究者已冻结 `PLANNER_V1_ROUTE_POLICY=EXPLICIT_NOOP_ONLY`：Planner v1 每个 horizon 的 Route family 必须为空，实际优化动作族为 Comm/Comp/Mob。pending、已有 Flow 同路径、多跳及改目的地的非空 Route 都在准入层拒绝，4.4/learned Route/checkpoint 未改。冻结 best.pt 的非锁定 validation anchor 上两个合法 H1–H4 候选经 CPU rollout 与五项 scorer 均有限，Route 编译为缺席哨兵；选定 anchor 的 Comm 分母仍为 50 个全局 RB ID（不是 242 条关系行）。`STEP_6_2B=PASS` 只表示 Objective scorer 与严格字典序比较器通过 CPU 合同验收；`CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`，`CLOSED_LOOP_READINESS=NOT_READY`，无性能结论。GPU、`locked_test`、训练、baseline 未执行。证据：`docs/implementation_records/STEP_06_2B_PATCH_ROUTE_NOOP_CLOSURE.md` 与 `code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928/`。下一动作仅为研究者审阅本 Step；后续方法另行授权。

## 2026-09-28 STEP 6.2B — 历史阻塞关口

`STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`。五项 Objective scorer 和严格字典序比较器已实现，CPU 合同测试与一个非锁定 validation anchor 的冻结 checkpoint H1–H4 集成通过；但尚不能宣称 6.2B 验收 PASS。当前 single-hop gate 允许 pending/no-current-Flow Route，正式 adapter 却映射为 `flow_index=-1`，4.4 不创建 Flow；Objective 合同只定义未来 Return birth 的 `H_sup`，未定义该 pending Route 的评分支持范围。scorer 已拒绝静默评分此类 trace。同路径 existing-Flow Route 还会在 hop 完成前改 Task-Agent Host，需与 frozen holder 语义核对。研究者尚未决定处理方式；Route 域和 4.4 本 Step 未改。Comm effort 分母已从误数通信关系行修为当前有效的全局 RB ID 数；选定 anchor 从 242 修为 50，旧 receipt 保持历史原值。best.pt SHA 不变；未用 GPU、locked_test、训练、baseline 或闭环。下一动作仅为研究者裁决上述两个 Route 语义冲突，随后再重算 6.2B acceptance。证据见 `docs/implementation_records/STEP_06_2B_PLANNER_OBJECTIVE_SCORER_AND_COMPARATOR.md` 和 `code/artifacts/protocols/pi_jwm_step6_2b_objective_scorer_v1_20260928/`。

## 2026-09-28 STEP 6.2A-CLOSURE — 历史关口

研究者正式接受 no-retrain salvage：原 `best.pt` 继续作为 `FORMAL_BEST_CHECKPOINT`，SHA-256=`941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`；不重训。形式 train/validation multi-hop coverage 均为 0，54 个 overlap windows 的 H1–H4 state/graph/prior/Motion/CSI legacy 与 patched 完全相同；这不构成 multi-hop 性能证明。旧 `LVal=0.07431338784170399` 仍是原 accepted run 观测，patched full validation 未执行。

Planner v1 Route 保持启用，但冻结为 `FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1`：每个实际 Route path 长度为 1，节点必须是已有 Flow 的 frozen logical destination；multi-hop 规则代码保留，移出 Planner v1，留待扩展/消融。Objective 保持 `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)` 字典序，单跳 `B_Tx=R_hop`。当前 `STEP_6_2B_READINESS=READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`，仅允许后续 scorer/comparator 实现与 CPU 合同测试；`CLOSED_LOOP_READINESS=NOT_READY`、`CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`。未开始 scorer、ranking、baseline、GPU、locked_test 或闭环。任何未来正式训练前必须 `CROSS_LAYER_RULE_SEMANTICS_GATE=PASS`。本 Step 收据和记录见 `code/artifacts/protocols/pi_jwm_step6_2a_closure_single_hop_v1_20260928/` 与 `docs/implementation_records/STEP_06_2A_CLOSURE_NO_RETRAIN_SINGLE_HOP_PLANNER_V1.md`。

## 2026-09-28 STEP 6.2A-ROUTE-RECOVERY — 历史关口

4.2C-B/C destination-list route semantics are now implemented in 4.4 deterministic intermediate-hop advancement. Same-destination Route uses rule-side full-path metadata and does not alter the frozen 11 learned action tensors. Real two-hop Raw → Tensor → Graph/State → rule tests and a 113-test cross-layer semantics gate pass. The frozen best checkpoint strict-loads unchanged: SHA-256 `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`, parameter digest unchanged. Full Formal Dataset static audit: 4416 train/1104 validation, route width 1 throughout, zero multi-hop rows; paired legacy/patched CPU rollout on 38 train + 16 validation overlap windows is exactly invariant H1–H4. Full patched validation is `NOT_EXECUTED_REQUIRES_SEPARATE_RUNTIME_AUTHORIZATION`. `CHECKPOINT_NO_RETRAIN_SALVAGE=SUPPORTED_WITH_LIMITATIONS`; no retraining decision is made. GPU, optimizer, `locked_test`, 6.2B, scorer, ranking, baseline and closed loop remain closed. Receipts: `code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928/`. Next action: researcher review after Git push.

`STEP_6_2B_READINESS=BLOCKED`。补丁已证实 4.2C-B/C 当前 Flow Ledger 保存端到端剩余量，旧 4.2B“无法恢复”是历史事实。一个非锁定 Formal Validation anchor 的真实 deadline 已通过同种子/配置/动作前缀/当前状态的重放精确对齐，Task ID 与模型槽位一致；Formal Raw 本就含返回数据大小，但实际是否需要 Return 还取决于预测计算节点与返回目的地。研究者已删除 Route effort、Priority prerequisite，冻结业务主吞吐为 E2E useful，all-hop service 仅诊断。新的 Planner-only side-state 不进训练模型。

原 route mismatch blocker 已由 STEP 6.2A-ROUTE-RECOVERY 修复；旧段落仅保留为历史记录。当前边界是 Formal Dataset 没有多跳/完整 reroute 激活，因此正式性能影响尚未被数据覆盖。未开始 scorer、6.2B、候选排序、baseline、GPU、`locked_test` 或闭环。唯一下一动作：研究者审阅本 Step 的有限 salvage 证据。

## 2026-09-28 STEP 6.2A — Objective Source Audit

`PLANNER_OBJECTIVE_SOURCE_AUDIT=PASS_WITH_READINESS_BLOCKERS`，`STEP_6_2B_READINESS=BLOCKED`。6.2A 核对了 deadline/lifecycle、Task cohort、multi-hop Flow、compute、support、throughput、effort、energy/priority 等来源，并冻结了 objective 与 baseline metric 接口定义；没有实现 scorer、candidate ranking 或 winner selection。AirFogSim/Observer 有 deadline 等来源，但 Formal Raw 到 Planner 未暴露完整 causal side-state；跨 hop `B_Tx` 和 Route effort denominator 也未闭合。证据见 `docs/implementation_records/STEP_06_2A_PLANNER_OBJECTIVE_SOURCE_SEMANTICS_AUDIT.md` 和 `code/artifacts/protocols/pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928/`。GPU、`locked_test`、baseline、MPC、closed-loop 均未执行。下一步须研究者审阅 blocker 并单独授权；不自动开始 6.2B。

## 2026-09-28 STEP 6.1 正式训练模型候选推演预检

`TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT=PASS`（以最新机器 receipt 为准）：从同一个正式 validation 当前样本及同一 Encoder/当前 posterior latent 出发，四类合法当前因果动作分别完成 H1–H4 逐步编译和 prior-only 递归 World Model 推演；动作注入、对应潜变量、规则状态更新、串行/批量一致性与三组配对随机种子诊断通过。正式 `best.pt` SHA 精确匹配且参数未改变。证据见 `docs/implementation_records/STEP_06_1_TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT_PREFLIGHT.md` 和 `code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/`。这是单 validation anchor 的 CPU 机制验收；没有候选优劣、预测质量或闭环性能结论。`MPC_OBJECTIVE=NOT_STARTED`、`CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`、`CLOSED_LOOP=NOT_STARTED`；baseline、GPU、locked_test 均未执行。唯一下一动作：研究者审阅 Step 6.1 的机制与边界，决定是否单独授权下一研究 Step。

## 2026-09-28 STEP 5.6C 正式训练最终验收

`STEP 5.6B=COMPLETE`：正式 run `pi_jwm_formal_train_v1_seed5601_20260924T112424Z` 的本地日志严格有 5520 步和 1104/2208/3312/4416/5520 五次完整 1104-window prior-only validation，无 NaN/Inf 或失败凭证。第 5520 步是严格最低 `L_Val=0.07431338784170399`；`best.pt` 和 `latest.pt` 均记录该最终步，428 个模型张量完全相同。CPU 单 validation window H1–H4 checkpoint reload/推理通过。`FORMAL_BEST_CHECKPOINT=FROZEN`，best SHA-256=`941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`。训练源码 SHA=`6e15ec2da0e3a6e0561dc821d0aaef90696a2387`，当前 main 的后续提交不改变训练身份。机器回执、五次全部分 horizon 指标和本地文件 SHA 见 `code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928/`；checkpoint 字节在 local-only `code/artifacts/formal_training/`。这是 **Formal Validation Observation**，未执行 baseline、locked_test、模型候选 Planner rollout 或性能声明；已有 6.0A–C 静态 Planner 合同不等于模型 rollout。唯一下一动作：研究者审阅验收后的 checkpoint 与验证证据，不自动进入 Planner。

## 2026-09-26 STEP 6.0C Planner v1 动作域（历史时点）

研究者已冻结第一版 Planner **操作动作域**：Comp 对每个节点/时隙按当前 Raw 观测的静态 CPU 容量限制请求总和；缺容量则拒绝正分配。UAV 使用正式采集策略的六档边际控制，heading/elevation 从当前 Raw 只进入 Planner 控制侧状态；HOLD 是每架当前 UAV 的显式零速行。6.0B 的事实未改变：动态可用 CPU 仍无可靠来源，仿真器无 UAV 数值硬界。此动作域不是安全或最优声明。6.0C 仅 CPU 合同与测试，5.6B 远端未联系，训练是否完成仍未核实；无模型候选 rollout/目标函数/`locked_test`。见 6.0C 合同、实施记录和机器 receipt。唯一下一步：等待正式 best checkpoint 并由研究者另行授权模型依赖 Planner Step。

## 2026-09-26 STEP 6.0B 来源审计（历史时点）

CPU 只读源码审计得到 `STATIC_CAPACITY_ONLY`：决策时刻有带 mask 的静态 CPU 容量，未证实动态可用量；AirFogSim 原生计算回调不按节点容量裁剪，PI-JWM 正式采集器另行限制自身分配。UAV 直接执行接口没有数值硬边界；示例配置是生成/初始化设定，正式数据动作范围只是行为支持。因此 6.0A 的两项 `UNKNOWN` 均保留，Candidate 代码未改。AirFogSim 本地目录没有独立 `.git`，`git -C` 上溯到 PI-JWM；不能声称本地 AirFogSim Git SHA/clean 状态，机器凭证提供相关源码文件哈希。5.6B 远端未联系，训练结束与 best checkpoint 未核实。详见 STEP 6.0B 实施记录及机器凭证；下一步先补齐仿真器历史 Git 身份（若需要精确 commit），模型依赖 Planner 仍需另行授权。

## 2026-09-26 STEP 6.0A 状态（历史时点）

Line A：STEP 5.6B 正式训练是独立远端 run `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`，源码 SHA `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`；本 Step 未连接远端，仓库只保留既有 2026-09-25 过程快照，不能据此断言当前远端进度或最终 best checkpoint。Line B：STEP 6.0A 已完成 CPU 静态候选生成合同和合成 fixture 验收；Search/Learned/Hybrid 只是可插拔接口，最终方法待研究者决定。当前正式训练动作适配器仍是 `build_step5_1d_unified_model_chain_v1.py::build_action`，新编译器包装它。没有 World Model 候选 rollout、objective、proposal training、GPU、closed loop、baseline 或 `locked_test`。证据：`docs/implementation_records/STEP_06_0A_UNIFIED_CANDIDATE_GENERATION_CONTRACT_CPU.md` 和对应机器 receipt。唯一下一动作：等待正式 best checkpoint，再由研究者授权模型依赖的 Planner Rollout Step。

## STEP 5.6B 训练中快照（2026-09-25）

正式训练已按冻结配置在 RTX 4090 上运行。远端 run ID 为 `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`，训练源码 Git SHA 为 `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`。2026-09-25 13:34 UTC 的只读心跳为 `RUNNING`、2646/5520 completed steps；已有 step 1104、2208 两次完整 1104-window prior-only validation。中途曲线和复制日志哈希见 `docs/figures/step5_6b_live_progress_20260925/`。这是过程诊断，不是最终性能结论；`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。下一动作仅为继续监控此 run。

## STEP 5.6B 预启动状态（2026-09-24，历史）

研究者已明确授权使用冻结 Formal Training Config v1 在 RTX 4090 启动一次正式训练。正式 runner 与监控/恢复路径正在做启动前验收；截至此源码快照，`formal_training=false`。5.6A 的完整 CUDA smoke 和 prior-only validation 仍是运行证据，不是性能结论。启动后的真实状态只以远端 `run_manifest.json`、`progress.json` 和 `heartbeat.json` 为准。`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。唯一下一动作：完成 Go/No-Go、提交精确 source、远端 detached launch，并在 2–3 步确认后停止。

## STEP 5.6A 当前状态（2026-09-24，GPU 验收完成）

正式数据身份保持 H=2/L=4、60 条 trajectory、48/12 split、4416/1104 windows，Dataset package/hash 未改。RTX 4090 上 H=4 CUDA 前向、反向、参数更新、跨轨迹 batch、checkpoint/错误身份拒绝和完整 1104-window prior-only GPU validation 已有通过凭证；验证集 12 条轨迹均覆盖且没有重复/遗漏。未训练 smoke checkpoint 的 `L_Val=0.829751` 仅作运行诊断，不是性能结果。研究者已冻结 Formal Training Config v1，机器状态为 `FORMAL_TRAINING_CONFIG=FROZEN`、`FORMAL_TRAINING_READINESS=READY_TO_START`；`formal_training=false`、`gpu_training_verified=false`、`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。唯一下一动作是另行授权 STEP 5.6B，不得自动启动。

## STEP 5.5-PATCH 历史状态（2026-09-23）

STEP 5.5 的原 CPU H=4 smoke 只消费 `runtime/` 的 1 train + 1 validation，不能作为 5520-window Trainer 证据。PATCH 新增 `FullFormalShardDataset → FullFormalTrainer`：4416/1104 全量索引、按请求加载 trajectory shard、跨 shard batch、H=4 CPU 参数更新、prior-only validation batch、checkpoint reload/错误身份拒绝已有独立凭证。原 runtime mini 与 8/4 development 路径仍保留。

旧 Dataset receipt 的 `unsupported/unresolved/fixed_support_blocked=0/0/0` 是字段计数，未检测真实 future Return birth；PATCH 对全部 5520 windows 的独立结构审计发现 8828 次按窗口与未来步计数的 Return-birth unsupported/fixed-support 事件，涉及 2901 个窗口和 60 条轨迹；143320 次已有支持的 Return continuation 未误判。新 detector 只写 target-side component 记录，不创建 current Return slot，也不删 Motion/CSI 监督。旧零值已被新审计替代。60 条 Raw 中有 213 次同一 Task 对象的 lifecycle collection 修复，涉及 50 条轨迹、124 个 trajectory-task；保留最远 lifecycle，直接 Task 状态字段不改。

机器凭证：`code/artifacts/audit/pi_jwm_step5_5_patch_20260923/`。`FULL_FORMAL_DATASET_LOADER=VERIFIED`、`H4_FULL_DATA_CONSUMPTION_PATH=VERIFIED` 指全量可索引的 CPU 数据路径和抽样执行，不表示 5520 窗口已完整正式训练。当时 `gpu=false`、`formal_training=false`、`locked_test_accessed=false`。

Formal Dataset v1 已由 60 条真实 AirFogSim trajectory 构建并机器验收：H=2/L=4、每条 96 transitions、48/12 trajectory split、4416/1104/5520 windows，五类 package、四动作 coverage、train-only normalization 与 deterministic rebuild 均通过。原 CPU H=4 trainer smoke 是 runtime 1+1 mini；PATCH 另行验证 full-shard CPU batch 路径。`formal_dataset=true` 只表示数据包 READY；在该 PATCH 时点 GPU execution 尚未开始。

以下为历史补充；该段原更新时间为 2026-09-22，当前状态以上方最新快照为准。

这是 ChatGPT 网页端进入仓库后的第一读取入口。它只提供当前快照和继续查证的路径，不替代源码、配置、checkpoint、metrics、manifest 或 audit。

## Source of truth

事实优先级固定为：当前源码/config/experiment → `AI_CONTEXT/` → 普通项目文档 → 历史聊天或推断。任何正式科研结论都必须回到第一层验证。

## 2026-09-22 历史状态

- 项目：PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）。AirFogSim 只是参考仿真器和数据生成工具。
- 当前 active workflow：研究者最新只读 `00–06` 定义链；工程执行入口为 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`。
- 当前状态：`STEP 5.5 = COMPLETE / FORMAL DATASET V1 ACCEPTED`，PATCH 的 full-shard CPU 数据路径另行验收。60/60 trajectory、五类 package、四动作真实 coverage、H1-H4 Motion/CSI、hash/identity、确定性重建和 runtime mini CPU H=4 smoke 均有机器证据。正式训练、GPU execution、planner、baseline、locked-test 仍未开始。
- `STEP 3.2-PATCH` 已补齐 Dataset isolation provenance、time-grid、development future-reference audit 与 normalization units；仍是 observation-only/non-locked evidence，不是正式 Dataset。
- `STEP 3.2-PATCH-RECEIPT` 已修正顶层 acceptance AND、显式 scope checks 和 frozen sample schema reuse；STEP 3.2 现正式 COMPLETE / FROZEN。
- STEP 4.4-PATCH3 已在源码中显式区分 Return requirement 的 known/unknown：冻结 current-side 不含 `Task.return_size`，所以 no slot 是 unknown，不是 no-return；unknown 或 known-required/no-slot 都不能错误 final-complete，但 side-state 可区分两者。DAG 只按有效前驱动态释放，terminal Flow completion 同步 remaining/presence/carrying/status。仍是 untrained CPU development evidence，不代表预测精度或训练结果。
- Definition 05 v1 decisions 已冻结：deterministic mean decoder；Motion/CSI family-wise mask-MSE；Phy/Comm analytic KL + warm-up/free-bits；overshooting OFF；Future Target 仅进入 family-specific training posterior；joint training；prior-only validation；checkpoint=`argmin L_Val`；逐 horizon raw-unit Motion/CSI MAE/RMSE。STEP 5.1A target、5.1B primitives/paired integration、5.1C development bundle、5.1D CPU chain 和 5.2 CPU training loop 均已有对应证据；full training 尚未开始。
- 新定义实现状态：Raw、最小 Dataset/Tensor、Typed Graph Builder、Dual-Graph Encoder、Structured RSSM World Model、5.1B loss/KL/metric primitives 已验收；5.1C additive bundle 将 12 个 paired window 的 support 对齐为 observed `10/74`，5.1D 从同一 bundle 完成 graph/encoder/world-model paired CPU integration。Physical topology、Encoder/World Model 参数仍是 development-only，模型权重未训练。
- 审计结论：时间因果、稳定 ID/index、mask/split、typed graph、`Z_t^{PI,L_g}→xi_t^Lat`、current-observation posterior、目标 RSSM 边界和逐步规则反馈已落地；完整 planner 闭环仍需后续授权与实现。
- 当前运行：没有正式 GPU 训练或远端同步任务；旧 `seed=20260832` 仍不得自动启动。
- `locked_test_accessed=false`；`formal_performance_claim_ready=false`；`formal_dataset=true`、`training_loop_implemented=true`、`cpu_optimizer_smoke=true`、`full_training=false`、`gpu=false`。

## 2026-09-22 当时最重要问题

当前实现不能按 Dataset READY、runtime mini smoke 或 PATCH full-shard CPU batch 外推性能。没有 formal training、GPU runtime、收敛泛化、校准或预测精度证据；Planner 真实反馈也未实现。

## 2026-09-22 当时建议的下一步

唯一建议是研究者另行授权 **STEP 5.6A — GPU Smoke + Formal Training Config Freeze**；在此之前不启动正式训练，也不访问 `locked_test`。

## 当前 Git

- Branch：`main`。
- Step 1 base commit：`829276241a0da72d3a5393946086daba40b0a0fe`。
- Latest commit：以 GitHub `main` 的 `HEAD` 为准；本文件不能稳定硬编码包含自身的 commit hash。读取时运行 `git rev-parse HEAD` 或查看 GitHub 分支头。

## 关键入口

- 当前方法与边界：`AI_CONTEXT/02_ARCHITECTURE.md`
- 新定义实现总表：`docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- Step 1 详细记录：`docs/implementation_records/STEP_01_AUDIT.md`
- 数据和张量：`AI_CONTEXT/03_DATA_FLOW.md`
- 问题到源码：`AI_CONTEXT/04_MODULE_MAP.md`
- 当前/历史实验：`AI_CONTEXT/05_EXPERIMENTS.md`
- 已确认决策：`AI_CONTEXT/06_DECISIONS.md`
- 冲突和阻塞：`AI_CONTEXT/07_KNOWN_ISSUES.md`
- 机器注册表：`docs/registries/`
- 原始状态权威：`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`

## 最近重要变化

- 2026-09-20：STEP 4.2A-PATCH 解耦 wireless structural relation validity 与 CSI observability；missing CSI 保留 relation 并使用 mask=false/zero placeholder。return size/priority/deadline 改为 observer available but frozen Raw not exposed；stateful Flow 仍未解决。
- 2026-09-20：STEP 4.2B source audit 证明 `transmitted_size` 为 hop-local stage progress，不能推出 end-to-end Flow remaining；DAG 只提供 gating，DepData transfer 未找到。综合 verdict=`FLOW_CONTRACT_NOT_YET_SUPPORTED`。
- 2026-09-20：STEP 4.2B-PATCH 将 verdict 改为 Flow-specific evidence 的实际计算；resource gaps 与 Flow readiness 解耦，stable ID/DepData 改为 implementation fact + researcher decision boundary；provenance 增加 symbol anchors。
- 2026-09-20：STEP 4.2C-A-PATCH 通过 audit-only replay 修正 verdict：logical destination 过滤 final delivery，E2E remaining 与 holder 可因果派生，same-destination reroute 可保持 Flow epoch；`flow_completed` 禁止作为 logical completion；机器 verdict=`CAUSAL_FLOW_LEDGER_FEASIBLE`。destination change/DepData 仍需研究者决定。
- 2026-09-20：STEP 4.2C-B 实现 FlowID/Epoch/RouteRevision、Flow/Carrying 分离、Input/Return lifecycle、clean-boundary destination change、lineage 与 Raw additive state；真实 non-locked trace覆盖 Input/Return，DepData runtime=0。随后 STEP 4.2C-C 完成 Sample/Tensor additive extension；在该 Step 当时 Graph Builder 尚未开始，之后已由 STEP 4.3A 完成。
- 2026-09-20：STEP 4.2C-B-PATCH 修正 logical destination provenance：Input 使用已成立 route terminal，Return 使用 `return_destination_id`；真实 `UAV_0→RSU_0→cloudServer_4` 两跳通过单 FlowID/Epoch、固定 destination、无重复 E2E 计数验收。
- 2026-09-20：STEP 4.2C-C-PATCH 将 normalization stats 收紧为 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`，并补齐 History/target Logical/Carrying 四组全字段 semantic equality、ID/provenance tamper 检查与 target carrying future-ground-truth namespace；23/23 focused、跨时隙真实 trace、deterministic/round-trip/receipt negative checks 通过。在该 Step 当时 Graph Builder 尚未开始；之后 STEP 4.3A/4.3B/4.4 已依次完成 Builder/Encoder/World Model Contract，训练仍未开始。

- 2026-09-19：STEP 4.1-PATCH 修正最小 gap 语义：wired relation 是 Raw/simulator 有来源但未暴露，无 CSI 时用 type + mask；wired 可选 numeric state 不阻塞 03 minimum；CPU capacity 是静态 capability，并与 allocation/service/available CPU 分离。在该 Step 当时 graph builder 保持关闭，之后已由 STEP 4.3A 完成。

- 2026-09-19：完成 Step 2.4 wireless/wired/total communication Outcome 语义最终验收；wired 服务来自真实 `WiredNetworkManager.step`，空 map 与 missing 分开，冻结 Raw Trajectory Layer / 01。
- 2026-09-19：完成 Step 2.2 真实 AirFogSim 6 步轨迹与独立下一 Decision 验收；补齐 Step 2.1/2.2 机器证据 Git 追溯。
- 2026-09-09：第二个正式 seed `20260830` 完成并通过单 seed 验收；第三 seed 暂停。
- 2026-09-09：建立项目文件、依赖、artifact、实验、结果、历史方法和问答路由索引。
- 2026-09-10：新增面向 ChatGPT 网页端的 `AI_CONTEXT/`，并将三方协作和同步规则写入 `AGENTS.md`。
- 2026-09-19：STEP 3.1 冻结最小 Model-ready Sample & Tensor Contract；`H=2/L=2` 真实样本、四类 action、input/target index 隔离和 mask 语义通过机器检查。正式 batch/split builder 未开始。
- 2026-09-19：STEP 3.1R 修正 History `[t-H+1,t]`、固定 index/presence、真实 DAG 接线、typed target namespaces 和 relation endpoints；v2 Raw 与最小样本证据已重建。
- 2026-09-19：STEP 3.1F 及其最小 PATCH 已冻结；History 保留 past Action/Outcome，input index 使用 History causal union，Future Action 先做 anchor visibility 检查再引用同一 static index，future-reference 观察审计 JSON 已纳入 provenance/Git。
- 2026-09-19：STEP 3.2 完成最小 non-locked batch/split/preprocessing validation；3 trajectories、12 windows、train-only mask-aware stats、deterministic rebuild 和 round-trip 已有 artifact/test 证据，不代表正式 Dataset。

Unverified：当前没有“最终 PI-JWM 方法已冻结”或“正式性能声明已开放”的证据。

## 2026-09-22 STEP 5.1C-PATCH

- Unified bundle 对 12 个 paired windows 完成 slot-wise Physical/Communication identity proof；统一容量为 `max_entity=10`、`max_comm_relation=74`，no-prefix/pairing crop 机器检查通过。
- Flow normalization 新 stats 只从 8 个 unified `dev_train` samples 的 History 拟合；4 个 `dev_validation` samples 排除，Future Target 未参与 fit；旧 5-sample stats 仅保留为历史 provenance。
- receipt、exact upstream train lineage、runtime prior-target isolation、tensor contract、deterministic rebuild 和 serialize/reload 通过；仍为 CPU/non-locked development evidence。Route/Comp non-empty coverage 在真实 12-sample bundle 中均为 0，只验证 explicit no-op；这是未来 formal training/data coverage gate，不是完整四动作族训练证据。

## 2026-09-22 STEP 5.2-PATCH

- Stage 2/Validation 初始 latent 改为 current-observation posterior；未来递推继续 prior-only。current posterior 纳入 joint optimizer，Future Target teacher 独立且 validation 调用为 0。
- Validation 改为每个 horizon 跨完整 validation set 的 Motion/CSI numerator/count 独立归一化，再计算 `L_Val`；补 unequal-mask fixture。
- Checkpoint load 增加 schema、data identity、normalization provenance、architecture-critical config 拒绝检查；compatible reload、wrong-data/normalization rejection 均通过。receipt 为 26/26。

## 2026-09-22 STEP 5.4

- 新增 manifest-driven `FormalTrainingInterface` 与 CPU-only readiness audit；5.2 development adapter 保持回归兼容。
- readiness：`TRAINING_STACK_READINESS=PASS`、`FORMAL_DATASET_READINESS=NOT_READY`、`GPU_CODEPATH_READINESS=PREPARED`、`FORMAL_TRAINING_READINESS=BLOCKED`。
- 12 个 development samples 实际 coverage：Mobility=24、Comm=1、Route=0、Comp=0；正式 `L`、topology、Dataset/seed/训练预算仍需 researcher decision。

## 2026-09-23 STEP 5.4-PATCH2

- 四动作 adapter 与 CPU device portability 已闭合；optimizer 在最终 device migration 后创建，Route/Comp adapter support 通过 synthetic fixture。
- 当时 readiness：`TRAINING_STACK_READINESS=PASS`、`FORMAL_DATASET_READINESS=NOT_READY`、`GPU_CODEPATH_READINESS=PREPARED`、`FORMAL_TRAINING_READINESS=BLOCKED`。
- Formal Dataset 不存在，真实 development Route/Comp coverage 仍为 `0/0`；formal L/topology/budget 未决，GPU/locked_test 未执行。

## 2026-09-22 STEP 5.2

- `code/src/pi_jwm/step5_2_training_loop_v1.py` 连接当前 Encoder、Structured RSSM、5.1B target/posterior/loss/KL 原语；Stage 1 使用 family-specific posterior teacher，Stage 2/Validation 从 current-observation posterior 初始化，之后 prior-only recursive rollout。
- 配置化 curriculum 为 `1→2→4`，当前 development `L=2` 自然为 `1→2`；`beta_KL`、warm-up、free bits、optimizer、clip、seed、batch size 和 epoch/step 均进入 config。
- 真实 8/4 unified non-locked bundle CPU smoke：两步 optimizer update、4 validation samples prior-only、checkpoint save/load/resume；5.2-PATCH receipt `26/26`，`passed=true`。随后 5.3 固定 dev_train 1/2-sample preflight receipt=`GO`；这两者都不是正式性能证据。
- optimizer audit 覆盖 Encoder、RSSM dynamics、两类 prior、两类 future posterior、两类 target encoder、Motion/CSI decoder；known rule parameter count=0。Route non-empty=0、Comp non-empty=0、Comm=1、Mobility=48，Route/Comp 仍是 future formal training/data coverage gate。

## Freeze chain（2026-09-20 历史快照）

- Raw Trajectory / 01、当前最小 Dataset/Tensor / 02、STEP 4.1 mapping、STEP 4.2A existing-source input extension、STEP 4.2C-B Raw Flow、STEP 4.2C-C Flow Sample/Tensor、STEP 4.3A Typed Dual-Graph Builder 与 STEP 4.3B Dual-Graph Encoder 均已冻结。
- Causal boundary: future task schedule is internal metadata only; canonical acceleration is backward speed difference with an explicit missing-history mask.
- 当时边界：Physical topology 的 `radius_knn/radius=1000m/k=2` 仅是 development config，`research_frozen=false`；此项后来已由 STEP 5.5 正式数据协议冻结。Return multi-hop 与 same-destination reroute runtime 的历史边界保留。
- 当时状态：Definition 05 decisions FROZEN；5.1A COMPLETE/FROZEN，5.1B COMPLETE/FROZEN FOR CPU DEVELOPMENT PRIMITIVES + PAIRED INTEGRATION，5.1C COMPLETE/FROZEN FOR DEVELOPMENT，5.1D COMPLETE/FROZEN FOR CPU DEVELOPMENT INTEGRATION，5.2 COMPLETE/FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION。该时点的 `formal_dataset=false`、`gpu=false` 不描述当前状态。
## 2026-09-30 STEP 6.3D-3080TI-MIGRATION-QUALIFICATION 当前状态

`STEP_6_3D_3080TI_MIGRATION_QUALIFICATION=PASS`：正式 Dataset、60 条 Raw 和冻结 checkpoint 在 RTX 3080 Ti 上完成逐项身份验收；32 TRAIN / 64 Validation 锚点可加载。CPU/GPU 离散等价、CUDA formal runner 与 resume/config 身份、小规模 TRAIN HRS/S-CEM smoke 均通过。正式 FP32 batch16，短稳态中位 10.9154 unique transitions/s。正式 TRAIN tuning、Validation 比较、`locked_test` 未运行；4090 源未停机。future Return birth 固定支持限制仍在，`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`。证据见 `docs/implementation_records/STEP_06_3D_3080TI_MIGRATION_QUALIFICATION.md` 和对应 `code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/`。下一步需研究者审阅。
