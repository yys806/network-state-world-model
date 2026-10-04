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

# PI-JWM Implementation Tracker

> 2026-10-01 当前：PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。

> 2026-10-01 当前：STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。

**2026-10-01 STEP 6.3D FORMAL TRAIN TUNING CLOSURE（PASS）：** 3080 Ti 上 768/768 TRAIN 调参结果已逐份身份/SHA 验收、本机归档并独立重算；S-CEM/MH-CEM 均冻结 `(K=4,rho=0.1)`。八组配置各 48/96 cases 找到可评分 H4，16/32 锚点始终无可评分路径，future Return birth 限制不变。名义/实际 unique transitions 均 393,216，正式耗时 21.38 小时。仅 TRAIN 选参完成；`VALIDATION_COMPARISON=NOT_STARTED`、`SEARCH_METHOD=NOT_SELECTED`、`locked_test=false`。见独立 Closure 实施记录和本机/机器 SHA 收据。

**2026-09-30 STEP 6.3D-PREFLIGHT-PATCH（输入/执行路径 PASS，正式比较暂停）：** 原选中 TRAIN 3、Validation 5 个零 Objective cohort 已按原同层哈希规则确定性替换；修正后 32/64 全部 CandidateDomain 非空、cohort>0，8 个 replacement deadline sidecar 精确对齐。TRAIN-only HRS seed6391/B64：16/32 锚点有 H4 可评分候选，16/32 无；275 条不可评分完整 H4 都有 future Return birth 支持边界。batch 1/4/8/16 一步与 H4 串行等价，batch 8 的本次 CPU 吞吐最高。Patch 验收不等于 H4 readiness；因 16/16 分布且无“大量”阈值，研究者需判断支持 blocker。正式 TRAIN tuning/Validation 比较未运行、方法未选；无 GPU、`locked_test`、训练、闭环。详见独立 Patch 实施记录和 25 接受收据。

**2026-09-29 STEP 6.3D（进行中，方法未选）：** 研究者冻结 HRS/S-CEM/MH-CEM 同一 CandidateDomain、H4-only 五项严格字典序 Objective 和 `B_WM={256,512,1024}` 的 TRAIN 调参与 Validation 成对比较。32/64 个非空锚点及 96/96 对齐因果侧状态已就绪；共享五层 proposal、固定预算搜索与 bootstrap 代码通过 9/9 focused tests 和 synthetic exact oracle。真实冻结模型首个 TRAIN anchor 的 B_WM=512 探针为 512 次独特一步转移、130 条完整 H4、0 条可评分，仅作诊断。完整矩阵尚未完成，`STEP_6_3D=IN_PROGRESS`、`SEARCH_METHOD=NOT_SELECTED`；本机仅 CPU，未用 GPU、`locked_test`、训练或闭环。见 STEP 6.3D 实施记录及机器探针。

**2026-09-29 STEP 6.3B/6.3C-PATCH（当前）：** 研究者正式将 Comm 当前任务域改为已有无线 Flow 唯一可绑定的 `E_comm(t)`，允许按 TRAIN 对应 Comm 结构观察到的 selected-task-count 选择子集；未选任务只是当前时隙未调度。原始 TRAIN 不改，依赖同决策 offload Route 的 Comm 历史行从 Planner-v1 投影目标排除并逐条说明。4416 TRAIN H1 投影任务数量拒绝为 0，728 条 Route 依赖行排除；Comp/未见结构残余照录。TRAIN/Validation 静态空域为 14/4416、6/1104；H1–H4 非空 Comm 机制路径与顺序 rollout 等价。`STEP_6_3BC_PATCH` 仅为 CPU 候选准入/投影审计，不是优化器、排名、闭环或性能验收；GPU、训练、`locked_test` 关闭。见补丁实施记录与 07 机器收据。

**2026-09-29 STEP 6.3B（当前）：** 研究者冻结 Planner v1 的 state-conditioned Comm/Comp/shared-Mob Candidate Grammar 与 Formal TRAIN 已见联合结构准入；Route 继续显式空，时间支持仅作标签。实现 TRAIN-only 目录、跨后端准入器和定向 CPU 测试，修正 6.0A 同 Task 多 Comm row 拒绝。`STEP_6_3B=PASS` 仅表示语法/支持标签/准入合同，不表示选择了 optimizer 或证明候选质量、闭环性能。见 `contracts/PIJWM_STEP_06_3B_STRUCTURED_CANDIDATE_GRAMMAR_V1.md` 与 STEP 6.3B 实施记录/收据。

**2026-09-28 STEP 6.2B-PATCH（当前）：** 研究者冻结 `PLANNER_V1_ROUTE_POLICY=EXPLICIT_NOOP_ONLY`，6.0C gate 在每 horizon 拒绝所有非空 Route。4.4/learned Route/checkpoint 保留。新 CPU 机器证据核对 pending 与 existing 同路径拒绝、Route 缺席编译、有效 Route 自由度 NONE、Comm 分母 50 个全局 RB ID、冻结 checkpoint 两个合法 no-Route H4 候选可评分。`STEP_6_2B=PASS` 仅为 Objective scorer 和字典序比较器合同验收；候选方法、闭环、baseline、GPU、`locked_test` 与性能声明未开始。证据见 `implementation_records/STEP_06_2B_PATCH_ROUTE_NOOP_CLOSURE.md` 和 `code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928/`。下方 6.2B BLOCKED 是 Patch 前历史状态。

**2026-09-28 STEP 6.2B（当前）：** 五项 Objective scorer 与严格字典序比较器已在 CPU 上实现，单 non-locked Formal Validation anchor 的冻结 checkpoint H4 机制集成通过；Comm RB 分母从关系行误计修为全局 RB ID。有效动作审计发现 pending Route 经 gate 通过却映射 `flow_index=-1`、规则不创建 Flow；同路径 Route 还提前改 Host。因此 `STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`，不宣称 scorer 完整验收；待研究者裁决这两个 Route 语义冲突。机器记录在 `code/artifacts/protocols/pi_jwm_step6_2b_objective_scorer_v1_20260928/`，实施记录见 `implementation_records/STEP_06_2B_PLANNER_OBJECTIVE_SCORER_AND_COMPARATOR.md`。无候选方法、闭环、baseline、GPU、locked_test 或性能结论。

**2026-09-28 STEP 6.1（冻结训练模型候选推演预检）：** 正式 `best.pt` SHA 与 CPU 加载核对后，在同一个 Formal Validation anchor 上用四类当前因果合法动作，完成 H1–H4 逐步编译、prior-only recursive rollout、规则/潜变量响应、配对随机种子与串行/批量 CPU 诊断。机器入口 `code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/`，实施记录见 `implementation_records/STEP_06_1_TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT_PREFLIGHT.md`。只验收模型候选推演机制，不含 objective/winner、closed loop、baseline、locked_test 或性能声明。

**2026-09-28 STEP 5.6C（正式训练最终验收）：** 本地 CPU 验收 run `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`：5520/5520 步、五次完整 prior-only 1104-window validation、最终 step 5520 严格最低 `L_Val=0.07431338784170399`。best/latest 都记录最终步且 428 个模型张量逐项相同；单 validation window H1–H4 CPU replay 通过。`STEP 5.6B=COMPLETE`、`FORMAL_BEST_CHECKPOINT=FROZEN`。机器回执和完整分 horizon 指标见 `code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928/`，实施记录见 `docs/implementation_records/STEP_05_6C_FORMAL_TRAINING_FINAL_ACCEPTANCE.md`。这是 Formal Validation Observation，未进入 baseline、locked_test、模型候选 Planner rollout 或性能声明。

**2026-09-26 STEP 6.0C（Planner v1 操作域冻结，CPU 合同）：** 研究者决定使用当前 Raw 静态 CPU 容量的每节点每时隙预算，并使用正式行为支持的 UAV 六档核心动作域。HOLD 为每架当前 UAV 的显式零速命令；H4 只更新 Planner 控制侧 heading/elevation，不预测物理状态。与正式 `build_action` 的 11 tensor 逐值等价已有合成合同测试。动态可用 CPU 仍无来源，UAV 无仿真器硬数值界；无安全/性能声明。见 `docs/contracts/PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md` 和实施记录。5.6B 未接触。

**2026-09-26 STEP 6.0B（CPU 只读来源审计）：** AirFogSim 静态容量可见，动态可用 CPU 未建立，verdict=`STATIC_CAPACITY_ONLY`；UAV 直接执行无硬数值边界，配置/正式数据仅是生成、初始化及行为支持。6.0A 两项 UNKNOWN 和 Candidate 代码保持不变。AirFogSim 本地目录无独立 Git 元数据，相关文件哈希可核验但 Git SHA 不可声称。记录：`docs/implementation_records/STEP_06_0B_PLANNER_ACTION_FEASIBILITY_SOURCE_AUDIT.md`；5.6B 未接触。

**2026-09-26 STEP 6.0A（CPU 静态合同验收）：** 新统一候选动作合同、四族语义、三态约束、固定支持、正式 `build_action` wrapper、Search/Learned/Hybrid 接口、暖启动、规则空动作和去重池完成。合成合同 fixture 的 11 个正式张量字段逐值等价；无模型 rollout/目标函数/GPU/`locked_test`。5.6B 独立远端训练未接触；最终候选方法待研究者决定。记录：`docs/implementation_records/STEP_06_0A_UNIFIED_CANDIDATE_GENERATION_CONTRACT_CPU.md`。

**2026-09-24 STEP 5.6B（预启动）：** 研究者已授权 RTX 4090 正式训练启动，冻结配置不变。runner、atomic heartbeat、完整 prior-only validation 和 identity-safe checkpoint 已增加，须经独立 Go/No-Go、source commit/push/remote SHA 核对后才启动。此源码快照尚无训练结果；真实运行状态以远端 `run_manifest.json`/`heartbeat.json` 为准。`locked_test`、baseline、Planner、性能声明均保持关闭。记录入口：`docs/implementation_records/STEP_05_6B_FORMAL_GPU_TRAINING_LAUNCH.md`。

**2026-09-24 STEP 5.6A（GPU 验收完成，配置待决）：** RTX 4090 上已完成正式数据 H=4 batch 1/2/4/8 的 CUDA 前向、反向、参数更新、跨轨迹 batch 与 checkpoint/错误身份拒绝。完整 1104-window prior-only validation 已通过四组互斥轨迹的唯一性合并；GPU readiness receipt 为 PASS。正式训练数值配置仍待研究者决定，formal training、baseline、Planner、locked_test 和性能声明均未执行。证据入口为 `docs/implementation_records/STEP_05_6A_GPU_SMOKE_FORMAL_CONFIG_EVIDENCE.md`。

**2026-09-24 STEP 5.6A-CONFIG-FREEZE：** 研究者批准的 Formal Training Config v1 已落入 `code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/`，状态为 `FORMAL_TRAINING_CONFIG=FROZEN`、`FORMAL_TRAINING_READINESS=READY_TO_START`。同时修正 validation availability bookkeeping：真实 validation target-mask 计算为 H1–H4 各 1104 个窗口，旧 GPU receipt、official numerator/count 与 `L_Val` 未改；无 GPU rerun、无 formal training。下一步需另行授权 STEP 5.6B。

**2026-09-23 STEP 5.5-PATCH：** 纠正原 H=4 `runtime/` 1+1 mini smoke 的边界。新增 `FullFormalShardDataset → FullFormalTrainer` 全量 4416/1104 索引与按 batch/shard 加载；真实 CPU H=4 参数更新、跨轨迹验证与 checkpoint 通过。全量 fixed-support audit 发现 8828 次 Return-birth unsupported/fixed-support；旧 receipt 的 0/0/0 为未检测字段读数，已被新凭证取代。AirFogSim lifecycle repair 为 213 次 collection-side 同对象去重。证据见 `docs/implementation_records/STEP_05_5_PATCH_FULL_CONSUMPTION_FIXED_SUPPORT_CLOSURE.md`。GPU/formal training/locked_test 均未执行。

STEP 5.5 当时状态：**Formal Dataset v1 的 60 条真实 trajectory、H=2/L=4 五类 package、48/12 split、四动作 coverage、deterministic rebuild 与 CPU H=4 trainer smoke 已完成**。`FORMAL_DATASET_READINESS=READY`、`TRAINING_STACK_READINESS=PASS`；当时尚未执行 GPU。当前 GPU smoke 状态见上方 STEP 5.6A。

## 当前依据与执行边界

- 目标研究定义：`D:\shen\OB\科研\PIJWM` 中七个 `00–06` Markdown 文件，只读。文件名、大小、行数和 SHA-256 见 `code/artifacts/audit/pi_jwm_new_definition_step01_20260918/initial_snapshot.json`。
- 实现事实：本仓库源码、配置、测试和原始 artifact。笔记中“当前代码已经……”的描述也必须核对。
- 工程工作区：`D:\shen\PKU\PIJWM`；旧 P4/P6/P0–P10 工作流为 **Historical / Archived**，不再是 active workflow。旧结果保留原验收含义，不变成新定义结果。
- 当前正式数据集协议与 artifact identity 已冻结；STEP 5.6A 只进行授权的 GPU smoke/validation，不进行正式训练，不改变模型科学结构/loss/planner，不访问 `locked_test`。
- 主报告：[STEP_01_AUDIT.md](implementation_records/STEP_01_AUDIT.md)；数据附件：[STEP_01_DATA_GRAPH_AUDIT.md](implementation_records/STEP_01_DATA_GRAPH_AUDIT.md)。下表中的 00–06 对应上述源文件章节；详细定位在报告中。

## 总体实施状态

Status 表示工程进度；Reuse 表示与目标定义的匹配类别。`DIRECT_REUSE` 只对该行明确划定的局部机制成立，不等于整个模块符合新定义。

| 模块 | Definition Source | Current Code | Status | Reuse | Main Gap | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 总体研究链路 | 00 四–六 | formal model / planner / r6 历史链 | AUDITED / NOT_IMPLEMENTED | STRUCTURAL_CHANGE | 各接口存在不等于新定义端到端闭环 | 先冻结一步数据合同 |
| AirFogSim trajectory | 01 原始轨迹与时间语义 | Raw contract、Step 2.1–2.4 runner/artifact | COMPLETE / FROZEN | MINOR_MODIFICATION | Raw 层已闭合；尚未映射到新 Dataset/Tensor | Step 3 冻结 Dataset/Tensor Contract |
| Decision / Execution / Outcome | 01；02 | Raw contract、causal helper、Step 2.3/2.4 真实 artifact | COMPLETE / FROZEN | MINOR_MODIFICATION | future schedule 已与 `O_t`/History/input index 隔离；wireless/wired slot outcome 已拆分并实测 | Step 3 保持同一因果边界 |
| Dataset / split / masks | 02 | `step3_2_batch_preprocessing_v1.py`、Step 3.2 bundle | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | trajectory split、lineage、time-grid、train-only normalization、mask/presence counterfactual 已验收；不是正式大规模 Dataset | 03 所需新增字段只能走 additive extension |
| Model-ready sample / tensor contract | 02 | `model_ready_sample_contract_v1.py`、Step 3.1F artifact、Step 3.2 bundle | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | History past A/Y、entity type、union input index、typed target、batch/split/preprocessing 已验收；不是正式大规模 Dataset | 进入 03 前先冻结对象-字段-关系映射 |
| tensor contract | 02；03 | `step3_3_model_input_tensor_v1.py`、`build_step3_3_model_input_tensor_v1.py` | COMPLETE / FROZEN FOR CURRENT MINIMAL DATA CONTRACT | MINOR_MODIFICATION | Past Outcome、完整 Target facts、固定 vocab、Comp 正式字段与 semantic receipt 已验收；03/04 feature selection 和正式容量未决定 | 单独授权双图字段映射 |
| Physical / Information 双图 | 03 | Step 4.1 mapping；Step 4.2A additive Sample/Tensor；Step 4.2C-B Ledger/Raw；Step 4.2C-C Flow Sample/Tensor；Step 4.3A typed builder；Step 4.3B encoder | BUILDER + ENCODER COMPLETE / FROZEN | STRUCTURAL_CHANGE | History temporal encoding、typed directed propagation、P2A/P2C 与 aligned `Z_t^{PI,L_g}` 已实现；topology/encoder config 仅为 development、未研究冻结；这不是 World Model latent | 仅建议 Definition 04 World Model Representation / Dynamics Contract；不自动执行 |
| entity alignment 局部工具 | 02；03 | `airfogsim_tensor_v2.py`、`formal_graph_ops_v1.py` | AUDITED | DIRECT_REUSE | ID/index/mask原则可复用；新增对象映射需扩展 | 保留身份稳定性检查 |
| Route action | 06 §2.1；04 §3.3 | Step 2 Raw + Step 3.3 past/future route tensors | INPUT TENSOR COMPLETE / FROZEN | MINOR_MODIFICATION | route kind/target/task node/hops 与 mask 已映射；尚未接新 graph/model | 后续按新对象路由，未授权 |
| Comm action | 06 §2.1；04 §2.3 | Step 2 Raw + Step 3.3 per-RB action tensors；Step 4.2A per-RB CSI additive tensor；Step 4.3A Comm relation | INPUT TENSOR + CURRENT COMM GRAPH COMPLETE / FROZEN | MINOR_MODIFICATION | CSI/typed relation 已进入 current Graph Builder；动作尚未接 Graph Encoder/model | 后续按独立授权接 Encoder/model，当前不执行 |
| Comp action | 06 §2.1；04 §2.3 | Step 2 Raw + Step 3.3 node/allocated CPU tensors | INPUT TENSOR COMPLETE / FROZEN | STRUCTURAL_CHANGE | `allocated_cpu_per_s` 已张量化；新 graph/model 执行语义尚未接入 | 后续单独授权模型路由 |
| UAV Mobility action | 06 §2.1/5.1 | Step 2 Raw + Step 3.3 UAV index/azimuth/elevation/speed tensors | INPUT TENSOR COMPLETE / FROZEN | MINOR_MODIFICATION | UAV action 已张量化、vehicle motion 仍为 SUMO external；尚未接新 graph/model | 后续单独授权模型路由 |
| action routing | 04 §3.3 | `step4_4_structured_rssm_world_model_v1.py` | COMPLETE / FROZEN | STRUCTURAL_CHANGE | 四类动作按 stable slot 局部路由并拒绝无效引用；尚未用于训练/Planner | Definition 05 单独冻结 loss/training |
| RSSM prior/posterior 局部机制 | 04 §3.1/4.1 | `step4_4_structured_rssm_world_model_v1.py` | COMPLETE / FROZEN | MINOR_MODIFICATION | Phy/Comm diagonal Gaussian prior/posterior、mean/sample 与 prior-only rollout 已实现；未训练 | Definition 05 单独冻结 loss/training |
| latent layout | 04 §3.2 | 同上 | COMPLETE / FROZEN | STRUCTURAL_CHANGE | 五类独立 h；仅 Vehicle Physical 与 Comm 有 z；slot 对齐且无 global pooling | 保持 v1 合同 |
| learned / deterministic boundary | 04 §2；05 §1 | Step 4.4 model/rule transition | COMPLETE / FROZEN FOR DEFINITION 04 CONTRACT | STRUCTURAL_CHANGE | 仅 vehicle motion/CSI 为 learned head；outage 为 known stochastic event；Flow/Task/CPU/UAV 为规则 | Definition 05 冻结监督与 Loss |
| 通信状态充分性 | 04 §4.2 第一个边界 | STEP 4.4 audit + model / ChannelManagerCP / WiredNetworkManager | COMPLETE / FROZEN | MINOR_MODIFICATION | 研究者已选择独立 known stochastic outage；wired capacity 从真实 config 读入，Carrying-derived membership 与 simulator equality 通过；无 learned residual | 保持 `learned_service_residual=false` |
| 外生事件 | 04 §2.3/4.2 | 固定entity slot/mask；presence head | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | 已知未来场景还是随机到达过程未冻结 | 研究者定观测与生成边界 |
| deterministic rule feedback | 04 §3.3–3.4 | Step 4.4 `deterministic_transition` | COMPLETE / FROZEN | STRUCTURAL_CHANGE | 每步 learned dynamics 后执行 known stochastic + deterministic rules，并反馈下一 latent | Definition 05 仅定义训练监督 |
| 动态重构图 | 04 §3.4/4.2 | Step 4.4 `rebuild_graph` | COMPLETE / FROZEN | MISSING→IMPLEMENTED | 每步由 predicted state 更新 Physical/Comm/Flow/Task/DAG/Align/GeoComm；topology config 仍 development-only | 不擅自研究冻结 topology |
| Definition 05 decision contract | 05；研究者 STEP 5.0 明确决定 | `contracts_PIJWM_DEFINITION_05_LOSS_TRAINING_EVALUATION_V1.md` | DECISION FROZEN / 5.2 CPU LOOP CLOSED | STRUCTURAL_CHANGE | 10 项决策已冻结；5.1B primitives、5.1D paired integration 与 5.2 CPU loop evidence 已闭合；formal training 仍未开始 | STEP 5.3 CPU Preflight |
| Future Motion / CSI target contract | 05 §1；STEP 5.1A-PATCH 授权 | `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`、`code/scripts/build_step5_1a_motion_csi_target_contract_v1.py` | COMPLETE / FROZEN FOR TARGET CONTRACT | MINOR_MODIFICATION | local one-step Motion；current physical/model comm slots；12 个 non-locked development samples；不是正式 Dataset；5.1D 只在 posterior/loss 路径读取 target | 复用 paired integration；不得自动进入 STEP 5.2 |
| STEP 5.1C unified identity/normalization lineage | 05 §1；4.2A/4.2C-B/C | `build_step5_1c_unified_development_bundle_v1.py`、unified bundle artifact | COMPLETE / FROZEN FOR DEVELOPMENT BUNDLE | MINOR_MODIFICATION | 12 paired samples 的 Physical/Communication slot identity、10/74 capacity、no-prefix 和 dev_train-only stats 已机器闭合；不是 formal Dataset | 复用 5.1D receipt；不得自动进入 STEP 5.2 |
| STEP 5.1D unified model-chain paired acceptance | 05 §1；4.3A/4.3B/4.4 | `build_step5_1d_unified_model_chain_v1.py`、`STEP_05_1D_UNIFIED_MODEL_CHAIN_PAIRED_ACCEPTANCE.md`、tracked evidence artifact | COMPLETE / FROZEN FOR CPU DEVELOPMENT INTEGRATION | STRUCTURAL_CHANGE | 47/47 checks、12/12 pairing、exact upstream train lineage、runtime prior isolation、actual prior/posterior/decoder、gradient 和 deterministic rebuild 已通过；模型仍 untrained；Route/Comp non-empty coverage=0 | 已由 STEP 5.2 复用；下一步 STEP 5.3 |
| STEP 5.2 training loop | 05 §2 | `code/src/pi_jwm/step5_2_training_loop_v1.py`、`run_step5_2_training_loop_smoke_v1.py`、`STEP_05_2_TRAINING_LOOP_CURRICULUM_JOINT_TRAINING.md` | COMPLETE / FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION | STRUCTURAL_CHANGE | Stage 1 future teacher、current-observation posterior initialization、Stage 2 prior recursive `1→2` curriculum、KL warm-up/free bits、joint optimizer groups、global mask-normalized prior-only validation、identity-safe checkpoint/resume；26/26 receipt checks | 仅 8/4 development bundle CPU smoke；full training/GPU/formal performance 未开始 | STEP 5.3 CPU Preflight |
| STEP 5.3 CPU preflight | 05 §2 | `code/scripts/step5_3_cpu_training_preflight_v1.py`、`code/tests/test_step5_3_cpu_training_preflight_v1.py`、`STEP_05_3_CPU_TRAINING_PREFLIGHT_TINY_OVERFIT_GO_NO_GO.md`、`code/artifacts/protocols/pi_jwm_step5_3_cpu_training_preflight_v1_20260922/` | LEARNING_SIGNAL_GO / TINY_OVERFIT_NO_GO | MINOR_MODIFICATION | 修正真实 Phase A pre/post、Phase C H=1→H=2 + KL warm-up、raw Motion/CSI bridge、phase-specific gradients、独立 fresh-run 和 resume；receipt passed=true 但 tiny-overfit 两项 false | 仅 bounded CPU learnability evidence；不可称 tiny-data overfit；full training/GPU/formal Dataset/performance 未开始；Route/Comp non-empty=0/0 | STEP 5.3D CSI scale diagnosis |
| STEP 5.3D CSI scale diagnosis | 05 §2 | `code/scripts/step5_3d_csi_scale_optimization_diagnosis_v1.py`、`code/tests/test_step5_3d_csi_scale_optimization_diagnosis_v1.py`、`code/artifacts/protocols/pi_jwm_step5_3d_csi_scale_optimization_diagnosis_v1_20260922/` | COMPLETE / DIAGNOSTIC-ONLY | MINOR_MODIFICATION | H1/H2 raw bridge exact match；baseline 200-step CSI remains above stronger gate；mean-bias diagnostic reaches normalized CSI MSE <1；formal initialization/decoder bridge not selected | CPU-only observation; no GPU/formal training/locked_test; researcher decision required between raw-head mean-bias and normalized-output bridge | researcher decision |
| STEP 5.3E CSI train-mean bias formalization | 05 §2 | `code/scripts/step5_3e_csi_train_mean_bias_formalization_v1.py`、`docs/implementation_records/STEP_05_3E_CSI_TRAIN_MEAN_BIAS_FORMALIZATION_TINY_OVERFIT_ACCEPTANCE.md`、tracked evidence artifact | FORMALIZATION_PASS / TINY_OVERFIT_GO | MINOR_MODIFICATION | Researcher-selected raw-dB decoder with train-only CSI mean bias formally initialized before optimizer; fixed [0,1] CPU run passes H1/H2 stronger gate; checkpoint contract guards and reproducibility pass | development tiny-data evidence only; full training/GPU/formal Dataset/locked_test/performance remain closed; Route/Comp non-empty=0/0 | researcher review before STEP 5.4 |
| STEP 5.4 GPU training readiness / formal preparation | 05 §2/5 | `code/src/pi_jwm/step5_4_formal_training_readiness_v1.py`、`code/scripts/step5_4_gpu_training_readiness_v1.py`、`docs/implementation_records/STEP_05_4_GPU_TRAINING_READINESS_FORMAL_TRAINING_PREPARATION.md`、readiness artifact | TRAINING_STACK_PASS / FORMAL_DATASET_NOT_READY / GPU_CODEPATH_PREPARED / FORMAL_TRAINING_BLOCKED | MINOR_MODIFICATION | Manifest-driven interface, CPU-only readiness audit, config/checkpoint schema and action coverage | Formal Dataset absent; Route/Comp non-empty=0/0; formal L/topology/training budget require researcher decision; CUDA not executed | researcher decides/formalizes Dataset before next Step |
| training | 05 §2 | STEP 5.2 training loop；历史 `run_formal_dual_graph_gpu_train_v1.py` | IMPLEMENTED FOR CPU DEVELOPMENT / FULL TRAINING NOT STARTED | STRUCTURAL_CHANGE | 旧 staged base-freeze/P4 gate 不复用；当前 loop 只执行少量 CPU optimizer smoke | STEP 5.3 CPU Preflight |
| loss / posterior / KL | 05 §1.2；STEP 5.0 决策 | `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py` + 5.1D/5.2 evidence | COMPLETE / FROZEN FOR CPU DEVELOPMENT PRIMITIVES + LOOP | STRUCTURAL_CHANGE | paired integration 与 CPU loop 已通过；无正式性能结论 | STEP 5.3 诊断 |
| checkpoint selection | 05 §2.2；STEP 5.0 决策 | `Step52Trainer.update_validation_state()` | COMPLETE / FROZEN FOR DEVELOPMENT | STRUCTURAL_CHANGE | selector/early stopping 只依据 prior-only horizon-mean `L_Val`；KL 只作 diagnostic | STEP 5.3 CPU Preflight |
| prediction evaluation | 05 §3；STEP 5.0 决策 | `code/src/pi_jwm/step5_1b_posterior_loss_metric_v1.py` + 5.2 validation smoke | CPU DEVELOPMENT EVIDENCE / NOT PERFORMANCE FROZEN | MINOR_MODIFICATION | 逐 horizon raw-unit Motion/CSI metrics 和 prior-only `L_Val` 可执行；未形成性能结果 | STEP 5.3 后再做 learning-signal/overfit 判断 |
| learned candidate generation | 06 §4.3 | planner的callback | AUDITED / NOT_STARTED | MISSING | 尚无新四类动作proposal模型/训练 | 后续单独授权 |
| legality / fallback / warm start | 06 §2.2/3.3/4.3 | CandidateAction.legal、空集raise | AUDITED / NOT_STARTED | MISSING | bool过滤不等于合法性规则；无安全fallback与warm start | 冻结场景约束 |
| candidate rollout | 06 §3.1 | `FormalCandidateRolloutPlanner.plan` | AUDITED / PROTOTYPE_ONLY | MINOR_MODIFICATION | 逐候选调用可复用；共同latent快照/随机评估/新动作未接通 | 保留原型，后续适配 |
| planner objective | 06 §3.2 | prediction_extractor/objective回调 | AUDITED / OPEN | RESEARCHER_DECISION_REQUIRED | 权重、风险形式、未来硬约束、proposal训练目标未定 | 决策前提交备选证据 |
| execute-first-action | 06 §3.3/5.1 | selected_first_action切片 | AUDITED / PROTOTYPE_ONLY | MINOR_MODIFICATION | 只返回第一步，未接真实四类动作执行器 | 后续接入AirFogSim |
| real feedback | 06 §5.2 | 旧R6运行器；formal replan接口 | AUDITED / NOT_STARTED | MISSING | 新planner无真实history/state/latent更新通路 | 后续真实一步反馈验收 |
| replanning / 完整closed loop | 06 §5 | replan(updated_batch) | AUDITED / NOT_STARTED | MISSING | 人工传入batch不等于真实环境反馈闭环 | 后续两决策步端到端验收 |
| 旧结果与checkpoint | 旧P4协议；新00–06 | 两seed验收、v5 tensor、旧实验 | ARCHIVED_IN_PLACE | HISTORICAL_ONLY | 新语义/布局/动作/损失不同，不能外推 | 保留原结果作历史回归 |

## 已完成、未开始与待决策

- 本轮完成范围：Step 1 审计矩阵、只读权限与新工作流、实施记录框架、历史逻辑归档、导航与注册表同步；实际验证见 Step 1 报告。
- 新方案的 Raw→Dataset/Tensor→Typed Graph Builder→Dual-Graph Encoder→Structured RSSM World Model Contract 已按当前最小合同完成并冻结；World Model 仅有未训练 CPU 机制证据，loss、训练、candidate proposal 和在线闭环仍未开始。
- 研究者已明确目标：严格Physical/Information划分，四类动作含CPU与UAV，结构化RSSM，只学习未知动态，混合proposal+世界模型选择。无需再次确认这些方向。
- Definition 05 的 distribution、loss、posterior、KL、overshooting、training stage、fixed-support mask、rule-state supervision、trainable modules 和 validation/evaluation 已决定；具体超参数、正式 Dataset资格和训练预算仍待后续独立 Step 冻结。未知未来到达与离开、proposal训练方式、objective权重/风险/硬约束/fallback仍待后续处理。当前通信 residual 明确关闭。

## 当前 STEP 4.4 结果

STEP 4.4 已将冻结 `Z_t^{PI,L_g}` 接入五类 entity/relation-aligned deterministic state，并只为 Vehicle Physical 与 Communication 建立 stochastic state。PATCH3 明确：冻结 current-side 输入没有 `Task.return_size`，所以无 Return slot 表示 requirement unknown，而不是 no-return；真实 adapter 的 unknown 状态在 computation finished 后输出 unresolved/blocking side-state，Existing Return 仍只按 `(task_index, flow_type_index=Return)` 绑定，Future Target 不改变 object support。DAG release 只计算有效前驱，并要求所有有效前驱完成；terminal Flow completion 使用冻结 `FLOW_STATUS_VOCAB` 同步 remaining/presence/carrying/status，partial/intermediate 不会错误 completed。focused tests 30/30、正式 receipt 92/92、6 文件独立重建 hash/size equality 均通过，STEP 4.4 = COMPLETE / FROZEN。证据始终仅为 untrained CPU development，`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false`。

## Checkpoint 与结果复用边界

| 变更 | 对旧checkpoint/tensor的影响 | 允许复用 |
| --- | --- | --- |
| Physical/Agent/Comm分拆、特征顺序/单位更改 | 编码器输入语义/尺寸改变；即使同shape也不兼容 | 通用算子、ID/mask工具；不得直接strict-load新模型 |
| Flow/Task去随机状态、Phy/Comm新latent | 参数键和含义改变 | prior/posterior/KL的实现思想；权重迁移需独立证明 |
| 四类动作与逐RB/UAV通路 | action encoder与训练分布改变 | 仿真器接口、旧事件记录作来源证据 |
| 单步规则反馈、动态图、loss/selector | 即使部分权重可加载，旧精度结论也不再适用 | 历史checkpoint复现和回归，不作新验收 |
| 旧两seed单seed验收 | 原数字与原协议保留；并非新00–06性能证据 | 有边界的历史对照；不自动补第三seed |

## Raw Trajectory Layer / 01 已完成并冻结

Step 2.1 v4 完成真实单步验收；Step 2.2 完成真实多步独立重采集与 no-op 连续性；Step 2.3 用 8 个连续真实步完成因果和字段收尾；Step 2.4 在真实 `RSU_0 ↔ cloudServer_4` 有线链路上补齐 Communication Outcome。AirFogSim 未来 schedule 只保留为 internal metadata，不进入 `O_t`、History 或 input-side Entity Index；Decision CSI、CPU capacity/missing mask、wireless/wired split slot service、total 聚合、`setTaskReturnRoute` 和 return lifecycle 均有真实证据。AirFogSim raw acceleration 与 PI-JWM canonical backward difference 使用不同字段，缺历史时为 `null + mask=false`。最终机器证据位于 `code/artifacts/protocols/pi_jwm_raw_contract_causal_complete_v2_20260919/` 和 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，并要求纳入 Git。

## STEP 3.1F Model-ready History Contract Finalization

最小真实证据位于 `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`。History frame `1,2`中明确保留 `O_1+A_1+Y_1+O_2`，Future Action/Target frame `2,3`；input-side index 是整个 History 对象并集，machine policy 为 `history_causal_observable_object_union`，不包含 target-only `Task_7/Task_8`。Future Action 先做 anchor visibility 检查，再使用同一 History-union static index；四类 action、历史 flow/relation/DAG 对齐、固定 presence/mask、ID↔index 校验和 round-trip 均有机器检查。该结果冻结最小 schema 语义，不等于正式数据集完成。Future-reference 观察扫描 JSON 已纳入 artifact provenance，不作自动丢弃或研究决策。

## 当前 Step 3.2 结果

STEP 3.2 已完成最小 non-locked validation：3 条独立 development trajectory、12 个 H=2/L=2 windows，trajectory-level split，train-only mask-aware normalization，deterministic rebuild 和 serialize/load 均有机器证据。该结果不是正式 Dataset、训练或泛化结论。

STEP 3.2-PATCH 已补齐 Dataset isolation evidence：Raw provenance 含真实 trajectory/seed/source SHA/config lineage/time range/slot duration/sample contract；time-grid 与 step start/end 对齐在 window construction 前检查；train/validation trajectory 无交集；batch future-reference audit 独立保存并统计 12/12/0/0，仍为 observation-only；normalization 单位为 `m/s`、`m/s^2`、`AirFogSim data-unit`。

STEP 3.2-PATCH-RECEIPT 已修正最终机器验收：顶层 `passed` 现在是全部 required checks 与 `locked_test/training/gpu/formal_dataset=false` scope checks 的逻辑 AND；负向 fixture 已证明单项失败会使 `passed=false`；provenance contract version 直接复用冻结的 `model_ready_sample_contract_v1.SCHEMA_VERSION`。STEP 3.2 现正式 COMPLETE / FROZEN。

## 当前 STEP 3.3 结果

STEP 3.3F 已补齐 JSON→Tensor 语义：Past Outcome 使用独立 `H-1` 轴，Target 保留 entity/task/flow/service future facts，Comp 读取 `allocated_cpu_per_s`，entity/lifecycle/route/transport 使用固定 vocab，Static entity type 来自 causal History。semantic validation receipt 由 required checks 实际 AND。artifact 位于 `code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/`。因此 STEP 3.3 正式 COMPLETE / FROZEN，02 数据集构建与模型输入已按当前最小数据合同冻结；这不等于正式大规模 Dataset 已生成，也不决定 03/04 最终 feature selection。

## 当前 STEP 4.1 结果

STEP 4.1 当时完成 Physical / Information object-field-relation mapping，并以 `DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1` 停止；该历史 readiness 已由后续 4.2A/4.2C 数据闭合和 STEP 4.3A builder 实施覆盖。其 CPU capacity/allocation/service/available resource 分离及禁止旧 mixed `physical_edge_state` 的语义仍有效。

## 当前 STEP 4.2A 结果

Step 4.1 中已有可靠来源的 position、wireless per-RB CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 已通过独立版本链贯穿 Raw amendment → Sample → train-only preprocessing → Tensor。PATCH 已把 wireless structural validity 与 CSI observability 解耦：CSI missing 不删除 relation，mask=false 的 numeric placeholder 为 0；wired valid/no-CSI 保持合法。三条 development trajectory 共形成 12 个样本；这是机器合同证据，不是正式 Dataset 或容量结论。return size/priority/deadline 已核实存在 simulator observer source，但冻结 Raw 未透传；stable stateful Flow 等仍为 Raw-insufficient。该 4.2A Step 当时尚无 Physical topology 与 graph object，之后已由 4.3A/4.3B/4.4 完成。机器 artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。
Step 4.1 中已有可靠来源的 position、wireless per-RB CSI、wired typed relation、CPU static capability、Task demand/progress/elapsed 与 Src/Host/Exec/Ret 已通过独立版本链贯穿 Raw amendment → Sample → train-only preprocessing → Tensor。PATCH 已把 wireless structural validity 与 CSI observability 解耦：CSI missing 不删除 relation，mask=false 的 numeric placeholder 为 0；wired valid/no-CSI 保持合法。三条 development trajectory 共形成 12 个样本；这是机器合同证据，不是正式 Dataset 或容量结论。return size/priority/deadline 已核实存在 simulator observer source，但冻结 Raw 未透传；stable stateful Flow 等仍为 Raw-insufficient。Physical topology 与 graph object 均未实现。机器 artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`。

## 当前 STEP 4.2C-C 结果

STEP 4.2C-C 已完成并冻结 Raw Flow → Model-ready Sample → CPU Tensor additive extension。新增独立 `logical_flow` History-union/target namespace，Flow 与 Carrying state 分离，completed/superseded 与 padding 分离，五个连续字段仅在 `dev_train` 且 `presence=true AND feature_mask=true AND value!=null` 上标准化，capacity overflow 显式拒绝；Sample/Tensor 不重新解释 C-B Raw。随后 `STEP 4.2C-C-PATCH` 补齐 presence-aware stats、History/target Logical/Carrying 四组全字段 semantic equality 和完整 target carrying namespace，并由 receipt 实际 AND 子检查、target namespace、future Epoch、placeholder/bounds/route mask 与 normalization policy。23 项 focused tests、4.2B/4.2A/3.3 回归、真实低 wired capacity 跨时隙 trace、deterministic rebuild/hash、serialize/load 和 receipt/semantic tamper 均通过。artifact 位于 `code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/`；真实 Return multi-hop、reroute runtime 与 formal capacity 仍未声称。

这是 STEP 4.2C-C 当时的历史范围：`graph_builder=false`、`information_graph=false`、`physical_topology=false`。之后 STEP 4.3A 已完成 Graph Builder；`training=false`、`gpu=false`、`locked_test=false`、`formal_dataset=false` 继续有效。

## 当前 STEP 4.3B 结果

STEP 4.3B 使用完整冻结 History Tensor 编码 Physical/Agent/Task/Flow 的对象级时间状态，并使用 STEP 4.3A current typed graph 完成 Physical/Comm/Flow/Task-Agent/DAG 分族有向传播、relation-wise masked mean、P2A Align 与 wireless-only P2C GeoComm。Logical Flow 与 Carrying 分支独立编码后融合，最终只产生一个 Flow relation latent。输出保持 entity/relation alignment，只到 `Z_t^{PI,L_g}`。artifact 是 `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`；不代表 `xi_t^Lat`、World Model、预测或性能。数值配置与 Physical topology 均为 development-only，`research_frozen=false`。

## 下一步边界

## STEP 6.2A-CLOSURE — No-Retrain + Single-Hop Planner v1

研究者已接受 ROUTE-RECOVERY 的 no-retrain salvage：best checkpoint 保留，Formal train/validation multi-hop coverage 均为零。Planner v1 Route 保持 enabled 但限制为 `FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1`；multi-hop deterministic code 保留为未来扩展/消融。当前 `STEP_6_2B_READINESS=READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`，只开放 scorer/comparator 的 CPU 合同实现；`CLOSED_LOOP_READINESS=NOT_READY`、candidate method `RESEARCH_PENDING`、multi-hop `NOT_IN_V1_DOMAIN`。任何未来 formal training 前必须通过 `CROSS_LAYER_RULE_SEMANTICS_GATE=PASS`。证据与实现记录见 closure receipt bundle 和 `STEP_06_2A_CLOSURE_NO_RETRAIN_SINGLE_HOP_PLANNER_V1.md`。

STEP 4.3A、STEP 4.3B 与 STEP 4.4 均正式 COMPLETE / FROZEN；STEP 5.0 已冻结 Definition 05 决策和复用审计。STEP 5.1A-PATCH 已冻结 local one-step Motion、current physical slots 与 current model CSI slots。STEP 5.1B-PATCH、STEP 5.1D-PATCH 与 STEP 5.2 已 COMPLETE/FROZEN FOR CPU DEVELOPMENT PRIMITIVES + PAIRED/TRAINING-LOOP INTEGRATION；full training、GPU、`locked_test`、formal Dataset、baseline 与 Planner 仍未开始。
# STEP 6.2A — Planner Objective Source & Semantics Audit (2026-09-28)

CPU-only source/provenance audit and protocol foundation completed with verdict `PASS_WITH_READINESS_BLOCKERS`; `STEP_6_2B_READINESS=BLOCKED`. The source receipt distinguishes AirFogSim runtime and observer availability from Formal Raw/Sample/Tensor/Graph/World Model/Planner exposure. Deadline is arrival-relative duration: completion accepts `delay <= deadline`, while the later active-task sweep fails only at `delay > deadline`; the no-Return completion path has its separately recorded `1e-5` tolerance. Current Planner causal side-state lacks aligned deadline/arrival/priority/Return support; cross-hop E2E burden proof and normalized Route effort denominator also remain incomplete. Contracts and evidence are in `docs/contracts/PIJWM_STEP_06_2_PLANNER_OBJECTIVE_CONTRACT_V1.md`, `docs/contracts/PIJWM_BASELINE_SYSTEM_METRIC_INTERFACE_V1.md`, and `code/artifacts/protocols/pi_jwm_step6_2a_planner_objective_semantics_audit_v1_20260928/`. No scoring, ranking, baseline, locked test, GPU, or closed-loop run was performed.
## 2026-09-28 STEP 6.2A-PATCH — objective readiness reconciliation

`STEP_6_2B_READINESS=BLOCKED` after current-source recalculation. 4.2C-B/C supplies conserved E2E Flow state; exact-aligned non-locked validation replay supplies Planner-only deadline, while accepted Formal Raw supplies Return size/destination. Effort and throughput decisions are synchronized. The current 4.2C-C destination-only route array conflicts with 4.4 cross-hop indexing, and Route action leaves the array stale. See `docs/implementation_records/STEP_06_2A_PATCH_PLANNER_OBJECTIVE_READINESS_RECONCILIATION.md` and the 01–12 receipt bundle. Scorer, ranking, baseline, closed loop, GPU and locked_test remain unopened; next action requires researcher decision on route semantics.
**2026-09-30 STEP 6.3D-3080TI-MIGRATION-QUALIFICATION（PASS）：** RTX 3080 Ti 上正式 Dataset/Raw/checkpoint、CPU/GPU 离散等价、正式 CUDA runner/resume 与 bounded TRAIN smoke 通过；FP32 batch16 冻结。正式 TRAIN tuning、Validation 比较、`locked_test` 未运行；future Return birth 限制不变。详见对应 Step 实施记录和机器收据。
