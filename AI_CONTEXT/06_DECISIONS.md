<!-- STEP6.4I CURRENT -->
## 2026-10-09 STEP 6.4I — Pure-search正式闭环CPU准备

STEP_6_4I=PASS（仅CPU正式闭环准备）；SEARCH_METHOD=S-CEM，K=4/rho=0.2；FINAL_CLOSED_LOOP_BUDGET=512，PLANNER_V1_CLOSED_LOOP_B_WM=512，H=4。研究者本轮明确接受6.4H所见Effort质量损失换取计算节约；B1024保留参考，不声称等价/最优，不冻结hybrid预算。Route=EXPLICIT_NOOP_ONLY；fallback=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1（一次A失败即C，无重试）；同步暂停仿真、非实时。

独立CPU真实S-CEM机制：固定TRAIN anchor0003，engineering-only B64/batch1，两次winner第一步真实执行0.6→0.7→0.8s；每次实际64 unique transitions、13个不同可评分H4。第二root由新真实观测/History/后验重新构建，未复用预测root，模型与TRAIN normalization不变。固定TRAIN anchor0041注入NO_SCOREABLE_H4，canonical fallback A经同桥真实执行4.5→4.6s一次；不是搜索性能结论。74项focused CPU tests通过，覆盖空域、缺字段、bridge/setter/step异常、部分写入、重复决策保护、Return支持与Future Target poison。

CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_PROTOCOL_REVIEW；真实CPU链READY，通用episode组件已测试，live CUDA provider=IMPLEMENTED_NOT_GPU_VERIFIED。正式episode来源/数量/seed/长度、指标分母、timeout及baseline仍待研究者批准；GPU_LAUNCH_AUTHORIZED=false，READY_FOR_GPU_LAUNCH=false。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；Stage B=NOT_STARTED/DEFERRED；GPU=NOT_USED；locked_test=false。

证据：docs/implementation_records/STEP_06_4I_PURE_SEARCH_CLOSED_LOOP_PREPARATION.md；docs/contracts/PIJWM_STEP_06_4I_FORMAL_CLOSED_LOOP_PROTOCOL_DRAFT_V1.md；code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2/15_acceptance.json。唯一下一动作：研究者审阅Protocol Draft，批准最小GPU pilot与首个live CUDA资格检查。本轮停止。以下较早内容均为历史时点，旧pending不覆盖本轮明确接受预算决定。

Researcher Decision（2026-10-09，原STEP6.4I请求）：正式接受S-CEM K4rho0.2 B512H4、修复域、冻结WM/数据/Objective、RouteNOOP、一次A否则C，非实时同步暂停；先慢Planner闭环再learned proposal。Codex建议的规模/metrics/timeout/费用不是研究决定。
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

Researcher Decision：本轮显式冻结E_comm(t)=有效唯一active wireless Flow绑定∩Task present∩offloading/transmitting∩not completed；只改Planner动作资格。K4/rho0.1/B512、Route NOOP保持。Codex证据建议57anchor targeted requalification不是新实验授权或研究者决定。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING。

证据：`docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/09_final_acceptance_receipt.json`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/07_formal_search_evidence_reuse_assessment.json`。历史6.4E BLOCKED及旧搜索accepted状态仅在旧定义内保留。

以下为历史状态，旧PASS不代表新资格定义下已经重新验证。

---

## 2026-10-04 STEP 6.4E 预算决定与真实执行阻塞（当前）

STEP_6_4E=BLOCKED，CLOSED_LOOP_PRE_FORMAL_READINESS=BLOCKED。研究者正式冻结 PLANNER_V1_CLOSED_LOOP_B_WM=512，仅 MH-CEM K4/rho0.1 pure-search；B1024保留高预算参考，不声称等价/最优，6.4D后两项目标质量损失仍记录。统一fallback接口CPU合同就绪，33 focused CPU tests通过。真实TRAIN固定anchor0041、t=4.5s的非空Comm Task_1/RB0被执行前安全门拒绝：Domain仍绑定已failed的Task_1/Task_6，无本轮setter或env.step，第二场景未启动，无换动作/样本或科学失败重试。候选A可确定性准备合法当前动作但本轮真实执行未取得；候选B缺独立current behavior offer provider；C无动作终止及空域/缺字段/bridge/setter负路径通过。当前Flow支持与真实task执行条件不一致，Status=Awaiting Researcher Decision；不改Domain/Objective/Return。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN；原Stage B=NOT_STARTED、DEFERRED / NOT_REQUIRED_FOR_CURRENT_MAINLINE；GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并决定一致性修复与复验范围；不自动开跑。

证据：`code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/10_pre_formal_readiness_receipt.json`；`docs/implementation_records/STEP_06_4E_BUDGET_COMM_FALLBACK_CLOSURE.md`。
研究者决定来源：本聊天 STEP6.4E 第1节，正式B512仅pure-search。Codex recommendation：一致性冲突及真实执行证据闭合后，NO_SCOREABLE_H4考虑A，关键错误考虑C；这不是Researcher Decision，最终策略仍待决定。

以下为历史状态，较早预算待决定或机制PASS不覆盖本轮阻塞。

---

## 2026-10-04 Researcher Decision / 人类确认

研究者此前明确授权本轮实验结束且结果本地SHA/归档通过后关闭对应AutoDL实例；本聊天回复“我已经关机了”。记录为人类手动关机确认，不是Codex UI核验。6.4D统计建议不是新的科研决定，最终closed-loop预算仍待研究者决定。

## Researcher Decision — STEP 6.4D

研究者本轮只授权完整64×5 MH-CEM B512及与既有B1024配对；可严格证明同义的6.4C48份B512必须优先复用；协议先提交再GPU。B256扩样/原Stage B/闭环未获授权，最终预算仍待研究者决定。

## STEP 6.4C 验收后的决定边界

研究者本轮仅授权16×3 MH-CEM B256/B512校准及复用B1024。CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；扩样64×5是Codex的ENGINEERING/RESEARCH RECOMMENDATION，不是Researcher Decision，未授权。METHOD仍仅pure-search backbone；Stage B=NOT_STARTED，locked_test=false。

## 2026-10-03 Researcher Decision：STEP 6.4C 小规模预算校准

研究者本轮明确授权：16个仅按运行前静态分层选择的Validation anchors×3 seeds，MH-CEM K4/rho0.1，仅新增B256/B512，复用配对B1024。只作预算取舍证据，不重新选择方法，不自动冻结closed-loop budget，不运行原完整Stage B或环境动作。3080Ti不可达时完成CPU预检后停止。

## 2026-10-03 STEP 6.4B 最小闭环真实反馈验收（当前）

STEP_6_4B=PASS，TWO_CYCLE_REAL_FEEDBACK_SMOKE=PASS，CLOSED_LOOP_MECHANISM_READINESS=PASS。固定TRAIN fixture、MH-CEM K4/rho0.1、CPU FP32 batch1、B64：仅执行首轮winner第一动作，仿真0.6→0.7秒，真实新观测重建state/graph/current posterior并完成第二次规划；下一轮root不是上一轮预测。两轮各64独特转移（合计128）；另保留一次动作执行前回执异常的64次尝试，实际总计192。live tensor/deadline、旧搜索不变性、防未来泄漏及参数/归一化不变全部PASS。SEARCH_METHOD=MH-CEM仍仅pure-search backbone；READY_FOR_BUDGET_CALIBRATION=true只是机制就绪，不是运行授权。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN；future Return-birth fixed-support limitation保持。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并单独授权预算校准协议，不自动运行。

Researcher Decision（2026-10-03，用户本轮明确决定）：SYNCHRONOUS_PAUSED_SIMULATION；0.1s仿真interval不是墙钟deadline，不宣称realtime。Smoke-only FAIL_CLOSED_AND_STOP_EPISODE，六种规定失败记录原因并终止；不是最终fallback。禁止H3/NOOP/旧policy fallback，不实现learned proposal/warm start/Route扩展。

记录：`docs/implementation_records/STEP_06_4B_MINIMAL_CLOSED_LOOP_TWO_CYCLE_SMOKE.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/13_final_acceptance.json`；合同：`docs/contracts/PIJWM_STEP_06_4B_LIVE_CLOSED_LOOP_SMOKE_V1.md`。以下为历史记录，其中“当前”仅表示记录时点。

---

## 2026-10-03 STEP 6.4A 闭环前就绪审计（当前）

STEP 6.4A审计完成（审计证据PASS），CLOSED_LOOP_READINESS=BLOCKED。SEARCH_METHOD=MH-CEM仍只为pure-search backbone，非最终hybrid。局部链路7 IMPLEMENTED/3 PARTIAL/5 MISSING；缺live history-only输入/belief/sidecar、winner动作序列/首动作及真实ID执行桥、研究者fallback和latency决定、两轮真实反馈MH证据。MH B1024搜索median90.506/P90 204.916/P95 284.152/max421.701秒；仿真slot0.1秒不是墙钟deadline，DECISION_LATENCY_REQUIREMENT=UNRESOLVED，尚无在线执行证据。Stage B用途由研究者冻结为online-budget trade-off，不能重选算法；A约30.46–41.85h、B NEW PROPOSAL约10.26–14.06h、C NEW PROPOSAL约1.54–2.11h，仅成本情景。4090_MIGRATION_NOT_YET_JUSTIFIED。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作研究者确定fallback及decision latency模式，再授权最小接口集成；不自动运行实验。

研究者本Step明确Stage B只服务预算选择，SEARCH_METHOD保持MH。Codex推荐C但没有冻结新subset/seed/runner或fallback；B/C须单独研究者授权。

记录：`docs/implementation_records/STEP_06_4A_CLOSED_LOOP_READINESS_STAGE_B_PURPOSE_FREEZE.md`；机器审计：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/06_go_no_go_receipt.json`。下方为历史状态，不能用Stage A PASS替代闭环readiness。

---

## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

这是执行研究者此前冻结的选择规则所得结果，不是Codex新增科研决定。S/MH都通过对HRS优势门，MH又通过对S优势门，因此选择MH。Stage B仍须研究者单独授权，最终hybrid结构/解释仍归研究者决定。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-01 研究者授权：Stage A backbone选择边界

研究者明确只授权B1024 Stage A正式比较，960 cases后硬停止，Stage B另行授权；SEARCH_METHOD只表示Planner v1 structured-search backbone/pure-search baseline，不等于最终hybrid PI-JWM planner冻结。该决定未改变算法、Objective、候选域、参数或既有Return-birth支持边界。Stage A已启动，结果尚未验收。

## 2026-10-01 STEP 6.3D BLOCKER CLOSURE

研究者2026-10-01明确 newly scoreable 指当前iteration实际完成且可评分的不同fingerprint≥2；历史重采样可计，轮内重复只计1，retained-only不计。研究者授权runner阶段/summary/身份修补，不授权开跑。冻结顺序1024→STOP→审阅；以后另行授权256→512。主选择/参数/模型/支持边界不改。 依据：`docs/contracts/PIJWM_STEP_06_3D_VALIDATION_STAGES_AND_CEM_UPDATE_SEMANTICS_V1.md`、`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md`。

# 已确认决策

## 2026-09-29 STEP 6.3D — Researcher Decision

研究者明确要求在同一 6.3BC CandidateDomain、interleaved frozen World Model rollout、6.2B 五项目标严格字典序与 `B_WM` 下比较 HRS、S-CEM、MH-CEM。三者共享五层结构化 proposal，第一层是 Comm+Comp+Mob 联合模式；HRS 全部合法层均匀且不更新，S-CEM 只更新模式/选中任务数，MH-CEM 还更新任务子集/行分配/RB 起点。CEM 只用当前轮完整可评分 elite，`η=0.5`、`ε=0.05`、保留精英最多 5；不足 2 条 H4 可评分候选不更新。正式 H=4 只比较 `H_sup=4`，禁止 H1–H3 回退。TRAIN 32 锚点/3 seeds/512 预算仅调各 CEM 的 K∈{3,4}、ρ∈{0.1,0.2}；Validation 64 锚点/5 seeds/三个预算成对比较，最终只以 1024 预算 anchor cluster bootstrap 的 95% CI 与冻结简约规则选方法。研究者当前指定先仅用本机 CPU；这不是对性能结果或最终方法的预先裁决。

## 2026-09-29 STEP 6.3B/6.3C-PATCH — Researcher Decision

研究者明确删除 6.3B 旧的“全部当前 eligible wireless Task 必须有 Comm row”要求。Planner v1 的 `E_comm(t)` 只含能从当前 causal/predicted state 中**已有** wireless Flow 唯一绑定的 Task；当步实际选中集合 `S_comm(t) ⊆ E_comm(t)`，未选任务只是本时隙未调度。对每个 TRAIN 已见 Comm row-width 结构，仅允许 TRAIN 在该结构下观察过的 unique selected-task-count；每个选中 Task 至少一行，同 Task 可多行。依赖同一 decision 的 Route 动作才新建/启动 Flow 的历史 Comm row 不进入 Planner v1 正式候选或自重放目标；保留原始 TRAIN，并分开报告原始动作、Planner v1 投影和排除行原因。Route 显式空、145 个 `(start,width)`、width 1/2/3、251 个 joint 签名、Comp/Mob、时间标签、CandidateDomain、interleaved rollout、`B_WM`、Objective、冻结模型/检查点均不变。其余残余失败不得为追求 100% 重放而放宽政策。

## 2026-09-29 STEP 6.3C — Protocol implementation boundary

Researcher-frozen 6.3B grammar and admission are unchanged. The implementation fact is that all future search methods must call one state-conditioned CandidateDomain and one interleaved one-step rollout protocol; `B_WM` counts unique admitted candidate-step transitions, with deterministic cache hits free. This is an implementation contract, not a choice of optimizer. Empty domains remain explicit dead ends and do not trigger horizon backoff.

## 2026-09-29 STEP 6.3B — Researcher Decisions

- Planner v1 允许一个 Task 多条 Comm row；修正旧 6.0A 重复 Task 拒绝。每 row 保持 TRAIN width 1/2/3、循环连续 block，正式池按 TRAIN 观察到的 `(start,width)` 对与当前因果关系绑定。旧“有 eligible wireless Task 时每个都要有 row”要求已由上方 6.3B/6.3C-PATCH 研究者决定取代。
- 正式 Mobility 候选只允许在场 UAV 共享同一个 HOLD/PROFILE_1–5；底层 6.0C 独立控制接口不因此删除。
- 正式候选池只准 Formal TRAIN 已见的 Comm–Comp–Mob 联合结构；边际已见但联合未见只保留未来消融标签。时间支持只贴标签，相邻历史转移不作硬门槛。
- 有当前可分配 computing Task 时 Comp 必须按现有因果 CPU base 与全局 alpha `{0.5,0.75,1.0}` 提交完整非空行；仅无该 Task 时 Comp 空 family。未来预测状态需显式 computing/唯一 Exec 才能构建 Comp base。
- 这些决定只冻结候选语法和支持政策，不选择 Random/CEM/其他优化器，也不修改既有 Objective。

## 2026-09-28 STEP 6.2B-PATCH — Researcher Decision

- `PLANNER_V1_ROUTE_POLICY=EXPLICIT_NOOP_ONLY`：Planner v1 每个 horizon 的 Route family 为空；pending、已有 Flow 同路径、多跳、改目的地等所有非空 Route 候选均不得进入正式 v1 rollout。
- Route schema/interface、4.4 learned Route encoder 与底层确定性多跳能力保留；Planner v1 实际优化动作族为 Comm/Comp/Mob。此策略是当前 learned-support 边界，不是 AirFogSim 物理合法性判断。
- Objective `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)` 及严格字典序不变。此前 6.2A-CLOSURE 的单跳 Route 决定已被本次更严格的 v1 决定取代；下文原记录保留为历史。

## 2026-09-28 STEP 6.2A-CLOSURE — Historical Researcher Decisions

- 保留当前 Formal `best.pt`，`RETRAIN_AFTER_ROUTE_RECOVERY=false`；SHA-256 保持 `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`。
- Planner v1 Route 保持启用，但 action domain 限定为 `FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1`：单节点直达当前冻结 destination；multi-hop code 保留但移出 v1，作为后续扩展/消融，不作正式 learned-performance claim。
- Future formal training 的 mandatory prerequisite 是 `CROSS_LAYER_RULE_SEMANTICS_GATE=PASS`。
- 以上是研究者决定；完整 patched validation 未执行，旧 `LVal` 不改标为 patched 结果。

## 2026-09-28 STEP 6.1 授权边界

研究者授权执行冻结模型上的 H1–H4 候选推演机制与敏感性预检，明确禁止目标函数、候选选择、训练、GPU、baseline、locked_test 和闭环。本 Step 未新增科学方法、指标阈值或 Search/Learned/Hybrid 选型决定；6.0C 操作域和 5.6C checkpoint 身份保持不变。

## 2026-09-26 STEP 6.0C 研究者决定

**Researcher Decision A**：Planner v1 Comp 使用 `STATIC_PER_SLOT_BUDGET_V1`，同一节点同一时隙的分配总和不得超过当前观察到的原始单位静态 CPU 容量；容量缺失时拒绝正分配。这不是动态可用 CPU。**Simulator Fact**：动态可用 CPU 没有可靠决策时刻来源；`dynamic_available_cpu_available=false`，但 `planner_requires_dynamic_available_cpu=false`。

**Researcher Decision B**：第一版正式 Planner rollout 的 UAV 域使用 `FORMAL_DATASET_MOBILITY_CORE_DOMAIN_V1` 六档控制，当前 heading/elevation 来自因果 Raw Planner-only side-state，HOLD 是显式零速命令。每 UAV 独立属于六档；不同 UAV 组合只标 marginal support，不称为已观测 exact joint support。

**Researcher Decision C**：这是保守的首次 rollout 操作域，不是最终最优动作空间、仿真器硬界或安全规则。

**Research Pending**：Search/Learned/Hybrid；连续或插值动作扩展；非零 elevation；空间/geofence；核心域外 OOD 处理；Planner objective/risk；fallback 安全声明；最终动作空间消融。

## 2026-09-26 STEP 6.0B 审计事实与待决

**Implementation/Simulator Fact（非 Researcher Decision）**：只证实静态 CPU 容量，未证实动态可用 CPU；UAV 示例配置和正式数据支持范围均非 simulator hard action bound。候选代码未改，两个 UNKNOWN 保留。

**Research Pending**：动态 CPU 可行性如何定义/新增因果来源；UAV Planner 数值与空间域、越界策略和安全声称；若要求精确 AirFogSim Git commit，需补本地目录缺失的独立 Git provenance。此处没有新的研究者方法决定。

## 2026-09-26：STEP 6.0A 研究者授权边界

**Researcher Decision**：后续使用 MPC 滚动选择框架，动作仍为 Route/Comm/Comp/Mob，Vehicle 移动外生；候选生成采用统一可插拔合同；正式模型完成前不做模型依赖的候选评价。当前实现 horizon 最大 4 是 H1–H4 证据边界，不是最优值结论。

**Research Pending**：Search/Learned/Hybrid 最终选型；优化器、proposal 结构与训练、数量/迭代/elite 等超参数；目标权重、风险、未来硬约束、动态 CPU 与 UNKNOWN 策略、fallback 安全性、延迟预算。Hybrid 没有被选定。

这里只记录研究者明确作出的科研或工程决策。Codex 分析、候选建议和实验现象不能自动写成 Researcher Decision。

## 2026-09-18：新定义与 Step 实施工作流

**Researcher Decision**

- `D:\shen\OB\科研\PIJWM` 中最新 `00–06` 是当前目标研究定义，该目录严格只读；所有工程修改留在 `D:\shen\PKU\PIJWM`。
- 旧 P4/P6/P0–P10 计划、实验、artifact 和 checkpoint 保留为 Historical / Archived evidence，不再作为当前执行主线。
- 当前只授权 Step 1 定义—实现审计、治理和 Tracker/记录框架；禁止自动进入 Step 2、正式训练、模型重构或新 planner。
- 每个有效 Step/Substep 必须验证、记录、commit、push、固定格式汇报并停止；新定义正确性优先于旧 checkpoint 复用。

## 2026-09-10：三方长期协作边界

**Researcher Decision**

- 研究者负责核心算法、总体架构、数学建模、科研 loss 设计、实验目的、消融变量、创新点和最终科研解释。
- ChatGPT Web 负责理论讨论、推导、架构分析、实验设计讨论和结果分析。
- Codex 作为 Research Engineer，负责实现、维护、实验执行、测试、代码事实检查和状态同步。
- Codex 可以主动报告冲突和风险，但不能自行替换科研算法。

影响范围：`AGENTS.md`、`AI_CONTEXT/`、以后所有代码与实验任务。

## 2026-09-10：GitHub 与 AI_CONTEXT 工作流

**Researcher Decision**

- GitHub `main` 是网页端可见的动态事实来源。
- `AI_CONTEXT/00_PROJECT_STATE.md` 是 ChatGPT 新对话的首入口；实现事实仍以源码/config/experiment 为最高优先级。
- 每个有效、经过合理验证的小任务应形成清晰 commit 并 push；broken state 不得推到 `main`。
- 完成本次授权更新后，Codex 不得自行修改 `AGENTS.md`，除非用户再次明确要求。

## 2026-09-09：第三 seed 与同步延期

**Researcher Decision**

- seed `20260832`、远端训练和远端同步延后。
- 现有入口保留，但不得自动启动；机器守卫见 `docs/registries/deferred_work.json`。

## 2026-09-09：项目重构优先

**Researcher Decision**

- 先完成无损知识与工程重构，以长期协作和快速找到正确信息为标准。
- 旧实验保留科研追溯价值；在引用、回归和回滚条件未闭合时不为目录整齐强行迁移。

## 尚无决策

- 最终 PI-JWM 方法是否冻结。
- P6 采用纯候选搜索、学习策略还是混合策略。
- 通信状态不足时是否增加 effective service/residual；外生到达/离开如何建模；planner objective/risk/fallback 的具体定义。
- 是否授权执行 Step 2。旧 `seed=20260832` 不属于当前 active queue。

Unverified：任何未在本文件或项目权威记录中标为 Researcher Decision 的科研取舍。

## 2026-09-24：STEP 5.6A-CONFIG-FREEZE Formal Training Config v1

**Researcher Decision**

- Formal Dataset remains H=2/L=4 with 4416 train and 1104 validation windows; dataset identity, split, normalization and Definition 05 semantics are unchanged.
- Formal training seed=5601, batch_size=8, 552 steps/epoch, AdamW learning_rate=3e-4, constant schedule, weight_decay=0, betas=(0.9,0.999), eps=1e-8, gradient_clip_norm=1.0.
- Budget is max_epochs=10 and max_steps=5520. Stage 1 is steps 0–551 posterior-assisted H1; from global_step=552 the path is prior-dominant. Curriculum starts are H1 at 0, H2 at 1104, H4 at 2208.
- KL target_beta=1.0, warmup_steps=1104, free_bits=0.1 per latent dimension, overshooting OFF. Validation is prior-only over all 1104 windows and H1–H4, every 1104 completed steps, with checkpoint selector `argmin L_Val`.
- Latest checkpoints save every 552 completed steps; best checkpoints save on strict L_Val improvement; patience is 3 full validations. Resume restores model, optimizer, RNG, progress, selector state and curriculum state; sampler order is derived deterministically from formal seed and global step. FP32 is retained and AMP is not introduced.
- This freezes the formal numerical protocol. It does not authorize starting Formal Training; `formal_training=false`, `gpu_training_verified=false`, `locked_test_accessed=false`, `baseline=false`, `planner=false`, and `performance_claim=false` remain required boundaries.

## 2026-09-18 Researcher Decision

Researcher explicitly fixed A_t=(A_t^Route,A_t^Comm,A_t^Comp,A_t^Mob), with A_t^Mob controlling UAV mobility only; vehicle motion remains SUMO external progression. This is a researcher decision, not an engineering inference.

## 2026-09-19：Raw 因果与 acceleration 定义

**Researcher Decision**

- AirFogSim 未来 task schedule 可以保留为 raw/internal metadata，但不得进入当前 `O_t`、History 或 input-side Entity Index。
- AirFogSim acceleration 原样保留为 raw simulator observation / audit 字段；PI-JWM canonical physical acceleration 定义为 `(v_t-v_{t-1})/delta_t`，只使用当前与历史信息。
- 首个有效时间点或缺少历史速度必须使用明确 mask，不得伪造数值；raw 与 canonical 不得混名。
- Step 2.3 只收尾 Raw Contract；后续 Dataset/Tensor 必须另行授权。

## 2026-09-20：Stateful Flow Ledger 决策

**Researcher Decision**

- Flow 是 logical end-to-end business Flow；Hop 是 carrying segment。Flow 不等于 Hop、Route 或 Communication edge。
- 使用 `AirFogSim real state/event → PI-JWM Causal Flow Ledger → Raw logical Flow state`，不修改 AirFogSim 核心传输状态机；Ledger 禁止读取 future action/outcome、target tensor、rollout prediction 或 future task schedule。
- FlowID 固定为 `(TaskID, FlowType, Epoch)`，FlowType 为 `Input/Return/DepData`；不依赖 Hop 或 RouteRevision。
- same-destination reroute 保持 FlowID/Epoch/E2E remaining，RouteRevision 增加。
- logical destination change 创建新 Epoch；旧 Epoch `SUPERSEDED`，新 source=current holder，新 total/remaining=旧 remaining。
- v1 destination change 只允许 clean hop boundary；partial active hop 必须拒绝或延迟，不在本 Step 实现 Planner。
- DepData vocabulary 保留，但当前 AirFogSim runtime instances=0；DAG 不得生成 fake DepData Flow。
- 本 Step 只授权 Ledger + Raw additive Flow contract；Sample/Tensor、Graph Builder、模型、训练、GPU 和 locked_test 均未授权。

## 2026-09-20：STEP 4.2C-C 实施授权

**Researcher Decision**

- 明确授权将 STEP 4.2C-B Raw logical Flow/Carrying state additive 贯穿到 Model-ready Sample 与 CPU Tensor；只允许 index、align、mask、train-only normalization 和 collation。
- 明确禁止 Graph Builder、Physical/Information graph、GNN/Encoder、World Model、Loss、Planner、Training、GPU、locked_test 和 formal Dataset；完成后停止等待审阅。
- 真实 Return multi-hop、same-destination reroute runtime 与 formal Flow capacity 不得因实现便利被默认为已解决；只记录机器证据和边界。

- Step 2.4（研究者明确批准）：Communication Outcome 在 Raw 层拆为 wireless、wired 和按 task 聚合 total；空 map 是已观测无服务，missing 必须是 null 加 mask/reason。该决定只冻结 Raw 语义，不授权 Dataset/Tensor 或模型实现。
- STEP 3.1（研究者明确批准）：model-ready sample 使用因果 History、严格对齐的 Future Action/Target、无 future-object leakage 的 stable input index、独立 target-side future object 表示、四类 action 和显式 presence/feature mask；本决定不授权正式数据集、模型、loss、planner 或训练。
- STEP 3.1R（研究者明确批准）：History 必须为 `[t-H+1,t]`；固定 input index/presence、真实 DAG source、typed target index、relation endpoint 和不可静默 `-1` 的 Action reference 属于修正合同。STEP 3.2 仍未授权。
- STEP 3.1F（研究者明确批准）：History 必须包含 `O_{t-H+1:t}` 和过去 `A_{t-H+1:t-1}/Y_{t-H+1:t-1}`；input index 必须覆盖 History 的因果对象 union，过去 flow/relation/DAG 必须与同一套 index 对齐；future unresolved reference 只做事实审计，不自行决定未来对象表示。STEP 3.2 仍未授权。
- STEP 3.1F-PATCH（研究者明确批准）：Future Action 先按 anchor visibility 判断是否允许引用，再使用同一 History-union `static.input_entity_index` 返回数值 index；`input_index_policy=history_causal_observable_object_union`、`future_action_index_policy=anchor_visibility_then_history_union_input_index`。STEP 3.2 仍未授权。
- STEP 4.1（研究者明确批准）：只冻结定义 03 的 Physical / Information object-field-relation mapping，必须区分真实可用、Raw 有但未暴露、Raw 不足、Derived、禁止归属和待研究者决定；禁止在本 Step 实现 graph builder、GNN、encoder、coupling、World Model、Loss、Planner、训练、GPU 或访问 `locked_test`。发现最小定义缺口时记录并停止在 mapping 层。
## 2026-09-20：STEP 4.3A 实施授权

- 研究者明确授权 Frozen Tensor → typed Physical / Information dual-graph objects + Align/GeoComm。
- Physical topology 参数只允许作为 deterministic development config，必须标记 `development_only=true`、`research_frozen=false`。
- 明确禁止 Encoder/MLP/GRU/message passing/P2A/P2C/GNN/World Model/Loss/Planner/Training/GPU/locked_test/formal Dataset；完成后停止。
# 2026-09-21 STEP 4.4 communication uncertainty decision

- Researcher explicitly selected `Learned Future CSI -> Rule Nominal Rate -> Known Stochastic Outage Event -> Actual Service`.
- Comm z represents CSI/channel uncertainty only. Outage is not a latent/head/input/target leak; both seeded `sample` and marked `expectation` modes are required. `learned_service_residual=false`.
- Wired capacity may be added from causal simulator configuration; active membership/count must first be derived from Flow Carrying and checked equal to `WiredNetworkManager`.
- STEP 4.4 implementation is authorized within the untrained CPU contract only; Loss, optimizer, Training, GPU, Planner, candidate generation, formal Dataset, and `locked_test` remain forbidden.

## 2026-09-22 STEP 5.3E — CSI train-mean bias initialization

**Researcher Decision**

- PI-JWM v1 采用 raw CSI decoder，直接输出 CSI `[dB]` 并进入冻结的规则转移链；不采用 normalized-output decoder 作为 v1 主线。
- CSI decoder 最后一层 bias 使用 frozen train-only CSI normalization mean 初始化，初始化发生在 optimizer 创建前；所有 RB 使用同一 global mean。
- 当前 development provenance 为 `dev_train`；validation、Future Target 和 `locked_test` 不参与 mean fit。normalized-output bridge 仅保留为未来可选 ablation。
# 2026-09-21 STEP 5.0 — Definition 05 researcher decisions

- Prediction distribution: retain stochastic `z^Phy/z^Comm`; Vehicle Motion/CSI observation decoders are deterministic mean heads; no learned observation variance.
- Prediction loss: Motion and CSI each use independently mask-normalized normalized-space MSE; v1 combines them equally.
- Posterior teacher: only corresponding future Vehicle Motion or CSI targets may enter family-specific training-only target encoders; prior never reads Future Target; Future Graph/Flow/Task/DAG/rule-state are forbidden posterior evidence.
- KL: separate analytic diagonal-Gaussian Physical/Communication KL. Physical mask is valid Vehicle slots only; Communication mask is valid+present+wireless+CSI-target-valid. Use configurable warm-up and small free bits; no KL balancing.
- Overshooting: OFF in v1. Total objective is Prediction plus beta-weighted family KL.
- Schedule: posterior-assisted short-horizon warm-up, then prior-dominant recursive curriculum `1→2→4→L`; Stage 2 posterior is KL teacher only; validation is always prior-only.
- Fixed support: unsupported future structure uses component-level mask/exclude/classify with separate counts; valid Motion/CSI supervision remains; whole-window deletion is forbidden.
- Rule states: no Flow/Task/DAG/Lifecycle/Completion learned head or independent loss; natural differentiable rule paths may carry gradients, but discrete rules stay exact.
- Trainable modules: jointly train Dual-Graph Encoder, RSSM dynamics, Prior, Posterior, Target Encoders and Motion/CSI Decoders. STEP 4.3B FROZEN means architecture/interface, not weights.
- Validation/evaluation: checkpoint and early stopping use prior-only horizon-mean `L_Val`; KL is diagnostic. Report Motion/CSI separately with per-horizon raw-unit MAE/RMSE; uncertainty sampling is auxiliary; system metrics remain closed-loop metrics.
- These explicit decisions supersede conflicting observation-NLL/Event/Residual/overshooting clauses in the read-only older Definition 05 note. The private note remains unchanged.

## 2026-09-23 STEP 5.5 — Formal Dataset v1 protocol

**Researcher Decision**

- Formal Dataset v1 使用 `H=2`、`L=4`；每条真实 AirFogSim trajectory 含 96 个连续 transition 和 97 个 Decision，由此每条构造 92 个 windows。
- 接受 60 条完整 trajectory；simulator primary seeds 为 `2026092300..2026092359`，对应 policy seeds 为 `2026092400..2026092459`。失败条目不得部分接受，按后续未用 deterministic seed pair 替换并记录 lineage。
- 固定 `split_seed=20260923`，先按 trajectory deterministic shuffle，再分为 48 train / 12 validation；禁止 window-level random split，禁止根据后续 loss 或统计量改 split。
- Formal Physical topology v1 固定为 `radius_knn(radius=1000m,k=2)`；这是 Dataset v1 protocol choice，不是 topology optimality claim。
- 数据采集采用 causal coverage-oriented 四动作 policy：`A_t=(Route,Comm,Comp,Mobility)`，动作只能依赖当前观察/因果 History，车辆仍由 SUMO 外生推进；policy 不是 Planner，也不作 reward 最优声明。
- Dataset 不包含或物化 `locked_test`。本 Step 仅 CPU Dataset/Interface acceptance；GPU、formal training、baseline、Planner 和 performance claim 均未授权。
# 2026-09-28 — Researcher-specified Planner Objective v1 target

- Target tuple: `(N_DDL, A_DDL, J_Delay, J_Burden, J_Effort)`, minimized lexicographically.
- Throughput is diagnostic/final metric, not a separate weighted objective. Energy and Fairness are outside Planner v1. Priority is inactive (`w_q=1`). Risk is defined but inactive.
- A common support-aware horizon is required. Future-only Return birth is a model-support boundary, not candidate illegality.
- These semantics are a target definition; 6.2A did not implement scoring. `CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING` remains unchanged.
# 2026-09-28 STEP 6.2A-PATCH — Researcher Decisions

- Planner Objective v1 仍按 `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)` 字典序最小化；RouteRevision 不单独进入 effort。`J_Effort` 只用 Comm/Comp/Mob 的当前 anchor 适用项和冻结分母。
- Priority 权重关闭，所有当前任务 `w_q=1`，Priority 字段不是 6.2B 前置条件。Risk 已定义但不启用；Energy、Fairness 不进入 Objective v1。
- 真实闭环业务主吞吐为 End-to-End Useful Throughput；all-hop Network Service Throughput 只作诊断。Planner 中 throughput 不是单独加权项。
- 允许独立 Planner-only causal side-state 携带当前已知、可确定递推的 deadline、Return/control route 元数据；不改 Formal Dataset、训练 Tensor 或 learned 模型。
- `B_Tx` 和 6.2B readiness 是工程审计事实，不是研究者决定。本补丁机器结论为 BLOCKED；路线语义修复方案尚待研究者决定。
