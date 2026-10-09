<!-- STEP6.4J CURRENT -->
## 2026-10-10 STEP 6.4J 当前状态

当前最终冻结为 r23，execution_config_id=103bca16ef006869fe4da1af5f3402481c5a549ca6166bcd5e15ba9d6906fbfc。4 次真实 env.step、2 条 episode、每条 2 个 fresh root，History/真实 post-step outcome 独立审计 PASS；GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行。协议部署 SHA verifier PASS，Git 同步后 READY_FOR_GPU_LAUNCH=true。

<!-- STEP6.4I CURRENT -->
## 2026-10-09 STEP 6.4I — Pure-search正式闭环CPU准备

STEP_6_4I=PASS（仅CPU正式闭环准备）；SEARCH_METHOD=S-CEM，K=4/rho=0.2；FINAL_CLOSED_LOOP_BUDGET=512，PLANNER_V1_CLOSED_LOOP_B_WM=512，H=4。研究者本轮明确接受6.4H所见Effort质量损失换取计算节约；B1024保留参考，不声称等价/最优，不冻结hybrid预算。Route=EXPLICIT_NOOP_ONLY；fallback=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1（一次A失败即C，无重试）；同步暂停仿真、非实时。

独立CPU真实S-CEM机制：固定TRAIN anchor0003，engineering-only B64/batch1，两次winner第一步真实执行0.6→0.7→0.8s；每次实际64 unique transitions、13个不同可评分H4。第二root由新真实观测/History/后验重新构建，未复用预测root，模型与TRAIN normalization不变。固定TRAIN anchor0041注入NO_SCOREABLE_H4，canonical fallback A经同桥真实执行4.5→4.6s一次；不是搜索性能结论。74项focused CPU tests通过，覆盖空域、缺字段、bridge/setter/step异常、部分写入、重复决策保护、Return支持与Future Target poison。

CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_PROTOCOL_REVIEW；真实CPU链READY，通用episode组件已测试，live CUDA provider=IMPLEMENTED_NOT_GPU_VERIFIED。正式episode来源/数量/seed/长度、指标分母、timeout及baseline仍待研究者批准；GPU_LAUNCH_AUTHORIZED=false，READY_FOR_GPU_LAUNCH=false。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；Stage B=NOT_STARTED/DEFERRED；GPU=NOT_USED；locked_test=false。

证据：docs/implementation_records/STEP_06_4I_PURE_SEARCH_CLOSED_LOOP_PREPARATION.md；docs/contracts/PIJWM_STEP_06_4I_FORMAL_CLOSED_LOOP_PROTOCOL_DRAFT_V1.md；code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2/15_acceptance.json。唯一下一动作：研究者审阅Protocol Draft，批准最小GPU pilot与首个live CUDA资格检查。本轮停止。以下较早内容均为历史时点，旧pending不覆盖本轮明确接受预算决定。

新执行层EpisodeController与run_episode组合旧live桥；LiveSCEMPlanner每solve重建posterior/domain，fresh real observation传到下一轮。WM、搜索器、Objective、Domain科学源码保持基线不变。
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

共享`step6_4f_comm_eligibility_v1`连接tensor grammar和raw-live bridge；模型状态/Flow Ledger不删除stale Flow。CandidateDomain科学实现只加Task资格交集；当前执行机制READY，不等于新域搜索科学证据PASS。

证据：`docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/09_final_acceptance_receipt.json`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/07_formal_search_evidence_reuse_assessment.json`。历史6.4E BLOCKED及旧搜索accepted状态仅在旧定义内保留。

以下为历史状态，旧PASS不代表新资格定义下已经重新验证。

---

## 2026-10-04 STEP 6.4E 预算决定与真实执行阻塞（当前）

STEP_6_4E=BLOCKED，CLOSED_LOOP_PRE_FORMAL_READINESS=BLOCKED。研究者正式冻结 PLANNER_V1_CLOSED_LOOP_B_WM=512，仅 MH-CEM K4/rho0.1 pure-search；B1024保留高预算参考，不声称等价/最优，6.4D后两项目标质量损失仍记录。统一fallback接口CPU合同就绪，33 focused CPU tests通过。真实TRAIN固定anchor0041、t=4.5s的非空Comm Task_1/RB0被执行前安全门拒绝：Domain仍绑定已failed的Task_1/Task_6，无本轮setter或env.step，第二场景未启动，无换动作/样本或科学失败重试。候选A可确定性准备合法当前动作但本轮真实执行未取得；候选B缺独立current behavior offer provider；C无动作终止及空域/缺字段/bridge/setter负路径通过。当前Flow支持与真实task执行条件不一致，Status=Awaiting Researcher Decision；不改Domain/Objective/Return。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN；原Stage B=NOT_STARTED、DEFERRED / NOT_REQUIRED_FOR_CURRENT_MAINLINE；GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并决定一致性修复与复验范围；不自动开跑。

证据：`code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/10_pre_formal_readiness_receipt.json`；`docs/implementation_records/STEP_06_4E_BUDGET_COMM_FALLBACK_CLOSURE.md`。以下为历史状态，较早预算待决定或机制PASS不覆盖本轮阻塞。

---

## 2026-10-04 STEP 6.4D 验收边界

新增6.4D orchestration、只读SHA备份、标准库独立统计审计；原MH/WM/Domain/Objective算法和checkpoint全部字节身份不变。 CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED；locked_test=false。 SEARCH_METHOD仍MH-CEM纯搜索骨架，非最终hybrid。

## 2026-10-03 STEP 6.4C 当前边界

新增独立calibration orchestration、静态cohort selector及raw auditor；复用原run_step6_3d_one_cpu_solve_v1.run CUDA16计算链，不修改WM/CandidateDomain/Grammar/Objective/MH数学。架构仍pure-search backbone，closed-loop机制PASS与正式性能NOT_STARTED分开。 CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。

## 2026-10-03 STEP 6.4B 最小闭环真实反馈验收（当前）

STEP_6_4B=PASS，TWO_CYCLE_REAL_FEEDBACK_SMOKE=PASS，CLOSED_LOOP_MECHANISM_READINESS=PASS。固定TRAIN fixture、MH-CEM K4/rho0.1、CPU FP32 batch1、B64：仅执行首轮winner第一动作，仿真0.6→0.7秒，真实新观测重建state/graph/current posterior并完成第二次规划；下一轮root不是上一轮预测。两轮各64独特转移（合计128）；另保留一次动作执行前回执异常的64次尝试，实际总计192。live tensor/deadline、旧搜索不变性、防未来泄漏及参数/归一化不变全部PASS。SEARCH_METHOD=MH-CEM仍仅pure-search backbone；READY_FOR_BUDGET_CALIBRATION=true只是机制就绪，不是运行授权。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN；future Return-birth fixed-support limitation保持。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作是研究者审阅并单独授权预算校准协议，不自动运行。

新增step6_4b_live_bridge_v1实时白名单history-only输入和当前真实Task deadline；run_step6_4b_two_cycle_smoke_v1负责暂停、fresh posterior、MH-CEM、first-action原生setter、一次env.step、fresh feedback再规划。预测state/graph/latent不进入下一轮root。

记录：`docs/implementation_records/STEP_06_4B_MINIMAL_CLOSED_LOOP_TWO_CYCLE_SMOKE.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/13_final_acceptance.json`；合同：`docs/contracts/PIJWM_STEP_06_4B_LIVE_CLOSED_LOOP_SMOKE_V1.md`。以下为历史记录，其中“当前”仅表示记录时点。

---

## 2026-10-03 STEP 6.4A 闭环前就绪审计（当前）

STEP 6.4A审计完成（审计证据PASS），CLOSED_LOOP_READINESS=BLOCKED。SEARCH_METHOD=MH-CEM仍只为pure-search backbone，非最终hybrid。局部链路7 IMPLEMENTED/3 PARTIAL/5 MISSING；缺live history-only输入/belief/sidecar、winner动作序列/首动作及真实ID执行桥、研究者fallback和latency决定、两轮真实反馈MH证据。MH B1024搜索median90.506/P90 204.916/P95 284.152/max421.701秒；仿真slot0.1秒不是墙钟deadline，DECISION_LATENCY_REQUIREMENT=UNRESOLVED，尚无在线执行证据。Stage B用途由研究者冻结为online-budget trade-off，不能重选算法；A约30.46–41.85h、B NEW PROPOSAL约10.26–14.06h、C NEW PROPOSAL约1.54–2.11h，仅成本情景。4090_MIGRATION_NOT_YET_JUSTIFIED。Stage B=NOT_STARTED，GPU=NOT_USED，locked_test=false。唯一下一动作研究者确定fallback及decision latency模式，再授权最小接口集成；不自动运行实验。

现有compile_candidate编译模型action tensor，不是simulator命令；prepare_anchor能建立posterior但缺live输入桥。shift_warm_start/learned wrappers不是MH集成。

记录：`docs/implementation_records/STEP_06_4A_CLOSED_LOOP_READINESS_STAGE_B_PURPOSE_FREEZE.md`；机器审计：`code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003/06_go_no_go_receipt.json`。下方为历史状态，不能用Stage A PASS替代闭环readiness。

---

## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

架构没有修改；选择既有 MH-CEM 路径。CandidateDomain、Grammar、World Model、Objective、预算/缓存、参数和checkpoint均保持冻结16项source SHA。learned proposal、Route NOOP支持边界、warm start/fallback尚需单独研究。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-01 STEP 6.3D BLOCKER CLOSURE

正式runner将历史TRAIN selection identity与新Validation execution identity分开验证；只有orchestration/统计修改。primary仅1024、完成return，diagnostic独立256/512，不改变主选择；原候选/rollout/目标/搜索架构不变。 依据：`docs/contracts/PIJWM_STEP_06_3D_VALIDATION_STAGES_AND_CEM_UPDATE_SEMANTICS_V1.md`、`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md`。

# 当前代码架构与新定义差异

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH 执行路径

串行与批量一步转移共用 6.1 的动作适配、冻结 `model.one_step`、确定性规则和当前/预测状态。批量路径复用 `_batch_tree`、`_stack_actions`、按候选拆回结果，并交给现有 `TransitionBudgetAccountant.evaluate_batch` 按每条 unique candidate-step 收费；缓存和 causal parent fingerprint 不变。真实 TRAIN fixture 上 batch 1/4/8/16 的动作、状态、图、模型 trace、Grammar/H_sup 和 Objective 排序与串行等价。搜索方法、候选域、Objective、模型及 checkpoint 未改变；正式方法矩阵仍暂停。

## 2026-09-29 STEP 6.3D CPU 搜索架构（进行中）

`step6_3d_structured_proposal_v1.py` 在唯一 `CandidateDomain` 上实现共享的联合结构→任务数→任务子集→通信行分配→RB 起点层级。`step6_3d_fixed_budget_search_v1.py` 让 HRS/S-CEM/MH-CEM 共享 `SearchNode`、一步 rollout 和 `TransitionBudgetAccountant`；只把完整可评分 H4 候选交给既有 6.2B 五项字典序 Objective。HRS 不更新，S-CEM 只更新粗层，MH-CEM 更新全部层。TRAIN/Validation 方法比较尚未完成，架构可运行不等于方法已选或闭环已建。

## 2026-09-29 STEP 6.3B/6.3C-PATCH 通信任务选择

研究者已批准当前可唯一绑定到**已有无线 Flow** 的 Task 集合 `E_comm(t)` 中按 TRAIN 通信结构的已见任务数量选择子集。`step6_3b_candidate_grammar_v1.py` 负责逐步绑定和准入；`step6_3c_candidate_domain_v1.py` 用相同条件计数并惰性枚举。未选任务只是不在当前时隙调度。依赖同决策 Route 才能产生 Flow 的历史 Comm 行由审计层排除，并与原始动作分开保存；Route 仍显式空。其余 Comp/Mob/联合结构/时间标签、World Model、`B_WM` 和 Objective 不变。

## 2026-09-29 STEP 6.3C CandidateDomain / rollout boundary

`step6_3c_candidate_domain_v1.py` is the only search-independent candidate-domain entry. It derives current eligibility, TRAIN-observed joint structures and concrete causal bindings lazily; it never materializes the full domain. `step6_3c_search_protocol_v1.py` supplies the shared prefix node and one-transition `B_WM` accounting/cache. Interleaved search rebuilds the domain after every frozen-model transition. Empty H1 is `NO_FORMAL_CANDIDATE`; later empty branches are `GRAMMAR_DEAD_END`; no fallback or horizon backoff is introduced.

## 2026-09-29 STEP 6.3B Candidate Grammar boundary

The Planner v1 candidate layer now has a search-independent structured grammar and Formal TRAIN support admission before the existing candidate adapter, trained World Model rollout and objective scorer. It binds Comm/Comp/Mob from each current or predicted state; Route stays explicit no-op. No optimizer, new World Model parameter or candidate ranking path was added.

## 2026-09-28 STEP 6.2B-PATCH Planner v1 Route 边界

Planner v1 的 6.0C 准入层只接受空 Route family；Route schema、11 tensor adapter、4.4 learned Route encoder 和多跳确定性规则都保留。合法候选实际变化来自 Comm/Comp/Mob。此策略是研究者对当前 learned-support 的限制，不是删除 Route 架构。STEP 6.2B scorer/字典序比较器通过 CPU 合同验收；候选生成方法和闭环仍未建立。下文 6.2B 阻塞段是 Patch 前历史状态。

## 2026-09-28 STEP 6.2B scorer 层（历史验收阻塞）

`CandidateRolloutTrace + PlannerObjectiveCausalSideState + CandidateActionSequence + anchor state → CandidateObjectiveScore` 是独立 CPU scorer 层，不进入 Encoder/RSSM learned tensors；先求共用 `H_eff`，再逐 Task/Horizon 汇总五项目标，最后严格字典序。当前只证明 fixed-support 机制。pending Route admission 与 same-path Host 规则冲突使 6.2B 完整验收阻塞；不改变原模型结构、训练 checkpoint 或 Planner v1 single-hop 决定。

## 2026-09-28 STEP 6.2A-CLOSURE Route boundary

4.4 repaired multi-hop deterministic rule and rule-side route metadata remain code capabilities. Planner v1 admission applies the separate `FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1` domain: one direct destination node per Route action. The domain gate does not alter the 11 learned action tensors or remove the multi-hop rule.

## STEP 6.1 冻结模型候选推演层

`step6_1_trained_candidate_rollout_v1.py` 在 6.0A/C Candidate/Domain 与冻结 5.6C Encoder/RSSM 之间建立 CPU 推演层：每个 anchor 只算一次当前 Encoder/posterior，候选克隆同一 state/graph/latent，每步在预测 state 上经本模块的 `compile_candidate_step` 调用未改的正式 `build_action`，再调用 4.4 `one_step` 递归更新 state/graph/latent。无目标函数、候选选择、闭环执行；历史 `formal_candidate_rollout_planner_v1.py` 不在当前路径。

## STEP 6.0C Planner v1 动作域层

6.0A 通用 Candidate 合同 → 6.0C 当前 Raw CPU 容量/UAV heading-elevation 控制侧状态 → 静态预算与六档 Mobility 验证 → 原 `build_action` 11 tensor 编译。6.0C 只更新 control-command side-state，不做物理状态或 World Model rollout。静态预算是研究者 Planner 操作规则，不是 AirFogSim 原生动态可用量。形式见 `docs/contracts/PIJWM_STEP_06_0C_PLANNER_ACTION_DOMAIN_V1.md`。

## STEP 6.0B 约束来源边界

Simulator Fact：AirFogSim CPU callback 直接给每个 Task 分配率，`Task.compute` 按率×时隙执行，无原生节点总量裁剪；静态 `FogProfile.cpu` 不等于动态可用 CPU。UAV setter/更新无数值硬边界。Implementation Fact：6.0A 的两项 UNKNOWN 保持，未改候选、World Model 或训练架构；世界模型的位置公式与仿真器一致，加速度使用已标注的 PI-JWM canonical 符号，与 simulator raw 符号相反。

## STEP 6.0A 静态候选生成层（2026-09-26）

当前因果支持与约束 → 四动作高层 CandidateActionStep → 长度 1–4 的 CandidateActionSequence → Search/Learned/Hybrid 接口 → 去重 CandidatePool → 包装正式训练 `build_action` 的张量编译器。到此为止。没有候选 World Model rollout、评价、选择或执行；旧 P6 `formal_candidate_rollout_planner_v1.py` 使用过期 `task_action*`，仅是历史原型。见新合同及 STEP 6.0A 实施记录。

> 下列 1–5 节描述被 Step 1 审计的现有旧协议实现，不表示它符合 2026-09-18 的新 `00–06` 目标。总体差异见 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/STEP_01_AUDIT.md`。

## 总体结构

```text
历史实体状态 + 静态拓扑 + 未来动作
→ FormalDualGraphWorldModel（确定性双图底座）
→ node / physical_edge / flow / task 的实体级 h,z
→ prior 逐步 rollout（验证与部署）
→ 多任务预测头 + 可选确定性规则递推
→ 20 步联合状态与事件预测
```

Source of truth：以下实现事实来自所列源码与冻结协议；运行采用情况还需核对具体实验 `config.json` 和 checkpoint manifest。

## 1. 确定性双图底座

- 文件：`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`
- 配置：`FormalWorldModelConfig`
- 模型：`FormalDualGraphWorldModel`
- 主入口：`FormalDualGraphWorldModel.forward()`
- latent 追踪：`FormalDualGraphWorldModel.forward_with_latent_trace()`
- 实体状态宽度：node=7、physical_edge=5、flow=5、task=8，定义于 `COMPONENT_FEATURES`。
- 图操作：`code/src/pi_jwm/formal_graph_ops_v1.py`，包含物理图、信息图、DAG 与跨图耦合。
- 规则层：`code/src/pi_jwm/formal_deterministic_rule_layer_v1.py::DeterministicRuleLayer.forward()`。

## 2. 实体对齐 RSSM

- 文件：`code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py`
- 配置：`FormalEntityAlignedRSSMConfig`
- 模型：`FormalEntityAlignedRSSMWorldModel`
- 主入口：`FormalEntityAlignedRSSMWorldModel.forward()`
- 当前冻结维度：hidden=32、stochastic=16、history=8、horizon=20、overshooting distance=5。
- entity latent：node、physical_edge、flow、task 各有独立确定性状态 `h` 和随机状态 `z`；agent 是双图底座中的确定性中间 latent，没有独立随机状态。
- `priors` 只依赖已知历史、动作和递推状态；`posteriors` 在训练期结合目标观测。
- prior 与 teacher 各有状态、存在性、DAG、链路、生命周期和能耗输出头。
- 验证/部署路径固定为 prior-only；训练固定使用 posterior teacher。

## 3. 确定性规则层

`DeterministicRuleLayer` 在物理单位中处理动作端点、RB、flow 守恒、任务阶段、CPU 服务、DAG release 等确定性更新，再返回归一化空间反馈下一步。它不应被描述为学习模型自动发现的规律。

## 4. Loss

- 文件：`code/src/pi_jwm/formal_world_model_loss_v1.py`
- 入口：`formal_world_model_loss()`
- 配置结构：`FormalLossWeights`
- 组成：masked Gaussian NLL/MAE、presence 与稀疏事件 BCE、生命周期、DAG、吞吐量/RB/时延等任务项，以及 RSSM KL、teacher reconstruction 和 overshooting。
- 正式权重以冻结协议为准：KL=0.1、teacher reconstruction=0.5、overshooting=0.1、KL balance=0.8。

## 5. 训练结构

- 正式包装入口：`code/scripts/run_formal_p4_entity_rssm_gpu_v1.py::main()`。
- 通用训练器：`code/scripts/run_formal_dual_graph_gpu_train_v1.py::run_gpu_training()` / `run_formal_training()`。
- 两阶段：deterministic base 训练 20 epochs；随后冻结 base，训练 RSSM 最多 40 epochs。
- checkpoint：逐 epoch 保存，使用 `p4_gate_aware_v1` 字典序选择；RSSM 最少 20 epochs，patience=10。
- 正式配置事实：`code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json`。

## 6. 新定义下的实现状态

- STEP 4.1 已冻结目标 mapping：Physical 只包含实体空间/运动与空间关系；Information Nodes 为 Agent/Task，Relations 为 Comm、Src/Host/Exec/Ret、Flow、DAG。
- 现有 `physical_edge` 混合空间、CSI、rate、任务数和 RB，不符合目标严格双图划分，禁止按原语义继续使用。
- STEP 4.2C-C（含 PATCH）已冻结 Flow Sample/Tensor additive extension：C-B Raw logical Flow/Carrying rows 进入独立 `logical_flow` History-union/target namespace；Flow 与 carrying 分离，History/target Logical/Carrying 四组语义逐字段核对；target carrying 明确是 future ground-truth/deterministic-transition state，不是 learned prediction head。该层在其完成时仍不是 Graph Builder；后续 STEP 4.3A 已完成 current typed graph materialization。
- STEP 4.3A 已冻结 current typed dual-graph representation：Physical nodes/relations 与 Agent/Task/Comm/Task-Agent/Flow/DAG 严格分离，Carrying 保持 side state，Align/GeoComm 仅为 structural cross-domain references。builder 不包含编码、传播、聚合或 latent。
- STEP 4.3B 已冻结 Definition 03 encoder：完整 History 对 Physical/Agent/Task/Flow 做独立 presence-gated GRU，current graph 做五类独立有向 relation update、分族 masked mean、独立 node update、P2A Align 与 wireless-only P2C GeoComm。Logical/Carrying 独立编码后融合为唯一 Flow relation latent；输出只到 aligned `Z_t^{PI,L_g}`，不是 `xi_t^Lat` 或 dynamics。
- 独立 Agent/Communication/Task-Agent/current stateful Flow graph 尚未实现；四类动作 tensor 已冻结但尚未接入新图或模型。
- 旧 entity RSSM 对 node/physical_edge/flow/task 均维护随机状态，与新定义的未知动态边界不同。
- base 的逐步规则和 RSSM 修正尚未组成“预测→规则→重构图→下一步”的完整闭环。
- 因此现有模型、tensor、checkpoint 和结果为 Historical / Archived；新定义模型尚未实现。

STEP 4.1 source of truth：`docs/contracts_PIJWM_PI_GRAPH_OBJECT_FIELD_RELATION_MAPPING_V1.md` 与 `code/artifacts/protocols/pi_jwm_step4_1_pi_graph_mapping_v1_20260919/`。其中 `DO_NOT_IMPLEMENT_GRAPH_BUILDER_IN_STEP_4.1` 是该 Step 当时的停止门；后续 STEP 4.3A 已在数据合同闭合后完成 builder。

## 7. 当前不属于新定义正式架构的内容

- `code/src/pi_jwm/formal_candidate_rollout_planner_v1.py` 是 P6 CPU 原型，不是已开放的正式规划器。
- `formal_complete_rssm_world_model_v1.py` 是历史 global RSSM，不是当前候选模型。
- v11 selector/ranking 是历史诊断，不是当前 PI-JWM 主线。

Unverified：任何只由旧 PPT、文件名或历史聊天提出但没有当前代码/config/experiment 支持的架构声明。
# STEP 4.4 Structured RSSM（2026-09-21）

STEP 4.4-PATCH3 已 COMPLETE / FROZEN，闭合未训练 CPU 机制的三个剩余语义。五类 aligned deterministic h 中仅 Vehicle Physical/Comm 有 z；只学习 vehicle motion 与 CSI。Wireless outage 是显式 known stochastic event，wired/Flow/Task/CPU/UAV 按规则推进，每步由 predicted state 重构图并反馈。5.1D 已补充 CPU Loss/KL/Metric integration；该实现仍不是精度、校准、规划或性能证据，Training Loop 尚未开始。

STEP 5.0 已冻结训练架构边界：Motion/CSI 是唯一直接 prediction supervision；对应 training-only Target Encoder 只允许读本 family future target；Phy/Comm 分别做 analytic KL；v1 overshooting OFF。Encoder、RSSM、Prior、Posterior、Target Encoder、Decoder 后续 joint train，规则无 optimizer 参数。此处是 target contract，不是已实现训练图。

STEP 5.1A-PATCH 已把 training target 与 STEP 4.4 对齐：Vehicle Motion 为逐步 `p_(t+k)-p_(t+k-1)`，并严格使用 current physical input slots；CSI 严格使用 current communication relation slots。Future GT 仍只在 target/training supervision，未进入 prior 或 current state。该 Patch 不实现 Posterior/Loss/Metric/Training。

PATCH3 保持 Return 只复用 current-support typed Flow identity：`task_index + flow_type_index=Return`。冻结输入未暴露 `Task.return_size`，所以 real adapter 用 `task_return_requirement_known=false` 表示 unknown；unknown 不等于 no-return，computation finished 后输出 unresolved/blocking side-state。known-required/no-slot 则单独输出 `return_birth_required`。v1 不生成 future-only Return Flow。
# 2026-09-28 STEP 6.2A objective boundary

The target Planner objective is the researcher-specified lexicographic tuple `(N_DDL, A_DDL, J_Delay, J_Burden, J_Effort)`. STEP 6.2A only audits source semantics and records the protocol interface. It does not add an objective scorer to the architecture. The rollout contract remains distinct from selection: `MPC_OBJECTIVE=NOT_STARTED`, and no winner selection or closed-loop planner is implemented by this Step.
# 2026-09-28 Planner Objective side-state boundary

新增 Planner-only Task/Route 因果 side-state，独立于 Encoder/RSSM；只接当前 Raw、已对齐 deadline sidecar、候选 Route 控制及预测状态，不读 Future Target。4.2C-C route 数组不含 holder，而 4.4 跨跳规则按含 holder 的数组推进，构成当前 blocker。Route action 仅改 endpoints/revision，不改完整数组；side-state 可检测分歧，不能代替模型修复。
2026-09-30 执行路径更新：正式 6.3D matrix runner 可在 CPU 或 RTX 3080 Ti CUDA FP32 上运行，GPU 复用同一 `rollout_one_step_batch`/`solve_fixed_budget`/`TransitionBudgetAccountant`；CUDA 搜索缓存和前缀使用主机内存保存精确张量，batch16 配置冻结。科研搜索定义未变；证据见 3080 Ti Step 实施记录。
