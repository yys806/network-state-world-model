## 2026-10-10 STEP 6.4J — r29 日志 schema 与独立落盘验收闭环

已统一正式日志字段为 `search_attempt`，修复 SEARCH_INTENT 与全局 attempts 的字段错配。独立验收器从实际落盘 `pilot_attempt.json`、已启动 episode 的 `journal.jsonl`、`episode_receipt.json` 和 `decision_*.json` 重建结果；未启动 episode允许没有日志，已启动 episode缺失/损坏/重复/预算/WM transition计数不一致则 BLOCKED。新增独立 CLI `code/scripts/audit_step6_4j_formal_v1.py`，22项专项测试全部通过，覆盖两 episode 16 步 COMPLETED/0、首轮失败 BLOCKED/1、资格后资源停止 PARTIAL/2、fallback C PARTIAL/2、缺失/重复/损坏日志、预算不匹配和不完整无合法原因。r25真实 CPU AirFogSim证据保留并通过：2 episode、4 次真实 env.step。r29 execution_config_id=`f2b71e6b06b372a21c891a432680663259ef2f5108b0704eff844433b39f9bdc`。
## 2026-10-10 STEP 6.4J — 最终日志与独立验收闭环修复

r28 统一正式日志 schema：所有 SEARCH_INTENT、WM_TRANSITION_ATTEMPT、FINAL_DECISION、decision receipt 和全局 search_attempts 使用 `search_attempt`。`audit_formal_result()` 现在只从落盘的 `pilot_attempt.json`、已启动 episode 的 `journal.jsonl`、`episode_receipt.json` 和 `decision_*.json` 重建结果；未启动 episode可没有日志，已启动 episode缺日志/损坏/重复/预算或序列不一致则 BLOCKED。新增 CLI 独立审计 `code/scripts/audit_step6_4j_formal_v1.py`，13项落盘/退出码测试覆盖 COMPLETED、首轮资格失败、合法 PARTIAL、fallback C、缺失/重复/损坏日志、预算不匹配和未完成无停止原因。r25真实 AirFogSim CPU证据保留：2 episode、4次真实 env.step、独立工程审计 PASS。GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行。
## 2026-10-10 STEP 6.4J — GPU Pilot 停止门修复与 r27 冻结

正式 GPU 分支采用全局一次性停止状态机，首个 B512 决策须完成 CUDA/FP32/batch16/执行身份、512 次独立 WM 转移、scorer 有限性、winner 或冻结 fallback A、真实 env.step 与 fresh observation 验收；失败立即 BLOCKED 且不启动第二 episode。每次真实 transition 调用前先同步 journal 与 attempted-search 记录，失败尝试计入 16 次总预算。600 秒单次规划使用硬中断，不返回 early-best；10200 秒资源门、预算溢出、重复动作/root、NaN/Inf 和异常均受停止门约束。最终结果从两条 episode 的决策 receipt、journal 与尝试记录独立重建：两条各 8 步为 COMPLETED/0；首个资格 PASS 后合法资源终止或协议允许的 fallback C 为 PARTIAL/2；资格/身份/预算/执行异常为 BLOCKED/1。r27 execution_config_id=`c019c075e35f10560f2a059ac8abf94809c8ef5d18b2f2be82585dec76ba3cf2`。r25 同核心真实 CPU 回归为 4 次 env.step、双 episode/fresh roots PASS；GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行。
## 2026-10-10 STEP 6.4J — GPU Pilot 停止门修复与 r26 冻结

正式 GPU 分支已改为全局一次性停止状态机：首个 B512 CUDA 决策必须完成身份、FP32/batch16、512 次独立 WM 转移、有限 scorer、winner/fallback A、真实 env.step 和 fresh observation；任一失败立即 BLOCKED，第二条 episode 不启动。搜索尝试在 Planner 调用前落盘并计入预算；600 秒单次规划、10200 秒实例资源门、重复 receipt/root、NaN/Inf、反馈/History 不一致均失败停止。最终状态由原始 receipt、journal、attempted-search 和 env.step 记录独立重建：COMPLETED=两条各8步；PARTIAL=资格通过后合法资源停止或冻结 fallback C；BLOCKED=资格/身份/预算/执行异常。r26 protocol 已重新冻结，execution_config_id=`bba80a237795e1fd6e03a6c0047005ae182d60b1178c3468c4f235bc5aced922`。r25 同一生产核心 CPU 回归仍为两 episode、4 次真实 env.step、独立审计 PASS；GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行。
## 2026-10-10 STEP 6.4J — r23 最终 CPU 一致性与部署验收

r23 已重新冻结源码与协议：`execution_config_id=103bca16ef006869fe4da1af5f3402481c5a549ca6166bcd5e15ba9d6906fbfc`。真实 AirFogSim engineering run 两条冻结 episode 各连续 2 次 `env.step`，合计 4 次；每条 2 个不同 fresh root；4 份 decision receipt 均 EXECUTED，History 动作逐字段与实际 indexed action 对齐，真实 post-step outcome 与 capture 对齐，独立审计 PASS。生产轨迹实际有 2 次合法 Comp 写回；Comm 在该冻结样本中为空，非空 Comm+Comp 的实体/任务/RB/CPU 映射由同一生产 helper 的 focused contract test 单独 PASS，未伪造运行数据。r23 protocol/manifest 已准备强制纳入 Git，部署 verifier 对协议、原始输入、checkpoint、normalization 和源码 SHA 全部 PASS。GPU、正式 Pilot、baseline、Hybrid、locked_test 未运行；完成提交同步后 `READY_FOR_GPU_LAUNCH=true`。

## 2026-10-09 STEP 6.4J — None 根因与 r19 证据

- 调用链：`EpisodeController.cycle → LiveSCEMPlanner.plan → history_tensor → build_live_sample → amend_raw_graph_inputs → _wired_pairs → enumerate(None)`。
- 另外两个真实阻塞：History 动作必须从 `uav_index` 还原 `uav_id`；fresh capture 必须读取 task object 的 `getReturnedSize()` 生成 `required_returned_size`。
- r19：4 次真实 env.step；两 episode 各2步；fresh roots 为 `69bd…764 → b200…61`、`5873…2a → d52b…420`；`ENGINEERING_PASS`。最终 r20 protocol source SHA/execution_config_id 已重冻并独立 audit PASS，READY_FOR_GPU_LAUNCH=true；GPU仍未启动。

## 2026-10-09 STEP 6.4J — S-CEM B512 最小真实 GPU Pilot 协议冻结

研究者已授权2条dev_validation源轨迹、每条最多8个决策、search seed6311，最多16次B512；前2轨迹按trajectory_id UTF8 SHA256排序，初始frame取该轨迹静态manifest最早frame，不依据搜索结果。CPU协议与manifest已冻结，Pilot执行身份见 `code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009/00_protocol.json`，execution_config_id=`31667b891be356017220e205e19a33b1e3aeaf7df8fcc8f11af147061303ddf8`。

配置：S-CEM K4/rho0.2、B512、H4、batch16、RTX3080Ti CUDA FP32、Route NOOP、fallback A一次失败即C、同步暂停仿真。CPU manifest/SHA/测试通过；GPU尚未启动。首个真实CUDA决策是资格门并计入16次，不额外搜索；单次>600秒或实例累计达到10200秒停止新规划，身份/源码/checkpoint/预算/scorer/NaN/重复任务异常立即停Pilot。locked_test=false，正式闭环性能仍NOT_STARTED。下一步是远端精确提交、硬件与依赖只读检查，通过后启动首个批准episode。
<!-- END STEP6.4J CURRENT -->

# STEP 6.4J — 当前授权计划

当前门：CPU协议冻结。研究者已批准2条dev_validation轨迹，SHA256(trajectory_id UTF8)排序前2；每轨迹静态manifest最早frame，seed6311，最多8步/16次S-CEM K4rho.2 B512H4。首个真实CUDA决策计入Pilot资格门，不额外搜索。

1. 核验静态manifest及causal prefix/初始Task cohort SHA；CPU只重放至初始观测，不执行任何新搜索或Planner动作。
2. 测试先行补最小真实环境组合、单次规划600s与安全边界累计实例上限10800s（10200s不再发起新规划）、独立source/input/config身份、逐决策原子日志/指标和whole-pilot失败停止。
3. CPU tests/compileall/index/Context/diff PASS后protocol-freeze commit+push。
4. 只读探测已授权RTX3080Ti；不可达则READY_FOR_GPU_LAUNCH=true后STOP。研究者负责云开关。可达则remote exactcommit/clean、CUDA FP32batch16、显存与磁盘20GiB/依赖/无旧runner/备份门PASS后单runner启动。
5. 首步资格PASS继续最多剩余15搜索；异常立即停，合法C保留分组，无episode内resume/动作重试。完成原始结果独立核验ZIP/SHA/D:备份、Context/Git收口后停止。

禁止24×64正式实验、StageB/baseline/ablation/hybrid/locked_test、科学源码修改。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；locked_test=false。下一交付：可启动的冻结Pilot协议与CPU preflight evidence。

---
## STEP 6.4I CPU准备完成（2026-10-09）

STEP_6_4I=PASS（仅CPU正式闭环准备）；SEARCH_METHOD=S-CEM，K=4/rho=0.2；FINAL_CLOSED_LOOP_BUDGET=512，PLANNER_V1_CLOSED_LOOP_B_WM=512，H=4。研究者本轮明确接受6.4H所见Effort质量损失换取计算节约；B1024保留参考，不声称等价/最优，不冻结hybrid预算。Route=EXPLICIT_NOOP_ONLY；fallback=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1（一次A失败即C，无重试）；同步暂停仿真、非实时。

独立CPU真实S-CEM机制：固定TRAIN anchor0003，engineering-only B64/batch1，两次winner第一步真实执行0.6→0.7→0.8s；每次实际64 unique transitions、13个不同可评分H4。第二root由新真实观测/History/后验重新构建，未复用预测root，模型与TRAIN normalization不变。固定TRAIN anchor0041注入NO_SCOREABLE_H4，canonical fallback A经同桥真实执行4.5→4.6s一次；不是搜索性能结论。74项focused CPU tests通过，覆盖空域、缺字段、bridge/setter/step异常、部分写入、重复决策保护、Return支持与Future Target poison。

CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_PROTOCOL_REVIEW；真实CPU链READY，通用episode组件已测试，live CUDA provider=IMPLEMENTED_NOT_GPU_VERIFIED。正式episode来源/数量/seed/长度、指标分母、timeout及baseline仍待研究者批准；GPU_LAUNCH_AUTHORIZED=false，READY_FOR_GPU_LAUNCH=false。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；Stage B=NOT_STARTED/DEFERRED；GPU=NOT_USED；locked_test=false。

证据：docs/implementation_records/STEP_06_4I_PURE_SEARCH_CLOSED_LOOP_PREPARATION.md；docs/contracts/PIJWM_STEP_06_4I_FORMAL_CLOSED_LOOP_PROTOCOL_DRAFT_V1.md；code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2/15_acceptance.json。唯一下一动作：研究者审阅Protocol Draft，批准最小GPU pilot与首个live CUDA资格检查。本轮停止。以下较早内容均为历史时点，旧pending不覆盖本轮明确接受预算决定。

验证：74 focused tests/22.416s；独立CPU证据核验PASS。预算已明确接受，正式protocol待审阅；无GPU启动。

以下为过程历史：

# STEP 6.4I 当前执行计划（CPU-only）

当前门：pure-search正式闭环CPU准备。研究者已接受S-CEM K4/rho.2 B512 H4与冻结fallback、暂停仿真。GPU/正式多episode/baseline/locked test禁止。

1. 审计既有6.4B/6.4F源码与6.4H接受证据，固定TRAIN anchor0003、seed6301、CPU B64/batch1工程smoke及新身份；不按运行结果换fixture。
2. 增加必要episode事务接口：fresh live root、S-CEM调用、唯一first-action执行、冻结A/C dispatch、env.step后真实重采集、异常保留/禁止重复与重试。测试先定义失败路径，保持科学核心文件不变。
3. 有界真实AirFogSim两轮机制smoke + domain empty/missing/bridge/setter/env.step异常测试；只用当前因果历史，normalization/checkpoint hash不变。
4. 形成真实任务指标记录接口与Protocol Draft；数量/seed/长度/baseline为建议，成本依据6.4H320raw，价格未确认不编造费用。
5. CPU检查/Context/registry/index/收口commit+push，然后停止等待研究者审阅和独立GPU授权。

已知缺口：旧脚本硬编码MH且缺episode管理；Route仅NOOP限制新任务路由；slot容量超限须fail closed；指标分母/研究规模尚未批准。

---

<!-- STEP6.4H ACCEPTED CURRENT -->
## 2026-10-09 STEP 6.4H — S-CEM B512 预算资格完成

STEP_6_4H=PASS；SEARCH_METHOD=S-CEM（K=4,rho=0.2，仅Planner v1 pure-search骨架）。新B512 320/320与修复域6.4G A的S-CEM B1024父320只读严格配对，本地原始结果独立重算、ZIP/逐文件SHA与持久D:备份PASS。两预算均160/320=50% H4可评分、集合完全相同，32/64困难起点保留；前三项N_DDL/A_DDL/J_Delay首差劣化0。B512对B1024为11胜/233平/76负，六类0/0/11/76/73/160，64-anchor整组抽样95% CI=[-0.290625,-0.11875]。差异仅在J_Effort首差：B512更好11、更差76；J_Burden首差0，双方可评分全等73。资格通过不表示整体目标等价；完整字典序比较显示Effort损失。

S_CEM_B512_QUALIFICATION=PASS；RECOMMENDED_CLOSED_LOOP_B_WM=512；FINAL_CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING。B512实际转移163840（名义相同）；不同可评分候选30209 vs 72587。单次内部搜索mean/median为74.548/51.416s vs175.249/126.359s，节省57.46%/59.31%。矩阵总墙钟42613.116s（11h50m12s），含内部搜索23855.506s、加载/setup10899.799s及其余核验/持久化开销，不能用总墙钟冒充单次搜索耗时。

future Return-birth fixed-support限制保留：B512/B1024边界49845/162239，grammar dead-end19882/46621，scorer exception/inconsistency均0。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_RESEARCHER_BUDGET_DECISION；已有闭环机制与fallback不改变，但FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED、HYBRID_PLANNER=NOT_FROZEN。6.4G Phase B=NOT_STARTED_BY_CONDITIONAL_STOP，原三方法Stage B=NOT_STARTED/DEFERRED；locked_test=false。GPU=RUNNER_STOPPED_NO_FURTHER_USE：原3080Ti CUDA FP32 batch16单runner自然结束，全部备份通过后已告知研究者可以关机，实例电源状态未由本任务确认。唯一下一动作：研究者审阅Effort损失与时间节约，决定最终pure-search闭环预算；不自动启动闭环或其他实验。

证据：code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2/{acceptance.json,local_backup_acceptance.json,qualification.json,summary.json,paired.json,objective_components.json,runtime_tradeoff.json,matrix_runtime.json,inventory.json,archive.json}；以下6.4G及更早内容为历史状态。
<!-- END STEP6.4H ACCEPTED CURRENT -->

<!-- STEP6.4H GPU RUNNING -->
## 2026-10-08 STEP 6.4H — S-CEM B512 正式矩阵运行中

研究者已开启RTX3080Ti并授权续跑。远端精确gate dc0e6c94b2353bdaf1c671e52bde2a978a991bf8、config5cf6685959ad8847150c1680ba55c399102858611e5c2d955fead8f32f6c46bf；CUDA FP32 batch16，checkpoint不变。无旧runner、tracked clean、约43.8GiB空闲容量及绑定检查PASS。网络git fetch超时后以逐SHA验证的增量Git包精确fast-forward，不改变源码/协议。唯一runner PID1615于2026-10-08T03:50:21.905759UTC启动，首个case已完成并本地身份/512预算/SHA备份核验通过；最新精确进度以namespace/runtime_status.json为准。

STEP_6_4H=RUNNING；S_CEM_B512_QUALIFICATION=RUNNING_NOT_DECIDED；FINAL_CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING。仅新S-CEM K4/rho0.2 B512共320，B1024父320只读、不重跑。6.4G Phase B条件停止、Stage B DEFERRED保持。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；HYBRID_PLANNER=NOT_FROZEN；GPU=IN_USE；locked_test=false。监控pi-jwm-6-4h-s-cem每10分钟只读快照/D:SHA备份，异常不重启，不操作云平台开关机。当前唯一下一动作：等待此320矩阵完成，再独立统计/ZIP/SHA、本地验收与Context/Git收口；运行期间不push新source/gate。

以下CPU门与旧阶段记录为此前状态。启动证据：code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2/07_gpu_launch_acceptance.json。
<!-- END STEP6.4H GPU RUNNING -->

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

## STEP 6.4G — 当前授权与执行计划（2026-10-04）

当前gate：CPU协议/runner/fallback/统计合同preflight；初始HEAD=origin/main=1f2b9fafeeb00251383ae4551a78b1a9ef5bc48a、tracked clean。研究者要求完整TRAIN768→完整StageA960→仅当新选择MH时B512320，不复用任何旧raw进入新统计。旧证据标签historical-under-pre-6.4F-domain；新结果仅为repaired-domain formal evidence。

依赖顺序：
1. 只读重建原TRAIN行为支持catalog并核验SHA/语义；核验修复域32/64非空、checkpoint与固定输入，不换anchor。
2. 测试先行落实正式fallback A-else-C、逐phase namespace/resume/条件门和预算验收gate；solver/scientific semantics不改。
3. 新6.4G protocol/manifest/phase identity模板、storage/runtime/statistics/independent acceptance runner冻结。CPU focused/compile/index/Context/diff通过后commit+push。
4. 3080Ti若不可达：READY_FOR_GPU_LAUNCH=true，停止等服务器；不启动/迁移服务器或CPU长搜索。若可达则精确Git/source/checkpoint/device preflight后PhaseT；每phase独立验收+Git gate。
5. PhaseA若新选择非MH：硬停止，budget待研究者；MH才可PhaseB。B512gate失效时budget待决定，不自行切1024。

本轮fallback研究者已冻结CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1：NO_SCOREABLE_H4且当前域非空/输入完整才A，其余C；A任何失败C，不重试、不换动作、不级联behavior。不启动正式闭环、B256、旧StageB、hybrid/baseline/ablation/locked_test。

当前缺项：新namespace/protocol/runner与CPU gate尚未完成；GPU服务器此前已关闭，需只读确认可达性。唯一下一动作：完成support catalog独立只读audit。

---

## 2026-10-04 STEP 6.4F（已完成当前授权范围）

STEP_6_4F=PASS；Comm当前资格修复与真实一步机制通过。共同谓词：Task present、offloading/transmitting、未完成，再与唯一合法活跃无线Flow绑定相交。Flow对象不删除，Flow存在不等于可执行。TRAIN4416/Validation1104完整静态前后审计与输入SHA一致；32/64搜索anchors中28/57的根候选域改变。真实TRAIN anchor0041：Task_24（offloading），UAV_1→vehicle_7，RB0/start0/width1，setter与真实无线传输事件一致，4.5→4.6s一次决策步；候选A独立episode、模拟NO_SCOREABLE_H4、同一bridge、无Comm的canonical合法动作亦4.5→4.6s。Future Target poison不改变当前输入/动作。FORMAL_SEARCH_EVIDENCE_REUSE=TARGETED_REQUALIFICATION_REQUIRED：7个Validation起点有覆盖全部H1–H4分支的规则不变量证明，另外57个旧raw无完整候选/预测轨迹，不足以证明winner/ranking不变。最小建议重新验证57×5×3方法B1024=855及57×5 MH B512=285，共1140cases；仅建议，未授权、未执行。旧TRAIN调参28/32根域受影响，不声称历史调参验证新域；K4/rho0.1按本轮研究者指令保持，不自动retune。B512_EVIDENCE_REQUALIFICATION_REQUIRED=true；PLANNER_V1_CLOSED_LOOP_B_WM=512作为研究者working-budget保留。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_FALLBACK_DECISION仅表示执行机制就绪，不解除搜索证据重新验证门。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING；SEARCH_METHOD=MH-CEM pure-search骨架，非最终hybrid。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED / DEFERRED；GPU=NOT_USED；locked_test=false。唯一下一动作是研究者审阅旧证据重新验证范围及fallback决定，不自动开跑。

证据：`docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/09_final_acceptance_receipt.json`；`code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/07_formal_search_evidence_reuse_assessment.json`。历史6.4E BLOCKED及旧搜索accepted状态仅在旧定义内保留。

---

## 2026-10-04 STEP 6.4F（当前：修复前只读影响审计）

研究者明确冻结当前Comm资格为唯一合法活跃无线Flow绑定∩当前offloading/transmitting∩未完成；Flow存在不代表Planner可执行。正在保存原始源码与TRAIN4416/Validation1104、32/64搜索anchors的静态基线；基线落盘后才能改资格predicate。下一动作只完成pre-patch audit。GPU/StageB/正式长闭环/locked_test禁止；B512工作预算保持512，旧搜索证据可复用性尚未判定，不自动重跑。

以下为历史状态。

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

## STEP 6.4D 运行期间的索引时点限制

新增raw/runtime持续写入时，knowledge-index --check捕获artifact_catalog.csv时点差异；协议提交前静态index write/check已PASS。该差异不来自科学源码或identity漂移，GPU未停止。最终结果固定后必须重新write/check并PASS，再完成收口提交；运行中不把index快照当最终验收。

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

# Findings

## 2026-10-01 STEP 6.3D FORMAL TRAIN TUNING CLOSURE

八组配置各 48/96 cases 成功找到可评分 H4；16/32 锚点在 3 seeds、两种方法和四组参数下均无可评分结果。57,797 次不可评分 H4 完成尝试均记录 future Return birth 支持边界；37,940 个 Grammar dead end、scorer 异常 0、静态空域/零 cohort 0。完整 H4 90,263、去重后可评分候选 31,667；因去重，完整数不等于可评分与不可评分两列简单相加。TRAIN 选参规则分别选 S-CEM/MH-CEM `(K=4,rho=0.1)`；这不证明两方法相等或相对 HRS 优劣。Validation 1,720,320 名义转移按 TRAIN 实测有效速度粗估 93.54 小时，B_WM1024 可能更慢；仅作资源规划。

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH

- 原 96 个方法比较锚点中 8 个 `cohort_count=0` 是冻结 6.2B scorer 的必然不可评分输入。静态同 split/四分位/Comp stratum/hash 替换后，32/64 全部非空且 cohort>0；样本选择不读取 rollout/search 结果。
- TRAIN-only 固定诊断的 32 个锚点恰好 16 有 H4 可评分候选、16 没有。275 条不可评分完整轨迹都具有 `UNSUPPORTED_FUTURE_RETURN_BIRTH`；27/32 个锚点至少有一次该原因，14/32 有语法死路。空 cohort 残余和 scorer inconsistency 均为 0。该诊断只说明固定 HRS seed6391/B64 下的评分支持，不能推断三种搜索器的胜负。
- 真实 TRAIN CPU 串行/批量转移在 batch 1/4/8/16 通过既有 `1e-4` 数值容差及 Grammar、`H_sup`、字典序排序检查；batch 8 在本次 64 转移探针中最快但只比 batch 1 快约 1.11 倍。`B_WM` 仍按 unique candidate-step 计费，重复请求命中缓存。
- 研究者未冻结“大量不可评分”的数值阈值，16/16 不构成“多数样本可评分”；因此 H4 readiness 需研究者基于完整分布判断，正式 6.3D 大矩阵暂停。本 Patch 的输入/执行路径可验收，不改变 Objective、模型、搜索方法或正式选型协议。

## 2026-09-29 STEP 6.3D（进行中）

- 6.3D 的 96 个分层选中锚点都能从原始轨迹只读重放建立 exact-aligned causal deadline sidecar；不需要改变 Raw、Objective 或 CandidateDomain。
- 真实冻结模型探针可严格计量 `B_WM`，但选定首个 TRAIN anchor 的 H4 完整轨迹在 8、32、256、512 预算探针里均无法按冻结 6.2B Objective 完整评分。512 次新源码重跑的 130 条完整轨迹均在 H1 触发 `UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_15`；旧 512 收据保留为诊断前对照。该事实只属于一个 anchor，不能推广为方法胜负或整体成功率。
- 正式矩阵若预算都耗尽共 2,113,536 次 unique one-step transition。256 次探针为 262.665 秒，512 次为 558.291 秒；串行 CPU 完成需要数周量级。运行时间是资源诊断，不可用于选择 HRS/S-CEM/MH-CEM。研究者当前仅要求本机 CPU，故未使用 GPU 或改动预算。

## 2026-09-29 STEP 6.3B

- Formal TRAIN: 293/4608 raw slots include repeated Comm rows for one Task, 251 joint structural signatures and 145 observed cyclic `(start,width)` pairs. Policy no-intervention is not empty action: Comm 674, Comp 913, Mob 2761 raw slots still have rows.
- Current model rule does not promote offloading to computing at Flow completion. Future Comp grammar therefore requires predicted `computing` and unique Exec binding. TRAIN-only future Return `H_sup` histogram remains unavailable; the grammar does not require it and does not invent one.
- Structural TRAIN support does not mean an exact concrete action was observed; current Task, relation, node, RB start and UAV are causally rebound. Temporal unseen labels are diagnostic under the researcher-approved policy.

## 2026-09-28 STEP 6.3A

- Formal TRAIN contains 1969 non-empty Comp entries; every entry reconstructs from decision-time computing tasks/static CPU using the existing `allocate_work_conserving_cpu` rule, with global alpha counts 0.5=248, 0.75=242, 1.0=1479.
- Comm support rows have widths 1/2/3, cyclic-contiguous RB IDs 0–49, and cross-task RB reuse in 100 slots; Raw does not carry a relation slot index.
- Two present UAVs use shared profile pairs only; independent profile marginals cannot be treated as exact joint support. Family product is `NOT_SUPPORTED`; temporal sequence uniqueness rises to 2842/4416 at H4.
- Historical Return fixed-support receipt aggregates train and validation: 8828 window-horizon events, 2901 windows; no train-only horizon histogram is available, so no such distribution was inferred.
- Recommendation remains proposal-only: measure a support-constrained structured syntax and conditional designs before method selection. No candidate optimizer, ranking, training, GPU, baseline, closed loop or `locked_test`.
- Full unittest discovery was not clean (1993 run, 34 errors, 1 historical receipt mismatch) and executed synthetic CPU trainer tests outside the audit scope. Formal model/data/checkpoint were not modified; the test deviation is disclosed in the implementation record.
- Closure state: first evidence commit `b2059c4dbe173669f296b9f20a1225c9695d459e` is pushed; a separate process-record commit will capture the final verification and stop boundary. `STEP_6_3A=PASS` remains limited to support audit evidence, not method selection or planner performance.

## 2026-09-28 STEP 6.2B-PATCH

- 旧 6.2B 阻塞来自 Planner v1 曾准入会改变 4.4 Task-Agent/learned latent 的非空 Route；研究者现冻结 Route no-op，故无需修改 4.4 或 checkpoint。
- 有界因果 fixture 的 pending、existing same-path、多跳、改目的地 Route 全部在 gate 拒绝；合法非空 Route 和合法可改变状态 Route 计数均为 0，`ROUTE_EFFECTIVE_FREEDOM_V1=NONE`。Route 接口与多跳代码保留。
- 选定真实 anchor 的 RB 支持是 50 个全局 ID；242 是 communication relation 行数，不能作 Comm effort 分母。
- CPU scorer 合同 PASS 不意味着候选方法、排序质量、闭环或真实吞吐有证据；未来 Route/offload 语义需另行研究。


## 2026-09-28 STEP 6.1

- 第一 Formal Validation anchor `...::anchor-0001` 的 Raw frame/capture 和 Sample 身份可逐项对齐；同一当前 Flow/Task-Agent/UAV/CPU/RB support 能构造四个独立 `CAUSAL_DOMAIN_PROBE`，无需第一未来已执行动作。
- 冻结模型在四类动作各自对应的 routed embedding 和 hidden latent 上有非零响应；Comp 的 H1 Motion/CSI 仍可相同，不能据此判为动作路径断裂。Route 对现有 Flow 改路且保持固定 identity，Comm 对 carrying relation 的 delivered service 有变化。
- 已查证期望服务 trace 的 `outage_uniform_draw` 全 NaN 是 `apply_known_stochastic_wireless_service` 对未抽样的显式记录；actual/nominal/probability 有限。Mob 大绝对坐标上的 FP32 相减会放大小位移表示误差，但按模型同一 FP32 运算的绝对下一位置相符。
- 机制证据限定一个 validation anchor/一个训练 seed；不推出反事实预测精度、最终 Planner latency、安全、目标函数或闭环收益。`locked_test` 未访问，GPU 未使用。


## 2026-09-26 STEP 6.0B

- 本地 `code/reference/AirFogSim/` 无 `.git`；`git -C` 的 SHA/remote/status 实为 PI-JWM。官方上游候选 commit 与本地抽检 3/5 blob 相同，不能当本地 source SHA。相关源码内容哈希已写 receipt。
- `FogProfile.cpu` 是可观察静态容量；TaskManager 无节点分配总额检查，callback 可超分配；动态可用 CPU 无决策时刻直接字段或严格推导公式。结论 `STATIC_CAPACITY_ONLY`。
- UAV setter 直接保存 angle/phi/speed，step 按三角公式×traffic_interval 更新，未限速/限角/限高/裁剪地图。示例 config 速度和高度范围只用于生成/初始化；正式 Raw 动作范围仅 DATASET_BEHAVIOR_SUPPORT_ONLY。
- World Model UAV 位置方程与 simulator 相同；simulator raw 加速度符号与 PI-JWM canonical 相反，已有 Raw 字段分离，不是新模型改动授权。
- 两项 Candidate UNKNOWN 留存；没有训练服务器接触或 Planner rollout。

## 2026-09-26 STEP 6.0A

- 当前正式 FullFormalTrainer 通过 `step5_2_training_loop_v1.py` 复用 `build_step5_1d_unified_model_chain_v1.py::build_action`；输出 11 个动作张量字段。旧 P6 `task_action*` 是历史合同，不能续用。
- Comm mask 继承当前 `rb_active_mask` 并增量置位；正式 World Model 按同时分配计算干扰，不能自行增加全局 RB 排他规则。
- CPU static capability、Comp 请求、actual service 与 dynamic available CPU 不等价；后者仍无已证实因果源。数值 UAV 约束同样未冻结。
- 新四动作合成 fixture 与当前 adapter 所有张量逐值相等。该结果限于 `SYNTHETIC_CONTRACT_EVIDENCE`，不表示真实候选覆盖/规划效果。
- 5.6B 独立远端训练未连接；仓库仅有 9 月 25 日过程快照，不能推断当前训练已结束。

## 2026-09-23 STEP 5.5-PATCH

- `DevelopmentBundle.from_formal_interface()` 只读取 `interface.runtime_package_paths`，原 CPU H4 证据实际上是 1+1 subset，不支持完整 5520-window Trainer 声明。
- 原 `unsupported_count/unresolved_count/fixed_support_blocked_count` 仅从不存在的 `unsupported_future_structure` 字段取长度；全量真实 typed Return 对照 current logical Flow index 后，unsupported/fixed-support 各 8828 次，unresolved 0。按窗口与 horizon 计，不能解读成 8828 个独立物理任务。
- Raw 日志的 duplicate Task repair 为 213 次、50 trajectory、124 trajectory-task；collector 遇不同对象同 task_id 会抛错。函数只移动相同对象的集合引用，fixture 的 transmitted/computed/returned/done 值不变；未做关修复反事实重跑。

## 2026-09-22 STEP 5.1B-PATCH

- 原实现把 `[B,L,S,F]` target 沿 horizon 求和为 `[B,S,D]`，导致 future posterior 可读取其他 horizon；Patch 保留 horizon，并验证篡改 horizon 2 不改变 horizon 1 embedding/q。
- 原 receipt 使用零 h、target==prediction 和硬编码 checks；Patch 真实调用 STEP 4.4 current latent/dynamics/priors/decoders，所有 required checks 由计算结果产生。
- 当前 5.1A target support 为 10 physical / 74 communication，STEP 4.4 development model support 为 8 / 44；receipt 使用显式固定支持子集，完整支持对齐仍是 blocker，不能因此宣称 COMPLETE/FROZEN。

## 2026-09-21 STEP 5.1A-PATCH

- Root cause 1: `extend_future_motion_csi_targets()` kept `history[-1]` as the position reference for every horizon, while STEP 4.4 recursively applies `state.position + vehicle_motion[..., :3]`; this produced cumulative anchor-to-future targets for horizon 2+ instead of local one-step targets.
- Root cause 2: Motion rows were enumerated from each future frame, and tensor rows were written by enumeration order. That made supervision depend on future row order/target index and allowed disappearance or future-only birth to change model slots.
- CSI values were already looked up by relation ID and RB ID, but the former receipt did not bind those rows to the actual current tensor relation slots used by STEP 4.3A/4.4. The patch adds and verifies the full slot/identity/endpoint/type/RB chain.
- The old STEP 5.1A receipt therefore did not prove multi-horizon Motion correctness or model-slot equality. The patched receipt requires all local-step, fixed-slot, current-model identity, normalization/mask, deterministic, real-development and scope checks to be true.

## 2026-09-21 STEP 5.1A

- Future Motion 的可核验实现是 Vehicle `[delta_x, delta_y, delta_z, next_speed]`；position delta 只除以冻结 position std，不减 position mean。
- Future CSI 必须由 target frame 对应 outcome 的 `channel_rows` 提取，不能读取 History CSI，也不能用 rate/service/outage 替代；输出只覆盖当前支持关系。
- 12 个真实 non-locked development samples 通过 focused contract、tensor round-trip、篡改检测和 receipt AND；这不是正式 Dataset、训练或性能结果。

## 2026-09-21 STEP 5.0 — Definition 05 audit

- The read-only Definition 05 note (SHA-256 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`) still specifies observation NLL, Event/Residual learning, and overshooting. The researcher's newer explicit ten decisions resolve this conflict: deterministic mean decoder + Motion/CSI MSE + family KL, Event/Residual heads absent in v1, and overshooting OFF.
- Current STEP 4.2A normalized samples contain future entity `position_m` only as raw value/mask/unit; it is neither normalized nor collated into the frozen tensor. `target_entity_features` contains only speed.
- Current target/sample/tensor namespace has no future per-RB CSI. History CSI and future communication service are not valid substitutes for CSI target.
- Current STEP 4.4 state adapter uses raw position while Comm CSI comes through the normalized graph path. STEP 5.1 needs an explicit normalized-loss/raw-rule bridge before training.
- Historical `_masked_mean`, analytic diagonal KL, horizon accumulation, seeding, AdamW, logging, checkpoint reload and manifest patterns are locally reusable. Old total loss, full-target posterior, overshooting tensors, staged encoder freezing, P4 gate selector, historical checkpoints and result numbers are not current Definition 05 evidence.
- No Definition 05 runtime was executed. Training remains blocked by target/data/loss implementation, not by GPU availability.
- Optional full-suite audit ran 1849 tests with 33 errors outside this documentation-only diff: Windows GBK output failure in AirFogSim import, absent historical artifact files, old teacher-tensor fixtures rejected by current RB-action validation, and clean-tree assumptions. Relevant Definition 05/STEP 4.4 audit regression remains 75/75 pass; do not report the whole suite as passing.

## 2026-09-21 STEP 4.4-PATCH3

- Frozen Task History features expose work/computed/transmitted/elapsed only; `return_size` is explicitly unavailable to the frozen Tensor/Graph contract. Therefore current-side no Return slot is epistemically unknown, not known-no-Return.
- The previous transition ignored `task_return_requirement_known`, so computation-finished unknown tasks could be falsely completed. PATCH3 adds a distinct unresolved side-state and keeps known-required/no-slot separate through `return_birth_required`.
- Previous DAG release counted invalid edges as predecessors. PATCH3 masks by `dag_validity` and machine-tests root, incomplete, completed, invalid, and multiple-predecessor cases.
- Previous terminal completion did not change `flow_status_index`. PATCH3 uses `FLOW_STATUS_VOCAB.index("COMPLETED")`; partial and intermediate-hop counterfactuals remain active.
- Builder acceptance now observes state changes rather than relying on field/module/policy-string existence. Focused 30/30, related regressions 82/82, formal receipt 92/92, compileall, and six-file independent hash/size equality pass; STEP 4.4 is COMPLETE / FROZEN.

## 2026-09-21 STEP 4.4-PATCH2

- 4.3A 已真实保留 Flow `task_index` 与 `flow_type_index`；此前 World Model adapter 丢弃前者并把所有 `return_flow_index` 初始化为 `-1`，这是已有 Return 无法绑定的直接根因。
- Return gate 必须由 typed structural identity 建立，不能依赖 slot 位置；完成的 Input Flow 不能替代 Return Flow。
- frozen Task tensor 没有 `return_size`，因此 future-only Return birth 不能从当前 support 可靠推断或创建；v1 必须显式声明不支持，并让外部已知的 `task_requires_return` side-state在缺少 slot 时阻止 final completion。
- Definition 05 应把跨越 unsupported Return birth 的 window mask/exclude/分类，不能把对象支持缺失当作普通预测误差。

## 2026-09-20 STEP 4.2A-PATCH

- `_physical_structure()` 先建立 V/U/I directed structural relation，再读取 CSI；`observed_mask=false` 只证明 CSI feature 缺失，不证明 relation 不存在。
- relation mask 与 feature mask 必须分层；否则 node-index 暂不可用会错误删除 Information Graph 的结构边。
- `_extract_tasks()` 已提供 return size、deadline、priority、task delay 的 simulator observer source；冻结 Raw 未透传不能写成 simulator 无可靠来源。
- 旧 `LogicalFlow`/`CarryingHop` 类型名和 past hop service 仍不足以证明定义 03 stateful Flow 的 identity、total/rem、multi-hop 与 route-revision 语义。

## 2026-09-20 STEP 4.2A

- wired topology 可以在动作执行前物化为 directed typed relation；CSI 缺失由 type + mask 表示，不能借用 wired service outcome。
- CPU capacity 应作为无 H 轴的静态 Agent capability；Comp allocation 和 actual service 的反事实变化都不改变该张量。
- Task progress/elapsed 与 Src/Host/Exec/Ret 可以从当前 Decision 因果构造；Future Route target 不得倒灌到当前关系。
- 已有来源字段贯穿 Tensor 后，stable stateful Flow 仍是独立的 Raw-insufficient blocker；不能用 past hop service 改名填补。

## 2026-09-19 STEP 4.1

- 当前数据缺口分层处理：position/无线 CSI/CPU static capacity 等是 Raw 已有但未暴露；wired 最小 relation 有 topology/`hasLink` 来源但未逐 Decision 物化；wired 可选动态 numeric state 与完整 stable stateful Flow 才是 Raw 本身不足，不能用同一种 patch 处理。
- `entity.getFogProfile()['cpu']` 在当前配置与真实轨迹中表示稳定的节点能力上限；它不能替代 `A_t^Comp` allocation、Outcome actual service 或尚无来源的 dynamic available CPU。
- `past_outcome_flow_service` 是过去某一 hop 的实际服务结果，不具备 current Flow 的 total/rem 语义；旧 `FLOW_FEATURES` 名称也不是当前数据证据。
- cloud 的 `[0,0,0]` 坐标不能证明它有独立空间建模意义；edge/cloud 的 Physical membership 必须由研究者决定。
- 旧 `EDGE_FEATURES` 把 distance、CSI、rate、active task 和 RB 混在一个 physical edge，和定义 03 冲突；仅通用 stable ID/mask 与无语义的 masked-index 算子可直接复用。

## 2026-09-19 STEP 3.3F

- fixed-shape 通过不代表语义完整；Past Outcome 和 Target future facts 必须作为独立 tensor 角色保存。
- category code 不能由 development batch 中“碰巧出现的值”决定；固定 vocab 才能保证 subset/reorder 稳定。
- past hop service event 不等于定义 03 的 current stateful Flow total/rem state；Gap Table 已明确区分。
- 实际 artifact 有 past offload Route，`max_route_hops=2`；旧记录中的“无 Route/max=0”与机器证据冲突，已纠正。

## 2026-09-19 STEP 3.3

- Tensor collation 必须使用 JSON sample 的 stable ID/index，不能按每帧可见集合重新编号。
- `presence=false` 与 `feature_mask=false` 均独立于 numeric value；padding 不进入有效 feature。
- target-only object 需要独立 target capacity；当前 development capacity 不是正式研究容量。
- 当前 batch 没有 route entry，因此 route hop capacity 为观测值 0；后续正式容量需研究者单独冻结。

## 2026-09-19 STEP 3.2 findings

- Existing Step 2.x Raw version directories were one seed-0 trajectory lineage, not independent split members; treating versions as trajectories would leak provenance.
- Fresh seed 1/2 non-locked collector runs passed all Raw checks without schema or simulator changes.
- A mask test exposed that object-level `feature_mask[field]=false` must override the field wrapper mask; the implementation now combines both masks and ignores masked extremes.
- The resulting bundle is observation-only validation evidence, not a formal Dataset or generalization result.

## 2026-09-19 STEP 3.1

- 新定义 `02数据集构建与模型输入.md` 要求已落实为最小合同；input-side index 不使用未来窗口 union，未来新对象在 target-side index 表达。
- STEP 3.1R 更正：真实 observer 已提供 DAG dependency rows；Raw v2 的 Decision 保存 2 条当前可见边，30 条含未来端点的边只在 internal metadata，model-ready input Static 不读取未来端点身份。
- Step 2.4 窗口的 target frames 没有 transfer event，故 flow rows 为 0；这是真实证据覆盖限制，不伪造服务记录。正式 batch 需要后续真实有服务窗口验收。

## 2026-08-26 双约束全仓审计与计划重构

### Teacher requirement (verbatim)

> “信息边特征可以不用这么多，就是可以考虑在消息不是那么完整的情况下面做预测；然后真实数据集可能没办法涵盖所有的内容，每一个真实的数据集可能各有侧重，所以可以考虑用不同的数据集对不同的方面进行微调训练增强；然后就是方法，不能只通过实验结果来说明每个模块选择哪个方法，而是要说明为什么这个方法适合我们的课题场景；然后就是不同的方法要针对我们的场景有调参这个过程”

### Audit scope and evidence

- 仓库清单按 `rg --files`/PowerShell inventory 复核，排除 `.git`、`.worktrees` 后约 725 个文件；文本/代码/配置/记录/JSON/JSONL 约 710 个、约 16.996 MB。核心目录 `code/src/pi_jwm`、`code/scripts`、`code/tests`、`记录`、`meeting`、`docs`、`literature`、`paper` 已逐目录核对；第三方 `code/reference/AirFogSim`、PDF/PPT/PNG 等二进制或外部资料按路径、manifest、哈希和元数据处理，不把二进制当普通文本解析。
- 权威入口为 `AGENTS.md`、`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`、`记录/接续记录/新对话接续说明_20260815.md` 及 planning files。早期“GPU未批准/P2未完成”文字均已由顶部最新状态覆盖，不能直接作为当前判断。
- 当前可复用证据：P0/P1/P2 contract、非 locked v4 tensor、确定性规则层、三 seed CPU rule replay、规则启用 aggregate candidate 的非 locked GPU smoke/固定数据训练；当前 `formal_performance_claim_ready=false`。
- 当前关键阻塞：candidate-action planner code-only audit 为 `blocked`；R6 是 direct belief/state-conditioned scorer，`formal_candidate_rollout_planner_v1.py` 仅 `prototype_only`。缺少合法候选生成、共同 belief 下逐候选 world-model 调用、未来 state/task/cost/risk 提取、预测后果选择和执行反馈重规划。

### Plan decisions

- 保留 P0–P10 编号，但以 2026-08-26 v2 入口重新定义依赖关系；P3 先补齐三模块理论适配与多源数据角色，P4 闭合 world model，P5 闭合 planner，P6 才做公平场景化调参，P7 才做多源微调/扩大训练，P8 后才锁方法，P9 最后访问 locked-test 与 baseline。
- 老师四项要求被固化为 T1 部分信息、T2 多源数据分工、T3 方法理论适配、T4 场景化调参；项目五项约束固化为 C1 一致性、C2 contract、C3 规则递推、C4 candidate rollout 闭环、C5 locked-test 最后。
- 已有结果只按其真实边界复用：aggregate/non-locked/diagnostic，不升级为逐 RB 方法、正式 planner、最终性能或 locked-test 结果。
- machine-readable 双约束门矩阵已落盘：`记录/双约束门矩阵_20260826.json`，包含 T1–T4、C1–C5、P0–P10 当前状态和 GPU/locked-test 边界。

## 2026-08-26 P3 方法适配与多源数据方案

- P3 设计包已完成：三个模块均写明“问题特点、方法、为什么适合、已知不足、检查指标”；说明使用尽量简单的语言，避免把设计目标写成已经完成的能力。
- 多源数据只按真实覆盖范围分工为仿真主线、无线遥测、边缘任务、移动/信道和 held-out transfer；尚未完成 source closure 的真实数据不得进入正式训练。
- 公平调参规则已预注册，但实际 sweep 仍属于 P6；planner 仍是 `prototype_only`/`blocked`，没有打开 GPU 或 locked-test。
- 下一门是 P4 的 CPU/non-locked 世界模型检查，不重复 P0–P2 和既有 aggregate GPU 证据。

## 2026-08-26 P4 世界模型机制门收口

- 冻结的 h20 tensor contract 已验收：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_unlocked_20260826`，history=8、horizon=20、54 个 unlocked seed、14,742 个窗口，统计量只来自 train split。
- CPU 1/5/20 horizon 机制审计通过：输出长度和 finite 性正确，动作扰动会改变预测，target 改写不会改变预测；报告为 `code/artifacts/audit/pi_jwm_formal_p4_horizon_mechanism_audit_20260826/p4_horizon_mechanism_audit.json`。
- 统一 P4 机制门通过：规则 replay、候选一致性、非 locked GPU 历史 artifact 边界和 horizon 接口检查均通过；报告为 `code/artifacts/audit/pi_jwm_p4_world_model_gate_h20_20260826/p4_world_model_gate.json`。
- 关键限制：上述 probe 使用的既有 checkpoint 训练 horizon=3，h20 只用于输入/输出接口机制审计，不是 h20 重训后的精度结果。因此 P4 只能标记为 `complete_for_mechanism_only`；正式 state/task/resource/uncertainty 精度门仍 pending。
- 当前边界保持：不启动新的 GPU、不访问 `locked_test`、不进入 P5/P6，`formal_performance_claim_ready=false`。

## Rule-layer hardening superseding findings (2026-08-23)

- The first interface closure was insufficient: label-time endpoint sourcing, action-occurrence stage completion, and full predicted-state feedback were invalid. These paths are removed or corrected.
- Learned heads predict physical edge rate and a bounded flow-service fraction. Deterministic rules convert that fraction to capped delivered data, write RB occupancy, allocate CPU by capped equal sharing, and enforce lifecycle/DAG conservation.
- CPU is not a policy action in the revised path. Logged `cpu/cpu_allocated/cpu_fraction` values are zeroed before action encoding and are not read by the deterministic CPU rule.
- Node CPU is not overwritten as a fake remaining-resource state. Capacity is a constraint; per-task allocation and served work are emitted separately.
- Rule feedback is the masked correction `rule_state - learned_state`, not the full state, preventing double injection of unconstrained node predictions.
- New tensor and training audits are current authority. Earlier rule-interface audit and pre-rule GPU checkpoints are historical only.

## Deterministic rule-layer resolution (2026-08-23)

- The four former interface blockers are closed at the code/data boundary: train-only stats are passed through model configuration; action source endpoints are explicit in each formal window; service outcomes are produced by dedicated learned service heads and consumed by the deterministic layer; deterministic masks are explicit static tensors.
- Rule-governed updates are physical-unit operations: edge rate/RB occupancy, node CPU availability, flow remaining/cumulative/slot delivery, task transmitted/computed fields, and DAG unfinished-parent/release state. Outputs are converted back to the training-normalized space before loss/feedback.
- A first backward pass on a real validation window exposed and fixed an autograd-breaking in-place write. The replacement uses functional feature reconstruction and `scatter_add`; real forward/backward now succeeds.
- This is an implementation/interface closure, not a performance result. The selected 2026-08-23 GPU runs predate the rule layer; consistency audit therefore reports `candidate_checkpoint_requires_rule_layer_retraining`, and no GPU/locked-test claim is authorized.
- Pre-retraining hardening found that the first window-level `future_source_node_index` implementation copied label-time `task_node_index`. That is future-state leakage even though the field is called an action input. Source endpoints must instead be tensorized directly from offload/return/RB/CPU action records.
- The authority document fixes the split as: deterministic action/endpoint write, RB counts, CPU inner rule and remaining-work conservation; learned dynamics supplies channel/effective service outcomes. Therefore a rule-enabled model must not consume logged future CPU allocations as a core action. Required new tensor fields are `task_action_source_node_index`, `flow_task_index`, and `slot_seconds`.
- Task progress must be service-based: input transmission, compute, and return progress are capped by their current remaining quantities and lifecycle eligibility. An offload or CPU indicator alone is not evidence that the whole task stage completed.

## Key-wise input perturbation diagnosis (2026-08-23)

- Audit: `code/artifacts/audit/pi_jwm_formal_tuning_keywise_robustness_cpu_20260823/`, same three seeds and validation samples, four isolated history keys, noise `0/0.05/0.10/0.20`.
- `node_state` alone produced mean node-x MAE deltas `+10.129/+20.972/+42.697`; `task_state` alone produced mean task-delay deltas `+0.188/+0.423/+0.897`. `physical_edge_state` and `flow_state` produced zero node-x/task-delay deltas to reported precision. Link-F1 changes were negligible (maximum about `0.001`).
- The residual path directly anchors predicted component means to the last observed state (`formal_dual_graph_world_model_v1.py:278-282, 412-414`), explaining the node-x channel; task-state enters the task-history encoder and explains the task-delay channel. This is a confirmed sensitivity mechanism, not a fix or final robustness claim.
- Execution remains CPU-only and non-locked: `evaluation_devices=["cpu"]`, `gpu_execution=false`, `locked_test_accessed=false`, `formal_performance_claim_ready=false`. Next is a theory-code consistency decision before any new GPU run or method freeze.

## Candidate theory-code consistency audit (2026-08-23)

- Machine report: `code/artifacts/audit/pi_jwm_formal_candidate_consistency_audit_20260823/candidate_consistency_audit.json`.
- Passed checks: aggregate-baseline boundary, residual configuration (`zero_init=true`, `residual_state_scale=0.5`), task-history conditioning, and future-action conditioning. The three selected runs share one protocol; only the independent `seed` differs. The per-RB sidecar remains diagnostic-only and is not required for the aggregate baseline.
- Critical mismatch: `per_step_deterministic_rule_update_missing`. The current model loop updates latent states and sends them directly through learned heads; the audited source does not call a deterministic rule layer before explicit-state prediction. This conflicts with the PI-JWM theory requirement that rule-governed fields be updated at every rollout step.
- Gate: `status=blocked`, `cpu_training_allowed=false`, `gpu_allowed=false`, `formal_performance_claim_allowed=false`, `locked_test_allowed=false`. Do not start another training run until the rule-layer resolution is recorded and verified.
- A real validation-window contract check confirms four missing inputs for a truthful rule layer: train-only normalization statistics are outside the model batch; future actions lack an explicit source endpoint mapping; future service outcomes (rate/delivered data/CPU allocation) are absent; and deterministic-target masks are absent. These are interface blockers, not tunable hyperparameters.

- The fixed-data three-seed gate is blocked only by validation node-x MAE ratio `1.3647403366171358`; activity and operational criteria pass.
- The training runner constructs `FormalLossWeights()` internally, so state-loss diagnosis requires an explicit runner parameter rather than an ad hoc artifact edit.
- The intended first intervention is a single-variable increase of `state_mae`; all other protocol and data fields remain unchanged.
- The override is now implemented in `run_formal_dual_graph_gpu_train_v1.py`, recorded in `config.json`, validated for non-negative values, and covered by the runner test suite (`7/7`).
- Increasing `state_mae` to `0.5` worsened the screen run (validation node-x MAE `5.9431`, link F1 `0.5943`).
- Enabling `zero_init_residual_state_heads` with the frozen default loss improved the full-budget seed `20260824` run: validation link F1 `0.7371` vs persistence `0.6879`, node-x MAE `1.8530` vs `1.4939` (ratio about `1.2407`), calibration link F1 `0.3003` vs `0.2109`. This is promising but only one seed.
- Explicit residual damping `residual_state_scale=0.5` with zero-init passed the frozen three-seed numerical gate: validation node-x MAE ratio `1.13897603387634`, mean validation link-F1 delta `0.06478173263384901`, mean calibration link-F1 delta `0.10730500845124441`, and all operational deltas non-positive.
- Hardware remains unavailable: `torch.cuda.is_available() == False`, device count `0`. Numerical eligibility does not authorize a GPU run on this machine.
- The supplied remote server has RTX 4090 and Conda PyTorch `2.8.0+cu128`; the isolated current-code/v3-tensor GPU smoke completed with `gpu_execution=true`, peak device memory `75049984` bytes, and zero manifest mismatches.
- Formal GPU seed `20260824` completed with the gate-passing configuration, train/validation/calibration `256/128/128`, peak device memory `201894912` bytes, train time about `66.77` seconds, and zero manifest mismatches. This is a non-locked single-seed training result, not a final performance claim.

- Controlled GPU tuning screening completed in a new isolated remote directory: 6 runs covering 4 modules x 2 learning rates x 3 seeds, with 24 candidate records and zero manifest mismatches across all runs.
- Only `coupled_dual_gnn_residual` at learning rate `3e-4` passed the frozen screening gates: mean validation link-F1 delta `+0.0647817326`, minimum per-seed delta `+0.0494443949`, mean calibration delta `+0.1073050085`, mean validation node-x ratio `1.1391798404`, maximum ratio `1.1783511224`, and all three operational deltas non-positive.
- The selected combination is a screening candidate, not a final method freeze or final performance claim; all results remain aggregate-baseline and non-locked, and the model still does not consume the per-RB target sidecar.

- Downstream horizon audit for the selected candidate improved communication/throughput/RB metrics at `k=1/2/3`, but node-x MAE deltas remained positive (`+0.209790/+0.208241/+0.205736`) and learned node-x error growth averaged `3.16999x`.
- CPU input-perturbation audit for the selected candidate showed node-x MAE deltas of approximately `+10.13/+20.97/+42.70` at normalized noise `0.05/0.10/0.20`, while link-F1 changed little; this is sensitivity evidence, not a final robustness claim.
- The next defensible step is a CPU-only single-variable root-cause diagnosis for state error/input sensitivity. No new GPU run or locked-test access is justified by the current downstream evidence.

## Created-flow reset and replay closure (2026-08-26)

- The remaining `flow_conservation` replay failure was not in the v3/v4 source tensors. On the actual v4 rule-contract tensor root, `total_data - remaining_data - delivered_cumulative` was within `5.21540641784668e-08` for all 155,875 active flow slots and 12,924 first-active flow slots across 54 unlocked trajectories.
- Root cause was rule-state initialization for a newly created flow: a nonexistent previous slot can denormalize to the training mean, so its cumulative delivered value must not be inherited. The corrected rule starts a newly created flow's cumulative delivered data and age from physical zero, then applies only the current slot service/time.
- A fresh CPU replay of the three existing rule-enabled GPU checkpoints against their actual v4 tensor root passed. Each seed has 64 validation batches, 192 expected/observed rule calls, and empty flow/RB/CPU/lifecycle/DAG violation counts. The report remains a non-locked aggregate-baseline diagnostic; it does not establish final performance, per-RB consumption, candidate-action rollout planning, or locked-test evidence.

## Candidate-action planner mechanism review (2026-08-26, in progress)

- `r6_joint_policy.py` implements a direct candidate scorer: its forward path encodes the present explicit/latent state and `candidate_descriptors`, then produces logits. It does not accept a world model, call a world-model rollout, or expose predicted future state/task/cost/risk values.
- `r6_rollout.py` records executed policy transitions and computes GAE. Its use of the word rollout means an observed training trajectory, not candidate-wise world-model simulation before action selection.
- `formal_dual_graph_world_model_v1.py` rolls the logged `future_action` sequence in a batch. The next audit must distinguish that action-conditioned prediction capability from an action-selection mechanism which independently rolls out every legal candidate from a common history.

### Result

- The new static audit reports `blocked`: the formal model is action-conditioned, but no audited path generates legal candidate sequences, invokes the world model per candidate from a common history, extracts candidate-specific state/task/cost/risk outcomes, selects from those outcomes, or closes feedback/replanning. Artifact SHA-256: `7A71C78F78058F8F0C19EC0509816A2A31D972B27DA8CF1D8FABC8F7CC3B6637`.

## Seven-paper close reading (2026-08-26)

- The local close-reading bundle is `literature/新增7篇精读笔记_20260826.jsonl` plus `literature/新增7篇精读汇总_20260826.md`; all seven sources are public arXiv preprints.
- The literature reinforces three current boundaries: (1) simulator/world-model faithfulness must be decomposed into observable action-to-state and state-to-response links; (2) a candidate-action planner requires candidate-wise model rollouts from a common history, predicted outcomes, selection, and replanning; (3) task-aware uncertainty and paired decision effects should be measured separately from global prediction error.
- Physical priors, multi-modal tokens, shared state, and inverse-dynamics anti-collapse are candidate design ideas only. They do not change the frozen PI-JWM tensor contract, rule-layer replay result, planner audit status, or `formal_performance_claim_ready=false`.

## 2026-08-26 file-tree and evidence-layer governance

- The repository is intentionally not physically cleaned in this pass. Its large `code/artifacts/`, meeting archive, and historical log populations contain provenance and unique outputs; moving or deleting them without a reference manifest and explicit authorization would risk evidence loss.
- The reliable organization is an indexed reading order plus evidence labels: root planning files and work logs are process records; `记录/本地计划表.md`, `记录/PIJWM主文档.md`, `记录/8.12之后推进.md`, approved manifests, and P8-approved artifacts are authority/final candidates.
- A filename such as `formal`, `best`, or `latest` is not evidence of finality. State fields, blockers, hashes, split/seed closure, and the applicable phase gate decide the evidence level.

## 2026-08-27 P4 h20 precision finding

- This is a genuine new P4 precision result, not a missing-file problem: all three completed h20 seed artifacts are present and the recorded manifests match. The learned model loses to persistence for node-x at k=20 in every seed (mean `22.8591 m` versus `22.1450 m`), and its nominal 95% node-x interval covers only `75.43%` of targets.
- Therefore the correct state is `P4 precision blocked`, not `h20 pending` and not a final method failure claim. Preserve the frozen contract and diagnose one possible cause on CPU before considering any minimal change or retraining. `locked_test` remains unused.

## 2026-08-27 P4 node-type contract finding

- Code/data chain: `formal_airfogsim_collector_adapter_v2.py` stores AirFogSim `node_type` verbatim; the formal source bundles contain short codes `V/U/I/C`; `airfogsim_tensor_v2.py` recognizes only `vehicle/uav/rsu/edge_server/cloud` and writes `-1` otherwise.
- Full h20 scan: all 54 unlocked trajectory tensors contain 2,484/2,484 invalid `node_kind_index` entries. Their 290,076 present-node time slots consequently have zero valid type masks. This is not missing data: the source graph contains concrete node identities, positions, and short-code types.
- Mechanism consequence: `formal_dual_graph_world_model_v1.py` uses `node_kind_index >= 0` as the node validity mask. Physical edge-node message passing, information flow-agent message passing, and agent-node coupling are therefore zeroed on the h20 runs. Flow-edge coupling and task-to-node writes can still execute, so the path is partially active rather than a clean no-op.
- Interpretation: k=20 node-x MAE `22.8591 m` remains a real measured error, not a metric-mask artifact. But it is evidence for a malformed-input partial model, not for the intended complete dual-graph PI-JWM method. P4 remains blocked; first repair the four-code normalization and enforce the present-node type contract, then repeat unlocked CPU acceptance before deciding on retraining.

## 2026-08-27 P4 节点类型修复证据（进行中）

- 已用测试先固定规则：`V/U/I/C` 必须映射到冻结契约中的 `vehicle/uav/rsu/cloud`；所有实际出现节点必须有有效类型；padding 节点可保留 `-1`。
- 失败测试确认根因后，最小修复只改变张量化入口，不改历史源数据、世界模型、损失、训练配置或 GPU 脚本。
- 30 项直接相关测试均通过。尚未生成修复后的 h20 产物，因此还不能声称双图消息已在正式数据上重新启用；下一门是独立非锁定 tensor 重建和 CPU 验收。
# 2026-08-27 P4 RB outcome reconstruction finding

- Root cause: source bundles have `source_rb_actions` and `source_transfer_events`, but no `source_rb_observations` or top-level `n_rb`; reading only the absent observation list produced `n_rb=1` and lost direct RB labels.
- Verified rule: action `rb_indices` are RB identities; event `path` is the physical edge; rate is `planned_capacity / 0.1s`. Two unlocked trajectories have one action with no event; those remain masked/unobserved.
- Repair is limited to the RB reconstruction, tensor-contract inference, and formal graph entry, with focused tests. No model, loss, protocol, GPU script, or source data changed.
- Rebuilt artifact `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827` passes tensor readiness with 54 unlocked trajectories, 14,742 windows, `n_rb=50`, 48,447 observed RB labels, valid present-node types, and no `locked_test` output.
- P4 remains blocked pending CPU tensor/window acceptance and checkpoint reload; this is data-contract repair evidence, not a performance result or GPU authorization.
## 2026-08-27 P4 repaired h20 acceptance finding

- The repaired h20 tensor is structurally usable: `formal_tensor_ready=true`, history=8, horizon=20, 54 unlocked trajectories, `n_rb=50`, and no materialized `locked_test`.
- Focused P4 tests are green: RB reconstruction 11/11, tensor build 8/8, world-model interface 8/8, CPU smoke 2/2.
- The CPU acceptance artifact is a small interface/training check (8/4/4 windows), not a converged performance result. It proves data-to-model wiring, finite forward/backward behavior, and checkpoint artifact creation only.
- The independent training-protocol report says the contract is ready for review, but its GPU launch gate remains false because the repaired CPU baseline comparison has not yet been audited. Keep `formal_performance_claim_ready=false`.
- Full-suite failures are bounded outside this gate: one repository root markdown expectation and 13 legacy AirFogSim/Windows fixture errors. Do not use them to claim either P4 success or a new model defect.

## 2026-08-27 P4 repaired h20 Go/No-Go finding

- The repaired h20 tensor now has the required three independent CPU seeds.
- The gate still blocks GPU: validation link-F1 deltas versus persistence are `+0.0213`, `-0.4492`, and `-0.0079`; mean delta `-0.1453`. The two negative seeds are decisive.
- Other checks are not the blocker: validation node-x MAE ratio is `1.0071`, and throughput/RB/task-delay errors improve on average. Calibration link-F1 improves on average but cannot replace validation evidence.
- This does not prove the repaired model is fundamentally wrong; it proves the current small CPU protocol is unstable for communication-activity prediction. The only next work is CPU-only cause diagnosis.

## 2026-08-27 P4 communication-activity F1 diagnosis

- The three seeds were replayed on the same selected non-locked windows. Validation contains 270 positive and 25,502 negative link-activity labels; calibration contains 86 positive and 27,564 negative labels. The training subset contains 669 positives and 62,755 negatives.
- Seed `20260827` has validation ROC-AUC `0.9611749` and average precision `0.4358372`; seed `20260829` has ROC-AUC `0.9579971` and average precision `0.3066603`. These two seeds separate positives from negatives reasonably well before thresholding.
- Seed `20260828` has validation ROC-AUC `0.3559574` and average precision `0.0107389`; calibration ROC-AUC is `0.0586247` and average precision `0.0018521`. Its positive scores are generally lower than its negative scores, so this is a ranking failure, not only a bad threshold.
- Calibration-selected thresholds are `0.9`, `0.1`, and `0.7` for the three seeds. The threshold changes are a visible symptom of seed-dependent score location and ordering; they cannot be used to replace validation evidence.
- Root-cause status: confirmed evidence limitation is the combination of a very small CPU training sample/one epoch and extreme class imbalance, which makes the communication-activity head seed-unstable. A deeper architectural cause is not established by this diagnostic.
- Boundary: keep the frozen tensor contract and formal training protocol unchanged; do not repeat the same GPU run. `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`.

## 2026-08-27 P4 expanded CPU stability finding

- Expanded CPU evidence now covers three independent seeds with the pre-registered `256/128/128` budget and the same repaired h20 contract.
- Gate report `code/artifacts/audit/pi_jwm_p4_h20_repair_expanded_cpu_go_no_go_20260827/cpu_to_gpu_gate.json` reports validation link-F1 deltas `-0.0411/-0.1753/+0.0104` (mean `-0.0686`), node-x MAE ratio `1.0849`, RB-occupancy delta `+0.1132`, and calibration link-F1 mean delta `+0.2404`.
- The larger budget improves evidence quality but does not pass the gate. Two of three seeds regress in validation F1 and the mean is below persistence; RB occupancy also regresses on average.
- Root-cause status: the earlier extreme seed inversion is not the only issue. The current candidate has not demonstrated validation non-inferiority under the expanded CPU protocol. No architectural or contract change is justified without a separate approved diagnosis.
- Boundary: `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`; no GPU, locked-test, or frozen-contract modification occurred.

## 2026-08-27 P4 failure diagnosis: link activity and RB

- A CPU-only replay of the three expanded runs shows validation link-activity ROC-AUC values near `0.991` for all seeds. Fixed threshold F1 is much less variable than calibration-selected F1; the current failure is threshold transfer across calibration/validation class proportions, not a ranking inversion.
- Action-derived RB occupancy is almost aligned with the target: exact edge agreement is `99.41%`--`99.61%`, and total MAE is `0.5949` on validation / `0.7473` on calibration. The learned RB output is exactly the deterministic action-derived output, so it is not independently learning a different RB quantity in this candidate.
- The existing RB metric path is inconsistent with its own data units: normalized prediction and raw `aggregate_rb_occupancy` are both multiplied by the train scale; prediction mean restoration is missing and the target is not kept in the same physical unit. This explains why the gate reports `3.2023` despite action-derived MAE below `0.75`; the old RB regression gate is invalid until the metric is corrected and regression-tested.
- This is a diagnosis only. Do not silently change the metric or threshold protocol; require one approved, focused repair plus independent Go/No-Go rerun. GPU and `locked_test` remain closed.

## 2026-08-28 P4 metric-repair finding

- The approved minimal repair restored the prediction mean before comparison and kept prediction/target in the same physical unit. Re-evaluation confirms validation RB occupancy delta=`-0.6477 RB` versus persistence.
- This removes the old RB failure as an evidence item; it does not authorize GPU because the independent gate still reports validation link-F1 mean delta=`-0.0686` and throughput MAE delta=`+0.7101 Mbps`.
- The re-evaluation was read-only over existing non-locked CPU checkpoints and sample IDs. No model, tensor contract, training protocol, GPU state, or `locked_test` state changed.
- Current interpretation: P4 has one fewer failed gate, but the candidate still lacks validation non-inferiority. Keep `gpu_allowed=false`, `formal_performance_claim_ready=false`, and do not enter P6.
- Single next action: human decision on the remaining link-F1/throughput gate handling; no duplicate GPU run before that decision.

## 2026-08-28 P4 threshold protocol audit finding

- Added `code/scripts/run_formal_p4_threshold_protocol_audit_v1.py` and two focused tests. The script is read-only and uses only the three existing expanded CPU runs; report: `code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`.
- With `pos_weight=50`, ordinary posterior correction is `p=s/(50-49s)`. Ordinary probability `0.5` corresponds to raw score `50/51`, which produces poor recall and is rejected for this gate.
- Fixed raw `0.5` gives validation F1 `0.6305/0.4924/0.6005` for seeds `20260830/20260831/20260832`, versus calibration-selected `0.5490/0.4148/0.6005`. Threshold transfer is a contributor, but seed `20260831` still fails against persistence and the mean remains below the pre-registered non-inferiority requirement.
- The remaining throughput MAE delta is `+0.7101 Mbps` after metric repair. The threshold audit does not authorize retraining, GPU, P6, or `locked_test`; it is a diagnosis, not a gate pass.
## 2026-08-28 P4 throughput diagnosis finding

- The read-only report `code/artifacts/audit/pi_jwm_p4_throughput_diagnosis_20260828/throughput_diagnosis.json` compares every one of 20 forecast steps for all three existing expanded CPU seeds.
- The failure is not uniform: seed `20260830` has validation bias about `-3.75 Mbps`, seed `20260831` about `-4.38 Mbps`, and seed `20260832` about `+2.03 Mbps`. For seeds `20260830/20260831`, the last five steps are more negative than the first five; seed `20260832` remains positive throughout.
- Therefore the remaining throughput gate is best described as unstable rollout magnitude/calibration across seeds, with error accumulation for two seeds. The diagnosis does not establish whether the cause is loss weighting, target scale, or model capacity; no such change is authorized yet.
- `gpu_allowed=false`, `formal_performance_claim_ready=false`, and `locked_test_accessed=false` remain unchanged. No training or code contract was changed.

## 2026-08-28 P4 epoch-stability hypothesis

- Existing seed `20260830` validation loss changed from `-0.2235` to `-1.1349` to `-1.7877` over epochs 1--3; the other seeds use the same 3-epoch budget. This is evidence that the short CPU check stopped while the objective was still moving, so under-training is a plausible single cause of the throughput and link-F1 instability.
- The next check changes only epochs (`3 -> 8`) in isolated CPU runs. It does not assume the hypothesis is true; the independent gate decides whether the evidence improves.

## 2026-08-28 P4 epoch-stability execution finding

- The first seed used the intended candidate only and completed with best epoch 6 and verified checkpoint reload.
- The initial second-seed direct-function invocation exposed a reproducibility hazard: the function default is all five V1 learned methods, whereas this diagnostic requires only `coupled_dual_gnn_residual`. The unintended run was stopped after a `pooled_gru` checkpoint appeared.
- This is a launch/configuration error, not a model result. The partial directory remains isolated and excluded from the independent gate. Future CPU invocations must pass the learned-method list explicitly.

## 2026-08-28 P4 epoch-stability gate finding

- The corrected three-seed 8-epoch CPU check passes the independent Go/No-Go gate. The report records `gpu_allowed=true` with no failed criteria.
- Validation link-F1 deltas are `+0.02496`, `-0.01173`, and `+0.05789` (mean `+0.02371`); node-x MAE ratio is `1.04629`; calibration link-F1 mean delta is `+0.28557`.
- Validation operational metrics improve over persistence: throughput MAE delta `-0.49943 Mbps`, RB occupancy MAE delta `-0.64768 RB`, and task-delay MAE delta `-1.34958`.
- The result opens only the next formal non-locked GPU execution decision. It does not open `locked_test` or `formal_performance_claim_ready`, and no GPU was started in this step.

## 2026-08-28 P4 GPU reachability finding

- The independent CPU Go/No-Go report authorizes a formal non-locked GPU run, but both recorded SSH ports (`14826`, `14507`) are currently unreachable and return `Connection refused`.
- This is an external server-availability block, not a model, data, metric, or protocol failure. No GPU training, upload, or `locked_test` access occurred.
- Do not switch endpoints, broaden the run, or retry repeatedly without a reachable server; resume with the same fixed configuration once connectivity is restored.

## 2026-08-28 P4 reachability recheck finding

- A second read-only check confirms the external block persists on both recorded ports; local evidence remains internally consistent and the P4 CPU gate remains passed.
- No new model or data diagnosis is justified while the approved GPU execution host is unavailable. Keep the mainline paused at the server boundary.
## 2026-08-28 P4 GPU launch finding

- The remote environment has CUDA, but the background launcher does not inherit the interactive precheck `PYTHONPATH` unless it is exported inside the command.
- Evidence: the precheck imported `pi_jwm` successfully with explicit `PYTHONPATH`; the first background run failed immediately at the first project import and used 0% GPU.
- No training result is valid from that attempt. Keep the directory as failed process evidence and restart the same seed with one explicit environment fix.

## 2026-08-28 P4 second GPU seed launch finding

- The first `20260831` background attempt did not train: it exited before model import with `ModuleNotFoundError: No module named 'pi_jwm'`, and GPU usage stayed at 0%.
- The failure is fully explained by missing environment propagation. Import succeeds with `PYTHONPATH=/root/autodl-tmp/pi_jwm_p4_h20_epoch8_gpu_20260828/code/src` and fails without it.
- This attempt is invalid process evidence only; no model/data/protocol conclusion changes, and `locked_test` remains untouched.

## 2026-08-28 P4 second GPU seed completion finding

- After explicitly exporting the remote source path, seed `20260831` completed without errors and passed the required completion, CUDA, checkpoint-reload, and non-locked boundary markers.
- This confirms the previous failure was limited to launcher environment propagation. No training parameter, tensor contract, or scope was changed.

## 2026-08-29 P4 third GPU seed completion finding

- Seed `20260832` completed with the same required markers as the first two seeds: training complete, CUDA execution, checkpoint reload verified, and `locked_test_accessed=false`.
- All three runs share the fixed data manifest and protocol. No formal performance claim is made until local manifest recovery and the independent multi-seed audit pass.

## 2026-08-29 P4 GPU three-seed audit finding

- Local recovery and `formal_gpu_multiseed_audit_v1` passed the reproducibility/boundary checks: three seeds share the same contract, all manifests have zero mismatches, GPU execution is true, and no `locked_test` content was used.
- The learned model improves link activity and operational errors, but node-x MAE is worse than persistence for all three seeds (`+0.4621`, `+0.8483`, `+1.3708 m`). This keeps the P4 performance gate closed and prevents any final-method claim.
- The correct next step is a focused decision about the remaining node-state error, not another unapproved training sweep or a phase jump.

## 2026-08-29 P4 节点位置误差聚焦处理

- 当前唯一性能阻塞是验证集 node-x MAE 相对 persistence 变差；三个 GPU seed 均变差。
- 现有仓库已有逐 horizon rollout 审计入口，可直接复用，避免重复实现。
- 待验证的单一诊断问题：位置误差是否主要在长 horizon 累积，还是由位置变化量/节点类型/归一化或反馈路径造成。当前不预设答案。

## 2026-08-29 P4 节点位置误差诊断发现

- 只读 CPU 回放报告：`code/artifacts/audit/pi_jwm_p4_position_diagnosis_20260829/position_diagnosis.json`。
- 三个 seed 在第 1 步到第 20 步均显示预测位移不足；第 20 步真实 x 位移均值约 `22.145 m`，预测 x 位移为 `0.774/2.027/3.816 m`。
- 车辆、UAV、RSU 分组均有同方向欠预测，静态节点误差不是唯一来源。
- 单一主因：residual state head 的位移更新幅度过小，造成长 horizon 位置 rollout 欠预测；该结论由三个独立 seed 的同方向证据支持。
- 风险：这仍是诊断结论，不是修复验证；P4 继续 blocked，不能进入 P6，不能访问 `locked_test`。
- 下一步只允许一个最小 residual 位移幅度修正的 CPU 单变量验证；若未通过，停止并报告，不扩大实验。

## 2026-08-29 P4 residual 幅度三 seed CPU 门发现

- 三 seed 的 scale=1.0 CPU 结果通过既有独立 Go/No-Go：`gpu_allowed=true`、`failed_gates=[]`。
- node-x MAE ratio=`1.0851`，验证 link-F1 delta=`+0.0476/-0.0400/+0.0945`（均值 `+0.0340`）；吞吐量、RB 占用和任务时延均值相对 persistence 改善。
- 这说明幅度修正方向在三 seed 上具有稳定的 CPU 证据，但它仍只是 GPU 放行证据，不是最终方法冻结或正式性能声明。
- 继续边界：只允许同一配置的 non-locked GPU 三 seed；不访问 `locked_test`，不进入 P6，不扩大参数搜索。

## 2026-08-29 P4 residual 幅度单变量发现

- `residual_state_scale=1.0` 的 seed `20260830` CPU 运行完成，配置和数据契约与 scale=0.5 仅一处不同。
- 第 20 步 node-x MAE 为 `22.9194 m`，persistence 为 `22.1450 m`，比 scale=0.5 的 `23.8567 m` 更接近真实移动；位置比例约 `1.035`，低于原门 `1.25`。
- 这只是一个 seed 的方向性证据，不能替代三 seed Go/No-Go；link-F1、吞吐量等完整门仍未重算。
- 风险边界：不要把 scale=1.0 直接写成定版参数；在三 seed CPU 复核前，P4 仍 blocked，GPU 和 `locked_test` 继续关闭。
- 唯一下一步：按同一配置运行 seed `20260831`，再运行 `20260832`，最后用既有独立审查脚本重算门。

## 2026-08-29 P4 scale=1.0 GPU 审计发现

- 三个 non-locked GPU seed 已完成；`gpu_execution=true`、checkpoint reload verified、`locked_test_accessed=false`，结构审计通过且 manifest mismatch 为 0。
- validation link-F1 delta 为 `-0.0010/-0.0337/+0.0524`，均值 `+0.0059`；node-x MAE delta 为 `+0.2504/+0.8316/+3.3702 m`，均值 `+1.4841 m`。
- 吞吐量 MAE、RB occupancy MAE、task-delay MAE 三项均值相对 persistence 改善，但不能抵消节点位置误差失败。
- 结论：residual scale=1.0 通过了 GPU 执行/可复现性门，没有通过 P4 性能门；不能把该配置写成最终方法或进入 P6。
- 下一步只允许：根据已有位置欠预测诊断，先定义并审核一个新的、最小且可证据检验的处理方案；不重复同一 GPU 配置，不访问 `locked_test`。

## 2026-08-29 为什么项目长期卡在 P4

- P4 的科学性能门从未真正闭合：当前 scale=1.0 GPU 结构审计通过，但 node-x 三 seed 全部变差，link-F1 平均提升很小且存在 seed 回退。
- 最大返工源头是顺序错误：数据契约、规则层和指标单位问题在较晚阶段才发现；旧 GPU 结果因此只能降级，必须重建或重训。
- 第二个返工源头是门定义不够严格：CPU gate 只判断有限统计条件，GPU audit 主要判断可复现性和边界，二者都不是最终性能门。
- 第三个返工源头是没有停机规则：每次失败都追加一个局部改动，却没有先判断是数据、实现、指标还是模型能力问题。
- 第四个返工源头是远端执行不稳定：路径、环境变量和默认方法列表没有在启动前硬性检查。
- 第五个返工源头是记录治理不够单一：旧状态和当前状态并存，容易重复做已经做过的工作。
- 改进原则：critical mismatch 未闭合不训练；执行通过和性能通过分开报告；同一失败最多做一个预注册最小验证，验证不支持就停止并重判路线。

## 2026-08-29 P4 状态接力修复发现

- 新失败测试证明旧 residual 逻辑在第二步仍使用历史最后状态；固定每步残差时，旧实现第二步只得到 `last + 1`，而不是应有的 `last + 2`。
- 最小修复后，两步 node state 和 task DAG state 都按上一轮预测值累积，规则层输出也会作为下一步递推状态；没有改数据、统计、损失或训练协议。
- 代码回归测试 `9/9` 通过，说明接口层行为符合预期；这不是模型性能门。
- 新 CPU seed `20260830` 已完成并 reload；seed `20260831` 因本轮运行时间过长被停止，未产生有效结果，seed `20260832` 未运行。
- 当前风险：长 horizon 真实数据三 seed 的位置、链路和业务指标尚未重新测量；P4 仍 blocked，GPU 和 `locked_test` 继续关闭。

## 2026-08-29 P4 状态接力修复三 seed CPU 门发现

- 三个固定 CPU seed 均已完成并通过 checkpoint reload，运行边界为 CPU-only、non-locked，未读取或生成 `locked_test`。
- 独立门审查报告为 `code/artifacts/audit/pi_jwm_p4_recursive_cpu_go_no_go_20260829/cpu_to_gpu_gate.json`；结构、位置、校准和运营指标门通过。
- 唯一失败是逐 seed validation link-F1 稳定性：`+0.1032/-0.0941/+0.2013`，其中 `20260831` 低于 `-0.05` 上限；均值虽为 `+0.0701`，不能掩盖单 seed 回退。
- 这说明状态接力修复已能完成三 seed CPU 运行，但链路活动预测仍有 seed 波动；不能放行 GPU、不能关闭 P4，也不能进入 P6。
- 下一步只允许基于已有证据分析该单一失败门，禁止继续叠加模型修复或扩大调参。

## 2026-08-30 `_CONTEXT.md` 交接审阅初始发现

- 现有过程记录的最新结论是 P4 仍 blocked；代码级状态接力修复已通过定向测试，但三 seed CPU Go/No-Go 因一个 seed 的 validation link-F1 回退而未放行 GPU。
- 该结论目前仅是交接审阅的起点，尚未与 `PROJECT/_CONTEXT.md`、权威记录、Git 工作区和机器可读审计完成交叉核验。
- 本次不启动 GPU、不读取 `locked_test`，也不把“执行链完成”写成“科研性能门通过”。
- 用户给出的 `PROJECT\_CONTEXT.md` 实际不存在，且 `PROJECT` 目录不存在；精确搜索仅找到根目录 `PROJECT_CONTEXT.md`，文件大小 33,221 bytes，修改时间为 2026-08-30 19:36:27，故将其作为本次交接源。
- Git 当前为 `main@0630515`，工作区有大量已修改和未跟踪的代码、测试、记录与脚本；它们是交接时必须保留的 active work，不能按干净提交状态理解。
- `PROJECT_CONTEXT.md` 明确把当前阶段限定为 P4：三 seed CPU 训练、验证、校准和 checkpoint reload 已完成，但最新 Go/No-Go 为 `gpu_allowed=false`；报告唯一失败项是逐 seed validation link-F1 回退。
- 直接阻塞数值是 seed `20260831` 的 validation link-F1 相对 persistence 为约 `-0.0941`，超过允许回退 `-0.05`；三 seed 平均提升不能掩盖单 seed 失败。
- 另一未闭合证据是逐步位置表现：当前 gate 的 node-x ratio `1.1869 <= 1.25` 是合并所有预测步的结果，不等于用户原计划要求的第 1/5/10/20 步长期位置门。
- 精确下一步只允许读取三 seed 指标、threshold selection、gate 定义和既有诊断，形成判断；如需改模型、loss、阈值协议、数据、预算或新实验，必须另行说明并确认。
- `记录/本地计划表.md` 第 1005--1010 行与 `记录/8.12之后推进.md` 第 681--687 行都把最新状态覆盖为 recursive CPU gate blocked；`记录/PIJWM主文档.md` 的 P4 进度只更新到 8 月 27 日，不能单独覆盖 8 月 29 日机器结果。
- `PIJWM主文档.md` 顶部当前统一命名明确禁止把现代码称为完整 RSSM；文档其他位置的 RSSM 历史审计语句不能越过该当前口径。
- 最新 gate JSON 的字段、数值、失败项和 SHA-256 与交接文件一致；这是当前状态最直接的机器证据。
- 三个 recursive CPU seed 的配置只有训练随机 seed 不同；共同 `data_seed=20260823`、manifest SHA-256 `d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`、CPU、h20、8 epochs、`residual_state_scale=1.0`、确定性规则层开启。
- 三个运行均为 `training_run_complete=true`、checkpoint reload verified、`gpu_execution=false`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- calibration 选择的 link threshold 为 `0.7/0.9/0.7`；该差异是后续诊断输入，当前不足以单独证明 seed 回退根因。
- gate 代码第 55--56 行对 `min(validation_deltas) < -0.05` 直接阻断，因此三 seed 平均为正不能覆盖 seed `20260831` 的失败。
- 当前模型代码中 `recursive_states` 和 `recursive_dag_state` 均在每步末尾更新；定向测试存在两步递推断言。这证明接力机制已实现，不证明真实长期精度通过。
- 三个 seed 的 `sample_ids.json` SHA-256 均为 `F9ED48D1AE1B532341FEEFA3E093AA44D0DF3FBFAECC94D4DA3EF0273744198C`，因此当前 link-F1 差异不是 validation 样本集合不同造成的。
- fresh 定向验证通过，但验证范围有限：世界模型 9 项、gate 2 项、compileall、diff check；未运行完整测试、AirFogSim、GPU、planner 全链或 `locked_test`。
- 交接结论：当前继续停在 P4。已完成证据是递推机制、三 seed CPU 运行/reload 和 gate 复算；阻塞是单 seed link-F1 回退及逐步位置证据缺口；唯一下一步是证据化诊断，不是训练。

## 2026-08-30 成本路由与零重复实验规则

- 用户要求后续大计划默认使用 `cost-aware-model-routing`，优先利用 Luna/Terra 的速度和成本优势，Sol 负责高判断任务和最终验收。
- 用户授权根据真实使用结果持续改进该技能，但改动必须由重复出现的路由问题或明确证据支持，不能因单次偶然快慢过拟合规则。
- 无技能压力测试暴露两类风险：一次安排 Terra 加两个 Luna 并行，容易过度委派；仅观察一次 Luna 更快就立即固化长期降级规则，存在单样本过拟合。
- PI-JWM 特定硬规则：先查历史证据和失败记录；无新增变量和新问题不得重跑；历史成功结果必须绑定配置/数据/seed/指标/哈希并受回归保护；历史失败也必须保留，不能通过覆盖或改名消失。
- 当前 P4 只允许复用现有证据诊断 link-F1 seed 不稳定性；任何模型、loss、threshold、数据、预算或实验变更都不在本轮授权内。
- 技能 GREEN 前向测试通过：读取修改后技能的 Terra 拒绝根据单次 40 秒结果改长期规则，改为记录候选路由证据并等待多个独立同类任务；项目门继续留在项目记录中。
- 技能经压缩后 534 词，`quick_validate.py` 返回 `Skill is valid!`；保留了实际模型委派、最多两个工作者、复用优先、证据驱动改进和协调者验收。
- 本轮实际路由：Luna 负责历史 artifact 去重清单；Terra 负责当前 recursive 三 seed 与历史诊断对比；两者均只读，Sol 不重复其机械扫描，只复核关键原始字段。

## 2026-08-30 P4 link-F1 去重清单与当前证据缺口

- Luna 去重确认：旧 link activity diagnosis 曾发现小样本 seed 排序反转；expanded failure diagnosis 后三 seed ROC-AUC 约 0.991，阈值迁移成为主要观察；RB 单位错误和 throughput 偏差也已分别诊断。这些都在状态接力修复前，不能直接解释最新 recursive checkpoint。
- Terra 对当前 JSON 的比较确认：三 seed 配置、样本、参数量和协议一致；`20260831` calibration 选阈值为 `0.9`，另两 seed 为 `0.7`；其 calibration F1/AUPRC 较弱，但 validation k=20 AUPRC 没有明显崩坏。
- Sol 复核原始字段得到 validation k=20 AUPRC `0.5601/0.6920/0.7136`、calibration k=20 AUPRC `0.2691/0.2340/0.3273`，与 Terra 一致。
- 当前目录没有 raw score/logit/prediction 文件，现有 JSON 不足以分离阈值迁移、排序和分数尺度的相对贡献。
- 现有 `run_formal_p4_link_activity_diagnosis_v1.py` 已实现 CPU-only 的 ROC-AUC、AP、正负分数分位数和阈值网格；但其 glob 只匹配旧目录名，当前可用临时 junction 复用而无需改代码。
- 修复后 checkpoint 的首次只读诊断已完成：`code/artifacts/audit/pi_jwm_p4_recursive_link_activity_diagnosis_20260830/link_activity_diagnosis.json`，SHA-256 `BF94979D9A761EE440E0EC39EFCCAB6D9056F200B82455ECD11E8E38E9897F01`；3 runs、CPU、GPU=false、locked-test=false。
- `20260831` validation 在冻结阈值网格中的最佳阈值仍为 `0.9`，最佳 F1=`0.4960`，低于 persistence `0.5901`；所以 calibration threshold 迁移不能单独解释失败。
- 在 threshold `0.9`，`20260831` validation 为 TP=`4594`、FP=`6175`、FN=`3163`；另两 seed 的 FP 为 `2186/1414`。当前直接失败机制是高置信假阳性显著增多，即顶部正负分离不稳定。

## 2026-08-30 P4 link-F1 逐步证据化判断

- validation 第 1/5/20 步的 `AUPRC/F1/precision/recall`：seed `20260830` 为 `0.9167/0/NA/0`、`0.9911/0.9498/0.9634/0.9365`、`0.5601/0.4525/0.2959/0.9604`；seed `20260831` 为 `0.8340/0/NA/0`、`0.7131/0.3263/0.6693/0.2157`、`0.6920/0.3802/0.2388/0.9314`；seed `20260832` 为 `0.8528/0.7235/0.9858/0.5714`、`0.9792/0.9050/0.9423/0.8706`、`0.7136/0.6673/0.5224/0.9235`。
- calibration 同口径下，seed `20260831` 第 1/5/20 步为 `0.2216/0/NA/0`、`0.4918/0.4201/0.4765/0.3757`、`0.2340/0.1944/0.1092/0.8818`。第 20 步 TP/FP/FN=`179/1460/24`，说明长步高置信假阳性在 calibration 内部也存在，不是 validation 阈值迁移单独造成。
- seed `20260831` validation 在同一冻结阈值 `0.9` 下，第 1/5/20 步 TP/FP/FN=`0/0/364`、`85/42/309`、`353/1125/26`：行为从短步漏报转为长步大量误报，直接指向 recursive rollout 过程中的分数尾部漂移。
- 全步诊断的 negative q90 为 `0.2420/0.2823/0.2992`，失败 seed 并不最高；overall q99 为 `0.7883/0.9229/0.6859`，threshold `0.9` 的 FP 为 `2186/6175/1414`。因此异常集中在极高分尾部，而不是整体负样本分数普遍抬升。禁止读取不存在的 `negative_score_quantiles.q99`；该字段不存在，PowerShell 空值转 `0` 不是证据。
- 直接失败机制已闭合到“长步递推放大高分负样本尾部”，但底层根因尚未闭合到具体的输入反馈、head、loss 或训练动态。按 systematic-debugging 规则，当前不提出模型、loss、阈值或训练修复。

## 2026-08-30 P4 link-F1 底层只读审计

- `link_activity_head` 直接读取每步的 physical-edge latent；该 latent 每步经过 edge GRU 更新，第 2 步起还接收 deterministic rule 产生的 physical-edge state feedback。因此“递推 edge latent 漂移”是与长步假阳性机制一致的候选路径，但当前仍是候选，不是已证根因。
- 三 seed 共用 train-only link `pos_weight=50`、`sparse_event=0.1` 和完全相同的样本；训练历史只记录 total train/validation loss，三个 seed 的 best epoch 均为 8，无法从现有曲线读取逐 horizon link loss。
- checkpoint 参数只读对比：seed `20260831` 的 link-head bias=`+0.1450`，另两 seed为 `-0.1650/-0.1526`；link-head weight L2=`0.5504/0.6372/0.7194`，edge GRU weight L2 约均为 `5.8--5.9`，physical-edge feedback weight L2=`3.3790/3.3568/3.4617`。没有参数范数爆炸证据。
- 正 bias 可能贡献分数上移，但约 `0.3` logit 的 seed 间差异不能单独解释阈值 `0.9` 以上 FP 的三倍差距。最小缺失证据是逐 horizon 负样本 logit 分位数，以及从 logit 中减去 head bias 后的对应分位数；获得前不能把根因写成 head bias 或 edge feedback。

## 2026-08-30 P4 严格收口计划发现

- Luna 的只读矩阵正确识别了 link-F1 和逐步位置两个剩余门，但误引用了 2026-08-26 旧 tensor 入口。Sol 用当前三个 run 的 `config.json` 纠正为 `pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827`，manifest SHA-256=`d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`。
- Terra 的诊断设计确认代数边界：`raw_logit = w·edge_latent + bias`，固定 bias 不能制造 h1 -> h20 的增量；逐 horizon `raw_logit-bias` 能判断 latent 在 head 方向是否漂移，但不能进一步未经干预就指认 GRU、图消息或规则反馈。
- 旧计划中的 CPU 三 seed 性能训练不再作为下一轮执行方式。按用户新要求和 pre-GPU hard gate，CPU 只保留极小 reload micro-smoke；正式训练转为用户开启 GPU 后的 sentinel `20260831` -> 其余两 seed，不做 sweep。
- P4 最终验收必须同时关闭：每 seed validation link-F1 回退、calibration link 优势、聚合 node-x、1/5/10/20 位置、运营指标、uncertainty/mask、manifest/reload 和 non-locked 边界。不能只修 link-F1 就宣布 P4 完成。

## 2026-08-30 P4 v2 bias/latent 诊断发现

- Terra 在冻结规格内只新增诊断脚本和测试，TDD RED/GREEN 证据完整；Sol 两阶段复核未发现范围扩张或评价口径漂移。本次单次成功不足以修改长期模型路由规则，只作为后续路由样本保留。
- v2 与 v1 在固定 `0.9` 阈值下的 validation/calibration 六组 raw FP 总数完全相同：`2186/2634`、`6175/8579`、`1414/1878`，证明分 horizon 重放没有改变原评价对象。
- seed `20260831` 正 bias=`+0.1450`；去 bias 后 validation 总 FP 只减少 `436`，h20 只减少 `39`。因此 bias 是放大因素，但不能直接解释大部分失败。
- 同一 seed 去 bias 后 validation q99 在 h1/h5/h20 为 `-0.19/-0.34/4.96`，q95 为 `-0.57/-1.19/1.18`；尾部不是从第一步单调上升，而是在后段递推中急剧抬高。准确表述应是“长 horizon edge latent 在 link-head 方向漂移”，不是笼统的全分布上移。
- forward 代码证明 link logit 在每步由更新后的 `edge` 经过同一线性 head 产生；loss 对真实 mask 使用同一 aggregate activity 加权 BCE，评价阈值由 calibration 的固定网格选择。未发现 v2 与模型/loss/metric 的关键口径冲突。
- 目前仍不能区分 direct physical-edge rule feedback、edge GRU/physical message 或 cross-flow coupling。最小下一诊断只干预 direct physical-edge rule-feedback 投影，其他路径全部保持不变；结果只判定该直接路径的因果贡献，不外推到整个规则层。

## 2026-08-30 direct physical-edge rule-feedback 因果路径闭合

- 干预保持 checkpoint、样本、head bias、edge GRU、physical message、cross-flow coupling 及 node/flow/task feedback 不变，只把 `state_feedback['physical_edge'](...)` 的输出临时置零。
- h1 基线/干预完全一致，证明 hook 没有提前改变无 rule-feedback 的第一步；原 baseline 逐 horizon negative count/raw FP 与冻结 v2 报告完全一致。
- h20 去 bias FP `1086 -> 0`，pre-bias q95 `1.18 -> -0.16`、q99 `4.96 -> -0.16`；全 20 步 raw FP `6175 -> 0`。按预注册条件，该直接路径对观测到的高分尾部具有充分解释力。
- 理论仍要求规则修正后的最终状态进入下一步，故完全关闭这条反馈会造成理论--实现不一致，且本报告没有正样本/召回保护证据，不能作为修复。
- 最窄的理论一致候选是把 physical-edge correction projection 从“直接加到 recurrent hidden state”改为“作为 edge GRU 的输入消息”，让 GRU 门控接收规则修正，同时避免无门控式 hidden 注入。该候选尚未获用户确认、尚未实现或验证。

## 2026-08-30 physical-edge feedback 输入路由实现发现

- 用户已确认方案 A，候选已实现并通过接口门。精确语义是：规则 correction 仍进入下一步 edge transition，但作为 GRU input message，由 GRU 门控处理；它不再预先改写 recurrent hidden。
- RED 测试证明旧行为真实存在而非命名差异；GREEN 证明新行为的 h2 hidden 不受直接注入，normal/suppressed 的 GRU input 差精确等于非零 projection。
- 本次未改参数模块，原 seed `20260831` checkpoint strict load 后 missing/unexpected keys=`0/0`，参数量保持 `83750`。
- 旧 CPU smoke runner 无 deterministic rule layer 且无 strict reload，不能验证本次改动；复用正式训练模块的 CPU 内部接口才与理论和实现一致。
- `2/1/1`、1 epoch micro-smoke 只证明接口可训练、保存、strict reload 和指标 JSON 有限，不提供任何性能结论。P4 是否修复仍需 GPU sentinel 和固定门验证。

## 2026-08-31 GPU sentinel 启动发现

- 远端默认环境没有 `python` 命令，但 `/root/miniconda3/bin/python` 为 Python 3.12.3，PyTorch `2.8.0+cu128` 且 CUDA 可用；必须在命令中显式使用该路径。
- 历史 GPU 流程可复用的是远端隔离、显式 `PYTHONPATH`、单方法和 manifest 审计；历史 residual scale `0.5` 不适用于当前已冻结 `1.0` 协议。
- 远端已有相同 canonical tensor manifest，可服务器内复制以节省上传；旧 checkpoint、旧 run summary 和旧训练结果均不复用。
- 单 seed 不能运行现有三-seed gate；sentinel 完成后必须直接读取 config/summary/runtime/manifest/comparison/class_weights/threshold report 做预注册字段判断，通过才授权另外两个 seed。

## 2026-08-31 GPU sentinel 结果发现

- 方案 A 没有破坏 node-x、throughput、RB occupancy 或 task-delay 门，但没有解决失败 seed 的 validation link-F1：delta 从冻结门角度仍为 `-0.2177`，因此不能继续三 seed。
- calibration 选择的 link threshold 仍为 `0.9`；validation candidate F1=`0.3724`，低于 persistence `0.5901`。当前只知道“性能未修复”，尚不能从 aggregate F1 指认新旧具体机制。
- sample IDs 的 parsed JSON 完全相同；raw hash 差异是跨平台换行，不是样本漂移。未来跨 Windows/Linux 冻结样本身份应使用 canonical JSON hash或逐 ID 对比，不能只比较原始字节哈希。
- 下一诊断的唯一新增问题是：GRU-input routing 后，新 checkpoint 是否仍出现长步高置信假阳性尾部。既有旧 checkpoint 诊断不能直接替代该回答。

## 2026-08-31 新 checkpoint 阈值迁移判定

- 新 checkpoint 已修掉旧的长步高置信 FP 尾部，但冻结 `0.9` 下错误转为 FN 主导：validation `TP/FP/FN=1902/557/5855`，precision=`0.7735`、recall=`0.2452`、F1=`0.3724`。
- 只读候选回放显示 `0.7` 可把 recall 提到约 `0.9309`，但同时产生 `16600` 个 FP，precision 只有约 `0.3031`，F1=`0.4573`；因此简单降低阈值只是用大量 FP 换回 recall，仍打不过 persistence。
- persistence 在同一 validation 汇总上 `TP/FP/FN=8944/5856/6570`，precision=`0.6043`、recall=`0.5765`、F1=`0.5901`。候选最优 `0.7` 与它的 F1 delta=`-0.1327`，仍低于冻结允许值 `-0.05`。
- 准确边界：可以说 calibration 到 validation 存在阈值迁移影响；不能说 No-Go 只由 `0.9` 导致，也不能用 validation 事后选择 `0.7` 作为新正式阈值。
- 当前问题已从“旧反馈路径造成长步高置信 FP”转为“现有 link score 与冻结候选阈值之间没有满足 persistence 门的工作点”。这是方法/指标协议问题，下一步必须先做理论--实现--指标一致决策，不能直接叠加第二个代码修复。

## 2026-08-31 link score/threshold 计划审计发现

- 当前训练对 link activity 使用 train-only `pos_weight=50.0` 的 weighted BCE；正式 threshold selection 对 raw sigmoid score 直接在 calibration 上选 `0.1/0.3/0.5/0.7/0.9`。
- 代码库已有 weighted-score 到 unweighted posterior 的数学修正 utility 和测试，但当前正式 GPU runner 的 threshold path 未接入该 utility。
- 主文档同时要求 link activity 为离散事件概率、类别加权后仍报告概率校准、calibration split 只确定概率阈值。因此必须先判断 raw weighted score 的方法语义，不能一边称概率、一边按未校准 decision score 使用。
- 历史 threshold audit 只能证明 correction 机制和旧 checkpoint 现象，不能当作当前 sentinel 性能。
- 路由复核再次证明嵌套 metrics 必须读取 `thresholds.link_activity`，不能用顶层默认 `threshold` 替代；当前有效 link threshold 已 fresh 核验为 `0.9`。

## 2026-08-31 link score/threshold 方法判定

- `protocol_coherent_candidate_failed` 被排除：主文档没有把 raw weighted sigmoid 定义为纯 decision score，而是明确要求事件概率与概率校准。
- `evidence_incomplete` 被排除：checkpoint/sample IDs/tensor、mask、aggregate、effective threshold、overall F1 和 persistence 均可核验。
- 当前状态为 `protocol_theory_mismatch`：weighted BCE 合法，但 raw sigmoid 在正式链中既未做概率语义修正/校准，也未报告 Brier/ECE，不能称为已经校准的事件概率。
- 性能边界不变：正式 `0.9` F1=`0.3724`；只读候选最优 `0.7` F1=`0.4573`；persistence=`0.5901`。No-Go 继续成立。
- 唯一推荐不是立即接入某个 correction，而是先冻结一个 post-training probability calibration boundary 设计对象；具体数学与接口只能在用户确认后单独设计，不能同时试多条路线。

## 2026-08-31 方案 B 设计发现

- weighted BCE 的理论赔率偏移可用 `z-log(pos_weight)` 固定反演；随后只拟合一个正 scalar temperature，既保留类别不平衡训练，也符合主文档对 temperature scaling 的候选口径。
- 将原 raw thresholds 通过同一严格单调函数映射到 probability thresholds，可以逐元素保持全部分类决策；因此不会返工方案 A、checkpoint、tensor 或 sentinel，但也不会改善现有 F1。
- 方案 B 能关闭的是 link probability 语义、calibration split、Brier/ECE/NLL 的一致性缺口；当前 P4 仍被 sentinel link-F1 性能门阻止。
- 如果 validation 被用于拟合温度、挑选方案或重新使用数值 `0.1/0.3/0.5/0.7/0.9`，就会变成新协议或事后调参，必须拒绝。

## 2026-08-31 方案 B 实施分解发现

- 最小可靠实现需要把概率数学、metric 消费、runner split 所有权和当前 sentinel audit 分成四个清晰接口；若只接 utility 而不进入 runner，仍不能关闭理论--实现链。
- 当前 sentinel 只需 CPU 只读 inference；训练、GPU 和 follow-up seeds 都不是概率语义门的必要条件。
- 为保护历史消费者，runner 的 legacy raw threshold identity 与正式 probability threshold 必须分别保存，不能复用一个含糊的 `threshold` 字段。

## 2026-08-31 方案 B Task 1 实现发现

- 仅有 happy-path 数学测试不足以证明概率边界可靠；极端 log-temperature、复数 tensor、非有限 class weight 和极小正权重都会暴露静默污染风险。
- `correct_positive_weighted_probability` 的分母对 `score∈[0,1]`、`w>0` 理论上严格为正；旧 `clamp_min(1e-12)` 会错误改变极小正权重的精确反演，现已由回归测试保护。
- raw threshold 到 probability threshold 必须使用与概率预测完全相同的 torch float64 路径，否则临界点舍入可能破坏逐元素 decision equivalence。

## 2026-08-31 方案 B Task 2 实现发现

- 只在 metadata 写“event_probability”不够；实际 `thresholds.link_activity` 必须是 mapped 值，legacy raw identity 必须单独保留。
- 最可靠接口是让调用方只传 legacy raw threshold，由 accumulator 内部调用同一 calibration 对象映射，消除外部双阈值和 tolerance 分叉。
- 浮点等值点上，形式上单调的 float64 threshold 与 float32 probability 仍可能翻转；分类统计必须显式复用同 dtype/device 的 legacy decision，概率指标继续使用正式概率。
- 既有 throughput/RB 全 mask 计数问题与本任务无关；本轮未顺手修改，以免主线扩张。

## 2026-09-01 方案 B Task 3 实现发现

- learned calibration 缺少有效样本或只有单类标签时，拟合出的 temperature 可能极端但看似有效；正式 runner 必须在拟合前要求正负类同时存在。
- RULE baseline 没有概率校准，不能因为共用 threshold report schema 就写 `probability_threshold`；raw/probability 字段必须按方法身份分开。
- 正式 runner 若边训练边写最终目录，后段校准失败会留下可能污染下次 manifest 的半成品；同级 staging + 成功一次发布是必要的证据完整性门。

## 2026-09-01 方案 B Task 4 实现发现

- float32 阈值等值点证明“数学映射严格单调”不等于有限精度下直接 probability comparison 必然逐元素相同；artifact 必须区分 direct comparison 与正式分类复用 legacy decision，不能用改写 mapped threshold 制造等价。
- provenance 不能只记录非空 SHA 和正 mask 数；mask count 必须绑定实际有效元素，canonical tensor manifest、共同元文件和 calibration/validation 所选 tensor 都必须在读取前核验。
- strict reload 的成功路径不足以证明严格性；缺 key、unexpected key、错误 model contract、错误 seed 与 sample ID 漂移都需要独立失败测试。
- 当前只证明 Task 4 audit 入口达到规格和代码质量门；真实 temperature、NLL/Brier/ECE、TP/FP/FN、manifest 与 No-Go 保留情况仍须由 Task 5 的冻结 CPU audit 决定。

## 2026-09-01 方案 B Task 5 真实证据发现

- 单元测试和接口审查通过不等于真实概率门通过；冻结 sentinel 上，calibration-only scalar temperature 使 validation NLL 相对 `T=1` 解析反演基线恶化，停止门实际触发。
- 当前可证结论只到：方案 B 的实现和审计入口可执行，但冻结的“解析反演 + 单温度”方法未通过预注册的 validation 泛化要求，`probability semantics gate = failed`。
- 因运行在 artifact 发布前停止，不能事后补写 temperature、Brier/ECE、TP/FP/FN 或 manifest 数值，也不能把旧 sentinel 数值冒充本次 probability audit 产物。
- 不允许用 validation 重新选温度、换 calibrator、放宽非恶化门或重复运行制造通过；下一方法动作必须是用户批准的独立单变量决策。

## 2026-09-04 P4 link低召回诊断发现

- 低召回的主量来自持续活跃链路，而不是新激活链路：持续组占正样本`6050/7757`，且persistence正确而candidate漏报的`2791/2806`来自持续组。
- candidate逐步recall不是单调下降，而是h1为零、中段升高、h20再次接近零；这排除了“只看overall recall就直接换阈值”的解释。
- 同一个link head用于全部20步，horizon形状必须来自edge latent轨迹；h1又没有上一预测步rule feedback，因此rule feedback不能单独解释全部现象。
- 最小可证伪下一步是只旁路`edge_transition`，保留其他计算并检查历史edge记忆能否找回主要漏报；该干预不是正式修复，不能把结果当作模型性能。
- 真实诊断保持`gpu_execution=false`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`，P4仍blocked。

## 2026-09-05 edge GRU旁路干预发现

- 让`edge_transition`直接返回进入GRU前的hidden state后，所有有效链路的raw score都未越过冻结阈值，candidate TP/FP/FN从`1902/557/5855`变为`0/0/7757`。
- 原始2,791个“持续活跃、persistence正确、candidate漏报”样本没有任何一个被找回；持续活跃recall从`0.2648`降为`0`，h1/h20均仍为`0`。
- 因此当前证据反对“GRU更新单独擦除了历史活跃记忆”这一解释；edge GRU更新反而是现checkpoint产生任何高于0.9链路分数的必要步骤。
- 该结果不证明edge GRU完全正确，也不证明CFE、rule feedback或link head错误；旁路同时阻断所有消息经GRU写入edge latent，只能否定本次充分原因假设，不能从失败结果跳到另一个模块结论。
- 后续若继续，应先设计一个更细的只读机制核验，把GRU门控/输入消息/hidden贡献拆成一个新的、单变量且可证伪的问题；未经用户确认不得执行。

## 2026-09-05 edge GRU接口只读追踪发现

- 冻结基线仍为TP/FP/FN=`1902/557/5855`。在5855个FN中，5559个是`pre_negative -> post_negative`，296个是`pre_positive -> post_negative`；即进入本步GRU前已偏低占`94.94%`，本步向下跨阈值占`5.06%`。
- h1的364个FN全部在更新前已低于阈值；h20的375个FN中370个如此。因此“本步edge GRU更新主导向下压分”被数据否定。
- h2以后的`hidden_before`是上一步`hidden_after`的递推结果，所以incoming主导不能被误读成history encoder单独失败；它只说明低分在当前步之前已形成并被递推带入。
- 数据--代码静态核对：`aggregate_link_activity` 精确定义为`physical_edge_state.active_task_count > 0`与mask的交；`active_task_count`在tensor contract的五个物理边特征中，而模型初始边编码消费完整`physical_edge_state`。因此不支持“训练输入没有链路活动信息”。
- 模型forward没有把`history.aggregate_link_activity`作为独立的二值递推状态，而是由edge hidden经link head直接预测绝对活动事件。这是一个可设计的候选方法边界，但当前证据尚不能宣称它已是根因或已验证修复。
- 下一步不再追个旁路旧模块，只允许先形成一份`candidate method`：用上一步链路活动作为持久性基准，学习action-conditioned变化量；必须先证明h2+只使用模型自身前一步预测而非真值target，并与weighted BCE和方案B概率语义一致。用户确认前不实现。

## 2026-09-05 link activity持久性残差候选设计发现

- 当前可复用的`last_persistence`会把历史最后一帧活动重复到20步，并用有限logit `+20/-20`表达确定性活动/非活动；因此候选可在零残差时精确复现该基线，而不另造标签或阈值。
- 唯一推荐参数化是：现有单路link head输出未加权事件log-odds变化量；h1以历史最后一帧的持久性logit为基准，h2-h20累加模型自身上一预测的未加权logit；正式输出再加`log(50)`供现有weighted BCE和legacy raw decision使用。
- 方案B的`weighted_logit-log(pos_weight)`会回到候选内部事件logit；temperature仍只能在训练后用calibration拟合，不能反馈到20步递推。
- 双路hazard虽有清晰转移语义，但新增输出参数且现证据不支持这份自由度；简单bias若使用硬阈值会把评价阈值塞进动力学，若改成软log-odds递推则等价于推荐方案。
- 旧checkpoint的参数形状可能仍能加载，但旧head是绝对logit、新head是变化量，语义不兼容；必须用方法/schema身份拒绝续训和正式评价，不能把`missing/unexpected keys=0/0`当兼容证据。
- 主要风险是20步log-odds变化量累积导致过度自信或漂移；只能由未来一次冻结sentinel证伪，不能失败后调持久性幅度或换第二方案补救。

## 2026-09-05 link activity持久性残差CPU实现发现

- 观测历史必须先从已有加权raw logit `+20/-20`减去`log(pos_weight)`得到内部`u0`；每步累加head输出的变化量，正式输出再加回`log(pos_weight)`。直接把`+20/-20`当内部`u`会破坏既有概率坐标。
- 新link机制必须挂在原`coupled_dual_gnn_residual`候选上；若注册为非残差模型，会在修link的同时改变其他状态头，违反单变量边界。
- 只核对`link_activity_method`不足以拒绝语义错误checkpoint；新方法还必须核对`mode=coupled_dual_gnn`和`residual_state_prediction=true`。
- canonical CPU micro证明接口、递推、规则反馈、保存和复载闭合，但不证明召回或概率质量改善；GPU sentinel仍是唯一性能证伪门。
- CPU门后提供的GPU端点在SSH握手前拒绝连接，因此本轮没有GPU执行事实；不能用历史GPU可用性或CPU micro代替新的sentinel。

## 2026-09-05 项目评估发现

- 当前方案B独立审计仍只支持旧`coupled_dual_gnn_residual`并硬锁raw threshold=0.9；新方法已注册/可复载不等于验收入口已支持。拒绝新身份的最小调用已复现，不涉及数据推理。
- 正式runner在阈值选择过程中拟合温度，再计算性能；需在后续入口闭合时对齐“先性能后概率”的书面顺序。没有据此发现validation参与拟合，也未改算法。
- 零delta复制persistence是条件契约；正式link head仍默认Linear初始化，zero-init残差状态头选项不等于link head置零。正负20起点可能阻碍状态切换，属于待sentinel检验的风险，不能表述为已证实失败。
- 当前持久性残差方向有针对性，应保持单变量；先补验收衔接，再做冻结单seed，不通过重复诊断、扩网格或新数据源掩盖P4未闭合。
- 详见`记录/研究进展/2026-09-05-项目现状与主线推进评估.md`；GPU/locked_test边界未改变。

## 2026-09-05 P4 验收入口修复发现

- 旧概率审计的硬编码已收缩为两种明确方法：旧`coupled_dual_gnn_residual`继续固定raw threshold=`0.9`；新`link_activity_persistence_residual_v1`从calibration threshold-selection产物读取冻结候选中的所选阈值。
- 新方法必须在run manifest中绑定threshold-selection文件，且推理使用对应方法身份；未知方法、错误坐标、非calibration选择、候选集外阈值和manifest漂移均拒绝。
- 这只是验收入口修复，不是概率门通过；GPU性能结果和新方法真实校准结果仍未知。
- GPU TCP检查失败，补查历史端口`14507`也失败；未上传、未训练、未访问locked_test，继续保持P4 blocked。

## 2026-09-05 P4 GPU sentinel 参数审计发现

- 启动后复核runner参数发现首次命令未包含样本上限，不能把全量窗口运行误写成`train/validation/calibration=256/128/128`。
- 证据边界已明确：偏差进程已停止，远端staging仅作失败追溯；不读取其指标、不运行方案B、不启动其他seed。
- 正确冻结命令必须显式传入`--train-limit 256 --evaluation-limit 128`；这仍是唯一下一动作，模型和实验协议不变。

## 2026-09-05 P4 persistence residual GPU sentinel evidence

- 远端retry按正确样本上限完成GPU训练；run摘要为`training_run_complete=true`、`gpu_execution=true`、`locked_test_accessed=false`，本地manifest核验`0 mismatch`。
- validation link-F1 candidate/persistence=`0.4801978088/0.5900903873`，delta=`-0.1098925785`，违反冻结下限`-0.05`。
- node-x candidate/persistence=`14.4161/11.8520`，ratio约`1.2163`；吞吐、RB occupancy、task delay未回退。
- 结论：sentinel performance gate=`no_go`，不执行概率门、不启动其他seed，P4继续blocked。

## 2026-09-05 完整 RSSM 论文与实现发现

- PlaNet 和 Dreamer 的关键可迁移条件不是“有一个随机层”，而是 deterministic state、action-conditioned prior、observation-conditioned posterior、训练 posterior 与部署 prior-only 的明确分工；PlaNet 还要求多步 latent supervision。
- 旧 `_GraphRSSMBackend` 的 context prior 和 posterior 都直接从同一个 `base_belief.joint` 计算；旧目标只计算 context KL，故名称可以保留为历史候选，但证据不足以支持完整 RSSM 结论。
- RDR 论文提示 teacher-forcing 与自由滚动分布不一致是长程误差来源；新候选把 prior rollout 与逐步 posterior 均值的一致性显式记录为辅助项。该项是 PI-JWM 适配，不等同于已证明 RDR 效果。
- 新 `complete_graph_rssm_v1` 使用 target 只构造训练 posterior；预测输出和 rollout prior 不读取 target。契约测试确认目标扰动不改变 prior-only 预测，且 KL 梯度到达 transition/prior/posterior。
- 完整性接口已闭合，但没有真实数据性能证据；不应把 4/4 测试、h20 finite 或论文机制直接写成性能提升。
- 当前 P4 仍 blocked；GPU 未使用，`locked_test_accessed=false`。下一步只做独立 CPU micro runner 和证据绑定，不扩展其它论文组件或参数网格。

## 2026-09-05 完整 RSSM canonical CPU preflight

- 首次运行将 target-conditioned posterior 辅助张量纳入整体输出等值比较，错误触发 `future_target_leakage_absent`；验收器已修正为只比较部署可见输出和 prior 分布，posterior 教师张量单独检查有限性与梯度。
- 修正后新候选 `complete_graph_rssm_v1` 在 train/validation/calibration 的 h1/h5/h20 共 9 个 canonical 窗口通过：训练步、有限梯度、非零梯度、全 rollout 有限、动作条件、target 泄漏、strict checkpoint roundtrip 均为 true。
- 产物：`code/artifacts/preflight/pi_jwm_complete_rssm_cpu_preflight_20260905_retry/`，`r4_cpu_preflight_ready=true`，`gpu_screening_ready=false`，`locked_test_accessed=false`。
- 该门只证明完整实现可在真实冻结数据上执行，不证明收敛、性能提升或最终方法选择；P4 仍 blocked，GPU 未启动。

## 2026-09-05 完整 RSSM teacher reconstruction 审查发现

- PlaNet/Dreamer/VGRNN 机制对照表明，posterior 除了 KL 还必须参与训练期观测或状态重构；否则只能称部分 variational 路径。
- 新增 `training_predicted_explicit/logits`，仅在 `model.train()` 生成并由 `compute_r4_objective` 用于重构；`predicted_*`、`rollout_prior_*` 和验证指标继续走 prior-only 路径。
- target 扰动测试确认部署可见输出不变，teacher 输出会随 target 改变；梯度、有限性、strict reload 和 manifest 均通过。
- 该修正补齐方法闭环但不证明性能改善；不启动 GPU 以外的额外变量，不调阈值，不访问 `locked_test`。

## 2026-09-05 完整 RSSM prior/teacher 双重重构结论

- 为避免训练目标丢失部署语义，最终保留 prior rollout 的 R3 重构，并额外加入权重 `0.5` 的 teacher 重构；KL、KL balancing 和 overshooting 保持不变。
- teacher 张量只在训练模式生成，验证/部署仍只读取 `predicted_*` prior 输出；target 泄漏测试保持通过。
- v2 canonical preflight 的 9 个 train/validation/calibration h1/h5/h20 窗口通过，artifact claim boundary 仍为 execution evidence only。
- 现在 GPU 才是必要的下一门：只运行 seed `20260831`，先性能门，失败即停；不访问 `locked_test`。

## 2026-09-05 完整 RSSM R4 GPU 筛选发现

- 新完整 RSSM 的 R4 单候选运行没有数值不稳定、保存或复载问题，27 epoch 内最佳 score出现在 epoch 22；这支持其作为可训练候选。
- 该 run 的训练 seed 为`20260803`，选择指标为R4 validation protocol score，且只有一个候选；不能外推到P4冻结 seed=`20260831`、正式link-F1门或多seed结论。
- 当前关键缺口不是再跑R4，而是将prior/posterior/teacher语义接入P4正式模型和正式指标路径，并以CPU一致性门证明训练、部署和评价定义一致。

## 2026-09-06 正式完整 RSSM sentinel 发现

- “完整 RSSM”必须让 stochastic latent 实际参与所有被正式评价的连续状态与事件输出；只补 prior/posterior/KL 接口而让分类头绕过 latent，仍属于不完整实现。
- `zero_init_residual_state_heads=true` 必须覆盖新增 RSSM 连续 correction heads。遗漏该契约会在 h1 直接破坏 persistence/residual 锚点；修复后 h1 node-x 从无效 v2 的 `14.239 m` 降到最终 v3 的 `2.205 m`。
- 最终 v3 证明完整 RSSM 能把 link-F1 控制在 persistence 允许回退范围内，并改善 throughput、RB 和 task delay；但 node-x ratio=`1.28774` 超出 `1.25`，P4 仍不能通过。
- v3 node-x 在 h1/h5/h10/h20 为 `2.205/6.948/13.763/30.005 m`，persistence 为 `1.664/5.332/11.047/22.145 m`。剩余问题是随 rollout 累积的位置误差，不再是初始随机 residual 偏移。
- 下一步应只读分解 `final prediction = formal dual-graph base + RSSM correction`，判断长期误差主要来自哪一项；在该证据前不应调 KL、decoder 权重、link 阈值或训练预算。

## 2026-09-06 node-x 修正项诊断发现

- 同一冻结 checkpoint、同一 validation 样本下，formal base ratio=`1.15756`，低于保护线 `1.25`；加入 RSSM correction 后 ratio=`1.28774`。故本轮 node-x No-Go 不是由 base 单独造成，而是 RSSM 连续修正把结果推过失败线。
- correction 平均幅度约 `3.04 m`，但仅 `23.79%` 的有效节点得到改善；从 h1 到 h20，完整输出的 MAE 均高于 base。这支持“修正可信度不足”的机制判断，不支持“多训练几个 epoch 就会好”的结论。
- 该分解是关联定位，不是因果修复实验。最小的新变量定义为训练期 `node_x_residual_non_degradation_v1`：只处罚修正后误差大于 base 的部分，同时保留 prior-only、posterior teacher、KL 和事件 decoder 语义；在 CPU 审计和新冻结协议之前不得启动下一次 GPU sentinel。

## 2026-09-06 node-x 非劣化约束实现发现

- 直接用完整输出和 base 计算约束会把新梯度同时传给 base，混入第二个变量；最终实现从显式 `rssm_node_state_correction` 重构 detached base，因此隔离损失只更新 correction 及其上游 RSSM 路径。
- posterior teacher reconstruction 必须把该新增权重置零，否则同一个安全约束会在 prior 与 teacher 两条路径重复出现；当前候选只约束部署实际使用的 prior correction。
- CPU 结果只证明公式、mask、梯度、checkpoint 身份和无未来泄漏成立。micro run 的任何 validation 数值均不用于性能判断。
- 协议比较确认除 `node_x_residual_non_degradation_v1` 外，v3 的数据、模型结构、训练预算、seed 和性能阈值均未变化。GPU 关闭期间没有必要开展其他实验。

## 2026-09-06 node-x 非劣化约束 GPU 发现

- 约束成功把 RSSM correction 平均绝对幅度从 `3.04245 m` 压到 `0.07621 m`，但这没有保护最终 node-x：base 自身 MAE 变为 `17.97475 m`，完整输出为 `18.02196 m`。
- 新损失对 base 的隔离直接梯度为零，不代表联合训练中 base 不会变化。原 state NLL/MAE 仍通过 `base + correction` 更新 base；当 correction 学习轨迹改变时，base 的最优轨迹也会改变。
- 旧、新 run 初始化哈希同为 `30efb3104c3de9b8ea4bc8fdfb9a321c6aa6036d6bf4bccd1c8ab4ed9d519095`，排除了初始化差异。当前证据反对继续增大非劣化权重或换 seed。
- 若继续主线，唯一有证据支撑的候选是复用旧 v3 已达到 ratio=`1.15756` 的 base 并冻结它，只训练 RSSM correction；这仍是待评审候选，不是已实施方法。

## 2026-09-06 P4 系统性根因发现

- “修正项过大”只解释了完整 RSSM v3 的一次失败，不能解释长期停滞。node-x-safe 已把 correction 压到 `0.07621 m`，base 仍恶化到 ratio=`1.51660`，说明共享优化轨迹是更底层问题。
- 正式 run 的 checkpoint 以 aggregate validation loss 最小为准，P4 却要求五项指标逐项过门；二者没有同一选择规则。局部总损失改善不能推出正式门改善。
- sentinel 的 `256` 个训练窗口只占全部 `14742` 个 unlocked 窗口的一小部分，且 8 epoch 结束时 validation loss 仍下降；当前证据不足以把 No-Go 全部归因于模型结构。
- 位置状态没有 heading、速度向量、路线或机动控制动作；link activity 正例约 `0.918%`。前者构成长步位置的信息充分性风险，后者使 F1 与校准高度敏感。
- “冻结 v3 base 再训 correction”只能隔离一条优化路径，无法排除输入、预算和目标冲突，故降级为待审候选。下一步必须先做现有数据/checkpoint 的 CPU-only 系统审计。

## 2026-09-06 文献对照与 latent 粒度发现

- 当前 `formal_complete_rssm_v1_1` 在“动作条件 prior、观测 posterior、KL、teacher reconstruction、overshooting、部署 prior-only”这些 RSSM 语义上是完整的；此前完整性结论在这一层仍成立。
- 新复核暴露了另一层边界：节点、边与任务先被 masked pooling，单个 stochastic latent 经过 decoder 后为每类实体广播同一 correction。它不能在同一步内为不同节点生成不同随机修正，也不能用单一 link offset 改变边之间的 logit 排序。
- G-RSSM、R-SSM、Graph Dreamer 和 GNS 都把动态状态保持在节点/对象粒度，再通过消息传播表达相互作用。由此得到的是一个有文献依据的结构候选，不是性能证明。
- 因此继续优化同一个 global broadcast correction、冻结 base 后只重训它，均不能消除其表达上限。当前更有信息量的动作是先计算这一上限，而不是立即重建模型。
- 位置运动信息、训练预算、checkpoint 选择和多任务梯度冲突仍是独立风险；不得与 latent 粒度在一次实验中同时修改。

## 2026-09-06 P4 实体级 RSSM 第一性原理结论

- P4 长期卡住不是单一超参数问题：旧随机状态是 global aggregate，link correction 无逐边排序能力；运动输入字段失真；共享目标量级和方向冲突；短预算仍未收敛；checkpoint 选择目标与 P4 门不一致。
- 旧 global node-x oracle 的 aggregate MAE ratio 约 `1.05346`，所以不能声称“全局修正一定过不了 aggregate node 门”；实体级方案的必要性主要来自逐边排序、实体差异和理论定义。
- 因果运动修复没有改变 split 或未来可见性，只把当前及更早位置差分成真实速度/加速度；它修复输入合同，不是加入未来标签。
- 实体级 RSSM 的 CPU 证据证明方法语义、训练可达性和正式 runner 接通，没有证明三 seed 性能。1-epoch gate selector 的失败数字不得作为 No-Go。
- 当前内部 CPU 链已经闭合，外部阻塞是远端 GPU 端口拒绝连接。恢复后必须从 frozen protocol v2 的 batch probe 开始，不得回退到旧 8-epoch sentinel、node-x safe loss 或 global RSSM。

## 2026-09-06 实体级 RSSM GPU 执行发现

- 4090 上 batch 8 的完整 forward/backward 峰值仅占总显存 `16.32%`，因此显存不是本方法正式训练的当前瓶颈；冻结协议仍按预注册候选顺序选 batch 8，不因余量临时扩大。
- batch probe 的 loss 随 batch 不同而变化是因为每档读取的样本集合不同，不能用来比较性能；四档共同证明 loss 和参与训练参数的梯度有限。
- 2 epoch sentinel 完成了真实 optimizer training，RSSM 阶段没有更新冻结 base，best checkpoint 已由 runner 严格重建并加载，说明此前 CPU 证明的分阶段合同在 CUDA 路径同样可执行。
- sentinel 的 validation loss 从 base 阶段 `0.77888656` 降至 RSSM 阶段 `0.76041028`，只属于执行观察；短预算结果不能接受或否决方法。
- 正式 seed `20260831` 已按唯一冻结配置运行。首 seed 结束前没有依据启动另外两个 seed、调整阈值、改变 loss 或访问 `locked_test`。

## 2026-09-06 正式训练证据保全发现

- 远端 runner 在隐藏 staging 中逐 epoch 写 checkpoint，正式目录在完成前为空；因此必须按 staging 做只读快照，不能只依赖最终发布目录。
- 已保存 base epoch `001–005` 的原始 checkpoint、配置、样本 ID、方法注册表、类别权重和进程/GPU 快照；本地证据包 manifest SHA-256=`e2ea66161b663cbbb1dc2807a946dbf5bda6d770d01be82ff96a165fc93b7ad7`。
- 复制和哈希查询未停止或暂停训练进程；当前运行仍由远端 PID `2307` 持有，`locked_test_accessed=false`。
- 监控期间检测到 epoch `006` 新增，立即完成第二次 staging 快照并回传该 checkpoint；这验证了中间证据保全可以与长时间训练并行，不需要中断或重启运行。
- 监控期间检测到 epoch `007` 新增，已按同一规则追加快照和哈希；历史证据没有覆盖。
- 监控期间检测到 epoch `008` 新增，已按同一规则追加快照和哈希；历史证据没有覆盖。

## 2026-09-07 正式训练阶段证据

- base 20 epoch 已完整结束，RSSM epoch 1–2 checkpoint 已生成；这证明 runner 已按冻结合同实际执行“先训练 base、再冻结 base 训练 RSSM”的阶段切换。
- 当前 RSSM 峰值显存约 5.0 GB，高于 base 阶段约 3.5 GB，符合实体 stochastic 分支进入训练后的资源变化；没有 OOM。
- 尚不能由前 2 个 RSSM epoch 判断 P4 性能或是否早停，最小训练轮数仍为 20。

## 2026-09-07 entity RSSM 中间性能发现

- epoch 7–14 连续八个 checkpoint 的单 seed 数值门均通过，说明当前改善不是某一个 checkpoint 的偶然波动。
- entity RSSM 当前同时改善链路排序、长期 node-x、throughput、RB occupancy 和 task delay；这与“逐实体 latent + 因果运动 proposal + 冻结 base”要解决的三个已证实根因方向一致。
- validation state NLL 从 `-3.10564` 单调改善到 `-3.10747`，gate-aware rank 仍在改善，因此现在不能提前停止或挑选 epoch 14 作为最终结果。
- calibration link-F1 delta 很大，但仍需最终独立审计确认阈值只来自 calibration、validation 没有参与选择；当前不据此声明 P4 已通过。

## 2026-09-06 正式 seed 全量运行时序

- runner 使用隐藏 staging 目录进行原子发布，正式输出目录在运行完成前保持为空；当前 staging 已有 base epoch `001/002/003` 三个 checkpoint。
- 实测全量 base epoch 约 26 分钟，当前瓶颈是完整窗口前向/反向和评估计算时间，不是显存容量或进程挂死。
- 当前配置仍是冻结协议：base 20 epoch 后冻结，再训练 RSSM 最多 40 epoch；没有修改 loss、阈值、seed 或数据。
## 2026-09-07 组会材料组织发现

- 这一个月的工作可以形成完整组会叙事：先关闭理论与数据缺口，再以正式P4门暴露模型问题，随后通过因果诊断和文献把方法收敛到实体级双图RSSM。
- 若逐项展示所有P4运行，汇报会变成实验流水账；按数据/指标错误、递推/链路错误、位置/共享训练错误三条失败链组织，可以保留每一步新增证据且不弱化工作量。
- 当前首seed中间结果足以支持“新方向获得正式unlocked数据的强正向证据”，尚不足以支持“P4已完成”或“最终方法已冻结”。
## 2026-09-07 组会PPT精简结构判断

- 10页已经能够覆盖本月所有关键模块；将采集器、字段合同和因果运动合并到两页数据，将图定义、消息传播和规则递推合并到两页双图编码，可以减少流水账且保留实际工作量。
- 文献依据最适合直接放在对应方法页：双图对应交互网络、无线GNN和edge-conditioned message passing；实体世界模型对应PlaNet/RSSM、GNS、Relational SSM、Trajectron++及节点级图世界模型。
- 面向组会时应使用论文式方法名称和自然语言阶段描述，内部阶段编号、脚本标签和gate名称只保留在证据记录中。

## 2026-09-07 组会PPT制作发现

- 直接用PowerPoint COM保存追加稿时，程序会自动重写少数未编辑旧页和一个版式XML；仅凭“没有选中旧页”不能证明第1–203页未变。最终稿在保留新增页面和全局页列表的同时恢复了原文件中的旧页、旧关系和共享版式，并对第1–203页共406个部件逐字节复核，差异为0。
- 第204/205页模板足以支持本次10页汇报：继承顶部标题、双线、徽标和字体体系，再用原生文本框、形状、连接线和表格表达流程与结构，无需把整页栅格化为图片。
- RSSM epoch 22 仍为单seed训练中间checkpoint。它继续同时通过链路、长期位置和运营指标数值门，但不能替代最小训练轮数后的严格复载、独立复算和另外两个seed验收。
- 组会页面对后续策略器的准确边界应写为：候选生成与合法性约束、逐候选世界模型推演、代价/风险选择、执行首动作和观测后重规划；当前只有CPU机制原型，尚未形成正式策略器。

## 2026-09-07 组会PPT层级纠正发现

- 继承页标题、徽标和配色并不足以称为“严格复用模板”；如果把正文改造成卡片和横向流程，原模板的红方框大标题、蓝菱形小标题和浅蓝箭头正文层级仍然会丢失。
- 本项目组会页的正确复用单位是第205页正文占位符及其段落结构，而不是只复用背景。最终版直接保留11个原生段落和Wingdings项目符号字符`113/118/216`，再替换段落文字和蓝色引导词。
- 双图编码与实体级RSSM属于串联的两个模型模块：前者生成关系感知的实体表示，后者在该表示上学习动作条件时间动力学。把它们写成“方法一/方法二”会错误暗示二选一或平行对比。
- 结构图只能作为某个小标题下的辅助证据，不能替代文字层级；本轮汇报采用纵向文字结构，优先保证老师能够按“问题—实现—依据—结果”顺序阅读。

## 2026-09-07 组会PPT图表折中发现

- 最合适的折中不是在“全图形”和“全正文”之间平均分配面积，而是先用原生层级给出判断与依据，再把图表放到对应小标题之后解释结构或比较数值。
- 流程、关系结构和递推机制适合用图；正式指标适合用表；定义、理论来源、能力边界和未完成条件仍应由层级正文承担。
- 最终折中版每页使用7个原生层级段落，图表全部位于正文下方；这样同时保留模板阅读顺序和模型结构的可解释性。
- 双图仍是模块一，RSSM仍是模块二；图形连接表达两者串联关系，没有恢复“方法一/方法二”的错误语义。

## 2026-09-07 组会PPT重点强化发现

- 数据页的轨迹划分数字不能解释方法的必要性，会分散对“字段是否可信、动作是否可追溯、样本是否因果合法”的注意力，因此从主汇报中移除。
- 图表的保留标准是能否承担关系结构、时间机制、训练阶段或数值对比；字段定义、适配理由和边界条件用原生层级文字说明更清楚。
- 本版形成单一叙事链：旧数据为什么不能直接学习 → 如何重建合法样本 → 双图如何编码关系 → 旧时间模型为什么失败 → 实体级RSSM如何修复 → 当前证据与后续决策闭环。

## 2026-09-07 当前双图四类对象的实现事实

- 物理节点主状态为 `x/y/z/speed/acceleration/cpu/storage`，并额外提供因果派生的三维速度和加速度给实体级运动 proposal。
- 物理边是物理设备之间的有向通信链路，特征为 `distance/csi_mean/rate_sum/active_task_count/allocated_rb_count`；因此这五项不能称为信息边特征。
- 信息节点是一物理节点一 agent 的附着代理。当前 agent 没有独立观测字段；模型用相同的 7 维 node history 初始化 agent latent，再通过信息图、任务和跨图消息更新。
- 信息边是 agent 之间的数据流，正式数组名为 `flow_state`，特征为 `total_data/remaining_data/delivered_cumulative/delivered_this_slot/age`，类型为任务输入、结果回传和有显式 payload 时的依赖数据流。
- `flow_endpoint_index`定义信息边端点，`flow_task_index`绑定任务，`flow_bearer_mask`逐时隙绑定承载该流的物理通信边。信息图消息沿 agent—flow—agent 传播，跨图消息沿 agent—physical-node 和 flow—physical-edge 传播。
- 实体级 RSSM 只为 node、physical_edge、flow、task 设置 prior/posterior 随机状态；agent latent 是基础双图里的确定性中间表示。这是当前实现边界，不能说四类对象都拥有独立观测和独立随机动力学。
## 2026-09-08 P4 首 seed 收敛发现

- 实体级双图 RSSM 在完整 unlocked 数据上并非只短暂过门：RSSM 40 个 epoch 均保存，冻结选择器最终选择 epoch 39，9 项单 seed 数值门全部通过。
- 最终检查点把旧 global complete RSSM 的 link-F1 delta `-0.04113`、node-x ratio `1.28774` 改善为 `+0.44415`、`0.75475`；当前证据支持因果运动合同、实体级 latent 和两阶段冻结训练这一组合方向。
- 该结论只覆盖 seed `20260831` 的 non-locked 数据。跨 seed 泛化仍未知，不能提前关闭 P4 或形成最终性能声明。
## 2026-09-08 用户锁定后续 seed 启动权限

- seed `20260830` 是当前唯一获授权继续运行的正式实验。
- seed `20260832` 的启动授权已明确收回到用户手动决定；现有通过结果不得被解释为自动续跑授权。
- seed `20260830` 完成后的正确动作是保全产物、独立验收、报告结果并暂停。
- 中间进度的有效证据是 checkpoint 内的完整 `p4_gate` 数值和哈希；单独报告轮次不能支持性能判断。

## 2026-09-08 组会 PPT 复核发现

- 当前 18 维历史记录应称为旧版通信边混合张量；本月工作不是简单从 18 维压到 5 维，而是把物理通信链路和业务数据流重新分成两类对象，各保留 5 维可信状态。
- `per_rb_target_sidecar` 仅为诊断保留，当前正式模型、损失和指标不消费；PPT 不能写成逐 RB 结果用于训练监督。
- 新链路头以上一时刻活动状态为 persistence 基线，再由逐物理边 latent 解码变化量；这是解释链路排序和相对 persistence 提升时不可省略的实现事实。
- 首 seed 结果支持当前结构方向，但跨 seed 泛化仍未知；汇报中必须把单 seed 验收、跨 seed 待验收和 `locked_test` 未访问分开表述。

## 2026-09-08 项目知识入口重构第一阶段发现

- 仓库在 2026-08-16 已完成顶层目录迁移；本次需求的缺口不是再次移动 `代码/文档`，而是缺少稳定的项目地图、架构、科研状态、实验和结果索引。
- 当前最安全的改造方式是新增导航文档并补充维护规则，不对正在同步的 `code/artifacts` 做批量整理。
- 当前 `code/artifacts` 同时存在正式 tensor、训练、audit、历史候选、tmp、transfer 和 live evidence；必须按证据层级索引，不能按 `final`/`best`/`latest` 文件名猜测当前结果。
- 当前 PPT 文件与对应验收 JSON 存在页数和 SHA-256 不一致，说明汇报材料需要独立版本验收，不能把旧验收记录自动套到当前文件。
- 保护边界有效：本轮未改变 `seed=20260830` 训练进程、同步文件、checkpoint、tensor、协议或 `locked_test` 状态。
## 2026-09-09 组会追问文档事实边界

- 当前实现的性能证据与图对象命名必须分开：`physical_edge_state` 承载通信链路、`flow_state` 承载任务数据流是代码事实，但主文档仍把通信关系定义为信息边；目前没有独立证据证明这次语义重分类更优。
- 首种子验收可以支持实体级 RSSM、逐边修正和两阶段训练的单种子可行性，不能支持跨种子、策略闭环或最终方法冻结。
- 截至本地最后可核验动态记录，seed `20260830` 已启动且至少到实体 RSSM 第 6 轮，本地尚无完成验收产物；文档没有推测远端后续状态。

## 2026-09-09 第二个正式 seed 收敛发现

- seed `20260830` 在同一冻结配置、完整 unlocked 数据和独立验收下再次通过 9 项单 seed 数值门，说明 seed `20260831` 的成功不是孤立的一次随机结果。
- 两个已完成 seed 的 link-F1、node-x 和运营指标方向一致；当前证据加强了实体级 latent、因果运动输入、逐边残差和两阶段冻结训练组合的合理性。
- 该证据仍不能替代预注册的三 seed 泛化门。seed `20260832` 未运行，三 seed均值和稳定性未知；P4不能关闭，`locked_test`不能开放。

## 2026-09-09 项目重构发现

- 当前问题的核心不是缺少顶层目录，而是 828 个项目文件、604 个 Python 节点和 802 个 artifact 目录缺少统一机器索引；第一阶段文档只能导航，不能回答反向依赖和批量实验定位。
- 211 个 Python 节点可按命名和证据边界标为历史；当前正式节点对它们的反向引用为 0，但 94 个仍有其他引用、93 个有直接测试，说明“当前主线隔离”与“历史可复现”可以同时成立。
- 物理移动历史模块当前没有可靠回归基线：全量套件包含 AirFogSim 环境缺失、过期 fixture、历史 artifact 权限和 calibration 样本问题。先逻辑归档比改变 import 路径更符合证据保全。
- 12 个所谓 Python 语法错误来自 UTF-8 BOM，而不是代码语法；生成器改为 `utf-8-sig` 后解析错误清零。14 个历史 manifest 仍因权限不可读，属于独立真实限制。
- 现有 GPU runner 已具备 seed 参数、输出目录和 locked-test 路径守卫；不需要创造新训练接口，只需把 `20260832` 的现有入口、冻结前提和用户授权门机器化登记。
- 重构后全量套件已消除根目录结构 failure，剩余 21 个 errors 没有来自本轮索引或当前 P4 核心；不能靠跳过严格合同或修改科研代码来换取表面全绿。

## 2026-09-09 长期协作闭环发现

- 单靠目录索引不能可靠回答“为什么旧方法不用了”；必须保存方法动机、真实结果、弃用原因、替代关系和证据路径。因此新增的历史方法注册表承担语义检索，artifact catalog 承担长尾精确定位。
- 正式数字写进摘要后仍可能随手工维护漂移；将注册表与 acceptance JSON 的文件哈希和 9 项指标逐项比较，才能让“容易找到”和“找到的是正确证据”同时成立。
- 中文口语提问不能只依赖英文关键词；CJK 字符片段匹配与人工问题路由结合后，“我们现在模型是啥”“第三个 seed 什么时候运行”等问法都能稳定落到正确入口。
- 当前达到的是无损的逻辑整理、语义检索和证据防漂移，不是历史文件的物理搬迁。21 个既有全量错误继续阻断物理归档，但不阻断日常问答和证据核实。

## 2026-09-10 AI_CONTEXT 重构发现

- 现有 `docs/` 和机器注册表已能支撑 Codex 检索，但缺少面向 ChatGPT 网页端的低上下文首入口；九文件 AI_CONTEXT 补的是角色化入口，不复制或替代既有证据。
- 静态文件无法可靠硬编码“包含自身的最新 commit hash”；`00_PROJECT_STATE.md` 因此记录创建基线 commit，并规定精确最新版本以 GitHub `main` 的 `HEAD` 为准，避免提交后立即自相矛盾。
- 2026-09-10 当前权限使原先 14 个 artifact 读取错误归零，全量错误由 21 降为 17；这是环境可读性变化，不是科研结果改善。
- `AGENTS.md` 中具体 P4/pre-GPU 临时步骤与“只放长期规则”冲突，已移入 AI_CONTEXT/计划层；永久文件仅保留通用冻结、审计和证据一致性门。
- 剩余 17 个全量 errors 仍阻断历史文件物理迁移，但不来自本次 AI_CONTEXT、注册表或当前 P4 正式路径。

## 2026-09-16 最新 AirFogSim 源轨迹分享包（仅本地交付）

- 用户明确要求最新版，交付 v6 formal source（B层），不含六月A层CSV、训练npz/checkpoint或locked_test；没有重跑仿真。
- 源：code/artifacts/formal_data/pi_jwm_v4_formal_candidate_v6_rb_v1_unlocked_20260821；54条、6场景、每条300步、0.1秒、合计16200轨迹时点。
- 完成：code/artifacts/packages/AirFogSim_raw_data_share_20260916.zip；658389567字节；SHA256=e2b73fc877a3c5117767a20e23c2527281876c0036e42be16835fb99a08fcb59。
- 验证：486项轨迹文件及8项顶层历史清单匹配；54个时间网格通过；ZIP全部800文件逐项解压读取、大小和SHA256通过；所选数据零缺失。
- 限制：历史完整config哈希重建0/54匹配；生成时项目/AirFogSim commit无法确认；最新v6源与当前tensor摘要所指v4不是同一来源声明。现有场景不构成严格单变量对照。上述差异仅报告，没有改写源证据。
- Context Consistency Check只读完成；按用户要求未修改AI_CONTEXT、模型或训练代码；本任务不commit/push。科研P4、第三seed、GPU、同步和locked_test边界不变。
- 本次单一下一动作：用户直接将ZIP交给同学；若需逐值重现历史仿真，先恢复完整历史配置和版本，不能自行重跑。

## 2026-09-18 新定义 Step 1 审计发现

- 当前 `physical_edge` 混合空间关系、CSI、rate、任务数和 RB，与目标严格 Physical/Information 分离不一致；独立 Agent state、Communication relation、Task-Agent typed relation 不完整。
- 旧 tensor/action 路径可复用 offload、RB、return 和 CPU 事件来源，但未闭合完整 Route、逐 RB Comm、CPU action 和 UAV Mobility 的采集、张量、执行及后果。
- 当前 RSSM 为 node/physical_edge/flow/task 都配置 `h,z`；新定义主要只对 Physical/Communication 未知动态设置随机状态，旧 checkpoint/layout 不能直接复用。
- base 内部有逐步规则递推，但实体 RSSM 修正在 base 整段推演后叠加；尚未实现“学习动态→规则更新→重构双图→下一步”的完整单步闭环。
- planner 有逐候选调用与首动作接口骨架，但合法候选、冻结 objective/risk/hard constraints/fallback、真实执行反馈和连续 replanning 仍缺失，只能标 `prototype_only`。
- 旧两 seed、旧 tensor/checkpoint 和旧 planner 结果全部保留为 Historical / Archived evidence，不能用于声称新 `00–06` 已实现或已有性能。

## 2026-09-18 STEP 2

- 已确认：`A_t=(A_t^Route,A_t^Comm,A_t^Comp,A_t^Mob)`，Mob 只控制 UAV，车辆由 SUMO 推进。
- 已确认：完整真实四类轨迹仍缺失；最小闭环不能替代真实场景验收。
- 已记录：本机缺少 `shapely/traci`，完整场景未运行；不影响 scheduler 源码 setter 的最小核验。
# 2026-09-19 Step 2.1 真实 AirFogSim 验收

- 真实非 locked AirFogSim 单轨迹通过四类 scheduler 与真实 `env.step()` 闭环；13 项 checks 全部为 true，未使用 `_MinimalEnv` 或手工 Outcome。
- vehicle traffic `angle` 是 degree，UAV `angle/phi` 按 rad；合同采用实体 `heading`，Mob action 保留 `azimuth_rad`。
- UAV 速度由 0 变 10 m/s 时 AirFogSim 真实 acceleration 为 `-100.0`，记录为仿真器现状，不在本 Step 修正。
- Step 2 专项测试真实为 6 项；generated registry 未纳入 `index-build.tmp.err`/`tmp.err`。

## 2026-09-19 Step 2.2 多步接口事实

- 下一 Decision 可以在下一循环对同一真实环境独立重采，并与上一 Outcome 严格对齐；不能用对象复制替代该证据。
- 6 步中 Route/Comm/Comp 均出现非空和空帧。空帧必须保留 action family 字段、空 entries 与原因；missing field 不等于 no-op。
- Decision 时刻固定 Comp 分配意味着同 slot 新完成传输的任务本 slot CPU 分配为 0，下一 Decision 才进入显式 Comp action；这保持 Decision 可见性边界。
- AirFogSim UAV acceleration 使用 `(last_speed-speed)/interval`，首次加速与常规前向差分符号相反；vehicle 本轨迹 6 行均匹配前向差分。
- `code/artifacts/*` 默认被忽略；仅在实施记录中写路径不能形成 GitHub 机器证据。Step 2.1 v3 和 Step 2.2 小型 JSON/manifest 需显式 force-add。

## 2026-09-19 Step 2.3 因果与字段发现

- `_to_generate_task_infos` 真实包含 `arrival_time_s > decision_time` 的未来任务。它是 simulator internal schedule，不能进入当前 `O_t`、History 或 input-side Entity Index。
- `channel_manager.getCSI` 可在 Decision 前得到 42 条逐 RB channel rows；同 slot transfer event 可给出真实 delivered data。
- `entity.getFogProfile()` 并非每个节点都有 `cpu` 键。合同必须允许 `null + mask + missing reason`，不能把缺失写成 0 capacity。
- `Task.getComputedSize()` 的 post-pre delta 可形成 slot served CPU work；`setTaskReturnRoute` 入队后由真实 env step 推进到 returning/done。
- AirFogSim raw acceleration 保留审计；PI-JWM canonical acceleration 使用只依赖当前/历史的 backward difference，首帧和新实体显式 mask。

## 2026-09-19 Step 2.4 通信 Outcome 发现

- AirFogSim wired service 的直接 slot 数据来自 `WiredNetworkManager.step(interval) -> {task_id: transmitted_bytes}`；`AirFogSimEnv._updateWiredCommunication` 随后调用 `Task.transmit_to_Node` 并在完成时推进 lifecycle。
- Step 2.3 的 `pi_jwm_transfer_events` 原先只覆盖 wireless；把 total 描述为 wireless+wired 与真实采集不一致，现已通过 transport split 修正。
- 已接线 transport 的无服务 slot 是空 map 且 mask=true；接口不可用必须是 null/mask=false/reason。只有两类分量均 observed 时，total 才可用。
- 真实两跳任务在一个 AirFogSim slot 内先完成 wireless、再完成 wired，post-step current node 为 cloud、lifecycle 为 computing；Action route 必须复制保存，否则 simulator 原地消费 route list 会污染记录。
- cloud 节点的现有 FogProfile 无 `cpu` 键；通信验收保持 Comp no-op，未改 Step 2.3 CPU missing 语义。

## 2026-09-19 STEP 3.1F-PATCH

- 发现：`994da0b` 的 Future Action 使用 anchor-only index，在 History union 包含已消失对象时会把合法对象重新编号。
- 证据：disappearing-object fixture 在修复前稳定复现 `History index=1` 被写成 `0`；修复后 validator 逐字段核对 ID 与 static index。
- 边界：audit 的 18 窗口/0 unresolved 仍只是当前非 locked Raw observation，不是正式 Dataset 可用率；STEP 3.2 未授权。

## 2026-09-19 STEP 3.2-PATCH findings

- 旧 batch provenance 只有 path/hash/trajectory/split/schema，无法完整证明 seed/config lineage 与 time-grid；本 Patch 从真实 Raw environment/execution 字段补齐，缺失字段保留 null，不伪造 metadata。
- Raw decisions 为 7 帧、steps 为 6 帧；冻结 slot duration 为 0.1 s。每个 step 的 execution start/end 与 decision/outcome 时间均通过逐项检查。
- Batch audit 必须重新按 Step 3.2 的三条 development trajectory 统计；当前为 12/12/0/0，不能引用 Step 3.1F 的 18-window audit 作为替代。
- task size 单位只能写 `AirFogSim data-unit`；现有源码和记录没有可靠 bit/byte 换算证据。
## 2026-09-20 STEP 4.2B

- `Task._transmitted_size` 每个 hop 完成后 reset；不能作为端到端 Flow remaining。
- `Task.getReturnedSize()` 是 return total requirement，不是 already-returned amount。
- `_task_dependencies` 是 DAG gating，不等于 DepData Flow；当前无 dependency payload/transfer source。
- 旧 `LogicalFlow`/`CarryingHop` 提供动作侧命名，但不是 simulator-issued stable Flow provenance。
- 机器 receipt：`FLOW_CONTRACT_NOT_YET_SUPPORTED`；停止进入 Graph Builder。

## 2026-09-20 STEP 4.2B-PATCH

- Flow verdict 现在由 identity/type/Task/端点/presence/total/remaining/causality/multi-hop/route evidence 计算；篡改 verdict 会被 validator 拒绝。
- dynamic available CPU、storage、wired queue/load/utilization 已移至 `other_information_graph_gaps`。
- simulator-issued Flow ID 缺失改为 implementation fact；logical identity 是否按 `task_id + input/return` 派生留给 researcher decision。
- DepData 当前无真实 transfer process，但不再作为 Input/Return readiness 的直接 blocker；DAG 仍禁止生成 fake Flow。
## 2026-09-20 STEP 4.2C-A-PATCH

- `Task._transmitted_size` remains hop-local, but it need not be the logical remaining source: logical-destination filtered real transfer events causally maintain E2E remaining.
- Final-destination delivery must be counted separately from intermediate hop service to avoid double counting.
- Future action changes do not alter replayed current ledger state; past outcome service remains non-current evidence.
- DepData has no audited transfer process; DAG gating cannot create a fake Flow.
## 2026-09-20 STEP 4.2C-B

- Flow identity is stable across carrying hops and same-destination reroute; only destination change at a clean boundary increments Epoch.
- Raw `O_t` exposes only Ledger state updated through the previous Outcome; same-slot delivery first appears in `O_{t+1}`.
- Real non-locked traces cover direct Input/Return and Input multi-hop. Return multi-hop, reroute, Epoch switch and local no-flow remain fixture-only observations.
- Legacy wireless Return event delivery may exceed observer return_size; frozen min-capping preserves logical conservation and the mismatch is retained as a real-trace limitation.

## 2026-09-20 STEP 4.2C-B-PATCH

- Root cause: Raw amendment used `entry.target_node_id` for logical destination, but ongoing actions expose the current carrying-hop target there.
- Proven Input source: `TaskManager.offloadTask` asserts `route[-1] == target_node_id`; `Task.offloadTo` stores that assigned target and remaining route; `transmit_to_Node` deletes only route element zero after each completed hop.
- Proven Return source: `Task.setToReturnRoute` stores its terminal in `_to_return_node_id`, and observer exposes it as `return_destination_id` independently from remaining route.
- Real `Task_1` service sequence is `UAV_0→RSU_0→cloudServer_4`; both events bind `flow::Task_1::Input::0`, Epoch 0 and destination `cloudServer_4`. Per-hop service sums to twice the payload, while E2E delivered counts only final-destination delivery once.
- Same-destination partial-hop reroute remains unresolved runtime evidence; no new research rule was invented.

## 2026-09-20 STEP 4.2C-C

- Sample/Tensor keeps logical Flow identity separate from hop carrying state: FlowID/Epoch/destination/E2E state come from C-B Raw, while stable history slots, target isolation, presence/mask, and route/holder arrays are collated independently.
- Only five continuous data-unit fields are fit with train-only, mask-aware normalization; identity/category/reference fields are not normalized and capacity overflow is rejected rather than truncated.
- The real cross-slot trace proves one Input Flow remains in one Tensor slot while intermediate hop service does not advance E2E delivery; it does not prove Return multi-hop, reroute runtime, formal capacity, Graph Builder, model, or training behavior.

## 2026-09-20 STEP 4.2C-C-PATCH

- Root cause of the first equality failure: the tensor was built from `apply_flow_normalization()`'s copy while the negative/coverage test passed the pre-normalization sample; equality now derives expected normalized values from the recorded stats when needed, while still comparing raw fields and masks.
- A second semantic gap was found: changing a logical destination/holder/route ID without changing its numeric index could pass numeric-only checks. Tensor metadata now preserves source ID/provenance projections and equality rejects that tamper.
- `target_carrying_*` arrays are a separate future target namespace. They are deterministic ground-truth transition state for this contract only; their presence does not imply a learned prediction head or Graph Builder input.
- Scope remains non-formal and CPU-only: no Graph Builder, model, Loss, Planner, training, GPU, or `locked_test`.

## 2026-09-20 STEP 4.3A

- The frozen current tensor already contains sufficient minimum fields to materialize typed Physical/Information graph objects without reading the simulator or target namespace.
- Physical membership can remain policy-driven: current presence plus valid XYZ admits a node without hard-coding entity classes. Development radius/kNN values are configuration evidence, not a research conclusion.
- Carrying hop endpoints cannot replace logical Flow endpoints. In the current real multihop frame the hop destination may equal the logical destination, so the negative fixture also checks the stage-local hop source.
- GeoComm is an endpoint-Physical dependency for wireless Comm and does not require an identical Physical edge; wired/no-spatial rows remain explicit but invalid.

## 2026-09-20 STEP 4.3A-CONTEXT-PATCH

- Stale current-state wording survived mainly in the `00_PROJECT_STATE` blocker, Tracker header/Step 4.2C-C summary, and generated PROJECT_INDEX/RESEARCH_STATUS summaries.
- Historical statements are still valid for their original Step, but require explicit “at that Step” wording so they cannot override current STEP 4.3A COMPLETE / FROZEN state.

## 2026-09-20 STEP 4.3B

- Current non-Flow Tensor arrays named normalized `*_features` are exactly equal to their raw counterparts; treating them as normalized would be an unsupported assumption. Fixed upstream train-only stats are therefore applied explicitly inside the encoder input path without fitting in forward.
- The stable Flow representation can exclude `e2e_delivered`, Epoch and Flow Index: total plus E2E remaining carry the minimum learned numeric state, while identity/provenance remain structural.
- Strong slot-permutation evidence must permute entity, task and flow rows and consistently remap every consumed reference. Passing only a batch permutation would not prove that numeric IDs are excluded.
- STEP 4.3B establishes encoder wiring and differentiability only. `Z_t^{PI,L_g}` remains distinct from `xi_t^Lat`; no representation-quality or prediction claim follows from deterministic untrained output.
## 2026-09-20 STEP 4.3B-PATCH

- Found and corrected two formula mismatches: P2A value had ignored Agent latent and P2C value had ignored Comm relation latent, despite their gates using joint context.
- Found and corrected incomplete `Z_t^{PI,L_g}` structural output: all eleven STEP 4.3A blocks are now preserved as side information, with semantic equality and tamper rejection checks.
- Removed the Comm CSI width source constant `50`; actual width is read from tensor contract `n_comm_rb` and mismatch fails explicitly.
- Evidence remains untrained CPU development wiring only; no World Model/RSSM/dynamics/training/GPU/locked-test/formal Dataset.
## 2026-09-21 STEP 4.4 communication service sufficiency finding

- `ChannelManagerCP.computeRate` computes nominal per-RB rate from signal/interference/noise and bandwidth, but independently samples Rayleigh outage with `random.rand` and zeroes rate on sampled outage.
- The frozen Decision observer exposes per-RB CSI including fast fading; the actual outage realization is only available in the execution collector with `temporal_role=outcome_only_not_same_frame_decision_input`.
- Therefore `CSI + A^Comm + known parameters` does not uniquely determine actual service. Promoting the outcome outage to input would be future leakage.
- Wired service needs configured capacity and active-flow count. Both exist in simulator state but are not frozen Raw/Tensor inputs, so they are additive gaps rather than unobservable residual evidence.
- Required verdict: `SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`; residual target/architecture remains unselected and STEP 4.4 implementation is stopped.

## 2026-09-21 STEP 4.4 resolved service boundary and model finding

- The researcher selected outage as a conditional known stochastic transition, not Comm latent and not learned residual. The earlier audit verdict is preserved as pre-decision evidence, not current blocker state.
- Complete RB allocation and predicted CSI feed channel-type power/interference/noise SINR. AirFogSim nominal-rate and Rayleigh-outage formulas are explicit; `sample` requires a generator and `expectation` is marked approximate.
- Existing Flow Carrying state exactly derives wired active membership/count in the real manager equality fixture; no extra membership tensor is needed. Capacity itself is read from the causal trajectory config and remains an explicit World Model state input.
- The implemented artifact proves untrained mechanism only. Accuracy, calibration, loss, training, planning and performance remain unknown.
## 2026-09-21 STEP 4.4-PATCH structural closure

- Root cause: canonical recursive receipt reused a route action after step one completed its Flow; the existing absent-index rejection was correct. The builder now uses a contract-valid negative-index no-op for step two while retaining complete RB allocation.
- The final receipt contains 87 required checks: 50 original, 21 service-transition, and 16 structural/rule checks. It passed with zero failures. Task lifecycle acceptance reads the frozen `LIFECYCLE_VOCAB` instead of a numeric literal.
- Evidence remains untrained CPU development only. No Raw/Tensor/graph schema, Loss, Training, Planner, GPU, `locked_test`, or formal Dataset was changed or used.
# 2026-09-21 STEP 4.4-PATCH2

- Root causes were weak hop cap, pre-route rebinding, scalar route revision input, unused Flow categorical embeddings, and static-only DAG receipt.
- Existing Return Flow is not fabricated or omitted: the adapter now binds frozen 4.3A `task_index + flow_type_index=Return`; future-only Return birth remains unsupported and missing required support is an explicit blocking side-state.
- Evidence remains mechanism-level, CPU-only, untrained, and non-formal.
# 2026-09-21 STEP 5.1B

- 当前实现只允许对应 family 的 normalized target 和 mask 进入 target encoder；prior predictor 的接口不接受 target，tampering invariance test 通过。
- family loss 在各自有效 mask 内归一化，空 mask 返回零和零计数；KL 显式返回 raw、free-bit adjusted 和 eligible count，未实现 balancing/overshooting。
- 12-sample CPU receipt 证明原语可执行，不构成训练收敛、性能或正式数据集证据。
# 2026-09-22 STEP 5.1C findings

- The support mismatch is lineage-level: historical 4.2C-C/4.3/4.4 artifacts contain 5 samples at 8/44, while STEP 5.1A contains 12 samples at 10/74.
- The suspected raw-support blocker did not reproduce. All 12 windows have real source provenance and pass the actual 4.2A + 4.2C-B + 4.2C-C path.
- The new unified bundle proves paired sample/window identity and capacity alignment, but does not yet prove rebuilt 4.3A/4.3B/4.4 execution. Do not call STEP 5.1C complete or enter STEP 5.2 until that rerun and paired 5.1B receipt pass.
# 2026-09-22 STEP 5.1C-PATCH findings

- 旧 5.1B/Flow artifact 的 8/44 support 与 5.1A 的 10/74 target 不能作为 paired evidence；当前统一 bundle 已消除该 lineage mismatch。
- `no_prefix_truncation` 不能由常量 receipt 证明；本 Patch 改为由逐样本 identity、support width、capacity 和 target tensor shape 的真实比较计算。
- 历史 `flow_train_normalization_stats.json` 未被删除，已明确标为历史 provenance；当前 bundle 使用独立 unified dev_train stats。
- 当前证据仍是 CPU/non-locked contract evidence；4.3A/4.3B/4.4 unified rebuild 是唯一后续候选动作，需研究者另行授权。
# 2026-09-22 STEP 5.1D

- 旧 5.1B receipt 的单 carrier/旧 support 不可作为 paired closure；新 receipt 只使用统一 12-sample lineage。
- 完整 Flow Tensor package 需要显式保留 sample IDs 和 upstream base Step 3.3 checks；已补齐并加 roundtrip test。
- CPU evidence seed 固定为 5101 后，两次独立 rebuild 的全部输出 SHA-256 相同。
- 结论仅为 non-locked CPU development integration，不是训练或性能结论。
# 2026-09-22 STEP 5.1D-PATCH findings

- `normalized_samples.json` 提供了 upstream 4.2A 逐窗口 metadata；与 unified bundle 的 8 个 dev_train sample_id/trajectory_id/anchor/split 顺序完全一致，未重新拟合 stats。
- 改变真实 paired target 后，actual `phy_prior`/`comm_prior` 的 mean 和 log_std 保持完全一致，而 posterior 改变；这是 runtime isolation，不只是 contract 字段检查。
- 旧 receipt 的全 false scope 会误导；现拆为 executed scope 与 forbidden scope，passed 只由 required checks 和 forbidden scope 共同决定。
- 真实 12-sample action coverage 中 Route/Comp non-empty 均为 0；不能外推为完整四动作族正式训练覆盖。

# 2026-09-22 STEP 5.2 findings

- 当前 5.2 loop 真正把 5.1D unified state/action adapter、4.3B encoder、4.4 RSSM 和 5.1B target/posterior/loss/KL/metric primitives 接成 CPU optimizer path；不是只检查接口存在。
- Stage 1 的 Future Motion/CSI 只进入对应 family posterior teacher；Stage 2/validation 由 `initialize_latent(posterior_mode="prior")` 和 prior recursive state feedback 驱动。receipt negative checks 证明 Future Target 与 Future GT state 不进入 prior rollout。
- optimizer audit 覆盖 encoder、RSSM dynamics、两类 prior、两类 future posterior、两类 target encoder 和两个 decoder；known deterministic rule 参数为 0。两步 smoke 后有真实 parameter update，validation `parameter_changed_count=0`。
- `L_Val=168.31609344482422` 仅为 8/4 development smoke diagnostic；不支持 tiny-data overfit、收敛、泛化、正式训练或性能结论。Route/Comp non-empty coverage 仍为 0，作为 future formal training/data coverage gate。
- 全量历史 suite fresh 运行结果为 `1887 tests / 33 errors`；错误集中于 AirFogSim GBK 输出、缺失 archived artifact/fixture drift 和旧 runner 环境边界。5.2 focused 与 current regression 无新错误。
# 2026-09-22 STEP 5.2-PATCH

- 事实：Stage 2/Validation 当前 latent 由 current-observation posterior 初始化，Future Target 不改变 current posterior 或 future prior。
- 事实：Validation 已按每个 horizon 的 Motion/CSI numerator/count 全集聚合；Future posterior teacher 与 Target Encoder 调用为 0。
- 事实：checkpoint 错误 data identity/normalization 被拒绝，compatible reload 保持 forward digest。
- 边界：仍无 tiny-data overfit、full training、GPU、formal Dataset、locked_test、baseline、Planner、性能结论；Route/Comp non-empty coverage=0/0。
# 2026-09-22 STEP 5.3

- GO 证据：Phase A Motion/CSI 均下降；Phase B prior H1/H2 均下降；Phase C 两样本双 family 均下降；无 NaN/Inf；resume trajectory 一致。
- CSI 数值 loss 较大与 normalized target scale/difficulty 相关，未发现 stats 实现错误，不改 0.5/0.5 权重。
- 部分 teacher groups 存在零梯度步骤但非全程，已由 zero_gradient_steps 审计；Route/Comp coverage、GPU、formal training、locked_test 仍关闭。
# 2026-09-22 STEP 5.3-PATCH

- 5.3 原 Phase C 没有执行冻结的 schedule；修正后直接复用 `Step52Trainer.train_step(global_step=...)`，step 0–9 为 H1/Stage1，step 10+ 为 H2/Stage2，KL 在 20 steps 内 warm-up。
- `motion_predictions`/`csi_predictions` 是 decoder raw outputs；raw metric 只 inverse-transform targets。CSI audit 实测 prediction 约 0 dB、target 约 98–99 dB，未发现二次 bridge bug。
- 真实 pre/post 与独立 fresh-run 均通过；learning signal 可接受，但强 tiny-overfit 双条件均未通过，不能进入 GPU/STEP 5.4。
# 2026-09-22 STEP 5.3D

- Observation：H1/H2 actual CSI normalized MSE 与 `(pred_raw-target_raw)/csi_std` expected MSE 完全一致；normalization bridge 无 bug 证据。
- Observation：baseline 200 steps 从约 `329/333` 降到 `117.66/119.74`，未达到 stronger gate；mean-bias diagnostic 从约 `1.09/1.10` 降到 `0.1405/0.1633`。
- Interpretation：更支持 raw-output initialization/conditioning bottleneck，而不是当前证据支持 decoder capacity 或 normalization bug；正式方案仍需 researcher decision。
# 2026-09-22 STEP 5.3E

- train CSI mean = 98.34974797337962 dB，provenance=dev_train；所有 RB bias 一致，optimizer 创建前初始化。
- H1/H2 normalized family loss final 均 <1，relative drop 均 >50%；结果仅是 development tiny-data capacity/optimization evidence。
- Route/Comp non-empty coverage 仍为 0/0；不得据此声称完整动作族训练或 formal performance。
# 2026-09-22 STEP 5.4

- 当前 5.2 `DevelopmentBundle.load()` 确实固定了 development artifact、12 samples 和 8/4 split；新增接口已把 formal 边界改为 manifest/split 驱动。
- 真实 development action coverage：Route=0、Comp=0、Comm=1、Mobility=24；没有用代理或伪造数据补齐 Route/Comp。
- `L=2`、`radius_knn(1000m,k=2)`、正式 seed/预算仍是 development 或 researcher decision required，不能包装成 formal config。
- 只完成 device-agnostic 静态审计与 CPU dry-run；`GPU_CODEPATH_PREPARED` 不等于 `GPU_TRAINING_VERIFIED`。

# 2026-09-22 STEP 5.4-PATCH

- Generic package now loads samples/tensor/graph/target/normalization and enters the real Trainer.
- `validate_readiness()` computes verdicts from checks; negative identity fixture passes.
- L=4 is config-only evidence, not runtime rollout verification.

# 2026-09-23 STEP 5.5 findings

- 原 Step 3.1 sample builder 的 H=2/L=2 固定拒绝是 Formal L=4 的真实工程阻塞；已泛化为正整数 H/L，同时保留 H2/L2 regression。
- 长 trajectory 暴露了 target-only future entity/task/route index 不应按 current input capacity 验证；validator 现使用独立 target namespace capacity，没有把 future-only object 偷放进 input index。
- `np.savez_compressed` 默认 ZIP member 时间戳会破坏 byte-level deterministic rebuild；Formal builder 对每个 NPZ 进行固定时间戳和排序 canonicalization，并计划用完整二次重建 package SHA 比较验收。
- 全量 package 必须按 trajectory 分 shard 构建，避免一次把 5520 windows 的 Tensor/Graph/Target 常驻内存；每个 shard 写出后立即 reload 并检查 sample identity、batch 维和 semantic digest。
- Raw coverage 不是伪造 action row：collector 真实调用 Route setter、RB scheduler、capacity-respecting CPU callback 与 UAV mobility setter，并在每个 Decision 记录 eligibility/intervention/no-op/signature。
# 2026-09-23 STEP 5.5 final findings

- 60 个 primary simulator/policy seed pair 全部成功，因此 rejected/replacement=0；这不是省略失败记录，而是 collection summary 的实际结果。
- Formal five-package hashes 与第二次完整 rebuild 完全一致；路径身份均为 portable relative paths，大型 Raw/package 保持 local-only。
- Formal action coverage 已消除 development Route/Comp 0/0 数据 blocker；pending Flow 时 Route/Comm 只证明模型路由，不被夸大为已执行服务 transition。
- CPU H=4 runtime 真实经过 optimizer、prior-only H1-H4 validation 与 checkpoint reload；GPU 未执行，formal training 仍由 5.6A 配置冻结/GPU smoke 阻塞。
# 2026-09-23 STEP 5.6A findings

- Training Bundle 不含 Raw；旧 `build_state()` 读取本地 Raw 才得到 wired capacity，导致远端 full-shard batch 无法启动。60/60 Raw 的有线链路结构完全一致，Formal loader 现传入已审计的 100 Mbps 双向当前环境值，未改变 Dataset package。
- 真实未来 Return birth 的 Comm 动作有时没有 current Flow slot；必须按 frozen fixed-support 边界记录 blocked，不能绑定同任务旧 Input Flow。未来目标不进入动作构造。
- prior-only rollout 可能比真实 continuation 更早把已有 Flow 预测为结束；Comm 仍可绑定该锚点原本活跃的 typed current relation，不复活 Flow，也不声称服务已发生。同任务已完成 Input 与当前活跃 Return 必须按锚点 Flow 身份消歧。
- CUDA checkpoint `map_location=cuda` 会把 RNG byte tensor 搬到 GPU；PyTorch RNG restore API 需要 CPU byte tensor。已做最小设备转换并通过 GPU reload/negative identity。
- 现有 development 数值不是研究者冻结的正式训练配置；GPU batch 兼容性只能给工程候选，不能替代训练 seed/LR/预算/阶段/KL 等研究决定。
# 2026-09-24 STEP 5.6A GPU 验收发现

完整 1104-window validation 的四组 serial worker wall 合计 7103.09 秒、shard 数据加载合计 72.15 秒（1.02%）；CUDA event 包含主机调度，不能称纯 kernel 时间。一次 profiler 探针因高 CPU 插桩开销超过 4 分钟而停止，不纳入验收。未训练 smoke checkpoint 的 `L_Val=0.829751` 不构成预测性能结论。CUDA checkpoint 两次重载参数与状态完全一致，H4 输出有微小浮点差异（Motion 8.20e-8、CSI 7.63e-6），在 rtol=atol=1e-6 内一致，不能称逐位一致。
# 2026-09-24 STEP 5.6A-CONFIG-FREEZE findings

原 merged GPU receipt 的 `available_sample_count=0` 是字段语义链问题：批内只保留 `available` 布尔值，merge 又把已聚合 row 当 sample row 处理。修复让每个 batch row 记录同时有 Motion 和 CSI 有效 target 的窗口数，merge 对已有 `available_sample_count` 直接累计；CPU correction receipt 从 1104 个真实 validation target windows 重算，四个 horizon 均为 1104，official loss/metric 不受影响。正式 config 与 Step 5.2 development defaults 分离；没有修改 Dataset、模型、loss 或历史 GPU receipt。
# 2026-09-24 STEP 5.6B 预启动发现

- 首次 detached attempt 在 1 步后因 config byte identity mismatch 终止并保存远端失败凭证。Git archive 将冻结 JSON 的 CRLF 工作树字节规范化为 LF，字段相同但 SHA 从 `a806c320...` 变为 `7c4358df...`；原始冻结 JSON 已单独传输并恢复精确 SHA。runner 新增 freeze receipt 的 byte-SHA 拒绝门，下一次必须是新 run ID。

- 5.6A 实测 H4 batch-8 约 25.026 s/step、完整验证约 7103.09 s；按 H1/H2 horizon 比例推算的最多 5 次验证总时长约 38.65 h，H1/H2 仅为估计。
- 新服务器 driver 为 595.71.05，RTX 4090 24,564 MiB，数据盘约 28 GB 可用；已有训练包沿用原 manifest 与五类 SHA，不上传 Raw。
- 远端当前 source 是旧 5.6A snapshot，无本 Step runner；必须提交后同步精确 Git archive 并校验，不能直接在旧 source 上训练。

# 2026-09-25 STEP 5.6B 中途过程图发现

- 活跃 run 已完成两次各 1104 window 的 prior-only validation，`L_Val` 从 0.1766124568 变为 0.0801012691。H4 Motion raw MAE 从 1.0051 到 0.2689，CSI raw MAE 从 3.8036 到 2.2791 dB；均是同一 validation split 的中途诊断，不是最终测试/泛化结论。
- 训练 loss 在 552、1104、2208 步发生 Stage/Horizon 改变，跨边界的高低不可直接解释成模型退化或改善；需要以同协议的完整 validation 序列判断。
- Motion raw aggregate 混合位置/速度等单位，因此不写成米；CSI raw error 才可写 dB。绘图脚本按 run ID、逐步完整性和 prior-only validation provenance 拒绝不一致快照。
# 2026-09-26 STEP 6.0C：6.0B 源码事实与本次研究者方法决定分离。当前 Raw 的原始单位静态 CPU 容量支持 per-slot operational budget；normalized feature 不可替代。当前 UAV heading(rad)/elevation 需 Planner-only 侧状态；正式采集的零速 HOLD 是显式动作，多 UAV 不同档仅是边际支持。没有新增 AirFogSim native hard bound 或安全声明。
# 2026-09-28 STEP 5.6C 发现

- 五次完整 `L_Val` 为 0.1766124568、0.0801012691、0.0775463209、0.0764608792、0.0743133878；最终 5520 步是唯一最低点。best/latest 文件 SHA 不同，但状态身份相同，428 个模型张量逐项相同。
- 冻结 checkpoint 的 config `device=cuda` 不应改写；CPU 验收仅构造 device=cpu 的等价 Trainer，先严格验证冻结配置/数据/架构，再使用原 base loader 的身份约束重载。单 validation window H1–H4 prior-only 有限值且参数不变。
- 当前只验证一个正式 seed 的 dev_validation；没有 baseline、locked_test、跨 seed 稳定性、Planner rollout 或闭环任务指标，不能由本次 `L_Val` 推出泛化/系统效果。Motion raw aggregate 混合单位。
# 2026-09-28 STEP 6.2A findings

- Deadline 不是绝对时间戳，而是相对允许时长；`arrival_time + deadline` 才是绝对截止时间。DONE 与 hard-failure 的边界分属不同运行阶段，不能用一个 `>=`/`>` 规则替代。
- Observer/Simulator 中存在 deadline、return size、priority 等来源，但 frozen Formal Raw 没有把它们作为 Planner 当前因果 side-state 暴露；不得从 Future Target、FAILED label 或 future schedule 反推。
- 当前 `B_Tx` 目标公式与中间 hop/terminal hop 语义部分一致，但跨 hop E2E remaining、稳定 route/epoch identity 的证据尚未完整；Route effort 也缺少 candidate-independent normalized denominator。
- EnergyManager source 已审计，但能量仍只属于最终闭环指标边界；priority v1 inactive，`w_q=1`。
- 结论：6.2A source semantics/protocol foundation 可记录，但 6.2B readiness 被上述 side-state 与 burden evidence blocker 阻塞。
- 收口验证：6.2A focused 1/1、6.0C 12/12、6.1 2/2、4.4 37/37、4.2C-B Flow ledger 25/25、5.6C checkpoint 3/3、5.5 Formal Dataset 21/21、6.0A adapter/fixed-support 6/6；compileall、knowledge index write/check、Context Consistency 与 diff check 通过。
## 2026-09-28 STEP 6.2A-PATCH findings

## 2026-09-28 STEP 6.2A-ROUTE-RECOVERY findings

- 4.2C-B/C canonical route 是不含当前 holder 的目的节点列表；修复后中间跳完成设置 holder/source=completed destination、index+1、next destination=route[index]、hop_remaining=E2E remaining。
- same-destination Route 通过规则侧 metadata 写完整 path，保留 FlowID/Epoch，RouteRevision 增 1；destination-change 在 fixed object support 下显式拒绝，不伪造新 Flow Epoch。
- Formal Dataset 4416/1104 的 anchor/target/future route 宽度均为 1，multi-hop 直接激活计数为 0。54 个 existing-Flow/Route-overlap 窗口 paired rollout 完全 invariant；这不能证明正式多跳性能。
- checkpoint strict load、428 state-dict tensors、SHA-256 与 parameter digest 全部保持；`CHECKPOINT_NO_RETRAIN_SALVAGE=SUPPORTED_WITH_LIMITATIONS`。

- 4.2B “Raw 无端到端 remaining”只适用于 4.2C-B Ledger 之前；当前 Raw Ledger、Sample/Tensor、Graph/4.4 均有 Flow E2E state。真实 Input 两跳和 direct Return 证据与合成 reroute/Epoch fixture 必须分开。
- 4.2C-C route 数组为剩余目的地、不含 holder；4.4 跨跳索引按含 holder 路径使用。确定性负例显示 hop service >0 且 hop remaining=0，holder/index 未推进。Route action 只改 endpoints/revision，完整 route array 未更新；side-state 只能发现不一致，不能修复冻结模型。6.2B BLOCKED。
- Accepted Formal Raw 含 `required_returned_size` 和 `return_destination_id`；但 AirFogSim `requireReturn()` 还依赖计算节点是否等于 Return 目的地。正 size 不是无条件 Return birth。
- deadline 不在 Formal Raw，但所选非锁定 anchor 可通过 exact-aligned 决策前重放因果获得；训练身份与 checkpoint 未变。
- Route effort 已由研究者删除；Priority 不启用。主业务吞吐是 E2E useful，all-hop network service 仅诊断，两者不能混用。
## 2026-09-28 STEP 6.2A-CLOSURE

- 研究者接受 no-retrain salvage；旧 LVal 保持 legacy observation，patched full validation 未执行。
- Formal train 4416 / val 1104 均无 multi-hop activation；54 个 existing-Flow overlap windows legacy/patched H1-H4 完全 invariant。
- Planner v1 Route enabled 但每条 path 只能一个 frozen destination；multi-hop code repaired and tested, outside v1 domain。

## 2026-09-28 STEP 6.2B

- `rb_active_mask[0]` 是二维 relation×RB，Python `sum(bool(row)...)` 误数 relation 行；实际 global RB 编号来自 AirFogSim `RB_Nos`，当前有效 RB support 在所选 anchor 为 50，旧值 242 仅是历史误计。
- 当前单跳 gate 会接受 pending/no-current-Flow Route，正式 adapter 给 `flow_index=-1`，4.4 只改变 learned latent，不创建 Flow。冻结 Objective 只有 future Return birth 的 `H_sup`，不应自行扩展到 pending Input Route。
- same-path existing Flow Route 在 Flow route fields 不变时仍改 Task-Agent Host；与 holder 在 hop 完成前不变的合同需核对。
- 6.2B scorer 的 fixed-support 机制证据已通过，但完整接受因以上冲突阻塞，不能作 Planner 性能声明。

2026-09-29 STEP 6.3C：静态候选域中 TRAIN 1935/4416 anchor 为空，主要原因是当前 state 无兼容 TRAIN joint structure；该结果是冻结支持政策下的机器事实，不允许自动放宽。非空域 cardinality 可达 313,949,952，因此 lazy/symbolic enumeration 是必要条件。H1-H4 canonical smoke 与旧 6.1 rollout exact equivalent。
## 2026-09-29 STEP 6.3B/6.3C-PATCH 恢复审计

- 未提交补丁把 Comm 从“覆盖所有当前 eligible 任务”改为“按结构签名选择 TRAIN 出现过的任务数量”。冻结 6.3B 合同仍明确要求前者，属于科研语义冲突，不能仅以测试更新掩盖。
- 旧 6.3C TRAIN 静态空域为 1935/4416；未提交补丁产物为 14/4416。精确候选最大数从 313949952 升至 61092600570。这是候选域定义变化，不是搜索性能改进。
- 现有 TRAIN H1 投影自重放产物记录 Comm 3688/4416、完整投影 3404/4416；主要失败类别为当前无线 Flow 因果绑定和 Comp 比例/零基准。脚本的“完整投影通过”仅验证解码、重绑定及结构准入，未逐字段比较原始动作与重绑定动作，不能称精确自重放闭合。
- 6.3B 定向测试 11 项中 3 项失败，断言仍按旧全覆盖语义；6.3C 定向测试 8/8 通过。未执行 GPU、训练或 `locked_test`。

2026-09-30 资料整理发现：正式训练首末 L_Total 不可直接比较（课程 horizon 与 KL beta 变化）；五次 L_Val 最低值 0.07431338784170399 在 5520 步。冻结配置的 formal_training_started=false 属启动前快照，原始运行清单和日志证实随后完成；AI_CONTEXT 当前正式训练摘要无数值冲突。stdout.log 为零字节，实际指标在 JSONL。当前 main 两个训练相关模块有训练后补丁，复现应使用训练源码 SHA 6e15ec2。

## 2026-09-30 STEP 6.3D-GPU-EXECUTION 鈥?宸插畬鎴恅n
RTX 4090 24GB GPU 鎵归噺鎺ㄦ紨闂ㄦ閫氳繃銆傜湡瀹?TRAIN fixtures 鐨?CPU FP32/GPU FP32 batch 1/4/8/16/32/64/128/256 鍗曟鍜?H4 绂绘暎绛変环閫氳繃锛涘悶鍚?batch 8/16/32/64 绋冲畾锛?28/256 OOM銆傛帹鑽?FP32銆乥atch=32銆佹棤 bucketing锛岀害 12.71 transitions/s銆佸嘲鍊?2.03 GiB銆佺害 13.19x CPU batch8銆傛湭杩愯姝ｅ紡 TRAIN tuning銆乂alidation comparison銆佹柟娉曢€夋嫨銆佽缁冦€侀棴鐜垨 locked_test銆傚敮涓€涓嬩竴鍔ㄤ綔锛氱爺绌惰€呭彟琛屾巿鏉冩寮?STEP 6.3D 姣旇緝銆俙n

## 2026-09-30 STEP 6.3D-GPU-PIPELINE-OPT

真实 TRAIN batch=32 profile：model.one_step 87.58%，scorer/H_sup 6.40%，proposal/domain/bind 3.90%，fingerprint/cache 2.12%。主要瓶颈是冻结模型前向；小型 cache/sync 优化不足以达到 20% 门槛。
2026-09-30 3080 Ti 发现：4090 的 GPU fixture 副本缺 58 Raw，经研究者授权以本地 SHA 一致副本补齐；完整 Dataset 与 checkpoint 身份匹配。直接 GPU 缓存随 B_WM 增长可 OOM，精确张量移至主机内存后 B_WM1024 容量通过。batch32/64 短探针虽更快，但 CEM K4/B256 无完整 H4；选 batch16。future Return birth 支持限制不变。

## 2026-10-02 09:09 Stage A 只读备份连接恢复

01:00 UTC心跳中SFTP连接重置（10054），发生在只读source核对；重新连接确认原runner PID1663持续正常运行，未重启搜索。重试后305份完成结果已逐文件SHA备份并独立验身份，无duplicate、source drift、NaN/Inf或scorer exception。此事件属于已恢复的备份连接中断，不是正式搜索崩溃或科研阻塞。Stage A继续RUNNING；唯一下一动作继续监控/备份，Stage B=NOT_STARTED、SEARCH_METHOD=NOT_SELECTED、locked_test=false。机器记录：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/snapshots/20261002T010938Z_backup_connection_recovery_receipt.json`。

### 2026-10-04 02:24 — Step 6.4D 只读监控重连

旧监控SSH连接失活退出，cell338已结束。重新只读核验远端原runner PID2838仍运行，未重启或修改GPU实验；新watch session49023恢复SHA备份，新189/272、覆盖237/320、本地189。后续heartbeat检查watch session49023（不能再wait cell338），若它失活仅重连只读监控。StageB NOT_STARTED、lockedfalse，下一动作继续监控至完成。

- STEP 6.4G: GPU启动前修正仅监控脚本的进程匹配；阳性/阴性识别通过，冻结科研源码 SHA 完全不变，服务器CPU preflight PASS，尚未启动GPU。下一动作：同步此工程修正提交，再启动Phase T。

- STEP 6.4G heartbeat 20261004T052639Z: Phase T 5/768，原runner持续运行未重启，身份/source/每case B512及D:SHA备份通过；A/B未开始、locked_test=false。下一动作：继续只读监控，完成T后独立验收/Git。

- STEP 6.4G heartbeat 20261004T053621Z: Phase T 8/768，原GPU进程正常，未重启；全量已完成raw身份/预算/source/D:SHA通过，A/B未开始、lockedfalse。下一动作：只读监控至T768后独立验收/Git。

- STEP 6.4G 20261004T054416Z用户进度查询：T 12/768，约24.1cases/h，早期剩余估算31.3h；原runner正常，完成raw身份/源码/预算/D:SHA通过，A/B未开始、lockedfalse。下一动作：继续监控和备份。

- STEP 6.4G 20261004T094701Z用户进度查询：T 109/768，本地验身份/SHA 91；速度23.9cases/h，剩余粗估27.6h，原runner持续运行未重启，A/B未开始、lockedfalse。下一动作：监控与备份至T完成。

- 20261004T094742Z 本轮SFTP已结束，T109份全部完成D:SHA备份及独立身份/预算复核；原GPU进程正常，未启动A/B。

- STEP6.4G 20261004T131015Z 进度查询：T190/768，原runner正常，无STOP，全190 raw source/身份/B512及D:SHA备份通过；速度24.1cases/h，剩余粗估24h，A/B未开始、lockedfalse。下一动作继续只读监控备份至T完整验收。

- 2026-10-05T14:12:05.218488+00:00：T768/768远端独立验收PASS，实际GPU矩阵耗时32.6小时；上次仅SFTP备份连接10054中断，原GPU未重启、已自然完成。现只读补齐会话56512仍运行，本地独立验收/Git未完成，A/B未启动、lockedfalse。下一动作：备份完整后本地重建统计。

- 20261005T142333Z：PhaseT768/768本地独立重建、ZIP SHA、67 CPUtests、compile/index/Context/diff通过；T阶段已commit+push757de2f5f5b9dbeea6117e5f7fee094d6e78cfec。新S4/rho.2、MH3/rho.1；A960 config a7c793ef7619e107fe7f3a5e3fdf16cffb8ae64af76b35422ee613fb69914db8。服务器deploy在git fetch25s超时，未执行remote ff/preflight、未launch A，T原GPU已经自然停止。此为工程网络门，下一动作只读确认远端Git状态、同步exact gate后再一次启动A，严禁重启T。

- 20261005T143328Z：Git超时已恢复，新A开始14:31:35UTC(北京时间22:31)，exact gate757de2f、config a7c793ef...，HRS/S4rho.2/MH3rho.1，RTX3080Ti FP32 batch16，trackedclean/results0；单runner，不重跑T，不启动B，lockedfalse。下一动作只读监控与D:SHA备份。

- 20261005T143530Z PhaseA实际启动与单进程已核，当前首case处理中；SFTP配置/启动资源SHA备份通过。自动监控pi-jwm-6-4g已update成功ACTIVE，新prompt明确T已PASS、A gate757/configa7c、新S4rho.2/MH3rho.1，避免重启T/A。下一动作只读监控A960。

- STEP6.4G 20261005T143753Z heartbeat：新A首case已完成，1/960，本地raw/source/identity/B1024/D:SHA通过，单runner正常、无STOP；T已PASS、B未开始、lockedfalse。仅早期diagnostic不作选法判断，下一动作继续只读监控备份。

- STEP6.4G 20261005T144754Z heartbeat：A5/960，单runner正常无STOP、全完成raw身份/source/B1024/D:SHA通过；T PASS、B未开始、lockedfalse。下一动作继续只读监控和备份，不根据早期结果选法。

- STEP6.4G 20261005T145755Z heartbeat：A9/960，原单runner正常无STOP，全raw身份/source/B1024/D:SHA通过；T PASS、B未开始、lockedfalse。下一动作继续只读监控与备份，不从早期case选法。

- STEP6.4G 20261005T150802Z heartbeat：A13/960，原单runner正常无STOP，全raw身份/source/B1024/D:SHA通过；T PASS、B未开始、lockedfalse。下一动作继续只读监控和备份，不根据早期结果选法，不关机。

- STEP6.4G 20261005T151809Z heartbeat：A16/960，原单runner正常无STOP，全raw身份/source/B1024/D:SHA通过；T PASS、B未开始、lockedfalse。下一动作继续只读监控与备份，不从早期case选法、不关机。

- 20261005T152515Z 用户询问耗时及为何960：已解释T768是搜索选参非WM重训，A960为原6.4G授权64×5×3正式选法；A18/960正常、raw身份/预算/D:SHA通过。下一动作继续监控，不改变协议，lockedfalse。

2026-10-08T04:25Z 6.4H只读监控：19/320，原PID1615单runner；本地19 raw身份/预算/参数/SHA与当前source/checkpoint核验PASS，实际9728转移，无STOP；继续等待唯一B512矩阵，不重启、不运行其他实验、locked_test=false。

2026-10-08T04:55Z 6.4H只读监控：31/320，原PID1615单runner；31 raw本地SHA/identity/预算/参数核验PASS，实际15872转移，无STOP。继续等待当前B512，不重启、不扩实验，locked_test=false。

2026-10-08T05:25Z 6.4H只读监控：43/320，原PID1615单runner；43 raw本地SHA/identity/预算/参数核验PASS，实际22016转移，无STOP。继续等待当前B512，不重启、不扩实验，locked_test=false。

2026-10-08T05:28Z 用户进度核验：45/320，原PID1615单runner；45 raw身份/SHA/预算/参数与source/checkpoint PASS，23040实际转移，无STOP；当前平均27.5cases/h，剩余搜索粗估10h（不含最终验收）。继续仅当前矩阵，locked_test=false。

2026-10-08T09:36:18.317121+00:00 用户进度核验：152/320，原PID1615单runner；152 raw身份/SHA/预算/参数与source/checkpoint PASS，77824实际转移，无STOP；当前平均26.6cases/h，剩余搜索粗估6.3h（不含最终验收）。继续仅当前矩阵，locked_test=false。

2026-10-08T11:56:39.719780+00:00 用户进度核验：215/320，原PID1615单runner；215 raw身份/SHA/预算/参数与source/checkpoint PASS，110080实际转移，无STOP；当前平均26.7cases/h，剩余搜索粗估3.9h（不含最终验收）。继续仅当前矩阵，locked_test=false。

2026-10-08T14:44:40.900668+00:00 用户进度核验：293/320，原PID1615单runner；293 raw身份/SHA/预算/参数与source/checkpoint PASS，150016实际转移，无STOP；剩余搜索粗估1h（不含最终验收）。继续仅当前矩阵，locked_test=false。
2026-10-09 STEP6.4J: The previous freeze lacked a runnable Pilot entry point. The new runner wires LiveSCEMPlanner, EpisodeController, AirFogSim collection and real metric ledger; local CPU gates pass. Remote CUDA execution remains pending exact commit deployment.
## 2026-10-10 STEP 6.4J — r29 GPU Pilot 预检证据

- 远端 `connect.nmb2.seetacloud.com:35569` 认证和 TCP 握手成功。
- HEAD 已精确检出 `93b3e806ee119545dd487fa6c7a8aa12a86c8390`；tracked 文件无 diff，但存在未跟踪 `step6_4j.bundle`。
- GPU/资源：RTX 3080 Ti，CUDA 可用，约11.9GiB显存空闲，约44GiB可用磁盘，无旧 Pilot 进程。
- `python code/scripts/verify_step6_4j_deployment_v1.py`：FAIL `SOURCE_SHA_MISMATCH:code/src/pi_jwm/step6_4i_episode_v1.py`。协议记录 `e3cc96d7...`，远端 checkout 文件 SHA256/Git blob对应 `8b185979...`。
- 运行时 `/root/miniconda3/bin/python3.12` 的 Torch 2.8.0+cu128/CUDA可用，但 `osmnx` 和 `traci` 缺失。
- 结论：BLOCKED/未执行；无 attempt、journal、decision 或备份结果。
