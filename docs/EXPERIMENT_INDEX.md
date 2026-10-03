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

# PI-JWM 实验索引

> 2026-10-01 当前：PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。

> 2026-10-01 当前：STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。

2026-10-01 STEP 6.3D 正式 TRAIN 调参：RTX 3080 Ti、冻结 FP32 batch16、32 TRAIN anchors×3 seeds×S-CEM/MH-CEM×4 configs，共 768/768 cases；名义/实际 unique transitions 各 393,216。原始求解与日志保存于本机 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/solve_results/train/`，同目录有逐组/逐锚点诊断、选参/验收收据及 ZIP/SHA manifest。两方法各选 `(K=4,rho=0.1)`；Validation 0、方法未选、`locked_test=false`。见 `docs/implementation_records/STEP_06_3D_FORMAL_TRAIN_TUNING_CLOSURE.md`。这是 TRAIN 选参，不是方法效果比较。

STEP 6.3D-PREFLIGHT-PATCH CPU 前置诊断：保留历史清单，静态替换 3 TRAIN / 5 Validation 零 cohort 样本并精确重放新 deadline sidecar；32 TRAIN 上固定 HRS seed6391/B_WM64 完成 2048 次 unique 一步转移的 H4 支持诊断。另在真实 TRAIN 样本检查 batch 1/4/8/16 的一步/H4 等价与 64 转移吞吐。机器收据 15–25 位于 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/`，解释见独立 Patch 实施记录。没有 Validation 搜索或三方法正式比较。

STEP 6.3B/6.3C-PATCH CPU 合同与投影审计：Formal TRAIN 4416 个 H1 锚点逐项记录 raw historical action、Planner-v1 projected action 和排除行；TRAIN/Validation 静态候选域分别审计 4416/1104 个锚点，Validation 只作描述。机器入口 `code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929/`，解释见 `docs/implementation_records/STEP_06_3BC_PATCH_COMM_TASK_SELECTION_TRAIN_SELF_REPLAY_AUDIT.md`。它不是候选排序或系统性能实验。

STEP 6.3A CPU-only support audit：Formal TRAIN 48 trajectories/4416 windows，validation 12/1104 descriptive-only。Comp 1969/1969 action entries 从 causal CPU rule 重建，alpha `{0.5,0.75,1.0}`；Comm/Mob/joint/temporal 和 anchor search-size receipts 见 `code/artifacts/protocols/pi_jwm_step6_3a_candidate_support_audit_v1_20260928/`。不是候选选择或性能实验。

STEP 6.2B-PATCH CPU 机制诊断：同一非锁定 validation `anchor-0001`、冻结 best.pt、两个 Route 空动作候选的 H1–H4 rollout 与五项 scorer；新机器入口 `code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928/`。这是合同测试，不是候选方法或性能实验。

STEP 6.1 CPU 机制诊断：正式 validation 样本 `...::anchor-0001`、冻结 `best.pt`、四类当前因果探针、H1–H4 递归、3 种配对随机 seed 与 K=1/2/4/8 CPU 实现计时。机器入口 `code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/`；实施记录在 `docs/implementation_records/STEP_06_1_TRAINED_WORLD_MODEL_CANDIDATE_ROLLOUT_PREFLIGHT.md`。非正式效果实验，不生成候选排名。

STEP 5.6B 正式训练 run `pi_jwm_formal_train_v1_seed5601_20260924T112424Z` 已由 STEP 5.6C 本地 CPU 验收：5520 步、五次完整 prior-only validation、最终 best checkpoint SHA 冻结。机器入口 `code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928/`，大文件在 local-only `code/artifacts/formal_training/`。这是 Formal Validation Observation，未做 locked-test、baseline、Planner rollout 或性能声明。

STEP 5.5-PATCH 是正式数据消费路径和语义审计，不是训练/性能实验；机器凭证见 `code/artifacts/audit/pi_jwm_step5_5_patch_20260923/`。原 STEP 5.5 H4 smoke 只消费 `runtime/` 的 1+1 mini subset，PATCH 另行验证 4416/1104 full-shard 可索引及跨轨迹 CPU batch。

> 本索引负责回答“实验做过没有、它回答什么问题、结果在哪里”。具体结论必须回到原始实验目录和机器可读产物。

重要实验的结构化记录见 `docs/registries/experiment_registry.json`；全部 artifact 一级目录的自动目录见 `docs/registries/generated/artifact_catalog.csv`。注册表 v2 对每条重要实验使用统一字段；未知或不适用的内容必须写成 `null` 并在 `field_notes` 解释，不能直接省略。

STEP 5.5 Formal Dataset v1 acceptance 不是性能实验：60 条真实 trajectory、H2/L4、四动作覆盖、五类 package、确定性重建和 CPU H=4 interface smoke 的可追踪证据位于 `code/artifacts/manifests/pi_jwm_step5_5_formal_dataset_v1_20260923/`。

> 2026-09-19：新 `00–06` 已完成 Raw Trajectory Layer 的真实接口验收，但没有新训练实验。下列 P4/P6 条目全部按旧定义作 Historical / Archived evidence。

Step 2.4 真实接口验收不属于训练实验：机器证据位于 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/`，覆盖 6 个 execution slot、7 个独立 Decision 和 14 项检查。

Step 3.1F 最小样本与 History 修正也不属于训练实验：证据位于 `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`，Raw source 为真实 v2 communication artifact，覆盖 `H=2/L=2`、24 项合同检查、12 项 focused tests、past action/outcome、History union index、Future Action ID↔index 对齐、history relation/DAG/flow 和 round-trip；`future_action_reference_audit.json` 对 4 个非 locked Raw artifact 的 18 个窗口做 observation-only 扫描，sample manifest 保存其 SHA-256 provenance；`locked_test=false`、training/gpu=false。

## 1. 旧协议正式实验（Historical / Archived）

| 实验 | 目的 | 入口/配置 | 结果和状态 |
| --- | --- | --- | --- |
| P4 entity RSSM seed 20260831 | 验证实体级双图 RSSM 在正式 unlocked tensor 上的单 seed 表现 | `code/scripts/run_formal_dual_graph_gpu_train_v1.py`；冻结协议 `code/artifacts/protocol/pi_jwm_p4_entity_rssm_frozen_protocol_20260906_v2/protocol.json` | `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260831_v1/`；单 seed 验收通过 |
| P4 entity RSSM seed 20260830 | 在相同冻结配置下检查另一 seed 的稳定性 | 同上；正式 run `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1/` | 已完成；单 seed 验收通过，audit 位于 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/` |
| P4 entity RSSM seed 20260832 | 第三个 seed 的跨 seed 证据 | 同上 | 未授权，不得自动启动 |

`20260832` 和后续远端同步已经登记在 `docs/registries/deferred_work.json`，状态为 `deferred`、`authorization_required=true`、`auto_start=false`。接口预留不等于运行许可。

## 2. 正式前置和机制实验

| 实验 | 所回答的问题 | 证据位置 |
| --- | --- | --- |
| 第一性原理审计 | 旧 global RSSM 是否能表达实体差异、运动和链路排序 | `code/artifacts/audit/pi_jwm_p4_first_principles_audit_20260906_v1/` |
| CPU 一致性审计 | 实体级 prior/posterior、运动合同、梯度和 strict reload 是否真实接通 | `code/artifacts/audit/pi_jwm_p4_entity_rssm_cpu_consistency_20260906_v2/` |
| GPU batch probe | batch 8 是否在显存边界内可执行 | `code/artifacts/audit/pi_jwm_p4_entity_rssm_gpu_batch_probe_20260906_v1/` |
| GPU execution sentinel | 正式 runner 是否能在 CUDA 执行两阶段训练 | 对应 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_sentinel_20260906_v1/` 和 audit |
| 候选规划器 audit | 逐候选 rollout 机制是否具备可审计骨架 | `code/artifacts/audit/pi_jwm_formal_candidate_rollout_planner_audit_20260826/`；当前 blocked/原型边界 |

## 3. P4 历史候选族

这些实验保留用于解释失败原因，不得自动提升为当前方法：

重要历史方法的“为什么尝试—实际结果—为什么不再使用—被谁替代—原始证据”见 `docs/registries/historical_method_registry.json`。

- global complete RSSM：`code/src/pi_jwm/formal_complete_rssm_world_model_v1.py`，对应 `code/artifacts/experiments/pi_jwm_p4_complete_rssm_*`。
- node-x safe correction：对应 `code/artifacts/experiments/pi_jwm_p4_complete_rssm_node_x_safe_*`。
- edge feedback GRU：对应 `code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_*`。
- persistence residual：对应 `code/artifacts/experiments/pi_jwm_p4_link_persistence_residual_*`。
- 旧概率校准 Scheme B：对应 2026-09-01 的校准审计和失败记录。
- h20、threshold、throughput、position、link recall 等诊断：对应 `code/artifacts/audit/pi_jwm_p4_*diagnosis*` 和相关 `记录/研究进展/`。

## 4. 早期阶段实验族

| 阶段/前缀 | 主要内容 | 当前口径 |
| --- | --- | --- |
| P0/P1/P2 | 理论一致性、信息边合同、采集器、Attempt/Reject Ledger、正式数据 | 当前数据和合同的历史来源，按最新冻结版本读取 |
| R3/R4/R5 | 世界模型候选筛选、模块预检和 GPU screening | 历史候选证据，不覆盖 P4 |
| R6 | belief-conditioned direct policy 和闭环策略预检 | 不是候选动作世界模型规划器 |
| v6/v7/v8/v11 | 旧双图、active-rate、selector、收益可辨识性和策略实验 | 历史研究线，失败结果必须保留 provenance |

## 5. 新实验登记要求

新增重要实验时，至少登记：

- 实验 ID 和日期；
- 它回答的研究问题；
- 方法/模型身份；
- 代码入口和代码版本；
- 配置、数据/tensor manifest、split 和 seed；
- checkpoint、原始 metrics、audit 和 SHA-256；
- 结果状态：`running`、`passed`、`failed`、`blocked`、`historical`；
- 当前结论、证据范围和不能推出的结论；
- 是否访问 `locked_test`。

新增实验只能在用户批准研究目的和方法变量后执行。索引登记不等于实验通过。
## 2026-09-28 STEP 6.2A-PATCH diagnostic only

One exact-aligned non-locked Formal Validation trajectory replay captured current Task deadlines; a deterministic 4.4 transition reproduced the destination-only route-index mismatch. Receipts: `code/artifacts/protocols/pi_jwm_step6_2a_patch_objective_readiness_v1_20260928/`. No performance experiment or baseline was run.
STEP 6.3D-3080TI-MIGRATION-QUALIFICATION：正式数据迁移、真实 TRAIN CPU/GPU 等价、batch 8/16/32/64 三次稳态吞吐、CUDA formal runner bounded HRS/S-CEM smoke。机器收据见 `code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/`；正式 TRAIN tuning 与 Validation 比较未运行。
