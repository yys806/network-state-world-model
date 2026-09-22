# 已知问题与冲突

Source of truth：本文件是入口；具体事实必须回到列出的代码、配置、测试或 artifact。

## STEP 5.2 当前边界

- STEP 5.2 CPU development training loop 已实现并通过 20/20 receipt checks；这只证明 Stage 1/Stage 2、optimizer smoke、prior-only validation 与 checkpoint/resume 的工程链路，不证明训练收敛、泛化或性能。
- 当前只使用 unified non-locked development bundle（`dev_train=8`、`dev_validation=4`）。Route/Comp non-empty coverage 均为 0，仅有 explicit no-op；正式训练数据覆盖仍需后续 gate。
- `training_loop_implemented=true`、`cpu_optimizer_smoke=true`，但 `full_training=false`、`gpu=false`、`formal_dataset=false`、`locked_test=false`、`performance_claim=false`。下一独立门是 STEP 5.3，不在本轮自动执行。

## STEP 5.1A/5.1D target boundary

- 初版 STEP 5.1A 的 horizon 2+ Motion anchor 与 future-row-order Motion alignment 已确认错误；旧 COMPLETE/FROZEN 证据被 PATCH 取代。当前代码和 non-locked development receipt 已覆盖 local-step semantics、current physical slots 和 current model CSI relation slots，PATCH target contract 已重新冻结。
- Future Target 保持 additive namespace；5.1D 只在 target encoder/posterior/decoder/loss 路径读取它，STEP 4.3B encoder 和 STEP 4.4 current prior 不读取 target。
- Loss/posterior/KL/metric 已有 CPU paired integration evidence；STEP 5.2 已补齐 CPU training-loop/optimizer smoke，但 full training、GPU、planner 和 `locked_test` 仍未开始，artifact 不是正式 Dataset 或性能证据。

## Step 2.4 raw boundary observations

- AirFogSim reports vehicle `angle` in degrees and UAV `angle`/`phi` in radians; the contract distinguishes observation `heading` from UAV action `azimuth_rad`.
- Across the real trace, the simulator can report `-100.0 m/s^2` when canonical `(v_t-v_{t-1})/delta_t` is about `+100.0 m/s^2`. The simulator remains unchanged; fields are now explicitly raw versus canonical.
- Some live nodes do not expose a `cpu` key in `FogProfile`; Raw records `null + observed_mask=false + CPU_NOT_EXPOSED_IN_FOG_PROFILE`. Dataset/Tensor must preserve this missingness.
- Historical Step 2.4 boundary: Raw Trajectory Layer was frozen while Dataset/Tensor and graph/model/loss/planner remained unverified. Current update: Dataset/Tensor and STEP 4.3A Typed Graph Builder are now frozen; Encoder/model/loss/planner remain unverified.

- AirFogSim cloud profile key mismatch remains an observed simulator/config limitation: `cloudServer_4` may expose no `cpu` key in the example profile, so the runner records Comp no-op rather than fabricating capacity. This is outside Step 2.4 communication semantics.

## 1. 新定义仅部分实现

- Documented Intent：最新 `00–06` 要求严格 Physical/Information 双图、Route/Comm/Comp/UAV 四类动作、主要面向 Physical/Communication 未知动态的 RSSM、逐步学习—规则—动态图闭环和真实反馈重规划。
- Actual Implementation：STEP 4.3A 已实现严格 typed graph，STEP 4.3B 已实现 Definition 03 encoder 与 aligned `Z_t^{PI,L_g}`，STEP 4.4 与 5.1D 已接入新的 World Model/loss CPU path；旧模型代码仍使用混合语义 `physical_edge`，不属于当前新链路。Training Loop 和 planner 尚未接管。
- Evidence：`docs/PIJWM_IMPLEMENTATION_TRACKER.md`、`docs/implementation_records/STEP_01_AUDIT.md`、`STEP_01_DATA_GRAPH_AUDIT.md`。
- Affected Files：旧 tensor/model/loss/training/planner、AI_CONTEXT 旧 P4 描述和旧 checkpoint/result。
- Conflict：STEP 4.3A builder acceptance 只能证明 current graph representation，不能证明 Graph Encoder、World Model 或完整新定义已经实现；旧接口、测试或两个 seed 验收也不能补足该证据。
- Status：Raw Trajectory / 01、当前最小 Dataset/Tensor / 02、STEP 4.3A builder、STEP 4.3B encoder、STEP 4.4 World Model、5.1B primitives/paired integration 与 5.1D CPU closure 已完成并冻结；Training Loop、Planner 与正式性能仍等待后续授权。

## 2. 研究边界仍需决定

- 通信状态是否足以规则计算 service、外生 task/entity/background load 的未来边界、proposal 训练方式、planner objective/risk/hard constraints/fallback 尚未冻结。
- Codex 不自行选择这些科研定义；相关实现保持停止。

## 2A. STEP 4.2A 当时的图输入缺口（后续状态见 2C-B 与 STEP 4.3A）

- STEP 4.2A 已暴露：position、wireless per-RB CSI、CPU static capacity、已有 Task demand/progress/time fields、wired typed relation 与 Src/Host/Exec/Ret。canonical motion direction 所需 heading/elevation 尚未加入本轮最小 extension。
- Simulator observer 已有但冻结 Raw 未暴露：return size、priority、deadline。`_extract_tasks()` 有真实 getter/TaskSnapshot 来源；后续是否透传属于下一合同，不是本 Patch 的数据实现。
- 当前 Raw 仍不足：可选 wired live queue/load/utilization（不是 03 minimum）；具有 stable ID、Input/Return/DepData type、endpoints、presence、total/rem、multi-hop/route-revision identity 与动作前因果性的 current stateful Flow；dynamic available CPU。
- Wireless structural relation validity 与 CSI observability 已解耦；missing CSI 不得删除 relation，mask=false 的 tensor placeholder 必须为 0。wired valid/no-CSI 同样合法。
- CPU capacity、`A_t^Comp` allocation、Outcome actual service 与 dynamic available CPU 是四种不同语义；不得互相替代。
- STEP 4.3A 使用“当前 presence + 有效 XYZ + 显式 membership policy”物化 Physical representation，没有硬编码 edge/cloud 类别结论；最终 radius/kNN topology 仍未 research freeze。
- 历史影响（STEP 4.2A 当时）：不得直接实现 graph builder；不得用旧 mixed `physical_edge_state`、past hop service 或 outcome 指标填补当前状态。该停止门已由后续数据闭合和 STEP 4.3A 授权解除，但禁止伪造字段的边界继续有效。
- Evidence：`docs/contracts_PIJWM_STEP_04_2A_GRAPH_INPUT_ADDITIVE_EXTENSION_V1.md` 与 Step 4.2A artifact。

## 2B. Stateful Flow source audit

- 当前 AirFogSim `Task._transmitted_size` 是阶段/当前 hop 累计量，完成 hop 后 reset；不能当作定义 03 的 end-to-end current remaining。
- 没有 simulator-issued stable Flow identity 或独立 Return identity；`LogicalFlow`/`CarryingHop` 是动作侧对象，不能替代运行时 provenance。
- `_task_dependencies` 只做 DAG completion gating，当前没有 DepData payload/transfer event。dynamic available CPU、storage、wired queue/load/utilization 也没有可靠 decision-time Raw source。
- Evidence：`code/artifacts/protocols/pi_jwm_step4_2b_stateful_flow_source_audit_v1_20260920/stateful_flow_source_audit.json`；综合 verdict=`FLOW_CONTRACT_NOT_YET_SUPPORTED`。
- STEP 4.2B-PATCH：顶层 verdict 已由 Flow-specific evidence 实际计算并加入篡改负例；其他 graph input gaps 不再参与 Flow verdict。真正的 Flow blocker 是 Input/Return 跨 multi-hop 的动作前 current remaining source-of-truth，以及尚未冻结的 identity/type/端点/route semantics。

## 2C-A. Causal Flow Ledger feasibility

- 真实 transfer event 可证明 task/phase/hop/端点和 delivered service，但不能单独证明 logical end-to-end remaining 或 final-destination delivery。
- 多 hop invariant 只统计最终目的地交付；中间 hop service 不得再次累加。reroute 仍缺 payload holder、保留/重传语义的 causal event。
- 4.2C-A-PATCH 证明 logical destination 已确定时，E2E remaining/final delivery/current holder/same-destination reroute 可由 existing event/state + audit-only replay 派生；`flow_completed` 仅是 stage/hop 语义，禁止当 logical completion。
- 当前 verdict=`CAUSAL_FLOW_LEDGER_FEASIBLE`，由 `ledger_specific_required_evidence` 计算；篡改 verdict 会被 validator 拒绝。destination-change epoch inheritance 和 DepData process 仍是 researcher decision，DAG 不得生成 fake Flow。
- Evidence：`code/artifacts/protocols/pi_jwm_step4_2c_a_causal_flow_ledger_feasibility_patch_v1_20260920/`。在 STEP 4.2C-A 当时长期 Ledger、Raw extension 和 Graph Builder 尚未授权；之后已分别由 4.2C-B 与 4.3A 完成。

## 2C-B. Ledger / Raw 已实现后的剩余边界

- Causal Flow Ledger 与 Raw additive Flow state 已实现；上述“长期 Ledger、Raw extension 未授权”是 4.2C-A 时点的历史描述，已由 4.2C-B 覆盖。
- 真实 non-locked trace 已覆盖直接 Input/Return 与独立 Input 两跳；尚未真实观察 Return multi-hop、same-destination reroute、destination-change Epoch 和 local execution no-flow，这些目前只有 contract fixture evidence。
- 旧无线 Return hook 的 delivered amount 可超过 observer return_size；Ledger 按冻结 min rule 封顶守恒。该 observation 不等于修改 simulator，也不能外推为正式 Dataset 结论。
- Flow Sample/Tensor additive extension 已在 STEP 4.2C-C-PATCH 完成并冻结；其后的 STEP 4.3A Graph Builder、STEP 4.3B Graph Encoder 与 STEP 4.4 World Model Contract 也已完成。Loss、Planner 和训练仍未开始。
- Input multi-hop logical destination continuity 已由真实 trace 闭合；Return multi-hop 和 same-destination partial-hop reroute 仍缺真实 runtime evidence。后者不得被当前 contract fixture 描述成 simulator 已支持。

## 3. 旧 P4 尚未闭合（Historical / Archived）

- Actual Implementation：当前正式 runner 支持三个冻结 seed，前两个已完成。
- Evidence：两份 `single_seed_acceptance.json` 均通过；`docs/registries/deferred_work.json` 将第三 seed 标为 deferred。
- Conflict：两个单 seed 证据不足以形成三 seed 结论。
- Status：保留旧协议边界；`20260832` 不在当前 active queue。
- Boundary：`locked_test_accessed=false`、`formal_performance_claim_ready=false`。

## 4. 全量测试存在历史/环境错误

- 2026-09-10 基线：1643 项为 0 assertion failure、17 errors。
- 已确认类别：缺少 AirFogSim `traci` 环境；旧 teacher fixture 与严格 RB 事件合同不符；旧 directed-dynamic fixture 无有效 calibration link 样本。
- 相比 2026-09-09 的 21 errors，历史 R5/R6 artifact 权限相关 4 个错误在当前权限下消失；这不是科研代码或实验结果变化。
- 影响：不能用全量套件证明物理迁移旧代码完全无回归。
- 不应采取：放宽当前合同、删除测试或修改科研逻辑来制造表面全绿。

## 5. 历史 artifact 控制文件读取曾受限

2026-09-09 受限环境下机器 catalog 曾记录 14 个历史 manifest/control file 不可读；不可读不等于文件不存在或实验失败。2026-09-10 在当前权限下重新生成后，802 个 artifact 记录的读取错误数为 0。该差异属于运行环境权限变化，不是实验状态变化。

## 6. 冻结协议状态字段是历史时刻

`protocol.json` 的 `status=ready_for_gpu_batch_probe` 表示协议冻结时状态；当前运行完成情况必须看后续 run 与 acceptance，不能只读这个字段。

## 7. 组会 PPT 结果滞后

`meeting/PI-JWM_组会汇报.pptx` 的结果页使用 seed 20260831 epoch 22 中间证据，晚于它的正式 epoch 39 结果和第二 seed 结果。PPT 不能作为当前两 seed 的最新验收来源。

## 8. 旧图术语冲突已由新定义替代，模型迁移未完成

- Documented Intent：部分较早材料把通信关系统称为信息边。
- Actual Implementation：`physical_edge_state` 表示有向通信链路；`flow_state` 表示任务数据流信息边；agent 没有独立观测张量。
- Evidence：`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`、`formal_airfogsim_window_v1.py`、当前 tensor contract。
- Affected Files：较早理论材料、旧 PPT 与后续论文表述。
- Conflict：最终理论术语如何命名仍属于科研决策。
- Status：目标定义已由最新 `00–06` 给出；typed Graph Builder、Graph Encoder 与 untrained World Model Contract 迁移已由 STEP 4.3A/4.3B/4.4 完成；Loss/Training/Planner 尚未迁移。

## 冲突记录模板

遇到新重大冲突时必须记录：`Documented Intent`、`Actual Implementation`、`Evidence`、`Affected Files`、`Conflict`、`Status: Awaiting Researcher Decision`，并停止相关科研逻辑修改。

Unverified：没有代码、config、experiment 或可读 audit 支持的问题只能标记为待核验，不能写成确认缺陷。
- STEP 3.1R 已关闭原 DAG source 判断错误：observer 已提供真实 DAG rows，Raw `_capture()` 已接线；当前样本过滤两端不在 anchor input task namespace 的 future-only edges。正式 batch/split preprocessing、数据规模和模型输入选择仍未验收。
- STEP 3.1F-PATCH 已修正 Future Action 的 anchor-only 重编号：anchor visibility 与 History-union numeric index 已分离，validator 和 disappearing-object fixture 已覆盖。4 个非 locked Raw artifact 的 18 个窗口扫描暂未发现 unresolved future reference；该短样本观察不能代替正式 dataset 可用率；未来对象到达的建模方案仍未决定。
- STEP 4.2C-C-PATCH 当时已解决 Flow Sample/Tensor 贯穿与 normalization/semantic completeness；其“Graph Builder Contract 未冻结”边界随后由 STEP 4.3A 关闭。Return multi-hop、same-destination partial-hop reroute runtime 与 formal capacities 仍未冻结；4.2C-C artifact 本身仍只支持 Raw→Sample→Tensor 合同。
- STEP 4.3A/4.3B/4.4 已冻结 typed Graph Builder、Encoder 与 untrained World Model Contract，但 Physical topology mode/radius/k 仍只是 development config；Return multi-hop、same-destination reroute runtime 和 formal graph capacities 仍未获得更强证据。5.1D 已有 CPU loss/KL/metric integration evidence，但 Training Loop、World Model 性能和 formal acceptance 尚未验证。

## 2026-09-19

The real `airfogsim` conda environment completed Step 2.1–2.3 acceptance. The earlier optional-dependency note is historical and no longer blocks Raw-layer verification.
# 2026-09-19 STEP 3.2 boundary

STEP 3.2-PATCH has finalized machine-readable Dataset isolation evidence for the three development trajectories. This remains observation-only and non-locked: formal dataset ratio, Tensor, model, training, GPU, and locked_test are still unopened. Task size is intentionally recorded as `AirFogSim data-unit` because no verified bit/byte conversion exists.

- The small bundle covers only three short development trajectories and cannot support formal split-ratio, scenario-coverage, generalization, or Dataset claims.
- Tensor/model/loss/planner integration remains unimplemented; `formal_performance_claim_ready=false`, `gpu=false`, `training=false`, `locked_test_accessed=false`.
# STEP 4.4 remaining boundary（2026-09-21）

The previous outage blocker is resolved: outage is an independent known stochastic event, and learned service residual is closed. STEP 4.4-PATCH3 is COMPLETE / FROZEN. Frozen current-side input cannot determine Return requirement when no typed Return slot exists; the adapter preserves this as unknown and blocks false final completion. Future-only Return Flow birth remains explicitly unsupported because v1 cannot create new object slots. Definition 05 must mask, exclude, or classify windows crossing that boundary. Model accuracy, posterior/prior training loss, KL/overshooting, calibration, formal capacities, Planner behavior, and performance remain unverified. Physical topology parameters remain development-only; Return multi-hop and same-destination partial-hop reroute retain their prior evidence limits.
# STEP 5.1B-PATCH Definition 05 implementation boundary（2026-09-22）

- Decisions are frozen and the 5.1B-PATCH CPU Loss/Posterior/Metric primitives are COMPLETE/FROZEN FOR CPU DEVELOPMENT PRIMITIVES + PAIRED INTEGRATION through 5.1D.
- 5.1A target tensor now carries normalized Motion and future per-RB CSI with explicit masks; 5.1D consumes them only in target encoder/posterior/loss.
- STEP 4.4 real state uses raw position while Comm CSI follows the normalized graph path; the paired receipt tests the normalized-loss/raw-rule bridge. Unified support is 10/74.
- Historical loss/runner/checkpoints are incompatible as complete implementations because they use old NLL/downstream losses/KL balancing/overshooting/staged freezing/P4 selection semantics.
- Therefore training is NO-START, not a GPU blocker. Receipt evidence remains development-only: `training=false`, `optimizer_step=false`, `gpu=false`, `formal_dataset=false`, `locked_test_accessed=false`, `performance_claim=false`.

# STEP 5.1C-PATCH remaining boundary（2026-09-22）

- Unified identity and normalization lineage is closed for the 12 non-locked development windows, and 5.1D proves rebuilt 4.3A/4.3B/4.4 execution on that bundle.
- Historical 8/44 model artifacts remain incompatible with the 10/74 target contract; the new unified bundle is the only paired development evidence.
# 2026-09-22 STEP 5.1C lineage alignment

- Historical model artifacts and STEP 5.1A targets came from different development lineages (5 vs 12 samples; 8/44 vs 10/74 support). The old 5.1B receipt reused one carrier and prefix-truncated targets; that receipt is not acceptable for closure.
- All 12 target windows now pass the real 4.2A graph amendment and 4.2C-B/C Flow path. Additive unified bundle: `code/artifacts/protocols/pi_jwm_step5_1c_unified_development_bundle_v1_20260922/`.
- Remaining issue: unified 4.3A/4.3B/4.4 paired rebuild and 5.1B model/target pairing are now closed by 5.1D-PATCH. Exact upstream train lineage and runtime prior-target isolation passed; the evidence remains CPU/non-locked and does not authorize training automatically.
# STEP 5.1D boundary（2026-09-22）

Unified chain 已闭合为 untrained CPU development evidence。真实 12-sample bundle 的 Route/Comp non-empty coverage 均为 0，仅 explicit no-op path 已验证；这保留为未来 formal training/data coverage gate。Training Loop/optimizer、CPU tiny-data overfit、GPU/formal Dataset、baseline、Planner 和 performance claim 仍未实现；下一步必须单独授权 STEP 5.2。
