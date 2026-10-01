## 2026-10-01 STEP 6.3D FORMAL VALIDATION STAGE A（当前运行）

研究者已单独授权并启动 Formal Validation Stage A（仅 B1024）：64锚点×5 seeds×3方法=960 cases，名义预算983,040。2026-10-01北京时间19:26在RTX3080Ti/CUDA/FP32/batch16启动，启动前源码/768 TRAIN/771文件SHA/64锚点/checkpoint/执行身份全部PASS，Validation起点0。首批2个case已通过身份检查并持久备份；这是运行快照，不是最终效果结论。STEP_6_3D_VALIDATION_STAGE_A=RUNNING，VALIDATION_COMPARISON=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；future Return-birth限制保留。唯一下一动作是监控并备份Stage A，960完成后独立验收并停止。选择范围仅Planner v1 structured-search backbone/pure-search baseline，不是最终hybrid planner冻结。

记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；启动前机器收据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/01_launch_preflight_receipt.json`；动态监控与本地备份在该运行目录。下方均为较早阶段快照，关于未授权/Validation0的表述不能代表当前运行。

## 2026-10-01 STEP 6.3D PRE-VALIDATION BLOCKER CLOSURE（当前）

PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。

入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md`；机器验收：`code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。下方为较早快照，原B1/B2阻塞已关闭。

## 2026-10-01 STEP 6.3D PRE-VALIDATION AUDIT（当前）

STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。

记录：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md`；机器验收：`code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。下方均为较早记录。

# 已知问题与冲突

## 2026-10-01 STEP 6.3D TRAIN 后仍存在的边界

正式 TRAIN 八组配置均仅 48/96 cases 有可评分 H4。16/32 锚点、48/96 锚点/种子组合在所有配置中都未找到可评分路径；57,797 次不可评分 H4 完成尝试均记录 future Return birth 支持边界事件。另有 37,940 次 Grammar dead end；scorer 异常为零。重复配置/种子的边界计数不能当独立 Task birth。静态零 cohort 和空域残余为零。保持 `SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`，不放宽 H4、删除困难锚点或推断两种方法相等。Validation 比较和方法选定尚未开始；正式 TRAIN wall-clock 21.38 小时，后续 Validation 的 93.54 小时仅为按 TRAIN 速度做的粗略资源估算。

## 2026-09-30 STEP 6.3D-PREFLIGHT-PATCH 后待决问题

静态零 cohort 输入已从正式清单中清除，但 TRAIN-only 固定 HRS/B64 的 H4 支持诊断恰好 16/32 锚点可评分、16/32 无可评分候选；27/32 锚点至少一条完整轨迹触发 future Return birth 边界。研究者未定义“大量不可评分”的比例阈值，所以 `H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`，正式 6.3D 可以继续；H4 scoreable success rate 单独报告。CPU batch 8 探针只带来约 1.11 倍于 batch 1 的吞吐，长期 CPU 资源仍是限制。没有据此放宽 H_sup、删除 Return 要求、改变评分器、增加搜索预算或筛除难样本。

## 2026-09-29 STEP 6.3D 执行资源与评分支持

完整冻结比较若预算均耗尽，需要 2,113,536 次独特候选一步 World Model 转移；本机只有 CPU，探针按单进程外推为数周量级。当前只证实一个 TRAIN 锚点的多条完整 H4 轨迹仍受既有 6.2B `H_sup` 支持边界限制，不能按完整 H4 评分。不得将此推广为整体成功率，也不得对其使用 H1–H3 backoff。TRAIN tuning、Validation paired comparison、cluster bootstrap 和正式方法选择尚未完成；不能标记 `STEP_6_3D=PASS`。

## 2026-09-29 STEP 6.3B/6.3C-PATCH 剩余边界

原全 eligible Task 覆盖规则与部分选择的冲突已由研究者正式决定解决。TRAIN 4416 H1 投影中，728 条依赖同决策 offload Route 的历史 Comm 行排除，任务数量条件拒绝为 0；1 个投影后未见通信结构、349 个 Comp alpha 不符、15 个正请求但 CPU base 为零、6 个未见联合结构仍拒绝，不因追求 100% 重放而放宽。TRAIN/Validation 静态空域为 14/4416、6/1104；结构准入 4045/4416 中，逐字段语义重放 3920 通过，125 个 Comp amount 差异超过既定绝对容差 `1e-7`（最大 `2.9793e-7`），未调整容差。没有 GPU、训练或 `locked_test`。

## 2026-09-29 STEP 6.3C

Static TRAIN feasibility shows 1935/4416 anchors with no formal candidate under the frozen joint-support and causal Comp rules. This is recorded evidence, not a license to widen support. The audit does not provide dynamic H2-H4 feasibility rates; `H_sup` remains a post-rollout scorer result. Candidate cardinalities can exceed 300 million, so full materialization is prohibited. Future optimizer work must preserve the shared domain, interleaved rebinding and `B_WM` contract.

## 2026-09-29 STEP 6.3B support boundaries

旧 6.3A 文档的 policy “no-op” 标签不能解释为空 action：TRAIN 中 Comm/Comp/Mob 分别有 674/913/2761 个该标签但 action 非空的 raw slots。6.0A 旧版同 Task 多 Comm row 拒绝与 TRAIN 293/4608 raw slots 冲突，现已修正。TRAIN-only future-Return `H_sup` 分布尚未统计；候选语法不依赖该直方图，实际 `H_sup` 仍需 rollout/scorer。4.4 预测状态不会在 Flow 完成时自动把 Task lifecycle 从 offloading 提升为 computing；未来 Comp 准入只用显式 predicted computing/唯一 Exec 关系，未证实的转变不能猜测。时间上未见的结构序列仅作标签且允许，泛化效果未验证。

## 2026-09-28 STEP 6.3A

Formal TRAIN Comp base is reproducible from current computing-task CPU work,
static node capacity and slot duration using the existing deterministic inner
CPU rule; observed global alpha is `{0.5,0.75,1.0}`. Independent Comm/Comp/Mob
factorization is `NOT_SUPPORTED` because observed joint and temporal structure
is sparse. Candidate syntax, support tiers and any optimizer remain unselected;
the recommendation is not a researcher decision.

## 2026-09-28 STEP 6.2B-PATCH 当前边界

先前的 pending Route `flow_index=-1` 与 existing same-path Route 改 Task-Agent Host/learned latent 是真实实现观察；研究者现以 Planner v1 Route 显式 no-op 策略把二者排除出正式候选域，故不再阻塞 `STEP_6_2B=PASS`。4.4 Route→Host 语义及真正 Route/offload/multi-hop Planner 留待未来单独研究；4.4 本 Patch 未改。Formal multi-hop learned coverage 仍为零，单 anchor CPU scorer 验收不等于候选排序质量或闭环性能。

## 2026-09-28 STEP 6.2B Objective acceptance blocker（历史）

`BLOCKED_ON_OBJECTIVE_SEMANTICS`：单跳 domain gate 实际允许 pending/no-current-Flow Route；正式 adapter 将它映射为 `pending_flow / flow_index=-1`，4.4 不创建 Flow，而 frozen Objective 只规定未来 Return birth 的 `H_sup`。scorer 现拒绝静默给此类 trace 正式评分，待研究者决定支持范围。另，同路径 existing-Flow Route 的 4.4 规则会在 hop 完成前修改 Task-Agent Host，尽管 holder/Flow 路线未变；需核对 Route/Host 因果语义。Comm effort 原分母把 242 条关系行当 RB support，已修为选定 anchor 的 50 个全局有效 RB ID；旧机器观测仍保留。scorer 的 fixed-support CPU 机制通过不等于 6.2B 完整验收。

## 2026-09-28 Route recovery closure

- Route deterministic mismatch：`RESOLVED_IN_CODE`，包括 intermediate-hop advancement 与 same-destination full-path reroute；证据见 ROUTE-RECOVERY receipts。
- `FORMAL_MULTIHOP_COVERAGE_MISSING`：train 4416 / validation 1104 windows 的 route width > 1 与 active multi-hop 均为 0。它是 coverage/generalization limitation，不是当前 checkpoint corruption evidence。
- Planner v1 明确排除 multi-hop，当前 readiness 只到 single-hop scorer implementation；patched full validation 未执行。

## STEP 6.1 之后的证据限制

四族机制验收只覆盖同一 Formal Validation anchor 和一个训练 seed；H1–H4 有响应不等于反事实预测准确、长时可靠、可用规划目标、候选生成覆盖或闭环收益。6.0C 的静态 CPU 预算不证明动态可用 CPU；UAV 空间/禁飞约束与 fallback 安全性未验证。配对随机诊断共用单个 generator，latent sampling 与无线 outage draw stream 未严格拆开。期望服务模式的 `outage_uniform_draw=NaN` 是未抽样哨兵值，有限性只针对实际 service 输出。

## STEP 5.6C 验收后的证据边界（2026-09-28）

- 正式训练已完成且 best checkpoint 冻结；此前“仍在运行/未核实”的段落是历史快照。当前只有一个 seed、12 条 validation trajectory 的观测；没有 baseline、locked-test、跨 seed 稳定性或闭环系统评价。`L_Val` 的下降不直接证明最终任务完成率、时延或 Planner 收益。Motion raw aggregate 混合不同物理量单位，不能标成单一米数。
- `best.pt` 和 `latest.pt` 均为 step 5520，模型张量逐项相同，但两个序列化文件 SHA 不同；冻结身份以 tracked final checkpoint manifest 中的 `best.pt` SHA 和本地 checkpoint 字节共同建立。CPU replay 只把冻结 CUDA 配置的运行设备映射到 CPU，未修改研究者配置或进行训练。
- STEP 6.0A–C 的静态 Planner 候选/动作域合同存在；本 Step 的 `planner=false` 指本训练 run/验收无模型候选 rollout、闭环执行或效果声明，不否定这些先前 CPU 合同。

## STEP 6.0C 之后仍未解决

6.0C 将 Comp/Mobility 的**Planner v1 操作域**冻结，不会把 6.0B 的 simulator `dynamic_available_cpu=unavailable` 或 UAV hard bounds absent 改写成已解决。缺 UAV 当前 heading/elevation 时域结果仍 UNKNOWN、不可编译；无当前静态 CPU 容量的正 Comp 请求为 VIOLATED。空间工作区、禁飞区、未来可行性与 fallback 安全性未验证。多 UAV 不同 profile 只是每架边际数据支持，非正式数据中精确联合采集模式。本地 AirFogSim 精确 Git 身份仍未取得。

## STEP 6.0B（2026-09-26）

- AirFogSim 本地源码目录无独立 `.git`；`git -C` 返回 PI-JWM 主仓库。相关文件 SHA-256 可核验，AirFogSim Git commit/branch/clean 状态不可声称。
- `dynamic_available_cpu`：静态容量可观察，但 simulator callback 没有原生总量约束/决策时刻可用量；维持 UNKNOWN。
- `mobility_numeric_bounds`：直接 setter/step 无硬数值检查；场景配置与行为数据均不能当 Planner hard bound，维持 UNKNOWN。
- Simulator raw UAV acceleration 是 `(old-new)/dt`，PI-JWM canonical/World Model 是 `(new-old)/dt`；Raw 已显式分离，不能混用。示例配置速度注释“每时隙距离”与执行公式乘以秒不一致，以执行源码为单位依据。

## STEP 6.0A（2026-09-26）

当前可可靠验证的是静态对象支持和动作张量结构；dynamic available CPU 没有已证实因果源，数值 UAV 控制边界未冻结，UNKNOWN 约束不得静默当成满足。未来专属 Return Flow birth 被 fixed-support 阻止。L>1 编译需要外部逐步因果状态，本 Step 不生成这些状态。合成四动作 fixture 不是现实 Planner 候选覆盖；无 fallback safety 或性能证据。

## STEP 5.6B 训练中证据边界（2026-09-25）

- 正式训练已启动并仍在运行；当前只有 step 1104、2208 两次完整 validation。中途图是过程诊断，不是训练完成、最终性能或泛化结论。训练 loss 的 Stage/Horizon 在 552/1104/2208 步变化，跨边界不可直接比较。Motion raw aggregate 混合不同单位；CSI raw error 使用 dB。图和原始日志 hash 见 `docs/figures/step5_6b_live_progress_20260925/`。locked_test、baseline、Planner 均未执行。

## STEP 5.6A-CONFIG-FREEZE 当时边界（2026-09-24，历史）

- GPU H4 few-step smoke 与完整 1104-window prior-only validation 均通过；这验证运行路径，不验证正式训练或预测性能。CUDA checkpoint 重载参数/状态完全一致；两次预测存在微小浮点差异（Motion 最大约 8.20e-8、CSI 最大约 7.63e-6），按记录的 `rtol=atol=1e-6` 均通过，不能声称逐位相同。
- （冻结前历史记录）正式训练数值配置曾待研究者决定；该状态已由下方 STEP 5.6A-CONFIG-FREEZE 记录取代。当前正式配置以 tracked freeze artifact 为准。
- 旧 STEP 5.5-SHARE README 所列代码版本只实证了 package/index 的 portable 解压加载；本次真实 H4 远端消费暴露并修复了模型状态构造对本地 Raw 的依赖。已有分享 ZIP 的 Dataset 字节和身份不变；完整 Trainer 使用应采用 STEP 5.6A 修复后的源码。
- Few-step smoke/未训练模型 validation 不是正式训练结果或性能证据。`formal_training=false`、`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。
- Formal Training Config v1 已由研究者批准并冻结：`FORMAL_TRAINING_CONFIG=FROZEN`、`FORMAL_TRAINING_READINESS=READY_TO_START`。配置凭证位于 `code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/`；这不表示已经启动 formal training。
- 原 GPU validation receipt 的 `available_sample_count=0` 是批内布尔字段在 merge 二次聚合时丢失的 bookkeeping 问题，不是有效目标不存在。CPU 从真实 validation target shards 重算为 `[1104,1104,1104,1104]`，numerator/count 与 `L_Val` 不变；原 receipt 保留，纠正凭证标记 `bookkeeping_only=true`、`no_gpu_rerun=true`。

## STEP 5.5-PATCH 修正的证据边界

- STEP 5.5 原 CPU smoke 使用 `runtime/` 的 1+1 subset；新的 full-shard CPU 路径单独验收。不能把 mini smoke 写成全部正式窗口进入 Trainer。
- 原 package acceptance receipt 将未生成的 unsupported 结构字段统计为 0/0/0；全量真实 Return-birth detector 得到 unsupported 8828、unresolved 0、fixed-support blocked 8828（窗口-未来步事件，重叠窗口会重复计）。旧三个零值不是无 future-birth 的证据。
- 213 次 AirFogSim lifecycle repair 涉及同一 Task 对象的重复集合引用；修复直接保留最远 lifecycle，不修改该对象的 transmitted/computed/returned/done 字段。此结论限于 collection-side sanitation，不是有/无修复两次仿真结果的因果比较。

Source of truth：本文件是入口；具体事实必须回到列出的代码、配置、测试或 artifact。

## STEP 5.5 当前边界

- Formal Dataset v1 已 READY：60 条真实 trajectory、H2/L4、48/12 split、五类 package 与四动作 coverage 均通过机器验收；development Route/Comp 0/0 已不再是 formal Dataset blocker。
- STEP 5.5 当时的正式训练 blocker 是 training seed/batch/epoch/max_steps/patience/budget 未冻结，且 GPU smoke 尚未执行。该历史状态已由上方 STEP 5.6A GPU smoke 证据更新；数值配置 blocker 仍在。
- `formal_dataset=true`、`training_stack=PASS`，但 `full_training=false`、`gpu=false`、`locked_test=false`、`performance_claim=false`。

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
## 2026-09-22 STEP 5.2-PATCH closure

- STEP 5.2 的 current latent 已改为 current-observation posterior；Future Target posterior teacher 与 Future Target Encoder 在 validation 的 runtime 调用为 0。
- Validation `L_Val` 现在按 horizon 跨完整 validation set 聚合 Motion/CSI numerator/count；不再平均 sample-level normalized loss。
- Checkpoint resume 现在拒绝错误 data identity、normalization provenance 或 architecture-critical config；compatible reload 已通过。
- 仍未解决且不属于本 Patch：tiny-data overfit、full training、GPU、formal Dataset、locked_test、baseline、Planner、performance claim；Route/Comp non-empty development coverage 仍为 0/0。
## 2026-09-22 STEP 5.3-PATCH result

- 修正后的 bounded CPU preflight 为 `LEARNING_SIGNAL_GO`，但 `TINY_OVERFIT_NO_GO`：1-sample Stage 1、1-sample prior H1/H2 和 2-sample 正常 1→2 均有双 family learning signal、无 NaN/Inf；强 tiny-overfit 的 50% relative-drop + final normalized MSE<=1.0 未通过。
- raw metric 已改为只反归一化 target 一次，并写出 Motion x/y/z/speed 与 CSI dB 指标；CSI scale audit 显示 decoder 初始约 0 dB 而目标约 98–99 dB，当前记录为初始化/优化难度观察，不改 0.5/0.5 权重。
- 独立 fresh-run、checkpoint resume、phase-specific gradient audit 和 leakage audit 均已写入 receipt/artifact。
- CSI loss 仍明显高于 Motion，但 train-only normalized CSI std 约 1.04、Motion std 约 0.16，当前证据不支持 normalization bug；没有修改冻结的 0.5/0.5 权重。
- 部分 future posterior/target encoder steps 梯度为 0，已记录 zero-gradient step count；各组均有 finite learning signal 和参数更新，不构成持续性 gradient starvation。
- 该结果不开放 STEP 5.4 或 GPU；需要先做 bounded CPU tiny-overfit diagnosis。Route/Comp non-empty coverage 仍为 0/0，full training/formal Dataset/locked_test/performance claim 仍关闭。

## 2026-09-22 STEP 5.3D result

- Scale bridge machine check passed exactly for fixed `[0,1]`: actual normalized CSI MSE equals `(raw prediction - raw target)^2 / csi_std^2` aggregation for H1/H2.
- Baseline 200-step CPU run remains `TINY_OVERFIT_NO_GO` (H1 `329.3315→117.6610`, H2 `332.8831→119.7404`). Mean-bias diagnostic reaches H1 `1.0935→0.1405`, H2 `1.1040→0.1633`, but this is diagnostic evidence only.
- Current interpretation supports raw-output initialization/conditioning bottleneck; no formal initialization or normalized-output bridge has been selected. Researcher decision is required before any such change.

## 2026-09-22 STEP 5.3E closure

- raw CSI decoder + train-only CSI mean bias 已由研究者明确选定并正式接入 CPU Trainer；normalized-output bridge 仍是未来可选 ablation。
- tiny-overfit gate 在固定 `[0,1]` development subset 上通过，但不外推为 formal training、泛化或性能结果。
- Route/Comp non-empty development coverage 仍为 `0/0`，继续作为 future formal data coverage gate。
- STEP 5.4：Formal Dataset 不存在；正式 `L`、Physical topology、Dataset scale/seeds 和 training budget 未冻结；CPU dry-run 只支持 `GPU_CODEPATH_PREPARED`，没有 CUDA 证据。
- STEP 5.4-PATCH：generic package 与真实 Trainer CPU dry-run 已闭合；L=4 只有 config fixture，`L_gt_2_runtime_verified=false`；formal Dataset、CUDA 和 formal training 仍阻塞。
- STEP 5.4-PATCH2：四动作 adapter 与 CPU device portability 已闭合；Route/Comp 仍无真实 development coverage，正式 L/topology/budget 未冻结，CUDA 未验证。
# 2026-09-24 STEP 5.6B 预启动边界

正式 runner 仍须通过独立 Go/No-Go、精确 Git source 同步及 detached launch 的持续进度检查。此源码快照中的 `formal_training=false` 为启动前状态；启动后的实时事实应读远端 heartbeat。H1/H2 step 时间尚无实测，38.65 小时是按 horizon 比例的估计，不是训练结果。`locked_test`、baseline、Planner 和性能声明仍关闭。
# 2026-09-28 STEP 6.2A blockers

- Deadline, aligned arrival/elapsed, priority and Return-support state need additive Planner-only causal exposure; current frozen Formal Raw/Tensor/World Model path does not provide the complete side-state.
- `B_Tx` remains partially supported because cross-hop E2E remaining and stable route/epoch evidence are incomplete.
- Route effort has no candidate-independent normalized denominator.
- Therefore `STEP_6_2B_READINESS=BLOCKED`. Do not derive missing fields from Future Target or start 6.2B automatically.
# 2026-09-28 当前阻塞：Route 状态语义

4.2C-C `route_node_indices` 是剩余目的节点列表；4.4 规则以“含 holder 的完整路径”索引并在 Route action 后保留旧数组。真实规则负例复现 hop 完成但下一跳不推进。Planner side-state 无法合法改写 frozen model state；因此多跳/重路由 `B_Tx` 和 STEP 6.2B BLOCKED。另需注意正 `required_returned_size` 不代表本地计算必有 Return：仿真器还比较计算节点与返回目的地。
2026-09-30：future Return birth 仍是固定支持限制；H4 readiness 为 `SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`。3080 Ti 短稳态 10.9154 tps 推算完整矩阵约 53.79 小时，但 B_WM1024 主机存储容量诊断的长预算速度较慢，故这不是实测完整矩阵工时。batch32/64 在最小预算 CEM K4 的完整 H4 为零，冻结 batch16。
