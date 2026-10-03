## 2026-10-03 STEP 6.3D Stage A 最终验收（当前）

STEP_6_3D_VALIDATION_STAGE_A=PASS：960/960 原始结果完成并独立验身份、选法、配对统计及 SHA 归档；名义/实际一步转移均983,040。冻结规则选择 SEARCH_METHOD=MH-CEM，仅表示 Planner v1 structured-search backbone / pure-search baseline，不是最终 hybrid PI-JWM planner 冻结。三方法各160/320可评分H4（50%）；MH 对 S 为78胜/198平/44负，anchor-cluster 95% CI=[0.03125,0.184375]。正式运行40.6074小时、23.6410 cases/hour；GPU RTX3080Ti/CUDA/FP32/batch16 已硬停止。Stage B=NOT_STARTED，locked_test=false。future Return-birth 固定支持限制保留；无未解决执行阻塞。唯一下一动作是研究者审阅本次结果，禁止自动扩展实验。

实施记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；独立验收：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/13_stage_a_acceptance_receipt.json`；归档 SHA：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/14_stage_a_archive_sha_manifest.json`。本机 D: 保留960 raw和ZIP，远端原始结果保留；Git仅保存小型证据。

---

以下均为历史记录；RUNNING / NOT_SELECTED 等较早状态不代表当前验收状态。

## 2026-10-01 STEP 6.3D FORMAL VALIDATION STAGE A（当前运行）

研究者已单独授权并启动 Formal Validation Stage A（仅 B1024）：64锚点×5 seeds×3方法=960 cases，名义预算983,040。2026-10-01北京时间19:26在RTX3080Ti/CUDA/FP32/batch16启动，启动前源码/768 TRAIN/771文件SHA/64锚点/checkpoint/执行身份全部PASS，Validation起点0。首批2个case已通过身份检查并持久备份；这是运行快照，不是最终效果结论。STEP_6_3D_VALIDATION_STAGE_A=RUNNING，VALIDATION_COMPARISON=RUNNING，SEARCH_METHOD=NOT_SELECTED，Stage B=NOT_STARTED，locked_test=false；future Return-birth限制保留。唯一下一动作是监控并备份Stage A，960完成后独立验收并停止。选择范围仅Planner v1 structured-search backbone/pure-search baseline，不是最终hybrid planner冻结。

记录：`docs/implementation_records/STEP_06_3D_FORMAL_VALIDATION_STAGE_A.md`；启动前机器收据：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/01_launch_preflight_receipt.json`；动态监控与本地备份在该运行目录。下方均为较早阶段快照，关于未授权/Validation0的表述不能代表当前运行。

# PI-JWM 结果索引

> 2026-10-01 当前：PRE_VALIDATION_BLOCKER_CLOSURE=PASS，PRE_VALIDATION_AUDIT=PASS，VALIDATION_GO。研究者已明确CEM更新门统计当前轮实际完成的不同可评分四步候选；历史重采样可计入，retained-only不计入，原solver无改动。runner已分primary/diagnostic，1024完成后硬停止，256/512另行授权；六类配对统计及Return/scorer诊断完整。旧TRAIN 768 raw/log/05/06及原执行身份保持，TRAIN_REUSE_VALID=true、TRAIN_RERUN_REQUIRED=false，两种CEM仍(4,0.1)。新3080Ti Validation身份单独冻结。VALIDATION_RESULT_COUNT=0，VALIDATION_COMPARISON=NOT_STARTED，SEARCH_METHOD=NOT_SELECTED，locked_test=false；future Return-birth限制保留。唯一下一动作是研究者审阅并单独授权Stage A；本次不启动GPU或Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_BLOCKER_CLOSURE.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/08_final_pre_validation_acceptance.json`。

> 2026-10-01 当前：STEP 6.3D Validation 前审计完成：PRE_VALIDATION_AUDIT=BLOCKED、VALIDATION_NO_GO。数学、预算/缓存、batch16四步可完成性、随机/顺序独立性及统计 oracle 通过；CEM 旧候选重放触发更新的“newly”定义待研究者确认，runner 缺六类配对/Return/scorer 汇总与 Stage A 停止门。TRAIN MH-vs-S=26胜/58平/12负，仅诊断。TRAIN closure PASS 和两种 (4,0.1) 配置保留；Validation=0、SEARCH_METHOD=NOT_SELECTED、locked_test=false，future Return birth 限制不变。唯一下一动作是明确语义并授权必要修补后再审计；本次不启动 GPU/Validation。 入口：`docs/implementation_records/STEP_06_3D_PRE_VALIDATION_SCIENTIFIC_IMPLEMENTATION_AUDIT.md` / `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001/09_pre_validation_audit_receipt.json`。

2026-10-01 TRAIN-only 选参观察：S-CEM/MH-CEM 的四组配置均各 48/96 cases 找到可评分 H4；按冻结 TRAIN 排序分别选 `(K=4,rho=0.1)`。16/32 锚点持续无可评分结果，future Return birth 限制保留。完整计数、配对净胜、Grammar/Return 残余和每锚点/种子见 `code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/train_tuning_diagnostic_summary.json`；验收状态见 `train_tuning_closure_acceptance.json`。这些不是 Validation HRS 对照、最终方法选择或闭环性能结果。

2026-09-30 STEP 6.3D-PREFLIGHT-PATCH：正式 32 TRAIN / 64 Validation 锚点静态候选域均非空且 Objective cohort>0。TRAIN-only HRS seed6391/B_WM64：16/32 锚点有 H4 可评分候选，16/32 无；406 条完整 H4 中 131 可评分、275 不可评分，275 条均有 future Return birth 支持边界；另有 240 个 Grammar dead-end 分支。CPU batch 1/4/8/16 与串行等价，本次吞吐 0.8695/0.8911/0.9640/0.9218 unique transitions/s。原始 20–25 机器收据可核验；这是前置诊断，不是 HRS/S-CEM/MH-CEM 胜负或闭环性能结果。

2026-09-29 STEP 6.3B/6.3C-PATCH：TRAIN 条件选中任务数的候选准入使静态空域从原合同的 1935/4416 变为 14/4416；Validation 描述性空域从 550/1104 变为 6/1104。TRAIN H1 投影排除 728 条依赖同决策 offload Route 的历史 Comm 行；任务数量条件的投影拒绝为 0。其余通信结构、Comp 和联合结构残余保留，详见 `code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929/07_step6_3bc_patch_acceptance.json`。这是候选域/准入证据，不是优化效果或闭环性能。

2026-09-28 STEP 6.3A：`STEP_6_3A=PASS` 表示候选支持审计有完整 CPU 证据。Formal TRAIN Comp 1969/1969 action entries 可按现有 causal CPU rule 重建，alpha 为 `{0.5,0.75,1.0}`；独立 Comm/Comp/Mob factorization 为 `NOT_SUPPORTED`。无候选排名、优化器或性能结论。

2026-09-28 STEP 6.2B-PATCH：机器 receipt 的 `STEP_6_2B=PASS` 是冻结 best.pt、非锁定单 anchor、mean-prior/expected-service H1–H4 的 Objective scorer/严格比较器 CPU 机制验收；Route 有效自由度 `NONE`，Comm effort 分母 50 全局 RB ID。见 `code/artifacts/protocols/pi_jwm_step6_2b_patch_route_noop_v1_20260928/`、实施记录和 6.0C/6.2B 源码。不得解释成候选优劣、性能或闭环结果。

2026-09-28 STEP 6.1：`action_family_response.json`、`recursive_feedback_audit.json` 和 `stochastic_common_seed_diagnostic.json` 仅证明冻结训练模型的动作条件、递归和数值稳定机制；CPU runtime 只说明本机实现开销。没有 reward/cost/risk、winner、baseline、locked-test、闭环收益或性能优越性结果。原始收据见 `code/artifacts/protocols/pi_jwm_step6_1_trained_candidate_rollout_preflight_v1_20260928/`。

2026-09-28：STEP 5.6C 的 `formal_validation_observation.json` 记录正式 run 五次 validation 和 H1–H4 `L_Pred`、Motion/CSI raw MAE/RMSE；最终 `L_Val=0.07431338784170399` 是同一验证集的严格最低值。对应 checkpoint SHA 和 CPU 验收见 `code/artifacts/manifests/pi_jwm_step5_6c_final_acceptance_20260928/`。这是 **Formal Validation Observation**，不是 test performance、baseline 对比、SOTA、泛化或闭环系统结论。

> 结果索引连接“数字—实验—配置—代码—数据—研究问题”。数字本身不是最终真相，必须回到原始 metrics、checkpoint、manifest 和 audit。

当前可引用数字的结构化副本见 `docs/registries/results_registry.json`；它只负责定位，最终仍以对应原始 metrics 和独立 audit 为准。`build_project_knowledge_index_v1.py --check` 会把 seed、最佳 epoch、9 项门控指标、audit SHA-256、验收状态和 `locked_test` 边界与原始 `single_seed_acceptance.json` 自动比较，不一致时直接失败。

STEP 5.5 只有 Dataset/CPU interface 验收结果，没有预测性能结果：60 trajectories、5520 windows、四动作 coverage、package hashes 与 machine receipts 定位在 `code/artifacts/manifests/pi_jwm_step5_5_formal_dataset_v1_20260923/`，不得写入性能比较。

> 2026-09-18：以下数字仍可在旧协议边界内引用，但全部属于 Historical / Archived evidence。新 `00–06` 尚无性能结果，不能用这些数字证明新双图、四类动作、目标 RSSM 或 planner 已实现。

## 1. 旧协议可引用结果

### P4 entity RSSM，seed 20260831

唯一正式单 seed 验收入口：

`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`

该文件记录了 79 项 manifest、strict reload、门控重算和 `locked_test_accessed=false`。核心数值：

| 指标 | 值 | 证据范围 |
| --- | ---: | --- |
| validation link-F1 delta | `+0.4441475764` | seed 20260831，unlocked validation |
| calibration link-F1 delta | `+0.8622921256` | seed 20260831，unlocked calibration |
| node-x overall ratio | `0.7547533605` | seed 20260831 |
| node-x h5/h10/h20 ratio | `0.7699514616 / 0.7507086696 / 0.7549541058` | seed 20260831 |
| throughput ratio | `0.9355544639` | seed 20260831 |
| RB occupancy ratio | `0.4736271290` | seed 20260831 |
| task delay ratio | `0.0128686245` | seed 20260831 |
| selected checkpoint | RSSM epoch 39 | base 20 + RSSM 40 epoch |

正确表述是：

> 实体级双图 RSSM 在 seed 20260831 的正式 unlocked 单 seed 验收中通过 9 项数值门。

不能表述为：最终性能已确定、跨 seed 泛化已证明、P4 已关闭或 locked test 已通过。

## 2. 第二个已验收结果

### P4 entity RSSM，seed 20260830

正式 run 位于 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1/`；独立验收入口为：

`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`

最佳 RSSM checkpoint 为 epoch 40。validation/calibration link-F1 delta=`+0.45710/+0.88509`；node-x 总体/h5/h10/h20 ratio=`0.75086/0.76033/0.74939/0.74960`；throughput/RB/task-delay ratio=`0.93831/0.47523/0.01175`。9 项单 seed 数值门全部通过，79 项 manifest 零缺失、零哈希差异，strict reload 为 `0/0`。

正确边界是：两个固定 unlocked seed 已分别通过单 seed验收；第三 seed 和三 seed 审计仍缺失，不能据此关闭 P4、开放 P6 或访问 `locked_test`。

## 3. 机制和执行结果

| 结果 | 说明 | 证据 |
| --- | --- | --- |
| causal motion | 运动特征由历史位置因果差分得到 | `formal_motion_state_v1.py`、P4 第一性原理审计 |
| entity-local stochastic state | node/physical edge/flow/task 分别维护 latent | 实体级模型与 CPU consistency audit |
| prior-only deployment | 正式预测不读取未来目标 | 模型代码、consistency audit、冻结协议 |
| two-stage training | base 训练后冻结，再训练 RSSM | runner、protocol、checkpoint metadata |
| batch 8 execution | CUDA batch probe 在预算内通过 | `pi_jwm_p4_entity_rssm_gpu_batch_probe_20260906_v1/` |

这些是机制/执行证据，不等于完整性能结论。

## 4. 历史结果的引用规则

历史失败结果可以用于回答“为什么改方法”，例如 global RSSM 的实体广播限制、运动输入缺失和训练目标冲突；但必须同时标记：

- 使用的旧方法身份；
- 数据和协议范围；
- 失败门或诊断指标；
- 是否被后续方法替代；
- 不能把它和当前实体 RSSM 数字混成同一实验。

## 5. 双向追踪模板

新增结果时按下列链路登记：

```text
研究问题
  → 方法/假设
  → 代码入口及版本
  → 配置和 tensor manifest
  → seed/split/checkpoint
  → 原始 metrics
  → 独立 audit
  → 结论和边界
```

反向查询时，从数字先找到 audit 或原始 metrics，再回到 run summary、checkpoint、manifest、配置和代码；不要只依赖 PPT、README 或摘要表。
## 2026-09-28 STEP 6.2A-PATCH

No new performance result. The only accepted result is a readiness diagnostic: deadline sidecar alignment passes for one non-locked validation anchor, and 6.2B remains blocked by a reproduced route transition mismatch. Primary future throughput is E2E useful delivery; network service throughput is diagnostic. See the PATCH machine readiness receipt and implementation record.
2026-09-30 3080 Ti 迁移/执行资格：正式 Dataset 与冻结 checkpoint SHA 一致，CPU/GPU 离散等价 PASS；batch 8/16/32/64 短稳态中位 10.3593/10.9154/12.4562/12.6699 tps，基于最小预算 CEM 完整 H4 约束冻结 batch16。bounded TRAIN smoke PASS；不是正式搜索方法比较或科学性能结论。
