# STEP 6.3D — CEM 更新语义与 Validation 分阶段执行合同

研究者于 2026-10-01 明确冻结本合同。它澄清已有更新门并改变 runner 的调度/汇总，不改变搜索方法、目标、域、预算、checkpoint 或 TRAIN 选择规则。

## 当前轮可评分候选计数

“至少两个 newly scoreable H4 candidates”表示：当前 CEM iteration **实际重新采样、完成四步并得到可评分结果**的不同 candidate fingerprints 至少两个。

- 同一个指纹在当前轮完成多次，计一个。
- 即使以前的轮次出现过，当前轮再次完成仍计一个；合法 transition cache hit 不妨碍完成。
- 只保留自前一轮、当前轮未重新采样完成的 elite 不计入门。
- 不要求整个 solve 历史上第一次出现，不改成 globally unseen gate。

现有 `solve_fixed_budget` 每轮 fresh `current`，只有 `score_complete` 在 depth=4 且 scoreable 时添加；`unique_current` 按指纹去重；更新门 `method != HRS and len(unique_current)>=2`。之后和 retained 合并的是 elite 选择池，不是更新门的分母。原概率公式、eta=0.5、epsilon=0.05、elite ratio、retain≤5 和严格目标字典序保持不变。B1 oracle 分别覆盖旧候选重放、轮内重复、单一候选及 retained-only 情况；无需重跑 TRAIN。

## Stage A — PRIMARY

固定 B_WM=1024；HRS=(K1,无elite ratio)，S-CEM/MH-CEM=(K4,rho0.1)。固定64 anchors、seeds6311–6315；共960 cases，名义983040次一步转移。只运行这一预算，汇总并产生 `07_validation_stage_a_primary_comparison_receipt.json` 和 `08_selected_method.json` 后直接 return/退出。不得继续256或512。执行前必须有研究者单独开跑授权；完成后停止待审阅。

## Stage B — DIAGNOSTIC

另行授权的明确 `diagnostic` 调用才运行256→512，1920 cases。读取并核对 Stage A 收据与选择文件的 execution identity、预算、case总数、停止门和 SHA 相互绑定。只写 `09_validation_stage_b_budget_diagnostic_receipt.json`；不调用 `select_method`，不覆盖 Stage A 收据或选择。

CLI 只允许 `--validation-stage primary|diagnostic`，没有 all 或自动 fall-through。Validation 缺阶段或新执行配置即拒绝；TRAIN 不接受 Validation stage 参数。run/resume 的结果文件名仍只取决于 anchor/method/seed/budget/K/rho，和调用顺序无关。

## 统计与输出

绑定上轮预注册 schema 的原文件 SHA。每个 method×budget 单列320 cases、H4可评分case数/比例、完整H4路径数、不同可评分指纹数、不可评分完成次数、Return-birth边界事件、Grammar死路、scorer异常、实际unique transitions、cache hits、proposed/admitted/rejected和计时。完整路径数和去重候选数不是同一计数单位，不强行相加。

每对方法分为 only-left-scoreable、only-right-scoreable、both-scoreable-left-win、both-scoreable-right-win、both-scoreable-tie、both-unscoreable 六类；总和必须320，缺case/多case/错误anchor或seed集合拒绝。保留win/tie/loss、每anchor五个按6311–6315排列的outcomes和冻结bootstrap。Return只计 `H4_SUPPORT_BOUNDARY:*UNSUPPORTED_FUTURE_RETURN_BIRTH*` 残余；scorer异常取 `support_horizon_counts.SCORER_EXCEPTION`，并与对应异常残余核对一致，不能两份相加。

只有1024进入冻结主方法选择：anchor-cluster bootstrap，五seed不拆分，10000次，seed6316，lower95>0；既定HRS/S/MH简约选择规则不变。三对比较的共享anchor重采样 Bonferroni percentile 敏感性区间仅作解释，不进入选择，不保证有限样本联合覆盖；Holm未启用。Stage B不能改变主选择。

## 两个执行身份

历史TRAIN raw/log/05/06/旧3080Ti资格配置不可修改；旧执行ID及source map保留。当前只允许 `run_step6_3d_formal_cpu_matrix_v1.py` 在16项source map中改变；其余source/anchors/catalog/sidecars/checkpoint都须相同。新配置为 `FORMAL_STEP_6_3D_VALIDATION_EXECUTION_CONFIG_3080TI`，记录当前source map及独立execution ID，并用文件SHA绑定历史TRAIN配置、tuning/closure、资格配置及schema。

CUDA、RTX3080Ti、FP32、batch16、cpu_cache_and_prefix、prior mean、service expectation、no AMP/BF16/FP16/quantization/compile/bucketing 保持。resume严格比较新source/config/device/batch/checkpoint；旧TRAIN config不能用作新Validation执行配置。新配置是未来运行合同，不代表本次GPU运行或方法选定。

冻结顺序：`1024 → STOP → researcher review`；另行授权后 `256 → 512`。本 Step 只做阻塞收口；`VALIDATION_COMPARISON=NOT_STARTED`、`SEARCH_METHOD=NOT_SELECTED`、`locked_test=false`，future Return-birth 固定支持限制保留。
