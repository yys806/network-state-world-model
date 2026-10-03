# Final acceptance

STEP_6_4C_BUDGET_CALIBRATION=PASS。静态预注册16 anchors×3 seeds，MH-CEM K4/rho0.1，B256/B512新增96/96，48份配对B1024身份SHA通过且未重跑；新GPU名义/实际独立一步转移36864。三预算各30/48 H4可评分（62.5%）。256vs1024=1胜/19平/28负；512vs1024=7/22/19；256vs512=1/21/26；每组六类合计48。逐case去重可评分候选1263/2850/6060；内部搜索median20.133/42.342/88.436秒，mean31.140/62.955/134.232秒，低预算平均耗时减少76.802%/53.100%。总墙钟7800.804秒（2h10m），其中内部搜索4516.527秒，加载准备等3284.277秒另列。独立raw验收、144文件inventory及ZIP/SHA PASS，原raw目录保留在D:。6 anchors×3 seeds三预算均无可评分H4；future Return-birth fixed-support limitation保留；scorer exception=0。前三项最佳objective逐对一致，排序差异在J_Burden/J_Effort（额外分项只作exploratory diagnostic）。建议另预注册64×5 MH-only扩样，属于NEW RESEARCH PROPOSAL，不自动运行。SEARCH_METHOD仍MH纯搜索骨架；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，FINAL_FALLBACK_POLICY=NOT_FROZEN，HYBRID_PLANNER=NOT_FROZEN，Stage B=NOT_STARTED，locked_test=false。GPU搜索和监控已结束。唯一下一动作研究者审阅并决定预算或是否另授权扩样。

## Actual Validation Commands and Results
- focused test_step6_4c*.py：4/4 PASS；test_step6_3d_*.py：34/34 PASS（35.392s）；compileall PASS。
- audit_step6_4c_completed_budget_calibration_v1.py：96新/48参考身份、参数摘要、三组48六类、runtime统计与169个ZIP entries逐份SHA独立PASS。
- 原Stage B B256/B512正式结果0，locked_test=false。
- 启动c3c1f34e4b41b690333c77e8333265a01c69b474；actual execution0973bbf43d3531175b1c5a2bba62bd360833acddebda277511d6af13df37204a；start08:13:54.185598 UTC，last case10:23:54.989130 UTC。
- knowledge-index/diff/Context final outputs见28/29 receipts，最终Git owner以本记录commit历史为准。

## Expected vs Actual / Remaining Issues
实现/身份/统计/归档符合预期，校准PASS。明显冻结排序损失但可评分率未下降；来源主要是晚序目标，不能声称任务延迟/完成率变差。6个困难anchors全部保留，不扩大H4支持、不重训、不重选方法。Closed-loop budget、final fallback/hybrid尚未冻结；小subset不能证明总体非劣性。下一步仅研究者审阅并决定预算或新扩样协议。

## Reproduction / Git
正式命令须显式指定`--execution-config .../07b_calibration_execution_config_git_lf.json`。07原身份因前置byte/clean gate被拒且从未GPU执行；不存在旧身份续算。只提交小型receipt/inventory/summary/source/tests，ZIP和raw保留本地及原远端目录，不提交Git。完成后commit+push origin/main并停止。

---

## Current runtime gate

STEP_6_4C_BUDGET_CALIBRATION=RUNNING（未最终验收）。仅静态分层16 anchors×seeds6311/6312/6313，MH-CEM K4/rho0.1，B256→B512共96 cases；48份B1024参考SHA身份PASS，未重跑。启动提交c3c1f34e4b41b690333c77e8333265a01c69b474，执行身份0973bbf43d3531175b1c5a2bba62bd360833acddebda277511d6af13df37204a；RTX3080Ti CUDA FP32 batch16。旧07从未运行，07b绑定Git LF规范字节，tracked clean/source identity gate同时PASS。逐case原子写入并在本地D:按SHA备份；当前进度以09_runtime_status.json及原始结果为准，部分结果不作预算结论。CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；SEARCH_METHOD仍MH纯搜索骨架；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。唯一下一动作继续监控与备份至96完成并独立验收；不启动其他实验。

---

# STEP 6.4C — MH-CEM BUDGET CALIBRATION

## Step Goal / Definition Basis
研究者2026-10-03授权的小规模预算校准；固定MH-CEM K4/rho0.1、H4、原CandidateDomain/Objective/Return support、3080Ti FP32 batch16。不得将subset观察外推为完整Validation、等价性或最终预算。

## Initial State
origin/main=db0eb13f94313763079c685b02dfa7a16615f9f9，tracked clean。6.4B机制PASS；原Stage B未启动，locked_test=false。

## Files Involved / Changes / Reuse
新增step6_4c_budget_calibration_v1.py、prepare/run/inspect scripts和focused tests；不修改原搜索、评分、模型或环境源码。直接调用已有run_step6_3d_one_cpu_solve_v1.run，独立calibration目录和execution identity；历史StageA/TRAIN不覆盖。

## Protocol Freeze
运行前静态16_validation_anchor_manifest_objective_eligible.json，SHA与StageA旧config一致；其静态祖先SHA一致。7个存在层按名称排序，各取floor(16/7)=2，前2层多取1；同层SHA256(UTF8 sample_id)升序，ID作平局排序。未用任何搜索outcome选样；在读取选中48份objective之前先写不可覆盖的cohort/protocol。此前仅查看过raw字段名称用于serializer核对，没有据此选样。

seeds=6311,6312,6313；顺序B256全48再B512全48；名义新增budget36864。48份B1024参考独立验证sample/seed/method/K/rho/historical source/config/device/precision/batch/checkpoint和StageA归档SHA。runtime分位采用线性插值；仅in-solve，setup/matrix overhead另记。三比较六类严格合计48；不评分值保持None。无需bootstrap显著性推断，预注册paired descriptive统计。

## Validation
RED: 新helper缺失导致test import失败；GREEN: 4个calibration selection/poison/order/paired/identity/isolation/NaN/scorer-stop测试PASS。旧focused Step6.3D 34 tests PASS。compileall、knowledge-index write/check、diff check及Context Consistency结果见08/15 receipts。

## Results / Expected vs Actual
协议与48参考PASS；CPU preflight PASS。SSH只读确认3080Ti空闲且checkpointSHA正确，但远端源码仍587efdf；此时未启动搜索。下一gate必须同步本轮协议提交，再通过HEAD=origin/main、tracked clean、当前byte-SHA、raw0等launch检查。正式结果待运行，不能声称96/96或性能结论。

## Known Issues / Boundaries
future Return-birth fixed-support limitation保持；CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；FINAL_FALLBACK_POLICY/HYBRID_PLANNER=NOT_FROZEN；Stage B=NOT_STARTED；locked_test=false。

## Git
本轮先提交并push协议/CPU gate，然后才能远端同步启动；最终运行验收另commit+push，owner commit通过Git历史和launch runtime确定，不将提交自身SHA写入其本身。

## Next Step
仅同步和验证当前授权96-case calibration runner，然后执行与独立验收；不得启动原Stage B/环境动作/其他实验。

### Deployment engineering check
首次SSH部署已fast-forward，但non-login shell无python PATH，尚未调用planner；改管理脚本使用StageA记录的同一/root/miniconda3/bin/python绝对路径。该脚本不进入frozen planner runtime/source closure；执行身份不变，不改搜索协议。

远端GitHub fetch因GnuTLS连接中断失败，GPU仍未启动。仅允许使用本地已push main生成的Git bundle做同一commit的传输备份，bundle verify和HEAD相等门保持；不改源文件内容或历史结果。

### Canonical Git byte correction before first GPU case
原07身份5681d3在remote tracked-clean gate失败（4个仅换行字节差异），GPU0 cases。保留07不覆盖，恢复本地到Git LF规范字节，solver数学/科学源码内容不变；添加07b新版身份及21 source byte bridge。runner新增显式execution-config参数，source SHA再次冻结；原cohort/protocol/48参考完全不变。远端只恢复本次newline transport影响的4个文件，然后以同一新提交同步重验，不隐瞒脏树或降低门槛。
