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
