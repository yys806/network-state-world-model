# STEP 6.4D — Full-cohort MH-CEM B512 Expansion

## Step Goal / Definition Basis
研究者2026-10-03本聊天授权：完整既有64×5 MH-only B512，严格配对原Stage A B1024；只比较预算质量与计算开销，不选择方法或冻结闭环预算。MH-CEM K4/rho0.1、H4、Objective、CandidateDomain、Grammar、Return/cache/RNG/prior/service/checkpoint不变。

## Initial State
origin/main=cb19ca74064a76bb28b98b1fa126b444259676e3；tracked clean；6.4C PASS；三原有untracked保留；Stage B/正式闭环/locked test未运行。

## Files Involved / Changes / Reuse
独立helper `code/src/pi_jwm/step6_4d_full_cohort_b512_v1.py`；prepare/run/manage三个script；focused tests。仅新增预算执行、复用验收、统计与SSH SHA备份。既有solver/CandidateDomain/Grammar/Objective/WM与6.4C冻结全source closure逐字节一致。320父结果逐文件身份、StageA inventorySHA验证；48B512按6.4C已验收inventory SHA引用，不复制或改写原raw。新272，无B256或B1024模型forward。

## Validation — Actual Commands / Outputs
RED：新helper缺失导致import失败；GREEN：`python -m unittest discover -s code/tests -p 'test_step6_4d*.py'` 6/6 PASS(0.454s)，6.4C 4/4 PASS，6.3D 34/34 PASS(30.508s)。compileall PASS，部署脚本三处regex字符串产生非致命SyntaxWarning（正则运行正确，不属于科学问题）。prepare与CPU runner均PASS，64/320父配对和48复用完整，新增272、新raw=0。check source/checkpoint身份PASS。

## Protocol / Scientific Statistical Contract
01完整static manifest原序，不采样；seeds6311..6315。02冻结唯一预算B512、配对None六类合计320。64anchor clusters保持每anchor5seeds，frozen bootstrap seed6316/10000replications。每pair首差分量：N_DDL/A_DDL/J_Delay/J_Burden/J_Effort/all_equal，分别记录左/右更优；前三项先劣的loss定义PRIMARY_OBJECTIVE_DEGRADATION_COUNT，后两项loss定义BURDEN_EFFORT_ONLY_DEGRADATION_COUNT。每预算320cases完整H4/独特scoreable候选/unscoreable/Return/grammar/scorer/budget/cache/proposal/runtime。原raw in-solve runtime含mean/median/P90/P95/max；准备与matrix overhead另记。困难anchors全保留并比较交集/两差集。

## Results / Expected vs Actual
当前是协议/CPU门PASS；GPU未开始，48/320已验复用覆盖，正式完整结果待272新case。不写实验最终PASS或预算结论。

## Known Issues / Claim Boundary
future Return-birth fixed-support limitation保留；B512是否值得冻结须研究者判定。禁止等价/最优/真实闭环性能/系统收益声称。Stage B、B256、正式closed loop、baseline、ablation、locked test均未启动。

## Git / Provenance / Execution Gate
05独立执行身份绑定530源码/输入条目（确切数以config为准）、冻结checkpoint、64×5 manifest、320父证据与48复用。protocol-owning commit必须先push；为避免commit自身引用循环，精确commit通过--protocol-commit指定并在launch/runtime/result保存。服务器须fetch+ff至该精确commit、tracked clean；runner再次核验Git blobs逐源码与config SHA，CUDA3080Ti FP32 batch16。每case原子保存；错误resume/scorer/NaNInf/source drift拒绝并停止，不自动重启。最终commit由Git历史定位。

## Next Step
协议commit/push后同步精确commit并核验，执行仅剩余272 B512；完成后独立验收/归档/Context/Git收口并停止。
