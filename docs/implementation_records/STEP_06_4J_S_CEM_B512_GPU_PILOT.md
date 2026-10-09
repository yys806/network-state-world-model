# STEP 6.4J — S-CEM B512 最小真实 GPU Pilot

## Goal / Decision Basis
研究者授权在6.4I CPU准备后执行最小GPU闭环Pilot：2条固定dev_validation轨迹、每条最多8步、search seed6311、S-CEM K4/rho0.2 B512 H4。只验证完整CUDA闭环机制、日志和成本，不作性能/优于baseline结论，不进入正式24×64。

## Initial State
基线973c62fa66a4898407803908e3a20383200f539d；origin/main一致；已有6.4I CPU mechanism PASS、正式episode草案未批准。无GPU操作，无locked_test。

## Freeze / Manifest
`prepare_step6_4j_pilot_v1.py --freeze`冻结namespace。按trajectory_id UTF8 SHA256升序选择：`formal-v1-sim-2026092326-policy-2026092426::anchor-0029`、`formal-v1-sim-2026092318-policy-2026092418::anchor-0006`；各取静态manifest最早frame。原raw、Task cohort、checkpoint、normalization和source SHA写入00/01 receipt。Pilot config ID=`31667b891be356017220e205e19a33b1e3aeaf7df8fcc8f11af147061303ddf8`。

## Changes / Reuse
新增step6_4j_pilot guard、protocol freeze script、independent audit和CPU tests。复用6.4I EpisodeController/runner/LiveSCEMPlanner、AirFogSim collector/capture、6.4F bridge/domain和冻结solver/scorer/WM；不改科学算法、checkpoint、Objective、CandidateDomain或fallback。

## Validation
`python code/scripts/prepare_step6_4j_pilot_v1.py --freeze` PASS；`python code/scripts/audit_step6_4j_pilot_v1.py` PASS；3 test cases PASS；compileall PASS；git diff --check PASS。GPU仍NOT_STARTED。

## Runtime gate / Stop
远端必须HEAD==本协议冻结commit、tracked clean、RTX3080Ti/CUDA/FP32/batch16、依赖、显存和20GiB空间、无旧runner、D备份通道PASS。先首个CUDA决策资格门，失败停止；单步超过600秒、累计接近3小时、异常身份/预算/scorer/NaN/重复runner/partial setter/env.step/feedback不一致均硬停。无episode内动作重试或中途恢复。

## Results / Boundary
CPU protocol PASS；正式CUDA和真实GPU动作结果尚不存在。所有Pilot失败保留分母和分组，不能early-best冒充B512。`FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED`，`locked_test=false`。

## Next
本协议commit/push后，仅在研究者已开启且硬检查通过的RTX3080Ti上执行；Pilot结束后硬停止等待审阅，不自动进入24×64、baseline、ablation或hybrid。
