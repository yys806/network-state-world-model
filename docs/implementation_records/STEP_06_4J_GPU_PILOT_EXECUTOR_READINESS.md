# STEP 6.4J — GPU Pilot Executor Readiness

## Step Goal

补齐已冻结的两轨迹 S-CEM B512 Pilot 的显式执行入口，并在 GPU 启动前完成 CPU 验收。

## Definition Basis

继承 `STEP_06_4J_S_CEM_B512_GPU_PILOT.md`：S-CEM K=4/rho=0.2、H4、B_WM=512、seed=6311、两条固定 dev_validation 轨迹、每条最多8次决策、最多16次搜索、同步暂停仿真、winner 首动作和 A-else-C fallback。

## Initial State

原冻结提交 `46ec14190b857b4e4b2b2a2921a00ea7ec514f8f` 只有协议保护与 CPU 证据，缺少正式 Pilot 多 episode 入口，因此不能启动 GPU。

## Changes

- 新增 `code/scripts/run_step6_4j_pilot_v1.py`：显式 engineering/execute 模式、CUDA/GPU/FP32/B512/范围身份门、结果目录拒绝重复、PilotClock 停止门、真实 LiveSCEMPlanner、EpisodeController、AirFogSim collector、原子每步 JSON、真实任务 ledger 和失败记录。
- 新增 `code/tests/test_step6_4j_runner_v1.py`：模式、范围、CPU CUDA 拒绝测试。
- 生成新 protocol revision `..._20261009_r3`，保留 r1/r2 历史证据；新的 execution_config_id 由源码 SHA、输入和冻结配置共同确定。

## Reuse

复用 6.4I 已验收的 LiveSCEMPlanner、EpisodeController、AirFogSim collector、动作桥和 RealTaskLedger；没有修改 World Model、CandidateDomain、Objective、搜索算法或 checkpoint。

## Validation

- `python code/scripts/prepare_step6_4j_pilot_v1.py --freeze`: PASS
- `python code/scripts/audit_step6_4j_pilot_v1.py`: PASS
- `python -m unittest discover -s code/tests -p 'test_step6_4j*.py'`: 6 PASS
- `python code/scripts/run_step6_4j_pilot_v1.py --engineering-smoke`: PASS, reused accepted real AirFogSim CPU evidence
- `python -m compileall -q code/src code/scripts code/tests`: PASS
- `git diff --check`: PASS

## Results and Boundary

`STEP_6_4J_IMPLEMENTATION=PASS`; `READY_FOR_GPU_LAUNCH=true` after this commit is synchronized remotely and runtime preflight passes. `GPU_PILOT_RESULTS=NOT_STARTED`; `FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED`; `locked_test=false`. CPU evidence is mechanism evidence, not Pilot performance.

## GPU Start Gate

After the researcher starts the instance: fast-forward remote to this exact commit, verify RTX 3080 Ti/CUDA/FP32/batch16, checkpoint/source/input SHA, clean worktree, dependencies, disk >=20 GiB, no runner and backup path; run the first approved CUDA decision as the qualification gate, counted in the 16-search cap. Stop on any identity, scorer, NaN/Inf, budget, setter, step or feedback failure. No retry or episode resume.

## Known Issues

The local host cannot execute CUDA. The remote Pilot remains unstarted until the exact commit is deployed and the runtime gate passes.

## Git

This record is closed by the commit that contains the r3 protocol and runner. The next action is the separately authorized remote GPU Pilot only.
