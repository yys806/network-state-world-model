# STEP 6.3D — FORMAL VALIDATION STAGE A

## Step Goal / Definition Basis

研究者于 2026-10-01 明确授权仅运行 PRIMARY Stage A：同一 CandidateDomain、冻结 World Model、H4 Objective 与 B_WM=1024 下比较 HRS、S-CEM K4/rho0.1、MH-CEM K4/rho0.1。64 Validation anchors × seeds6311–6315 ×3 methods =960 cases；名义预算983,040。选出的 SEARCH_METHOD 仅表示 Planner v1 structured-search backbone / pure-search baseline，不表示最终 hybrid PI-JWM planner 完全冻结。Stage B256/512、learned proposal、Route 支持扩展、warm start/fallback、closed-loop、baseline/ablation/locked_test 均不在本授权内。

依据：研究者本次指令，`docs/contracts/PIJWM_STEP_06_3D_VALIDATION_STAGES_AND_CEM_UPDATE_SEMANTICS_V1.md` 与已接受的 PRE_VALIDATION_BLOCKER_CLOSURE receipts。

## Initial State

本机 fetch 后 HEAD=origin/main=`587efdfbdb460f7fa810c5c23ac651afc5a4b339`，tracked clean；既有三个 untracked 项保留。远端起初为已接受 TRAIN source `5b48ad6...`，仅历史 relay manifest 改动。该改动已通过专用 Git stash 保留，再 fast-forward 至冻结 Validation 提交；没有改写历史 TRAIN raw/log/05/06。原 qualification 与新 Validation 执行身份保持分离。

## Files Involved / Changes / Reuse

- 冻结 formal runner / solver / proposal / comparator / method-selection / anchors / checkpoint 均不修改。
- 使用 frozen Validation config `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json`，ID `3b3fcfe775f947790a982929a292fb3f3b8f30396996f7f9b33270e7d6c9f215`。
- 运行控制、启动前收据、监控和本地快照目录：`code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001/`。这属于运行管理，未进入冻结科学源码 hash 集。
- 原始结果原子写入既有 `pi_jwm_step6_3d_fixed_budget_search_v1_20260929/solve_results/validation/`。只读 SFTP 将完成结果逐文件验身份和 SHA 后备份到本机 D:；原远端数据保留。
- 外部监控核对源码 hash、case identity、非有限数、scorer exceptions；异常停止 runner，不修改方法或自动重启。监控初稿引用的预算收据字段已在首 case 完成前改为实际 `budget_receipt`，通过附加监控进程接管；原 runner PID1663 和启动时间保留，没有重启或改变搜索。

## Validation — Actual Commands / Outputs

- `git fetch origin` / `git rev-parse HEAD origin/main`：同为587efdf...，本机 tracked clean。
- 本地独立 `validate_validation_provenance`、`selected_ids`、771文件 SHA inventory：PASS，768 TRAIN raw完整、64锚点、新旧身份桥及两种(4,0.1)冻结参数通过。
- 远端 fast-forward 后 tracked clean，重复 provenance、全部771文件SHA、best.pt SHA、CUDA可用性与GPU型号核对：PASS；正式 Validation count=0。
- GPU：NVIDIA GeForce RTX3080Ti；CUDA FP32 batch16，cpu_cache_and_prefix、no AMP/BF16/FP16/quantization/compile/bucketing。checkpoint SHA=`941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`。
- 正式命令：`/root/miniconda3/bin/python -u code/scripts/run_step6_3d_formal_cpu_matrix_v1.py --phase validation --validation-stage primary --device cuda --batch-size 16 --execution-config code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json`。
- 启动 UTC=`2026-10-01T11:26:16.174919+00:00`，北京时间19:26；runner PID1663。只调用 primary，没有 Stage B fall-through。
- 定期监控与完成验收续办：当前聊天 heartbeat `pi-jwm-stage-a`，每30分钟；不含凭据。SSH密码仅在交互提示中使用，不写入代码/配置/日志。
- 启动记录验证：`python -m unittest discover -s code/tests -p 'test_step6_3d*.py'` 27/27 PASS（20.253s）；`python -m compileall -q code/src code/scripts code/tests` PASS；冻结16项科学source hash未变。Context Consistency Check核对01–08研究范围、架构、数据流、模块、实验、研究者决定、已知问题和变更，运行状态与历史快照明确区分。knowledge-index write通过；最终check/diff及Git身份以本启动记录提交历史为准。
- 2026-10-01 11:37 UTC快照：4/960完成并本地SHA备份，实际4,096 unique transitions，累计665.702s，约21.63 cases/hour；早期ETA约44.2h，尚不足以代表整个矩阵的速度。未发现执行身份错误或scorer exception；仍未完成正式统计/方法选择。

## Results / Expected vs Actual

`STEP_6_3D_VALIDATION_STAGE_A=RUNNING`；尚无最终统计/验收。`SEARCH_METHOD=NOT_SELECTED`，Stage B=`NOT_STARTED`，`locked_test=false`。启动前条件符合预期；运行与960结果验收仍待完成。约53小时估算来自正式TRAIN总吞吐，仅是资源安排，后续以Stage A已完成case实测更新。

## Known Issues

future Return-birth固定支持边界继续保留并单独统计；Grammar dead-end不等同scorer exception。GPU利用率只作诊断，不授权改变batch/precision。任何进度或早期scoreability不构成方法优劣结论。

## Git / Next Step

运行源码固定于587efdf...；本地仅同步运行记录与导航。运行中不向远端拉取新提交。唯一下一动作：监控并持久备份Stage A，960完成后独立验收、正式汇总、Context/Git收口，然后硬停止等待研究者审阅；不会启动Stage B。
