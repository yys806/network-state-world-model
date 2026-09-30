# STEP 6.3D-3080TI-MIGRATION-QUALIFICATION

## Step Goal and Definition Basis

将冻结 Formal Dataset 和 `best.pt` 迁至 RTX 3080 Ti，验收数据身份、FP32 候选推演、吞吐与正式 CUDA matrix runner；停止于正式 TRAIN 调参和 Validation 比较之前。研究依据为研究者授权的只读 `06策略器与候选动作规划.md` §3.1，SHA-256 `f20294bd8708076ae7583be679a8182a0990ecf6777e05df378ae4c2855431f1`；具体执行边界由本 Step 用户指令冻结。

## Initial State and Files Involved

开始时 `origin/main=fc719dec323a8b74c59c14c7a6cdbe4274ac2ce1`。4090 是唯一正式数据迁移源及 fallback；目标为 `NVIDIA GeForce RTX 3080 Ti`，12 GiB，driver 595.71.05、PyTorch 2.8.0+cu128。Git 跟踪的协议、清单、目录从 `origin/main` 获取；本地大文件从 4090 迁移。冻结 checkpoint SHA-256 为 `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`。

主要源码：`code/scripts/run_step6_3d_one_cpu_solve_v1.py`、`run_step6_3d_formal_cpu_matrix_v1.py`、GPU environment/equivalence/throughput 脚本、迁移校验/小规模 smoke/acceptance 脚本。机器证据位于 `code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/`，大 Dataset/checkpoint 不入 Git。

## Changes and Reuse

4090 到 3080 Ti 使用 `rsync -a --partial --append-verify`，保留目录结构、不删除源文件。源 4090 的工作数据集 266/266 文件独立 SHA 校验通过；其 GPU fixture 副本只有 2 条 Raw，剩余 58 条按研究者明确授权从本地副本补齐，并逐项与冻结 Raw provenance SHA 比对。60/60 Raw 一致。一次中断的早期 relay 留下 8 MiB partial 文件，核对直接 rsync 完成后仅删除该目标 partial；源正式数据和 checkpoint 未改。

复用 `rollout_one_step_batch`、`TransitionBudgetAccountant`、`solve_fixed_budget` 及原评分器。CUDA 上保留 FP32 推演，精确张量结果和搜索前缀放主机内存，避免 B_WM 增长时 GPU 缓存耗尽。CPU path 保留。每个 solve 记录设备、GPU、精度、batch、state storage、checkpoint/source SHA、execution config ID、bucket strategy、`locked_test=false`；resume 逐字段拒绝旧 CPU/new GPU 或不同 batch/config 混用。未改 World Model、CandidateDomain、Grammar、HRS/S-CEM/MH-CEM、Objective、H_sup、B_WM 定义或 anchors。

## Validation and Results

- 正式 loader/manifest 校验 PASS：48/12 trajectory、4416/1104 windows、32/64 选中锚点、零 cohort 0、静态空域 0；normalization、sample/tensor/graph shards、Raw、deadline sidecars、6.3B catalog、checkpoint data/config/source 身份均 PASS。4090/3080 Ti Dataset identity 一致。
- 真实 TRAIN fixture CPU↔3080 Ti 单步与 H4 离散等价 PASS；覆盖 action、latent/state/graph、支持/Grammar/H_sup、future Return birth 边界、scoreability、Objective/字典序、cache 与 B_WM。GPU 结果移入 CPU 存储与原直接 GPU 执行在 B_WM256 的离散结果和预算一致。
- offload 稳态 batch 8/16/32/64 各三次：中位 10.3593/10.9154/12.4562/12.6699 unique transitions/s；peak VRAM 44.43/80.33/153.34/291.09 MB。B_WM1024、batch64 的单 TRAIN HRS 容量诊断完成。原全 GPU 缓存 batch64 在 B_WM256 OOM，已废弃。
- 冻结 batch16：虽然 32/64 短探针更快，但相同 TRAIN anchor 的 MH-CEM K4、B_WM256 在 batch32/64 得到 0 个完整 H4，batch16 得到 66 个完整、63 个可评分 H4；因此批量不能只按短吞吐最大化。正式配置 ID `3834e6b2e76c93e66dc92f0b3d0c56ad7e753c07be9ef0b8cc022a2596b7b680`。
- 正式 runner bounded TRAIN smoke：HRS 和 S-CEM 各一次、B_WM256、batch16，均真实走 CUDA；分别 64/48 个完整且可评分 H4，unique transitions=256，cache hits=2/3；resume 等值并拒绝 batch 篡改。没有进入正式 TRAIN tuning。
- 本地 `python -m unittest discover -s code/tests -p 'test_step6_3d*.py'`：15 tests PASS；`python -m compileall -q code/src code/scripts code/tests`：退出码 0；`python code/scripts/build_project_knowledge_index_v1.py` 与 `--check`：均 `passed=true`、`mismatches=[]`；`git diff --cached --check`：退出码 0。`build_step6_3d_3080ti_acceptance_v1.py` 重建接受收据为 PASS，SHA 清单为 `15_sha_manifest.json`。AI_CONTEXT 00–08 逐项检查：01 研究背景和 06 决定无新科研变更；00/02/03/04/05/07/08 已同步本次事实。

## Expected vs Actual and Known Issues

符合迁移、身份、GPU 等价、runner 和小规模 smoke 的预期。batch16 选用是吞吐与 H4 完整度共同约束的工程配置。10.9154 tps 为短稳态估计：393,216 transitions 约 10.01 h，2,113,536 transitions 约 53.79 h，与 4090 12.71 tps 比为 0.8588；长预算 B_WM1024 主机存储诊断更慢，完整矩阵耗时未经实测。future Return birth 仍是已知 fixed-support limitation；`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`。未运行正式 TRAIN tuning、Validation comparison、locked_test、重训或 5090 测试。

## Git and Next Step

源码、small receipts、上下文、索引和本记录已推送 `origin/main`。实现提交：`c344eb12bb0be953b3e434c7c3783a49ac0d242d` (`feat(step6.3d): qualify 3080 Ti formal GPU runner`)。`SOURCE_4090_CAN_BE_STOPPED=true` 只表示本 Step 证据允许研究者决定停机，未执行云平台停机/释放/删盘。唯一下一动作：研究者审阅本 Step；正式 STEP 6.3D TRAIN tuning 需后续单独指令。
