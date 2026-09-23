# STEP 5.5-PATCH — Formal Dataset Full-Consumption & Fixed-Support Closure

## Step Goal

在不改变 Formal Dataset v1 冻结的 H2/L4、60 trajectory、48/12 split、96 transitions、`radius_knn(1000m,k=2)`、模型和 loss 的前提下，补齐全量训练数据消费路径，查明 future Return birth 的固定支持语义，并审计 AirFogSim Task lifecycle collection 修复。CPU only；不做正式训练、GPU、baseline、Planner 或 locked_test。

## Definition Basis

研究者 2026-09-23 STEP 5.5-PATCH 授权；冻结的 Definition 02/03/04/05 与 STEP 4.2C、4.4-PATCH3、5.1A、5.2、5.5 实现合同。当前目录的 `AGENTS.md`、`记录/本地计划表.md` 和 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 是工程治理入口。只读科研定义未修改。

## Initial State

- `main` 起点 `a65bd6d5934568ff186b1b42ab6907e6700e3036`；原有未跟踪 `TASK/` 和 `code/scripts/plot_step5_3e_tiny_overfit.py` 保持不动。
- 五类 Formal Dataset package 已接受，5520 windows；但 `Step52Trainer.from_formal_interface()` 经 `DevelopmentBundle.from_formal_interface()` 读取 `runtime_package_paths`，只消费 1 train + 1 validation。
- 原 acceptance receipt 的 unsupported/unresolved/fixed-support 三个零值仅来自目标帧已有字段，Raw/Sample 真正的 future Return birth 未被检测。
- collector 的 `_repair_duplicate_task_references` 已运行并在 Raw 写入日志，但此前无 60-trajectory 聚合审计。

## Files Involved / Changes / Reuse

- 新增 `code/src/pi_jwm/step5_5_full_sharded_loader_v1.py`：读取完整 index，仅对请求 batch 解压所需 trajectory shard；每 shard 四类 payload 按 Sample ID、split、hash 对齐，跨 shard padding 后交给原 `Step52Trainer` 模型/loss/optimizer。验证按 batch 顺序遍历 validation index。train-only normalization 复用 Formal package base/extension/Flow stats，并从 48 个 train graph shard 流式计算 Physical relation stats。
- `code/src/pi_jwm/step5_2_training_loop_v1.py` 只增加可传入 encoder normalization stats 的构造参数；原 development 和 runtime mini 默认路径不变。
- 新增 `step5_5_fixed_support_audit_v1.py`；`step5_1a_motion_csi_target_contract_v1.py` 在 target-only side metadata 中把真实 future typed Return Flow 对照 current logical Flow index。已完成但 `presence=false` 的已知 Return 也按真实 birth 检查；存在而身份未知的 Return 单独计 `unresolved`。缺少 current slot 不创建 slot、不删 window，也不改 Motion/CSI mask。已有 Return continuation 不标成 birth。
- 将 collector 的同对象集合修复函数原样移至无 AirFogSim 依赖的 `step5_5_lifecycle_repair_v1.py`，便于独立测试；collector 仍调用它，未删除修复。
- 新增 `audit_step5_5_patch_v1.py`、`accept_step5_5_patch_cpu_v1.py` 和 focused tests。未修改五类 frozen package、Raw 或原协议；原 package 目标数组与新增 side metadata 下重新构造的真实样本逐项相等。
- `build_project_knowledge_index_v1.py` 的 Git path 枚举收紧为已跟踪文件；原未跟踪 `TASK/` 和绘图脚本不进入生成索引或本次提交。

## Validation / Results

- CPU full-shard acceptance：`code/artifacts/audit/pi_jwm_step5_5_patch_20260923/cpu_acceptance_receipt.json`，4416/1104 全量索引；跨不同 train/validation trajectory 的 batch；H4 forward、一次 optimizer 参数更新、prior-only H1-H4 validation batch、checkpoint save/load 和 wrong dataset identity rejection。`full_shard_traversal_receipt.json` 逐个读取并核对全部 60 shard 的 Sample/Tensor/Graph/Target 哈希、索引和唯一身份，内存只缓存一个 trajectory shard。此处的“full”指完整数据源可索引且 Trainer 按需消费；没有对 5520 windows 执行正式训练。
- 全量结构与 lifecycle 审计：`code/artifacts/audit/pi_jwm_step5_5_patch_20260923/audit_receipt.json`。5520 windows 中 Return-birth unsupported/fixed-support 均为 8828 次窗口-未来步事件，unresolved=0；涉及 2901 个窗口、60 条轨迹。已有 current Return support 的合法 future continuation 出现 143320 次且未标成 birth。身份未知 Return 的 `unresolved` 正向 fixture 可触发，正式数据未出现该结构。重叠窗口中的同一物理事件可能重复计。旧 receipt 的 0/0/0 已被此检测覆盖。
- lifecycle repair：Raw 记录 213 次、50 条轨迹、124 个 trajectory-task；只允许同一 Python Task 对象重复引用，遇到不同对象同 ID 时拒绝；保留 collection priority 最远 lifecycle。fixture 验证 transmitted/computed/returned/done 直接对象字段不变。没有做“关闭修复再重跑 60 trajectory”的反事实比较。
- 最终 `readiness_receipt.json` 的 8/8 AND checks 为 true：`FORMAL_DATASET_ARTIFACT=READY`、`FULL_FORMAL_DATASET_LOADER=VERIFIED`、`H4_FULL_DATA_CONSUMPTION_PATH=VERIFIED`、`FUTURE_FIXED_SUPPORT_ACCOUNTING=VERIFIED`、`LIFECYCLE_REPAIR_AUDIT=PASS`、`TRAINING_STACK_READINESS=PASS`、`FORMAL_DATASET_READINESS=READY`、`GPU_CODEPATH_READINESS=PREPARED`。`GPU_TRAINING_VERIFIED=false`、`FORMAL_TRAINING=false`、`locked_test_accessed=false`。
- 回归与最终命令/计数以本文件后续验收记录和 process files 为准；`compileall`、`git diff --check`、knowledge index write/check 必须在最后编辑后执行。

## Expected vs Actual / Known Issues

已按 frozen fixed-support 边界识别真实 future-only Return 并保留 Motion/CSI 监督，未开放 future Return birth。正式训练预算和 GPU smoke 尚未进入本 Step；本 Step 的 CPU 数值没有性能含义。旧 STEP 5.5 说明中把 1+1 runtime smoke 写成“Formal package → Trainer”容易误解，现统一限定为 `RUNTIME_MINI_SMOKE=PASS`。

## Git / Next Step

最终 commit、push 和 `HEAD==origin/main` 在完成全部验收后填入。唯一建议下一步：研究者另行授权 STEP 5.6A — GPU Smoke + Formal Training Config Freeze。
