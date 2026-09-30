# STEP 6.3D-GPU-EXECUTION — 24GB GPU 批量推演闭合

## Step Goal
在不改变 Planner v1 语义、候选域、H4 scorer 或 B_WM 定义的前提下，验证冻结 checkpoint 的 FP32 CUDA 批量一步推演，并确定 24GB GPU 的稳定配置。正式 TRAIN tuning、Validation comparison、方法选择和 locked_test 均未运行。

## Evidence
机器收据位于 `code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930/`：GPU 环境、CPU/GPU 等价、吞吐、分桶、验收和 SHA manifest。

## Results
RTX 4090 24564 MiB，PyTorch 2.8.0+cu128，checkpoint SHA 与冻结值一致。真实 TRAIN 两个 fixture 的 batch 1/4/8/16/32/64/128/256 单步等价通过；H4 grammar、H_sup、scoreability、Objective tuple 和严格排序一致。中间浮点诊断最大差 1.220703125e-4，保留原值且不改 1e-7 Comp 容差或离散门槛。

吞吐探测 batch 8/16/32/64 稳定，batch 128/256 OOM。batch 32 中位数约 12.71 unique transitions/s，峰值约 2.03 GiB；相对 CPU batch8 0.964/s 约 13.19x，2,113,536 次转移理论约 46.2 小时。结构分桶等价通过但中位吞吐约为未分桶 0.910，故不采用。

## Implementation
仅增加 device-aware frozen loader、GPU equivalence/throughput/bucket/environment receipts 和 CUDA 分支的等价 SINR 批量执行；CUDA 分支保持原公式与索引累加语义，CPU 分支不变。anchor encoder 只初始化一次；重复 transition 命中 cache 且不消耗 B_WM。

## Limitations
未执行正式 TRAIN tuning、Validation comparison、HRS/S-CEM/MH-CEM 选择、训练、closed loop 或 locked_test。GPU 结果是执行路径/吞吐证据，不是方法性能结果。
