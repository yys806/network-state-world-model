# STEP 6.3D-GPU-PIPELINE-OPT — GPU 流水线性能剖析与有限优化

## Step Goal
在不改变 6.3D 科研语义的前提下定位 RTX 4090 batch=32 路径瓶颈，并执行停止规则。

## Definition Basis / Initial State
沿用已接受的 FP32、batch=32、无 bucketing、冻结 checkpoint 与逐 unique transition 的 B_WM 计费。基线为 12.71 transitions/s，峰值约 2.03 GiB。

## Changes
新增 `run_step6_3d_gpu_pipeline_profile_v1.py` 与 08–11 机器收据；未改变候选域、Grammar、Objective、cache key、模型或搜索方法。

## Validation
真实 non-locked TRAIN anchor 的 32 请求波次 profiling：model.one_step 2.3360 s（87.58%），scorer/H_sup 0.1707 s（6.40%），proposal/domain/bind 0.1039 s（3.90%），fingerprint/cache 0.0566 s（2.12%），总计 2.6672 s。既有 CPU/GPU 与 H4 等价收据继续作为正式等价门证据。未运行正式 TRAIN tuning、Validation comparison 或 locked_test。

## Results / Decision
模型前向是主要瓶颈。fingerprint/cache fast path 预期最多影响约 2.12%，CPU control shadow 与 multi-anchor wavefront 没有 profiler 支持，均不采用。优化吞吐按已接受配置保持 12.71/s，speedup 1.0，未达到 20% 阈值；正式配置保持 FP32、batch=32、no bucketing。2,113,536 transitions 理论约 46.2 小时。

## Readiness / Limitations
`H4_SEARCH_COMPARISON_READINESS=SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`。future Return birth 是固定支持限制；H4 scoreable success rate 仍需在正式比较中单独报告。正式方法比较尚未运行。

## Evidence
`code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930/08_gpu_pipeline_bottleneck_profile.json`、`09_pipeline_optimization_assessment.json`、`10_pipeline_optimization_acceptance.json`、`11_pipeline_optimization_manifest.json`。
