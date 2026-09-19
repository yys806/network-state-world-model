# STEP 3.3 — Model Input Tensor / Collation Contract Freeze

状态：最小 CPU fixed-shape tensor/collation contract 已完成；不代表双图、World Model、Loss、Planner 或训练已实现。

## Step Goal

将冻结的 STEP 3.1F JSON sample / STEP 3.2 batch 转为可复现的固定形状 CPU NumPy tensor，保持 History union index、anchor visibility、presence/mask 和四类动作引用语义。

## Definition Basis

- `docs/contracts_PIJWM_MODEL_READY_SAMPLE_TENSOR_CONTRACT_V1.md`
- `code/src/pi_jwm/model_ready_sample_contract_v1.py`
- STEP 3.2 batch/provenance bundle

## Initial State

STEP 3.2 已生成 12 个 H=2/L=2 non-locked JSON sample；尚无新定义的 fixed-shape tensor/collation 实现。

## Files Involved

- `code/src/pi_jwm/step3_3_model_input_tensor_v1.py`
- `code/scripts/build_step3_3_model_input_tensor_v1.py`
- `code/tests/test_step3_3_model_input_tensor_v1.py`
- `code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/`

## Changes

- 增加 `TensorContract`：H/L、development capacities、feature order、dtype、padding、categorical vocabulary、sample contract version 和 input/future-action index policy。
- History entity/task/flow、relation、DAG、past/future action、route hop、Comm RB、target namespaces 均生成固定形状数组；padding index 为 `-1`，numeric padding 为 `0`，presence/mask 独立保存。
- 生成前显式检查 stable ID ↔ JSON index ↔ tensor slot；容量超限拒绝，禁止静默截断。
- target-only namespace 使用独立 target capacities，不写入 input-side index。
- 同时保留 raw/normalized 连续值、flow task/source/destination/transport 引用、route/Comm/Comp/Mobility 的 past/future validity、entry、hop/RB mask；当前 development bundle 未暴露的 channel numeric、CPU capacity 和 Comp allocation 写入 Gap Table，不伪造字段。
- NPZ save/load 保留 contract 与 sample IDs；builder 生成 schema/manifest/hash。

## Reuse and Boundary

复用 STEP 3.1F sample 的真实字段和既有 ID/index/mask 合同；未修改 Raw、Flow/DAG 科研语义、双图、World Model、Loss、Planner、训练、GPU 或 `locked_test`。Development capacities 只表示当前三条 development trajectory 的观测上界，不是正式研究容量。

## Validation

- `test_step3_3_model_input_tensor_v1.py`: 3/3 focused tests passed。
- builder：12 samples，H=2/L=2，manifest 绑定 Step 3.2 输入 manifest SHA。
- manifest 保存全部数组 shape/dtype、mask/presence counts、categorical vocab、units、Gap Table 和 capacity receipt。
- 覆盖 fixed shape、missing != real zero、presence padding、capacity overflow、round-trip；动作 ID/index 校验沿用并在 tensorization 入口再次执行。
- scope：`locked_test=false`、`training=false`、`gpu=false`、`formal_dataset=false`。

## Results and Limits

符合最小 contract 冻结目标。得到的是 CPU JSON-to-tensor/collation 机器证据，不是模型读路径、图语义、训练输入或正式 Dataset 结论。当前 development batch 没有 route action entry，因此 `max_route_hops=0` 是观测结果；正式容量仍未决定。

## Git

本记录与代码、测试、artifact 一并提交；推送状态以最终 Git 回执为准。

## Next Step

研究者审阅 STEP 3.3 contract；若继续，单独授权并先冻结 Physical/Information 双图字段映射，不自动进入 03。
