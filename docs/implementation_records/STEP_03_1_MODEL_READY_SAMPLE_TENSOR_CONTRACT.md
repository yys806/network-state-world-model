# STEP 3.1 / 3.1R — Model-ready Sample & Tensor Contract Freeze and Correction

## Step 3.1R Correction Record

审阅 commit `09dbbd7` 后修正六项边界：History 改为 `[t-H+1,t]`；input index 在整个 History 固定并显式输出不存在对象的 presence/feature mask/padding；Raw observer DAG rows 接入并按 anchor 可见 task 过滤 future-only endpoints；target index 拆为 physical/task/flow namespace；channel relation endpoints 映射到 stable physical index；所有 Action reference 解析失败即拒绝，不再静默写 `-1`。

真实 v2 Raw 证据来自 `code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/`。anchor frame 2 的样本为 History `[1,2]`、Future Action/Target `[2,3]`；Decision 可见 DAG 2 条，internal future DAG 30 条；relation endpoints 42 条；`Task_7/Task_8` 仍只在 target-side task namespace。

## Step Goal

从已冻结的 Step 2 Raw Trajectory 构造一条 `D_t = (History, Static, Future Action, Target, Metadata)` 最小真实样本，冻结时间窗口、输入 index、presence/mask/padding、四类 action、通信 service 与 task progress 分离、Flow/DAG gap、Static/Metadata 边界和 train-only preprocessing 原则。本步不生成正式大数据集、不训练、不进入双图/World Model/Loss/Planner。

## Definition Basis

- 研究者只读定义：`D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`。
- Step 2 Raw Contract 与最终通信证据：`docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`、`code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/`。
- 研究笔记目录未修改。

## Initial State

旧 formal dataset/window/tensor 已有 ID/index、mask、DAG endpoint、trajectory split 和 masked normalization 机制，但旧 `flow_state` 语义不能直接代表新定义 Flow/Data。Step 3.1 当时误判 Raw DAG source 缺失；Step 3.1R 已确认 observer 的真实 `_extract_dag_edges` 并完成接线。

## Files Involved

- `code/src/pi_jwm/model_ready_sample_contract_v1.py`
- `code/scripts/build_step3_1_minimal_real_model_ready_sample_v1.py`
- `code/tests/test_model_ready_sample_contract_v1.py`
- `docs/contracts_PIJWM_MODEL_READY_SAMPLE_TENSOR_CONTRACT_V1.md`
- `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`
- Tracker、authority records、AI_CONTEXT 和项目索引。

## Changes

- 固定最小 `H=2, L=2`：anchor frame 2 的 History 是 frame 1–2；Future Action/Target 是 frame 2–3；当前 action 不进入 History。
- 输入 index 只使用 anchor 前可见对象；Target 单独建立 target-side index 和 `target_only_objects`，不把未来对象加入 input index。
- 三态语义固定为 object presence、feature mask、真实零值；padding/missing 不参与 normalization。
- Route/Comm/Comp/UAV Mobility 四类 action 保持统一结构；empty/no-op 与 missing 分开；车辆运动保留 `SUMO external`。
- Raw simulator acceleration 只作 audit；canonical causal acceleration 是候选 model condition；future schedule 只作 internal metadata。
- communication service 保存 wireless/wired/total；明确它们是 hop-level transport volume，不是 task end-to-end progress；task transmitted progress/lifecycle 单独保存。
- Flow/Data 依据真实 transfer event 的 task/transport/source/destination 生成 hop-level flow ID。DAG rows 来自真实 observer；future-only endpoint 只保留 raw count，不进入 input-side Static。
- Static 保存 index、关系端点和对应关系；Metadata 保存 trajectory/seed/split/time/contract/audit，不作为模型输入。
- 固定 `Trajectory-level Split -> Window Construction -> Train-only preprocessing fit -> Validation/Test apply`，统计只使用 valid/masked-in 值。

## Reuse

复用旧实现的稳定 ID/index、presence/mask、trajectory split、DAG endpoint serialization 和 masked normalization 原则；不复用旧 `flow_state` 的新语义，不为兼容旧 tensor 改变 02 定义。

## Validation

```powershell
python .\code\scripts\build_step3_1_minimal_real_model_ready_sample_v1.py
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python -m unittest discover -s .\code\tests -p 'test_model_ready_sample_contract_v1.py' -v
python -m compileall -q .\code\src .\code\scripts .\code\tests
```

真实样本 artifact 的 10/10 checks 通过；测试 4/4 通过；serialize→load equality 通过；source/artifact hash 在 manifest 中保存。证据 scope 为 non-locked、`H=2`、`L=2`、training=false、gpu=false。

## Results

- History frame `1,2`，Future Action/Target frame `2,3`；History 最后一帧等于 anchor，Future Action 第一帧等于 anchor。
- 输入 task index 为 `Task_1, Task_2, Task_4, Task_5, Task_6`；Target-only 新对象包含 `Task_7, Task_8`，没有进入 input index。
- 四类 action 均存在正式 tensor 结构；empty/no-op 与 missing 可区分。
- DAG source 为 `airfogsim_full_dual_graph_observer_v1._extract_dag_edges`；Decision 可见 edge count=2，internal future edge count=30，Dataset 不读取未来端点身份。
- Target 保存 communication service split/total 字段；service 语义明确为 hop service，task transmitted progress 独立保存。

## Expected vs Actual

时间、索引、mask、动作和 round-trip 均符合预期。真实 Step 2.4 窗口的两个 target slot 没有 transfer event，因此该最小样本的 flow rows 为空；Flow 合同仍由真实 event schema 定义，需后续样本覆盖有服务 slot。该限制不被隐藏。

## Known Issues

- Raw observer 已提供 DAG rows，Step 3.1R 已将其接入 Raw v2 artifact；后续仍需在批量构建中保持 future-only endpoint 过滤。
- 本步只完成一个最小样本，不代表正式数据集规模、模型输入最终选择或训练资格。
- 旧 tensor/checkpoint 不能因 shape 相同而视为新合同兼容。

## Git

本步完成验证后使用独立 Conventional Commit 推送 `main`；artifact 和 manifest 需 force-add。

## Next Step

研究者审阅后，唯一建议下一步是单独授权 **STEP 3.2 — Raw-to-Dataset Batch / Split Preprocessing Validation**。
