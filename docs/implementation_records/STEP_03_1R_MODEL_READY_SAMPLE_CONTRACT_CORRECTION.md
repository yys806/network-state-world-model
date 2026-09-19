# STEP 3.1R — Model-ready Sample Contract Correction

## Step Goal

修正 commit `09dbbd7` 中的时间窗口、固定 index/presence、DAG 来源、typed target index、relation endpoint 和 Action reference 合同；重新生成一条非 locked `H=2/L=2` 真实样本。STEP 3.2、正式数据集、模型和训练不在本步范围。

## Definition Basis

- 研究者只读定义：`D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`。
- 研究者明确修正规则：`History=H_{t-H+1:t}`，当前 `O_t` 在 History，当前 `A_t` 从 Future Action 开始。
- 已冻结 Raw 合同；本步仅增加 Decision-time DAG capture amendment，不改变通信、加速度、CPU 或 future-task 语义。

## Initial State

- Step 3.1 使用 History `[0,1]`，漏掉 anchor `O_2`。
- History 每帧只写存在 row，没有为 anchor index 中但早期不存在的对象输出 presence/padding。
- observer 已有 `_extract_dag_edges(env)`，但 Raw JSON 未接线。
- target index 混合对象类型；relation endpoints 为空；Action reference 可静默变成 `-1`。

## Files Involved

- `code/src/pi_jwm/model_ready_sample_contract_v1.py`
- `code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py`
- `code/scripts/run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py`
- `code/scripts/build_step3_1_minimal_real_model_ready_sample_v1.py`
- `code/tests/test_model_ready_sample_contract_v1.py`
- Raw v2 和 model-ready sample artifacts、合同、Tracker、authority records、AI_CONTEXT 和索引。

## Changes

- History 改为 `[t-H+1,t]`；validator 明确检查 History 最后一帧和 Future Action 第一帧均等于 anchor。
- input index 由 anchor `O_t` 可见 physical/task 建立并固定贯穿 History；早期不存在对象输出 `presence=false`、feature mask false、padding/null。
- Raw `_capture()` 序列化真实 observer DAG。两端当前可见的边进入 Decision；含未来端点的边仅进入 internal metadata。
- target index 拆为 `physical/task/flow` namespace。
- anchor `channel_rows` 的 source/target 映射为 stable physical index，保存 validity mask。
- Action reference 必须解析到合法 input index；否则抛出带 `RESEARCHER_DECISION_REQUIRED` 的错误，不再生成 `-1`。

## Reuse

复用已有真实 observer、Step 2.4 runner、稳定排序 ID/index 和 JSON round-trip。没有修改第三方 AirFogSim，也没有从旧 tensor 推断新语义。

## Validation

```powershell
$env:PYTHONUTF8='1'
conda run -n airfogsim --no-capture-output python .\code\scripts\run_step2_4_real_airfogsim_communication_outcome_semantics_v1.py
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python -m unittest discover -s .\code\tests -p 'test_model_ready_sample_contract_v1.py' -v
python -m unittest discover -s .\code\tests -p 'test_*.py'
python -m compileall -q .\code\src .\code\scripts .\code\tests
python .\code\scripts\build_project_knowledge_index_v1.py
python .\code\scripts\build_project_knowledge_index_v1.py --check
git diff --check
```

真实 AirFogSim 14/14 checks 通过；model-ready 17/17 checks 通过；focused tests 6/6 通过；serialize/load equality 通过。scope 为 `locked_test=false`、`training=false`、`gpu=false`。

仓库级历史套件共运行 1662 项，其中 33 项 error。报错包括主环境 GBK 无法输出 AirFogSim emoji、已归档 dataset/evaluation artifact 在当前 checkout 不存在，以及历史 teacher-tensor fixture 与旧 RB 约束不一致。这些报错不位于本次修改的 Step 3.1R 合同测试，不作为本步验收通过证据；本步验收仍以上述真实 Raw、样本 manifest 和 6 项专项测试为准。

## Results

- History `[1,2]`；Future Action/Target `[2,3]`；当前 `O_2` 在 History。
- 42 条真实 channel relation endpoints 全部映射到 stable physical index。
- anchor Decision 有 2 条当前可见 DAG edges；30 条含未来端点的 edges 只在 internal metadata。
- target namespaces 为 physical/task/flow；target-only task 为 `Task_7/Task_8`。
- sample 中没有负 index；fixture 证明未解析 future-only Action 会被明确拒绝。
- fixed-index fixture 证明较早帧缺失 physical/task row 使用 presence false、mask false 和 padding/null。

## Expected vs Actual

六项审阅问题全部按预期修正。真实轨迹本身没有 History 中途出现的 physical/task，因此该边界由使用真实 Raw shape 的最小 fixture 验收；没有把 fixture 冒充真实事件。

## Known Issues

- 本步只验收一条最小样本，不代表正式 batch/split preprocessing 已完成。
- 若后续真实 Future Action 引用 anchor 时刻不可见 task，当前合同会阻断并要求研究者决定，不自行建立未来对象输入 index。
- 历史全量套件对已归档大 artifact 和特定控制台编码有环境依赖；本步没有为让它们通过而恢复历史数据或改动旧合同。

## Git

本记录与代码、Raw v2、sample artifact、manifest、测试和索引使用一个 Conventional Commit 推送 `main`；最终 hash 在 Completion Report 记录。

## Next Step

停止并等待研究者审阅；不自动进入 STEP 3.2。
