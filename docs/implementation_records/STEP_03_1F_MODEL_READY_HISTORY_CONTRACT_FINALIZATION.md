# STEP 3.1F — Model-ready History Contract Finalization

## Step Goal

将 Model-ready History 最终冻结为 `O_{t-H+1:t} + A_{t-H+1:t-1} + Y_{t-H+1:t-1}`，并以整个 History 的因果可见对象并集建立稳定 input index。只完成最小样本与合同验收，不进入 Step 3.2、正式数据集、模型或训练。

## Definition Basis

- 研究者只读定义：`D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`。
- 研究者明确要求当前 `A_t/Y_t` 不进入 History，Future Action/Target 仍从 t 开始。
- STEP 3.1R 的 DAG、typed target namespace、relation endpoint 和 unresolved-reference 保护继续有效。

## Initial State

- History 只有 `O_{t-1},O_t`，没有过去已执行 `A_{t-1}` 和真实 `Y_{t-1}`。
- input index 只由 anchor `O_t` 建立，会漏掉过去存在但当前离开/完成的对象。
- 历史 Outcome 的 flow/service、relation、DAG 尚未映射到 History stable index。
- 未形成跨当前非 locked Raw artifacts 的 future-action unresolved-reference 机器审计。

## Files Involved

- `code/src/pi_jwm/model_ready_sample_contract_v1.py`
- `code/scripts/build_step3_1_minimal_real_model_ready_sample_v1.py`
- `code/scripts/audit_step3_1f_future_action_references_v1.py`
- `code/tests/test_model_ready_sample_contract_v1.py`
- Model-ready sample、future-reference audit、合同、Tracker、authority records、AI_CONTEXT 和索引。

## Changes

- History 每帧保留 observation；所有 `tau<t` 的行同时保存同 frame 的四类 action 和真实 outcome；当前 t 行不含 action/outcome。
- input physical/task index 改为 History observation 的因果 union；flow index 来自过去 Outcome 中真实出现的 transfer events。
- 每个 History observation/outcome 为 union 中所有 physical/task 输出固定行；late-entry 和 disappearing 使用 presence/mask/padding 表达。
- past action、flow、relation endpoint、DAG endpoint 全部解析到同一 History index；禁止负 index。
- Future Action 仍按 anchor `O_t` 可见 task/entity 解析；History union 中仅过去可见而当前不可见的对象不会被用来绕过 unresolved-reference 保护。
- 新增 observation-only 扫描器，对可用非 locked Raw 的 `A_{t+1:t+L-1}` 引用进行统计，不丢弃 window 或自行决定未来对象建模。

## Reuse

复用 STEP 3.1R 的四类 action tensor、strict index resolver、真实 DAG observer、channel relation、Flow ID 和 JSON round-trip。未修改 Raw Layer 或第三方 AirFogSim。

## Validation

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src;D:\shen\PKU\PIJWM\code\scripts'
python -m unittest discover -s .\code\tests -p 'test_model_ready_sample_contract_v1.py' -v
python .\code\scripts\build_step3_1_minimal_real_model_ready_sample_v1.py
python .\code\scripts\audit_step3_1f_future_action_references_v1.py
python -m compileall -q .\code\src .\code\scripts .\code\tests
python .\code\scripts\build_project_knowledge_index_v1.py
python .\code\scripts\build_project_knowledge_index_v1.py --check
git diff --check
```

最终输出、测试数量、checks 和哈希在提交前重新核对。

## Results

- 最小真实样本为 `O_1+A_1+Y_1+O_2`，当前 `A_2/Y_2` 不在 History；Future Action/Target 仍为 frame 2、3。
- 历史真实 frame 1 的 action/outcome 已进入 History；其 relation/DAG endpoint 使用稳定 physical/task index。
- 真实 anchor=2 的过去 step 1 没有 flow event；Flow identity/presence 机制另由真实 Raw shape 的最小 fixture 验收，不能把 fixture 写成真实事件。
- late-entry 与 disappearing physical/task 的 `false->true`、`true->false` 已由 fixture 验收。
- future-reference audit 扫描 4 个版本不同的非 locked Raw artifacts，共 18 个可构造窗口；0 个窗口、0 个引用受到影响，四个 action family 和 task/physical_entity 两类对象均为 0。机器汇总同时记录 affected window rate=0.0、审计 offset=[1]。该结果只描述当前短轨迹，不决定未来任务到达的建模方案。

## Expected vs Actual

两个 History 合同问题均按定义实现。当前真实样本能够直接证明 past action/outcome 与当前帧无泄漏；对象进出和历史 flow presence 的稀有边界由最小 fixture 验收并明确标注。

## Known Issues

- 18 个可构造窗口来自 4 个版本不同的短 Raw artifacts，不等于正式 dataset 的可用率，也不能证明未来大规模轨迹不会出现 unresolved reference。
- future action 出现 anchor 不可见引用时仍会明确阻断并要求研究者决定。
- 正式 batch/split/preprocessing 尚未实现。

## Git

本记录与代码、artifact、测试、索引使用一个 Conventional Commit 推送 `main`；最终 hash 在 Completion Report 记录。

## Next Step

研究者审阅后，单独授权 STEP 3.2 — Raw-to-Dataset Batch / Split Preprocessing Validation；不自动执行。
