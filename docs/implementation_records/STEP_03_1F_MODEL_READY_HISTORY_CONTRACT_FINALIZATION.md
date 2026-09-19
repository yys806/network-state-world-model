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

## STEP 3.1F-PATCH — Future Action index namespace and provenance correction

### Step Goal

在进入 Batch Dataset 前，修正 Future Action 使用 anchor-only 重新编号而造成的 History-union index 错位；同步修正 machine-readable input-index policy，并把 future-reference audit JSON 纳入可追溯 artifact。

### Definition Basis

沿用 `02数据集构建与模型输入.md` 的 History causal union 定义和本记录的 STEP 3.1F 合同；本 patch 不改变 History 时间边界、anchor visibility 保护、Raw 语义或后续模型范围。

### Initial State

`994da0b` 已完成 History past `A/Y` 和 union index，但 Future Action 使用 `anchor_node_index/anchor_task_index` 写入数值 index；validator 只检查非负值。`TensorContract.input_index_policy` 仍为 `decision_visible_objects_only`，audit JSON 尚未被该 commit 纳入 Git。

### Files Involved

- `code/src/pi_jwm/model_ready_sample_contract_v1.py`
- `code/scripts/build_step3_1_minimal_real_model_ready_sample_v1.py`
- `code/tests/test_model_ready_sample_contract_v1.py`
- `code/scripts/audit_step3_1f_future_action_references_v1.py`
- `code/artifacts/protocols/pi_jwm_model_ready_sample_v1_20260919/`
- 本合同、Tracker、authority records、AI_CONTEXT 和知识索引。

### Changes

- Future Action 先用 anchor visibility map 拒绝不可见 object，再用 History-union `static.input_entity_index` 返回正式 task/node/UAV index。
- validator 新增 `future_action_indices_match_input_index`，逐一核对 action object ID 与 numeric index。
- 新增 disappearing-object fixture：anchor-only 集合缩小时仍保持统一 History-union index；新增 validator 错位拒绝测试。
- `input_index_policy` 改为 `history_causal_observable_object_union`，并显式记录 `future_action_index_policy=anchor_visibility_then_history_union_input_index`。
- builder 增加显式 `--refresh-existing`，重新生成带新 policy/checks 的 sample；manifest 保存 future-reference audit 的路径、SHA-256 和 `observation_only=true`。
- 重新生成并纳入 `future_action_reference_audit.json`；结果仍仅为 observation，不外推正式 Dataset 可用率。

### Reuse

复用既有 History union、四类 action schema、strict reference resolver、真实 Raw artifact、future-reference observation audit 和 round-trip 机制；未修改 Raw Layer、Flow/DAG 语义、双图、World Model、Loss、Planner 或训练。

### Validation

- 旧实现上新增 disappearing-object fixture 先失败：`Task_2` 的 History-union index `1` 被错误写成 anchor-only index `0`。
- 修正后专项测试 `12/12 PASS`。
- `build_step3_1_minimal_real_model_ready_sample_v1.py --refresh-existing` 通过，manifest checks 全部为 `true`，包括 `future_action_indices_match_input_index`。
- future-reference audit 重新运行，4 个非 locked Raw artifact、18 个窗口、0 个 unresolved reference；JSON SHA-256 已写入 sample manifest。
- compileall、knowledge index write/`--check`、`git diff --check` 均需在提交前重新执行。

### Results

Future Action 的合法引用现在共享 `static.input_entity_index` 的正式数值 namespace；anchor visibility 仍独立负责拒绝不可见引用。sample artifact 的 machine contract policy 已准确记录 History causal union，audit JSON 具备 Git provenance。

### Expected vs Actual

符合预期：修正了 disappearing-object 场景的 index 错位，未扩大科研范围或改变因果边界。

### Known Issues

正式 batch/split Dataset、模型、训练和 future-arrival 研究决策仍未开始；当前 audit 仍只是短 Raw 轨迹 observation。

### Git

本 patch 使用独立 Conventional Commit 推送 `main`；最终 commit/hash 和远端回执在 Completion Report 中记录。

### Next Step

研究者审阅后，唯一建议下一步仍为 STEP 3.2 — Raw-to-Dataset Batch / Split Preprocessing Validation；本 patch 不自动进入该 Step。
