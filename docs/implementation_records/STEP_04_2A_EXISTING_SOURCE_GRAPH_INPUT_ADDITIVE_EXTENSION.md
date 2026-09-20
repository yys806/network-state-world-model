# STEP 4.2A — Existing-Source Graph Input Additive Extension

状态：COMPLETE；Graph Builder NOT STARTED。

## Step Goal

把 STEP 4.1 已确认 Raw/Simulator 已存在或可由当前 Decision 因果推导的定义 03 minimum inputs，以版本化 additive extension 贯穿 Raw → Model-ready Sample → trajectory split/train-only preprocessing → Tensor。明确不补造 stateful Flow。

## Definition Basis

- 只读 `D:\shen\OB\科研\PIJWM\02数据集构建与模型输入.md`、`03物理-信息双图建模.md`。
- STEP 3.1F/3.2/3.3 frozen causal/index/split/mask contract。
- STEP 4.1 mapping 与 PATCH 的 object-field-relation、wired 和 CPU 四语义边界。

## Initial State

Step 3 Tensor 只有 speed/canonical acceleration/task size 等当前最小输入；position、numeric CSI、CPU static capability、Task current demand/progress/time 和 typed Task–Agent relation 尚未暴露。wired topology 只在 environment 保存，未逐 Decision 物化。stable stateful Flow 的 Raw 信息不足。

## Files Involved

- `code/src/pi_jwm/step4_2a_graph_input_extension_v1.py`
- `code/scripts/build_step4_2a_graph_input_extension_v1.py`
- `code/tests/test_step4_2a_graph_input_extension_v1.py`
- `docs/contracts_PIJWM_STEP_04_2A_GRAPH_INPUT_ADDITIVE_EXTENSION_V1.md`
- `code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920/`
- Step 4.1 mapping status reference、Tracker、authority/process records、AI_CONTEXT 和 registries。

## Changes

- Raw v1 amendment 为三条 development trajectory 的每个 Decision 增加 wired typed relation rows；旧 Raw 文件不覆盖。
- Sample v5 增加 position、typed Comm、Agent static CPU capability、Task demand/progress/elapsed 和 Src/Host/Exec/Ret；保留 Step 3.1F History union 与 Future Action index policy。
- Dataset/preprocessing v1 继续 trajectory-level split；新增 9 项 train-only mask-aware normalization statistics 与单位。
- Tensor v3 additive 增加 Physical position、typed Comm、static CPU、Task extended features 和 typed Task–Agent arrays；保留 Step 3.3 v2 arrays 和 semantic checks。
- 形成 Step 4.1 gap-resolution overlay；不改写 Step 4.1 当时“未暴露”的历史事实。

### STEP 4.2A-PATCH — Comm Mask Semantics & Gap Reclassification

- 依据 `_physical_structure()` 修正 wireless 语义：structural relation presence/validity 不再依赖 CSI `observed_mask`；CSI 缺失时保留 relation，记录 missing reason，CSI mask=false、placeholder=0。
- Sample/Tensor validator 增加 `relation_validity_independent_of_feature_observability`、masked-zero、inactive-relation feature-mask 检查；machine receipt 实际运行 missing-CSI 正向与删除-relation 负向 counterfactual。
- 核实 `_extract_tasks()`：return size、priority、deadline 改为 `SIMULATOR_OBSERVER_AVAILABLE_BUT_FROZEN_RAW_NOT_EXPOSED`；task delay 已由 Step 4.2A 因果 elapsed 表达。本 Patch 未修改 Raw collector 或输入化前三项。
- stateful Flow 继续 `RAW_INSUFFICIENT`；旧 `LogicalFlow`、`CarryingHop` 和 `past_outcome_flow_service` 均不作为定义 03 current Flow 已存在的证据。

## Reuse

复用三条 Step 3.2 development trajectory、Step 3.1F sample builder 的时间/index/action/target 语义、Step 3.2 Raw continuity/split 和旧字段 normalization、Step 3.3 fixed-shape/action/target arrays。未修改第三方 AirFogSim、旧 artifact、模型或图算子。

## Validation

- TDD red：新增 focused test 先以 `ModuleNotFoundError` 失败。
- PATCH TDD red：missing-CSI fixture 先以 `ValueError: wireless CSI and RB identity lengths differ` 失败，validator negative fixture 先因缺少 `relation_validity_independent_of_feature_observability` 返回项失败；gap classification fixture 先以缺少机器常量的 `ImportError` 失败。
- focused：PATCH 后 17/17，新增 wireless missing-CSI relation 保留、masked-zero、inactive relation/tampered validity rejection 和 Task gap source classification；其余 future isolation、wired no-CSI、CPU 四语义、Task–Agent、stable slot 与 NPZ round-trip 保持通过。
- development artifact：3 trajectories、12 samples；machine receipt `passed=true`；semantic deterministic rebuild=true。
- Tensor observation shapes：position `[12,2,10,3]`，Comm CSI `[12,2,74,50]`，static CPU `[12,10]`，Task–Agent endpoint `[12,2,22]`。这些容量只描述当前 development bundle，不是正式研究容量。
- PATCH regressions：Step 4.1 7/7、Step 3.3 8/8、Step 3.2 11/11、Raw causal 4/4，全部通过。
- PATCH deterministic artifact rebuild：两个独立临时目录各生成 11 个文件，逐文件 SHA-256 `DifferenceCount=0`；正式 `manifest.json=cacdee9c...4d440edf`、`tensor.npz=489bf4c2...7a1981df`、`validation_report.json=07361ac5...da6f69af`。JSON 固定 UTF-8/LF，manifest 内 7 个文件 hash 均与 Git 可保存字节一致。
- `python -m compileall -q code/src code/scripts code/tests`：通过；knowledge index write / `--check`：5 个输出、`mismatches=[]`、`passed=true`；`git diff --check`：通过。

## Results

Step 4.1 中 position、wireless CSI、wired relation、CPU static capability、既有 Task demand/progress/elapsed 与 typed Task–Agent gaps 已由 Step 4.2A 贯穿到 Sample/Tensor。Graph readiness 仍为 false，因为 stable stateful Flow 和其他 Raw-insufficient 字段未解决，且 Physical topology policy 未决定。

## Expected vs Actual

与授权范围一致。没有把 wired service outcome 变成 current relation，没有把 CPU capacity 变成 available resource/Comp Action/Outcome，没有用 Future Route target 构造动作前 Host/Exec/Ret，也没有构造 Physical Edge。

## Known Issues

- 当前只有三条短 development trajectory 和 12 个窗口，不代表正式 Dataset 或 feature capacity。
- stable stateful Flow、dynamic available CPU、storage、wired queue/load/utilization 继续 `RAW_INSUFFICIENT`。return size/priority/deadline 仍未进入 Raw/Sample/Tensor，但已有 simulator observer source，不能再称为 simulator 无可靠来源。
- edge/cloud Physical membership、Physical neighborhood 和后续模型字段选择继续等待研究者决定。

## STEP 4.2B follow-up reference

本记录中的 stable stateful Flow、dynamic available CPU、storage 和 wired queue/load/utilization gap 已由 STEP 4.2B 独立 source audit 复核；`transmitted_size` hop 完成后 reset，DAG 不等于 DepData Flow，综合 verdict=`FLOW_CONTRACT_NOT_YET_SUPPORTED`。本记录不把这些 gap 改写为已解决。详见 `docs/implementation_records/STEP_04_2B_REMAINING_RAW_SOURCE_STATEFUL_FLOW_AUDIT.md`。

## Git

最终 commit/hash/push 状态见 Completion Report。

## Next Step

唯一建议：STEP 4.2B — Remaining Raw Source & Stateful Flow Contract Audit；不要自动实现 Graph Builder。
