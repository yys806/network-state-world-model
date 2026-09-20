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

## Reuse

复用三条 Step 3.2 development trajectory、Step 3.1F sample builder 的时间/index/action/target 语义、Step 3.2 Raw continuity/split 和旧字段 normalization、Step 3.3 fixed-shape/action/target arrays。未修改第三方 AirFogSim、旧 artifact、模型或图算子。

## Validation

- TDD red：新增 focused test 先以 `ModuleNotFoundError` 失败。
- focused：14/14，覆盖未来 position/progress 隔离、validation CSI counterfactual、wired valid+no-CSI、outcome independence、CPU capacity/allocation/service 分离、Task progress、Future Route isolation、endpoint triple consistency、late-entry/disappearing stable slot 和 NPZ round-trip。
- development artifact：3 trajectories、12 samples；machine receipt `passed=true`；semantic deterministic rebuild=true。
- Tensor observation shapes：position `[12,2,10,3]`，Comm CSI `[12,2,74,50]`，static CPU `[12,10]`，Task–Agent endpoint `[12,2,22]`。这些容量只描述当前 development bundle，不是正式研究容量。
- regressions：Step 4.1 7/7、Step 3.3 8/8、Step 3.2 11/11、Raw causal 4/4，全部通过。
- deterministic artifact rebuild：两个独立临时目录各生成 11 个文件，逐文件 SHA-256 `DifferenceCount=0`；正式 `manifest.json=c6341b0a...74bb649a`、`tensor.npz=fc36946c...3d38620`、`validation_report.json=d0281b2d...785c42c3`。JSON 固定 UTF-8/LF，manifest 内 7 个文件 hash 均与 Git 可保存字节一致。
- `python -m compileall -q code/src code/scripts code/tests`：通过；knowledge index write / `--check`：5 个输出、`mismatches=[]`、`passed=true`；`git diff --check`：通过。

## Results

Step 4.1 中 position、wireless CSI、wired relation、CPU static capability、既有 Task demand/progress/elapsed 与 typed Task–Agent gaps 已由 Step 4.2A 贯穿到 Sample/Tensor。Graph readiness 仍为 false，因为 stable stateful Flow 和其他 Raw-insufficient 字段未解决，且 Physical topology policy 未决定。

## Expected vs Actual

与授权范围一致。没有把 wired service outcome 变成 current relation，没有把 CPU capacity 变成 available resource/Comp Action/Outcome，没有用 Future Route target 构造动作前 Host/Exec/Ret，也没有构造 Physical Edge。

## Known Issues

- 当前只有三条短 development trajectory 和 12 个窗口，不代表正式 Dataset 或 feature capacity。
- stable stateful Flow、return size/priority/deadline、dynamic available CPU、storage、wired queue/load/utilization 继续 blocked。
- edge/cloud Physical membership、Physical neighborhood 和后续模型字段选择继续等待研究者决定。

## Git

最终 commit/hash/push 状态见 Completion Report。

## Next Step

唯一建议：审阅仍然 Raw-insufficient 的 minimum graph gaps，特别是 stable stateful Flow；不要自动进入 Graph Builder。
