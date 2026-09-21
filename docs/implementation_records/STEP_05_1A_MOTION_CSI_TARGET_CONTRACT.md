# STEP 5.1A — Motion / CSI Target & Normalization Contract Closure

## Step Goal
在不改变冻结 History/Input、STEP 4.4 transition 或 Definition 05 决策的前提下，补齐 future Motion/CSI target、mask、alignment、normalization、serialization 与真实 non-locked development receipt。

## Definition Basis
用户授权附件明确冻结 Vehicle Motion 为 `[delta_x, delta_y, delta_z, next_speed]`，Future CSI 为 directed wireless relation × per-RB 的 future observation；要求 current-support alignment、train-only stats、raw-rule bridge、additive target namespace 和真实 non-locked AirFogSim development trajectory。只读 Definition 05 未被修改。

## Initial State
初始 HEAD 为 `667e560c30f019e5ed576cdba3ee1caa838b3cdd`，分支为 `main`，工作树干净。STEP 4.2A sample 有 future raw position/velocity，当前没有 future normalized Motion tensor 或 future CSI target；STEP 4.3B stats 是 `encoder_normalization_stats.json`。

## Files Involved
- `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`
- `code/scripts/build_step5_1a_motion_csi_target_contract_v1.py`
- `code/tests/test_step5_1a_motion_csi_target_contract_v1.py`
- `code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921/`
- 本记录、Tracker、AI_CONTEXT、authority/process records 与项目索引。

## Reuse and Non-goals
复用 STEP 4.2A Sample/raw amendment 的实体与 communication relation identity、STEP 4.3B frozen train-only position/speed/CSI stats、STEP 4.4 的 raw Motion transition 语义。未实现 Posterior、Loss、KL、Metric、optimizer、training、GPU、Planner、formal Dataset 或 `locked_test`，未改变 Definition 05 十项决策。

## Changes and Evidence
- 新增 `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`、`code/scripts/build_step5_1a_motion_csi_target_contract_v1.py` 和 focused tests。
- Vehicle Motion 为 delta xyz + next speed；CSI 仅从 future outcome `channel_rows` 读取，并按 current communication support 对齐。
- `python -m unittest discover -s .\code\tests -p 'test_step5_1a_motion_csi_target_contract_v1.py' -v`：12/12 passed。
- build script 生成 12 samples、`manifest.json` 和 receipt `passed=true`，digest `57280ca746aee5fcbe6c065d080ab719f6c79d65f1e884cc288ea2770dcc880f`；独立重建 digest 相同。
- Scope：`formal_dataset=false`、`training=false`、`gpu=false`、`locked_test_accessed=false`。

## Results and Expected vs Actual
预期是把 Raw/Sample future facts → target tensor → stable support alignment → masks → frozen normalization → raw bridge 串成可验证的 additive contract；实际 12 个真实 non-locked development samples、12/12 focused checks、deterministic rebuild、serialization 和 tamper rejection 均闭合。该结果只证明数据合同，不证明模型读取、训练收敛或性能。

## Known Issues and Boundary
这是合同和真实 non-locked development artifact 证据，不是正式 Dataset、训练结果、性能结果或当前模型已读取该 target 的证据。Loss、Posterior、Metric、optimizer、GPU、Planner、`locked_test` 未进入。

## Git
实现主提交：`e815b99efb304aa60974f008695f8372bc0ee985` / `feat(pi-jwm): close step 5.1a motion csi targets`；已推送 `origin/main`，GitHub branch 为 `main`。本段记录的末次同步提交由最终 `git log -1 --oneline` 回执确认。

## Next Step
后续若要实现 Definition 05 Loss/Posterior/Metric，需研究者另行授权 STEP 5.1B；本 Step 停止，不自动推进。
