# STEP 5.1A-PATCH — Multi-Horizon Motion Semantics & Stable Slot Alignment Fix

## Patch Goal

在不进入 STEP 5.1B、不改变 STEP 4.4 transition、不重新拟合 normalization 的前提下，修正初版 STEP 5.1A 的 multi-horizon Motion 语义和 Motion entity slot 对齐，并把 CSI current model relation slot alignment 做成机器验收。

## Patch Definition Basis

- 研究者 2026-09-21 明确授权的 `STEP 5.1A-PATCH` 附件。
- STEP 4.4 实际代码 `deterministic_transition()`：Vehicle 下一位置为 `state["position"] + learned["vehicle_motion"][..., :3]`，因此每个 decoder step 的 delta 必须是相邻 rollout state 的 local one-step displacement。
- current Physical model slots 来自 `static.input_entity_index["physical"]`；current Communication model slots 来自 current tensor `comm_*[:, -1, relation_slot, ...]` 并由 STEP 4.3A 原样物化给 STEP 4.4。
- 只读 `D:\shen\OB\科研\PIJWM\05模型训练、Loss与评价.md` 仍保持未修改，SHA-256 `62eebb05e2eeefe9edb0038f12964f59915a7acce03a683d6c7e5c71f85684e9`。

## Patch Initial State

- 起始 `main == origin/main == 7ea9e4bed621e6d9e201ed4fe9175c26ca2c2252`，worktree clean。
- 初版代码在 horizon loop 外固定 `current_entities=history[-1]`，所以 horizon 2+ 得到 `p_(t+k)-p_t`。
- 初版 Motion rows 直接枚举 future frame entities，tensor 再按枚举序号写入，因而依赖 future row order/target index，并在 future-only birth 或 disappearance 时改变槽位。
- 初版 CSI 已按 relation ID/RB ID 找数值，但 receipt 未与 STEP 4.3A/4.4 当前 tensor relation slot、endpoint slot、type index 建立机器绑定。

## Patch Files Involved

- `code/src/pi_jwm/step5_1a_motion_csi_target_contract_v1.py`
- `code/tests/test_step5_1a_motion_csi_target_contract_v1.py`
- `code/scripts/build_step5_1a_motion_csi_target_contract_v1.py`
- `code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921/`
- 本记录、Tracker、authority/process records、AI_CONTEXT 与知识索引。

## Patch Changes and Reuse

- Motion target 改为 `[p_(t+k)-p_(t+k-1), v_(t+k)]`；第一步 reference 为 History current，后续为上一 Future GT frame。Future GT 仍只属于 target/training supervision，不进入 History/Graph/Prior。
- delta 每个坐标分量要求 reference 与 current future 两端都有效；不跨缺失帧。next speed 只依赖 current future speed 和 current-support Vehicle eligibility。
- Motion rows 固定按 `input_entity_index["physical"]` slot 构造并按 `input_slot` 写 tensor。future-only Vehicle 只进 side metadata；disappearing current Vehicle 保留 slot；current non-Vehicle 全 mask=false。
- CSI 每行保存 `relation_slot/relation_id/source_id/target_id/source_input_slot/target_input_slot/relation_type_index/rb_indices`，并与 STEP 4.2A current tensor（即 STEP 4.3A/4.4 read path）逐项核对。
- 继续复用冻结 train-only position/speed/CSI stats；没有 refit，没有读取 Future Target 拟合 stats。

## Patch Validation

- TDD red：旧实现运行新增测试得到 5 failures + 1 error，分别命中 cumulative Motion、missing previous future state、future-order/target-index、future-only Vehicle、disappearance、缺失 CSI relation-slot metadata。
- TDD green：focused suite 19/19 passed，覆盖 local H1/H2、missing previous state、Motion/CSI permutations、future-only/disappearance、input slot mapping、CSI model-slot identity、normalization round-trip、unsupported Return birth、serialization/reload、deterministic rebuild、tamper negative 与 receipt AND。
- 真实 non-locked AirFogSim development artifact：12 samples，horizon 1/2 local-step witness、Motion/CSI slots、current model tensor identity 均为 true；receipt top-level `passed=true`。
- patched digest 与 deterministic rebuild digest 均为 `dc6c5b0b0d957f0e1e57ee09c19d2b54e7b2ecd0bb631a9612a8200078ab579a`。
- Related regression：STEP 4.2A 17/17、STEP 4.3A 15/15、STEP 4.3B 21/21、STEP 4.4 30/30、STEP 5.1A-PATCH 19/19，共 102/102 passed。
- `python -m compileall -q .\code\src .\code\scripts .\code\tests`、knowledge index write/check 与 `git diff --check` 通过；最终 Git 同步在提交后核验。

## Patch Results and Expected vs Actual

实现结果符合目标合同：Motion supervision 与 STEP 4.4 recursive transition 逐步一致，entity 轴与 current model Physical slots 一致，CSI relation 轴与 current model Communication slots 一致。该证据只证明 target contract 与 development artifact，不证明 Posterior/Loss/Metric、训练收敛、预测精度或系统性能。

## Patch Known Issues and Boundary

- Loss、Posterior Target Encoder、KL、Metric、optimizer、training loop 均为 NOT STARTED。
- `training=false`、`gpu=false`、`formal_dataset=false`、`locked_test_accessed=false`；Planner/baseline 未进入。
- `future_return_birth_supported=false` 不变；unsupported/unresolved/fixed-support-blocked 只做 component mask/classification/side metadata，不删除 window。

## Patch Git

- Primary implementation commit：`a1989da971c0caec918e64400b9758cc3bb286f9` / `fix(pi-jwm): align step 5.1a local motion targets`。
- 已推送 `origin/main`；提交后核验 local HEAD 与 `origin/main` 一致。最终文档回执 commit 和 clean worktree 状态由 Completion Report 给出。

## Patch Next Step

仅在本 Patch 完整验证、commit、push 后，建议研究者审阅并决定是否另行授权 **STEP 5.1B — Definition 05 Loss / Posterior / Metric Implementation**。不得自动执行。

---

## Historical initial STEP 5.1A record (superseded by PATCH above)

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
