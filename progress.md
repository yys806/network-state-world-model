# Progress

## 2026-09-19 STEP 3.2 completion

- Implemented batch/split/preprocessing module and builder; reused frozen Step 2.4 collector with explicit seed/output arguments.
- Validation bundle: 3 independent trajectories, 12 windows, dev_train 8 / dev_validation 4; trajectory isolation, train-only mask-aware stats, deterministic rebuild and round-trip passed.
- Focused tests: Step 3.2 5/5, Step 3.1F model-ready 12/12. GPU=false, training=false, locked_test=false.

## 2026-09-19 STEP 3.1

- 已完成最小真实 Model-ready Sample & Tensor Contract：`H=2/L=2`、因果 History、Future Action/Target 对齐、input/target index 分离、四类 action、presence/feature mask、通信 service/task progress 分离。
- STEP 3.1R 真实 artifact 已重建：History `[1,2]`、Action/Target `[2,3]`，17 项机器检查、6 项 focused tests 和 serialize→load equality 通过；旧 Step 3.1 的 10/4 记录已被本修正取代。
- 当前边界：正式 batch/split、模型、loss、planner、训练、GPU、locked_test 均未开始。唯一下一动作是研究者审阅后授权 STEP 3.2。

## 2026-09-19 STEP 3.2-PATCH completion

- 已补齐 provenance：trajectory_id/seed/source path+SHA/split/schema/config lineage、frame/time range、slot duration、3.1F sample contract version。
- 已加入真实 slot time-grid 与 execution start/end 对齐检查，time-gap fixture 在 window construction 前拒绝。
- development batch future-reference audit：12 candidate、12 constructed、0 unresolved window、0 unresolved reference，结果仅 observation。
- normalization metadata 已保存 `m/s`、`m/s^2`、`AirFogSim data-unit`；presence=false 极端 padding 不改变 train stats。

## 2026-09-19 STEP 3.2-PATCH-RECEIPT completion

- 修正 validation receipt 顶层 `passed`：由全部 required isolation checks、deterministic rebuild 和四项 non-locked scope boolean 逻辑 AND 计算。
- 新增 negative fixture：人为设置 `scope.gpu=true` 后，required check 为 false 且整体 acceptance 为 false。
- provenance contract version 改为直接复用冻结 `model_ready_sample_contract_v1.SCHEMA_VERSION`。
- 最终 receipt 重新生成，`passed=true`，STEP 3.2 正式 COMPLETE / FROZEN。
- validation report 改为从最终 bundle/provenance/stats 计算；仍为 formal_dataset=false、training=false、gpu=false、locked_test_accessed=false。

## 2026-08-26 双约束计划重构（本次会话）

- 用户明确给出老师组会原话，已逐字写入 `记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md` 以及记忆更新；原话与解释分开保存。
- 完成当前仓库清单和关键权威文件复核：约 725 个文件，核心 PI-JWM 源码/脚本/测试/记录已核对；第三方、生成物和二进制按 manifest/哈希/元数据边界处理。
- 新增当前唯一执行入口：P0–P10 双约束计划 v2。老师要求线为部分信息、多源数据分工、方法适配理论、场景化调参；PI-JWM 线为理论—实现—数据—指标一致、contract、规则递推、candidate rollout 闭环、locked-test 最后。
- 新增机器可读门矩阵 `记录/双约束门矩阵_20260826.json`，固定 T1–T4/C1–C5、P0–P10 状态和当前 GPU/locked-test 边界。
- 计划明确复用既有 P0–P2、规则 replay 和 non-locked aggregate GPU 证据，不重复已通过工作；P3 方法适配设计包现已完成，当前下一步是 P4 CPU/non-locked 世界模型闭合，之后按 P5 -> P6 -> P7 -> P8 -> P9/P10 顺序推进。
- 本次未启动 GPU、未访问 locked-test、未修改 AirFogSim 第三方源码。

## 2026-08-26 P3 方法适配与多源数据方案

- 已完成通俗版三模块适配说明、限制、预测/决策指标、多源数据角色、数据准入检查和公平调参预注册：`记录/设计/2026-08-26-P3方法理论适配与多源数据方案.md`。
- 已完成对应机器状态：`记录/P3方法适配与多源数据方案_20260826.json`。T3 设计门完成；T2 仍等待真实数据逐源核验，T4 留在 P6 执行。
- 当前状态保持：`formal_performance_claim_ready=false`、`locked_test_accessed=false`、`gpu_authorized_now=false`。下一步只进入 P4，不启动新的 GPU。

## 2026-08-26 P4 机制门收口

- 用户反馈已确认：项目回答必须使用通俗、完整、先结论后证据的中文，并解释术语、剩余工作、阻塞和 GPU/locked-test 边界；该要求已写入 `AGENTS.md`、`记录/本地计划表.md` 和 `记录/PIJWM主文档.md`，作为长期硬约束。

- [x] 复用并验收 h20 formal tensor：`code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_unlocked_20260826`；history=8、horizon=20、54 unlocked seed、14,742 windows、train-only normalization。
- [x] 完成 CPU 1/5/20 horizon mechanism audit：`code/artifacts/audit/pi_jwm_formal_p4_horizon_mechanism_audit_20260826/p4_horizon_mechanism_audit.json`；shape/finite/action sensitivity/target leakage checks pass。
- [x] 修复并验证 P4 unified gate 的 checkpoint-vs-tensor horizon contract；回归测试 `test_run_formal_p4_world_model_gate_v1.py` 覆盖长 tensor + 短 checkpoint 场景。
- [x] 统一 P4 mechanism gate 通过：`code/artifacts/audit/pi_jwm_p4_world_model_gate_h20_20260826/p4_world_model_gate.json`，规则 replay、candidate consistency、1/5/20 mechanism 和 non-locked 边界均通过。
- [ ] 正式精度门未完成：当前 checkpoint horizon=3；h20 checkpoint 的 20 步 state/task/resource/uncertainty 指标尚未获得。不得将机制门结果写成最终 P4 性能。
- 本轮不启动 GPU、不访问 `locked_test`，不进入 P5/P6；`formal_performance_claim_ready=false` 保持。
- [x] 收尾验证通过：P4 gate 回归 3/3，世界模型 8/8，规则层 14/14，loss 9/9，metrics 8/8，window 5/5，tensor builder 7/7；`compileall` 与 `git diff --check` 通过。
- [x] 机器字段复核通过：机制报告和统一 gate 均 `status=passed`；统一 gate `checkpoint_horizons=[3]`，`horizon_gate=true`，`formal_performance_claim_ready=false`，`gpu_execution=false`，`locked_test_accessed=false`。

## 2026-08-25 revised-candidate GPU smoke and formal execution

- Recovered remote smoke archive `pi_jwm_rule_v2_smoke_20260825.tar.gz`; local SHA-256 `5D2DA544051120B18CBCCE1798E49BB35EB7A6A415C6CE169229D1C3C173E87C`, matching the remote archive.
- Smoke manifest audit passed: 18 files, 0 mismatches. Summary confirms `gpu_execution=true`, strict checkpoint reload, train/validation/calibration `2/1/1`, `locked_test_accessed=false`, and `formal_performance_claim_ready=false`.
- Remote independent execution root: `/root/autodl-tmp/pi_jwm_rule_v2_20260825`; rule tensor manifest SHA-256 `8722b2f801e20cb8c26d233c6d7538b21461fc66beb373223814358323075b91`; `locked_test` absent.
- Formal three-seed GPU run is now in progress with the frozen rule-enabled candidate configuration; no locked-test access and no formal performance claim are authorized.
- The first GPU batch completed but was rejected by the multiseed audit because `data_seed` drifted with the training seed; those artifacts remain preserved as failed-protocol evidence and are not used for claims.
- Corrected batch `gpu_formal_rule_v2_fixed_data_20260825` completed with common `data_seed=20260823`; archive SHA-256 is `B044B0546FDAF96A943FACBF7B96400D7CCBDFEBF1A7E08A6BBF38CB6804D37F`.
- Corrected three-seed audit passed: contract identical, manifest mismatch `0`, train/validation/calibration `256/128/128`, `gpu_execution=true`, `locked_test_accessed=false`. Validation link-F1 delta mean `+0.11279246` (min `+0.09827085`), calibration link-F1 delta mean `+0.36119906`; validation node-x MAE delta mean `+0.16214928` remains positive, so this is not a joint-state performance claim.
- Rule-enabled consistency audit passed with no critical mismatches: `deterministic_rule_update_implemented=true`, `candidate_checkpoint_rule_layer_enabled=true`, `rule_layer_input_contract_ready=true`, `no_future_state_endpoint_leakage=true`, and `per_rb_outputs_consumed=false`.
- Phase 11 is complete for non-locked GPU execution. Final method freeze, locked-test evaluation, and formal performance claim remain closed.

## 2026-08-23 rule-layer contract hardening and CPU-to-GPU gate

- Removed label-time endpoint leakage. `task_action_source_node_index` now comes from offload/return/RB/CPU action records; rule-enabled rollout has no task-state fallback.
- Added `flow_task_index` and `slot_seconds`; rebuilt 54 unlocked trajectories and 15,660 windows at `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_unlocked_20260823`.
- Full tensor audit passed: all four action-source missing counts 0, flow-task missing 0, slot `0.1`, manifest mismatch 0, locked-test accessed/materialized false.
- Replaced action-occurrence completion with learned service fraction plus physical conservation, vectorized capped-equal-share CPU service, lifecycle/DAG legality, and recursive explicit-state emission. Logged future CPU allocation is ignored.
- A first full CPU seed exposed full-state double feedback. TDD changed it to masked rule correction only; failed/partial v1 runs are excluded from final evidence.
- Final v2 CPU runs completed for seeds `20260824/25/26`, each 256/128/128 windows, hidden 32, 3 epochs, batch 2, lr `3e-4`, zero-init and residual scale `0.5`.
- Frozen numerical gate passes: validation link-F1 delta mean `+0.11279246`, calibration delta mean `+0.36119906`, node-x MAE ratio `1.10900068`, and all operational deltas are non-positive.
- Consistency audit passes 10/10 checks with no critical mismatch. The revised aggregate candidate is eligible for non-locked GPU execution. No revised GPU run or locked-test access occurred; formal performance claim remains false.
- Verification: formal tests 139/139, teacher tensor 5/5, teacher builder 4/4, compileall and `git diff --check` pass. Full suite ran 1436 tests; residual 7 AirFogSim GBK/third-party environment errors and 1 pre-existing root-plan directory-policy failure are outside this change.

## 2026-08-23 deterministic rule-layer resolution

- Added `code/src/pi_jwm/formal_deterministic_rule_layer_v1.py`: explicit endpoint resolution, train-stat physical/normalized conversion, RB/CPU conservation, flow service update, task lifecycle/DAG transition, and deterministic masks.
- Extended `FormalAirFogSimWindowDataset` with `future_source_node_index` (plus compatibility alias), deterministic target masks, and a machine-readable rule-layer contract. No locked-test data was read or materialized.
- Integrated the rule layer into `FormalDualGraphWorldModel` behind the explicit `deterministic_rule_layer` configuration. Learned service heads provide rate/delivery/CPU outcomes; each step applies rules and feeds rule-updated normalized state into the next rollout step.
- TDD and CPU evidence: rule-layer tests 5/5, model 8/8, formal window 5/5, consistency audit 5/5; real non-locked validation window forward/backward finite; one-window train/reload smoke completed with `gpu_execution=false` and `locked_test_accessed=false`.
- New audit: `code/artifacts/audit/pi_jwm_formal_candidate_consistency_audit_20260823_rule_layer_interface/`. Input contract is `ready`; status remains `blocked` because the prior three GPU checkpoints have `deterministic_rule_layer=false` and are not valid evidence for the revised method.
- Pre-retraining risk review reopened the source-endpoint subgate: label-time `task_node_index` is not a leak-safe source. No multi-seed training has started; Phase 9 now requires explicit action-source tensorization and a new unlocked tensor artifact before any gate run.

## 2026-08-23 key-wise CPU root-cause diagnosis

- Completed four isolated history-key perturbation audits for the selected `coupled_dual_gnn_residual + lr=3e-4` candidate across seeds `20260824/25/26`, validation-only, noise `0/0.05/0.10/0.20`.
- Artifact: `code/artifacts/audit/pi_jwm_formal_tuning_keywise_robustness_cpu_20260823/`; all 12 reports have `sample_count=128`, `evaluation_device=cpu`, `gpu_execution=false`, `locked_test_accessed=false`.
- `node_state` explains node-x sensitivity (`+10.129/+20.972/+42.697` mean MAE deltas); `task_state` explains task-delay sensitivity (`+0.188/+0.423/+0.897`); edge/flow perturbations leave those metrics unchanged to reported precision. Link-F1 remains effectively unchanged.
- CPU root-cause phase is complete. Next phase is a theory-code consistency decision around residual anchoring and task-state encoding; no new GPU run or locked-test access is justified yet.

## 2026-08-23 candidate theory-code consistency audit

- Added and ran `code/scripts/run_formal_candidate_consistency_audit_v1.py` against the three selected checkpoints and current v3 tensor; artifact is `code/artifacts/audit/pi_jwm_formal_candidate_consistency_audit_20260823/`.
- Residual config, task-history conditioning, future-action conditioning, aggregate-baseline boundary, and same-protocol checks pass. Only seed identity differs across configs.
- The audit blocks on `per_step_deterministic_rule_update_missing`: current `formal_dual_graph_world_model_v1.py` transitions latent states and emits explicit state means, but does not invoke a deterministic rule update at each rollout step as required by the theory contract.
- No GPU, training, or locked-test access occurred. Next phase is rule-layer resolution or an explicit theory-boundary revision, followed by tests and a new consistency audit.
- Real validation-window inspection added to the audit: `rule_layer_input_contract.status=blocked` with missing `stats_passed_into_model`, `future_source_endpoint_mapping`, `future_service_outcome`, and `deterministic_target_masks`. No rule implementation was guessed or inserted.

## 2026-08-23

- Read project-local constraints and planning-with-files instructions.
- Confirmed no CUDA device is available and no locked-test access is allowed.
- Confirmed current gate artifact and loss construction path.
- Next: add a tested state-loss override to the CPU/GPU training runner, preserving the default configuration.
- Added and tested `state_mae_weight`; default path is unchanged.
- Started the controlled CPU screen: fixed `data_seed=20260823`, non-locked tensor, learned method `coupled_dual_gnn_residual` only.
- State-weight screen completed: `state_mae=0.5` failed to improve either core metric.
- Residual zero-init screen completed; full-budget seed `20260824` is individually within the frozen gate. Next: two independent zero-init seeds, then one gate audit.
- Residual damping `scale=0.5` completed for seeds `20260824/25/26`; the frozen gate report is `code/artifacts/audit/pi_jwm_formal_cpu_gpu_gate_20260823_zero_init_scale05_3seed/cpu_to_gpu_gate.json` with `gpu_allowed=true`.
- Final hardware check remains blocked because CUDA is unavailable; no GPU process was started and locked-test remains sealed.
- Verification complete: model tests `8/8`, runner tests `7/7`, gate tests `2/2`, compileall and diff check pass; all three selected run manifests/configs verify; CUDA reports unavailable.
- Remote GPU smoke completed on RTX 4090 using the explicit gate-passing configuration; local evidence is in `code/artifacts/experiments/pi_jwm_formal_gpu_smoke_nonlocked_20260823_server_4090/`. No formal long training was started.
- First formal GPU run completed on remote RTX 4090: seed `20260824`, train/validation/calibration `256/128/128`, 3 epochs; local evidence is in `code/artifacts/experiments/pi_jwm_formal_gpu_seed20260824_server_4090/`. Multi-seed GPU training remains pending.

## Controlled GPU tuning screening started (2026-08-23)

- Remote readiness rechecked: RTX 4090, CUDA available, no pre-existing training process, tensor contract present, and local/remote training runner SHA-256 identical (`925d793f...ff1b2e`).
- Started a new isolated remote directory `/root/autodl-tmp/pi_jwm_current_gpu_20260823/formal_tuning_screen_20260823` with six serial jobs: four learned modules per job, two learning rates, three seeds.
- Launcher PID is `1447`; first job is `formal_tune_lr0.0002_seed20260824`. No locked-test path is included in the command.

- All six launcher jobs completed successfully; remote summary checks found `training_run_complete=true`, `gpu_execution=true`, `locked_test_accessed=false`, and `256/128/128` sample counts for every run.
- Recovered archive `code/artifacts/experiments/pi_jwm_formal_tuning_screen_20260823_server_4090.tar.gz` and extracted six run directories locally; the audit produced 24 candidate records and selected `coupled_dual_gnn_residual @ 3e-4` under the frozen gates.
- Final method freeze is still pending; next work is downstream rollout/robustness/consistency closure for the selected screening candidate.

- Downstream CPU audits completed for the selected candidate: horizon rollout retained communication/resource gains but node-x deltas stayed positive; input perturbation caused large node-x/task-delay degradation.
- Next phase is CPU-only root-cause diagnosis for state error and input sensitivity; no new GPU run or locked-test access is justified yet.

## 2026-08-26 created-flow replay closure

- Recomputed flow conservation on the actual v4 rule-contract tensor root used by the rule-enabled GPU checkpoints: 54 unlocked trajectories, 155,875 active flow slots, 12,924 first-active slots, maximum absolute balance error `5.21540641784668e-08`, and zero violations at `1e-5`.
- Fixed the newly created flow path so an unobserved previous slot cannot contribute a denormalized training-mean `delivered_cumulative` or age. The focused deterministic-rule-layer suite had already passed 14/14 after the fix.
- Replayed seeds `20260824/20260825/20260826` on CPU against the v4 root. The fresh report records 64 validation batches and 192/192 observed/expected rule steps per seed, with no flow, RB, CPU, lifecycle, or DAG violations. No GPU or locked-test access occurred; `formal_performance_claim_ready=false` remains false.

## 2026-08-26 candidate-action planner consistency audit

- Started the planned code-only, non-locked mechanism audit. No GPU, training, simulator execution, or locked-test path has been accessed.
- Historical R6 source review confirms direct belief/state-conditioned candidate scoring. Current formal-model review confirms action-conditioned sequence prediction but has not established per-candidate action injection, future objective extraction, or predicted-value-based selection.

- Added the audit gate, script, and three focused tests. The code-only report is `blocked`; it records no GPU start, training, or locked-test access. Authority records now preserve P6 as unimplemented rather than relabeling the direct scorer as a planner.

## 2026-08-26 seven-paper close reading

- Added structured notes for the seven newly indexed 2026 arXiv preprints in `literature/新增7篇精读笔记_20260826.jsonl` and a Chinese synthesis in `literature/新增7篇精读汇总_20260826.md`.
- The strongest directly relevant evidence is methodological: WorldSimProbe provides an action-realization/interaction-response contract; WONDER provides a true candidate-wise world-model rollout and replanning loop; RMWorld provides task-aware uncertainty and paired decision diagnostics.
- RFWM, EMWM, Khora, and AC-MTM are recorded as design inspiration only. No literature result is promoted to PI-JWM implementation, performance, planner, or formal claim.
- This literature phase used local PDFs only; no GPU, locked-test, or new experiment was run. The next authorized work remains the non-locked planner mechanism audit/implementation gate.

## 2026-08-26 file-tree and evidence-layer governance

- Added `记录/文件树与证据分层_20260826.md` as a read-only navigation index. It records the mandatory pre-task read order, canonical evidence entry points, and the rule that process/smoke/candidate/history files cannot be treated as final method evidence.
- Updated `AGENTS.md`, root `README.md`, `记录/README.md`, and `code/artifacts/README.md` so the same process/final distinction is visible from every main entry point.
- Verification scope was documentation-only: no GPU start, no locked-test access, no file moves/deletions, and no changes under `code/reference/AirFogSim/`. The research state remains `formal_performance_claim_ready=false`, `locked_test_accessed=false`, and planner `blocked`.

## 2026-08-27 P4 h20 precision audit

- Corrected the stale record that called h20 precision pending. The three h20 GPU runs for seeds `20260824/20260825/20260826` have already completed; each used 128 validation windows, `gpu_execution=true`, and `locked_test_accessed=false`.
- The h20 audit artifact is `code/artifacts/audit/pi_jwm_p4_h20_precision_audit_20260826/p4_h20_precision_audit.json` (SHA-256 `6908318279AFD880459D6EF3EADBE34EA783C70DA2B4C072AE7631252B2D1B3B`). Its result boundary is non-locked aggregate-baseline only.
- At k=20, learned node-x MAE averages `22.8591 m`; persistence averages `22.1450 m`; the learned model is worse by `0.7140 m` on average and worse on all three seeds. Node-x 95% interval coverage averages only `75.43%`, while task-delay coverage is `96.93%`.
- Result: P4 final precision is `blocked`, not passed. `formal_performance_claim_ready=false` remains unchanged. The sole next action is a CPU-only, one-variable diagnosis of the node-x long-horizon error. No new GPU run, model/tensor/protocol edit, later-phase transition, or `locked_test` access is authorized by this record.

## 2026-08-27 P4 node-type contract diagnosis

- Confirmed the h20 root cause without GPU or `locked_test`: the source uses `V/U/I/C`, while tensorization recognizes only full names. Across all 54 unlocked h20 trajectories, all 2,484 static node-type entries are `-1`; none of 290,076 present-node time slots has a valid node type.
- This makes the coupled model treat every physical node and attached information agent as invalid. The h20 GPU results therefore cannot be used to assess the intended complete dual-graph method, even though the reported position errors themselves are genuinely computed.
- Next action: normalize the four source short codes at the data entry point, add a present-node type contract test, then rebuild the affected unlocked h20 tensor and run CPU acceptance before any GPU decision. No model, protocol, or locked-test change has occurred.

## 2026-08-27 P4 节点类型契约修复（进行中）

- 新增短码回归测试后，旧实现在 `V -> node_kind_index=-1` 处按预期失败。
- 最小修复仅位于 `code/src/pi_jwm/airfogsim_tensor_v2.py`：在原有完整名称识别前，将 `V/U/I/C` 分别归一化为 `vehicle/uav/rsu/cloud`。未知类型与 padding 仍保持 `-1`。
- 验证通过：张量测试 8/8；正式 tensor 构建测试 7/7；collector adapter 测试 7/7；世界模型测试 8/8。未启动 GPU，未访问 `locked_test`。
- 下一步是重建独立的 h20 非锁定 tensor，再做 CPU 验收；不改模型、损失或训练协议。
# 2026-08-27 P4 RB outcome reconstruction repair

- The source bundle stores RB actions and runtime transfer events, but no direct RB observation list or top-level `n_rb`. Reading only the absent list collapsed the rebuilt contract to one RB and erased labels.
- Added a tested reconstruction path in `code/src/pi_jwm/formal_rb_targets_v1.py`: match by `(time, task_id, source, target)`, keep RB IDs from the action, use the event path and `planned_capacity / slot_seconds` for observed rate, and preserve unmatched actions as masked/unobserved. `n_rb` also comes from RB actions.
- Focused tests pass: `test_formal_rb_targets_v1.py` 11/11 and `test_build_formal_airfogsim_tensor_v1.py` 8/8.
- Rebuilt isolated unlocked h20 tensor: `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827`; 54 trajectories, 14,742 windows, `n_rb=50`, 48,447 observed RB labels, all present nodes have valid types, no locked-test directory. `formal_training_ready=false` remains.
- Next action: CPU tensor/window acceptance and checkpoint reload on this repaired tensor. No GPU start and no `locked_test` access.
## 2026-08-27 P4 repaired h20 CPU acceptance

- Re-read the canonical plan and authority records; current gate remains P4.
- Verified repaired tensor readiness and focused regression tests: RB targets 11/11, formal tensor build 8/8, world model 8/8, CPU smoke 2/2.
- Reused the existing repaired CPU acceptance artifact `code/artifacts/experiments/pi_jwm_formal_p4_cpu_h20_repair_acceptance_20260827/`: train/validation/calibration sample counts are 8/4/4, horizon=20, finite outputs, and checkpoint artifacts are present. Its summary correctly keeps `formal_performance_claim_ready=false` and `gpu_execution=false`.
- Full repository discovery is not a P4 green suite: 1457 tests yielded 1 root-file expectation failure and 13 legacy real-AirFogSim/Windows encoding or optional-fixture errors. No locked-test path was accessed.
- Next: independent CPU Go/No-Go audit for the repaired evidence; no GPU launch yet.

## 2026-08-27 P4 repaired h20 Go/No-Go

- Added two independent CPU seeds with the same repaired h20 tensor, `data_seed=20260823`, model, rule layer, hidden size, batch size, learning rate, and one-epoch acceptance budget.
- Independent gate report: `code/artifacts/audit/pi_jwm_p4_h20_repair_cpu_go_no_go_20260827/cpu_to_gpu_gate.json`.
- Three-seed gate result is `gpu_allowed=false`: seed count=3; validation node-x ratio=1.0071 and operational deltas are non-regressive, but validation link-F1 deltas are `+0.0213`, `-0.4492`, `-0.0079`, giving mean `-0.1453`. Calibration mean delta remains positive (`+0.3326`) but does not override validation failure.
- No GPU or locked-test access occurred. `formal_performance_claim_ready=false` remains.
- Next: one CPU-only communication-activity F1 diagnosis; no identical GPU rerun and no frozen-contract change before cause verification.

## 2026-08-27 P4 communication-activity F1 diagnosis

- Reused the three existing repaired h20 CPU checkpoints and their exact sample IDs; no training, GPU, tensor-contract edit, protocol edit, or `locked_test` access occurred.
- Added the read-only diagnostic artifact `code/artifacts/audit/pi_jwm_p4_link_activity_diagnosis_20260827/link_activity_diagnosis.json`.
- All seeds use the same validation population (270 positive and 25,502 negative link-activity labels) and calibration population (86 positive and 27,564 negative labels). The training subset has only 669 positives versus 62,755 negatives.
- Seed `20260827`: validation ROC-AUC `0.9612`, AP `0.4358`; seed `20260829`: ROC-AUC `0.9580`, AP `0.3067`; seed `20260828`: ROC-AUC `0.3560`, AP `0.0107`. The bad seed is already badly ordered before thresholding, so changing the threshold alone cannot repair the gate.
- Thresholds selected from calibration were `0.9`, `0.1`, and `0.7`; this variation follows the seed-dependent score distributions and explains the large F1 swings.
- Result: the small 8/4/4, one-epoch CPU check cannot establish seed-stable link-activity behavior. P4 remains blocked; `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`.
- Next: run the single expanded CPU stability check specified in `task_plan.md`; keep GPU and `locked_test` closed until its independent gate is passed.

## 2026-08-27 P4 expanded CPU stability check

- Completed three expanded CPU runs: `code/artifacts/experiments/pi_jwm_formal_p4_cpu_h20_repair_expanded_seed20260830`, `...seed20260831`, and `...seed20260832`.
- All three used the same repaired non-locked h20 tensor, `data_seed=20260823`, `256/128/128` windows, hidden=32, 3 epochs, batch=2, `lr=3e-4`, deterministic rule layer, zero-init residual heads, and residual scale `0.5`.
- Independent gate report: `code/artifacts/audit/pi_jwm_p4_h20_repair_expanded_cpu_go_no_go_20260827/cpu_to_gpu_gate.json`.
- The expanded gate still blocks GPU: validation link-F1 deltas are `-0.0411`, `-0.1753`, and `+0.0104` (mean `-0.0686`); validation node-x ratio is `1.0849`; RB occupancy error delta is `+0.1132`; calibration link-F1 mean delta is `+0.2404`.
- This larger check shows that the earlier problem was not only a tiny 8-window sample effect. The current candidate does not meet validation F1 non-inferiority and RB non-regression requirements, so there is no GPU authorization.
- No GPU or `locked_test` access occurred. No frozen model, tensor contract, or training protocol was edited.
- Next: pause for human review of the failed P4 gates; no new training or later-phase work until a specific decision is made.

## 2026-08-27 P4 failure diagnosis: communication threshold and RB metric

- Re-read the P4 plan and authority records, then ran a CPU-only, non-locked replay over the exact validation/calibration IDs from seeds `20260830/20260831/20260832`.
- New diagnostic script: `code/scripts/run_formal_p4_failure_diagnosis_v1.py`; report: `code/artifacts/audit/pi_jwm_p4_failure_diagnosis_20260827/p4_failure_diagnosis.json`; SHA-256: `C067195787D43630459764ECA83AC2E527F2A937DAE21D99414CD577515BD9EA`.
- Link activity: validation ROC-AUC is `0.99128`, `0.99292`, and `0.99150`. Fixed-threshold validation F1 peaks at `0.6305`, `0.5524`, and `0.6005`, while calibration-selected thresholds are not consistently transferable. The score ordering is stable; threshold calibration under different class proportions is the remaining diagnostic issue.
- RB: action-derived occupancy exact-edge fraction is `0.9961`, `0.9961`, and `0.9941`; action-derived total MAE is `0.5949` (validation) and `0.7473` (calibration), and the learned RB output is numerically identical to it. Persistence is worse (`1.1777` / `1.1313`).
- Code audit found the current metric compares normalized predicted RB and raw aggregate target RB after multiplying both by `scale`; it omits prediction `+ mean` restoration and does not keep the target in the same physical unit. Therefore the prior `3.2023` RB MAE gate is a units mismatch, not yet a valid model regression result.
- No model, tensor contract, training protocol, GPU, or `locked_test` state changed. P4 remains blocked until the metric and threshold-transfer checks are independently approved and rerun.

## 2026-08-28 P4 metric-repair re-evaluation

- Completed the approved CPU-only metric repair verification with `code/scripts/reevaluate_formal_cpu_runs_v1.py`; metric regression tests passed 9/9, tool tests passed 3/3, and `compileall` passed.
- Reused seeds `20260830/20260831/20260832`, their exact checkpoints and sample IDs, with calibration-only threshold selection. Historical runs were not overwritten.
- Independent report: `code/artifacts/audit/pi_jwm_p4_metric_repair_cpu_go_no_go_20260828/cpu_to_gpu_gate.json`.
- RB is now compared in a consistent physical unit: validation RB occupancy delta=`-0.6477 RB`; the old RB failure is removed.
- Remaining gates still fail: validation link-F1 mean delta=`-0.0686` and validation throughput MAE delta=`+0.7101 Mbps`. Therefore `gpu_allowed=false` and `formal_performance_claim_ready=false` remain.
- Boundary: no GPU was started and `locked_test` was not accessed. P4 remains blocked; P6 and unrelated planner work stay closed.
- Next: obtain a focused decision on the remaining validation link-F1 and throughput gates before any new run.

## 2026-08-28 P4 threshold protocol audit

- Reused the three expanded non-locked CPU checkpoints and exact calibration/validation sample IDs in `code/artifacts/audit/pi_jwm_p4_threshold_protocol_audit_20260828/threshold_protocol_audit.json`.
- Calibration-only threshold selection was enforced. Fixed raw `0.5` validation F1 was `0.6305`, `0.4924`, and `0.6005`; calibration-selected raw thresholds gave `0.5490`, `0.4148`, and `0.6005`.
- This explains part of the old F1 drop as threshold transfer, but not all of it: seed `20260831` remains below persistence and the three-seed mean still does not pass the existing gate. Corrected ordinary-probability `0.5` performs worse and is not adopted.
- CPU/GPU training call sites were rechecked: all production loss calls pass `normalization_stats`; no code path was changed in this audit.
- Boundary remains `gpu_allowed=false`, `formal_performance_claim_ready=false`, `locked_test_accessed=false`; throughput MAE delta `+0.7101 Mbps` is still unresolved.
- Single next action: focused decision on the remaining F1/throughput gate handling before any new training or GPU run.
## 2026-08-28 P4 throughput diagnosis

- Replayed the exact validation/calibration sample IDs from the three expanded CPU runs, one forecast step at a time, against the target and last-step persistence baseline.
- Validation total-throughput MAE deltas were `+0.0356/-0.1543/+2.2492 Mbps` for seeds `20260830/20260831/20260832`; the first two show later-step under-prediction, and the third is over-predicting across the horizon.
- This narrows the remaining failure to rollout magnitude/calibration and seed stability. It does not justify changing the frozen contract or claiming a final method.
- No GPU was started and `locked_test` was not accessed. `gpu_allowed=false` and `formal_performance_claim_ready=false` remain.
- Next: wait for a focused decision before any new CPU experiment or training change.

## 2026-08-28 P4 epoch-stability check started

- Root-cause review found the existing 3-epoch CPU runs were still improving in validation loss at their final epoch, so they cannot distinguish rollout under-training from a model limitation.
- Authorized focused check: only `epochs=8` changes; all frozen data/model/loss/optimizer/gate settings stay fixed, with outputs written to new isolated directories.
- GPU and `locked_test` remain closed until the three-seed independent gate is rerun.

### 2026-08-28 P4 epoch-stability launch correction

- Seed `20260830` finished the intended 8-epoch CPU run. Its summary reports `training_run_complete=true`, `gpu_execution=false`, and `locked_test_accessed=false`; best epoch was 6 and checkpoint reload passed.
- The initial second-seed direct-function invocation used the function default `LEARNED_METHODS` instead of the planned single `coupled_dual_gnn_residual` method. It began an unintended multi-model run and produced a `pooled_gru` checkpoint before being stopped.
- The unintended directory is retained as process/failed evidence and is excluded from all comparisons. No GPU or `locked_test` access occurred.
- Single next action: rerun seed `20260831` with the explicit one-method list in a fresh directory.

### 2026-08-28 P4 epoch-stability CPU Go/No-Go

- All three corrected 8-epoch CPU seeds completed with `coupled_dual_gnn_residual` only; each checkpoint reload was verified.
- Independent gate report: `code/artifacts/audit/pi_jwm_p4_h20_repair_epoch8_cpu_go_no_go_20260828/cpu_to_gpu_gate.json`.
- Result: `gpu_allowed=true`, with mean validation link-F1 delta `+0.0237`, validation node-x MAE ratio `1.0463`, mean calibration link-F1 delta `+0.2856`, and validation operational deltas of throughput `-0.4994 Mbps`, RB occupancy `-0.6477 RB`, and task delay `-1.3496`.
- This is CPU permission evidence for a formal non-locked GPU run, not a final method/performance claim. No GPU was started and `locked_test` was not accessed.
- Single next action: wait for user confirmation before the GPU run.

### 2026-08-28 P4 formal GPU launch attempt

- User confirmed execution. The canonical CPU gate remains `gpu_allowed=true` for the fixed h20 rule-enabled `coupled_dual_gnn_residual` configuration.
- Read-only SSH probes to `connect.nmb1.seetacloud.com:14826` and `:14507` both failed with `Connection refused`; training was not launched and no files were uploaded.
- `locked_test_accessed=false` and `formal_performance_claim_ready=false` remain unchanged.
- Next: wait for the recorded server endpoint to accept SSH, then run only the approved non-locked GPU training.

### 2026-08-28 P4 reachability recheck

- The follow-up read-only port check again returned `14826=False` and `14507=False`.
- Local verification confirms no PI-JWM training process is running and the CPU Go/No-Go artifact still has `gpu_allowed=true` with no failed gates.
- No code, data, protocol, upload, GPU, or `locked_test` state changed. The next action is still the same fixed non-locked GPU run after SSH becomes reachable.
## 2026-08-28 P4 GPU launch and environment diagnosis

- Confirmed the new endpoint is live: RTX 4090, CUDA 12.8/PyTorch 2.8, and about 45 GB free disk; no existing PI-JWM training process.
- Synced the repaired h20 non-locked tensor (114 files) and current source package into a new isolated remote directory. Manifest hash matched local exactly; no `locked_test` directory exists.
- First seed command exited with `ModuleNotFoundError: No module named 'pi_jwm'` because the background command did not export `PYTHONPATH`; GPU remained idle and no checkpoint/result was created.
- Corrective action is limited to remote environment propagation. Training configuration and scope are unchanged.

## 2026-08-28 P4 second GPU seed launch correction

- Seed `20260830` finished on CUDA with `training_run_complete=true`, `gpu_execution=true`, `checkpoint_reload_verified=true`, and `locked_test_accessed=false`.
- The first `20260831` background launch failed immediately because `PYTHONPATH` was not exported; its log contains only the `pi_jwm` import error, with no valid artifact.
- A minimal remote check reproduced the cause and confirmed that the explicit remote `code/src` path fixes the import. The training contract remains unchanged.

## 2026-08-28 P4 second GPU seed completed

- Corrected seed `20260831` completed on the RTX 4090; summary and runtime report `training_run_complete=true`, `gpu_execution=true`, and `checkpoint_reload_verified=true`.
- It used the repaired h20 tensor, `data_seed=20260823`, 8 epochs, `256/128/128` windows, and only `coupled_dual_gnn_residual`; `locked_test_accessed=false`.
- No formal performance claim is opened. The only remaining execution action in this gate is seed `20260832`.

## 2026-08-29 P4 third GPU seed completed

- Corrected seed `20260832` completed on the RTX 4090; summary and runtime report training completion, CUDA execution, and checkpoint reload verification.
- It used the same repaired h20 non-locked tensor and frozen configuration as seeds `20260830/20260831`; `locked_test_accessed=false`.
- The three-seed GPU execution is complete, but this is still non-locked evidence only. Next is local recovery and independent audit.

## 2026-08-29 P4 GPU three-seed recovery and audit

- Recovered all three remote runs into `code/artifacts/experiments/pi_jwm_p4_h20_epoch8_gpu_20260829/`; the downloaded archive hash matches the remote SHA-256.
- Independent audit report: `code/artifacts/audit/pi_jwm_p4_h20_epoch8_gpu_multiseed_audit_20260829/gpu_multiseed_audit.json`; structural audit passed with zero manifest mismatches and fully consistent frozen contracts.
- Aggregate validation deltas: link-F1 mean `+0.0452`, throughput MAE `-0.5328 Mbps`, RB occupancy MAE `-0.6477 RB`, task-delay MAE `-1.3590`; node-x MAE is worse by `+0.8937 m` on average.
- Scope remains non-locked aggregate-baseline evidence; `formal_performance_claim_ready=false` and `locked_test_accessed=false`. Focused audit tests passed `2/2`.

## 2026-08-29 P4 residual 幅度单变量 CPU 验证完成

- 新建隔离 run `code/artifacts/experiments/pi_jwm_formal_p4_cpu_h20_repair_scale1_seed20260830`，只改变 `residual_state_scale=1.0`，其余配置与通过的 scale=0.5 CPU 协议一致。
- 训练完成，checkpoint reload verified；validation link-F1=`0.63765`，node-x MAE=`12.49098`（逐步平均），20 步位置诊断 node-x MAE=`22.9194 m`，persistence=`22.1450 m`，比例约 `1.035`。
- 该结果支持“增大残差幅度可减轻长步位置欠预测”的单变量假设，但仍不能证明三 seed 稳定或 P4 已通过。
- 本轮 CPU-only、non-locked；未启动 GPU、未访问 `locked_test`。
- 下一步：同配置运行 seed `20260831`，完成后检查再运行 `20260832`，然后重跑独立 Go/No-Go。

## 2026-08-29 P4 节点位置误差聚焦处理启动

- 已重新读取当前计划、AGENTS.md、P4 权威记录和文件证据分层规则。
- 已确认本次只处理节点位置 MAE 失败，不扩展到 planner、鲁棒性、论文 baseline 或 P6。
- 已找到可复用的 `run_formal_gpu_rollout_audit_v1.py`，下一步用已有三个 non-locked GPU run 做 CPU 只读逐步审计。
- 边界：不修改冻结 contract，不启动 GPU，不访问 `locked_test`。

## 2026-08-29 P4 节点位置误差 CPU 诊断完成

- 新增只读诊断：`code/scripts/run_formal_p4_position_diagnosis_v1.py`；focused test 通过。
- 复用三个 GPU seed 的 128 个 validation sample IDs，按 1--20 步、x/y/z、节点类型和 `target.node_present` 拆解位置误差；报告：`code/artifacts/audit/pi_jwm_p4_position_diagnosis_20260829/position_diagnosis.json`。
- 三 seed 都显示预测位移远小于真实位移：第 20 步 x 真实位移约 `22.145 m`，预测仅 `0.774/2.027/3.816 m`；主因收敛为 rollout 位移幅度严重欠预测。
- 结果意味着当前模型学到的主要是“位置变化很小”，长 horizon 后自然落后于真实移动节点；P4 精度门不能据此关闭。
- 仍未修改冻结 contract/model/training protocol，未启动 GPU，未访问 `locked_test`；P4 继续 blocked。
- 唯一下一步：对 residual 位移幅度做一个单变量 CPU 验证，未通过前不重训 GPU、不进入 P6。

## 2026-08-29 P4 residual 幅度三 seed CPU Go/No-Go 完成

- `residual_state_scale=1.0` 的三个 CPU seed 已完成，均使用修复后 h20 tensor、`data_seed=20260823`、256/128/128、hidden=32、batch=2、8 epochs、确定性规则层和单一 learned method。
- 独立门报告 `code/artifacts/audit/pi_jwm_p4_h20_repair_scale1_cpu_go_no_go_20260829/cpu_to_gpu_gate.json`：`gpu_allowed=true`，无失败门。
- 关键结果：validation link-F1 delta 均值 `+0.0340`；node-x MAE ratio `1.0851`；calibration link-F1 delta 均值 `+0.3157`；throughput、RB occupancy、task-delay MAE 均值均改善。
- 测试：formal dual-graph 8/8、CPU-GPU gate 2/2、compileall 通过。
- 边界：CPU-only、non-locked；没有 GPU、没有 `locked_test`，`formal_performance_claim_ready=false` 保持。
- 下一步：服务器可用时，原样启动同配置的 non-locked GPU 三 seed并做回收审计；不改变张量契约、不扩大调参网格。

## 2026-08-29 P4 residual 幅度 GPU 三 seed 回收与独立审计

- 已在 RTX 4090 完成并回收 scale=1.0 的 seeds `20260830/20260831/20260832`；三份运行均完成 CUDA 训练和 checkpoint reload，且 `locked_test_accessed=false`。
- 审计报告：`code/artifacts/audit/pi_jwm_p4_h20_scale1_gpu_multiseed_audit_20260829/gpu_multiseed_audit.json`；结构门通过，三 seed 配置一致、manifest mismatch=`0`、calibration-only threshold。
- 性能仍未达标：validation link-F1 delta mean=`+0.0059`，seed `20260831` 为 `-0.0337`；node-x MAE delta mean=`+1.4841 m`。吞吐、RB、任务时延平均优于 persistence。
- 边界：这是 non-locked aggregate-baseline GPU 证据，不是最终方法/性能声明；P4 保持 blocked，`formal_performance_claim_ready=false`，不进入 P6。
- 唯一下一步：基于 GPU 审计和已有位置诊断，先完成节点状态误差的证据化处理决策，不再重复同配置训练。

## 2026-08-29 P4 停滞原因复盘

- 当前状态：GPU 三 seed 已真实完成，但 P4 性能门未通过；必须区分“执行链完成”和“科研目标完成”。
- 流程失误：前置数据/实现契约没有一次性硬阻断；旧 GPU 结果在发现节点类型问题后被降级；RB 指标曾有单位错误；CPU 放行门与最终 GPU 性能门不同。
- 实验失误：在失败原因尚未分类前连续做 3->8 epoch、scale 0.5->1.0 等修正，缺少单一假设和明确停机点。
- 工程失误：远端启动路径、`PYTHONPATH` 和 learned-method 参数没有统一 preflight，产生了无效运行。
- 治理失误：计划和权威记录是追加式长日志，缺少机器可读的唯一当前 gate，增加了重复工作的概率。
- 下一步不是继续开 GPU，而是先只读整理当前 gate，区分必须修复的问题和可接受的边界，再由证据决定是否进行一个最小 CPU 验证。

## 2026-08-29 P4 预测状态接力修复实施

- 根因对应的最小修改已完成：residual state head 和 DAG state head 不再每步回到历史最后帧，而是把上一轮最终预测状态作为下一步基准；规则层修正后的状态也继续接力。
- TDD 证据：新增测试先在旧实现上失败，修复后 `test_formal_dual_graph_world_model_v1.py` 全部 `9/9` 通过；compileall 和 diff 检查通过。
- 新 CPU 隔离目录：`code/artifacts/experiments/pi_jwm_p4_recursive_cpu_20260829/`。seed `20260830` 完成，checkpoint reload verified，使用原 h20 non-locked 数据和固定配置；没有 GPU、没有 `locked_test`。
- seed `20260831` 曾启动并保持正常响应，但单 seed 运行时间过长，本轮停止；目录只含 config、manifest 输入清单和 sample IDs，不含有效 checkpoint/指标。seed `20260832` 未启动。
- 结果边界：修复接口已通过，三 seed CPU 性能门未完成，不能据此放行 GPU 或宣称 P4 通过。

## 2026-08-29 P4 状态接力修复三 seed CPU 复验完成

- `20260830/20260831/20260832` 三个 CPU seed 均完成训练、验证、校准和 checkpoint reload；统一使用修复后的 h20 tensor、`data_seed=20260823`、256/128/128、8 epochs、`residual_state_scale=1.0`，未启动 GPU、未访问 `locked_test`。
- Go/No-Go 报告：`code/artifacts/audit/pi_jwm_p4_recursive_cpu_go_no_go_20260829/cpu_to_gpu_gate.json`。
- 结果：位置比例 `1.1869`、校准 F1 均值增益 `0.3676`、吞吐/RB/任务时延门通过；validation link-F1 差值为 `+0.1032/-0.0941/+0.2013`，单个 seed 超过回退上限，故 GPU 不放行。
- 当前状态：P4 仍 blocked；这证明修复可稳定运行，但没有证明性能门通过，也没有改变 `formal_performance_claim_ready=false`。
- 唯一下一步：只做 link-F1 seed 不稳定性的证据化处理决策，不新增训练或参数搜索。

## 2026-08-30 `_CONTEXT.md` 项目交接开始

- 已完整读取 `project-handoff` 与 `planning-with-files` 技能说明及交接模板，并检查现有三份过程记录。
- 已确认本次仅做证据化交接审阅；暂不执行测试、训练、GPU 任务或 `locked_test` 访问。
- 下一步：逐项核验交接文件、权威记录、Git 状态、相关审计及文件路径，完成后回写交接结论和唯一续接动作。
- 已完成路径和 Git 基线核验：指定路径缺失，唯一对应文件为根目录 `PROJECT_CONTEXT.md`；仓库为 `main@0630515` 且工作区非干净，本次原样保留。
- 已完整通读根目录 `PROJECT_CONTEXT.md` 532 行；已提取当前 P4 gate、GPU/locked-test 边界、link-F1 失败项、逐步位置证据缺口和精确续接点。
- 下一步：用权威记录和机器 JSON/CSV 核验上述交接结论，不运行新训练。
- 已核验权威记录的当前覆盖段；`本地计划表`、`8.12之后推进` 与交接一致，`PIJWM主文档` 的 P4 进度较旧但顶部方法边界仍有效。
- 已核验最新 Go/No-Go JSON 及 SHA-256；交接中的失败门、三 seed 数值、GPU 和 `locked_test` 边界均匹配。
- 已按实际 schema 核验三份 run summary、runtime、config 和 threshold selection；完成/reload/配置/边界均与交接一致。
- 已定向核对 gate 与状态接力模型源码/测试，交接中的失败门定义和递推机制描述与当前代码一致。
- 已确认三 seed 使用完全相同的 sample IDs；comparison.csv 的 link-F1 差值与 gate JSON 一致。
- fresh 验证完成：世界模型 `9/9`、gate `2/2`、`compileall` 和 `git diff --check` 全部退出码 0。
- 本次交接审阅完成；仅更新 root `task_plan.md`、`findings.md`、`progress.md` 过程记录，没有改代码、配置、方法或实验资产。
- 下一步保持单一：只读或 CPU-only 分析 seed `20260831` 的 validation link-F1 回退，形成判断后再决定是否需要向用户申请任何方法或实验变更。

## 2026-08-30 成本路由与 P4 零重复计划启动

- 已读取 `cost-aware-model-routing`、技能编写/TDD、文件规划、计划编写和系统调试规则，并重新核对 PI-JWM 当前权威入口。
- 已完成一次无技能基线压力测试，记录了过度并发和单样本路由过拟合风险。
- 已把“默认成本路由、证据驱动改进、禁止重复实验、保护既有结果”写入当前过程计划。
- 本轮没有训练、GPU、`locked_test`、模型或实验配置变更。
- 下一步：由 Luna 只读建立历史 link-F1 证据去重清单，同时由 Terra 比较当前三 seed 的已有指标；Sol 负责独立复核。
- 已最小增强 `cost-aware-model-routing` 的“Reuse Before Spend”和“Improve Routing From Evidence”规则；结构校验通过。
- RED/GREEN 行为测试完成：无技能时会基于单一样本立即固化路由；使用修改后技能时会拒绝过拟合并要求重复证据与前向测试。
- 已实际委派 Luna/Terra 两个互不写文件的只读子任务，当前没有第三个工作者，也没有在主线程重复同一清单工作。
- Luna 历史去重清单和 Terra 最新三 seed 对比均已返回；Sol 已抽查 threshold grid、k=20 AUPRC 和 raw-score 文件缺失，关键字段一致。
- 阶段 A 完成，阶段 B 开始；确认需要的不是重训，而是对修复后 checkpoint 的首次只读分数诊断。
- 已审查旧诊断脚本：只读取 validation/calibration、CPU checkpoint 和原 sample IDs，报告固定 `gpu_started=false`、`locked_test_accessed=false`；目录 glob 与当前命名不兼容，计划用系统临时 junction 适配，不修改项目代码。
- 第一次诊断启动在进程创建前被策略以 `blocked by policy` 拒绝，原因是同一命令含 `Remove-Item` 清理；没有运行 Python，也没有生成诊断输出。
- 下一次采用项目 `code/artifacts/tmp/` 下可追溯、保留的 junction 适配目录，不做删除，不改变三个源运行目录。
- 只读诊断完成并生成新 audit JSON；三 seed 共六个 split 全部处理，未训练、未使用 GPU、未访问 `locked_test`。
- 初步判断已从“阈值/排序/尺度三者不明”收敛为：阈值迁移参与但不是充分解释，seed `20260831` 的高分假阳性过多是当前 F1 失败的直接机制。
- 下一步只读取现有第 1/5/20 步 metrics 和真实分位数字段，判断问题出现的 horizon；不再运行 checkpoint 推理。
- 已从现有 validation/calibration metrics 提取三 seed 第 1/5/20 步 link activity AUPRC、F1、precision、recall；没有再次运行 checkpoint 推理。
- seed `20260831` validation 第 1/5/20 步 TP/FP/FN=`0/0/364`、`85/42/309`、`353/1125/26`；calibration 第 20 步=`179/1460/24`。
- 已读取诊断 JSON 的真实字段：validation negative q90=`0.2420/0.2823/0.2992`、negative q100 约均为 `1.0`、overall q99=`0.7883/0.9229/0.6859`；未使用不存在的 negative q99。
- 处理判断完成：高置信假阳性主要随长步递推累积，失败 seed 的异常集中在极高分尾部；阈值迁移不是充分解释。底层根因仍未闭合，因此未提出或实施修复。
- 边界：未训练、未改模型/loss/tensor/阈值协议、未启动 GPU、未访问 `locked_test`、未进入 P6。
- 已只读核对模型、loss、训练入口、三份 class weights、training history 和 checkpoint 相关参数；没有执行模型前向或新增实验。
- 确认 link head 使用递推 edge latent，三个 seed 共用 `pos_weight=50` 和相同 loss weights；现有 training history 只有总损失，不能分解 link/horizon 原因。
- checkpoint 参数范数总体一致；仅 seed `20260831` link-head bias 为正。该证据只能形成候选解释，不能关闭底层根因。
- 当前停止新增计算；下一步如执行，只补逐 horizon 负样本 logit 与去 bias 分位数这一项新增证据，不重复现有 27 分钟汇总诊断。

## 2026-08-30 P4 严格收口计划制定

- 已重新读取成本路由、详细计划和系统调试规则，并核对 P4 权威入口、当前 gate、真实 tensor config、诊断脚本和测试模式。
- 已实际路由 Luna (`gpt-5.6-luna`) 完成验收矩阵、Terra (`gpt-5.6-terra`) 完成最小诊断方案；二者只读、未改文件、未训练、未访问 `locked_test`。
- Sol 已纠正子模型的旧 tensor 路径，确认当前 canonical tensor 为 2026-08-27 RB 修复版，manifest hash 与三个 run config 一致。
- 已创建 `记录/实施计划/2026-08-30-P4严格收口实施计划.md`，包含九个门控任务、模型路由、CPU/GPU 边界、训练通知模板、回归保护和最终 P4 验收矩阵。
- 当前已完成计划 Task 1；详细计划草案等待用户确认，确认前不修改代码。确认后先写 v2 bias/latent 诊断失败测试；现阶段不需要训练，也不需要开启 GPU。
- 用户已确认详细计划，并再次要求主线不发散、不增加重复或表面实验。执行进入 Task 2/3 的单一 v2 诊断 TDD；不并行启动其他 P4 问题。

## 2026-08-30 P4 v2 bias/latent 诊断执行记录

- 实际路由：Terra 按冻结规格实现 `run_formal_p4_link_activity_bias_latent_diagnosis_v2.py` 及对应测试；Sol 完成规格审查、代码质量审查、fresh 测试、真实 replay 和原始 JSON 复核。
- TDD 证据：RED 为 6 个预期 `ModuleNotFoundError`；GREEN 定向测试 `6/6`。fresh 相关测试：formal dual-graph v1 `9/9`、v2 `11/11`、metrics `9/9`、CPU/GPU gate `2/2`；`compileall` exit 0。
- 单次 CPU 只读 replay 正常结束，未训练、未启 GPU、未访问 `locked_test`。输出覆盖 3 seeds × 2 splits × 20 horizons，报告 SHA-256=`4fbbd9f41bd4e300200e4e276ea5f3401f63c21b4d58e8962641b50723df0783`。
- v2 raw FP 与 v1 逐 seed/逐 split 完全一致；`raw_logit-pre_bias_logit==bias` 最大误差为 `0`。canonical tensor manifest 未变化。
- 结论：失败 seed 的正 bias 只放大少量 FP，h20 高分尾部主要来自 pre-bias 的 `w·edge_latent` 长步漂移。当前仍不能定位具体递推子路径，因此没有修改模型、loss、threshold 或训练协议。
- 当前状态：P4 blocked，`gpu_allowed=false`、`formal_performance_claim_ready=false`、`locked_test_accessed=false`。唯一下一动作是 direct physical-edge rule-feedback 单变量 forward intervention；仍不需要 GPU。

## 2026-08-30 P4 direct physical-edge rule-feedback 干预记录

- Terra 按冻结规格以 TDD 新增一个 intervention 脚本和测试；Sol 规格审查发现并修复了 locked-test 缺失默认放行及结论边界缺失两项问题。fresh 测试 `8/8`、v2 `6/6`、compileall 和 diff check 通过。
- 单次 CPU 只读干预报告 SHA-256=`6b89049612e97bcb50ad929662423f17630f5a220a6ac78b4bd31ee8f5ea1f2e`；输入 provenance 与 v2 完全一致，未训练、未启 GPU、未访问 `locked_test`。
- 结果：h20 zero-bias FP `1086 -> 0`、pre-bias q99 `4.96 -> -0.16`，h1 不变，判定 `sufficient_to_explain`。这关闭了 direct physical-edge feedback 的因果路径，但不证明关闭反馈是可用修复。
- 当前需要用户确认一个根因专属实现设计。确认前不改模型；若确认，下一步只先写失败测试，再做一个代码变更和 CPU 极小 reload smoke。真实性能训练仍需用户开启 GPU。

## 2026-08-30 P4 Task 6 执行记录

- 实际路由：Terra 完成冻结规格下的 TDD 与最小实现；Luna 只读核验 smoke 接口；两个 Sol reviewer 依次完成规格和代码质量审查，主 Sol 复核全部 diff、测试和产物。
- RED：新增测试在旧实现上准确失败于 h2 hidden，48/48 元素不同，最大绝对差 `0.3548906`；其余 14 项通过。GREEN：八组 fresh 定向回归共 `70/70`。
- 代码只改变 physical-edge correction 的进入位置：不再直接加到 recurrent hidden，而是加入 edge GRU input message；node/flow/task feedback 未改。
- 发现并纠正计划差异：旧 CPU smoke runner不启用 deterministic rule layer，也不 reload checkpoint，因此改为复用现有正式训练模块的 CPU 内部接口。
- micro-smoke：canonical tensor hash 未变；一个方法、`2/1/1`、1 epoch，20.5 秒完成。manifest 18 文件 `0 mismatch`，checkpoint SHA-256=`99b8fdc87f221513007b48f87c2b75d7bc37aac4512447dc328aaa7cba4bd13c`。
- 新 checkpoint 和原 seed `20260831` checkpoint 均 strict load；参数量 `83750`、missing/unexpected keys=`0/0`。输出有限，`gpu_execution=false`、`locked_test_accessed=false`、`formal_performance_claim_ready=false`。
- Task 6 只证明实现、梯度、保存和 reload 接口成立。当前停止 CPU 工作，等待用户开启 GPU 后只跑 sentinel seed `20260831`。

## 2026-08-31 P4 GPU sentinel 运行记录

- 用户提供远端 GPU 后，第一次并行 SSH 探测进入密码等待并被中断；未启动训练。已终止残留 SSH PID `26040`，改为可控交互连接。
- 远端环境：RTX 4090 24GB，`/root/miniconda3/bin/python`，PyTorch `2.8.0+cu128`，CUDA available。默认 shell 无 `python` 命令，因此正式命令显式使用完整解释器路径。
- 新远端根目录：`/root/autodl-tmp/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831`。canonical tensor 从远端已验副本复制，manifest SHA-256=`d5c9edb4200840ad085adaab970855555c19198ed0031234513ea1f078660781`。
- 当前代码和两个训练脚本已上传；远端 compile/import help 通过。模型、GPU runner、protocol 远端 SHA-256 与本地一致。
- seed `20260831` 已作为唯一 sentinel 后台启动，PID=`2028`。配置固定为 `256/128/128`、8 epochs、hidden=32、batch=2、lr=`3e-4`、weight decay=`1e-5`、residual scale=`1.0`、规则层和 zero-init 开启。
- 当前尚未宣称训练完成或性能通过；另外两个 seed 未启动，`locked_test_accessed=false`、`formal_performance_claim_ready=false`。

## 2026-08-31 P4 GPU sentinel 完成与停止

- seed `20260831` 训练耗时 `744.95s`，best epoch=8，peak GPU memory=`690515968` bytes；远端和本地 strict reload 均通过，参数量 `83750`。
- 本地已恢复完整 run：`code/artifacts/experiments/pi_jwm_p4_edge_feedback_gru_input_gpu_20260831/seed_20260831`；manifest 18 文件重算 `0 mismatch`。
- 逐字段审计确认配置、tensor、样本内容、train-only class weights、calibration-only threshold 和 non-locked 边界未漂移。
- validation link-F1 delta=`-0.2177333`，未达到 `-0.05`；node-x 与三个运营指标均通过。机器审计判定 `no_go`。
- 远端确认 `20260830/20260832` 输出目录均不存在，GPU 无训练进程后已退出 SSH。没有重跑、调参或访问 `locked_test`。
- 当前停止 GPU 配置。下一步仅允许对新 checkpoint 做一个有明确新增变量的只读机制判断。

## 2026-08-31 P4 threshold candidate 只读回放

- 先读现有 `threshold_selection.json`、validation metrics 和 `comparison.csv`；确认 validation 缺少 `0.1/0.3/0.5/0.7` 的候选统计，不能仅凭已有汇总作二选一结论。
- 复用既有 link-activity diagnosis，通过保留的本地路径 adapter 和只读 junction 加载 sentinel checkpoint；checkpoint 与原 run SHA-256 均为 `0d203fa46b89e3b4267f387371e4a8c38682810482d30af7c8058f2ea2ac141f`，sample IDs SHA-256 均为 `1f01dade9a0a673fd95bb3d5953a00617c09f124cdc019673765983ea7cde582`。
- 单次 CPU score replay 正常退出，输出 `device=cpu`、`gpu_started=false`、`locked_test_accessed=false`，没有训练或参数更新。
- validation 候选 F1 在阈值 `0.1/0.3/0.5/0.7/0.9` 下为 `0.0721/0.2834/0.3715/0.4573/0.3724`；`0.7` 是候选内最好值，但相对 persistence `0.5901` 仍回退 `-0.1327`。
- calibration replay 在 `0.5/0.7/0.9` 与原阈值报告精确一致；`0.1/0.3` 分别有 `12/3` 个 FP 边界差异，原因未证明，但不改变 calibration 选择 `0.9` 或 validation 候选排序。
- 已达到本轮停止条件：阈值迁移存在，但预注册候选内仍无一超过 persistence。未启动 follow-up seeds，未提出第二个实现变化。

## 2026-08-31 P4 link score/threshold 决策计划制定

- 已读取当前 gate、主文档概率/校准定义、weighted BCE loss、threshold selection、event-specific metrics、probability correction utility 和当前/历史 threshold artifacts。
- 已建立单一决策链：provenance → theory → loss/score/threshold → current evidence → three-state decision → user confirmation。
- Luna/Terra 已真实路由；Sol 发现 Luna 把 validation 顶层默认 `0.5` 误当 link effective threshold，已用原 JSON `thresholds.link_activity=0.9` 和 comparison.csv 纠正。
- 新计划只授权只读审计和 decision memo；当前不需要 GPU，不训练、不访问 `locked_test`、不进入 P6。

## 2026-08-31 P4 link score/threshold decision memo

- 按已确认计划完成 Tasks 1--7；所有当前 sentinel/tensor/audit 哈希与冻结值一致，合同为 seed `20260831`、data seed `20260823`、8/20、256/128/128、non-locked。
- 主文档核验：link activity 是离散事件概率；允许 train-only class weight，但要求概率校准、Brier/ECE 和 calibration-only probability threshold。
- 实现核验：当前 link loss 使用 `pos_weight=50.0` 的 weighted BCE；runner 对 raw sigmoid 直接选阈值；概率反演 utility 只在历史 audit 使用，正式 runner 未接入；当前 metrics 未发现 Brier/ECE。
- 三态判定为 `protocol_theory_mismatch`。当前 sentinel No-Go 和候选内无通过阈值的结果继续有效；不把方法语义缺口当作推翻失败的理由。
- decision memo 只推荐一个设计对象：post-training link event probability calibration boundary。未选择 correction、temperature scaling 或新 grid，也未授权实现。
- Terra 审查无数值/哈希错误；三处 overall/语义文字已最小修正。当前等待用户确认，不需要 GPU。

## 2026-08-31 P4 方案 B 书面设计记录

- 用户明确批准“固定类别权重解析反演 + calibration-only scalar temperature”。Terra 已真实路由并完成三方案、字段、指标和停止门复核，Sol 负责最终数学定义与边界收敛。
- 新增书面规格 `记录/设计/2026-08-31-P4-link事件概率校准边界设计.md`；没有修改代码、模型、loss、数据、阈值候选或性能门。
- 设计通过映射既有五个 raw thresholds 强制保持全部分类决策不变，因此当前 sentinel No-Go 原样保留；方案 B 只修复概率语义与 Brier/ECE/NLL 缺口。
- 当前完全 CPU-only 设计阶段，尚未运行 inference 或测试；GPU 不需要，`locked_test_accessed=false`，P6 未开放。
- Terra 书面规格复核发现并已修正三项实现歧义：raw 坐标 tie-break、`V_cal` mask/reduction、identity-temperature 与 ECE 分箱边界；没有改变方案 B 或增加实验路线。

## 2026-08-31 P4 方案 B 进入实施

- 用户已通过书面规格，实施计划已写入 `记录/实施计划/2026-08-31-P4-link事件概率校准边界实施计划.md`。
- 计划分六个顺序门，任何一步失败即停止，不并行修改 runner/metrics，不用 validation 选择补救路线。
- 当前执行从 calibration math 的 TDD RED 开始；尚未修改实现、未运行 inference、未启 GPU、未访问 `locked_test`。

## 2026-08-31 P4 方案 B Task 1 执行记录

- 初始 baseline `1/1`；首轮 RED 后 GREEN，规格审查发现接口/极端值/覆盖缺口并回退修复；质量审查发现复数、非有限权重和极小权重反演问题并再次回退修复。
- 最终 `test_formal_binary_calibration_v1.py` 为 `16/16`，两阶段复审均通过；root fresh 复跑与 compileall 通过。
- 新边界只有一个可拟合参数 `T`，拟合报告显式记录 CPU/float64/full-batch/LBFGS；旧 correction API 保留且数值边界更严格。
- 当前没有 inference、训练、GPU 或 locked-test 行为；P4 和 sentinel No-Go 未变化。

## 2026-08-31 P4 方案 B Task 2 执行记录

- metrics baseline `9/9`；首轮新增测试准确 RED，后经规格 reviewer 两轮纠正双阈值输入和坐标混用，经质量 reviewer 纠正 float32 threshold tie point。
- 最终接口只有 `link_probability_calibration` 一个新增校准输入；legacy raw threshold 在 accumulator 内唯一映射，避免调用方提供两套阈值。
- root fresh 复跑 metrics `15/15`、calibration `16/16`；没有训练、GPU、locked-test 或 runner 修改。
- throughput/RB 全 mask 的既有计数问题已记录为历史问题，不纳入本次 link 概率单变量实施。

## 2026-09-01 P4 方案 B Task 3 执行记录

- runner baseline `7/7`；正常 learned path 接入后，规格审查阻止空 calibration fallback 和 RULE probability 字段；质量审查阻止单类 calibration 与失败残留污染。
- 最终 runner `10/10`，连同 metrics/calibration 共 `41/41` fresh 通过；root 复跑、隔离 pycache compileall 和 diff check 通过。
- sidecar 记录 train-only weight provenance、温度、raw/mapped thresholds、decision equivalence 和 non-locked flags；manifest 成功后才发布。
- 当前仍未对真实 sentinel 执行新 inference，P4/No-Go 状态不变。

## 2026-09-01 P4 方案 B Task 4 执行记录

- Task 4 严格按 Terra 实现、Sol 规格复审、fresh Terra 质量复审、Sol root fresh 验收顺序执行；审查问题均回到同一实现者做新 RED 后最小修复。
- 最终新增 CPU-only audit core/CLI 与 `11/11` 定向测试；共享 calibration 回归 `16/16`，compileall 与 diff check 通过。
- 实现锁定 canonical tensor/checkpoint/sample IDs/train-only class weight，并在读取前核验 calibration/validation 所选 tensor；strict reload、真实 mask 交集和原子发布路径均有失败测试。
- 当前完成的是 audit 入口，不是真实 sentinel 验收；未运行 GPU、训练或 `locked_test`，`formal_performance_claim_ready=false`。
- 唯一下一步：运行 Task 5 完整定向回归和冻结 sentinel 的一次 CPU-only audit。

## 2026-09-01 P4 方案 B Task 5 停止记录

- 四组定向回归合计 `52/52` 通过，全目录 compileall 通过；实现链本身可执行。
- 冻结真实 sentinel CPU audit 随后运行一次，在 validation NLL 相对 identity-temperature 基线恶化时按机器门抛错停止。
- 精确错误：`ValueError: validation nll regressed after temperature fitting`。这表示 calibration split 拟合出的单温度没有通过 validation 泛化门，不能据此关闭概率语义门。
- 原子发布保护生效：目标 audit 目录不存在、同名前缀 staging 为 0；没有半成品 JSON/manifest。
- 当前仍为 P4 blocked；GPU 未使用，`locked_test` 未访问，另外两个 seed 未启动。下一步必须先由用户确认新的单变量方法边界，不能自动换 calibrator或重跑。

## 2026-09-04 P4低召回诊断执行记录

- 新增CPU-only recall diagnosis runner及9项定向测试；Sol规格审查PASS，质量审查发现并修复float32阈值身份、manifest字段和class-weight provenance问题。
- root复跑诊断`9/9`、世界模型`9/9`、现有概率audit`11/11`，compileall与diff check通过。
- 真实validation只读诊断一次成功，报告与manifest哈希一致，严格复现TP/FP/FN=`1902/557/5855`；没有训练、GPU或`locked_test`访问。
- 用户已同意只对edge GRU更新进行一次旁路核验；已写书面设计和机械支持/不支持门，尚未写干预代码。
- 唯一下一步：用户复核书面设计后进入TDD实现。

## 2026-09-05 P4 edge GRU单次干预执行记录

- 按批准设计新增CPU-only edge GRU旁路runner，并把冻结validation准备逻辑抽为共享私有函数；正式世界模型、loss、数据、阈值和方案B均未修改。
- Terra先获得模块不存在的RED，再完成GREEN；Sol规格审查阻止未硬锁checkpoint/sample IDs、关键集合未锁2791、mask指纹假绿等问题。Terra最后因服务401中断，主Sol只修复二维mask展平这一剩余缺口。
- 主Sol最终复跑四组测试`42/42`和compileall通过；目标目录事前不存在，真实CPU干预只启动一次。
- 输出manifest的size/SHA与磁盘一致；hook calls=`1280/1280`且已移除，输入checkpoint/sample IDs/tensor manifest SHA均与冻结证据一致。
- 机器结果为`not_sufficient_to_explain_dominant_low_recall`：旁路后TP/FP/FN=`0/0/7757`，关键集合找回`0/2791`，五项性能/召回门失败。
- 当前路线已按预注册规则停止；未使用GPU、未访问`locked_test`、未启动其他seed。

## 2026-09-05 P4 edge GRU接口追踪设计记录

- 在旁路假设被否定后，没有自动尝试第二个模块。用户只批准下一步做无干预的接口追踪。
- 已完成书面设计，冻结输入、hook只读语义、GRU精确重算、FN四类转换、三态`0.80`判定和停止门。
- 尚未实现或运行；当前等待用户复核书面规格。

## 2026-09-05 P4 edge GRU接口追踪执行记录

- 只新增`run_formal_p4_edge_gru_interface_trace_v1.py`及对应测试。Terra首轮两次实现仍遗留collector形状/测试覆盖缺口，按成本路由规则升级给Sol，没有重复发布实验。
- 独立Sol规格复审PASS，fresh Sol质量审查修复了canonical SHA可覆盖和hook完整模块身份两个证据链缺口，复审APPROVED。
- 主Sol最终定向回归`75/75`与compileall通过。正式追踪前目标不存在、staging=0，三个冻结SHA与磁盘一致。
- 唯一CPU trace退出码0；报告/manifest SHA-256分别为`87b4fe4b17d06928a1a3583be45f0521b51aa90866d39ce43a95075d868027d9`/`9fccddfbf43bc0832bdeea28dbe8d765963d2126c2016d852b78816d5e1147b4`，manifest的size/SHA与磁盘一致。
- 真实结果：`incoming_readout_below_threshold_dominant`；overall/h1/h20的incoming FN比例分别为`94.94%/100%/98.67%`，downcross比例为`5.06%/0%/1.33%`。
- hook位于`pi_jwm.formal_dual_graph_world_model_v1.FormalDualGraphWorldModel.edge_transition`，命中`1280/1280`且已移除；post/official与GRU重算误差均为0。未训练、未用GPU、未访问`locked_test`。
- 本结果只定位到“本步更新前的读头方向已偏低”，不足以指认单个旧模块。P4继续blocked，下一步只允许先设计无target泄漏的link activity持久性残差候选方法并交用户确认。

## 2026-09-05 P4 link activity持久性残差候选完成方法审查

- Luna完成机械盘点：现有数据已含活动信号，已有persistence基线、单路link head、weighted BCE和方案B接口，但没有正式的previous-activity递推实现。
- Sol完成三方案审查：唯一推荐单路persistence log-odds residual；不采用双路activation/deactivation hazard，不采用无完整概率定义的previous-state bias。
- 已闭合h1合法历史输入、h2-h20纯预测递推、缺失历史mask的train-only先验、`z=u+log(50)`与方案B反演、temperature不进入动力学、旧checkpoint语义拒载等边界。
- 本轮未改模型代码、未训练、未使用GPU、未访问`locked_test`，也未启动另外两个seed；当前仍不是修复结果。
- 单一下一动作：用户确认候选定义后，才形成设计文档和单变量TDD实施计划。

## 2026-09-05 P4 link activity持久性残差CPU实现

- Terra完成机械实现，Sol两轮审查发现并闭合`u/z`坐标、保留残差状态头、config落盘顺序、checkpoint强语义校验和summary证据字段。
- 最终定向与既有保护回归`88/88`通过，compileall与diff check通过。
- 唯一真实CPU micro-smoke产物：`code/artifacts/experiments/pi_jwm_p4_link_persistence_residual_cpu_micro_20260905/`；使用canonical h20 tensor、`2/1/1`、1 epoch、hidden=4、规则层和零初始化残差状态头。
- micro证据：内部方法身份一致，train正/负=`260/8736`，先验=`0.028901734104046242`，pos_weight=`33.6`且与train-only报告一致；h1/h20有限，规则物理边反馈投影调用19次，strict reload缺/多键=`0/0`，manifest 19项零不一致。
- GPU未使用，`locked_test_accessed=false`，性能与方案B概率双门尚未运行。下一动作仅为经用户确认的sentinel seed `20260831`。

## 2026-09-05 GPU sentinel启动阻塞

- 对已提供端点`connect.nmb1.seetacloud.com:14826`只读探测一次，返回`Connection refused`；无文件上传、无训练进程、无远端产物。
- 当前阻塞是GPU服务器不可达，不是candidate性能失败。唯一下一动作是服务器恢复后运行同一seed `20260831`冻结sentinel。

## 2026-09-05 全仓阅读与现状评估

- 主代理阅读当前交接/权威记录并核验P4实现与产物；Luna只读清点目录，主代理复核源码/脚本/测试Python文件数为128/228/238，文献PDF及索引均85项。部分历史artifact枚举返回os error 5，未声称全仓逐文件验收。
- fresh：micro manifest 19项大小/SHA零不一致，tensor manifest SHA与交接一致，micro checkpoint严格加载成功；模型契约5项测试通过。未重复训练或完整88项历史回归。
- fresh最小复现：概率审计传入新方法后立即报 `the frozen sentinel method is coupled_dual_gnn_residual`，未读数据、未创建输出；同时核对旧0.9阈值锁定和runner提前温度拟合顺序。
- 评估报告：`记录/研究进展/2026-09-05-项目现状与主线推进评估.md`；新增阻塞是验收入口与新候选不匹配。GPU可达性本轮未重查；无GPU/locked_test执行。下一建议只有验收入口闭合，未改实验实现。

## 2026-09-05 P4 新候选验收入口实现与GPU阻塞

- 新方法`link_activity_persistence_residual_v1`现可进入独立概率审计；审计按方法读取checkpoint、调用对应模型、校验threshold-selection和run manifest，并从calibration选定raw threshold。旧方法仍固定0.9。
- TDD RED复现旧方法拒绝错误；GREEN后概率审计`13/13`、持久性模型`5/5`、持久性runner`8/8`、formal模型`9/9`通过，compileall和diff check通过。
- runner生成的训练manifest实际已绑定新方法threshold-selection文件；本轮未修改训练入口。未启动训练，未访问locked_test。
- GPU TCP检查`connect.nmb1.seetacloud.com:14826`失败，补查历史端口`14507`也失败；未上传或创建远端产物，这仍是当前外部阻塞。
- 下一步只有GPU恢复后按冻结配置运行seed`20260831`，先性能门后概率门。

## 2026-09-05 P4 GPU sentinel 启动参数偏差

- 远端首次启动已连接并进入数据准备，但命令漏传`--train-limit 256 --evaluation-limit 128`，实际不是冻结协议；PID`1771`已停止。
- 该运行只生成了未发布staging中的`config.json`、`sample_ids.json`和`class_weights.json`，没有正式metrics/checkpoint/manifest，全部不纳入性能判断。
- GPU可用性已确认（RTX 4090、CUDA可用），`locked_test_accessed=false`保持；下一步只在新目录正确重启同一sentinel。

## 2026-09-05 P4 persistence residual GPU sentinel No-Go

- 正确冻结GPU sentinel已完成并归档；样本`256/128/128`、strict reload、manifest和`locked_test_accessed=false`均已核验。
- 性能门失败：candidate validation link-F1=`0.4801978`，persistence=`0.5900904`，delta=`-0.1098926`，超过允许回退`-0.05`；node-x ratio约`1.2163`及保护项通过不能抵消失败。
- h1到h20 link-F1从`0.8065`降至`0.2989`，记录为长期递推漂移风险；不开放事后调参。
- 按停止规则未运行概率门、未启动follow-up seeds，P4仍blocked，`formal_performance_claim_ready=false`。
- 单一下一动作：等待新的单变量设计确认。

## 2026-09-05 完整 RSSM 论文对照与实现

- 读取并对照 PlaNet、Dreamer、Graph Network-based Simulators、TD-MPC2、Rollout-Decoded Reconstruction、latent ensemble 和 UTG 的原始资料，形成论文到 PI-JWM 的机制映射与边界记录：`记录/研究进展/2026-09-05-完整RSSM论文对照与实现审计.md`。
- 审计结论：旧 `graph_rssm_v1` 是可运行候选，但 context prior/posterior 同源且没有逐步观测 posterior、balanced KL 和多步 latent 监督，不能称完整 RSSM。
- 新增 `complete_graph_rssm_v1`，训练路径逐步读取 target observation 形成 posterior，预测路径只使用 history/action/prior；新增逐步 balanced KL 与自由 rollout 一致性项。
- 新增契约测试 `4/4` 通过；旧 R4 RSSM 测试 `4/4` 通过；compileall 通过。测试只证明接口完整和无 target 泄漏，不证明性能提升。
- 独立 R4 CPU preflight 已接入新候选并通过真实 canonical train/validation/calibration 的 h1/h5/h20 共 9 个窗口；训练步、梯度、有限性、target 泄漏、动作条件和 strict checkpoint roundtrip 均通过，manifest 已生成。
- 该 artifact 仍是执行/完整性证据，不是性能提升证据；未启动 GPU、未访问 `locked_test`，P4 继续 blocked，`formal_performance_claim_ready=false`。
- 单一下一动作：完成新候选理论—代码—数据—指标最终审查，确认后才决定是否进入单 seed sentinel。

## 2026-09-05 完整 RSSM teacher reconstruction 补齐

- 理论—代码审查发现原候选 posterior 只进入 KL，缺少直接重构训练信号。
- 已实现训练期 posterior teacher 显式状态/分类输出，并在 R4 objective 中使用；部署/验证输出和 prior 参数仍 target-independent。
- 定向测试 `5/5`、R4 preflight 回归 `3/3`、canonical CPU preflight `9/9` 通过，未访问 `locked_test`。
- 当前仍无真实性能提升证据，P4 blocked；单一下一动作是 GPU 可用后运行 seed `20260831` 冻结 sentinel。

## 2026-09-05 完整 RSSM prior/teacher 双重重构收口

- 复核发现不能用 teacher 重构替代自由 prior 重构；R4 objective 已改为保留 prior 重构，并增加权重 `0.5` 的 posterior teacher 重构。
- 定向测试 `5/5`、CPU preflight 回归 `3/3`、GPU screening 回归 `6/6`、compileall 和 diff check 通过。
- 最新 canonical 产物：`code/artifacts/preflight/pi_jwm_complete_rssm_cpu_preflight_20260905_teacher_reconstruction_v2/`，9 个窗口全部通过。
- 该结果仍只证明方法完整性和可执行性，不证明性能提升；GPU 未启动、`locked_test_accessed=false`，下一步需要 GPU sentinel。

## 2026-09-05 完整 RSSM R4 GPU 单候选筛选

- RTX 4090 完成`complete_graph_rssm_v1`单候选 GPU 筛选，27 epoch 后按 patience=5 早停；最佳 validation protocol score=`4.4865667891`。
- 本地结果目录`code/artifacts/experiments/pi_jwm_complete_rssm_r4_gpu_screening_20260905/`的 manifest 13/13大小与SHA-256一致；checkpoint复载分数差=`4.13e-10`；未访问`locked_test`。
- 筛选只证明 GPU 可运行、训练/复载稳定；候选数=1，`r4_gpu_screening_complete=false`，没有 winner，不构成P4性能改善或方法定版证据。

## 2026-09-06 正式 P4 完整 RSSM GPU sentinel

- 完整 RSSM 已进入正式双图、损失、runner、checkpoint 和指标路径；最终版本 `formal_complete_rssm_v1_1` 同时覆盖连续状态与分类事件，并严格记录 residual-head 零初始化语义。
- 首次/第二次 GPU 运行分别因“事件 decoder 未接入 latent”和“RSSM 连续 residual decoder 未遵守零初始化合同”而作废，只保留过程追溯；没有用其数值形成正式结论。
- 最终 v3 run 位于 `code/artifacts/experiments/pi_jwm_p4_complete_rssm_gpu_20260905_seed_20260831_v3/`；checkpoint strict reload、19 项 manifest 和冻结 sample IDs 均通过。
- sentinel 仅失败 node-x 聚合门：ratio=`1.2877405 > 1.25`。link-F1 delta=`-0.0411296` 在允许范围内，throughput/RB/task-delay delta=`-0.33246/-0.63831/-1.36127` 均通过。
- 审计：`code/artifacts/audit/pi_jwm_p4_complete_rssm_sentinel_20260905_v3/sentinel_audit.json`，SHA-256=`2a64334caebde5018a9f69fdd22b3072d3ac309c0ec00ff8f198e43b8c268bf0`。
- P4 仍 blocked；其他 seed、独立概率门和 `locked_test` 均未运行。下一步只做现有 checkpoint 的 CPU node-x base/correction 分解。

## 2026-09-06 node-x base/correction 只读分解

- 最终 v3 checkpoint 在固定 validation 样本上的分解严格满足 `full = base + correction`，checkpoint strict reload 和 sample IDs 复用均通过。
- base node-x MAE=`13.71936 m`，persistence=`11.85200 m`，ratio=`1.15756`；完整 RSSM 输出 MAE=`15.26231 m`，ratio=`1.28774`。RSSM correction 是把原本通过的 base 推过 `1.25` 门槛的充分因素。
- correction 绝对值平均=`3.04245 m`，改善比例仅=`0.23788`；这说明当前 latent 连续修正多数方向不可靠，不能通过增加 seed 或训练预算解释掉。
- 本次仅 CPU 诊断，无训练、无 GPU、无 `locked_test`。候选设计已写入 `记录/研究进展/2026-09-06-P4-node-x-RSSM修正诊断与单变量候选.md`；P4 保持 blocked，下一步只评审训练期 node-x residual 非劣化约束。

## 2026-09-06 node-x residual 非劣化约束 CPU 与协议门

- 已按批准范围新增 `complete_rssm_node_x_safe_dual_graph_v1`，旧完整 RSSM 方法及 v3 失败证据保持不变。唯一新变量是训练期 `node_x_residual_non_degradation=1.0`。
- 新损失从 `full - correction` 得到 detached base，只对 harmful correction 产生梯度；隔离审计中 correction gradient sum=`1.94717`、base gradient sum=`0.0`。
- canonical h20 CPU micro run 已训练并 strict reload；一致性审计全部通过，状态=`ready_for_protocol_freeze`，不含性能提升结论。
- 冻结协议及独立协议门已通过，run contract 和 v3 完全相同，五个性能门保持不变；协议状态=`ready_when_gpu_available`。
- GPU 未启动，`locked_test_accessed=false`，P4 仍 blocked。下一步只在 GPU 恢复后运行一次冻结 seed `20260831` sentinel。

## 2026-09-06 node-x residual 非劣化约束 GPU No-Go

- 冻结 GPU sentinel 已完成并回传：训练约 `859.98 s`，最佳 epoch=`8`，checkpoint strict reload、同一 sample IDs 和 manifest `19/19` 通过。
- node-x ratio=`1.52058`，超过 `1.25`，因此正式状态=`no_go`；其余四项保护门通过不能抵消失败。
- 只读分解显示 correction 平均幅度从旧 v3 的 `3.04245 m` 降至 `0.07621 m`，但 base ratio 从 `1.15756` 升至 `1.51660`。相同初始化下，新增损失改变了联合训练结果。
- 已停止该候选，不调权重、不补跑、不运行其他 seed/概率门/`locked_test`。P4 仍 blocked；下一步只评审是否冻结并复用旧 v3 base。

## 2026-09-06 P4 长期停滞系统性复盘

- 已把八月中旬以来的失败按数据/实现合同、模型机制、训练与选择协议、最终性能门重新归类。结论是长期停滞来自多层问题叠加，以及按最近失败指标串行打补丁的推进方式，不是一个尚未找到的简单超参数。
- 现有正式 tensor 共 `14742` 个 unlocked 窗口，sentinel 只训练 `256` 个；完整 RSSM 两条正式曲线到第 8 轮仍改善。当前短预算可以检查执行稳定性，但尚无证据说明它足以否决复杂模型。
- 当前 checkpoint 用 aggregate validation loss 选择，而最终用五个独立门验收；位置输入缺少未来机动意图，链路正例约 `0.918%`；这些均可能使局部修复在另一指标上回退。
- 未运行训练或推理，GPU 不需要，`locked_test` 未访问。下一步只做 CPU-only 信息充分性与目标冲突审计，不直接训练冻结-base 候选。

## 2026-09-06 P4 相关文献与图结构 RSSM 对照

- 已完成面向当前失败机制的文献对照。G-RSSM、R-SSM、Graph Dreamer 和 GNS 的共同点是保留逐节点/逐对象状态并通过关系传播更新；无线 latent dynamics 与 Trajectron++ 进一步说明速度、方向或计划条件对长时预测的重要性。
- 正式代码复核发现：当前完整 RSSM 的训练语义成立，但物理节点、边和任务被池化成一个全局上下文，随后同一 correction 被广播回同类实体。因此其准确边界是 global aggregate RSSM adapter，不是 entity-aligned Graph-RSSM。
- 该结构事实可以解释当前随机分支无法分别修正不同节点和边，但尚不能证明改成实体级 latent 后一定过门。已有 prior/posterior、KL、overshooting、teacher 和双图规则层应继续复用。
- 详细报告与机器审计已落盘；本轮没有新训练或代码改动，GPU 不需要，`locked_test_accessed=false`，P4 仍 blocked。
- 下一步只做已有 checkpoint 上的 CPU 实体级可表达性审计；审计前不再增加 loss、输入字段、训练预算或新架构。

## 2026-09-06 P4 实体级双图 RSSM CPU 收口

- 完成第一性原理审计、因果运动 tensor、实体级双图 RSSM、两阶段训练、P4 gate-aware checkpoint selector、CPU consistency audit、GPU batch probe 入口和冻结 GPU runner。
- 第一性原理审计复用 v3 checkpoint 与 128 个固定 validation 样本：历史 speed/acceleration 非零计数均为 0，但从历史位置得到 174338 个移动节点对；旧 link correction 步内最大范围仅 `2.384e-07`；node/operations/task/link/KL 平均梯度范数约为 `28.96/8.95/1.94/1.06/0.645`，并存在负冲突。
- 新 tensor 保持 9828/3276/1638 的 train/validation/calibration 窗口；只在 history 暴露 `node_motion_state/mask`，speed 非零 175467、acceleration 非零 174075，future motion 未进入模型。
- 新方法按 node/physical-edge/flow/task 分别维护 h/z；deterministic base 先训练再冻结，RSSM 独立训练。失败的 node-x 非劣化损失未进入新候选。
- 真实 CPU run `pi_jwm_p4_entity_rssm_cpu_micro_20260906_v2` 与 audit `pi_jwm_p4_entity_rssm_cpu_consistency_20260906_v2` 通过 strict reload、manifest、prior target invariance、posterior locality、四类动作敏感性、mask、h20 和冻结梯度检查。
- frozen protocol v2 固定 base 20 epoch + RSSM 最多 40 epoch，总预算 60；RSSM 前 20 epoch 不早停、patience=10，全部 unlocked 数据，按 P4 gate-aware 字典序选 checkpoint。
- 本地无 CUDA；远端 23874 返回 `Connection refused`。GPU 未运行，`locked_test_accessed=false`，P4/P6 状态不变。

## 2026-09-06 P4 实体级 RSSM GPU 执行门通过并启动正式 seed

- 新远端 RTX 4090 已核验可用；上传的 v5 tensor、CPU consistency audit 与 frozen protocol 均按 manifest 逐文件核验，无差异。
- GPU batch probe 在不创建 optimizer、不执行 optimizer step 的条件下完成。batch `8/4/2/1` 均无 OOM 且 loss/gradient 有限；冻结选择 batch `8`，峰值约 `4.12 GB`（总显存的 `16.32%`）。
- GPU execution sentinel 使用 seed `20260831`、`256/128/128`、base 1 epoch + RSSM 1 epoch。两阶段 mean validation loss 为 `0.77888656` 与 `0.76041028`；该数值只证明运行正常，不承担 P4 性能判定。
- sentinel checkpoint 记录正确 method/model/latent identity，runner strict reload 为 true，base 在 RSSM 阶段冻结，峰值显存约 `4.14 GB`，19 项 manifest 全部一致。
- 首个正式 seed `20260831` 已在远端启动，配置保持冻结：全量 unlocked 数据、batch 8、hidden 32、AdamW `3e-4/1e-5`、base 20 + RSSM 最多 40 epoch、gate-aware checkpoint 选择。
- 当前仍缺少首 seed 收敛结果和正式数值门结论；另外两个 seed 未启动。P4 保持 blocked，`locked_test_accessed=false`、`formal_performance_claim_ready=false`、P6 未开放。
- 单一下一动作：读取并独立核验 seed `20260831` 完成产物；只有全门通过才允许继续 `20260830/20260832`。

## 2026-09-06 正式 seed 全量训练进展

- 远端正式 seed `20260831` 仍在运行，隐藏 staging 已产生 `base_epoch_001.pt`、`base_epoch_002.pt`、`base_epoch_003.pt`。
- 当前 base 预训练进度为 `3/20`；前 3 个 epoch 每个约 26 分钟。全量数据计算耗时较长，但进程 CPU/GPU 均持续活跃，没有异常退出或 OOM。
- RSSM 阶段尚未开始，因此尚无 gate-aware 性能选择结果；P4 仍 blocked，另外两个 seed 未运行，`locked_test_accessed=false`。
- 单一下一动作：持续监控 staging 的新增 epoch checkpoint，待首 seed 完成后再做原始产物核验。

## 2026-09-06 正式训练中间证据保全

- 在不影响远端 PID `2307` 的前提下，建立只读运行证据包 [live evidence](D:\shen\PKU\PIJWM\code\artifacts\experiments\pi_jwm_p4_entity_rssm_gpu_formal_seed_20260831_live_evidence_20260906)。
- 快照记录了完整启动命令、正式冻结配置、tensor manifest、进程/GPU 状态、远端 staging 路径、已完成文件列表及逐文件 SHA-256。
- 当前已回传 base epoch `001–005` checkpoint 和运行元数据；证据包自身 manifest 已核验，`process_untouched=true`、`locked_test_accessed=false`。
- 后续新增 epoch 将追加独立快照并保留旧副本，确保失败时可以直接按时间和 hash 重建完整训练历程。
- 训练期间又完成 base epoch `006`；对应 checkpoint 和第二份只读快照已追加到同一证据包，旧快照未覆盖，远端进程未受影响。
- 训练随后完成 base epoch `007`；第三份只读快照和 checkpoint 已追加，进程仍正常。
- 训练随后完成 base epoch `008`；第四份只读快照和 checkpoint 已追加，进程仍正常。

## 2026-09-07 正式训练阶段切换

- seed `20260831` 的 deterministic base 已完成全部 20 epoch，正式进入冻结 base 的 entity RSSM 训练；当前 RSSM epoch `2/40`。
- 阶段切换前后的 checkpoint 已追加到 live evidence，进程/GPU 正常，无 OOM 或异常退出。
- 当前尚未完成 RSSM 最小 20 epoch，也尚无 gate-aware 正式性能结论；P4 仍 blocked，另外两个 seed 未启动，`locked_test_accessed=false`。

## 2026-09-07 首 seed 中间性能趋势

- RSSM epoch 7–14 的 checkpoint 内置 P4 gate 连续 8 次通过全部单 seed 数值门；验证使用完整 unlocked validation/calibration，`evaluation_limit=null`。
- epoch 14 相对 persistence：validation link-F1 `+0.44123`、calibration link-F1 `+0.85695`；node-x 总体/h5/h10/h20 ratio=`0.75057/0.75523/0.74395/0.75849`；throughput/RB/task-delay ratio=`0.93588/0.47263/0.01723`。
- 这明显优于旧 global complete RSSM 的 validation link-F1 delta `-0.04113` 和 node-x ratio `1.28774`，也优于失败 node-x-safe 候选的 node-x ratio `1.52058`。
- 当前结论仅为“首 seed 中间结果强烈正向”；最小 epoch、最终产物、机制门和跨 seed 泛化尚未完成，P4 继续 blocked，`locked_test_accessed=false`。
## 2026-09-07 组会 PPT 大纲完成

- 新增 `meeting/2026-09-09-PI-JWM组会汇报PPT大纲.md`，形成22页主汇报和6页备份页的制作级大纲。
- 大纲完整覆盖8月12日以来的数据、规则、模型、指标、诊断、文献、实体级RSSM和正式训练证据，并将重复实验压缩为可解释的三条失败链。
- 当前结果页记录实时核验的seed `20260831` RSSM epoch 17中间门；明确三 seed、独立审计和最终P4关闭仍未完成，`locked_test`未访问。
## 2026-09-07 组会PPT大纲按汇报需求压缩

- `meeting/2026-09-09-PI-JWM组会汇报PPT大纲.md` 已从22页加备份页压缩为10页主汇报。
- 工作总览直接对应老师提出的信息精简、多源分工、方法适配理由和公平调参四点；随后按新数据、双图编码、世界模型和完整决策链展开。
- 世界模型部分已将原模型问题、文献启发、实体级结构、机制验证、两阶段训练、旧方案对比和首seed结果组织为连续三页；组会页面不再使用内部阶段编号。

## 2026-09-07 组会PPT追加完成

- 已在 `meeting/PI-JWM_组会汇报.pptx` 后追加10页原生PowerPoint页面，页码为206–215；封面复用第204页，正文复用第205页的页标题、徽标、分隔线和层级样式。
- 内容完整覆盖本月三条主线：最小可靠数据合同与正式因果窗口、严格物理—信息耦合双图、实体对齐动作条件双图RSSM，以及世界模型完成后的候选动作滚动规划闭环。
- 第215页使用远端 checkpoint `entity_aligned_dual_graph_rssm_v1__epoch_022.pt` 的内置门控证据：validation/calibration link-F1 delta=`+0.44198/+0.85982`，node-x总体/h5/h10/h20 ratio=`0.75223/0.76016/0.74626/0.75840`，throughput/RB/task-delay ratio=`0.93793/0.47776/0.01543`，state NLL=`-3.10873`；这是单seed中间结果。
- 追加稿通过PowerPoint原生重开和10页1920×1080渲染；新增页最小正文18磅，中文楷体、英文Times New Roman，未发现字体或页面边界违规。
- 第1–203页的页面XML和关系文件共406项与追加前备份哈希一致；`locked_test`未访问，远端训练进程未被修改或中断。

## 2026-09-07 组会PPT按原生层级重做

- 用户指出首版追加页的卡片和框图削弱了原模板的大标题、小标题和正文层级，并纠正双图与RSSM应为模块一、模块二，而不是方法一、方法二。
- 第207–215页已全部重做为第205页同款纵向原生层级：每页1个红色方框大标题、2个蓝色菱形小标题、8条浅蓝箭头正文，蓝色引导词与黑色说明保留在同一原生段落中。
- 双图页面标题现为“模块一：严格物理—信息耦合双图编码/双图编码与逐步状态更新”；世界模型页面标题现为“模块二：原有世界模型的问题分析/实体对齐双图RSSM/机制验证与可靠训练”。
- 重做后第1–205页的410个页面XML/关系文件与原始205页模板逐字节一致；PowerPoint原生重开、10页渲染、18磅最小字号、楷体/Times New Roman、层级项目符号和文字溢出检查全部通过。
- 正式文件SHA-256=`f33dac592e5eaca480dd6fac568adfbec905e40aeff301a6b0cfc98645eb8898`；训练结果仍严格标注为seed `20260831` epoch 22单seed中间证据，`locked_test`未访问。

## 2026-09-07 组会PPT层级与图表折中完成

- 正文页继续保留模板原生的页面标题、大标题、小标题和正文层级，同时将图表限定在层级文字下方，避免图形取代叙事结构。
- 数据两页恢复字段合同、动作规则、正式规模、轨迹划分和8步历史—20步预测图；模块一恢复严格双图与逐步状态更新图；模块二恢复旧global latent瓶颈、实体级RSSM和两阶段训练图。
- 第215页使用原生表格展示epoch 22的link-F1、node-x、运营指标和state NLL，并在页底保留“合法候选—模型推演—代价/风险—执行首动作—观测重规划”链路。
- 正式文件SHA-256=`f387e7f41553872a9609910a6bc42a4f235a5a3bed9305d075c671bd43e2773d`；第1–205页保持逐字节一致，新增内容检查无违规，`locked_test`未访问。

## 2026-09-07 组会PPT重点强化完成

- 第208、209、211页改为纯文字上下结构，第207、210、212–215页仅保留能够解释流程、模型机制、训练方式或结果的原生图表。
- 删除了轨迹划分数量等与汇报主线关系弱的内容，文字围绕字段可信、因果样本、双图编码、递推状态、旧模型瓶颈、实体级RSSM和可靠训练连续展开。
- 正式文件SHA-256=`2513cedaca6badb19512dc8c1a08e78cf73fa08f0cc39cfbccb0736896a06299`；第1–205页410个部件与追加前原稿一致，新页最小18磅、楷体/Times New Roman、无正文溢出，PowerPoint重开和渲染通过。
- 结果边界未改变：第215页仍是seed `20260831` epoch 22的单seed中间证据，跨seed验收尚未完成，`locked_test`未访问。

## 2026-09-07 当前双图对象与特征重新核验

- 当前正式 h20 tensor 的 5 项 `distance/csi_mean/rate_sum/active_task_count/allocated_rb_count` 确实属于 `physical_edge_state`，由物理通信链路快照产生。
- 当前信息节点是与活动物理节点一一附着的 agent；信息边是 agent 之间的任务输入、结果回传或显式依赖数据流，在 tensor 中使用 5 维 `flow_state`，而不是名为 `information_edge_state` 的数组。
- 模型分别执行 node—physical-edge 和 agent—flow 消息传播，再通过 agent—node 附着和 flow—physical-edge 承载关系交换消息。实体级 RSSM 的随机状态覆盖 node、physical_edge、flow、task；agent 仍是确定性中间 latent。
- 此次复核纠正了“缺少 `information_edge_features` 键等于没有信息边”的错误推断。未修改当前 GPU 进程、checkpoint、tensor 或 PPT 文件；`locked_test`边界不变。
## 2026-09-08 P4 实体级 RSSM 首 seed 正式验收

- `20260831` 正式全量训练已完成，最佳 epoch=`39`；validation/calibration link-F1 相对 persistence 为 `+0.44415/+0.86229`，node-x 总体/h5/h10/h20 ratio=`0.75475/0.76995/0.75071/0.75495`，throughput/RB/task-delay ratio=`0.93555/0.47363/0.01287`。
- 独立验收记录：`code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260831_acceptance_20260908/single_seed_acceptance.json`。正式 manifest 79 项零缺失、零哈希差异，strict reload 为 `0/0`，原始 metrics 重算全门通过。
- 后续 seed `20260830` 已使用 batch `8` 和完全相同的冻结协议启动；未访问 `locked_test`，P4 仍等待另外两个 seed 和三 seed 独立审计。
## 2026-09-08 正式训练暂停边界更新

- 已将监控自动化更新为：seed `20260830` 完成后只回传、独立验收并停止。
- 无论 seed `20260830` 通过或失败，均不得自动启动 seed `20260832`，必须等待用户指令。
- 当前运行中的 seed `20260830` 不受影响；P4、P6 与 `locked_test` 边界不变。
- 已纠正自动监控输出：后续不得只报 epoch/GPU，必须从 checkpoint 的 `p4_gate` 报告具体门控数值。

## 2026-09-08 组会 PPT 内容一致性审计

- 已对照正式数据合同、双图和实体级 RSSM 代码、首 seed 验收 JSON 逐页检查第 204–210 页，未改动 PPT。
- 必须修正四类表述：逐 RB sidecar 当前不参与训练；旧训练张量的 speed/acceleration 为零不等于原始轨迹不移动；实体级结构目前只有单 seed 证据；结果页必须声明 `locked_test` 未访问且不构成最终性能结论。
- seed `20260830` 当前进程存活，base 20 轮已完成，实体 RSSM 已产生第 6 轮 checkpoint；`20260832` 未启动。

## 2026-09-08 项目知识入口重构第一阶段

- 已完成只读盘点：确认当前仓库存在运行中的 Python 训练相关进程、dirty worktree 和正在变化的 seed `20260830` live evidence；保护区未移动、重命名、删除或覆盖。
- 已新增项目地图、架构说明、科研状态、实验索引、结果索引、重构计划和结构变更记录，全部位于 `docs/`。
- 已更新根 `README.md`、`docs/README.md` 和 `AGENTS.md` 的导航/持续维护规则；未修改算法实现、冻结协议、训练配置、checkpoint、tensor 或 `locked_test` 边界。
- 已验证新增 Markdown 链接目标存在，`git diff --check` 对本轮受影响的导航文件通过，当前训练证据目录仍持续产生原始文件。
- 发现当前 PPT 二进制与旧验收 JSON 的页数/SHA-256 不匹配；该问题只登记，不在训练期间修复或重新验收。
- 当前门：P4 blocked，`formal_performance_claim_ready=false`，`locked_test_accessed=false`，P6 未开放。
- 单一下一动作：继续保护并只读观察 seed `20260830`，完成后独立验收再更新索引。
## 2026-09-09 组会追问手册完成

- 新增 `meeting/2026-09-09-PI-JWM组会追问高密度问答.md`，以 56 个高密度问答覆盖课题主线、数据合同、双图编码、RSSM、旧方案根因、指标、首种子证据和预期追问。
- 文档字符数为 19,164；不存在乱码或未闭合代码块，14 个关键证据入口均已核对存在。
- 对当前图语义冲突采用事实边界：首种子结果验证逐链路实体级动力学，不证明把通信链路归入 `physical_edge_state` 的理论划分更优。
- 未改动 PPT、训练产物、模型或 `locked_test`。

## 2026-09-09 P4 实体级 RSSM 第二个正式 seed 验收

- seed `20260830` 已完成完整两阶段训练，gate-aware 最佳 checkpoint 为 epoch `40`；validation state NLL=`-3.10336359`。
- 独立重算的 validation/calibration link-F1 delta=`+0.45710/+0.88509`；node-x 总体/h5/h10/h20 ratio=`0.75086/0.76033/0.74939/0.74960`；throughput/RB/task-delay ratio=`0.93831/0.47523/0.01175`，9 项单 seed 数值门全部通过。
- 正式 run 已回传到 `code/artifacts/experiments/pi_jwm_p4_entity_rssm_gpu_formal_seed_20260830_v1`；独立验收位于 `code/artifacts/audit/pi_jwm_p4_entity_rssm_seed_20260830_acceptance_20260909/single_seed_acceptance.json`。manifest `79` 项零差异，strict reload 为 `0/0`。
- 远端 PID `51974` 已退出，GPU 为 `0 MiB/24564 MiB`，可以释放。按用户要求暂停，不启动 seed `20260832`。
- `locked_test_accessed=false`、`formal_performance_claim_ready=false`；P4 仍等待第三个 seed 和三 seed审计，P6不开放。单一下一动作是等待用户决定是否运行 `20260832`。

## 2026-09-09 全项目知识与工程结构重构

- 完成阶段 2 机器映射：828 个项目文件、604 个 Python 节点、802 个 artifact 一级目录；生成文件/哈希、依赖/反向引用、artifact 控制证据和归档候选表。
- 新增用户学习路径、AI 检索指南、代码状态、冲突、归档、目录导航和四类人工注册表。
- 索引生成器使用 TDD 完成 6 项测试；UTF-8 BOM 最小复现先失败，改用 `utf-8-sig` 后 12 个误报降为 0。
- 14 个历史 manifest 权限不可读被保留在 catalog；没有修改权限、删除文件或用缺失状态替代真实错误。
- 第三个 seed 和远端同步已登记为延后、需用户授权且禁止自动启动；本轮未运行 GPU、未执行同步、未触碰 `locked_test`。
- 验收结果：索引 `--check` 通过，项目知识/结构测试 15/15、当前正式 P4 测试 210/210 通过，compileall 和 `git diff --check` 通过；全量 1624 项为 0 failure/21 errors，剩余均已登记为环境、历史 fixture 或历史 artifact 边界。

## 2026-09-09 长期协作与快速问答闭环

- 已将科研决策权、AI 独立判断、通俗解释阶梯和证据限定写入永久规则与协作指南。
- 重要实验统一为 7 项完整字段记录；6 类关键历史方法增加动机、结果、弃用原因、替代关系与原始证据；5 类常见问题建立只读路由。
- 正式结果注册表已与两份原始 acceptance JSON 自动比较 SHA-256、seed、epoch、状态、9 项指标及封存边界，当前不一致数为 0；负向测试证明任一登记指标漂移都会被拒绝。
- 查询工具和自动测试现可定位当前模型/训练入口、完整 RSSM 弃用原因、正式 seed 指标来源、第三 seed 延后条件、精确旧 artifact 与精确代码文件。
- 最终映射为 834 个项目文件、606 个 Python 节点、802 个 artifact 一级目录；项目知识/结构 27/27、正式 P4 210/210、compileall、`git diff --check` 和生成器 `--check` 通过。
- 全量套件为 1636 项、0 failure/21 errors；错误仍来自已登记环境、历史 fixture 或历史 artifact 边界，故不物理迁移历史文件。未启动 GPU、第三 seed、同步或 `locked_test`。

## 2026-09-10 AI_CONTEXT 与三方 GitHub 协作

- 重新以当前源码、冻结协议、两份正式 acceptance、机器注册表和 Git `main@e382d79` 为依据核实项目；未读取私人笔记。
- TDD 红灯确认缺少九个 AI_CONTEXT 文件、ChatGPT 路由、验证函数和长期规则；实现后 AI_CONTEXT 契约 6/6、网页端入口查询通过。
- `AGENTS.md` 按本次明确授权保留长期 Research Engineer/研究者/ChatGPT Web 边界，加入 Context Consistency Check、Git commit/push、冲突模板和私人笔记禁区，并移除具体 P4 临时步骤。
- 自动索引覆盖 844 个项目文件、607 个 Python 节点、802 个 artifact 目录；9 个 AI_CONTEXT 文件、6 类问题路由、2 个正式结果证据核对均无错误。
- 项目知识/结构 28/28、正式 P4 210/210、compileall、索引 `--check` 和 diff 检查通过；全量 1643 项为 0 assertion failure/17 errors，均属已登记外部环境或历史 fixture 边界。
- 本轮未修改模型、loss、metrics、tensor、协议、checkpoint 或 artifact，未启动 GPU、seed `20260832`、远端同步或 `locked_test`。

## 2026-09-16 最新 AirFogSim 源轨迹分享包（仅本地交付）

- 用户明确要求最新版，交付 v6 formal source（B层），不含六月A层CSV、训练npz/checkpoint或locked_test；没有重跑仿真。
- 源：code/artifacts/formal_data/pi_jwm_v4_formal_candidate_v6_rb_v1_unlocked_20260821；54条、6场景、每条300步、0.1秒、合计16200轨迹时点。
- 完成：code/artifacts/packages/AirFogSim_raw_data_share_20260916.zip；658389567字节；SHA256=e2b73fc877a3c5117767a20e23c2527281876c0036e42be16835fb99a08fcb59。
- 验证：486项轨迹文件及8项顶层历史清单匹配；54个时间网格通过；ZIP全部800文件逐项解压读取、大小和SHA256通过；所选数据零缺失。
- 限制：历史完整config哈希重建0/54匹配；生成时项目/AirFogSim commit无法确认；最新v6源与当前tensor摘要所指v4不是同一来源声明。现有场景不构成严格单变量对照。上述差异仅报告，没有改写源证据。
- Context Consistency Check只读完成；按用户要求未修改AI_CONTEXT、模型或训练代码；本任务不commit/push。科研P4、第三seed、GPU、同步和locked_test边界不变。
- 本次单一下一动作：用户直接将ZIP交给同学；若需逐值重现历史仿真，先恢复完整历史配置和版本，不能自行重跑。

## 2026-09-18 STEP 1 — New Definition → Current Implementation Audit

- 读取了研究者只读目录中的七个 `00–06` 定义，并将文件大小、行数、SHA-256 和起始 Git 状态写入 `initial_snapshot.json`。
- 审计了轨迹/时间、dataset/tensor、双图、entity alignment、RSSM、规则反馈、四类动作、training/loss/evaluation 和 planner/closed loop。
- 新增 `docs/PIJWM_IMPLEMENTATION_TRACKER.md`、`docs/implementation_records/README.md`、`STEP_01_AUDIT.md` 和数据/双图附件；更新治理、权威记录和上下文导航。
- 运行既有 synthetic CPU 合同测试 49 项，返回码 0；该结果明确不作为新定义验收。
- 未修改模型、数据、loss、planner 或 checkpoint；`gpu_started=false`、`locked_test_accessed=false`、`step2_started=false`。

## 2026-09-18 STEP 2

结论：单决策步合同已冻结，四类 AirFogSim setter 的最小环境闭环通过。历史真实非 locked ledger 证明 Route/Comm/Comp 的 accepted env-step 记录，未证明 UAV mobility 真实采集。当前不进入 Step 3。
# 2026-09-19 Step 2.1 真实 AirFogSim 验收

- 真实非 locked 单轨迹的四类动作、真实 `env.step()`、Outcome 与 next Decision 已通过；当前 Git 可追溯重跑为 `code/artifacts/protocols/pi_jwm_raw_single_decision_step_real_airfogsim_v4_20260919/`。
- 未训练、无 GPU、未访问 `locked_test`；Dataset/Tensor、双图、World Model、Loss、Planner 未进入。
- 真实接口合同已修正为 vehicle heading degree、UAV heading rad，Mob action 仍为 azimuth_rad；Step 2 测试记录修正为 6 项。

## 2026-09-19 Step 2.2 真实 AirFogSim 多步验收

- 真实 6 步轨迹生成 7 个独立 Decision，frame `0..6`、time `2.4..3.0 s`；18 项 checks 全部通过。
- Outcome 与下一循环 Decision 的 entity/task ID、lifecycle 和状态一致；Route/Comm/Comp 各自同时覆盖非空与显式 no-op，Mob 每步只覆盖 UAV。
- Task_1 生命周期从 waiting_to_offload 进入 computing，再进入 waiting_to_return；车辆由 SUMO 推进。
- 两个 UAV 首次 `0 -> 10 m/s` 时常规差分约 `+100`，AirFogSim acceleration 为 `-100`；只记录语义，不修改 simulator 或决定 Dataset。
- Step 2.1 追加 v4 真实证据，准备与 Step 2.2 v2 JSON/manifest 一起纳入 Git。无训练、GPU 或 `locked_test`。

## 2026-09-19 Step 2.3 Raw Contract 最终验收

- 新增 3 项 causal helper 测试并通过；单步合同专项测试现为 7/7。
- 真实 8 slot、9 Decision 轨迹从 `2.4 s` 到 `3.2 s`，17 项 checks 全部通过。
- 首帧有 9 个 future schedule task 和 5 个 observable task；逐帧未发生 future-object leakage。
- 首帧 42 条真实 channel row；CPU capacity 7 个 observed、1 个 explicit missing。
- 6 个 slot 有 delivered-data，6 个 slot 有 served-CPU；真实 return setter 调用 3 次并进入 returning/done。
- Raw/canonical acceleration 已分字段和 mask。无训练、GPU 或 `locked_test`；后续层未开始。

## 2026-09-19 Step 2.4 Communication Outcome 语义收尾

- 真实 AirFogSim wired 路径已核对并接线：`WiredNetworkManager.step` 的逐 task 返回值形成 wired event，wireless event 显式标记 transport。
- 新 Raw Outcome 分为 wireless map、wired map 和两者按 task 求和的 total；空 map 表示已观测但无服务，missing 使用 `null + mask=false + reason`。
- 真实 6 slot/7 Decision 轨迹完成 `UAV_0 → RSU_0 → cloudServer_4`，14 项 checks 通过；无 GPU、无训练、未访问 `locked_test`。
- Raw Trajectory Layer / 定义 01 正式 COMPLETE / FROZEN；Step 3 未执行。

## 2026-09-19 STEP 3.1F-PATCH

- 修正 Future Action anchor-only 重编号：合法 action 现在引用统一 `static.input_entity_index`，anchor visibility 仍单独拒绝不可见 object。
- 新增 disappearing-object fixture、ID↔index validator 和错位拒绝测试；machine policy 改为 `history_causal_observable_object_union`。
- 重新生成 sample/manifest/audit JSON，manifest 保存 audit SHA-256 provenance；未进入 STEP 3.2、模型或训练。
