# PI-JWM Current State Snapshot

## 2026-09-26 STEP 6.0B 来源审计

CPU 只读源码审计得到 `STATIC_CAPACITY_ONLY`：决策时刻有带 mask 的静态 CPU 容量，未证实动态可用量；AirFogSim 原生计算回调不按节点容量裁剪，PI-JWM 正式采集器另行限制自身分配。UAV 直接执行接口没有数值硬边界；示例配置是生成/初始化设定，正式数据动作范围只是行为支持。因此 6.0A 的两项 `UNKNOWN` 均保留，Candidate 代码未改。AirFogSim 本地目录没有独立 `.git`，`git -C` 上溯到 PI-JWM；不能声称本地 AirFogSim Git SHA/clean 状态，机器凭证提供相关源码文件哈希。5.6B 远端未联系，训练结束与 best checkpoint 未核实。详见 STEP 6.0B 实施记录及机器凭证；下一步先补齐仿真器历史 Git 身份（若需要精确 commit），模型依赖 Planner 仍需另行授权。

## 2026-09-26 STEP 6.0A 当前状态

Line A：STEP 5.6B 正式训练是独立远端 run `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`，源码 SHA `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`；本 Step 未连接远端，仓库只保留既有 2026-09-25 过程快照，不能据此断言当前远端进度或最终 best checkpoint。Line B：STEP 6.0A 已完成 CPU 静态候选生成合同和合成 fixture 验收；Search/Learned/Hybrid 只是可插拔接口，最终方法待研究者决定。当前正式训练动作适配器仍是 `build_step5_1d_unified_model_chain_v1.py::build_action`，新编译器包装它。没有 World Model 候选 rollout、objective、proposal training、GPU、closed loop、baseline 或 `locked_test`。证据：`docs/implementation_records/STEP_06_0A_UNIFIED_CANDIDATE_GENERATION_CONTRACT_CPU.md` 和对应机器 receipt。唯一下一动作：等待正式 best checkpoint，再由研究者授权模型依赖的 Planner Rollout Step。

## STEP 5.6B 训练中快照（2026-09-25）

正式训练已按冻结配置在 RTX 4090 上运行。远端 run ID 为 `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`，训练源码 Git SHA 为 `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`。2026-09-25 13:34 UTC 的只读心跳为 `RUNNING`、2646/5520 completed steps；已有 step 1104、2208 两次完整 1104-window prior-only validation。中途曲线和复制日志哈希见 `docs/figures/step5_6b_live_progress_20260925/`。这是过程诊断，不是最终性能结论；`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。下一动作仅为继续监控此 run。

## STEP 5.6B 预启动状态（2026-09-24，历史）

研究者已明确授权使用冻结 Formal Training Config v1 在 RTX 4090 启动一次正式训练。正式 runner 与监控/恢复路径正在做启动前验收；截至此源码快照，`formal_training=false`。5.6A 的完整 CUDA smoke 和 prior-only validation 仍是运行证据，不是性能结论。启动后的真实状态只以远端 `run_manifest.json`、`progress.json` 和 `heartbeat.json` 为准。`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。唯一下一动作：完成 Go/No-Go、提交精确 source、远端 detached launch，并在 2–3 步确认后停止。

## STEP 5.6A 当前状态（2026-09-24，GPU 验收完成）

正式数据身份保持 H=2/L=4、60 条 trajectory、48/12 split、4416/1104 windows，Dataset package/hash 未改。RTX 4090 上 H=4 CUDA 前向、反向、参数更新、跨轨迹 batch、checkpoint/错误身份拒绝和完整 1104-window prior-only GPU validation 已有通过凭证；验证集 12 条轨迹均覆盖且没有重复/遗漏。未训练 smoke checkpoint 的 `L_Val=0.829751` 仅作运行诊断，不是性能结果。研究者已冻结 Formal Training Config v1，机器状态为 `FORMAL_TRAINING_CONFIG=FROZEN`、`FORMAL_TRAINING_READINESS=READY_TO_START`；`formal_training=false`、`gpu_training_verified=false`、`locked_test_accessed=false`、`baseline=false`、`planner=false`、`performance_claim=false`。唯一下一动作是另行授权 STEP 5.6B，不得自动启动。

## STEP 5.5-PATCH 历史状态（2026-09-23）

STEP 5.5 的原 CPU H=4 smoke 只消费 `runtime/` 的 1 train + 1 validation，不能作为 5520-window Trainer 证据。PATCH 新增 `FullFormalShardDataset → FullFormalTrainer`：4416/1104 全量索引、按请求加载 trajectory shard、跨 shard batch、H=4 CPU 参数更新、prior-only validation batch、checkpoint reload/错误身份拒绝已有独立凭证。原 runtime mini 与 8/4 development 路径仍保留。

旧 Dataset receipt 的 `unsupported/unresolved/fixed_support_blocked=0/0/0` 是字段计数，未检测真实 future Return birth；PATCH 对全部 5520 windows 的独立结构审计发现 8828 次按窗口与未来步计数的 Return-birth unsupported/fixed-support 事件，涉及 2901 个窗口和 60 条轨迹；143320 次已有支持的 Return continuation 未误判。新 detector 只写 target-side component 记录，不创建 current Return slot，也不删 Motion/CSI 监督。旧零值已被新审计替代。60 条 Raw 中有 213 次同一 Task 对象的 lifecycle collection 修复，涉及 50 条轨迹、124 个 trajectory-task；保留最远 lifecycle，直接 Task 状态字段不改。

机器凭证：`code/artifacts/audit/pi_jwm_step5_5_patch_20260923/`。`FULL_FORMAL_DATASET_LOADER=VERIFIED`、`H4_FULL_DATA_CONSUMPTION_PATH=VERIFIED` 指全量可索引的 CPU 数据路径和抽样执行，不表示 5520 窗口已完整正式训练。当时 `gpu=false`、`formal_training=false`、`locked_test_accessed=false`。

Formal Dataset v1 已由 60 条真实 AirFogSim trajectory 构建并机器验收：H=2/L=4、每条 96 transitions、48/12 trajectory split、4416/1104/5520 windows，五类 package、四动作 coverage、train-only normalization 与 deterministic rebuild 均通过。原 CPU H=4 trainer smoke 是 runtime 1+1 mini；PATCH 另行验证 full-shard CPU batch 路径。`formal_dataset=true` 只表示数据包 READY；在该 PATCH 时点 GPU execution 尚未开始。

以下为历史补充；该段原更新时间为 2026-09-22，当前状态以上方最新快照为准。

这是 ChatGPT 网页端进入仓库后的第一读取入口。它只提供当前快照和继续查证的路径，不替代源码、配置、checkpoint、metrics、manifest 或 audit。

## Source of truth

事实优先级固定为：当前源码/config/experiment → `AI_CONTEXT/` → 普通项目文档 → 历史聊天或推断。任何正式科研结论都必须回到第一层验证。

## 2026-09-22 历史状态

- 项目：PI-JWM（Physical-Information Joint World Model，物理—信息联合世界模型）。AirFogSim 只是参考仿真器和数据生成工具。
- 当前 active workflow：研究者最新只读 `00–06` 定义链；工程执行入口为 `docs/PIJWM_IMPLEMENTATION_TRACKER.md` 和 `docs/implementation_records/`。
- 当前状态：`STEP 5.5 = COMPLETE / FORMAL DATASET V1 ACCEPTED`，PATCH 的 full-shard CPU 数据路径另行验收。60/60 trajectory、五类 package、四动作真实 coverage、H1-H4 Motion/CSI、hash/identity、确定性重建和 runtime mini CPU H=4 smoke 均有机器证据。正式训练、GPU execution、planner、baseline、locked-test 仍未开始。
- `STEP 3.2-PATCH` 已补齐 Dataset isolation provenance、time-grid、development future-reference audit 与 normalization units；仍是 observation-only/non-locked evidence，不是正式 Dataset。
- `STEP 3.2-PATCH-RECEIPT` 已修正顶层 acceptance AND、显式 scope checks 和 frozen sample schema reuse；STEP 3.2 现正式 COMPLETE / FROZEN。
- STEP 4.4-PATCH3 已在源码中显式区分 Return requirement 的 known/unknown：冻结 current-side 不含 `Task.return_size`，所以 no slot 是 unknown，不是 no-return；unknown 或 known-required/no-slot 都不能错误 final-complete，但 side-state 可区分两者。DAG 只按有效前驱动态释放，terminal Flow completion 同步 remaining/presence/carrying/status。仍是 untrained CPU development evidence，不代表预测精度或训练结果。
- Definition 05 v1 decisions 已冻结：deterministic mean decoder；Motion/CSI family-wise mask-MSE；Phy/Comm analytic KL + warm-up/free-bits；overshooting OFF；Future Target 仅进入 family-specific training posterior；joint training；prior-only validation；checkpoint=`argmin L_Val`；逐 horizon raw-unit Motion/CSI MAE/RMSE。STEP 5.1A target、5.1B primitives/paired integration、5.1C development bundle、5.1D CPU chain 和 5.2 CPU training loop 均已有对应证据；full training 尚未开始。
- 新定义实现状态：Raw、最小 Dataset/Tensor、Typed Graph Builder、Dual-Graph Encoder、Structured RSSM World Model、5.1B loss/KL/metric primitives 已验收；5.1C additive bundle 将 12 个 paired window 的 support 对齐为 observed `10/74`，5.1D 从同一 bundle 完成 graph/encoder/world-model paired CPU integration。Physical topology、Encoder/World Model 参数仍是 development-only，模型权重未训练。
- 审计结论：时间因果、稳定 ID/index、mask/split、typed graph、`Z_t^{PI,L_g}→xi_t^Lat`、current-observation posterior、目标 RSSM 边界和逐步规则反馈已落地；完整 planner 闭环仍需后续授权与实现。
- 当前运行：没有正式 GPU 训练或远端同步任务；旧 `seed=20260832` 仍不得自动启动。
- `locked_test_accessed=false`；`formal_performance_claim_ready=false`；`formal_dataset=true`、`training_loop_implemented=true`、`cpu_optimizer_smoke=true`、`full_training=false`、`gpu=false`。

## 2026-09-22 当时最重要问题

当前实现不能按 Dataset READY、runtime mini smoke 或 PATCH full-shard CPU batch 外推性能。没有 formal training、GPU runtime、收敛泛化、校准或预测精度证据；Planner 真实反馈也未实现。

## 2026-09-22 当时建议的下一步

唯一建议是研究者另行授权 **STEP 5.6A — GPU Smoke + Formal Training Config Freeze**；在此之前不启动正式训练，也不访问 `locked_test`。

## 当前 Git

- Branch：`main`。
- Step 1 base commit：`829276241a0da72d3a5393946086daba40b0a0fe`。
- Latest commit：以 GitHub `main` 的 `HEAD` 为准；本文件不能稳定硬编码包含自身的 commit hash。读取时运行 `git rev-parse HEAD` 或查看 GitHub 分支头。

## 关键入口

- 当前方法与边界：`AI_CONTEXT/02_ARCHITECTURE.md`
- 新定义实现总表：`docs/PIJWM_IMPLEMENTATION_TRACKER.md`
- Step 1 详细记录：`docs/implementation_records/STEP_01_AUDIT.md`
- 数据和张量：`AI_CONTEXT/03_DATA_FLOW.md`
- 问题到源码：`AI_CONTEXT/04_MODULE_MAP.md`
- 当前/历史实验：`AI_CONTEXT/05_EXPERIMENTS.md`
- 已确认决策：`AI_CONTEXT/06_DECISIONS.md`
- 冲突和阻塞：`AI_CONTEXT/07_KNOWN_ISSUES.md`
- 机器注册表：`docs/registries/`
- 原始状态权威：`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`

## 最近重要变化

- 2026-09-20：STEP 4.2A-PATCH 解耦 wireless structural relation validity 与 CSI observability；missing CSI 保留 relation 并使用 mask=false/zero placeholder。return size/priority/deadline 改为 observer available but frozen Raw not exposed；stateful Flow 仍未解决。
- 2026-09-20：STEP 4.2B source audit 证明 `transmitted_size` 为 hop-local stage progress，不能推出 end-to-end Flow remaining；DAG 只提供 gating，DepData transfer 未找到。综合 verdict=`FLOW_CONTRACT_NOT_YET_SUPPORTED`。
- 2026-09-20：STEP 4.2B-PATCH 将 verdict 改为 Flow-specific evidence 的实际计算；resource gaps 与 Flow readiness 解耦，stable ID/DepData 改为 implementation fact + researcher decision boundary；provenance 增加 symbol anchors。
- 2026-09-20：STEP 4.2C-A-PATCH 通过 audit-only replay 修正 verdict：logical destination 过滤 final delivery，E2E remaining 与 holder 可因果派生，same-destination reroute 可保持 Flow epoch；`flow_completed` 禁止作为 logical completion；机器 verdict=`CAUSAL_FLOW_LEDGER_FEASIBLE`。destination change/DepData 仍需研究者决定。
- 2026-09-20：STEP 4.2C-B 实现 FlowID/Epoch/RouteRevision、Flow/Carrying 分离、Input/Return lifecycle、clean-boundary destination change、lineage 与 Raw additive state；真实 non-locked trace覆盖 Input/Return，DepData runtime=0。随后 STEP 4.2C-C 完成 Sample/Tensor additive extension；在该 Step 当时 Graph Builder 尚未开始，之后已由 STEP 4.3A 完成。
- 2026-09-20：STEP 4.2C-B-PATCH 修正 logical destination provenance：Input 使用已成立 route terminal，Return 使用 `return_destination_id`；真实 `UAV_0→RSU_0→cloudServer_4` 两跳通过单 FlowID/Epoch、固定 destination、无重复 E2E 计数验收。
- 2026-09-20：STEP 4.2C-C-PATCH 将 normalization stats 收紧为 `known=true AND presence=true AND feature_mask=true AND value!=null AND split=dev_train`，并补齐 History/target Logical/Carrying 四组全字段 semantic equality、ID/provenance tamper 检查与 target carrying future-ground-truth namespace；23/23 focused、跨时隙真实 trace、deterministic/round-trip/receipt negative checks 通过。在该 Step 当时 Graph Builder 尚未开始；之后 STEP 4.3A/4.3B/4.4 已依次完成 Builder/Encoder/World Model Contract，训练仍未开始。

- 2026-09-19：STEP 4.1-PATCH 修正最小 gap 语义：wired relation 是 Raw/simulator 有来源但未暴露，无 CSI 时用 type + mask；wired 可选 numeric state 不阻塞 03 minimum；CPU capacity 是静态 capability，并与 allocation/service/available CPU 分离。在该 Step 当时 graph builder 保持关闭，之后已由 STEP 4.3A 完成。

- 2026-09-19：完成 Step 2.4 wireless/wired/total communication Outcome 语义最终验收；wired 服务来自真实 `WiredNetworkManager.step`，空 map 与 missing 分开，冻结 Raw Trajectory Layer / 01。
- 2026-09-19：完成 Step 2.2 真实 AirFogSim 6 步轨迹与独立下一 Decision 验收；补齐 Step 2.1/2.2 机器证据 Git 追溯。
- 2026-09-09：第二个正式 seed `20260830` 完成并通过单 seed 验收；第三 seed 暂停。
- 2026-09-09：建立项目文件、依赖、artifact、实验、结果、历史方法和问答路由索引。
- 2026-09-10：新增面向 ChatGPT 网页端的 `AI_CONTEXT/`，并将三方协作和同步规则写入 `AGENTS.md`。
- 2026-09-19：STEP 3.1 冻结最小 Model-ready Sample & Tensor Contract；`H=2/L=2` 真实样本、四类 action、input/target index 隔离和 mask 语义通过机器检查。正式 batch/split builder 未开始。
- 2026-09-19：STEP 3.1R 修正 History `[t-H+1,t]`、固定 index/presence、真实 DAG 接线、typed target namespaces 和 relation endpoints；v2 Raw 与最小样本证据已重建。
- 2026-09-19：STEP 3.1F 及其最小 PATCH 已冻结；History 保留 past Action/Outcome，input index 使用 History causal union，Future Action 先做 anchor visibility 检查再引用同一 static index，future-reference 观察审计 JSON 已纳入 provenance/Git。
- 2026-09-19：STEP 3.2 完成最小 non-locked batch/split/preprocessing validation；3 trajectories、12 windows、train-only mask-aware stats、deterministic rebuild 和 round-trip 已有 artifact/test 证据，不代表正式 Dataset。

Unverified：当前没有“最终 PI-JWM 方法已冻结”或“正式性能声明已开放”的证据。

## 2026-09-22 STEP 5.1C-PATCH

- Unified bundle 对 12 个 paired windows 完成 slot-wise Physical/Communication identity proof；统一容量为 `max_entity=10`、`max_comm_relation=74`，no-prefix/pairing crop 机器检查通过。
- Flow normalization 新 stats 只从 8 个 unified `dev_train` samples 的 History 拟合；4 个 `dev_validation` samples 排除，Future Target 未参与 fit；旧 5-sample stats 仅保留为历史 provenance。
- receipt、exact upstream train lineage、runtime prior-target isolation、tensor contract、deterministic rebuild 和 serialize/reload 通过；仍为 CPU/non-locked development evidence。Route/Comp non-empty coverage 在真实 12-sample bundle 中均为 0，只验证 explicit no-op；这是未来 formal training/data coverage gate，不是完整四动作族训练证据。

## 2026-09-22 STEP 5.2-PATCH

- Stage 2/Validation 初始 latent 改为 current-observation posterior；未来递推继续 prior-only。current posterior 纳入 joint optimizer，Future Target teacher 独立且 validation 调用为 0。
- Validation 改为每个 horizon 跨完整 validation set 的 Motion/CSI numerator/count 独立归一化，再计算 `L_Val`；补 unequal-mask fixture。
- Checkpoint load 增加 schema、data identity、normalization provenance、architecture-critical config 拒绝检查；compatible reload、wrong-data/normalization rejection 均通过。receipt 为 26/26。

## 2026-09-22 STEP 5.4

- 新增 manifest-driven `FormalTrainingInterface` 与 CPU-only readiness audit；5.2 development adapter 保持回归兼容。
- readiness：`TRAINING_STACK_READINESS=PASS`、`FORMAL_DATASET_READINESS=NOT_READY`、`GPU_CODEPATH_READINESS=PREPARED`、`FORMAL_TRAINING_READINESS=BLOCKED`。
- 12 个 development samples 实际 coverage：Mobility=24、Comm=1、Route=0、Comp=0；正式 `L`、topology、Dataset/seed/训练预算仍需 researcher decision。

## 2026-09-23 STEP 5.4-PATCH2

- 四动作 adapter 与 CPU device portability 已闭合；optimizer 在最终 device migration 后创建，Route/Comp adapter support 通过 synthetic fixture。
- 当时 readiness：`TRAINING_STACK_READINESS=PASS`、`FORMAL_DATASET_READINESS=NOT_READY`、`GPU_CODEPATH_READINESS=PREPARED`、`FORMAL_TRAINING_READINESS=BLOCKED`。
- Formal Dataset 不存在，真实 development Route/Comp coverage 仍为 `0/0`；formal L/topology/budget 未决，GPU/locked_test 未执行。

## 2026-09-22 STEP 5.2

- `code/src/pi_jwm/step5_2_training_loop_v1.py` 连接当前 Encoder、Structured RSSM、5.1B target/posterior/loss/KL 原语；Stage 1 使用 family-specific posterior teacher，Stage 2/Validation 从 current-observation posterior 初始化，之后 prior-only recursive rollout。
- 配置化 curriculum 为 `1→2→4`，当前 development `L=2` 自然为 `1→2`；`beta_KL`、warm-up、free bits、optimizer、clip、seed、batch size 和 epoch/step 均进入 config。
- 真实 8/4 unified non-locked bundle CPU smoke：两步 optimizer update、4 validation samples prior-only、checkpoint save/load/resume；5.2-PATCH receipt `26/26`，`passed=true`。随后 5.3 固定 dev_train 1/2-sample preflight receipt=`GO`；这两者都不是正式性能证据。
- optimizer audit 覆盖 Encoder、RSSM dynamics、两类 prior、两类 future posterior、两类 target encoder、Motion/CSI decoder；known rule parameter count=0。Route non-empty=0、Comp non-empty=0、Comm=1、Mobility=48，Route/Comp 仍是 future formal training/data coverage gate。

## Freeze chain（2026-09-20 历史快照）

- Raw Trajectory / 01、当前最小 Dataset/Tensor / 02、STEP 4.1 mapping、STEP 4.2A existing-source input extension、STEP 4.2C-B Raw Flow、STEP 4.2C-C Flow Sample/Tensor、STEP 4.3A Typed Dual-Graph Builder 与 STEP 4.3B Dual-Graph Encoder 均已冻结。
- Causal boundary: future task schedule is internal metadata only; canonical acceleration is backward speed difference with an explicit missing-history mask.
- 当时边界：Physical topology 的 `radius_knn/radius=1000m/k=2` 仅是 development config，`research_frozen=false`；此项后来已由 STEP 5.5 正式数据协议冻结。Return multi-hop 与 same-destination reroute runtime 的历史边界保留。
- 当时状态：Definition 05 decisions FROZEN；5.1A COMPLETE/FROZEN，5.1B COMPLETE/FROZEN FOR CPU DEVELOPMENT PRIMITIVES + PAIRED INTEGRATION，5.1C COMPLETE/FROZEN FOR DEVELOPMENT，5.1D COMPLETE/FROZEN FOR CPU DEVELOPMENT INTEGRATION，5.2 COMPLETE/FROZEN FOR CPU DEVELOPMENT TRAINING-LOOP INTEGRATION。该时点的 `formal_dataset=false`、`gpu=false` 不描述当前状态。
