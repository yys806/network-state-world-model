# STEP 6.0A — Unified Candidate Generation Contract v1

Status: CPU static implementation. Definition basis: read-only `D:\shen\OB\科研\PIJWM\06策略器与候选动作规划.md`, SHA-256 `F20294BD8708076AE7583BE679A8182A0990ECF6777E05DF378AE4C2855431F1`, sections 1–2 and 4. Researcher-approved four-family action and downstream MPC boundary are also in this Step authorization. This is not a final candidate-generation method.

## Scope and data flow

Current causal `PlannerCandidateContext` → known constraints and eligibility → `CandidateActionStep` → `CandidateActionSequence` (1–4 steps) → pluggable backend → deduplicated `CandidatePool` → formal action tensor compiler. The chain stops there. **NO WORLD-MODEL ROLLOUT WAS PERFORMED.** H1–H4 formal supervision/validation is the reason for the maximum horizon 4; this is no optimal-horizon claim.

The context carries current typed identity/support via `static.input_entity_index`, causal `history`, current state tensors and provenance. `latent_reference` is opaque, optional and unused. Neither future truth nor labels are inputs. For a sequence longer than one step, the compiler requires one supplied causal/predicted state per step; this Step does not create those later states. A candidate sequence alone is never executable as an MPC plan.

## Four action families

`CandidateActionStep` preserves Sample/Action entry names and semantics. Route rows carry `task_id`, `task_index`, `route_kind` (`offload`/`return`), current task node, target node, intended route nodes and optional existing `flow_id`. Comm rows carry task identity, explicit current `relation_index` and RB indices; the formal adapter resolves the current Flow/hop communication relation or causal pending Route endpoint, and the compiler checks it equals the declared relation. Comp rows carry current node ID, task ID and `allocated_cpu_per_s`. Mob rows carry a current UAV slot and `azimuth_rad`, `elevation_rad`, `speed_mps`. A Vehicle is never a planner mobility row. Empty tuples in each family mean explicit no-op; missing/unknown actions are not silently converted to no-op.

The actual formal adapter is `build_action` in `code/scripts/build_step5_1d_unified_model_chain_v1.py`, still imported by `step5_2_training_loop_v1.py` in the full formal Trainer. The compiler wraps this adapter without editing training source. Its eleven fields are `mobility_entity_index`, `mobility_values`, `comm_relation_index`, `comm_values`, `comm_allocation_mask`, `comp_agent_index`, `comp_task_index`, `comp_values`, `route_task_index`, `route_flow_index`, `route_values`. Route no-op is one row with task/flow `-1` and values `[-1,-1,0,0]`; Comm/Comp/Mob empty rows have zero width. Comm allocation starts from the current `rb_active_mask` and adds requested RBs. No global RB exclusivity is assumed: the current service model explicitly handles interference from simultaneous relations.

## Constraints and fixed support

Each constraint records name, `SATISFIED`/`VIOLATED`/`UNKNOWN`, reason code and source. Known violation rejects candidate creation. Unknown remains visible in the pool; compilation rejects it pending researcher policy. This choice freezes no scientific feasibility rule. No new physical, Task or Flow slot may be created. Future-only Return Flow birth is blocked, even if a Route row names `return`. Existing `task_return_requirement_known`/`task_requires_return` semantics remain in the world model and are not rewritten here.

Nonempty Comp automatically adds `dynamic_available_cpu=UNKNOWN`; nonempty Mob adds `mobility_numeric_bounds=UNKNOWN`. A clearly named `synthetic_contract_test_only` switch permits tensor-equivalence fixtures to compile through these unresolved checks, but does not license execution, feasibility or safety claims. It is not a planner runtime policy.

Comp validates current node/task identity and finite nonnegative allocation request. Static CPU capability, allocation request, actual service and dynamic available CPU remain distinct. No causal dynamic CPU availability source is proven here; dynamic feasibility is unknown. Mobility validates current UAV identity and finite values. Numeric speed/angle bounds remain unknown until source and protocol evidence freeze them. Comm rejects invalid current relations and duplicate task/RB assignments. It does not invent a global exclusivity rule.

## Backends, warm start and fallback

`SearchStubBackend` emits a deterministic contract fixture only. `LearnedProposalBackend` accepts a deterministic callable; no architecture, optimizer, training target or checkpoint exists. `HybridCompositionBackend` merges learned/search/warm-start/fallback outputs without scoring, elite selection or refinement. Search, Learned and Hybrid remain comparable research choices, not a selected winner.

Warm start removes the old first action, shifts the remaining actions, and fills an explicit `TAIL_REQUIRED` position through a supplied tail function; metadata retains parent ID and shift count. This is a seed for a future replan using the next real observation, never an automatically executed plan. `RULE_FALLBACK` uses all-family explicit no-op. Its claim is structural representability under known constraints only; dynamic feasibility and safety are unverified. The pool orders by SHA-256 fingerprint of action semantics, deduplicates cross-backend identical sequences and merges source tags. Backend/source metadata never changes semantic identity.

## Research pending

Final Search/Learned/Hybrid choice; CEM/iCEM/MPPI/other optimizer; K, iterations, elite count/ratio, smoothing, colored-noise beta; autoregressive Route→Comm→Comp→Mob factorization; proposal architecture/dimensions/training target/distillation temperature; objective weights, risk and future hard constraints; dynamic CPU treatment; UNKNOWN policy; fallback safety; planning latency. `proposal_training=false`. Formal Dataset behavior is coverage-oriented, not an optimal expert.

## Evidence boundary

Focused tests use `SYNTHETIC_CONTRACT_EVIDENCE`; real Formal Dataset four-action coverage is not Planner candidate-generation coverage. No intermediate or final checkpoint was consumed. No candidate objective, world-model rollout, GPU, baseline, closed loop or `locked_test` was used.
