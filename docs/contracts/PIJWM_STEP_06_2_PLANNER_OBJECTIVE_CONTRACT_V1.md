# PI-JWM Planner Objective Contract v1

**Status:** `RESEARCHER-FROZEN TARGET CONTRACT` / `SOURCE-RECONCILED` / `PLANNER_V1_ROUTE_POLICY=EXPLICIT_NOOP_ONLY` / `STEP_6_2B=PASS` (CPU scorer contract only)

## Scope

The contract applies to generic computation tasks that are causally visible at the anchor and are not terminal `DONE` or `FAILED`. Candidate scores use one common support-aware horizon. A candidate that first reaches an unsupported future Return state at state index `u_k` has `H_sup = u_k - 1`; the comparison horizon is `H_eff = min_k H_sup`. `H_eff=0` is `OBJECTIVE_UNSCOREABLE`, with no fallback winner.

## Objective

The Planner v1 target is the lexicographically minimized tuple:

`J_v1 = (N_DDL, A_DDL, J_Delay, J_Burden, J_Effort)`.

Deadline violation is first, followed by violation time area, unfinished-task time area, remaining service burden, and resource/control effort. This is a tuple comparison, not a weighted sum. A stable candidate fingerprint may break exact equality but is not a research objective.

`N_DDL` counts current-cohort tasks whose predicted lifetime violates the source-backed deadline rule. `A_DDL` sums the violation indicator over the common horizon and normalizes by `|Q_t| * H_eff`. `J_Delay` is the mean unfinished-task indicator over the same cohort and horizon; it is a finite-horizon Planner surrogate, not Average/P95/P99 closed-loop delay. `J_Burden` is the anchor-masked mean of normalized communication and computation burden. `J_Effort` is the final tie-break only.

The frozen target definitions are:

- `v(q,h)=1` when task `q` violates its lifetime deadline by predicted `t+h` and has not completed within the source-defined completion boundary.
- `N_DDL` counts current-cohort tasks with any violation in the common horizon; `A_DDL=sum(v(q,h))/(|Q_t|*H_eff)`.
- `u(q,h)=1` exactly when the task has not successfully reached final completion. `FAILED` is not successful completion. `J_Delay=mean(u(q,h))` over the anchor cohort and common horizon.
- `b_Tx=R_hop+(N_hop-1-current_hop_index)*R_e2e` where all route and Flow identity terms must be supported.
- `b_Comp=W_rem(q,h)/max(W_rem(q,0),epsilon)`. If `W_rem(q,0)=0`, compute burden is zero. A computation-finished task is not final-complete when a required Return is unresolved.
- `b(q,h)=(m_Tx*b_Tx+m_Comp*b_Comp)/(m_Tx+m_Comp)`. Masks are frozen from the anchor, candidate-independent, and completed task burden is zero. `J_Burden=mean(b(q,h))`.

At STEP 6.2A-ROUTE-RECOVERY these were target formulas only; that Step executed no scoring. The selected Formal Validation anchor has a causally aligned deadline sidecar. The repaired canonical route convention is a destination list excluding the current holder; deterministic cross-hop state and same-destination full-path reroute follow that convention. Formal train/validation multi-hop coverage is zero. At the historical 6.2A-CLOSURE, Planner v1 used the researcher-frozen `FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1`; STEP 6.2B-PATCH further restricted it to explicit Route no-op. Multi-hop code remains available but outside v1 and carries no formal learned-performance claim. For a legal single-hop active Flow, `N_hop=1` and `current_hop_index=0`, so `remaining_hops_after_current=0` and `B_Tx=R_hop`; the general formula remains unchanged for later domains.

## Source boundaries

AirFogSim defines task deadline as relative allowed latency: `arrival_time + deadline`. Return completion accepts `delay <= deadline`; no-Return completion accepts `delay <= deadline + 1e-5`; the later active-task sweep fails only when `delay > deadline`. Completion handling precedes that sweep. Accepted Formal Raw has arrival and `required_returned_size`, but not deadline. A non-locked validation anchor now has an exact-aligned Planner-only deadline sidecar; deadline still does not enter the trained Tensor or 4.4 state. Priority remains outside Objective v1.

**Historical fact:** the pre-Ledger STEP 4.2B source audit could not reconstruct E2E remaining from Raw. **Current implementation fact:** STEP 4.2C-B introduced `FlowID=(TaskID,FlowType,Epoch)`, conserved `total_data=e2e_delivered+e2e_remaining`, holder/hop state and RouteRevision; STEP 4.2C-C tensors them. Real traces cover direct Input/Return and two-hop Input; Return multi-hop, same-destination reroute and destination-change Epoch are contract fixtures. DepData has zero runtime instances and is not applicable to Planner v1.

**Historical STEP 6.2A boundary:** 4.2C-C destination-list semantics and 4.4 deterministic transition were aligned by STEP 6.2A-ROUTE-RECOVERY. Same-destination reroute writes the full rule-side path without changing learned action dimensions. Formal Dataset activation is zero for route length greater than one, so multi-hop/reroute `B_Tx` was repaired in code but not formally revalidated. Destination-change Epoch still requires a new Flow identity and remains unsupported by fixed object support. At that historical point STEP 6.2B remained closed; current acceptance is recorded below.

Planner-only causal task side-state contains current Task ID/slot, arrival/elapsed/deadline, absolute deadline, returned size and Return destination. Its deadline comes from a decision-before-action replay of one non-locked Formal Validation anchor with exact seed/config/action-prefix/current-state alignment. It is separate from Encoder/RSSM and training data. Deadline failure is Planner-derived, not a learned lifecycle output. AirFogSim's `requireReturn()` also checks the assigned compute host: a positive returned size does not force a Return Flow when computation occurs at the Return destination. The support layer must compare the predicted compute host with the known Return destination when computation finishes. A required Return with no current Return slot creates only a derived support boundary; no Flow is fabricated. If unsupported Return birth first appears at state `u`, `H_sup=u-1`; all candidates use `H_eff=min H_sup`, and zero means `OBJECTIVE_UNSCOREABLE`.

Computation uses `cpu_service = allocated_cpu_per_s * slot_duration_s` and `W_next=max(W_current-cpu_service,0)`. A computation-finished task is not final complete when a required Return is unresolved or blocked by fixed support. The normalized compute burden is `W_rem(q,h)/max(W_rem(q,0), epsilon)`, with zero initial remaining work contributing zero compute burden, not an invented positive value.

## Inactive or diagnostic dimensions

`J_Effort` uses only Comm, Comp and Mob: allocated RB fraction over current valid allocatable RB support; requested CPU over applicable observed raw static CPU capacity; and UAV speed over 15 m/s per present UAV. The 15 m/s value is the Formal Dataset mobility core profile maximum, not a physical limit. The applicable component mask and denominators are anchor-frozen and candidate-independent. `ROUTE_EFFORT_IN_OBJECTIVE_V1=false`: RouteRevision has no source-backed physical/resource cost mapping. Priority is inactive (`w_q=1`) and is not a readiness prerequisite. Energy and fairness are outside Planner v1. Risk is defined but inactive; primary rollout uses mean prior and expectation service.

Main final system throughput is **End-to-End Useful Throughput**: actual terminal-hop E2E delivered application bytes divided by actual simulation elapsed time, including wired and wireless. All-hop **Network Service Throughput** is diagnostic and can exceed useful throughput. Both PI-JWM and future baselines must use the same Flow/Outcome extractor. Throughput is not an independent weighted Planner term; rollout rates, hop bytes and E2E bytes remain separately labeled predicted diagnostics.

Only source-backed hard constraints are allowed. Future task schedules, Future Target, FAILED labels, and candidate-dependent denominators must not be used to fill missing causal state. Unsupported future Return birth is a model support boundary, not an illegal candidate. The anchor cohort excludes already terminal `DONE` and `FAILED` tasks; task presence, release, computation-finished, and final completion are distinct states.

**Historical blocker:** the 4.2C-C/4.4 route mismatch blocked the earlier 6.2A-PATCH readiness calculation. STEP 6.2A-ROUTE-RECOVERY repaired deterministic route semantics, and this closure freezes Planner v1 to Formal Dataset single-hop support. The existing formal best checkpoint is retained by researcher decision; retraining is false. The original `LVal=0.07431338784170399` remains a legacy accepted observation, not patched full validation; patched full validation was not executed.

**Historical 6.2A-CLOSURE readiness:** `STEP_6_2B_READINESS=READY_FOR_SINGLE_HOP_SCORER_IMPLEMENTATION`. This authorized only implementation of the objective scorer and lexicographic comparator with CPU contract tests. It did not mean Planner/MPC/closed-loop readiness, candidate method selection, ranking quality, multi-hop readiness, baseline readiness or performance readiness. `CLOSED_LOOP_READINESS=NOT_READY`; `CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`; `MULTIHOP_PLANNER_READINESS=NOT_IN_V1_DOMAIN`. At that closure no scorer, ranking or performance run had begun.

## STEP 6.2B implementation observation (2026-09-28)

The five-part scorer and strict comparator now execute on supplied fixed-support H1–H4 traces; the 6.2A readiness statement above remains a historical authorization gate. `STEP_6_2B=BLOCKED_ON_OBJECTIVE_SEMANTICS`, so this is not an accepted Planner objective. A pending/no-current-Flow single-hop Route currently passes 6.0C admission, but the formal adapter maps it to `flow_index=-1` and 4.4 creates no Flow. This conflicts with the v1 action-domain exclusion of unsupported Flow birth. The current Objective contract defines `H_sup` for future Return birth, not this pending Input Route. The scorer refuses to silently score that mapping pending a researcher decision. A same-path existing-Flow Route also updates the Task-Agent Host relation before hop completion while the route holder remains unchanged; that deterministic rule behavior requires source/contract reconciliation. The general objective tuple, ordering and formulas above have not changed. No candidate method, performance, baseline or closed-loop result follows from the bounded CPU scorer evidence.

## STEP 6.2B-PATCH current acceptance (2026-09-28)

The previous paragraph is a **historical blocked observation**. The researcher now freezes `PLANNER_V1_ROUTE_POLICY=EXPLICIT_NOOP_ONLY`: every horizon has an empty Route family. Candidate admission rejects all nonempty Route rows as `OUTSIDE_PLANNER_ROUTE_NOOP_ONLY_V1`, so both pending `flow_index=-1` and existing same-path Route are outside Planner v1 before rollout. The Route interface and 4.4 learned/deterministic Route behavior remain unchanged. On legal single-hop existing Flow state, `B_Tx=R_hop` remains the objective burden implication; the general multi-hop formula is retained for future research.

The five-part tuple `(N_DDL,A_DDL,J_Delay,J_Burden,J_Effort)`, strict lexicographic order, shared `H_eff`, deadline latch, anchor-frozen burden and Comm/Comp/Mob effort definitions remain unchanged. CPU tests and one non-locked frozen-checkpoint H1–H4 no-Route integration yield `STEP_6_2B=PASS`, `PLANNER_OBJECTIVE_SCORER=IMPLEMENTED_AND_CPU_CONTRACT_VERIFIED`, `MPC_OBJECTIVE=FROZEN_AND_IMPLEMENTED`. `CANDIDATE_METHOD_SELECTION=RESEARCH_PENDING`, `CLOSED_LOOP_READINESS=NOT_READY`, `MULTIHOP_PLANNER_READINESS=NOT_IN_V1_DOMAIN`, and `ROUTE_OPTIMIZATION_ACTIVE_V1=false`. No ranking-quality, performance or closed-loop conclusion follows.
