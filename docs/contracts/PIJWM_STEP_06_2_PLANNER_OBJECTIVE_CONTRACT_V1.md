# PI-JWM Planner Objective Contract v1

**Status:** `RESEARCHER-FROZEN TARGET CONTRACT` / `SOURCE-RECONCILED IN STEP 6.2A-PATCH` / `PLANNER SIDE-STATE FOUNDATION AVAILABLE` / `SCORER IMPLEMENTATION NOT STARTED`

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

These are target formulas only. STEP 6.2A-PATCH executes no scoring. The selected Formal Validation anchor now has a causally aligned deadline sidecar. The 4.2C-B/C Flow contract supplies E2E remaining, but the frozen 4.4 cross-hop route-index rule conflicts with the current route array; full Tx burden remains blocked.

## Source boundaries

AirFogSim defines task deadline as relative allowed latency: `arrival_time + deadline`. Return completion accepts `delay <= deadline`; no-Return completion accepts `delay <= deadline + 1e-5`; the later active-task sweep fails only when `delay > deadline`. Completion handling precedes that sweep. Accepted Formal Raw has arrival and `required_returned_size`, but not deadline. A non-locked validation anchor now has an exact-aligned Planner-only deadline sidecar; deadline still does not enter the trained Tensor or 4.4 state. Priority remains outside Objective v1.

**Historical fact:** the pre-Ledger STEP 4.2B source audit could not reconstruct E2E remaining from Raw. **Current implementation fact:** STEP 4.2C-B introduced `FlowID=(TaskID,FlowType,Epoch)`, conserved `total_data=e2e_delivered+e2e_remaining`, holder/hop state and RouteRevision; STEP 4.2C-C tensors them. Real traces cover direct Input/Return and two-hop Input; Return multi-hop, same-destination reroute and destination-change Epoch are contract fixtures. DepData has zero runtime instances and is not applicable to Planner v1.

**Current blocker:** 4.2C-C stores `route_node_indices` as the remaining destinations, excluding the current holder. The frozen 4.4 intermediate-hop rule indexes it as though it included the holder. A deterministic rule test delivers a complete intermediate hop yet leaves `current_hop_index=0`, holder unchanged, `hop_remaining=0` and E2E remaining unchanged. A Route action updates current endpoints and RouteRevision, but not the full route array; its `hop_count` field is not consumed by the deterministic rule. The Planner-only route side-state carries the candidate route and rejects divergence, but cannot repair the trained model's future hop transition. Consequently multi-hop/reroute `B_Tx` and STEP 6.2B are blocked. Destination-change Epoch also requires a new Flow identity, which this model does not create.

Planner-only causal task side-state contains current Task ID/slot, arrival/elapsed/deadline, absolute deadline, returned size and Return destination. Its deadline comes from a decision-before-action replay of one non-locked Formal Validation anchor with exact seed/config/action-prefix/current-state alignment. It is separate from Encoder/RSSM and training data. Deadline failure is Planner-derived, not a learned lifecycle output. AirFogSim's `requireReturn()` also checks the assigned compute host: a positive returned size does not force a Return Flow when computation occurs at the Return destination. The support layer must compare the predicted compute host with the known Return destination when computation finishes. A required Return with no current Return slot creates only a derived support boundary; no Flow is fabricated. If unsupported Return birth first appears at state `u`, `H_sup=u-1`; all candidates use `H_eff=min H_sup`, and zero means `OBJECTIVE_UNSCOREABLE`.

Computation uses `cpu_service = allocated_cpu_per_s * slot_duration_s` and `W_next=max(W_current-cpu_service,0)`. A computation-finished task is not final complete when a required Return is unresolved or blocked by fixed support. The normalized compute burden is `W_rem(q,h)/max(W_rem(q,0), epsilon)`, with zero initial remaining work contributing zero compute burden, not an invented positive value.

## Inactive or diagnostic dimensions

`J_Effort` uses only Comm, Comp and Mob: allocated RB fraction over current valid allocatable RB support; requested CPU over applicable observed raw static CPU capacity; and UAV speed over 15 m/s per present UAV. The 15 m/s value is the Formal Dataset mobility core profile maximum, not a physical limit. The applicable component mask and denominators are anchor-frozen and candidate-independent. `ROUTE_EFFORT_IN_OBJECTIVE_V1=false`: RouteRevision has no source-backed physical/resource cost mapping. Priority is inactive (`w_q=1`) and is not a readiness prerequisite. Energy and fairness are outside Planner v1. Risk is defined but inactive; primary rollout uses mean prior and expectation service.

Main final system throughput is **End-to-End Useful Throughput**: actual terminal-hop E2E delivered application bytes divided by actual simulation elapsed time, including wired and wireless. All-hop **Network Service Throughput** is diagnostic and can exceed useful throughput. Both PI-JWM and future baselines must use the same Flow/Outcome extractor. Throughput is not an independent weighted Planner term; rollout rates, hop bytes and E2E bytes remain separately labeled predicted diagnostics.

Only source-backed hard constraints are allowed. Future task schedules, Future Target, FAILED labels, and candidate-dependent denominators must not be used to fill missing causal state. Unsupported future Return birth is a model support boundary, not an illegal candidate. The anchor cohort excludes already terminal `DONE` and `FAILED` tasks; task presence, release, computation-finished, and final completion are distinct states.

**Readiness:** `STEP_6_2B_READINESS=BLOCKED`. Deadline and Return source readiness are closed for the selected anchor; E2E Flow fields and the revised effort/throughput definitions are closed. The remaining critical blocker is the current 4.2C-C/4.4 route-array semantic mismatch, including stale arrays after Route action. Repair requires a separate researcher decision because it changes trained-model route semantics or the accepted Planner action domain. No scorer, ranking or performance run has begun.
