# PI-JWM Planner Objective Contract v1

**Status:** `RESEARCHER-FROZEN TARGET CONTRACT` / `SOURCE-AUDITED IN STEP 6.2A` / `IMPLEMENTATION NOT STARTED`

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

These are target formulas only. STEP 6.2A executes no scoring. Deadline scoring remains blocked until the causal side-state is exposed; Tx burden is only partially source-supported; Route effort has no accepted candidate-independent normalized denominator.

## Source boundaries

AirFogSim defines task deadline as relative allowed latency: `arrival_time + deadline`. Return completion accepts `delay <= deadline`; no-Return completion accepts `delay <= deadline + 1e-5`; the later active-task sweep fails only when `delay > deadline`. Completion handling precedes that sweep. The runtime source is real, but deadline/priority are absent from accepted Formal Raw and neither deadline nor elapsed time enters the trained Tensor/4.4 Planner state. Arrival and elapsed fields available upstream still require explicit aligned Planner side-state. Therefore deadline objective readiness is `BLOCKED` until an additive causal side-state contract is supplied.

The World Model rule decreases `hop_remaining` for intermediate service without decreasing `flow_remaining`; terminal service decreases end-to-end `flow_remaining`. The target `B_Tx = R_hop + (N_hop - 1 - i) * R_e2e` is only `PARTIALLY_SUPPORTED` because end-to-end remaining across completed hops, full route identity/revision provenance, and dependency-data transfer are not universally source-backed.

Computation uses `cpu_service = allocated_cpu_per_s * slot_duration_s` and `W_next=max(W_current-cpu_service,0)`. A computation-finished task is not final complete when a required Return is unresolved or blocked by fixed support. The normalized compute burden is `W_rem(q,h)/max(W_rem(q,0), epsilon)`, with zero initial remaining work contributing zero compute burden, not an invented positive value.

## Inactive or diagnostic dimensions

Throughput and delivered data remain diagnostics and final closed-loop metrics; they are not a second weighted Planner objective. Energy and fairness are final metric boundaries, not Planner v1 terms. Priority is present in the simulator but v1 uses `w_q=1`. Risk is defined but inactive: primary rollout uses mean prior and expectation service; latent variance is not risk, and no chance or tail guarantee is implied.

Only source-backed hard constraints are allowed. Future task schedules, Future Target, FAILED labels, and candidate-dependent denominators must not be used to fill missing causal state. Unsupported future Return birth is a model support boundary, not an illegal candidate. The anchor cohort excludes already terminal `DONE` and `FAILED` tasks; task presence, release, computation-finished, and final completion are distinct states.

**Readiness:** `STEP_6_2B_READINESS=BLOCKED`. Minimal prerequisites are additive Planner-only causal exposure for deadline/arrival/priority/return support, source-backed end-to-end Flow remaining, and a candidate-independent Route effort denominator.
