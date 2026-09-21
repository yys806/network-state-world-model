# PI-JWM STEP 4.4 Structured RSSM World Model Contract v1

## 1. Scope

This contract maps frozen `Z_t^{PI,L_g}` and the four frozen future-action families to an untrained structured world-model rollout. It proves wiring, causal boundaries, rule execution, stochastic replay, dynamic graph rebuild, serialization, and CPU differentiation only. It does not prove prediction accuracy, calibration, planning quality, or performance.

Scope flags are fixed to `loss=false`, `optimizer=false`, `training=false`, `gpu=false`, `planner=false`, `candidate_generation=false`, `locked_test=false`, `formal_dataset=false`, and `performance_claim=false`.

## 2. Structured latent

The aligned recurrent state is

`h = {Physical, Agent, Communication, Flow, Task}`.

Only unknown dynamics receive stochastic state:

`z = {Physical vehicle motion, Communication CSI}`.

UAV motion is action/rule driven; fixed infrastructure has no motion stochastic state; Agent, Flow, and Task have no stochastic state. Each family has an independent initializer, state encoder, dynamics processor, and recurrent cell. No global pooling is used.

Physical and Communication prior/posterior distributions are diagonal Gaussians with explicit `mean`, clamped `log_std`, reparameterized sample API, posterior teacher-compatible initialization, and prior-only future rollout. `prior_mode=mean` is deterministic development evidence; `prior_mode=sample` requires an explicit generator.

## 3. Action and dynamics boundary

Route, Communication, Computation, and UAV Mobility actions retain the frozen tensor namespaces and are locally routed through validated stable slots. Invalid/absent/type-incompatible references are rejected; raw numeric IDs are not learned scalar features.

Future steps use independent `DynamicsGraphInteraction`, not the STEP 4.3B History encoder and not shared parameters. Physical, Comm, Flow, Task-Agent, DAG, P2A, and P2C processors consume the current predicted graph. Explicit state encoders feed the current predicted system state back into every recurrent transition.

Allowed learned heads are only:

- vehicle `delta_xyz` and next speed;
- wireless per-RB CSI/channel condition.

There is no learned acceleration, Physical-edge, rate, outage, service, Flow remaining/completion, Task progress/lifecycle, DAG, or presence head.

## 4. Communication transition

Wireless service follows:

`predicted CSI + complete RB allocation + audited channel-type power + derived interference + noise -> SINR_dB`;

`R_nominal = RB_bandwidth_MHz * log2(1 + 10^(SINR_dB/10))` in Mbps;

`p_out = 1 - exp(-outage_snr_threshold / max(SINR_dB, 1e-9))`;

`O ~ Bernoulli(p_out)` and `R_actual = (1-O) * R_nominal`.

The outage realization is a known stochastic transition, not a Decision input, future Target input, Communication latent, classifier, or residual head. `sample` mode requires an explicit `torch.Generator`; `expectation` mode uses `(1-p_out)R_nominal` and must report `expected_service_approximation=true`.

Wired capacity is read from the causal simulator configuration. Active membership/count is derived from current Flow Carrying `active + hop source/destination + wired relation` and is machine-compared with the real `WiredNetworkManager` membership. The rule is `min(capacity_mbps * 1e6 / 8 * slot_duration / N_active, remaining_bytes)`.

## 5. Deterministic transition and graph rebuild

Vehicle position integrates the learned delta; acceleration is derived from speed difference. UAV position uses the frozen mobility rule. Static nodes remain unchanged. Communication action, known outage event, and service rules produce delivered bytes. Intermediate-hop delivery does not reduce end-to-end remaining; only `HopDst == LogicalDst` does. CPU allocation updates work; Task progress/lifecycle and DAG satisfaction remain rules.

Each predicted state rebuilds Physical relations, Comm CSI/validity, Flow presence, Task state, static DAG identity, Align, and wireless-only GeoComm. The next recurrent step consumes this rebuilt graph and predicted state. Object support is fixed to the History/input namespace; future-only Target objects are never created or read.

## 6. Evidence and provenance

### PATCH2 transition closure

Hop delivery is `min(raw_service, hop_remaining, flow_remaining)`. `hop_progress` accumulates delivered bytes and `hop_remaining` is reduced; completion is defined by the next remaining value reaching zero. Intermediate hops preserve end-to-end remaining, advance holder/segment, and reset the next hop progress. Terminal hops alone reduce logical Flow remaining and synchronize `presence=false`, `carrying_active=false`, and `status=COMPLETED`.

After natural advancement or route action, active Flow carrying endpoints are rebound to a communication relation by source/target plus relation presence/validity; no relation yields `-1`. `flow_route_revision` is structural provenance only and is not a learned continuous input. Flow type and status are typed embeddings in the Flow state representation. Task final completion is gated by Return Flow semantics; DAG identity remains static while `task_completed`, `task_released`, and `dag_satisfied` are dynamic transition state.

An existing Return Flow is bound only by the typed structural pair `(task_index, flow_type_index=Return)` over the fixed current History/input support. Input Flow and another Task's Return Flow cannot satisfy this gate. The model declares `future_return_birth_supported=false`: it does not read Future Target, create a new Flow slot, or use a future route to fabricate a Return Flow. If current side state says a Task requires Return but the current support has no matching Return Flow, computation completion leaves the frozen lifecycle unchanged and sets `return_birth_required=true` plus `final_completion_blocked_by_fixed_support=true`; it does not set `task_completed=true`. Definition 05 must mask, exclude, or explicitly classify windows that cross this unsupported birth boundary rather than score them as ordinary prediction error.

### PATCH3 final closure semantics

The frozen current-side Tensor/Graph contract does not expose `Task.return_size`; therefore slot absence cannot prove that no Return is required. The adapter represents the three states explicitly with `task_return_requirement_known` and `task_requires_return`: an existing typed Return establishes `known=true, requires_return=true`; otherwise the canonical real adapter uses `known=false` and treats the `requires_return` value as non-authoritative. A separately supplied current-side semantic may represent `known=true, requires_return=false`. If computation finishes while the requirement is unknown and no Return slot exists, `final_completion_unresolved_by_return_requirement=true` and `final_completion_blocked_by_fixed_support=true`, while `task_completed=false` and `return_birth_required=false`. This differs from the known-required/no-slot case, which sets `return_birth_required=true`.

DAG release uses only valid incoming edges. A root task is released; one incomplete valid predecessor blocks release; every valid predecessor must complete before release; invalid edges never block. `dag_satisfied` is recomputed from the current predecessor completion state. Terminal logical Flow completion atomically synchronizes `flow_remaining=0`, `flow_presence=false`, `carrying_active=false`, and `flow_status_index=FLOW_STATUS_VOCAB[COMPLETED]`. Partial and intermediate-hop completion keep the non-completed status, and intermediate service does not reduce end-to-end remaining.

Canonical artifact: `code/artifacts/protocols/pi_jwm_step4_4_structured_rssm_world_model_v1_20260921/`.

The receipt ANDs all 50 original checks, 21 communication-event checks, and 21 structural/rule-transition checks. PATCH3 semantic counterfactuals exercise the canonical real adapter's unknown Return state, typed Return identity and completion gate, Future-Target support isolation, valid/invalid/multiple-predecessor DAG transitions, partial/intermediate/terminal Flow completion, exact RouteRevision change-only behavior, effective type/status embeddings, and raw identity/revision exclusion from learned Flow features. A recursive rollout uses a contract-valid negative-index no-op after a Flow completes; invalid or absent action references remain rejected. Source hashes and symbols point to `ChannelManagerCP.computeRate`, `_get_power_db`, `rayleigh_outage_prob`, `WiredNetworkManager.step`, and the active AirFogSim example configuration. Development topology `radius_knn/radius=1000m/k=2` remains `development_only=true` and `research_frozen=false`.
