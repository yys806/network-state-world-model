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

Canonical artifact: `code/artifacts/protocols/pi_jwm_step4_4_structured_rssm_world_model_v1_20260921/`.

The receipt ANDs all 50 original checks plus 21 communication-event checks. Source hashes and symbols point to `ChannelManagerCP.computeRate`, `_get_power_db`, `rayleigh_outage_prob`, `WiredNetworkManager.step`, and the active AirFogSim example configuration. Development topology `radius_knn/radius=1000m/k=2` remains `development_only=true` and `research_frozen=false`.
