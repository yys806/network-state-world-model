# PI-JWM STEP 4.4 Communication Service Sufficiency Audit v1

## Scope

This is the mandatory pre-implementation gate for the Structured RSSM World Model. It audits whether actual communication service can be recovered from causal state and action. It does not implement the World Model, a learned service residual, Loss, Planner, or Training.

Definition basis: read-only `D:\shen\OB\科研\PIJWM\04世界模型预测边界与当前模型.md`, SHA-256 recorded in the implementation record.

## Wireless dependency result

The simulator nominal rule is recoverable from channel attenuation including fast fading, all active RB allocations, channel-type power, interference, noise, and RB bandwidth. It computes `bandwidth * log2(1 + SINR)`.

Actual rate is not uniquely recoverable from those causal inputs. `ChannelManagerCP.computeRate` samples a per-RB outage with `random.rand` and then sets sampled-outage rate to zero. The frozen Decision snapshot contains CSI, but not the future outage realization. PI-JWM records outage only after runtime channel computation as `outcome_only_not_same_frame_decision_input`.

Therefore outcome-only outage must not be promoted to an input, and nominal rate must not be mislabeled actual service.

## Wired dependency result

Wired service is `min(capacity_per_slot / active_flow_count, remaining)`. Link capacity and active flow membership exist in `WiredNetworkManager`, but are not exposed by the frozen Raw/Tensor contract. This is a minimal additive-state gap, not the reason for the residual verdict.

## Machine verdict

`SERVICE_RESIDUAL_RESEARCH_DECISION_REQUIRED`

The verdict is derived from the dependency matrix. If any required actual-service factor is `UNAVAILABLE_UNOBSERVABLE`, the residual decision gate dominates additive-state gaps. Tampering the verdict or promoting outcome-only outage to a causal input fails validation.

## Stop boundary

No residual target or architecture is selected. The researcher must choose whether the random outage/effective-service uncertainty belongs to predicted communication condition, a separate stochastic service event, or a learned residual, and freeze the corresponding target and interface before STEP 4.4 implementation resumes.

Scope remains `training=false`, `optimizer=false`, `loss=false`, `gpu=false`, `planner=false`, `candidate_generation=false`, `locked_test=false`, `formal_dataset=false`, and `performance_claim=false`.
