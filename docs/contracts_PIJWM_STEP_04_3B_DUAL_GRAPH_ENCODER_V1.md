# PI-JWM STEP 4.3B Dual-Graph Encoder Contract v1

## Scope

This contract maps the frozen History tensor and STEP 4.3A current typed graph to the entity/relation-aligned same-time representation `Z_t^{PI,L_g}`. It is an untrained encoder contract, not `xi_t^Lat`, a world model, a dynamics transition, a prediction head, or a planner.

Definition basis: read-only `D:\shen\OB\科研\PIJWM\03物理-信息双图建模.md`, SHA-256 `6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e`, especially §5.1–§5.4 and §6.1.

## Inputs and normalization

- Physical Node, Agent, Task, and Flow temporal encoders read only frozen History. Future Target and Future Action are not consumed.
- Current Physical/Comm/Task-Agent/DAG relations, Flow logical endpoints, Align, GeoComm, presence, validity, and structural references come from STEP 4.3A without changing its schema.
- Raw continuous values are normalized with fixed `dev_train` statistics before encoding. Forward never fits batch statistics.
- STEP 3.2/4.2A/4.2C-C statistics are reused. Only STEP 4.3A-derived Physical relation features receive additive, mask-aware, `dev_train`-only statistics.
- Every continuous encoder receives both `mask * normalized_value` and the explicit mask. Missing is therefore distinguishable from a real numeric zero.
- Unavailable optional inputs remain `UPSTREAM_FIELD_NOT_AVAILABLE`; no value is fabricated.

## Typed initial encoding and temporal state

Each object/relation family has an independent `Linear → SiLU → Linear → LayerNorm` encoder with a common output width `d_h`. Physical Node, Agent, Task, Physical relation, Comm relation, Logical Flow, Carrying, Flow fusion, Task-Agent, and DAG parameters are not shared.

Physical, Agent, Task, and Flow use four independent object-wise GRUs over stable slots. Hidden state starts at zero and updates only when `presence=true`; absent frames keep the preceding hidden state. Physical relations and Comm relations have no temporal GRU in v1.

Logical Flow learns only `total_data` and `e2e_remaining` plus masks/type. `e2e_delivered`, Epoch, Flow Index, and raw object indices are excluded from numeric features. Carrying independently encodes hop progress, hop remaining, masks, and active status. Logical and Carrying latents are fused before the single Flow GRU; Carrying remains structural side state and does not create a second relation latent.

## Typed directed graph refinement

- Five relation processors are independent: Physical, Comm, Flow, Task-Agent, and DAG.
- Semantic graph direction is unchanged. Comm, Flow, and Task-Agent add explicit reverse computational messages with a direction embedding. DAG is forward-only.
- The current STEP 4.3A Physical topology contains both directed rows for each admitted pair; the encoder consumes each row once and does not synthesize duplicates.
- Incoming messages use masked mean. Agent and Task aggregate each relation family separately and fuse family messages with family-valid masks.
- Physical Node, Agent, and Task have independent node-update MLPs. Node and relation updates use residual plus LayerNorm.
- Graph layers refine the representation at the same decision time; they do not perform `t→t+1` dynamics or mutate the system graph.

## Cross-domain coupling

- P2A exists only on valid Align rows and uses a vector sigmoid gate.
- P2C exists only on valid wireless GeoComm rows and uses a vector sigmoid gate. Wired and invalid rows receive zero P2C.
- P2A value and gate processors both consume `[h_phy, h_agent]` with independent parameters. P2C value and gate processors both consume `[h_src, h_dst, r_comm]` with independent parameters; a joint-context gate with a physical-only value path is not compliant.
- P2C requires valid endpoint Physical representations but does not require a matching Physical relation row.
- There is no Information→Physical path and no Task↔Physical or Flow↔PhysicalEdge shortcut.

## Configuration and output

All dimensions, `L_g`, coupling/reverse toggles, aggregation/update policies, and initialization seed are explicit. The development evidence uses `d_h=16`, hidden width `24`, embedding widths `4`, and `L_g=2`; these values are `development_only=true`, `research_frozen=false`.

The output contains aligned Physical node/relation latents; Information Agent/Task/Comm/Flow/Task-Agent/DAG latents; and an immutable structural side interface copied from all eleven STEP 4.3A blocks: node indices/presence, relation endpoints/indices/presence/validity, Align, GeoComm, and Flow Carrying state. Structural equality is checked against the source graph and is not appended to learned numeric inputs. Communication CSI width is read from tensor-contract `n_comm_rb`, never from a source-code constant.

## Evidence boundary

The artifact class is `UNTRAINED_DEVELOPMENT_ENCODER_EVIDENCE`. It proves wiring, causality, masking, direction, coupling, permutation behavior, serialization, determinism, and CPU differentiability. It does not establish learned representation quality, prediction accuracy, performance improvement, or a trained model.

Scope: `training=false`, `optimizer=false`, `gpu=false`, `locked_test=false`, `formal_dataset=false`.
