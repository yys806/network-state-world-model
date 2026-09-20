# PI-JWM STEP 4.3A Typed Dual-Graph Builder Contract v1

## 1. Scope and authority

This contract maps the frozen STEP 4.2C-C Tensor to the current typed dual graph

`G_t^PI = (G_t^Phy, G_t^Info, R_t^PI)`.

The research authority is the read-only `03物理-信息双图建模.md`, SHA-256 `6f3e17b0691aa81c60c2ea1e76a1031d2f7f622abf4b40884e9ae4380953c15e`, especially Physical semantics/edges (lines 88–147), Information nodes/relations (151–338), and Align/GeoComm boundaries (548–650). The implementation source is `step4_3a_typed_dual_graph_builder_v1.py`.

The builder contains no encoder, MLP, GRU, message passing, reverse computational path, P2A/P2C processor, latent fusion, world model, loss, planner, or training.

## 2. Current-time and identity policy

- Current graph input is exactly frozen History `history[-1]`; target tensors are never consumed.
- Stable entity/task/Flow indices are retained from the Tensor. Row deletion is not used to encode feature missingness.
- `presence=false`, `feature_mask=false`, and a real numeric zero remain distinct.
- Optional sequence materialization and temporal aggregation are not implemented in this Step.

## 3. Typed blocks

| Block | Source | Semantics |
| --- | --- | --- |
| `physical_nodes` | current entity position and motion | spatial existence only |
| `physical_relations` | current physical nodes plus explicit topology config | directed spatial relation only |
| `agent_nodes` | current entities and CPU static capability | Information Agent; CPU is not available CPU |
| `task_nodes` | frozen current task state | demand/progress/time/lifecycle |
| `comm_relations` | current typed communication rows | Agent→Agent; validity independent of CSI mask |
| `task_agent_relations` | current Src/Host/Exec/Ret rows | Task→Agent |
| `flow_relations` | current logical Flow rows | logical Agent→Agent stateful multiedge |
| `dag_relations` | current DAG rows | predecessor Task→dependent Task |
| `flow_carrying_state` | current Carrying rows | Flow side state, not a second relation |
| `align_relations` | shared stable entity index | Physical↔same Information Agent |
| `geo_comm_relations` | wireless Comm endpoints plus Physical membership | structural spatial dependency; does not require a matching Physical edge |

Flow relation features retain total, E2E delivered, and E2E remaining verbatim. E2E delivered is redundant provenance in this builder, not a recomputed state. Return size, priority, and deadline remain `AVAILABLE_UPSTREAM_BUT_NOT_IN_FROZEN_TENSOR`.

## 4. Physical policy

A Physical Node is materialized only when the current entity is present and all three frozen position coordinates are valid. The policy intentionally does not hard-code an edge/cloud class decision. Minimum features are `[x,y,z,speed,canonical_acceleration]` with masks.

Physical relations are derived only from those current spatial states. Features are `[delta_x,delta_y,delta_z,distance,relative_speed,relative_acceleration]`. No CSI, rate, RB allocation, Task, Flow, CPU, or service state is permitted.

The topology contract supports `radius`, `knn`, and `radius_knn`, with explicit radius, k, and self-loop values. The acceptance artifact uses `radius_knn/radius=1000 m/k=2/self_loop=false` only as a deterministic development setting: `development_only=true`, `research_frozen=false`.

## 5. Information and cross-domain policy

- Comm rows preserve structural presence/validity when CSI is missing; wired/no-CSI is legal.
- Logical Flow endpoints are the frozen logical source/destination. Carrying holder/hop/route never overwrites them. Parallel Flow slots remain independent.
- DAG never creates a DepData Flow; runtime DepData remains zero in current evidence.
- Align is identity-only. GeoComm is valid only for a wireless Comm relation whose two endpoints have valid Physical representations; wired or missing-spatial endpoints remain explicit invalid rows.
- No Task↔Physical or Flow↔PhysicalEdge shortcut is created.

## 6. Machine acceptance

The receipt evaluates every required check by logical AND and includes twenty named negative/counterfactual fixtures. It covers exact Tensor mapping, semantic separation, topology independence, Comm/Flow independence, logical multi-hop Flow, parallel Flow, Align/GeoComm, masks, future-target isolation, deterministic rebuild, overflow/truncation rejection, serialize/load, and scope. Artifact evidence is development CPU/non-locked evidence, not a formal Dataset or model result.
