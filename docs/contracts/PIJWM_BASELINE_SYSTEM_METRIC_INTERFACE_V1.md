# PI-JWM Baseline System Metric Interface v1

**Status:** `INTERFACE RECORDED`; no baseline method is selected or executed.

Future PI-JWM and baseline systems must share the same simulator scenario, task generation, random-seed protocol, decision interval, action execution semantics, episode horizon, warm-up rule, failure/deadline semantics, and metric definitions.

## Metric families

The final closed-loop report must keep these distinct from Planner scores: Task Completion Rate, Deadline Violation Rate, Throughput, Average Task Delay, P95 Task Delay, P99 Task Delay, Energy Consumption, Resource Utilization, and Fairness/Jain's Index when its service population is explicitly defined.

Planner Objective is not the final Evaluation Metric. Predicted metric is not real closed-loop metric. Evaluation Metric is not an acceptance gate. Each report must include absolute values, relative improvement, and cross-seed statistics; no improvement threshold is frozen here.

For higher-is-better metrics, `Delta_rel=(PI-JWM-Baseline)/Baseline*100%`. For lower-is-better metrics, `Delta_rel=(Baseline-PI-JWM)/Baseline*100%`. Positive means PI-JWM improvement. Zero baselines require an explicit denominator policy before reporting.

If a baseline value is zero, relative improvement is `METRIC_SEMANTICS_PENDING`; report absolute values and do not calculate a percentage until the denominator policy has been separately specified. Reports require absolute metrics, relative improvement where defined, and cross-seed statistics. No improvement threshold is frozen here.

Throughput must state whether it counts network-carried service bytes or end-to-end useful bytes. The current audit leaves this distinction `METRIC_SEMANTICS_PENDING` for multi-hop flows. Energy and fairness require their own source/definition audits before use in a Planner objective.

No baseline, locked-test, GPU, closed-loop, or performance experiment is authorized by this interface document.
