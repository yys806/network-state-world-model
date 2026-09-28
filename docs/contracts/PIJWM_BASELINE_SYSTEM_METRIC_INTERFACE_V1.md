# PI-JWM Baseline System Metric Interface v1

**Status:** `INTERFACE RECORDED`; no baseline method is selected or executed.

Future PI-JWM and baseline systems must share the same simulator scenario, task generation, random-seed protocol, decision interval, action execution semantics, episode horizon, warm-up rule, failure/deadline semantics, and metric definitions.

## Metric families

The final closed-loop report must keep these distinct from Planner scores: Task Completion Rate, Deadline Violation Rate, Throughput, Average Task Delay, P95 Task Delay, P99 Task Delay, Energy Consumption, Resource Utilization, and Fairness/Jain's Index when its service population is explicitly defined.

Planner Objective is not the final Evaluation Metric. Predicted metric is not real closed-loop metric. Evaluation Metric is not an acceptance gate. Each report must include absolute values, relative improvement, and cross-seed statistics; no improvement threshold is frozen here.

For higher-is-better metrics, `Delta_rel=(PI-JWM-Baseline)/Baseline*100%`. For lower-is-better metrics, `Delta_rel=(Baseline-PI-JWM)/Baseline*100%`. Positive means PI-JWM improvement. Zero baselines require an explicit denominator policy before reporting.

If a baseline value is zero, relative improvement is `METRIC_SEMANTICS_PENDING`; report absolute values and do not calculate a percentage until the denominator policy has been separately specified. Reports require absolute metrics, relative improvement where defined, and cross-seed statistics. No improvement threshold is frozen here.

Main Throughput is `END_TO_END_USEFUL_THROUGHPUT`: the sum of real terminal-hop logical E2E application bytes delivered to the Flow destination, divided by real elapsed simulation time. Wired and wireless terminal deliveries both count. The Causal Flow Ledger's decrease in `e2e_remaining` defines the useful bytes, so a simulator event exceeding remaining data is capped by the conserved Flow state.

Diagnostic Throughput is `NETWORK_SERVICE_THROUGHPUT`: all real successful hop-carried bytes divided by the same elapsed time, including intermediate hops. It measures network service activity and can exceed useful throughput. For a three-hop delivery of one MB, service can be about three MB while useful delivery is one MB. PI-JWM and every future baseline must use the same Flow/Outcome-level extractor, `pi_jwm.step6_2a_throughput_metric_v1.extract_real_throughput`, with the same time window and Flow identities. Predicted rollout versions must be labeled proxies and never reported as real closed-loop metrics.

Energy and fairness retain separate final-metric boundaries. This interface does not activate them in Planner Objective v1.

No baseline, locked-test, GPU, closed-loop, or performance experiment is authorized by this interface document.
