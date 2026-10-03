"""Full-cohort budget diagnostics, never algorithm selection or budget freeze."""
import math
from pi_jwm.step6_3d_method_selection_v1 import paired_outcome, cluster_bootstrap_advantage
from pi_jwm.step6_4c_budget_calibration_v1 import compare_pairs

SEEDS = (6311, 6312, 6313, 6314, 6315)
COMPONENTS = ('N_DDL', 'A_DDL', 'J_Delay', 'J_Burden', 'J_Effort')


def plan_cases(anchors, reused):
    if len(anchors) != 64 or len(set(anchors)) != 64:
        raise ValueError('exact frozen 64 unique anchors required')
    expected = {(a, s) for a in anchors for s in SEEDS}
    if not set(reused).issubset(expected):
        raise ValueError('reused identity outside frozen cohort/seeds')
    return [(a, s) for a in anchors for s in SEEDS if (a, s) not in reused]


def require_same_sources(historical, current):
    for path, digest in historical.items():
        if current.get(path) != digest:
            raise ValueError('scientific/source reuse mismatch: ' + path)


def component_analysis(left, right):
    if set(left) != set(right): raise ValueError('incomplete paired objectives')
    counts = {n: {'B512_better': 0, 'B1024_better': 0} for n in COMPONENTS}
    equal = both = 0
    for key in sorted(left):
        a, b = left[key], right[key]
        for x in (a, b):
            if x is not None and (len(x) != 5 or not all(math.isfinite(v) for v in x)):
                raise ValueError('invalid objective')
        if a is None or b is None: continue
        both += 1
        outcome = paired_outcome(tuple(a), tuple(b))
        if outcome == 0: equal += 1; continue
        index = next(i for i in range(5) if a[i] != b[i])
        counts[COMPONENTS[index]]['B512_better' if outcome > 0 else 'B1024_better'] += 1
    assert equal + sum(sum(v.values()) for v in counts.values()) == both
    return {'components': counts, 'all_equal': equal, 'both_scoreable_count': both,
            'PRIMARY_OBJECTIVE_DEGRADATION_COUNT': sum(counts[n]['B1024_better'] for n in COMPONENTS[:3]),
            'BURDEN_EFFORT_ONLY_DEGRADATION_COUNT': sum(counts[n]['B1024_better'] for n in COMPONENTS[3:])}


def full_pair_analysis(left, right):
    pairs = compare_pairs(left, right, expected_count=320)
    clusters = {a: [r['outcome'] for r in rows] for a, rows in pairs['per_anchor_seed_outcomes'].items()}
    if len(clusters) != 64 or any([r['seed'] for r in rows] != list(SEEDS)
                                for rows in pairs['per_anchor_seed_outcomes'].values()):
        raise ValueError('64 anchor clusters with frozen five paired seeds required')
    pairs['statistical_scope'] = 'Full frozen Validation cohort budget comparison; no equivalence or closed-loop claim'
    pairs['cluster_bootstrap'] = cluster_bootstrap_advantage(clusters, seed=6316, replications=10000)
    return pairs, component_analysis(left, right)
