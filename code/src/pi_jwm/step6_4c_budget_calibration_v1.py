"""Outcome-blind cohort and descriptive budget accounting, not method selection."""
from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from pi_jwm.step6_3d_method_selection_v1 import paired_outcome


def select_cohort(rows, count=16):
    groups = defaultdict(list)
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or count <= 0 or len(rows) < count:
        raise ValueError('duplicate IDs or insufficient static population')
    for row in rows:
        sid = row['sample_id']
        groups[row['stratum']].append({k: row[k] for k in
            ('sample_id', 'stratum', 'concrete_count', 'eligible_compute_tasks')})
    labels = sorted(groups)
    selected = []
    for j, label in enumerate(labels):
        quota = count // len(labels) + int(j < count % len(labels))
        if len(groups[label]) < quota:
            raise ValueError('stratum cannot meet preregistered balanced quota')
        ordered = sorted(groups[label], key=lambda r:
                         (hashlib.sha256(r['sample_id'].encode()).hexdigest(), r['sample_id']))
        for row in ordered[:quota]:
            selected.append(dict(row, sample_id_sha256=hashlib.sha256(
                row['sample_id'].encode()).hexdigest()))
    return selected


def validate_identity(result, expected):
    for key, value in expected.items():
        if key not in result or result[key] != value:
            raise ValueError('identity mismatch: ' + key)


def compare_pairs(left, right, *, expected_count=48):
    if set(left) != set(right) or len(left) != expected_count:
        raise ValueError('missing/extra paired case')
    bins = dict.fromkeys(('only_left_scoreable', 'only_right_scoreable',
        'both_scoreable_left_better', 'both_scoreable_right_better',
        'both_scoreable_tie', 'both_unscoreable'), 0)
    anchors = defaultdict(list)
    outcomes = []
    for (anchor, seed) in sorted(left):
        a, b = left[(anchor, seed)], right[(anchor, seed)]
        v = paired_outcome(None if a is None else tuple(a), None if b is None else tuple(b))
        category = ('both_unscoreable' if a is None and b is None else
                    'only_right_scoreable' if a is None else
                    'only_left_scoreable' if b is None else
                    'both_scoreable_left_better' if v > 0 else
                    'both_scoreable_right_better' if v < 0 else 'both_scoreable_tie')
        bins[category] += 1
        outcomes.append(v)
        anchors[anchor].append({'seed': seed, 'outcome': v})
    assert sum(bins.values()) == expected_count
    return {'six_categories': bins, 'total_paired_cases': expected_count,
            'win': outcomes.count(1), 'tie': outcomes.count(0), 'loss': outcomes.count(-1),
            'per_anchor_seed_outcomes': dict(anchors),
            'statistical_scope': 'paired descriptive CALIBRATION SUBSET; no non-inferiority claim'}


def summarize(rows):
    if not rows:
        raise ValueError('empty summary')
    totals = defaultdict(int)
    times = []
    for row in rows:
        o = row['outcome']; b = o['budget_receipt']
        totals['scoreable_cases'] += int(o['best_objective'] is not None)
        for dest, source in [('complete_h4', 'complete_sequence_count'),
                             ('distinct_scoreable_h4', 'h4_scoreable_count'),
                             ('unscoreable_h4', 'h4_unscoreable_count')]:
            totals[dest] += o[source]
        for key in ('N_unique_transition_evals', 'N_cache_hits', 'N_proposed_steps',
                    'N_admitted_steps', 'N_rejected_steps', 'N_dead_end_branches'):
            totals[key] += b[key]
        totals['scorer_exceptions'] += row['support_horizon_counts'].get('SCORER_EXCEPTION', 0)
        totals['return_birth_boundary'] += sum(n for k, n in row['score_residuals'].items()
            if k.startswith('H4_SUPPORT_BOUNDARY:') and 'UNSUPPORTED_FUTURE_RETURN_BIRTH' in k)
        t = b['wall_clock_seconds_diagnostic_only']
        if not math.isfinite(t) or t <= 0:
            raise ValueError('invalid runtime')
        times.append(t)
    times.sort()
    def quantile(p):
        position = (len(times) - 1) * p
        lower = int(position); upper = min(lower + 1, len(times) - 1)
        return times[lower] + (position - lower) * (times[upper] - times[lower])
    return {'case_count': len(rows), **totals,
            'scoreable_rate': totals['scoreable_cases'] / len(rows),
            'runtime_seconds': {'median': quantile(.5), 'P90': quantile(.9),
                'P95': quantile(.95), 'max': max(times), 'mean': sum(times)/len(times),
                'sum_in_solve': sum(times)},
            'actual_transitions_per_second': totals['N_unique_transition_evals']/sum(times),
            'runtime_scope': 'in-solve only; matrix/setup overhead recorded separately'}
