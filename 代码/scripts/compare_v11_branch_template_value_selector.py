'''CPU-first branch/template value selector triage for PI-JWM v11 candidate.

This script does not launch a new expensive training job.  It consolidates
existing v11 candidate CSVs, re-ranks candidates with one validation objective,
and reports whether the next controlled experiment should focus on support
ranking, RB/CPU magnitude calibration, high-load repair, or GPU confirmation.
'''

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'artifacts/experiments/pi_jwm_v11_branch_template_value_selector_cpu_20260628'


@dataclass(frozen=True)
class ObjectiveConfig:
    link_budget: float = 90.0
    link_weight: float = 1.0
    high_load_weight: float = 0.25
    f1_floor: float = 0.024
    f1_weight: float = 500.0
    ood_weight: float = 0.02


def _safe_float(value, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == '' or text.lower() in {'nan', 'none', 'null'}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _finite_or(value: float, default: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(default)


def objective_score(row: dict, config: ObjectiveConfig | None = None) -> float:
    '''Lower is better; active RMSE is primary, constraints add penalties.'''

    cfg = config or ObjectiveConfig()
    active = _finite_or(_safe_float(row.get('active_rate_rmse')), 1e9)
    link = _safe_float(row.get('link_rmse'))
    f1 = _safe_float(row.get('activity_f1'), default=cfg.f1_floor)
    high_load = _safe_float(row.get('high_load_active_rate_rmse'))
    ood = _safe_float(row.get('action_ood_distance'), default=0.0)

    score = active
    if math.isfinite(link):
        score += cfg.link_weight * max(0.0, link - cfg.link_budget)
    if math.isfinite(high_load):
        score += cfg.high_load_weight * max(0.0, high_load - active)
    if math.isfinite(f1):
        score += cfg.f1_weight * max(0.0, cfg.f1_floor - f1)
    if math.isfinite(ood):
        score += cfg.ood_weight * max(0.0, ood)
    return float(score)


def _is_diagnostic_candidate(row: dict) -> bool:
    candidate = str(row.get('candidate', ''))
    experiment = str(row.get('experiment', ''))
    source_csv = str(row.get('source_csv', ''))
    support_model = str(row.get('support_model', ''))
    selector = str(row.get('selector', ''))
    family = str(row.get('family', ''))
    diagnostic_text = ' '.join([candidate, experiment, source_csv]).lower()
    return (
        'diagnostic_only' in candidate
        or 'oracle' in candidate
        or 'true_value' in candidate
        or 'diagnostic' in diagnostic_text
        or 'oracle_value_scope' in diagnostic_text
        or support_model == 'diagnostic_only'
        or selector == 'diagnostic_only'
        or family == 'diagnostic_only'
    )


def is_diagnostic_row(row: dict) -> bool:
    return _is_diagnostic_candidate(row)


def is_smoke_row(row: dict) -> bool:
    text = ' '.join([str(row.get('experiment', '')), str(row.get('source_csv', ''))]).lower()
    return '_smoke' in text or 'smoke_' in text


def _has_finite_active_metric(row: dict, min_active_count: int = 0) -> bool:
    if not math.isfinite(_safe_float(row.get('active_rate_rmse'))):
        return False
    if int(min_active_count) > 0:
        active_count = _safe_float(row.get('active_count'))
        if not math.isfinite(active_count) or active_count < int(min_active_count):
            return False
    return True


def _candidate_key(row: dict) -> str:
    return str(row.get('candidate', '')).strip()


def select_matched_test_by_validation(
    rows: Iterable[dict],
    config: ObjectiveConfig | None = None,
    include_diagnostic: bool = True,
    min_val_active_count: int = 0,
    min_test_active_count: int = 0,
) -> dict:
    cfg = config or ObjectiveConfig()
    paired: dict[str, dict[str, dict]] = {}
    for row in rows:
        if not include_diagnostic and _is_diagnostic_candidate(row):
            continue
        candidate = _candidate_key(row)
        split = str(row.get('split', '')).strip().lower()
        if not candidate or split not in {'val', 'test'}:
            continue
        paired.setdefault(candidate, {})[split] = row

    best: tuple[float, str, dict] | None = None
    for candidate, split_rows in paired.items():
        val = split_rows.get('val')
        test = split_rows.get('test')
        if val is None or test is None:
            continue
        if not _has_finite_active_metric(val, int(min_val_active_count)):
            continue
        if not _has_finite_active_metric(test, int(min_test_active_count)):
            continue
        score = objective_score(val, cfg)
        if best is None or score < best[0]:
            best = (score, candidate, split_rows)

    if best is None:
        raise ValueError('no validation rows available for matched selection')

    score, _candidate, split_rows = best
    val_row = dict(split_rows['val'])
    test_row = dict(split_rows.get('test', {}))
    val_row['objective_score'] = score
    if test_row:
        test_row['objective_score'] = objective_score(test_row, cfg)
    return {'val': val_row, 'test': test_row}


def oracle_gap_summary(
    learned_active_rmse: float,
    oracle_active_rmse: float,
    autonomous_reference: float,
    target_rmse: float,
) -> dict:
    learned = float(learned_active_rmse)
    oracle = float(oracle_active_rmse)
    target = float(target_rmse)
    reference = float(autonomous_reference)
    return {
        'learned_active_rmse': learned,
        'oracle_active_rmse': oracle,
        'autonomous_reference': reference,
        'target_rmse': target,
        'learned_to_oracle_gap': learned - oracle,
        'needed_improvement_to_target': learned - target,
        'improvement_vs_autonomous_reference': reference - learned,
        'oracle_has_sub200_potential': oracle < target,
    }


def recommend_next_action(
    matched_test_active_rmse: float,
    matched_test_link_rmse: float,
    oracle_active_rmse: float,
    target_rmse: float = 200.0,
    ranked_reference_rmse: float = 213.16087389710646,
    autonomous_reference_rmse: float = 217.237962,
    link_budget: float = 90.0,
) -> dict:
    active = float(matched_test_active_rmse)
    link = float(matched_test_link_rmse)
    oracle = float(oracle_active_rmse)
    target = float(target_rmse)

    if active < target and link <= link_budget:
        return {
            'decision': 'gpu_ready_sub200_confirm',
            'reason': 'validation-selected matched test is already below target with acceptable link RMSE',
        }
    if active < ranked_reference_rmse and link <= link_budget:
        return {
            'decision': 'gpu_ready_ranked_reference_beat',
            'reason': 'candidate beats ranked allocation reference and should be confirmed with larger GPU run',
        }
    if oracle < target and active > ranked_reference_rmse:
        return {
            'decision': 'continue_cpu_support_value_research',
            'reason': 'oracle gap is large; focus on support/value ranking before GPU',
        }
    if active < autonomous_reference_rmse:
        return {
            'decision': 'continue_cpu_scale_and_validation_gate',
            'reason': 'candidate improves autonomous reference but does not beat ranked allocation reference yet',
        }
    return {
        'decision': 'method_not_ready_research_new_candidate',
        'reason': 'matched test does not improve current autonomous reference enough for GPU',
    }


def discover_csvs(input_csvs: list[str], input_globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in input_csvs:
        path = Path(item)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            paths.append(path)
    for pattern in input_globs:
        base_pattern = pattern
        root = PROJECT_ROOT
        if ':' in pattern[:3]:
            matched = [Path(p) for p in Path().glob(pattern)]
        else:
            matched = list(root.glob(base_pattern))
        paths.extend(path for path in matched if path.is_file())

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def read_candidate_rows(csv_paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for csv_path in csv_paths:
        try:
            with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if 'candidate' not in row or 'split' not in row or 'active_rate_rmse' not in row:
                        continue
                    item = dict(row)
                    item['source_csv'] = str(csv_path.relative_to(PROJECT_ROOT))
                    item['experiment'] = csv_path.parent.name
                    rows.append(item)
        except UnicodeDecodeError:
            with csv_path.open('r', encoding='gbk', newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if 'candidate' not in row or 'split' not in row or 'active_rate_rmse' not in row:
                        continue
                    item = dict(row)
                    item['source_csv'] = str(csv_path.relative_to(PROJECT_ROOT))
                    item['experiment'] = csv_path.parent.name
                    rows.append(item)
    return rows


def ranked_validation_pairs(
    rows: list[dict],
    config: ObjectiveConfig,
    include_diagnostic: bool,
    min_val_active_count: int = 0,
    min_test_active_count: int = 0,
) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        if not include_diagnostic and _is_diagnostic_candidate(row):
            continue
        key = (str(row.get('source_csv', '')), _candidate_key(row))
        split = str(row.get('split', '')).lower()
        if split in {'val', 'test'}:
            grouped.setdefault(key, {})[split] = row

    ranked: list[dict] = []
    for (source_csv, candidate), split_rows in grouped.items():
        val = split_rows.get('val')
        test = split_rows.get('test')
        if val is None:
            continue
        if test is None or not _has_finite_active_metric(test, min_test_active_count):
            continue
        if not _has_finite_active_metric(val, min_val_active_count):
            continue
        ranked.append(
            {
                'source_csv': source_csv,
                'experiment': val.get('experiment', ''),
                'candidate': candidate,
                'is_diagnostic': _is_diagnostic_candidate(val),
                'val_objective_score': objective_score(val, config),
                'val_active_rate_rmse': _safe_float(val.get('active_rate_rmse')),
                'test_active_rate_rmse': _safe_float(test.get('active_rate_rmse')),
                'val_link_rmse': _safe_float(val.get('link_rmse')),
                'test_link_rmse': _safe_float(test.get('link_rmse')),
                'val_activity_f1': _safe_float(val.get('activity_f1')),
                'test_activity_f1': _safe_float(test.get('activity_f1')),
                'family': val.get('family', ''),
                'support_model': val.get('support_model', ''),
                'selection_score_mode': val.get('selection_score_mode', ''),
                'selection_group_mode': val.get('selection_group_mode', ''),
                'top_k': val.get('top_k', ''),
                'alpha': val.get('alpha', ''),
                'step_total_cap_scale': val.get('step_total_cap_scale', ''),
                'edge_value_cap_scale': val.get('edge_value_cap_scale', ''),
                'new_edge_value_cap': val.get('new_edge_value_cap', ''),
            }
        )
    ranked.sort(key=lambda item: (item['val_objective_score'], _finite_or(item['test_active_rate_rmse'], 1e9)))
    return ranked


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    config = ObjectiveConfig(
        link_budget=float(args.link_budget),
        link_weight=float(args.link_weight),
        high_load_weight=float(args.high_load_weight),
        f1_floor=float(args.f1_floor),
        f1_weight=float(args.f1_weight),
        ood_weight=float(args.ood_weight),
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_csvs(args.input_csv, args.input_glob)
    rows = read_candidate_rows(csv_paths)
    if not rows:
        raise RuntimeError('no candidate rows found')
    if not bool(args.include_smoke):
        rows = [row for row in rows if not is_smoke_row(row)]

    learned_ranked = ranked_validation_pairs(
        rows,
        config,
        include_diagnostic=False,
        min_val_active_count=int(args.min_val_active_count),
        min_test_active_count=int(args.min_test_active_count),
    )
    all_ranked = ranked_validation_pairs(
        rows,
        config,
        include_diagnostic=True,
        min_val_active_count=int(args.min_val_active_count),
        min_test_active_count=int(args.min_test_active_count),
    )
    if not learned_ranked:
        raise RuntimeError('no learned validation/test candidate pairs found')

    learned_rows = [
        row
        for row in rows
        if not _is_diagnostic_candidate(row)
    ]
    learned_selected = select_matched_test_by_validation(
        learned_rows,
        config,
        include_diagnostic=False,
        min_val_active_count=int(args.min_val_active_count),
        min_test_active_count=int(args.min_test_active_count),
    )
    all_selected = select_matched_test_by_validation(
        rows,
        config,
        include_diagnostic=True,
        min_val_active_count=int(args.min_val_active_count),
        min_test_active_count=int(args.min_test_active_count),
    )

    learned_test_active = _safe_float(learned_selected['test'].get('active_rate_rmse'))
    learned_test_link = _safe_float(learned_selected['test'].get('link_rmse'))
    gap = oracle_gap_summary(
        learned_active_rmse=learned_test_active,
        oracle_active_rmse=float(args.oracle_active_rmse),
        autonomous_reference=float(args.autonomous_reference_rmse),
        target_rmse=float(args.target_rmse),
    )
    recommendation = recommend_next_action(
        matched_test_active_rmse=learned_test_active,
        matched_test_link_rmse=learned_test_link,
        oracle_active_rmse=float(args.oracle_active_rmse),
        target_rmse=float(args.target_rmse),
        ranked_reference_rmse=float(args.ranked_reference_rmse),
        autonomous_reference_rmse=float(args.autonomous_reference_rmse),
        link_budget=float(args.link_budget),
    )

    ranked_path = output_dir / 'branch_template_value_ranked.csv'
    write_csv(ranked_path, learned_ranked[: int(args.max_ranked_rows)])
    all_ranked_path = output_dir / 'branch_template_value_ranked_with_diagnostics.csv'
    write_csv(all_ranked_path, all_ranked[: int(args.max_ranked_rows)])

    summary = {
        'framework': 'PI-JWM',
        'candidate': 'v11',
        'mode': 'branch_template_value_selector_cpu_triage',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'output_dir': str(output_dir.relative_to(PROJECT_ROOT)),
        'command': ' '.join(str(part) for part in ['scripts/compare_v11_branch_template_value_selector.py']),
        'input_csv_count': len(csv_paths),
        'candidate_row_count': len(rows),
        'learned_pair_count': len(learned_ranked),
        'diagnostic_pair_count': len(all_ranked) - len(learned_ranked),
        'objective_config': config.__dict__,
        'oracle_active_rmse': float(args.oracle_active_rmse),
        'target_rmse': float(args.target_rmse),
        'autonomous_reference_rmse': float(args.autonomous_reference_rmse),
        'ranked_reference_rmse': float(args.ranked_reference_rmse),
        'best_learned_by_validation': learned_selected,
        'best_all_by_validation': all_selected,
        'best_learned_ranked_top10': learned_ranked[:10],
        'oracle_gap_summary': gap,
        'recommendation': recommendation,
        'ranked_csv': str(ranked_path.relative_to(PROJECT_ROOT)),
        'ranked_with_diagnostics_csv': str(all_ranked_path.relative_to(PROJECT_ROOT)),
    }
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    next_steps_path = output_dir / 'next_steps.md'
    next_steps_path.write_text(
        '\n'.join(
            [
                '# PI-JWM v11 Candidate Branch/Template Triage',
                '',
                f"- Learned matched-test active RMSE: `{learned_test_active}`",
                f"- Learned matched-test link RMSE: `{learned_test_link}`",
                f"- Oracle active RMSE used for gap: `{float(args.oracle_active_rmse)}`",
                f"- Decision: `{recommendation['decision']}`",
                f"- Reason: {recommendation['reason']}",
                '',
                '## Suggested Next CPU Command',
                '',
                '```powershell',
                'python scripts\\compare_v11_graph_support_generator.py --output-dir artifacts\\experiments\\pi_jwm_v11_branch_support_value_focus_1024_20260628 --device cpu --batch-size 32 --max-train-samples 1024 --max-val-samples 512 --max-test-samples 512 --limit-after-stats --streaming-stats --stats-chunk-size 512 --support-model-kinds hgb rank_hgb --support-training-strategies hard_negative --rf-trees 80 --support-max-train-rows 200000 --support-negative-ratio 40 --hard-negative-fraction 0.85 --rank-target-mode value_gain_norm --top-k 4 8 16 32 --selection-score-modes support support_value support_gain --selection-group-modes baseline_active_count support_threshold --support-thresholds 0.01 0.02 0.05 --blend-alpha 0.95 1.0 --step-total-cap-scale 1.02 1.05 1.1 --edge-value-cap-scale 1.15 1.25 --new-edge-value-cap 1.0 2.0',
                '```',
                '',
            ]
        ),
        encoding='utf-8',
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        '--input-glob',
        action='append',
        default=['artifacts/experiments/pi_jwm_v11_*/*.csv'],
        help='Project-root relative glob for candidate CSV files.',
    )
    parser.add_argument('--input-csv', action='append', default=[])
    parser.add_argument('--target-rmse', type=float, default=200.0)
    parser.add_argument('--oracle-active-rmse', type=float, default=122.09518244236105)
    parser.add_argument('--autonomous-reference-rmse', type=float, default=217.237962)
    parser.add_argument('--ranked-reference-rmse', type=float, default=213.16087389710646)
    parser.add_argument('--link-budget', type=float, default=90.0)
    parser.add_argument('--link-weight', type=float, default=1.0)
    parser.add_argument('--high-load-weight', type=float, default=0.25)
    parser.add_argument('--f1-floor', type=float, default=0.024)
    parser.add_argument('--f1-weight', type=float, default=500.0)
    parser.add_argument('--ood-weight', type=float, default=0.02)
    parser.add_argument('--max-ranked-rows', type=int, default=200)
    parser.add_argument('--include-smoke', action='store_true')
    parser.add_argument('--min-val-active-count', type=int, default=50)
    parser.add_argument('--min-test-active-count', type=int, default=50)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        'output_dir': summary['output_dir'],
        'learned_pair_count': summary['learned_pair_count'],
        'best_learned_test_active_rate_rmse': summary['best_learned_by_validation']['test'].get('active_rate_rmse'),
        'best_learned_test_link_rmse': summary['best_learned_by_validation']['test'].get('link_rmse'),
        'decision': summary['recommendation']['decision'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
