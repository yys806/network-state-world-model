'''Summarize a PI-JWM v11 candidate GPU batch output directory.'''

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _row_from_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    best_val = data.get('best_val') or {}
    matched_test = data.get('matched_test_for_best_val') or {}
    return {
        'output_dir': str(path.parent),
        'mode': data.get('mode'),
        'device': data.get('device'),
        'best_val_candidate': best_val.get('candidate'),
        'val_active_rate_rmse': _safe_float(best_val.get('active_rate_rmse')),
        'val_link_rmse': _safe_float(best_val.get('link_rmse')),
        'val_activity_f1': _safe_float(best_val.get('activity_f1')),
        'test_active_rate_rmse': _safe_float(matched_test.get('active_rate_rmse')),
        'test_link_rmse': _safe_float(matched_test.get('link_rmse')),
        'test_activity_f1': _safe_float(matched_test.get('activity_f1')),
        'command': data.get('command'),
    }


def summarize_batch(batch_dir: Path) -> list[dict[str, Any]]:
    summaries = sorted(Path(batch_dir).glob('*/summary.json'))
    return [_row_from_summary(path) for path in summaries]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'output_dir',
        'mode',
        'device',
        'best_val_candidate',
        'val_active_rate_rmse',
        'val_link_rmse',
        'val_activity_f1',
        'test_active_rate_rmse',
        'test_link_rmse',
        'test_activity_f1',
        'command',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, default=None)
    parser.add_argument('--output-json', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_batch(args.batch_dir)
    output_csv = args.output_csv or Path(args.batch_dir) / 'candidate_metric_summary.csv'
    output_json = args.output_json or Path(args.batch_dir) / 'candidate_metric_summary.json'
    write_csv(output_csv, rows)
    output_json.write_text(json.dumps({'batch_dir': str(args.batch_dir), 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'batch_dir': str(args.batch_dir), 'count': len(rows), 'output_csv': str(output_csv), 'output_json': str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
