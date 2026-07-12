import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11GpuCandidateBatchSummaryTest(unittest.TestCase):
    def test_summarize_batch_extracts_val_and_matched_test_metrics(self):
        from summarize_v11_gpu_candidate_batch import summarize_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / '01_candidate'
            candidate.mkdir()
            (candidate / 'summary.json').write_text(
                json.dumps(
                    {
                        'mode': 'demo_mode',
                        'device': 'cuda',
                        'command': 'python demo.py',
                        'best_val': {
                            'candidate': 'demo_candidate',
                            'active_rate_rmse': 210.5,
                            'link_rmse': 88.0,
                            'activity_f1': 0.12,
                        },
                        'matched_test_for_best_val': {
                            'active_rate_rmse': 212.5,
                            'link_rmse': 89.0,
                            'activity_f1': 0.11,
                        },
                    }
                ),
                encoding='utf-8',
            )

            rows = summarize_batch(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['mode'], 'demo_mode')
        self.assertEqual(rows[0]['best_val_candidate'], 'demo_candidate')
        self.assertEqual(rows[0]['val_active_rate_rmse'], 210.5)
        self.assertEqual(rows[0]['test_link_rmse'], 89.0)


if __name__ == '__main__':
    unittest.main()
