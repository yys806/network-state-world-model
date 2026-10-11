import json,unittest
from pathlib import Path
NS=Path('code/artifacts/audit/pi_jwm_step6_4j_live_root_capture_20261011_v4')
class LiveRootReceiptTests(unittest.TestCase):
 def test_live_roots_valid_and_r39_unproven(self):
  audit=json.loads((NS/'independent_audit.json').read_text()); self.assertTrue(audit['passed']); self.assertEqual(audit['r39_exact_equivalence'],'UNPROVEN'); self.assertEqual(audit['totals']['roots'],2)
 def test_h4_and_reason_counts_are_non_overlapping(self):
  audit=json.loads((NS/'independent_audit.json').read_text()); self.assertEqual(audit['totals']['completed_h4'],2); self.assertEqual(audit['totals']['scoreable_h4'],0); self.assertEqual(audit['totals']['scorer_exceptions'],0); self.assertEqual(audit['totals']['wm_transitions'],8)
 def test_future_return_birth_is_recorded_per_candidate(self):
  for p in NS.glob('*_final.json'):
   d=json.loads(p.read_text()); scored=[c for c in d['candidates'] if c.get('scorer_status')]; self.assertEqual(len(scored),1); self.assertIn('UNSUPPORTED_FUTURE_RETURN_BIRTH',str(scored[0]['diagnostics']))
if __name__=='__main__': unittest.main()
