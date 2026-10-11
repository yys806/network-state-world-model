import json,unittest
from pathlib import Path
NS=Path('code/artifacts/audit/pi_jwm_step6_4j_h1_return_action_sensitivity_20261011_v1')
class H1ReturnActionAuditTests(unittest.TestCase):
 def setUp(self): self.audit=json.loads((NS/'independent_audit.json').read_text())
 def test_budget_and_counts(self):
  t=self.audit['totals']; self.assertTrue(self.audit['passed']); self.assertEqual(t['legal_distinct_actions'],18); self.assertEqual(t['successful_h1_predictions'],18); self.assertEqual(t['wm_transitions'],18); self.assertEqual(t['return_birth_triggers'],18)
 def test_no_scorer_or_rollout_exception(self):
  t=self.audit['totals']; self.assertEqual(t['scorer_exceptions'],0); self.assertEqual(t['rollout_exceptions'],0); self.assertEqual(t['h1_support_pass'],0); self.assertEqual(t['h1_scoreable'],0)
 def test_categories_are_state_b(self):
  for row in self.audit['rows']: self.assertTrue(all('B' in str(x) for x in row['target_classifications']))
if __name__=='__main__': unittest.main()
