import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4j_pilot_v1 import select_episodes,PilotClock,validate_spent,validate_resume
class PilotTests(unittest.TestCase):
 def test_static_selection_ignores_outcomes_and_keeps_earliest(self):
  rows=[{'sample_id':f'{t}::anchor-{f:04d}','objective':object()} for t in ('a','b','c') for f in (8,3)]
  a=select_episodes(rows);b=select_episodes(list(reversed(rows)))
  self.assertEqual(a,b);self.assertEqual(len(a),2);self.assertTrue(all(x.endswith('0003') for x in a))
 def test_safe_boundary_and_incomplete_budget_rejected(self):
  clock=[0.];c=PilotClock(lambda:clock[0]);c.before_plan();clock[0]=10200
  with self.assertRaisesRegex(RuntimeError,'PILOT_TIME_LIMIT'):c.before_plan()
  validate_spent({'B_WM':512,'N_unique_transition_evals':512})
  with self.assertRaisesRegex(RuntimeError,'BUDGET'):validate_spent({'B_WM':512,'N_unique_transition_evals':511})
 def test_resume_identity_then_refuses_interrupted_or_duplicate(self):
  current={'execution_config_id':'new','checkpoint':'same','source':'frozen'}
  with self.assertRaisesRegex(RuntimeError,'IDENTITY'):validate_resume(current,{'execution_config_id':'other'},False)
  with self.assertRaisesRegex(RuntimeError,'NO_MID'):validate_resume(current,current,False)
  self.assertEqual(validate_resume(current,current,True),'READ_ONLY_COMPLETE')
if __name__=='__main__':unittest.main()
