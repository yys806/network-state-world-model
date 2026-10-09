import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4j_pilot_v1 import select_episodes,PilotClock,engineering_acceptance,validate_spent,validate_resume
from run_step6_4j_pilot_v1 import reconstruct_history_action
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
 def test_engineering_acceptance_rejects_zero_steps_fail_closed_and_duplicate_roots(self):
  good=[{'status':'COMPLETED','decisions':2,'root_ids':['a','b'],'actual_action_history_aligned':True,'failures':[]} for _ in range(2)]
  self.assertTrue(engineering_acceptance(good))
  for mutate in (
   lambda x:x.update(decisions=0),
   lambda x:x.update(status='FAILED',failures=[{'reason':'FAIL_CLOSED_C'}]),
   lambda x:x.update(root_ids=['same','same']),
   lambda x:x.update(actual_action_history_aligned=False),
  ):
   bad=[dict(row) for row in good];mutate(bad[0]);self.assertFalse(engineering_acceptance(bad))
 def test_history_reconstructs_nonempty_comm_comp_and_mobility_ids(self):
  current={'entities':[{'entity_id':'UAV_0'},{'entity_id':'RSU_0'}], 'tasks':[{'task_id':'Task_7'}]}
  action={'comm':{'entries':[{'task_index':0,'relation_index':2,'rb_indices':[1,3]}]},'comp':{'entries':[{'task_index':0,'node_index':1,'allocated_cpu_per_s':1.5}]},'mobility':{'entries':[{'uav_index':1,'azimuth_rad':.1,'elevation_rad':.2,'speed_mps':3.}]},'route':{'entries':[]}}
  row=reconstruct_history_action(action,current)
  self.assertEqual(row['comm']['entries'][0]['task_id'],'Task_7');self.assertEqual(row['comp']['entries'][0]['node_id'],'UAV_0');self.assertEqual(row['mobility']['entries'][0]['uav_id'],'UAV_0')
  self.assertEqual(row['comm']['entries'][0]['rb_indices'],[1,3]);self.assertEqual(row['comp']['entries'][0]['allocated_cpu_per_s'],1.5)
if __name__=='__main__':unittest.main()
