import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/tests')]
import test_step6_4i_episode_v1 as fixtures
from pi_jwm.step6_4i_episode_runner_v1 import run_episode
class RunnerTests(unittest.TestCase):
 def test_feedback_two_cycles_and_existing_directory_refuses_duplicate_episode(self):
  f=fixtures.EpisodeTests();f.setUp();current=dict(f.decision)
  def observe():return dict(current)
  def capture():
   current.update(frame_index=current['frame_index']+1,simulation_time_s=f.env.simulation_time,capture_event_id='real'+str(current['frame_index']+1));return dict(current)
  def plan(decision):
   # Real environment causality is separately exercised by the AirFogSim smoke.
   from dataclasses import replace
   f.packet=replace(f.packet,domain=replace(f.domain,context=replace(f.domain.context,history=({'frame_index':decision['frame_index'],'simulation_time_s':decision['simulation_time_s']},))))
   return f.packet
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'episode'
   r=run_episode(episode_id='e',output=out,env=f.env,initial_observation=observe(),runtime_tasks=lambda:f.tasks,planner=plan,installer=f.installer,step=f.step,capture=capture,decision_steps=2,engineering_only=True)
   self.assertEqual(r['executed_decisions'],2);self.assertEqual(r['scheduled_decisions'],2)
   with self.assertRaises(FileExistsError):run_episode(episode_id='e',output=out,env=f.env,initial_observation=observe(),runtime_tasks=lambda:f.tasks,planner=plan,installer=f.installer,step=f.step,capture=capture,decision_steps=2,engineering_only=True)
 def test_formal_unapproved_denied_before_output_or_step(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(PermissionError):run_episode(episode_id='e',output=Path(tmp)/'e',env=None,initial_observation={},runtime_tasks=None,planner=None,installer=None,step=None,capture=None,decision_steps=64,engineering_only=False)
 def test_initial_metric_failure_is_preserved_without_step(self):
  f=fixtures.EpisodeTests();f.setUp()
  class BrokenLedger:
   def observe(self,*args):raise ValueError('real task getter mismatch')
   def report(self,**kwargs):return {}
  with tempfile.TemporaryDirectory() as tmp:
   output=Path(tmp)/'e'
   r=run_episode(episode_id='e',output=output,env=f.env,initial_observation=f.decision,runtime_tasks=lambda:f.tasks,planner=lambda _:f.packet,installer=f.installer,step=f.step,capture=f.capture,decision_steps=2,engineering_only=True,task_ledger=BrokenLedger())
   self.assertEqual(r['status'],'STOPPED');self.assertEqual(f.env.simulation_time,0.)
   self.assertTrue((output/'receipt.json').exists())
 def test_metric_report_failure_counts_termination_after_real_step(self):
  f=fixtures.EpisodeTests();f.setUp()
  class BrokenReport:
   def observe(self,*args):pass
   def report(self,**kwargs):raise ValueError('metric aggregation inconsistency')
  original_capture=f.capture;f.capture=lambda:{**original_capture(),'tasks':[]}
  with tempfile.TemporaryDirectory() as tmp:
   r=run_episode(episode_id='e',output=Path(tmp)/'e',env=f.env,initial_observation=f.decision,runtime_tasks=lambda:f.tasks,planner=lambda _:f.packet,installer=f.installer,step=f.step,capture=f.capture,decision_steps=1,engineering_only=True,task_ledger=BrokenReport())
   self.assertEqual(r['status'],'STOPPED');self.assertEqual(r['reason'],'METRIC_INCONSISTENCY')
   self.assertEqual(r['termination_C_count'],1);self.assertEqual(r['executed_decisions'],1)
   self.assertEqual(f.env.simulation_time,.1)
if __name__=='__main__':unittest.main()
