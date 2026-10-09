"""Fault-injection tests for one-shot real-step orchestration, not performance."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/tests')]
from pi_jwm.step6_4i_episode_v1 import EpisodeController, PlanPacket
from pi_jwm.step6_4b_live_bridge_v1 import SmokeFailure
import test_step6_4e_fallback_v1 as previous
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionSequence,Backend

class EpisodeTests(unittest.TestCase):
 def setUp(self):
  self.domain,self.decision,self.tasks=previous.FallbackTests().inputs()
  self.decision.update(capture_event_id='real0')
  self.action=next(self.domain.iter_bound()).action
  self.result=SimpleNamespace(winner_first_action=self.action,winner_sequence=CandidateActionSequence((self.action,self.action,self.action,self.action),'winner',Backend.RULE_FALLBACK,6301,self.domain.context.causal_provenance),best_fingerprint='winner',best_objective=(0,0,0,0,0),budget_receipt={'N_unique_transition_evals':64},h4_scoreable_count=1)
  self.packet=PlanPacket(self.result,self.domain,{'state':'real-root'},'predicted-h1',0.1)
  self.env=SimpleNamespace(simulation_time=0.,simulation_interval=.1)
  self.calls=[]
  self.driver=EpisodeController(self.calls.append)
 def installer(self,env,commands):self.calls.append({'installed':commands})
 def step(self):self.env.simulation_time+=.1
 def capture(self):return {'capture_event_id':'real1','simulation_time_s':self.env.simulation_time}
 def run_cycle(self,planner=None,installer=None,step=None):
  return self.driver.cycle(self.env,self.decision,self.tasks,planner or (lambda:self.packet),installer or self.installer,step or self.step,self.capture)
 def test_winner_only_first_action_and_no_repeat(self):
  r=self.run_cycle();self.assertEqual(r['status'],'EXECUTED');self.assertEqual(r['executed_horizon'],1)
  self.assertEqual(r['action'],self.action.frame());self.assertEqual(self.env.simulation_time,.1)
  r=self.run_cycle();self.assertEqual(r['reason'],'DUPLICATE_DECISION');self.assertEqual(self.env.simulation_time,.1)
 def test_no_scoreable_uses_one_A_and_empty_terminates(self):
  self.result.winner_first_action=None;self.result.winner_sequence=None
  r=self.run_cycle();self.assertEqual(r['dispatch'],'FALLBACK_A');self.assertEqual(self.env.simulation_time,.1)
  self.setUp()
  def empty():raise SmokeFailure('DOMAIN_EMPTY')
  r=self.run_cycle(planner=empty);self.assertEqual(r['dispatch'],'FAIL_CLOSED_C');self.assertEqual(self.env.simulation_time,0.)
 def test_missing_bridge_setter_and_environment_failure_no_retry(self):
  for reason in ('REQUIRED_LIVE_OBSERVATION_MISSING','ACTION_BRIDGE_REJECTED','SETTER_FAILURE'):
   self.setUp()
   def fail():raise SmokeFailure(reason)
   r=self.run_cycle(planner=fail);self.assertEqual(r['reason'],reason);self.assertEqual(self.env.simulation_time,0.)
   self.assertEqual(self.run_cycle()['reason'],'EPISODE_ALREADY_STOPPED')
  self.setUp()
  def bad_setter(env,commands):raise SmokeFailure('SETTER_FAILURE','partial buffer mutation')
  r=self.run_cycle(installer=bad_setter);self.assertTrue(r['setter_partial_mutation_risk']);self.assertEqual(self.env.simulation_time,0.)
  self.setUp()
  def bad_step():self.env.simulation_time+=.1;raise RuntimeError('after mutation')
  r=self.run_cycle(step=bad_step);self.assertEqual(r['reason'],'ENV_STEP_FAILURE');self.assertEqual(self.env.simulation_time,.1)
  self.assertTrue(r['environment_step_attempted']);self.assertEqual(self.run_cycle()['reason'],'EPISODE_ALREADY_STOPPED')
 def test_simulator_motion_during_planning_is_rejected(self):
  def drift():self.env.simulation_time+=.1;return self.packet
  r=self.run_cycle(planner=drift);self.assertEqual(r['reason'],'SIMULATION_ADVANCED_DURING_PLANNING')
  self.assertFalse(any('installed' in row for row in self.calls))
 def test_feedback_inconsistency_stops(self):
  self.capture=lambda:{'capture_event_id':'real0','simulation_time_s':0.}
  r=self.run_cycle();self.assertEqual(r['reason'],'REAL_FEEDBACK_INCONSISTENCY');self.assertEqual(self.env.simulation_time,.1)
if __name__=='__main__':unittest.main()
