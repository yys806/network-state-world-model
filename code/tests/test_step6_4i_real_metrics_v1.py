import sys,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/src'))
from pi_jwm.step6_4i_real_metrics_v1 import RealTaskLedger
class MetricsTests(unittest.TestCase):
 def task(self,finish):return SimpleNamespace(getTaskArrivalTime=lambda:0.,getTaskDeadline=lambda:2.,getLastOperationTime=lambda:finish,getReturnedSize=lambda:3.,isFinished=lambda:True)
 def test_real_completion_end_to_end_credit_not_hop_bytes_or_prediction(self):
  ledger=RealTaskLedger();ledger.observe([{'task_id':'T','lifecycle':'computing'}],{'T':self.task(-1)},.5)
  ledger.observe([{'task_id':'T','lifecycle':'completed'}],{'T':self.task(1.)},1.)
  ledger.observe([{'task_id':'T','lifecycle':'completed'}],{'T':self.task(1.)},1.1)
  r=ledger.report(planned_duration_s=2.,actual_duration_s=1.)
  self.assertEqual(r['initial_cohort_count'],1);self.assertEqual(r['completed_count'],1);self.assertEqual(r['on_time_count'],1)
  self.assertEqual(r['useful_result_data_units'],3.);self.assertEqual(r['useful_result_units_per_planned_second'],1.5)
  self.assertEqual(r['completion_delays_s'],[1.]);self.assertEqual(r['pending_count'],0)
 def test_early_stop_keeps_denominator_and_birth_cohort_separate(self):
  l=RealTaskLedger();t=self.task(-1)
  l.observe([{'task_id':'T','lifecycle':'computing'}],{'T':t},.5)
  l.observe([{'task_id':'T','lifecycle':'failed'},{'task_id':'new','lifecycle':'offloading'}],{'T':t,'new':t},.6)
  r=l.report(planned_duration_s=2.,actual_duration_s=.1)
  self.assertEqual(r['initial_cohort_count'],1);self.assertEqual(r['completion_rate'],0.)
  self.assertEqual(r['failed_count'],1);self.assertEqual(r['observed_new_task_count'],1)
 def test_terminal_tasks_present_initially_are_not_new_births(self):
  l=RealTaskLedger();t=self.task(-1)
  l.observe([{'task_id':'T','lifecycle':'computing'},{'task_id':'oldfailed','lifecycle':'failed'}],{'T':t,'oldfailed':t},.5)
  r=l.report(planned_duration_s=1.,actual_duration_s=.1)
  self.assertEqual(r['observed_new_task_count'],0)
if __name__=='__main__':unittest.main()
