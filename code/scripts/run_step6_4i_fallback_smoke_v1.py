"""CPU-only real one-step fallback A with injected no-winner status; no WM search."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,gzip,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/reference/AirFogSim')]
import run_step6_4e_comm_fallback_audit_v1 as prior
from pi_jwm.step6_4i_episode_v1 import EpisodeController,PlanPacket
from pi_jwm.step6_4i_real_metrics_v1 import RealTaskLedger
from pi_jwm.step6_4b_live_bridge_v1 import apply_commands
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2'
class EpisodeStopped(BaseException):pass
def write(n,d):
 p=OUT/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def freeze():
 p=ROOT/'code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/01_fixture_and_execution_freeze.json'
 f=json.loads(p.read_text());assert prior.sha(ROOT/f['raw_path'])==f['raw_sha256']
 if (OUT/'06_fallback_fixture.json').exists():raise SystemExit('Existing freeze')
 f={k:f[k] for k in ['sample_id','raw_path','raw_sha256','simulator_seed','policy_seed','anchor']}
 f.update(parent_sha256=prior.sha(p),source_sha256={x:prior.sha(ROOT/x) for x in ['code/scripts/run_step6_4i_fallback_smoke_v1.py','code/src/pi_jwm/step6_4i_episode_v1.py','code/src/pi_jwm/step6_4i_real_metrics_v1.py']},
  method='FROZEN_FALLBACK_A_NO_SEARCH',reason_injection='NO_SCOREABLE_H4; not observed S-CEM search failure',selection='Reuse previously frozen6.4F causal TRAIN anchor0041; no new outcome selection',WM_forwards=0,GPU='NOT_USED',locked_test=False)
 f['execution_config_id']=prior.digest(f);write('06_fallback_fixture.json',f)
def execute():
 if (OUT/'07_fallback_attempt.json').exists():raise SystemExit('No duplicate or retry')
 f=json.loads((OUT/'06_fallback_fixture.json').read_text());assert all(prior.sha(ROOT/p)==v for p,v in f['source_sha256'].items())
 write('07_fallback_attempt.json',{'started':True,'execution_config_id':f['execution_config_id'],'GPU':'NOT_USED'})
 raw=json.loads(gzip.decompress((ROOT/f['raw_path']).read_bytes()));prefix={**raw,'decisions':raw['decisions'][:f['anchor']+1],'steps':raw['steps'][:f['anchor']]}
 expected,domain=prior.current_domain(prefix)
 import collect_step5_5_formal_raw_v1 as collector
 from airfogsim.scheduler.computation_sched import ComputationScheduler
 receipt={'verdict':'BLOCKED','reason_injection':'NO_SCOREABLE_H4','WM_forward_count':0,'environment_steps':0,'GPU':'NOT_USED','locked_test':False}
 def hook(env,decisions,steps,config,communication):
  if decisions[-1]['frame_index']!=f['anchor']:return
  try:
   sample,domain=prior.current_domain({'environment':{'seed':f['simulator_seed'],'wired_edges':collector.WIRED_EDGES},'decisions':decisions,'steps':steps})
   assert prior.digest(sample)==prior.digest(expected)
   tasks=collector.step23._all_runtime_tasks(env);current=decisions[-1]
   ledger=RealTaskLedger();ledger.observe(current['tasks'],tasks,float(env.simulation_time));start=float(env.simulation_time)
   outcome=SimpleNamespace(winner_first_action=None,winner_sequence=None,best_fingerprint=None,h4_scoreable_count=0,budget_receipt={'N_unique_transition_evals':0,'N_cache_hits':0})
   journal=[]
   def log(r):journal.append(r);write('08_fallback_journal.json',journal)
   driver=EpisodeController(log)
   def capture():
    fresh=collector.step23._capture(env,frame=f['anchor']+1,phase='loop_start_decision',event_index=2*(f['anchor']+1),previous_speed_by_entity={e['entity_id']:float(e['speed_mps']) for e in current['entities']},delta_t_s=float(env.simulation_time)-start)
    return fresh
   row=driver.cycle(env,current,tasks,lambda:PlanPacket(outcome,domain,{'state':prior.digest(sample)},None,0.),
    lambda e,c:apply_commands(e,c,communication,ComputationScheduler,collector.step23.TrafficScheduler),env.step,capture)
   assert row['dispatch']=='FALLBACK_A' and row['status']=='EXECUTED'
   ledger.observe(row['fresh_observation']['tasks'],collector.step23._all_runtime_tasks(env),float(env.simulation_time))
   receipt.update(verdict='PASS',environment_steps=1,execution=row,real_task_metric_getters_validated=ledger.report(planned_duration_s=.1,actual_duration_s=float(env.simulation_time)-start),
    future_target_read=False,behavior_overwrite=False,normalization_refit=False,performance_claim=False)
  except Exception as exc:receipt.update(blocker=getattr(exc,'reason',type(exc).__name__),detail=str(exc))
  finally:write('09_real_fallback_and_metrics.json',receipt)
  raise EpisodeStopped()
 try:collector.collect_trajectory(f['simulator_seed'],f['policy_seed'],f['sample_id'].split('::')[0],on_decision=hook)
 except EpisodeStopped:pass
 except Exception as exc:receipt.update(blocker=type(exc).__name__,detail=str(exc));write('09_real_fallback_and_metrics.json',receipt)
 print(json.dumps({'verdict':receipt['verdict'],'steps':receipt['environment_steps'],'blocker':receipt.get('blocker')}));return 0 if receipt['verdict']=='PASS' else 1
if __name__=='__main__':
 if '--freeze' in sys.argv:freeze()
 elif '--execute' in sys.argv:raise SystemExit(execute())
 else:raise SystemExit('Explicit --freeze / --execute CPU-only mechanism')
