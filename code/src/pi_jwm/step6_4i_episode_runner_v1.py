"""Bounded episode orchestration; formal runs require separate approved protocol.

 Capture owns the actual observer/causal-history append. Planner is called again
 only with the fresh real observation, never a prior PlanPacket or prediction.
 This file provides no cloud/GPU launcher or automatic recovery.
 """
import json,time
from pathlib import Path
from .step6_4i_episode_v1 import EpisodeController
from .step6_4b_live_bridge_v1 import SmokeFailure

def run_episode(*,episode_id,output,env,initial_observation,runtime_tasks,planner,installer,step,capture,
                decision_steps,engineering_only,approved_protocol=None,task_ledger=None):
 if not isinstance(decision_steps,int) or decision_steps<1:raise ValueError('positive decision steps required')
 if engineering_only:
  if decision_steps>2:raise PermissionError('engineering-only CPU mechanism limit: two decisions')
 elif not approved_protocol or approved_protocol.get('status')!='APPROVED_BY_RESEARCHER' or episode_id not in approved_protocol.get('episode_ids',[]) or approved_protocol.get('decision_steps')!=decision_steps:
  raise PermissionError('formal episode/seed/length and GPU authorization pending')
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 def save(name,value):
  p=output/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8');tmp.replace(p)
 journal=[]
 def log(row):journal.append(row);save('journal.json',journal)
 driver=EpisodeController(log);start=time.perf_counter();current=initial_observation;rows=[]
 initial_time=float(env.simulation_time)
 receipt={'episode_id':episode_id,'scheduled_decisions':decision_steps,'executed_decisions':0,'engineering_only':engineering_only,
          'status':'RUNNING','locked_test':False,'resume_within_episode':False,'automatic_retry':False}
 save('attempt.json',receipt)
 try:
  if task_ledger is not None:task_ledger.observe(current['tasks'],runtime_tasks(),initial_time)
  for _ in range(decision_steps):
   def fresh_plan():
    packet=planner(current)
    if not engineering_only:
     outcome=packet.outcome
     if (outcome.method!='S-CEM' or outcome.budget!=512 or outcome.iterations!=4 or outcome.elite_ratio!=.2 or outcome.batch_size!=16):
      raise SmokeFailure('FROZEN_PLANNER_CONFIG_MISMATCH')
    return packet
   row=driver.cycle(env,current,runtime_tasks(),fresh_plan,installer,step,capture);rows.append(row)
   if row['status']!='EXECUTED':receipt.update(status='STOPPED',reason=row['reason']);break
   receipt['executed_decisions']+=1;current=row['fresh_observation']
   if task_ledger is not None:task_ledger.observe(current['tasks'],runtime_tasks(),float(env.simulation_time))
  else:receipt['status']='BOUNDED_WINDOW_COMPLETED'
 except Exception as exc:
  receipt.update(status='STOPPED',reason=getattr(exc,'reason',type(exc).__name__),detail=str(exc))
 finally:
  receipt.update(attempted_decisions=len(rows),unattempted_scheduled_decisions=decision_steps-len(rows),elapsed_seconds=time.perf_counter()-start,
     winner_count=sum(r.get('dispatch')=='WINNER' and r['status']=='EXECUTED' for r in rows),
     fallback_A_count=sum(r.get('dispatch')=='FALLBACK_A' and r['status']=='EXECUTED' for r in rows),
     termination_C_count=int(receipt['status']=='STOPPED'),rows=rows)
  if task_ledger is not None:
   try:receipt['draft_real_metrics']=task_ledger.report(planned_duration_s=decision_steps*float(env.simulation_interval),actual_duration_s=float(env.simulation_time)-initial_time)
   except Exception as exc:receipt.update(status='STOPPED',reason='METRIC_INCONSISTENCY',detail=str(exc),termination_C_count=1)
  save('receipt.json',receipt)
 return receipt
