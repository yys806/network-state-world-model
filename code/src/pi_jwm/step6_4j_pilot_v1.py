"""CPU-side Pilot contract guards; no launcher or cloud control."""
import hashlib,time
import math

def select_episodes(rows):
 groups={}
 for row in rows:
  sid=str(row['sample_id']); trajectory=sid.split('::',1)[0]
  groups.setdefault(trajectory,[]).append(row)
 ranked=sorted(groups,key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))[:2]
 return [min(groups[t],key=lambda r:(int(str(r['sample_id']).rsplit('-',1)[1]),str(r['sample_id'])))['sample_id'] for t in ranked]
class PilotClock:
 def __init__(self,now=time.monotonic):self.now=now;self.started=self.now();self.total_limit_s=3*3600;self.stop_before_s=10200
 def before_plan(self):
  if self.now()-self.started>=self.stop_before_s:raise RuntimeError('PILOT_TIME_LIMIT')
 def check_total(self):
  if self.now()-self.started>=self.total_limit_s:raise RuntimeError('PILOT_INSTANCE_LIMIT')
def validate_spent(receipt, expected_budget=512):
 if receipt.get('B_WM')!=expected_budget or receipt.get('N_unique_transition_evals')!=expected_budget:raise RuntimeError('BUDGET_INCOMPLETE_OR_MISMATCH')
def engineering_acceptance(episodes):
 if len(episodes)!=2:return False
 for episode in episodes:
  steps=episode.get('decisions',0);roots=episode.get('root_ids',[])
  if (episode.get('status')!='COMPLETED' or steps<2 or len(roots)<2 or len(set(roots))<2
      or episode.get('actual_action_history_aligned') is not True or episode.get('failures')):
   return False
 return True
def validate_resume(current,stored,complete):
 if current!=stored:raise RuntimeError('IDENTITY_MISMATCH')
 if not complete:raise RuntimeError('NO_MID_EPISODE_RESUME')
 return 'READ_ONLY_COMPLETE'

RESOURCE_STOPS = {'PILOT_TIME_LIMIT', 'PILOT_INSTANCE_LIMIT', 'PILOT_SINGLE_PLAN_TIMEOUT'}

def validate_formal_decision(row, identity):
 if row.get('status')!='EXECUTED':raise RuntimeError(row.get('reason','DECISION_NOT_EXECUTED'))
 validate_spent(row.get('budget_receipt',{}))
 if row.get('execution_identity')!={'device':'cuda','precision':'FP32','batch_size':16,'execution_config_id':identity}:raise RuntimeError('EXECUTION_IDENTITY_MISMATCH')
 if row.get('planning_validation')!='PASS':raise RuntimeError('PLANNING_NOT_VALIDATED')
 latency=float(row['internal_search_seconds'])
 if not math.isfinite(latency) or latency>600:raise RuntimeError('PILOT_SINGLE_PLAN_TIMEOUT')
 if (row.get('dispatch') not in {'WINNER','FALLBACK_A'} or row.get('setter_attempted') is not True or row.get('environment_step_attempted') is not True or row.get('executed_horizon')!=1 or row.get('setter_partial_mutation_risk') is True):raise RuntimeError('EXECUTION_INCONSISTENCY')
 token=row['decision_token'];fresh=row['fresh_observation']
 if (not fresh.get('capture_event_id') or fresh['capture_event_id']==token[0] or fresh.get('capture_method')!='fresh_direct_real_environment_read' or fresh.get('capture_phase')!='loop_start_decision' or fresh.get('frame_index')!=token[1]+1 or not math.isclose(float(fresh['simulation_time_s']),float(row['simulation_time_after']),abs_tol=1e-8) or float(row['simulation_time_after'])<=float(row['simulation_time_before']) or row.get('history_outcome')!=fresh or not row.get('history_action') or row.get('action_history_validation')!='PASS' or 'slot_transfer_events' not in fresh or 'communication_observation' not in fresh or not row.get('root',{}).get('state')):raise RuntimeError('REAL_FEEDBACK_OR_HISTORY_INCONSISTENCY')
 def finite(v):
  if isinstance(v,float) and not math.isfinite(v):raise RuntimeError('NONFINITE_EXECUTION_EVIDENCE')
  if isinstance(v,dict):
   for x in v.values():finite(x)
  elif isinstance(v,(list,tuple)):
   for x in v:finite(x)
 finite(row)

class FormalPilotGate:
 """Global one-shot execution guard, also replayed by independent acceptance."""
 def __init__(self,episodes,identity):
  if len(episodes)!=2 or len(set(episodes))!=2:raise RuntimeError('PILOT_SCOPE_MISMATCH')
  self.episodes=list(episodes);self.identity=identity;self.attempts=[];self.qualified=False;self.stop=None;self.tokens=set();self.roots=set()
 def begin_search(self,episode):
  if self.stop:raise RuntimeError('PILOT_STOPPED')
  if episode not in self.episodes:raise RuntimeError('EPISODE_IDENTITY_MISMATCH')
  if len(self.attempts)>=16 or sum(a['episode']==episode for a in self.attempts)>=8:raise RuntimeError('PILOT_SEARCH_BUDGET_EXHAUSTED')
  if episode==self.episodes[1] and not self.qualified:raise RuntimeError('FIRST_CUDA_QUALIFICATION_PENDING')
  n=len(self.attempts)+1;self.attempts.append({'episode':episode,'number':n});return n
 def resource_stop(self,reason):
  self.stop={'status':'PARTIAL' if self.qualified and reason in RESOURCE_STOPS else 'BLOCKED','reason':reason}
 def accept(self,episode,row):
  try:
   if row.get('status')!='EXECUTED':
    reason=row.get('reason','DECISION_NOT_EXECUTED')
    if self.qualified and reason in RESOURCE_STOPS:self.resource_stop(reason);return
    if (self.qualified and row.get('dispatch')=='FAIL_CLOSED_C' and row.get('fallback_reason')=='NO_SCOREABLE_H4' and reason in {'DOMAIN_EMPTY','ACTION_BRIDGE_REJECTED'} and row.get('planning_validation')=='PASS' and not row.get('setter_attempted') and not row.get('environment_step_attempted')):
     validate_spent(row.get('budget_receipt',{}));self.stop={'status':'PARTIAL','reason':'LEGAL_FALLBACK_C'};return
    raise RuntimeError(reason)
   validate_formal_decision(row,self.identity)
   token=(episode,tuple(row['decision_token']));root=(episode,row['root']['state'])
   if token in self.tokens or root in self.roots:raise RuntimeError('DUPLICATE_DECISION_OR_ROOT')
   self.tokens.add(token);self.roots.add(root);self.qualified=True
  except Exception as exc:self.stop={'status':'BLOCKED','reason':str(exc)}

def formal_result(episodes,rows_by_episode,attempts,stop,identity):
 """Derive result from raw decisions and search intents, never a summary PASS."""
 gate=FormalPilotGate(episodes,identity);by_number={};error=None
 try:
  for ep,rows in rows_by_episode.items():
   for row in rows:
    n=row.get('search_attempt')
    if n in by_number:raise RuntimeError('DUPLICATE_RECEIPT')
    by_number[n]=(ep,row)
  if len(by_number)!=len(attempts):raise RuntimeError('ATTEMPT_RECEIPT_COUNT_MISMATCH')
  for a in attempts:
   n=gate.begin_search(a['episode'])
   if n!=a['number'] or n not in by_number:raise RuntimeError('ATTEMPT_SEQUENCE_MISMATCH')
   ep,row=by_number[n]
   if ep!=a['episode']:raise RuntimeError('EPISODE_RECEIPT_MISMATCH')
   gate.accept(ep,row)
  if stop and not gate.stop:
   if stop.get('reason')=='LEGAL_FALLBACK_C':raise RuntimeError('UNPROVEN_FALLBACK_C')
   gate.resource_stop(stop.get('reason'))
 except Exception as exc:error=str(exc)
 counts={ep:sum(r.get('status')=='EXECUTED' for r in rows_by_episode.get(ep,[])) for ep in episodes}
 if error:status='BLOCKED';reason=error
 elif gate.stop:status=gate.stop['status'];reason=gate.stop['reason']
 elif gate.qualified and all(counts[ep]==8 for ep in episodes) and len(attempts)==16:status='COMPLETED';reason=None
 else:status='BLOCKED';reason='UNEXPLAINED_INCOMPLETE_PILOT'
 return {'status':status,'exit_code':{'COMPLETED':0,'PARTIAL':2,'BLOCKED':1}[status],'reason':reason,'qualification':'PASS' if gate.qualified else 'FAIL','attempted_searches':len(attempts),'env_step_count':sum(counts.values()),'step_attempt_count':sum(r.get('environment_step_attempted') is True for rs in rows_by_episode.values() for r in rs),'planned_decisions':16,'episode_decisions':counts,'pilot_pass':status=='COMPLETED'}

def audit_formal_result(result_dir,episodes,attempts,stop,identity):
 import json
 from pathlib import Path
 rows={ep:[] for ep in episodes}; journal_attempts=[]; journal_decisions=[]; error=None
 try:
  for ep in episodes:
   folder=Path(result_dir)/ep
   rows[ep]=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('decision_*.json'))]
   events=[json.loads(line) for line in (folder/'journal.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
   journal_attempts.extend({'episode':ep,'number':x['number']} for x in events if x.get('event')=='SEARCH_INTENT')
   journal_decisions.extend((ep,x['row'].get('search_attempt')) for x in events if x.get('event')=='FINAL_DECISION')
  if journal_attempts!=[{'episode':x['episode'],'number':x['number']} for x in attempts]:raise RuntimeError('JOURNAL_ATTEMPT_MISMATCH')
  expected=[(ep,r.get('search_attempt')) for ep,rs in rows.items() for r in rs]
  if sorted(expected)!=sorted(journal_decisions):raise RuntimeError('JOURNAL_DECISION_MISMATCH')
 except Exception as exc:error=str(exc)
 result=formal_result(episodes,rows,attempts,stop,identity)
 if error:result.update(status='BLOCKED',exit_code=1,reason=error,pilot_pass=False)
 return result
