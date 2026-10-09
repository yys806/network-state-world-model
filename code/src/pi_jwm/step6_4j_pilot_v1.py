"""CPU-side Pilot contract guards; no launcher or cloud control."""
import hashlib,time

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
