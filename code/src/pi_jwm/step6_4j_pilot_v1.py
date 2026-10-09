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
def validate_spent(receipt):
 if receipt.get('B_WM')!=512 or receipt.get('N_unique_transition_evals')!=512:raise RuntimeError('BUDGET_INCOMPLETE_OR_MISMATCH')
def validate_resume(current,stored,complete):
 if current!=stored:raise RuntimeError('IDENTITY_MISMATCH')
 if not complete:raise RuntimeError('NO_MID_EPISODE_RESUME')
 return 'READ_ONLY_COMPLETE'
