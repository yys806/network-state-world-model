"""Researcher frozen phase gates and A-else-C policy; no solver changes."""
from collections import Counter
from dataclasses import dataclass, field
import math
from .step6_4e_fallback_v1 import (FallbackReason, FallbackCandidate,
                                 prepare_fallback, execute_fallback)

FINAL_FALLBACK_POLICY='CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1'
TRAIN_SEEDS=(6301,6302,6303)
VALIDATION_SEEDS=(6311,6312,6313,6314,6315)
GRID=((3,.1),(3,.2),(4,.1),(4,.2))
METHODS=('HRS','S-CEM','MH-CEM')

def phase_cases(phase,train_anchors,validation_anchors,configs=None):
    if len(train_anchors)!=32 or len(set(train_anchors))!=32 or len(validation_anchors)!=64 or len(set(validation_anchors))!=64:
        raise ValueError('exact frozen32 TRAIN/64 Validation anchors required')
    if phase=='T':
        return [dict(sample_id=a,seed=s,method=m,iterations=k,elite_ratio=r,budget=512,split='dev_train')
          for m in METHODS[1:] for k,r in GRID for a in train_anchors for s in TRAIN_SEEDS]
    if phase not in ('A','B') or configs is None or set(configs)!=set(METHODS[1:]):
        raise ValueError('explicit phase and newly frozen tuning configs required')
    for c in configs.values():
        if (c['K'],c['elite_ratio']) not in GRID:raise ValueError('tuning configuration outside frozen grid')
    methods=METHODS if phase=='A' else ('MH-CEM',)
    return [dict(sample_id=a,seed=s,method=m,iterations=1 if m=='HRS' else configs[m]['K'],
      elite_ratio=None if m=='HRS' else configs[m]['elite_ratio'],budget=1024 if phase=='A' else 512,split='dev_validation')
      for m in methods for a in validation_anchors for s in VALIDATION_SEEDS]

def require_phase_parent(phase,tuning,selection):
    if phase=='T':return
    if tuning is None or tuning.get('verdict')!='PASS':raise ValueError('Phase T independent acceptance required')
    if phase=='B' and (selection is None or selection.get('verdict')!='PASS' or selection.get('selected_method')!='MH-CEM'):
        raise ValueError('STOP: new Stage A did not formally reselect MH-CEM')
    if phase not in ('A','B'):raise ValueError('unknown phase')

def budget_requalification_gate(stats):
    checks={'paired_identity_complete':stats['paired_cases']==320,
      'scorer_clean':stats['scorer_exceptions']==stats['scorer_inconsistencies']==0,
      'same_scoreability_set':stats['only_B512_scoreable']==stats['only_B1024_scoreable']==0,
      'no_primary_objective_degradation':stats['PRIMARY_OBJECTIVE_DEGRADATION_COUNT']==0,
      'mean_runtime_lower':stats['mean_512']<stats['mean_1024'],
      'median_runtime_lower':stats['median_512']<stats['median_1024'],
      'valid_runtimes':all(math.isfinite(stats[k]) and stats[k]>0 for k in ('mean_512','mean_1024','median_512','median_1024')),
      'budget_accounting':stats['budget_accounting_passed'] is True}
    passed=all(checks.values())
    return {'checks':checks,'passed':passed,'B512_EVIDENCE_REQUALIFICATION':'PASS' if passed else 'FAIL',
      'closed_loop_budget':512 if passed else 'RESEARCHER_DECISION_PENDING',
      'claim_boundary':'Burden/Effort loss allowed and reported; not equivalence or optimality'}

@dataclass
class FallbackCounters:
    fallback_trigger_count:int=0
    reason_distribution:Counter=field(default_factory=Counter)
    A_execution_count:int=0
    C_termination_count:int=0
    def receipt(self,decision_count=None):
        return {'fallback_trigger_count':self.fallback_trigger_count,
          'fallback_trigger_rate':None if not decision_count else self.fallback_trigger_count/decision_count,
          'reason_distribution':dict(self.reason_distribution),'A_execution_count':self.A_execution_count,
          'C_termination_count':self.C_termination_count,'rate_denominator':decision_count}

def dispatch_frozen_fallback(reason,domain=None,decision=None,runtime_tasks=None,
    env=None,communication=None,computation=None,traffic=None,*,counters=None):
    """One canonical A attempt, otherwise C. Same bridge; never env.step here."""
    reason=FallbackReason(reason)
    if counters is not None:
        counters.fallback_trigger_count+=1;counters.reason_distribution[reason.value]+=1
    def stop(why,detail=''):
        if counters is not None:counters.C_termination_count+=1
        return {'policy':FINAL_FALLBACK_POLICY,'selected_candidate':FallbackCandidate.FAIL_CLOSED_TERMINATION.value,
          'reason':why,'trigger_reason':reason.value,'detail':detail,'action_executed':False,
          'terminate_episode':True,'no_retry':True,'environment_steps':0}
    if reason!=FallbackReason.NO_SCOREABLE_H4:return stop(reason.value)
    try:
        prepared=prepare_fallback(reason,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,
                                  domain,decision,runtime_tasks)
    except Exception as exc:return stop(FallbackReason.ACTION_BRIDGE_REJECTED.value,str(exc))
    if prepared.terminate_episode:return stop(prepared.reason.value,prepared.detail)
    try:receipt=execute_fallback(prepared,env,communication,computation,traffic)
    except Exception as exc:return stop(FallbackReason.SETTER_FAILURE.value,str(exc))
    if receipt['terminate_episode'] or not receipt['action_executed']:
        return {**receipt,**stop(receipt.get('reason',FallbackReason.SETTER_FAILURE.value),receipt.get('detail',''))}
    if counters is not None:counters.A_execution_count+=1
    return {**receipt,'policy':FINAL_FALLBACK_POLICY,'selected_candidate':FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION.value,
            'trigger_reason':reason.value,'no_retry':True}
