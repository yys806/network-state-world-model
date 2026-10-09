"""One-shot closed-loop orchestration. No model, search or simulator rule changes."""
from dataclasses import dataclass
import math,time,traceback
from .step6_4b_live_bridge_v1 import SmokeFailure,validate_command
from .step6_4e_fallback_v1 import prepare_fallback,FallbackReason,FallbackCandidate
from .step6_3b_candidate_grammar_v1 import admit_structured_candidate
from .step6_0a_candidate_generation_v1 import CandidateActionSequence

SEARCH_METHOD='S-CEM'
K=4
RHO=.2
B_WM=512
HORIZON=4
FALLBACK_POLICY='CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1'
LATENCY_MODE='SYNCHRONOUS_PAUSED_SIMULATION'

@dataclass(frozen=True)
class PlanPacket:
 outcome:object
 domain:object
 root:dict
 predicted_h1_state:str|None
 planning_seconds:float

class EpisodeController:
 """Consumes a decision token before planning; any failure permanently stops.

 Installer must reuse validate_command/apply_commands for winner and fallback.
 Step is attempted once, never resumed mid-episode; recovery needs a separately
 authorized new episode identity. Journal is synchronous before any mutation.
 """
 def __init__(self,journal):
  self.journal=journal;self.used=set();self.stopped=False
 def cycle(self,env,decision,runtime_tasks,planner,installer,step,capture):
  token=(decision.get('capture_event_id'),decision.get('frame_index'),decision.get('simulation_time_s'))
  row={'decision_token':token,'status':'PREPARING','dispatch':'FAIL_CLOSED_C','executed_horizon':0,
       'environment_step_attempted':False,'setter_attempted':False,'setter_partial_mutation_risk':False,'no_retry':True}
  start=time.perf_counter();before=float(env.simulation_time);row['simulation_time_before']=before
  def stop(reason,detail=''):
   self.stopped=True;row.update(status='STOPPED',reason=reason,detail=detail,
     simulation_time_after=float(env.simulation_time),end_to_end_decision_seconds=time.perf_counter()-start)
   self.journal(dict(row));return row
  if self.stopped:return stop('EPISODE_ALREADY_STOPPED')
  if token in self.used:return stop('DUPLICATE_DECISION')
  self.used.add(token);self.journal(dict(row))
  try:
   if token[0] is None or not math.isclose(before,float(decision['simulation_time_s']),abs_tol=1e-9):
    raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','real decision identity/time')
   packet=planner();outcome=packet.outcome;domain=packet.domain
   row.update(root=packet.root,predicted_h1_state=packet.predicted_h1_state,
              internal_search_seconds=packet.planning_seconds,budget_receipt=dict(outcome.budget_receipt),
              h4_scoreable_count=outcome.h4_scoreable_count,best_fingerprint=outcome.best_fingerprint)
   if float(env.simulation_time)!=before:raise SmokeFailure('SIMULATION_ADVANCED_DURING_PLANNING')
   action=outcome.winner_first_action
   if action is None:
    fallback=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,decision,runtime_tasks)
    row['fallback_reason']='NO_SCOREABLE_H4'
    if fallback.terminate_episode:return stop(fallback.reason.value,fallback.detail)
    action=fallback.action;commands=fallback.commands;row['dispatch']='FALLBACK_A'
   else:
    if len(outcome.winner_sequence.steps)!=HORIZON:raise SmokeFailure('ACTION_BRIDGE_REJECTED','winner horizon')
    single=CandidateActionSequence((action,),outcome.winner_sequence.candidate_id,
        outcome.winner_sequence.generator_backend,outcome.winner_sequence.seed,domain.context.causal_provenance)
    admission=admit_structured_candidate(single,domain.context,domain.operational_domain,(domain.state,),domain.catalog,slot_seconds=domain.slot_seconds)
    if not admission.admitted:raise SmokeFailure('ACTION_BRIDGE_REJECTED',admission.reason_codes)
    commands=validate_command(action,domain.context,decision,runtime_tasks);row['dispatch']='WINNER'
   row.update(action=action.frame(),commands=commands,status='VALIDATED')
   self.journal(dict(row))
   row['setter_attempted']=True
   try:installer(env,commands)
   except Exception as exc:
    row['setter_partial_mutation_risk']=True
    return stop('SETTER_FAILURE',str(exc))
   if float(env.simulation_time)!=before:return stop('SETTER_TIME_DRIFT')
   row.update(status='STEP_INTENT',executed_horizon=1,environment_step_attempted=True)
   self.journal(dict(row))
   try:step()
   except Exception as exc:return stop('ENV_STEP_FAILURE',str(exc))
   after=float(env.simulation_time)
   if not math.isclose(after-before,float(env.simulation_interval),abs_tol=1e-8):return stop('ENV_STEP_TIME_INCONSISTENCY')
   fresh=capture()
   if fresh.get('capture_event_id')==decision['capture_event_id'] or not math.isclose(float(fresh.get('simulation_time_s',-1)),after,abs_tol=1e-8):
    return stop('REAL_FEEDBACK_INCONSISTENCY')
   row.update(status='EXECUTED',fresh_observation=fresh,simulation_time_after=after,
       end_to_end_decision_seconds=time.perf_counter()-start)
   self.journal(dict(row));return row
  except Exception as exc:
   return stop(getattr(exc,'reason',type(exc).__name__),
               str(exc) + "\n" + traceback.format_exc())
