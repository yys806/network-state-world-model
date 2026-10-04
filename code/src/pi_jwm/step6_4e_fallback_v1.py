"""Explicit fallback candidates; no default policy, model score, or env.step."""
from dataclasses import dataclass, replace
from enum import Enum
import math
from .step6_0a_candidate_generation_v1 import Backend, CandidateActionSequence
from .step6_3b_candidate_grammar_v1 import admit_structured_candidate
from .step6_4b_live_bridge_v1 import SmokeFailure, validate_command, apply_commands

PLANNER_V1_CLOSED_LOOP_B_WM=512
FINAL_FALLBACK_POLICY='RESEARCHER_DECISION_PENDING'


class FallbackReason(str,Enum):
    NO_SCOREABLE_H4='NO_SCOREABLE_H4'
    DOMAIN_EMPTY='DOMAIN_EMPTY'
    REQUIRED_LIVE_OBSERVATION_MISSING='REQUIRED_LIVE_OBSERVATION_MISSING'
    ACTION_BRIDGE_REJECTED='ACTION_BRIDGE_REJECTED'
    SETTER_FAILURE='SETTER_FAILURE'
    LIVE_SLOT_UNSUPPORTED='LIVE_SLOT_UNSUPPORTED'


class FallbackCandidate(str,Enum):
    CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION='CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION'
    PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR='PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR'
    FAIL_CLOSED_TERMINATION='FAIL_CLOSED_TERMINATION'


@dataclass(frozen=True)
class FallbackResult:
    reason: FallbackReason
    candidate: FallbackCandidate
    action: object=None
    commands: object=None
    terminate_episode: bool=True
    detail: str=''


def prepare_fallback(reason,candidate,domain=None,decision=None,runtime_tasks=None,*,behavior_offer=None):
    """The caller explicitly chooses a candidate for audit; no automatic policy.

    A uses the first lazy canonical grammar choice on a fresh current domain.
    B only removes Route from an already generated *current causal* offer;
    it never repairs RBs, CPU allocation or mobility into a different policy.
    Existing collector behavior has no detached offer provider (audited missing).
    Critical input/bridge/setter failures terminate; no cascading retries.
    """
    reason=FallbackReason(reason);candidate=FallbackCandidate(candidate)
    def stop(why,detail=''):return FallbackResult(why,candidate,detail=detail)
    if reason!=FallbackReason.NO_SCOREABLE_H4 or candidate==FallbackCandidate.FAIL_CLOSED_TERMINATION:
        return stop(reason)
    if domain is None or domain.is_empty:return stop(FallbackReason.DOMAIN_EMPTY)
    if decision is None or runtime_tasks is None:return stop(FallbackReason.REQUIRED_LIVE_OBSERVATION_MISSING)
    required=('frame_index','simulation_time_s','entities','tasks','node_cpu_capacity_observation_rows','n_rb')
    if any(k not in decision for k in required):return stop(FallbackReason.REQUIRED_LIVE_OBSERVATION_MISSING)
    # Current-only identity gate: a later predicted branch domain is forbidden.
    if not domain.context.history:return stop(FallbackReason.REQUIRED_LIVE_OBSERVATION_MISSING,'causal history missing')
    anchor=domain.context.history[-1]
    if (domain.prior_signatures or
        ('frame_index' in anchor and decision['frame_index']!=anchor['frame_index']) or
        ('simulation_time_s' in anchor and not math.isclose(float(decision['simulation_time_s']),float(anchor['simulation_time_s']),abs_tol=1e-9))):
        return stop(FallbackReason.ACTION_BRIDGE_REJECTED,'NOT_CURRENT_REAL_DOMAIN')
    try:
        if candidate==FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION:
            # modes/count/subset/rows/RB starts already have canonical ordering.
            # Consume one lazy bound; never materialize alternatives.
            action=next(domain.iter_bound()).action
        else:
            if behavior_offer is None:return stop(FallbackReason.ACTION_BRIDGE_REJECTED,'CURRENT_BEHAVIOR_PROVIDER_NOT_IMPLEMENTED')
            if any(behavior_offer.get(k)!=decision[k] for k in ('frame_index','simulation_time_s')):
                return stop(FallbackReason.ACTION_BRIDGE_REJECTED,'BEHAVIOR_OFFER_NOT_CURRENT')
            action=replace(behavior_offer['action'],route=())
        sequence=CandidateActionSequence((action,),candidate.value,Backend.RULE_FALLBACK,None,domain.context.causal_provenance)
        admission=admit_structured_candidate(sequence,domain.context,domain.operational_domain,(domain.state,),domain.catalog,slot_seconds=domain.slot_seconds)
        if not admission.admitted:return stop(FallbackReason.ACTION_BRIDGE_REJECTED,','.join(admission.reason_codes))
        commands=validate_command(action,domain.context,decision,runtime_tasks)
    except (SmokeFailure,ValueError,KeyError,IndexError,TypeError,AttributeError,StopIteration) as exc:
        return stop(FallbackReason.ACTION_BRIDGE_REJECTED,str(exc))
    return FallbackResult(reason,candidate,action,commands,False)


def execute_fallback(result,env,communication,computation,traffic):
    """Reuse winner bridge buffers/setters. Never advances time or retries.

    Native three-buffer rollback is inherited from apply_commands. This does
    not promise arbitrary setter atomicity; failure always ends the episode.
    """
    if result.terminate_episode:
        return {'action_executed':False,'terminate_episode':True,'reason':result.reason.value,'detail':result.detail}
    before=float(env.simulation_time)
    try:
        apply_commands(env,result.commands,communication,computation,traffic)
    except SmokeFailure as exc:
        return {'action_executed':False,'terminate_episode':True,'reason':exc.reason,
                'detail':str(exc),'no_retry':True,'native_decision_buffers_restored':True,
                'arbitrary_setter_atomicity_guaranteed':False}
    assert float(env.simulation_time)==before
    return {'action_executed':True,'terminate_episode':False,'reason':result.reason.value,'environment_steps':0}
