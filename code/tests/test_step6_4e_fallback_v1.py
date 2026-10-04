import copy
import gzip
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/tests')]
from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_4e_fallback_v1 import (
    FallbackReason, FallbackCandidate, prepare_fallback, execute_fallback,
    PLANNER_V1_CLOSED_LOOP_B_WM)


class FallbackTests(unittest.TestCase):
    def inputs(self):
        context,operational,state,control,catalog=fixture()
        context=replace(context,history=({'frame_index':0,'simulation_time_s':0.0},))
        domain=CandidateDomain.from_state(context,operational,state,control,catalog)
        decision={'frame_index':context.history[-1].get('frame_index',0),
          'simulation_time_s':context.history[-1].get('simulation_time_s',0),
          'entities':[{'entity_id':n,'entity_type':'uav' if n.startswith('UAV') else 'rsu'} for n in context.static['input_entity_index']['physical']],
          'tasks':[{'task_id':'input','lifecycle':'offloading'},{'task_id':'compute','lifecycle':'computing'}],
          'node_cpu_capacity_observation_rows':[{'node_id':'RSU_0','observed_mask':True,'capacity_per_s':10.}], 'n_rb':50}
        tasks={'input':SimpleNamespace(),'compute':SimpleNamespace(getAssignedTo=lambda:'RSU_0')}
        return domain,decision,tasks

    def test_A_deterministic_lazy_and_future_poison_invariant(self):
        self.assertEqual(PLANNER_V1_CLOSED_LOOP_B_WM,512)
        domain,decision,tasks=self.inputs()
        first=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,decision,tasks)
        self.assertFalse(first.terminate_episode)
        self.assertEqual(first.action.route,())
        poison=copy.deepcopy(decision);poison.update(target=object(),future_schedule=object(),predicted_root=object())
        again=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,poison,tasks)
        self.assertEqual(first.action,again.action)
        self.assertEqual(first.commands,again.commands)
        self.assertEqual(first.action,next(domain.iter_bound()).action)

    def test_critical_reasons_terminate_without_input_or_setters(self):
        for reason in [FallbackReason.DOMAIN_EMPTY,FallbackReason.REQUIRED_LIVE_OBSERVATION_MISSING,
                       FallbackReason.ACTION_BRIDGE_REJECTED,FallbackReason.SETTER_FAILURE]:
            result=prepare_fallback(reason,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION)
            self.assertTrue(result.terminate_episode)
            self.assertIsNone(result.action)
            self.assertFalse(execute_fallback(result,None,None,None,None)['action_executed'])

    def test_empty_domain_and_missing_field_fail_closed(self):
        domain,decision,tasks=self.inputs()
        empty=replace(domain,modes=(),empty_reason='strict test empty')
        result=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,empty,decision,tasks)
        self.assertEqual(result.reason,FallbackReason.DOMAIN_EMPTY)
        decision.pop('n_rb')
        result=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,decision,tasks)
        self.assertEqual(result.reason,FallbackReason.REQUIRED_LIVE_OBSERVATION_MISSING)

    def test_B_requires_current_offer_and_rejects_unsupported_action(self):
        domain,decision,tasks=self.inputs()
        B=FallbackCandidate.PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR
        result=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,B,domain,decision,tasks)
        self.assertTrue(result.terminate_episode)
        self.assertEqual(result.detail,'CURRENT_BEHAVIOR_PROVIDER_NOT_IMPLEMENTED')
        step=next(domain.iter_bound()).action
        offer={'frame_index':decision['frame_index'],'simulation_time_s':decision['simulation_time_s'],'action':step}
        self.assertFalse(prepare_fallback(FallbackReason.NO_SCOREABLE_H4,B,domain,decision,tasks,behavior_offer=offer).terminate_episode)
        offer['frame_index']-=1
        self.assertTrue(prepare_fallback(FallbackReason.NO_SCOREABLE_H4,B,domain,decision,tasks,behavior_offer=offer).terminate_episode)
        offer['frame_index']=decision['frame_index'];offer['action']=replace(step,comm=({'task_id':'input','task_index':0,'relation_index':0,'rb_indices':[50]},))
        self.assertTrue(prepare_fallback(FallbackReason.NO_SCOREABLE_H4,B,domain,decision,tasks,behavior_offer=offer).terminate_episode)

    def test_C_and_setter_failure_stop_without_step(self):
        domain,decision,tasks=self.inputs()
        C=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.FAIL_CLOSED_TERMINATION,domain,decision,tasks)
        self.assertTrue(C.terminate_episode)
        result=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,decision,tasks)
        old=object();env=SimpleNamespace(activated_offloading_tasks_with_RB_Nos={'old':[1]},alloc_cpu_callback=old,uav_mobility_patterns={},simulation_time=0.6)
        def fail(e,c):e.alloc_cpu_callback=c;raise RuntimeError('injected')
        actual=execute_fallback(result,env,SimpleNamespace(setCommunicationWithRB=lambda e,k,v:None),SimpleNamespace(setComputingCallBack=fail),None)
        self.assertTrue(actual['terminate_episode']);self.assertEqual(actual['reason'],'SETTER_FAILURE')
        self.assertFalse(actual['action_executed']);self.assertEqual(env.simulation_time,0.6)
        self.assertIs(env.alloc_cpu_callback,old);self.assertEqual(env.activated_offloading_tasks_with_RB_Nos,{'old':[1]})

    def test_malformed_behavior_offer_fails_closed_before_setters(self):
        domain,decision,tasks=self.inputs()
        offer={'frame_index':decision['frame_index'],'simulation_time_s':decision['simulation_time_s']}
        result=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR,domain,decision,tasks,behavior_offer=offer)
        self.assertTrue(result.terminate_episode)
        self.assertEqual(result.reason,FallbackReason.ACTION_BRIDGE_REJECTED)

    def test_frozen_real_replay_failed_task_comm_is_rejected(self):
        # A factual regression of the unresolved Domain/live-eligibility conflict.
        # This test neither runs the simulator nor relaxes the scientific Domain.
        sys.path.insert(0,str(ROOT/'code/scripts'))
        from run_step6_4e_comm_fallback_audit_v1 import current_domain,actions
        from pi_jwm.step6_4b_live_bridge_v1 import validate_command,SmokeFailure
        frozen=json.loads((ROOT/'code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/01_fixture_and_execution_freeze.json').read_text())
        raw=json.loads(gzip.decompress((ROOT/frozen['raw_path']).read_bytes()))
        frame=frozen['anchor']
        prefix={**raw,'decisions':raw['decisions'][:frame+1],'steps':raw['steps'][:frame]}
        _,domain=current_domain(prefix)
        from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep
        old=frozen['nonempty_comm_action']
        comm=CandidateActionStep(route=(),comm=tuple(old['comm']['entries']),comp=tuple(old['comp']['entries']),mob=tuple(old['mobility']['entries']))
        rows={t['task_id']:t for t in prefix['decisions'][-1]['tasks']}
        self.assertEqual(rows[comm.comm[0]['task_id']]['lifecycle'],'failed')
        with self.assertRaisesRegex(SmokeFailure,'communication not current eligible'):
            validate_command(comm,domain.context,prefix['decisions'][-1],{k:SimpleNamespace() for k in rows})

if __name__=='__main__':unittest.main()
