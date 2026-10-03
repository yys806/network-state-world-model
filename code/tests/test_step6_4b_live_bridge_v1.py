import copy
import gzip
import json
from pathlib import Path
import sys
import unittest
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4b_live_bridge_v1 import build_live_sample, SmokeFailure
from pi_jwm.step6_4b_live_bridge_v1 import apply_commands, validate_command, require_nonempty_domain
from types import SimpleNamespace

class LiveBridgeTests(unittest.TestCase):
    def test_receipt_serializes_real_candidate_tags_and_stop_bypasses_policy_wrapper(self):
        import run_step6_4b_two_cycle_smoke_v1 as runner
        old=runner.OUT
        try:
            with tempfile.TemporaryDirectory() as d:
                runner.OUT=Path(d)
                runner.write('fixture.json',{'source_tags':frozenset({'structured_search'})})
                self.assertEqual(json.loads((runner.OUT/'fixture.json').read_text())['source_tags'],['structured_search'])
        finally:runner.OUT=old
        self.assertFalse(issubclass(runner.EpisodeStopped,Exception))
    def test_future_poison_and_deletion_do_not_change_live_sample(self):
        f=json.loads((ROOT/'code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/01_fixture_freeze.json').read_text())
        raw=json.loads(gzip.decompress((ROOT/f['raw_path']).read_bytes()))
        prefix={'environment':raw['environment'],'decisions':raw['decisions'][:4],'steps':raw['steps'][:3]}
        clean=copy.deepcopy(prefix)
        for d in clean['decisions']:d.pop('internal_metadata',None)
        poison=copy.deepcopy(prefix);poison['target']={'fatal':object()};poison['future_action']=object()
        for d in poison['decisions']:d['internal_metadata']={'future_task_schedule':object()}
        self.assertEqual(build_live_sample(clean),build_live_sample(poison))
        missing=copy.deepcopy(clean);missing['decisions'][-1].pop('node_cpu_capacity_observation_rows')
        with self.assertRaises(SmokeFailure):build_live_sample(missing)

    def test_fail_closed_none_and_illegal_route_never_execute_setters(self):
        for step,reason in [(None,'NO_SCOREABLE_H4'),(SimpleNamespace(route=({'task_id':'illegal'},)),'ACTION_BRIDGE_REJECTED')]:
            with self.assertRaises(SmokeFailure) as caught:
                validate_command(step,None,{}, {})
            self.assertEqual(caught.exception.reason,reason)

    def test_setter_failure_restores_decision_buffers_without_environment_step(self):
        env=SimpleNamespace(activated_offloading_tasks_with_RB_Nos={'old':[1]},alloc_cpu_callback=object(),uav_mobility_patterns={'old':{'speed':0}},simulation_time=0.6)
        old=env.alloc_cpu_callback
        comm=SimpleNamespace(setCommunicationWithRB=lambda e,k,v:e.activated_offloading_tasks_with_RB_Nos.update({k:v}))
        def fail(e,c):
            e.alloc_cpu_callback=c
            raise RuntimeError('injected setter failure')
        with self.assertRaises(SmokeFailure) as caught:
            apply_commands(env,{'rb':{'new':[2]},'cpu':{},'mobility':{}},comm,SimpleNamespace(setComputingCallBack=fail),None)
        self.assertEqual(caught.exception.reason,'SETTER_FAILURE')
        self.assertEqual(env.simulation_time,0.6)
        self.assertEqual(env.activated_offloading_tasks_with_RB_Nos,{'old':[1]})
        self.assertIs(env.alloc_cpu_callback,old)
        self.assertEqual(env.uav_mobility_patterns,{'old':{'speed':0}})

    def test_empty_domain_is_an_explicit_stop(self):
        with self.assertRaises(SmokeFailure) as caught:
            require_nonempty_domain(SimpleNamespace(exact_unique_single_step_count=0))
        self.assertEqual(caught.exception.reason,'DOMAIN_EMPTY')

    def test_comm_comp_mob_units_ids_and_illegal_rb(self):
        sys.path.insert(0,str(ROOT/'code/tests'))
        from test_step6_3b_candidate_grammar_v1 import fixture
        from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep
        context,*_=fixture()
        decision={'entities':[{'entity_id':n,'entity_type':'uav' if n.startswith('UAV') else 'rsu'} for n in context.static['input_entity_index']['physical']],
            'tasks':[{'task_id':'input','lifecycle':'offloading'},{'task_id':'compute','lifecycle':'computing'}],
            'node_cpu_capacity_observation_rows':[{'node_id':'RSU_0','observed_mask':True,'capacity_per_s':10.}], 'n_rb':50}
        tasks={'input':SimpleNamespace(), 'compute':SimpleNamespace(getAssignedTo=lambda:'RSU_0')}
        step=CandidateActionStep(comm=({'task_id':'input','task_index':0,'relation_index':0,'rb_indices':[3,4]},),
            comp=({'task_id':'compute','node_id':'RSU_0','allocated_cpu_per_s':5.},),
            mob=({'uav_index':0,'azimuth_rad':0.5,'elevation_rad':0.1,'speed_mps':0.},))
        commands=validate_command(step,context,decision,tasks)
        self.assertEqual(commands['rb'],{'input':[3,4]})
        self.assertEqual(commands['cpu'],{'compute':5.})
        self.assertEqual(commands['mobility'],{'UAV_0':{'angle':0.5,'phi':0.1,'speed':0.}})
        from dataclasses import replace
        bad=replace(step,comm=({**step.comm[0],'rb_indices':[50]},))
        with self.assertRaises(SmokeFailure) as caught:validate_command(bad,context,decision,tasks)
        self.assertEqual(caught.exception.reason,'ACTION_BRIDGE_REJECTED')

    def test_history_only_tensor_equals_frozen_fixture_and_future_poison(self):
        import numpy as np
        from run_step6_4b_two_cycle_smoke_v1 import history_tensor
        from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
        from pi_jwm.step5_5_full_sharded_loader_v1 import _one
        p=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1'
        tid='formal-v1-sim-2026092325-policy-2026092425'
        raw=json.loads(gzip.decompress((p/f'raw/{tid}.json.gz').read_bytes()))
        raw['decisions']=raw['decisions'][:4];raw['steps']=raw['steps'][:3]
        frozen=_one(load_flow_tensor_batch(p/f'packages/tensor/{tid}.npz'),2,92)
        stats=json.loads((p/'packages/normalization/stats.json').read_text())
        _,actual,_=history_tensor(raw,stats,frozen)
        poison=copy.deepcopy(raw);poison['target']=object();poison['future_action']=object()
        for d in poison['decisions']:d['internal_metadata']=object()
        _,poisoned,_=history_tensor(poison,stats,frozen)
        for key,value in actual.items():
            if isinstance(value,np.ndarray) and not key.startswith(('target_','future_')):
                np.testing.assert_array_equal(value,frozen[key],err_msg=key)
                np.testing.assert_array_equal(value,poisoned[key],err_msg=key)

if __name__=='__main__':unittest.main()
