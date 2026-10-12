"""Whole-decision validation against the saved causal real anchor."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import torch
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext,CandidateActionStep
from pi_jwm.step6_4n_joint_bridge_v2 import validate_joint_commands_v2,apply_joint_commands_v2

ROOT=Path(__file__).resolve().parents[2]

class JointBridgeTests(unittest.TestCase):
    def setUp(self):
        a=torch.load(ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v4/common_anchor.pt',weights_only=False)
        self.ctx=PlannerCandidateContext.from_sample(a['sample'],a['state'],'joint-test')
        self.raw=copy.deepcopy(a['current'])
        self.ids=self.ctx.static['input_entity_index']; self.row=next(r for r in self.raw['tasks'] if r['task_id']=='Task_5')
        r=self.row
        task=SimpleNamespace(getTaskId=lambda:'Task_5',getCurrentNodeId=lambda:r['current_node_id'],
            getToReturnNodeId=lambda:r['return_destination_id'],getReturnedSize=lambda:r['required_returned_size'],
            getTaskCPU=lambda:r['task_cpu_work'],getComputedSize=lambda:r['computed_cpu_work'])
        self.env=SimpleNamespace(task_manager=SimpleNamespace(_waiting_to_return_tasks={'UAV_0':[task]}),
            channel_manager=SimpleNamespace(n_RB=50),task_return_routes={},activated_offloading_tasks_with_RB_Nos={},
            _getNodeTypeById=lambda n:'U' if n.startswith('UAV') else 'I',_getNodeIdxById=lambda n:self.ids['physical'].get(n,-1))
        self.tasks={'Task_5':task};self.catalog=SimpleNamespace(comm_start_width_pairs={(0,1)})
        ti=self.ids['task']['Task_5'];dst=self.ids['physical']['UAV_1']
        self.step=CandidateActionStep(route=({'task_id':'Task_5','task_index':ti,'route_kind':'return',
            'route_node_ids':['UAV_1'],'route_node_indices':[dst]},),
            comm=({'task_id':'Task_5','task_index':ti,'relation_index':38,'rb_indices':[0]},))

    def validate(self,step=None):
        return validate_joint_commands_v2(self.env,self.step if step is None else step,self.ctx,self.raw,self.tasks,self.catalog)

    def test_joint_and_both_no_service_controls(self):
        from pi_jwm.step6_4n_joint_bridge_v2 import DynamicCandidateDomainV2
        domain=DynamicCandidateDomainV2(self.env,self.ctx,self.raw,self.tasks,self.catalog)
        declared=domain.declared_return_controls(self.step.route[0],38,([0],))
        self.assertEqual(len(declared),3)
        self.assertEqual(declared[0].route,())
        self.assertEqual(declared[1].comm,())
        self.assertEqual(declared[2].comm,self.step.comm)
        self.assertEqual(self.validate()['rb'],{'Task_5':[0]})
        self.assertEqual(self.validate(replace(self.step,comm=()))['rb'],{})
        self.assertEqual(self.validate(CandidateActionStep())['routes'],{})
        self.assertEqual(self.env.task_return_routes,{})

    def test_invalid_whole_decision_does_not_mutate_any_buffer(self):
        bads=[replace(self.step,route=self.step.route*2),replace(self.step,comm=self.step.comm*2),
            replace(self.step,comm=({**self.step.comm[0],'rb_indices':[50]},)),
            replace(self.step,comm=({**self.step.comm[0],'relation_index':-1},)),
            replace(self.step,route=({**self.step.route[0],'task_index':999},)),
            replace(self.step,mob=({'uav_index':self.ids['physical']['UAV_0'],'azimuth_rad':0.,'elevation_rad':0.,'speed_mps':16.},))]
        for step in bads:
            with self.subTest(step=step),self.assertRaises((ValueError,KeyError,IndexError)): self.validate(step)
            self.assertEqual(self.env.task_return_routes,{})
            self.assertEqual(self.env.activated_offloading_tasks_with_RB_Nos,{})

    def test_no_demand_no_relation_failed_and_poison(self):
        for patch in ({'required_returned_size':0.},{'lifecycle':'failed'},{'lifecycle':'done'},
                      {'computed_cpu_work':self.row['task_cpu_work']-1e-16}):
            old=copy.deepcopy(self.row);self.row.update(patch)
            with self.assertRaises(ValueError):self.validate()
            self.row.clear();self.row.update(old)
        old=self.raw['channel_rows'];self.raw['channel_rows']=[]
        with self.assertRaises(ValueError):self.validate()
        self.raw['channel_rows']=old;self.raw['future_target']=object()
        with self.assertRaisesRegex(ValueError,'FUTURE'):self.validate()

    def test_all_four_partial_setter_failures_quarantine_without_step(self):
        commands=self.validate()
        for fail in ('route','comm','comp','mob'):
            e=copy.copy(self.env);e.task_return_routes={};e.activated_offloading_tasks_with_RB_Nos={};e.step=lambda:None
            def setter(name,fn):
                def run(*args):
                    fn(*args)
                    if name==fail:raise RuntimeError('injected '+name)
                return run
            route=SimpleNamespace(setTaskReturnRoute=setter('route',lambda e,t,p:e.task_return_routes.update({t:p})))
            comm=SimpleNamespace(setCommunicationWithRB=setter('comm',lambda e,t,v:e.activated_offloading_tasks_with_RB_Nos.update({t:v})))
            comp=SimpleNamespace(setComputingCallBack=setter('comp',lambda e,cb:None))
            mob=SimpleNamespace(setUAVMobilityPatterns=setter('mob',lambda e,p:None))
            with self.subTest(fail=fail),self.assertRaisesRegex(RuntimeError,'PARTIAL'):
                apply_joint_commands_v2(e,commands,route_scheduler=route,communication=comm,computation=comp,traffic=mob)
            with self.assertRaisesRegex(RuntimeError,'QUARANTINED'):e.step()

    def test_existing_returning_with_comp_mob_and_resource_overflow(self):
        from test_step6_3b_candidate_grammar_v1 import fixture
        ctx,*_=fixture()
        tx=SimpleNamespace(getCurrentNodeId=lambda:'vehicle_0',getToOffloadRoute=lambda:['RSU_0'])
        compute=SimpleNamespace(getAssignedTo=lambda:'RSU_0')
        env=SimpleNamespace(task_return_routes={},activated_offloading_tasks_with_RB_Nos={},
            channel_manager=SimpleNamespace(n_RB=50),task_manager=SimpleNamespace(_offloading_tasks={},_returning_tasks={'vehicle_0':[tx]}))
        raw={'tasks':[{'task_id':'input','lifecycle':'returning'},{'task_id':'compute','lifecycle':'computing'}],
             'entities':[{'entity_id':n,'entity_type':'uav' if n.startswith('UAV') else 'rsu'} for n in ctx.static['input_entity_index']['physical']],
             'channel_rows':[],'n_rb':50,'node_cpu_capacity_observation_rows':[{'node_id':'RSU_0','observed_mask':True,'capacity_per_s':10.}]}
        step=CandidateActionStep(comm=({'task_id':'input','task_index':0,'relation_index':0,'rb_indices':[0]},),
            comp=({'task_id':'compute','node_id':'RSU_0','allocated_cpu_per_s':5.},),
            mob=({'uav_index':0,'azimuth_rad':.5,'elevation_rad':.1,'speed_mps':0.},))
        def validate(s):return validate_joint_commands_v2(env,s,ctx,raw,{'input':tx,'compute':compute},self.catalog)
        self.assertEqual(validate(step)['cpu'],{'compute':5.})
        ctx.current_state['task_lifecycle_index'][0,0]=0
        with self.assertRaisesRegex(ValueError,'STALE'):validate(step)
        ctx.current_state['task_lifecycle_index'][0,0]=4
        with self.assertRaises(ValueError):validate(replace(step,comp=({**step.comp[0],'allocated_cpu_per_s':11.},)))
        for life in ('failed','done','waiting_to_return'):
            raw['tasks'][0]['lifecycle']=life
            with self.assertRaises(ValueError):validate(step)
        raw['tasks'][0]['lifecycle']='returning'
        self.assertEqual(env.task_manager._offloading_tasks,{})
        env.task_manager._returning_tasks={}
        with self.assertRaisesRegex(ValueError,'STALE'):validate(step)

if __name__=='__main__':unittest.main()
