"""Four bounded paired AirFogSim runs; v2 behavior is explicitly opt-in."""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, os, random, sys, time
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/reference/AirFogSim/examples')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONUTF8']='1'
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf8',errors='replace')
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_v5'
MODES=('LEGACY_ROUTE','V2_ROUTE_NO_RB','V2_ROUTE_RB','V2_NOOP')

def atomic(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    with q.open('w',encoding='utf8') as f:
        json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,default=str); f.write('\n'); f.flush(); os.fsync(f.fileno())
    q.replace(p)

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

class StopCapture(BaseException): pass

def main(mode):
    import numpy as np
    import collect_step5_5_formal_raw_v1 as c
    from run_step6_4l_r1_task5_runtime_trace_v1 import task5, task_snapshot, bucket_ids, install_trace_wrappers
    from pi_jwm.step6_4n_runtime_v2 import stage_aware_return_tti, validate_joint_return, commit_joint_return
    from pi_jwm.airfogsim_full_dual_graph_collector_v1 import _install_lifecycle_alias_guard
    from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import CausalFlowLedger
    case=OUT/mode
    if case.exists(): raise RuntimeError('NAMESPACE_ALREADY_EXISTS')
    case.mkdir(parents=True)
    start=time.perf_counter(); receipt={'mode':mode,'status':'STARTED','GPU':False,'locked_test':False,
        'simulator_seed':2026092318,'policy_seed':2026092418,'target_frame':6,'diagnostic_steps':0,
        'prefix_replay_steps':0,'schema':'STEP6.4N-REAL-V2','events':[]}
    receipt['executed_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'code/src/pi_jwm/step6_4n_runtime_v2.py')}
    atomic(case/'00_start.json',receipt)
    def hook(env,decisions,steps,config,communication):
        if time.perf_counter()-start>180: raise TimeoutError('ROOT_CPU_TIMEOUT')
        if int(decisions[-1]['frame_index'])!=6:return
        current=decisions[-1]; t=task5(env)
        raw=next(row for row in current['tasks'] if str(row['task_id'])=='Task_5')
        prefix=copy.deepcopy({'decisions':decisions,'steps':steps,'trajectory_id':current['trajectory_id']})
        atomic(case/'01_prefix.json',prefix)
        side={tid:{'arrival_time_s':float(x.getTaskArrivalTime()),'deadline_s':float(x.getTaskDeadline())}
              for tid,x in c.step23._all_runtime_tasks(env).items() if tid in {r['task_id'] for r in current['tasks']}}
        links={(str(row['source_id']),str(row['target_id'])) for row in current['channel_rows']
               if str(row.get('channel_type','')) in {'U2U','U2V','U2I','V2U','V2V','V2I','I2U','I2V','I2I'}}
        env._pi_jwm_v2_wireless_relations=links
        route=['UAV_1']
        # The fixed baseline RB is (start=0,width=1), selected before outcomes.
        catalog=json.loads((ROOT/'code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json').read_text(encoding='utf8'))
        if not any(int(x['start']) == 0 and int(x['width']) == 1 for x in catalog['comm_start_width_pairs']):
            raise ValueError('RB_PATTERN_OUTSIDE_TRAIN')
        if mode == 'V2_ROUTE_RB':
            validate_joint_return(env,raw,route,[0],links)
        root={'current':current,'task':task_snapshot(t),'task_raw':raw,'buckets':bucket_ids(env.task_manager),
              'deadline_sidecar_tasks':side,'prefix_sha':digest(prefix),'current_sha':digest(current),
              'rng_sha':digest({'python':random.getstate(),'numpy':str(np.random.get_state())}),
              'last_input_clock':float(t.getLastTransmissionTime()),'time':float(env.simulation_time),
              'runtime_source':{str(Path(inspect.getfile(type(x))).resolve()):hashlib.sha256(Path(inspect.getfile(type(x))).read_bytes()).hexdigest()
                 for x in (env,env.task_manager,t)},'comm_relations':sorted(links)}
        atomic(case/'02_root.json',root)
        receipt['prefix_replay_steps']=len(steps)
        install_trace_wrappers(env,receipt['events'])
        ledger=CausalFlowLedger()
        with stage_aware_return_tti(env.task_manager) if mode!='LEGACY_ROUTE' else nullcontext({}) as clocks:
            if mode=='V2_ROUTE_RB':
                setter=commit_joint_return(env,'Task_5',route,[0],c.step23.TaskScheduler.setTaskReturnRoute,communication.setCommunicationWithRB)
            elif mode!='V2_NOOP':
                c.step23.TaskScheduler.setTaskReturnRoute(env,'Task_5',route); setter={'route':route,'rb_indices':[]}
            else:setter={'route':[],'rb_indices':[]}
            if mode!='V2_NOOP':
                fid=ledger.create_return_flow(raw,logical_destination=route[-1],total_data=float(raw['required_returned_size']),route=route,
                    logical_destination_source='v2_current_task_return_destination',
                    logical_destination_capture_phase='v2_validated_route_action_before_service')
            else:fid=None
            atomic(case/'03_action.json',{'setter':setter,'flow_id':fid,'ledger':ledger.snapshot(),'pre_route_anchor_sha':root['prefix_sha']})
            restore=_install_lifecycle_alias_guard(env.task_manager)
            try:
                for h in range(1,5):
                    if time.perf_counter()-start>180:raise TimeoutError('ROOT_CPU_TIMEOUT')
                    if h>1 and mode=='V2_ROUTE_RB' and any(str(x.getTaskId())=='Task_5' for xs in env.task_manager._returning_tasks.values() for x in xs):
                        communication.setCommunicationWithRB(env,'Task_5',[0])
                    # Identical Comp NOOP/Mob HOLD in all four branches.
                    c.step23._install_cpu_callback(env,{})
                    patterns={str(uid):{'angle':float(info.get('angle',0.)),'phi':float(info.get('phi',0.)),'speed':0.}
                              for uid,info in env.traffic_manager.getUAVTrafficInfos().items()}
                    c.step23.TrafficScheduler.setUAVMobilityPatterns(env,patterns)
                    event_start=len(env.pi_jwm_transfer_events)
                    native_start=len(clocks.get('transmission_events',[]))
                    before=task_snapshot(t); env.step(); receipt['diagnostic_steps']+=1
                    transfers=[dict(x) for x in env.pi_jwm_transfer_events[event_start:]]
                    atomic(case/f'raw_h{h}.json',{'transfers':transfers,'after':task_snapshot(t),'events':receipt['events']})
                    native_events=copy.deepcopy(clocks.get('transmission_events',[])[native_start:])
                    for ev in native_events:
                        if fid and str(ev.get('task_id'))=='Task_5' and ledger.flow(fid)['status']=='ACTIVE':
                            ledger.apply_transfer_event(ev,observed_task_current_node_id=str(t.getCurrentNodeId()))
                    obs=c.step23._capture(env,frame=6+h,phase='post_env_step_outcome',event_index=12+h,
                        previous_speed_by_entity={r['entity_id']:r['speed_mps'] for r in current['entities']},delta_t_s=.1)
                    row={'horizon':h,'before':before,'after':task_snapshot(t),'time':float(env.simulation_time),
                         'transfers':transfers,'native_transfers':native_events,'ledger':ledger.snapshot(),'clocks':copy.deepcopy(clocks),
                         'raw':obs,'buckets':bucket_ids(env.task_manager),'elapsed_s':time.perf_counter()-start}
                    atomic(case/f'04_h{h}.json',row)
                    atomic(case/'runtime_events.json',receipt['events'])
            finally:restore()
        receipt.update(status='COMPLETED',elapsed_s=time.perf_counter()-start)
        atomic(case/'05_receipt.json',receipt)
        raise StopCapture()
    try:
        c.collect_trajectory(2026092318,2026092418,'formal-v1-sim-2026092318-policy-2026092418',on_decision=hook)
    except StopCapture: pass
    except BaseException as exc:
        receipt.update(status='BLOCKED',exception_type=type(exc).__name__,exception=str(exc),elapsed_s=time.perf_counter()-start)
        atomic(case/'failure.json',receipt); raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True)
    main(p.parse_args().mode)
