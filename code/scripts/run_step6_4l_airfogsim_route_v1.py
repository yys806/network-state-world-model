"""Real AirFogSim A/B Route setter probe for STEP 6.4L.

No model is imported in this process.  A and B replay the same frozen
seed/policy prefix; B calls only the real TaskScheduler return-route setter.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/reference/AirFogSim/examples')]
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v7'
SIM_SEED,POLICY_SEED=2026092318,2026092418
TARGET_FRAME=6
class StopCapture(BaseException): pass
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def atomic(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8'); q.replace(p)
def run(mode):
    import collect_step5_5_formal_raw_v1 as c
    started=time.perf_counter(); rec={'schema':'PI-JWM-STEP-6.4L-AIRFOGSIM-ROUTE-v1','mode':mode,'simulator_seed':SIM_SEED,'policy_seed':POLICY_SEED,'target_frame':TARGET_FRAME,'GPU':False,'locked_test':False,'formal_semantics_modified':False,'stages':{}}
    atomic(OUT/f'{mode}_00_start.json',rec)
    def hook(env,decisions,steps,config,communication):
        frame=int(decisions[-1]['frame_index'])
        if frame==TARGET_FRAME:
            current=decisions[-1]; task=next((x for x in current.get('tasks',[]) if str(x.get('task_id'))=='Task_5'),None)
            if task is None:
                rec['stages']['root']='BLOCKED_TASK5_MISSING'; atomic(OUT/f'{mode}_failure.json',rec); raise StopCapture()
            prefix={'environment':{'seed':SIM_SEED,'policy_seed':POLICY_SEED,'wired_edges':c.WIRED_EDGES},'decisions':decisions,'steps':steps}
            runtime_tasks=c.step23._all_runtime_tasks(env)
            deadlines={}
            for tid,obj in runtime_tasks.items():
                try: deadlines[str(tid)]={'arrival_time_s':float(obj.getTaskArrivalTime()),'deadline_s':float(obj.getTaskDeadline())}
                except Exception: pass
            rec['root']={'frame':frame,'simulation_time_s':current['simulation_time_s'],'capture_event_id':current['capture_event_id'],'decision_count':len(decisions),'step_count':len(steps),'causal_prefix_sha256':digest(prefix),'task_5':task,'route_table_before':dict(env.task_return_routes),'deadline_sidecar_tasks':deadlines}
            rec['stages']['root']='LIVE_ROOT_CAPTURE_VALID'; atomic(OUT/f'{mode}_01_root.json',rec['root']); atomic(OUT/f'{mode}_01_causal_prefix.json',prefix)
            route=[str(task['return_destination_id'])]; evidence={'task_id':'Task_5','source':str(task['current_node_id']),'destination':str(task['return_destination_id']),'route':route,'lifecycle':task.get('lifecycle'),'task_cpu_work':task.get('task_cpu_work'),'computed_cpu_work':task.get('computed_cpu_work'),'required_returned_size':task.get('required_returned_size'),'route_table_before':dict(env.task_return_routes)}
            if mode=='B':
                c.step23.TaskScheduler.setTaskReturnRoute(env,'Task_5',route)
                evidence['setter_called']='TaskScheduler.setTaskReturnRoute'; evidence['route_table_after_setter']=dict(env.task_return_routes); evidence['setter_verified']=list(env.task_return_routes.get('Task_5',[]))==route
                env._pi_jwm_diagnostic_route_entries=[{'task_id':'Task_5','task_node_id':str(task['task_node_id']),'route_kind':'return','target_node_id':route[-1],'route_node_ids':route}]
            else: evidence.update({'setter_called':None,'route_table_after_setter':dict(env.task_return_routes),'setter_verified':False})
            env._pi_jwm_diagnostic_disable_route_policy=True
            rec['route']=evidence; atomic(OUT/f'{mode}_02_route.json',evidence)
            if mode=='B' and not evidence['setter_verified']:
                rec['stages']['route']='BLOCKED_SETTER_REJECTED'; atomic(OUT/f'{mode}_failure.json',rec); raise StopCapture()
            rec['stages']['route']='PASS_NOOP_CONTROL' if mode=='A' else 'PASS_REAL_SETTER'
        elif frame==TARGET_FRAME+1:
            selected_step=next((x for x in steps if int(x.get('frame_index',-1))==TARGET_FRAME),None)
            post={'frame':frame,'simulation_time_s':decisions[-1]['simulation_time_s'],'capture_event_id':decisions[-1]['capture_event_id'],'step_count':len(steps),'causal_prefix_after_step_sha256':digest({'decisions':decisions,'steps':steps}),'task_5_after':next((x for x in decisions[-1].get('tasks',[]) if str(x.get('task_id'))=='Task_5'),None),'route_table_after':dict(env.task_return_routes),'selected_step6':None if selected_step is None else {'action_route':selected_step.get('action',{}).get('route'),'execution':selected_step.get('execution'),'outcome_task_5':next((x for x in selected_step.get('outcome',{}).get('tasks',[]) if str(x.get('task_id'))=='Task_5'),None),'slot_transfer_events':[x for x in selected_step.get('outcome',{}).get('slot_transfer_events',[]) if str(x.get('task_id'))=='Task_5']}}
            rec['real_observation']=post; rec['stages']['real_simulator_alignment']='OBSERVED_ONE_REAL_STEP'; atomic(OUT/f'{mode}_03_real_observation.json',post); rec['stages']['complete']='PASS'; rec['elapsed_seconds']=time.perf_counter()-started; atomic(OUT/f'{mode}_04_final.json',rec); raise StopCapture()
    try: c.collect_trajectory(SIM_SEED,POLICY_SEED,f'step6_4l_{mode}',on_decision=hook)
    except StopCapture: pass
    except BaseException as exc: rec['stages']['exception']={'type':type(exc).__name__,'message':str(exc)}; rec['elapsed_seconds']=time.perf_counter()-started; atomic(OUT/f'{mode}_failure.json',rec)
    return rec
def main():
    OUT.mkdir(parents=True,exist_ok=True); atomic(OUT/'00_protocol.json',{'schema':'PI-JWM-STEP-6.4L-AIRFOGSIM-ROUTE-v1','modes':['A','B'],'target_frame':TARGET_FRAME,'same_seed_prefix':True,'max_real_steps_per_mode':1,'GPU':False,'locked_test':False})
    results={}
    for mode in ('A','B'):
        print(json.dumps({'mode_start':mode}),flush=True); results[mode]=run(mode); print(json.dumps({'mode_done':mode,'stage':results[mode].get('stages')}),flush=True)
    atomic(OUT/'05_acceptance_receipt.json',{'status':'COMPLETED','root_identity':{m:results[m].get('stages',{}).get('root') for m in results},'route_setter':{m:results[m].get('stages',{}).get('route') for m in results},'real_simulator_alignment':{m:results[m].get('stages',{}).get('real_simulator_alignment') for m in results},'GPU':False,'locked_test':False})
if __name__=='__main__': main()
