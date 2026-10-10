"""Freeze and verify the 6.4J two-trajectory Pilot; never starts GPU."""
import argparse,hashlib,json,gzip,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MAN=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/16_validation_anchor_manifest_objective_eligible.json'
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261010_r36'
RAW=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/raw'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git_blob_sha(commit, relative):
 out=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{relative}'])
 return hashlib.sha256(out).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
 if not a.freeze:raise SystemExit('CPU freeze only; GPU launch is a separately gated operation')
 if OUT.exists():raise SystemExit('namespace exists; no overwrite')
 commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip():
  raise SystemExit('tracked worktree must be clean before freezing')
 m=json.loads(MAN.read_text(encoding='utf-8')); groups={}
 for r in m['selected']:
  t=r['sample_id'].split('::',1)[0];groups.setdefault(t,[]).append(r)
 trajectories=sorted(groups,key=lambda s:(hashlib.sha256(s.encode()).hexdigest(),s))[:2]
 eps=[]
 for t in trajectories:
  r=min(groups[t],key=lambda x:(int(x['sample_id'].rsplit('-',1)[1]),x['sample_id']))
  raw=RAW/(t+'.json.gz')
  payload=json.loads(gzip.decompress(raw.read_bytes()))
  frame=int(r['sample_id'].rsplit('-',1)[1]);decision=payload['decisions'][frame]
  eps.append({'trajectory_id':t,'sample_id':r['sample_id'],'simulator_seed':payload['environment']['seed'],'policy_seed':payload['environment']['policy_seed'],'initial_frame':frame,'initial_capture_event_id':decision['capture_event_id'],'initial_simulation_time_s':decision['simulation_time_s'],'raw_path':str(raw.relative_to(ROOT)).replace('\\','/'),'raw_sha256':sha(raw),'initial_task_ids':sorted(str(x['task_id']) for x in decision['tasks']),'initial_task_cohort_sha256':hashlib.sha256(json.dumps(decision['tasks'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'source_manifest_sha256':sha(MAN)})
 sources=['code/src/pi_jwm/step6_4i_episode_v1.py','code/src/pi_jwm/step6_4i_episode_runner_v1.py','code/src/pi_jwm/step6_4i_real_metrics_v1.py','code/scripts/step6_4i_live_planner_v1.py','code/src/pi_jwm/step6_4j_pilot_v1.py','code/scripts/run_step6_4j_pilot_v1.py','code/scripts/collect_step5_5_formal_raw_v1.py','code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py','code/src/pi_jwm/step6_3c_candidate_domain_v1.py','code/src/pi_jwm/step6_4b_live_bridge_v1.py','code/src/pi_jwm/step6_4f_comm_eligibility_v1.py','code/src/pi_jwm/step4_2a_graph_input_extension_v1.py','code/src/pi_jwm/model_ready_sample_contract_v1.py','code/scripts/verify_step6_4j_deployment_v1.py','code/scripts/audit_step6_4j_engineering_v1.py','code/scripts/audit_step6_4j_formal_v1.py','code/scripts/preflight_step6_4j_sumo_v1.py']
 ck=ROOT/'code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt'; norm=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'
 cfg={'schema':'PI-JWM-STEP-6.4J-PILOT-v1','status':'CPU_PROTOCOL_FROZEN_GPU_NOT_STARTED','method':'S-CEM','K':4,'rho':.2,'B_WM':512,'H':4,'batch_size':16,'precision':'FP32','gpu_model':'NVIDIA GeForce RTX 3080 Ti','Route':'EXPLICIT_NOOP_ONLY','fallback':'CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1','latency_mode':'SYNCHRONOUS_PAUSED_SIMULATION','search_seed':6311,'decision_steps_per_episode':8,'episodes':2,'max_searches':16,'formal_performance':'NOT_STARTED','locked_test':False,'checkpoint_path':str(ck.relative_to(ROOT)).replace('\\','/'),'checkpoint_sha256':sha(ck),'normalization_path':str(norm.relative_to(ROOT)).replace('\\','/'),'normalization_sha256':sha(norm),'anchor_manifest':str(MAN.relative_to(ROOT)).replace('\\','/'),'anchor_manifest_sha256':sha(MAN),'selection_rule':'trajectory_id SHA256 UTF8 ascending; first static anchor frame per trajectory; no outcome use','episodes_manifest':eps,'source_hash_basis':'sha256(final_git_commit_blob_bytes)','source_git_commit':commit,'source_sha256':{s:git_blob_sha(commit,s) for s in sources},'stop_gates':{'single_plan_seconds_gt':600,'instance_elapsed_seconds_ge':10200,'scorer_nan_identity_source_duplicate':'STOP_PILOT','setter_or_step_or_feedback':'STOP_EPISODE_NO_RETRY'},'backup_requirement':'raw+zip+per_file_sha+D persistent storage','execution_config_id':None}
 cfg['execution_config_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest();OUT.mkdir(parents=True);(OUT/'00_protocol.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(OUT/'01_manifest.json').write_text(json.dumps({'episodes':eps,'manifest_sha256':sha(MAN),'selection_rule':cfg['selection_rule']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'verdict':'PASS','execution_config_id':cfg['execution_config_id'],'episodes':[e['sample_id'] for e in eps],'GPU':'NOT_STARTED'}))
if __name__=='__main__':main()
