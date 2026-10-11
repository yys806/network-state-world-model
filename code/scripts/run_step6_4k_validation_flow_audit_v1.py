"""All frozen dev_validation decisions, official causal Ledger replay, read-only."""
from __future__ import annotations
import gzip,json,hashlib,os,sys,time,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4k_return_shadow_20261011_v4'
DATA=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def write_gz(p,rows):
    tmp=p.with_suffix(p.suffix+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,p)

def classify(t,flows):
    required=('task_cpu_work','computed_cpu_work','required_returned_size','current_node_id','return_destination_id','lifecycle','task_id')
    missing=[k for k in required if k not in t]
    if missing:
        return {'task_id':t.get('task_id'),'lifecycle':t.get('lifecycle'),'mapping_error':'MISSING_FIELDS:'+','.join(missing),'raw':t}
    total,done=float(t['task_cpu_work']),float(t['computed_cpu_work'])
    rem=total-done; rem32=float(np.float32(total)-np.float32(done))
    rs=[f for f in flows if f['task_id']==t['task_id'] and f['flow_type']=='Return']
    terminal=t['lifecycle'] in ('done','failed','completed')
    requires=float(t['required_returned_size'])>0 and t['current_node_id']!=t['return_destination_id']
    known=bool(rs);active=any(f['presence'] for f in rs)
    return {'task_id':t['task_id'],'lifecycle':t['lifecycle'],'terminal':terminal,
      'cpu_total_fp64':total,'cpu_computed_fp64':done,'remaining_fp64':rem,'remaining_fp32':max(0.,rem32),
      'exact_computed':rem<=0,'fp32_computed':rem32<=0,'fp64_fp32_conflict':rem>0 and rem32<=0,
      'requires_return_current_host':requires,'return_known':known,'return_presence':active,
      'return_ids':[f['flow_id'] for f in rs],'return_states':[f['status'] for f in rs],
      'computing_unfinished':t['lifecycle']=='computing' and rem>0,
      'waiting_route_absent':t['lifecycle']=='waiting_to_return' and not known,
      'established_untransmitted':any(f['status']=='ACTIVE' and f['e2e_delivered']==0 for f in rs),
      'return_completed':any(f['status']=='COMPLETED' for f in rs),
      'potential_fixed_support_block_fp32':not terminal and rem32<=0 and requires and not known,
      'potential_fixed_support_block_fp64':not terminal and rem<=0 and requires and not known,
      'raw':t}

def replay(raw):
    tagged=[('logical_flow_rows' in d) for d in raw['decisions']]
    if any(tagged) and not all(tagged):raise ValueError('PARTIAL_LEDGER_RAW')
    if all(tagged):return raw,{'passed':None,'reason':'already_ledger_amended_no_second_birth'},'ALREADY_AMENDED'
    amended,receipt=amend_raw_with_causal_flow_ledger(raw)
    return amended,receipt,'OFFICIAL_LEDGER_REPLAY'

def scan(tid,raw):
    amended,receipt,mode=replay(raw);rows=[];counts=Counter();unique=set();conflicts=[];examples=[]
    created={};steps={int(s['frame_index']):s for s in raw.get('steps',[])}
    for d in amended['decisions']:
        fs=d['logical_flow_rows'];frame=int(d['frame_index'])
        if len({f['flow_id'] for f in fs})!=len(fs) or len({f['flow_index'] for f in fs})!=len(fs):raise ValueError('DUPLICATE_LEDGER_IDENTITY')
        for f in fs:
            if f['flow_id'] not in created:
                evidence=[{'frame':int(s['frame_index']),'route':e} for s in raw.get('steps',[]) if int(s['frame_index'])<frame
                    for e in s['action']['route']['entries'] if e['task_id']==f['task_id'] and e.get('route_kind','offload')==('return' if f['flow_type']=='Return' else 'offload')]
                created[f['flow_id']]={'first_visible_decision_frame':frame,'creation_action_evidence':evidence[-1:]}
        task_rows=[classify(t,fs) for t in d['tasks']]
        prior_return=set(e['task_id'] for e in steps.get(frame-1,{}).get('action',{}).get('route',{}).get('entries',[]) if e.get('route_kind')=='return')
        for t in task_rows:
            if 'mapping_error' in t:
                counts['task_records']+=1;counts['mapping_errors']+=1;unique.add(t.get('task_id'))
                continue
            counts['task_records']+=1;counts['nonterminal_task_records']+=not t['terminal'];unique.add(t['task_id'])
            counts['raw_lifecycle_'+t['lifecycle']]+=1
            for k in ('computing_unfinished','waiting_route_absent','return_known','return_presence','established_untransmitted','return_completed','fp64_fp32_conflict','potential_fixed_support_block_fp32','potential_fixed_support_block_fp64'):
                counts[k]+=bool(t[k])
            counts['old_previous_frame_waiting_721']+=t['lifecycle']=='waiting_to_return' and t['task_id'] not in prior_return
            counts['old_previous_frame_potential_724']+=not t['terminal'] and t['fp32_computed'] and t['requires_return_current_host'] and t['task_id'] not in prior_return
            if t['fp64_fp32_conflict']:conflicts.append({'trajectory':tid,'frame':frame,**t})
            if t['return_known'] and len(examples)<4:examples.append({'trajectory':tid,'frame':frame,'task_id':t['task_id'],'kind':'EXISTING_RETURN'})
        blocking=[t['task_id'] for t in task_rows if 'mapping_error' not in t and t['potential_fixed_support_block_fp32']]
        counts['roots_with_block']+=bool(blocking);counts['roots_multiple_blocks']+=len(blocking)>1;counts['decision_roots']+=1
        rows.append({'trajectory':tid,'frame':frame,'time':d['simulation_time_s'],'capture_event_id':d['capture_event_id'],
                     'tasks':task_rows,'flows':fs,'carrying':d['carrying_rows'],'blocking_tasks_fp32':blocking})
        for e in steps.get(frame,{}).get('action',{}).get('route',{}).get('entries',[]):
            if e.get('route_kind')=='return':examples.append({'trajectory':tid,'frame':frame,'task_id':e['task_id'],'kind':'RETURN_ROUTE_ACTION','route':e})
    counts['independent_tasks']=len(unique)
    return rows,dict(counts),conflicts,examples,created,receipt,mode

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    manifest=json.loads((DATA/'split_manifest.json').read_text());tids=sorted(manifest['dev_validation'])
    atomic(OUT/'00_protocol.json',{'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'scope':'SHADOW_ONLY','selection':tids,'manifest_sha256':sha(DATA/'split_manifest.json'),
      'GPU':False,'locked_test':False,'max_wm_unique_transitions':64,'wm_timeout_seconds':1800,'audit_timeout_seconds':600,
      'old_statistics':'721/724 are previous-frame proxies, not missing-Flow facts'})
    total=Counter();files=[];conf=[];examples=[]
    try:
        for tid in tids:
            if time.perf_counter()-start>600:raise TimeoutError('VALIDATION_AUDIT_TIMEOUT')
            p=DATA/'raw'/f'{tid}.json.gz'
            with gzip.open(p,'rt',encoding='utf8') as f:raw=json.load(f)
            rows,c,errors,cases,created,receipt,mode=scan(tid,raw)
            target=OUT/f'{tid}_ledger.jsonl.gz';write_gz(target,rows)
            atomic(OUT/f'{tid}_receipt.json',{'source_path':str(p.relative_to(ROOT)),'raw_sha256':sha(p),
                'ledger_path':target.name,'ledger_sha256':sha(target),'counts':c,'created':created,'official_ledger_receipt':receipt,'mode':mode})
            total.update(c);conf.extend(errors);examples.extend(cases);files.append(target.name)
            print(json.dumps({'phase':'VALIDATION','trajectory':tid,'seconds':time.perf_counter()-start,'counts':c}),flush=True)
            atomic(OUT/'01_progress.json',{'completed_trajectories':len(files),'counts':dict(total),'elapsed_seconds':time.perf_counter()-start})
        atomic(OUT/'02_validation_summary.json',{'trajectories':len(tids),'counts':dict(total),'numeric_conflicts':conf,
            'case_catalog':examples,'ledger_files':files,'elapsed_seconds':time.perf_counter()-start,
            'unit':'task records repeat the same task across decisions; independent_tasks is trajectory+task',
            'classification':'offline causal snapshots, potential support boundaries, NOT executed scorer rejections'})
    except BaseException as e:
        atomic(OUT/'failure.json',{'type':type(e).__name__,'error':str(e),'seconds':time.perf_counter()-start});raise

if __name__=='__main__':main()
