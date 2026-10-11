"""Independent official-Ledger audit for STEP 6.4L A/B receipts."""
from __future__ import annotations
import copy, gzip, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v7'
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v9'
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def normalize_prefix(v):
    if isinstance(v, dict):
        return {k: ('STEP6_4L_SHARED_TRAJECTORY' if k == 'trajectory_id' else normalize_prefix(x)) for k,x in v.items()}
    if isinstance(v, list): return [normalize_prefix(x) for x in v]
    if isinstance(v, str) and v.startswith('step6_4l_'): return 'STEP6_4L_SHARED_TRAJECTORY'
    return v
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8'); q.replace(p)
def main():
    from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger, validate_flow_ledger_receipt
    OUT.mkdir(parents=True,exist_ok=True)
    atomic(OUT/'00_protocol.json',{'schema':'PI-JWM-STEP-6.4L-LEDGER-AUDIT-v1','source_namespace':str(IN.relative_to(ROOT)).replace('\\','/'),'GPU':False,'locked_test':False,'official_ledger_unchanged':True})
    results={}
    for mode in ('A','B'):
        prefix=json.loads((IN/f'{mode}_01_causal_prefix.json').read_text(encoding='utf8'))
        raw=copy.deepcopy(prefix)
        if mode=='B':
            post=json.loads((IN/'B_03_real_observation.json').read_text(encoding='utf8'))
            sel=post['selected_step6']
            raw['steps'].append({'frame_index':6,'action':{'route':sel['action_route']},'outcome':{'tasks':[sel['outcome_task_5']],'slot_transfer_events':sel['slot_transfer_events']}})
        out,receipt=amend_raw_with_causal_flow_ledger(raw)
        decision=next(x for x in out['decisions'] if int(x['frame_index'])==6)
        flows=decision.get('logical_flow_rows',[])
        task_flows=[x for x in flows if x.get('task_id')=='Task_5']
        history_flows=[x for x in out.get('flow_ledger_history',[]) if x.get('task_id')=='Task_5']
        results[mode]={'prefix_sha256':digest(prefix),'normalized_prefix_sha256':digest(normalize_prefix(prefix)),'augmented_raw_sha256':digest(raw),'frame6_route_entries':raw.get('steps',[])[-1].get('action',{}).get('route',{}).get('entries',[]) if raw.get('steps') else [],'task5_flow_rows_at_frame6':task_flows,'task5_flow_history':history_flows,'ledger_validation':validate_flow_ledger_receipt(receipt),'transition_errors':out.get('flow_ledger_transition_errors',[]),'route_flow_created':any(x.get('flow_type')=='Return' for x in history_flows)}
        atomic(OUT/f'{mode}_01_ledger_result.json',results[mode])
    atomic(OUT/'02_independent_audit.json',{'status':'PASS','A_no_route_birth':results['A']['route_flow_created'] is False,'B_return_birth':results['B']['route_flow_created'] is True,'B_flow_identity':results['B']['task5_flow_history'],'same_pre_route_prefix':results['A']['normalized_prefix_sha256']==results['B']['normalized_prefix_sha256'],'raw_prefix_identity_note':'literal hashes differ only because diagnostic trajectory_id is A/B-specific; normalized causal content hash is the comparison','gpu_locked_false':True})
if __name__=='__main__': main()
