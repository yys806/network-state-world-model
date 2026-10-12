"""Independent reconstruction of Gate 1 from saved native runtime evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code/artifacts/audit/pi_jwm_step6_4n_20261012_v5'
MODES = ('LEGACY_ROUTE', 'V2_ROUTE_NO_RB', 'V2_ROUTE_RB', 'V2_NOOP')

def audit(path=OUT):
    roots = {m: json.loads((path/m/'02_root.json').read_text(encoding='utf8')) for m in MODES}
    rows = {m: [json.loads((path/m/f'04_h{h}.json').read_text(encoding='utf8')) for h in range(1,5)] for m in MODES}
    receipts = {m: json.loads((path/m/'05_receipt.json').read_text(encoding='utf8')) for m in MODES}
    same = all(len({r[k] for r in roots.values()}) == 1 for k in ('prefix_sha','current_sha','rng_sha','time'))
    rb = rows['V2_ROUTE_RB'][0]
    transfers = [e for e in rb['native_transfers'] if e['task_id']=='Task_5']
    flow = rb['ledger']['logical_flow_rows'][0]
    allocation = [e for e in receipts['V2_ROUTE_RB']['events'] if e['event']=='_allocate_communication_RBs']
    checks = {
        'paired_pre_route_prefix_state_rng':same,
        'bounded_cpu':all(r['status']=='COMPLETED' and r['diagnostic_steps']==4 and not r['GPU'] and not r['locked_test'] for r in receipts.values()),
        'legacy_tti_failure':rows['LEGACY_ROUTE'][0]['after']['failure_reason_code']==1,
        'return_independent_input_clock':all(r['after']['last_transmission_time']==roots[m]['last_input_clock'] for m in MODES for r in rows[m]),
        'no_service_clock_not_refreshed':all(r['clocks']['return_clocks']['Task_5']==roots['V2_ROUTE_NO_RB']['time'] and r['after']['failure_reason_code']==-1 for r in rows['V2_ROUTE_NO_RB']),
        'same_slot_native_rb_consumed':bool(allocation and allocation[0]['time']==roots['V2_ROUTE_RB']['time'] and allocation[0]['activated'].get('Task_5',{}).get('RB_Nos')==[0]),
        'native_return_transmission_completion':bool(len(transfers)==1 and transfers[0]['stage_or_hop_completed'] and transfers[0]['time']==roots['V2_ROUTE_RB']['time'] and transfers[0]['delivered_data']==roots['V2_ROUTE_RB']['task_raw']['required_returned_size']),
        'ledger_conservation':flow['e2e_delivered']+flow['e2e_remaining']==flow['total_data'] and flow['status']=='COMPLETED',
        'noop_keeps_waiting_return':all(any(x['task_id']=='Task_5' for x in r['buckets']['_waiting_to_return_tasks']) and not r['ledger']['logical_flow_rows'] for r in rows['V2_NOOP']),
    }
    return {'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,
        'real_roots':1,'branch_steps':16,'horizon':4,'GPU':False,'locked_test':False,
        'native_transfer':transfers,'collector_event_discrepancy':{
            'legacy_collector_delivered_data':rb['transfers'][0]['delivered_data'],
            'native_useful_return_delivery':transfers[0]['delivered_data'],
            'reason':'historical collector caps event by Input task_size; v2 uses native Return requirement and method result',
            'historical_artifacts_modified':False},
        'inventory':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*.json') if p.name!='gate1_acceptance.json'}}

if __name__=='__main__':
    result=audit()
    (OUT/'gate1_acceptance.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'checks':result['checks']}))
    raise SystemExit(0 if result['status']=='PASS' else 1)
