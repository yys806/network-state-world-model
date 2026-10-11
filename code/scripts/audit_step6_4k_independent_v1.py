"""Independent checks for Step 6.4K receipts; does not trust summary counters."""
from __future__ import annotations
import gzip,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/'code/artifacts/audit/pi_jwm_step6_4k_return_shadow_20261011_v4'
WM=ROOT/'code/artifacts/audit/pi_jwm_step6_4k_frozen_compatibility_20261011_v4'
OUT=LEDGER/'03_independent_audit.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def main():
 summary=json.loads((LEDGER/'02_validation_summary.json').read_text())
 roots=task_records=conflicts=blocks=waiting=known=mapping=0; flow_ids=[]
 for p in sorted(LEDGER.glob('*_ledger.jsonl.gz')):
  for line in gzip.open(p,'rt',encoding='utf8'):
   r=json.loads(line); roots+=1
   if len({f['flow_id'] for f in r['flows']}) != len(r['flows']): raise ValueError('DUPLICATE_FLOW_ID')
   for f in r['flows']:
    if f['flow_id'] not in flow_ids: flow_ids.append(f['flow_id'])
   for t in r['tasks']:
    task_records+=1
    if 'mapping_error' in t: mapping+=1; continue
    conflicts+=bool(t['fp64_fp32_conflict']); blocks+=bool(t['potential_fixed_support_block_fp32']); waiting+=bool(t['waiting_route_absent']); known+=bool(t['return_known'])
 wmrec=json.loads((WM/'04_acceptance_receipt.json').read_text()); ident=json.loads((WM/'00_identity.json').read_text())
 checks={'trajectory_ledger_files':len(list(LEDGER.glob('*_ledger.jsonl.gz')))==summary['trajectories'],'task_records':task_records==summary['counts']['task_records'],'decision_roots':roots==summary['counts']['decision_roots'],'fp64_fp32_conflicts':conflicts==summary['counts']['fp64_fp32_conflict'],'potential_blocks_fp32':blocks==summary['counts']['potential_fixed_support_block_fp32'],'waiting_without_return':waiting==summary['counts']['waiting_route_absent'],'known_return_records':known==summary['counts']['return_known'],'mapping_errors_reproduced':mapping==summary['counts']['mapping_errors'],'frozen_existing_h4_pass':wmrec['frozen_wm_existing_return_h4']=='PASS','frozen_shadow_h4_pass':wmrec['frozen_wm_shadow_return']=='FROZEN_WM_SHADOW_H4_PASS','formal_scoreability_not_claimed':wmrec['formal_planner_scoreability']=='NOT_TESTED','gpu_locked_false':not ident['GPU'] and not ident['locked_test']}
 out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'recomputed':{'task_records':task_records,'decision_roots':roots,'fp64_fp32_conflicts':conflicts,'potential_blocks_fp32':blocks,'waiting_without_return':waiting,'known_return_records':known,'mapping_errors':mapping,'unique_flow_ids':len(flow_ids)},'ledger_sha256':{p.name:sha(p) for p in sorted(LEDGER.glob('*_ledger.jsonl.gz'))},'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
