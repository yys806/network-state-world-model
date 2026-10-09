"""Independent CPU protocol audit; no GPU/cloud actions."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009_r23'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=json.loads((OUT/'00_protocol.json').read_text(encoding='utf-8'));m=json.loads((OUT/'01_manifest.json').read_text(encoding='utf-8'))
 assert c['status']=='CPU_PROTOCOL_FROZEN_GPU_NOT_STARTED' and c['episodes']==2 and c['max_searches']==16 and c['search_seed']==6311
 assert c['method']=='S-CEM' and c['K']==4 and c['rho']==.2 and c['B_WM']==512 and c['H']==4 and c['batch_size']==16 and c['locked_test'] is False
 assert c['checkpoint_sha256']==sha(ROOT/c['checkpoint_path']) and c['normalization_sha256']==sha(ROOT/c['normalization_path'])
 assert c['anchor_manifest_sha256']==sha(ROOT/c['anchor_manifest']) and m['manifest_sha256']==c['anchor_manifest_sha256']
 assert [e['sample_id'] for e in m['episodes']]==[e['sample_id'] for e in c['episodes_manifest']]
 for e in c['episodes_manifest']:
  assert sha(ROOT/e['raw_path'])==e['raw_sha256'] and len(e['initial_task_ids'])==len(set(e['initial_task_ids']))
 for p,h in c['source_sha256'].items():assert sha(ROOT/p)==h,p
 return {'verdict':'PASS','episodes':2,'max_searches':16,'GPU':'NOT_STARTED','locked_test':False,'execution_config_id':c['execution_config_id']}
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
