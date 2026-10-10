"""Verify a transferred 6.4J protocol bundle without regenerating identity."""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def verify_bundle(bundle_root, proto):
    manifest=proto.parent/'01_manifest.json'
    if not proto.exists() or not manifest.exists(): raise RuntimeError('FROZEN_PROTOCOL_BUNDLE_MISSING')
    class Args: pass
    a=Args();a.bundle_root=bundle_root
    c=json.loads(proto.read_text(encoding='utf-8'));m=json.loads(manifest.read_text(encoding='utf-8'))
    if c.get('status')!='CPU_PROTOCOL_FROZEN_GPU_NOT_STARTED' or c.get('locked_test') is not False: raise SystemExit('PROTOCOL_SCOPE_INVALID')
    identity=dict(c);identity['execution_config_id']=None
    if c.get('execution_config_id')!=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise SystemExit('EXECUTION_CONFIG_ID_MISMATCH')
    if m.get('manifest_sha256')!=sha(a.bundle_root/c['anchor_manifest']): raise SystemExit('ANCHOR_MANIFEST_SHA_MISMATCH')
    for item in c['episodes_manifest']:
        if sha(a.bundle_root/item['raw_path'])!=item['raw_sha256']: raise SystemExit('RAW_SHA_MISMATCH')
    for rel,expected in c['source_sha256'].items():
        if sha(a.bundle_root/rel)!=expected: raise SystemExit(f'SOURCE_SHA_MISMATCH:{rel}')
    for rel,key in ((c['checkpoint_path'],'checkpoint_sha256'),(c['normalization_path'],'normalization_sha256')):
        if sha(a.bundle_root/rel)!=c[key]: raise SystemExit(f'DEPENDENCY_SHA_MISMATCH:{rel}')
    if m.get('episodes')!=c['episodes_manifest']:raise RuntimeError('EPISODES_MANIFEST_MISMATCH')
    return {'verdict':'PASS','execution_config_id':c['execution_config_id'],'protocol':str(proto.relative_to(a.bundle_root)).replace('\\','/'),'gpu':'NOT_STARTED','locked_test':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle-root',type=Path,default=ROOT);a=ap.parse_args()
    proto=a.bundle_root/'code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009_r27/00_protocol.json'
    print(json.dumps(verify_bundle(a.bundle_root,proto),ensure_ascii=False))
if __name__=='__main__': main()
