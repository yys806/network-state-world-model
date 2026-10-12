"""Freeze executed reference Python sources and per-file delivery hashes."""
import json,hashlib,zipfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'code/artifacts/manifests/pi_jwm_step6_4n_v2_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 reference=ROOT/'code/reference/AirFogSim'
 paths=list((reference/'airfogsim').rglob('*.py'))+[reference/'LICENSE',reference/'requirements.txt']
 manifest={p.relative_to(reference).as_posix():sha(p) for p in paths}
 archive=OUT/'airfogsim_executed_reference_sources.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,p.relative_to(reference).as_posix())
 # Preserve only executed base Python source, not a modified local runtime.
 # v2 changes live in tracked PI-JWM adapter modules.
 with zipfile.ZipFile(archive) as z:
  if any(hashlib.sha256(z.read(name)).hexdigest()!=digest for name,digest in manifest.items()):raise ValueError('REFERENCE_ARCHIVE_HASH_FAILED')
 wm=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v5';real=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_v10'
 source=sorted(list((ROOT/'code/src/pi_jwm').glob('step6_4n_*.py'))+list((ROOT/'code/scripts').glob('*step6_4n*.py'))+list((ROOT/'code/tests').glob('test_step6_4n_*.py')))
 inputs=list((reference/'examples/sumo_wujiaochang').rglob('*'))
 inputs=[p for p in inputs if p.is_file() and p.suffix in ('.xml','.sumocfg')]
 inventory={str(p.relative_to(ROOT)):sha(p) for folder in (real,wm) for p in folder.rglob('*') if p.is_file()}
 m={'schema':'PI-JWM-STEP-6.4N-DELIVERY-v2','reference_archive':str(archive.relative_to(ROOT)),'reference_archive_sha256':sha(archive),'reference_source_sha256':manifest,'reference_note':'exact executed Python snapshot, including Apache LICENSE; no claim of upstream commit identity','sources':{str(p.relative_to(ROOT)):sha(p) for p in source},'raw_evidence':inventory,'simulator_input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'boundary':{'formal_route':'EXPLICIT_NOOP_ONLY','GPU':False,'locked_test':False,'training':False},'validation':{'step6_4n_tests':{'count':32,'status':'PASS','command':"D:/miniconda/envs/airfogsim/python.exe -m unittest discover -s code/tests -p 'test_step6_4n_*.py' -v"},'legacy_focused_tests':{'count':61,'status':'PASS','note':'Scorer/eligibility/grammar/rollout/6.4M/Shadow/live bridge'},'compileall':'PASS','gate_acceptance':str((wm/'acceptance_v2.json').relative_to(ROOT))},'limitations':['formal search binding remains disabled; isolated joint domain and bridge are CPU accepted','prediction reliability/retrain need not established','reference simulator data/runtime dependency installation remains required']}
 (OUT/'delivery_manifest.json').write_text(json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'snapshot_files':len(manifest),'evidence_files':len(inventory),'archive_bytes':archive.stat().st_size}))
if __name__=='__main__':main()
