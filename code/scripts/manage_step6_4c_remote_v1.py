"""Authorized calibration deployment/launch or read-only SHA-checked snapshot.

The password is read interactively and never serialized. No historical raw writes.
"""
import argparse
import getpass
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
import paramiko

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/root/autodl-tmp/pi-jwm-step6-3d'
PYTHON='/root/miniconda3/bin/python'
REL='code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003'

def main():
    p=argparse.ArgumentParser();p.add_argument('user');p.add_argument('action',choices=['deploy','launch','snapshot'])
    args=p.parse_args()
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.nmb2.seetacloud.com',port=18448,username=args.user,
                   password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=15)
    def command(cmd):
        _,o,e=client.exec_command(cmd);text=o.read().decode();error=e.read().decode();code=o.channel.recv_exit_status()
        if code: raise RuntimeError('remote command failed '+str(code)+': '+error[-1500:])
        return text
    sftp=client.open_sftp()
    if args.action=='deploy':
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        command(f'git -C {REMOTE} fetch origin')
        command(f'git -C {REMOTE} merge --ff-only origin/main')
        actual=command(f'git -C {REMOTE} rev-parse HEAD').strip()
        if actual!=head:raise ValueError('remote HEAD mismatch')
        config=json.loads((ROOT/REL/'07_calibration_execution_config.json').read_text())
        # Verify all frozen bytes in one remote read, then fix only newline differences
        # in source checkout files. Never replace historical metadata or raw results.
        payload=json.dumps(config['source_sha256'])
        script='import json,pathlib,hashlib; d=json.loads('+repr(payload)+'); print(json.dumps([n for n,h in d.items() if not pathlib.Path(n).exists() or hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()!=h]))'
        wrong=json.loads(command(f'cd {REMOTE} && {PYTHON} -c {shlex.quote(script)}'))
        for name in wrong:
            local=(ROOT/name).read_bytes()
            with sftp.open(REMOTE+'/'+name,'rb') as f: remote=f.read()
            if not name.endswith('.py') or local.replace(b'\r\n',b'\n')!=remote.replace(b'\r\n',b'\n'):
                raise ValueError('non-newline source mismatch: '+name)
            with sftp.open(REMOTE+'/'+name,'wb') as f:f.write(local)
        output=command(f'cd {REMOTE} && {PYTHON} code/scripts/run_step6_4c_mh_budget_calibration_v1.py')
        dirty=command(f'git -C {REMOTE} status --porcelain --untracked-files=no').strip()
        if dirty:raise ValueError('remote tracked dirty after deployment')
        print(json.dumps({'remote_commit':actual,'exact_byte_newline_transfers':wrong,'preflight_output':output.strip(),'tracked_clean':True}))
    elif args.action=='launch':
        # Runner independently repeats every gate. No restart or Stage B command exists here.
        count=command(f'find {REMOTE}/{REL}/solve_results/calibration -name "*.json" 2>/dev/null | wc -l').strip()
        if count!='0':raise ValueError('initial launch count not zero')
        output=command(f'cd {REMOTE} && mkdir -p {REL}/runtime && nohup {PYTHON} -u code/scripts/run_step6_4c_mh_budget_calibration_v1.py --execute > {REL}/runtime/formal_calibration.log 2>&1 < /dev/null &')
        print(json.dumps({'launch_requested':True,'scope':'MH-only 96-case calibration, no Stage B','output':output}))
    else:
        script="import pathlib,json,hashlib; p=pathlib.Path("+repr(REL)+"); print(json.dumps([{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob('*')) if f.is_file() and (f.suffix=='.json' or f.name=='formal_calibration.log')]))"
        files=json.loads(command(f'cd {REMOTE} && {PYTHON} -c {shlex.quote(script)}'))
        copied=[]
        for item in files:
            name=item['path'];dest=ROOT/name
            # Immutable preregistration must never be overwritten by snapshot.
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']:continue
            if '/solve_results/' not in name and '/runtime/' not in name and Path(name).name[:2] not in ['09','10','11','12','13','14']:
                raise ValueError('immutable protocol SHA changed: '+name)
            with sftp.open(REMOTE+'/'+name,'rb') as f:data=f.read()
            if hashlib.sha256(data).hexdigest()!=item['sha256']:
                if '/runtime/' in name or Path(name).name=='09_runtime_status.json':continue
                raise ValueError('non-atomic result changed during snapshot')
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.download');tmp.write_bytes(data);tmp.replace(dest)
            copied.append(name)
        status=ROOT/REL/'09_runtime_status.json'
        print(json.dumps({'files_copied':len(copied),'local_persistent_storage':str(ROOT/REL),
            'status':json.loads(status.read_text()) if status.exists() else None,
            'log_tail':(ROOT/REL/'runtime/formal_calibration.log').read_text()[-2500:] if (ROOT/REL/'runtime/formal_calibration.log').exists() else None}))
    sftp.close();client.close()

if __name__=='__main__':main()
