"""Operational SSH control/backup for the independent 6.4H namespace only."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import paramiko

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/root/autodl-tmp/pi-jwm-step6-3d'
REL='code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2'
PYTHON='/root/miniconda3/bin/python'
GATE='dc0e6c94b2353bdaf1c671e52bde2a978a991bf8'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('probe','deploy','deploy-bundle','launch','snapshot'))
    a=p.parse_args();user=getpass.getpass('SSH user: ');password=getpass.getpass('SSH password: ')
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.nmb2.seetacloud.com',port=18448,username=user,password=password,
              look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=15,banner_timeout=15)
    password=None;user=None
    def cmd(command,timeout=120):
        _,out,err=c.exec_command(command,timeout=timeout)
        data=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
        if code: raise RuntimeError('remote exit '+str(code)+': '+error[-2500:])
        return data.strip()
    processes=cmd("ps -eo pid,args | grep -E '([r]un_step6_.*[.]py|[s]tep6_4h_s_cem_budget_v1[.]py)' || true")
    gpu=cmd('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv,noheader')
    dirty=cmd('git -C '+REMOTE+' status --porcelain --untracked-files=no')
    disk=cmd('df -B1 --output=avail '+REMOTE+' | tail -1')
    if 'NVIDIA GeForce RTX 3080 Ti' not in gpu: raise ValueError('wrong GPU')
    if int(disk)<5*1024**3: raise ValueError('insufficient persistent disk')
    if a.action=='probe':
        print(json.dumps(dict(SSH_authenticated=True,GPU=gpu,free_disk_bytes=int(disk),
            remote_commit=cmd('git -C '+REMOTE+' rev-parse HEAD'),tracked_clean=not dirty,
            scientific_processes=processes,GPU_started=False)))
    elif a.action in ('deploy','deploy-bundle','launch'):
        if processes: raise ValueError('existing scientific process; no duplicate or automatic restart')
        if dirty: raise ValueError('remote tracked dirty')
        if a.action=='deploy':
            cmd('git -C '+REMOTE+' fetch origin')
            cmd('git -C '+REMOTE+' merge --ff-only '+GATE)
        if a.action=='deploy-bundle':
            bundle=ROOT/REL/'deployment_dc0e6c9.bundle'
            if not bundle.exists(): raise ValueError('local exact Git bundle missing')
            expected=hashlib.sha256(bundle.read_bytes()).hexdigest()
            target=REMOTE+'/'+REL+'/deployment_dc0e6c9.bundle'
            cmd('mkdir -p '+REMOTE+'/'+REL)
            sf=c.open_sftp();sf.put(str(bundle),target);sf.close()
            actual=cmd('sha256sum '+shlex.quote(target)).split()[0]
            if actual!=expected: raise ValueError('bundle transfer SHA mismatch')
            cmd('git -C '+REMOTE+' bundle verify '+shlex.quote(target))
            cmd('git -C '+REMOTE+' fetch '+shlex.quote(target)+' refs/heads/main:refs/remotes/origin/main')
            cmd('git -C '+REMOTE+' merge --ff-only '+GATE)
            print(json.dumps(dict(deployment='SHA_verified_incremental_Git_bundle',bundle_SHA256=expected)),flush=True)
        for ref in ('HEAD','origin/main'):
            if cmd('git -C '+REMOTE+' rev-parse '+ref)!=GATE: raise ValueError('exact freeze gate mismatch')
        runner='code/scripts/step6_4h_s_cem_budget_v1.py'
        preflight=cmd('cd '+REMOTE+' && '+PYTHON+' '+runner)
        if a.action=='launch':
            # Refuse prior H activity here; resume needs a separate explicit operation.
            state=cmd('cd '+REMOTE+' && '+PYTHON+' -c '+shlex.quote(
                "from pathlib import Path; p=Path('"+REL+"'); print(len(list((p/'solve_results').glob('*.json')))+len(list(p.glob('launch_*.json')))+len(list(p.glob('STOP_*.json'))))"))
            if state!='0': raise ValueError('not a fresh empty H invocation; inspect rather than restart')
            log=REL+'/formal_run.log'
            command='cd '+REMOTE+' && nohup '+PYTHON+' -u '+runner+' --execute --gate-commit '+GATE+' > '+log+' 2>&1 < /dev/null &'
            cmd('sh -c '+shlex.quote(command))
        print(json.dumps(dict(action=a.action,gate_commit=GATE,CPU_preflight=preflight,
                              launch_requested=a.action=='launch',GPU=gpu,free_disk_bytes=int(disk))))
    else:
        sf=c.open_sftp()
        program="import pathlib,json,hashlib; p=pathlib.Path("+repr(REL)+"); print(json.dumps([{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob('*')) if f.is_file() and f.suffix in ('.json','.zip','.log')]))"
        files=json.loads(cmd('cd '+REMOTE+' && '+PYTHON+' -c '+shlex.quote(program)))
        copied=0
        for row in files:
            local=ROOT/row['path'];name=local.name
            if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==row['sha256']: continue
            if local.exists() and (name[:2].isdigit() or '/solve_results/' in row['path']): raise ValueError('immutable protocol/raw SHA mismatch')
            with sf.open(REMOTE+'/'+row['path'],'rb') as f: data=f.read()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:
                if name in ('runtime_status.json','formal_run.log'): continue
                raise ValueError('non-atomic SFTP evidence')
            local.parent.mkdir(parents=True,exist_ok=True);tmp=local.with_suffix(local.suffix+'.download');tmp.write_bytes(data);tmp.replace(local);copied+=1
        sf.close()
        progress=ROOT/REL/'runtime_status.json'
        print(json.dumps(dict(copied=copied,local_persistent_storage=str(ROOT/REL),GPU=gpu,
            scientific_processes=processes,progress=None if not progress.exists() else json.loads(progress.read_text()),
            STOP_receipts=[p.name for p in (ROOT/REL).glob('STOP_*.json')],raw_count=len(list((ROOT/REL/'solve_results').glob('*.json'))))))
    c.close()


if __name__=='__main__': main()
