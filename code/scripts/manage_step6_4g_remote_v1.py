"""Credential-interactive deployment/probe/snapshot; never boot an instance."""
import argparse,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[2]
REMOTE='/root/autodl-tmp/pi-jwm-step6-3d'
REL='code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004'
PYTHON='/root/miniconda3/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('probe','deploy','launch','snapshot'))
    p.add_argument('--phase',choices=('T','A','B'));p.add_argument('--commit');p.add_argument('--resume',action='store_true')
    a=p.parse_args();user=input('SSH user: ');password=getpass.getpass('SSH password: ')
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:c.connect('connect.nmb2.seetacloud.com',port=18448,username=user,password=password,look_for_keys=False,allow_agent=False,timeout=12,auth_timeout=12,banner_timeout=12)
    except (OSError,paramiko.SSHException) as e:
        print(json.dumps({'SSH_authenticated':False,'error_type':type(e).__name__,'GPU_started':False,'instance_start_attempted':False}));return
    password=None
    def cmd(value):
        _,o,e=c.exec_command(value,timeout=25);data=o.read().decode();code=o.channel.recv_exit_status()
        if code:raise RuntimeError('remote read/command failed exit='+str(code))
        return data.strip()
    active=cmd("pgrep -af '[r]un_step6_(3d|4[cdefg])_.*\\.py' || true")
    gpu=cmd('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv,noheader')
    if 'NVIDIA GeForce RTX 3080 Ti' not in gpu:raise ValueError('STOP wrong GPU')
    if a.action=='probe':
        print(json.dumps({'SSH_authenticated':True,'read_only':True,'gpu':gpu,
          'remote_commit':cmd('git -C '+REMOTE+' rev-parse HEAD'),
          'tracked_clean':not cmd('git -C '+REMOTE+' status --porcelain --untracked-files=no'),
          'active_scientific_process':bool(active),'GPU_started':False,'instance_start_attempted':False}));c.close();return
    if a.action in ('deploy','launch'):
        if not a.commit or len(a.commit)!=40 or not a.phase:raise ValueError('exact phase/commit required')
        if active:raise ValueError('another scientific runner exists; never duplicate/restart')
        if cmd('git -C '+REMOTE+' status --porcelain --untracked-files=no'):raise ValueError('remote tracked dirty')
        if a.action=='deploy':
            cmd('git -C '+REMOTE+' fetch origin');cmd('git -C '+REMOTE+' merge --ff-only origin/main')
        if cmd('git -C '+REMOTE+' rev-parse HEAD')!=a.commit or cmd('git -C '+REMOTE+' rev-parse origin/main')!=a.commit:raise ValueError('exact Git gate mismatch')
        runner='code/scripts/run_step6_4g_requalification_v1.py'
        preflight=cmd('cd '+REMOTE+' && '+PYTHON+' '+runner+' --phase '+a.phase)
        if a.action=='launch':
            # Deployment after T/A must include committed, SHA verified local
            # phase receipts. Phase and condition checks also repeat in runner.
            log=REL+'/'+a.phase+'/formal_run.log';extra=' --resume' if a.resume else ''
            cmd('cd '+REMOTE+' && mkdir -p '+REL+'/'+a.phase)
            command='cd '+REMOTE+' && nohup '+PYTHON+' -u '+runner+' --phase '+a.phase+' --phase-gate-commit '+shlex.quote(a.commit)+' --execute'+extra+' > '+log+' 2>&1 < /dev/null'
            _,o,e=c.exec_command('sh -c '+shlex.quote(command+' &'));o.channel.recv_exit_status()
        print(json.dumps({'action':a.action,'phase':a.phase,'commit':a.commit,'CPU_preflight':preflight,'launch_requested':a.action=='launch'}))
    else:
        sf=c.open_sftp()
        script="import pathlib,json,hashlib; p=pathlib.Path("+repr(REL)+"); print(json.dumps([{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob('*')) if f.is_file() and f.suffix in ('.json','.zip','.log')]))"
        files=json.loads(cmd('cd '+REMOTE+' && '+PYTHON+' -c '+shlex.quote(script)));copied=0
        for row in files:
            path=ROOT/row['path'];phase=Path(row['path']).relative_to(REL).parts[0]
            # Top protocol freezes must match; only new phase runtime/receipts
            # are mutable locally during sync. Never touch older namespaces.
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']:continue
            if phase not in ('T','A','B') and path.exists():raise ValueError('immutable protocol differs')
            if path.exists() and '/solve_results/' in row['path']:raise ValueError('completed raw SHA mismatch')
            with sf.open(REMOTE+'/'+row['path'],'rb') as f:data=f.read()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:
                if path.name in ('runtime_status.json','formal_run.log'):continue
                raise ValueError('non-atomic result during SFTP')
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.download');tmp.write_bytes(data);tmp.replace(path);copied+=1
        sf.close();print(json.dumps({'copied':copied,'local_persistent_storage':str(ROOT/REL),'GPU':gpu,'active_scientific_process':bool(active)}))
    c.close()
if __name__=='__main__':main()
