"""Deploy exact protocol commit; read-only monitoring and SHA backups thereafter."""
import argparse
import getpass
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/pi-jwm-step6-3d'
PYTHON = '/root/miniconda3/bin/python'
REL = 'code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003'
RUNNER = 'code/scripts/run_step6_4d_full_cohort_b512_v1.py'


def main():
    p = argparse.ArgumentParser(); p.add_argument('user'); p.add_argument('action', choices=['probe','deploy','launch','snapshot'])
    p.add_argument('--commit'); p.add_argument('--bundle', type=Path); args = p.parse_args()
    c = paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect('connect.nmb2.seetacloud.com', port=18448, username=args.user,
            password=getpass.getpass('SSH password: '), look_for_keys=False, allow_agent=False, timeout=15)
    except (OSError, paramiko.SSHException):
        print(json.dumps({'reachable': False, 'gpu_started': False})); return
    def command(cmd):
        _, o, e = c.exec_command(cmd); data = o.read().decode(); error = e.read().decode(); code = o.channel.recv_exit_status()
        if code: raise RuntimeError('remote exit '+str(code)+': '+error[-1200:])
        return data
    sftp = c.open_sftp()
    if args.action == 'probe':
        result = {'reachable': True, 'read_only': True,
            'gpu': command('nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader').strip(),
            'remote_commit': command(f'git -C {REMOTE} rev-parse HEAD').strip(),
            'tracked_status': command(f'git -C {REMOTE} status --porcelain --untracked-files=no').strip(),
            'running': command("pgrep -af '[r]un_step6_4[cd]_.*\.py' || true").strip()}
        path = ROOT/REL/'06_remote_readonly_probe.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(result))
    elif args.action in ('deploy','launch'):
        if not args.commit or len(args.commit) != 40: raise ValueError('exact protocol commit required')
        if command(f'git -C {REMOTE} status --porcelain --untracked-files=no').strip(): raise ValueError('remote tracked dirty')
        if command("pgrep -af '[r]un_step6_4[cd]_.*\.py' || true").strip(): raise ValueError('another scientific runner is active')
        if args.action == 'deploy':
            if args.bundle:
                transport = '/tmp/pi-jwm-step6-4d-deployment.bundle'; sftp.put(str(args.bundle), transport)
                command(f'git -C {REMOTE} bundle verify {transport}')
                command(f'git -C {REMOTE} fetch {transport} main:refs/remotes/origin/main')
            else: command(f'git -C {REMOTE} fetch origin')
            command(f'git -C {REMOTE} merge --ff-only origin/main')
        if command(f'git -C {REMOTE} rev-parse HEAD').strip() != args.commit: raise ValueError('remote exact commit mismatch')
        # CPU-only preflight verifies source, config, historical raw SHA and commit blobs.
        output = command(f'cd {REMOTE} && {PYTHON} {RUNNER} --protocol-commit {shlex.quote(args.commit)}')
        if command(f'git -C {REMOTE} status --porcelain --untracked-files=no').strip(): raise ValueError('remote tracked dirty after deploy')
        if args.action == 'launch':
            if command(f'find {REMOTE}/{REL}/solve_results/expansion -name "*.json" 2>/dev/null | wc -l').strip() != '0':
                raise ValueError('initial expansion result count not zero')
            # Both source/clean gates and GPU/checkpoint qualification repeat inside runner.
            command(f'cd {REMOTE} && mkdir -p {REL}/runtime && nohup {PYTHON} -u {RUNNER} --protocol-commit {shlex.quote(args.commit)} --execute > {REL}/runtime/formal_expansion.log 2>&1 < /dev/null &')
        print(json.dumps({'commit': args.commit, 'cpu_preflight': output.strip(), 'launch_requested': args.action=='launch'}))
    else:
        script = "import pathlib,json,hashlib; p=pathlib.Path("+repr(REL)+"); print(json.dumps([{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob('*')) if f.is_file() and (f.suffix=='.json' or f.name=='formal_expansion.log')]))"
        files = json.loads(command(f'cd {REMOTE} && {PYTHON} -c {shlex.quote(script)}')); copied = 0
        for item in files:
            dest = ROOT/item['path']; name = dest.name
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == item['sha256']: continue
            if '/solve_results/' not in item['path'] and '/runtime/' not in item['path'] and not ('09' <= name[:2] <= '16'):
                raise ValueError('immutable remote protocol differs: '+item['path'])
            with sftp.open(REMOTE+'/'+item['path'], 'rb') as f: data = f.read()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                if name in ('09_runtime_status.json','formal_expansion.log'): continue
                raise ValueError('non-atomic raw changed during transfer')
            dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix(dest.suffix+'.download'); tmp.write_bytes(data); tmp.replace(dest); copied += 1
        status = ROOT/REL/'09_runtime_status.json'; log = ROOT/REL/'runtime/formal_expansion.log'
        print(json.dumps({'copied': copied, 'local_persistent_storage': str(ROOT/REL),
            'status': json.loads(status.read_text()) if status.exists() else None,
            'gpu': command('nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader').strip(),
            'running': command("pgrep -af '[r]un_step6_4d_.*\.py' || true").strip(),
            'log_tail': log.read_text()[-1600:] if log.exists() else None}))
    sftp.close(); c.close()


if __name__ == '__main__': main()
