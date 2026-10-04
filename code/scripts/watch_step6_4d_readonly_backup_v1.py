"""Read-only SSH/SFTP monitor; stop only this runner on evidence of failure."""
import getpass
import hashlib
import json
import shlex
import sys
import time
import paramiko
from manage_step6_4d_remote_v1 import ROOT, REMOTE, PYTHON, REL
from run_step6_4d_full_cohort_b512_v1 import new_identity, validate_row


def main():
    commit = sys.argv[2]; config = json.loads((ROOT/REL/'05_execution_config_3080ti.json').read_text())
    parent = json.loads((ROOT/REL/'03_b1024_parent_receipt.json').read_text())
    parameter_digest = json.loads((ROOT/parent['files'][0]['path']).read_text())['parameter_digest']
    c = paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.nmb2.seetacloud.com', port=18448, username=sys.argv[1], password=getpass.getpass('SSH password: '),
              look_for_keys=False, allow_agent=False, timeout=15)
    c.get_transport().set_keepalive(30); sftp = c.open_sftp(); last = -1
    def cmd(script):
        _, o, e = c.exec_command(script); text = o.read().decode(); error = e.read().decode(); code = o.channel.recv_exit_status()
        if code: raise RuntimeError('remote read exit '+str(code)+': '+error[-600:])
        return text
    checks = {**config['source_sha256'], **config['input_sha256'], config['checkpoint_path']: config['checkpoint_sha256']}
    script = 'import pathlib,json,hashlib; d=json.loads('+repr(json.dumps(checks))+'); p=pathlib.Path('+repr(REL)+'); print(json.dumps({"wrong":[n for n,h in d.items() if not pathlib.Path(n).exists() or hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()!=h],"files":[{"path":str(f),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob("*")) if f.is_file() and (f.suffix==".json" or f.name=="formal_expansion.log")]}))'
    try:
        while True:
            evidence = json.loads(cmd(f'cd {REMOTE} && {PYTHON} -c {shlex.quote(script)}'))
            if evidence['wrong']: raise ValueError('SOURCE_INPUT_CHECKPOINT_DRIFT: '+str(evidence['wrong']))
            for item in evidence['files']:
                dest = ROOT/item['path']; name = dest.name
                if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == item['sha256']: continue
                mutable = '/solve_results/' in item['path'] or '/runtime/' in item['path'] or '09' <= name[:2] <= '16'
                if not mutable: raise ValueError('immutable protocol mismatch: '+item['path'])
                with sftp.open(REMOTE+'/'+item['path'], 'rb') as f: data = f.read()
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    if name in ('09_runtime_status.json','formal_expansion.log'): continue
                    raise ValueError('raw changed during SHA backup')
                if '/solve_results/' in item['path']:
                    row = json.loads(data)
                    validate_row(row, new_identity(config, row['sample_id'], row['seed'], commit), parameter_digest)
                dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix(dest.suffix+'.download'); tmp.write_bytes(data); tmp.replace(dest)
            status_path = ROOT/REL/'09_runtime_status.json'; log_path = ROOT/REL/'runtime/formal_expansion.log'
            status = json.loads(status_path.read_text()) if status_path.exists() else None
            tail = log_path.read_text()[-1400:] if log_path.exists() else ''
            running = cmd("pgrep -af '[r]un_step6_4d_full_cohort_b512_v1.py' || true").strip()
            count = status['new_results_completed'] if status else 0
            if count != last:
                diagnostic = cmd('nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader').strip()
                if not diagnostic.startswith(config['gpu_model']): raise ValueError('GPU identity drift')
                rate = status['cases_per_hour_this_invocation'] if status else None
                print(json.dumps({'status': status, 'gpu': diagnostic, 'ETA_hours': (272-count)/rate if rate else None,
                                  'local_new_raw': len(list((ROOT/REL/'solve_results/expansion').glob('*.json'))),
                                  'local_persistent_storage': str(ROOT/REL)}, ensure_ascii=False), flush=True)
                last = count
            if 'Traceback (most recent call last)' in tail: raise ValueError('RUNNER_CRASH: '+tail)
            if not running:
                if count == 272 and 'STEP_6_4D_FULL_COHORT_B512=PASS; STOP' in tail:
                    print('REMOTE_COMPLETED_LOCAL_BACKUP_READY', flush=True); break
                raise ValueError('runner stopped before accepted full coverage: '+tail)
            time.sleep(30)
    except Exception:
        # No restart or scientific modification; terminate only the authorized runner.
        c.exec_command("pkill -TERM -f '^/root/miniconda3/bin/python -u code/scripts/run_step6_4d_full_cohort_b512_v1.py' || true")
        raise
    finally:
        sftp.close(); c.close()


if __name__ == '__main__': main()
