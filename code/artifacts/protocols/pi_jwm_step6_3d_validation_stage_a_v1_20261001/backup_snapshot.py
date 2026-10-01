"""Read-only remote Stage-A snapshot; credentials are prompted, never saved."""
import getpass
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import paramiko

root=Path(__file__).resolve().parents[4]
rel=Path('code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929')
control=Path('code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001')
remote='/root/autodl-tmp/pi-jwm-step6-3d/'
client=paramiko.SSHClient(); client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(sys.argv[1],port=int(sys.argv[2]),username=sys.argv[3],password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=15)
try:
    sftp=client.open_sftp()
    def read(path):
        with sftp.open(remote+path,'rb') as stream: return stream.read()
    def save(path,data,immutable=False):
        dest=root/path; dest.parent.mkdir(parents=True,exist_ok=True)
        if immutable and dest.exists():
            assert dest.read_bytes()==data, f'Existing local evidence differs: {path}'
            return
        tmp=dest.with_name(dest.name+'.download.tmp'); tmp.write_bytes(data); tmp.replace(dest)
    config=json.loads((root/'code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json').read_text())
    for path,expected in config['source_sha256'].items():
        assert hashlib.sha256(read(path)).hexdigest()==expected, f'Remote SOURCE_DRIFT: {path}'
    status=json.loads(read((control/'02_runtime_status.json').as_posix()))
    assert status['execution_config_id']==config['execution_config_id'] and status['locked_test'] is False
    entries=sftp.listdir((remote+rel.as_posix()+'/solve_results/validation'))
    inventory=[]; actual_transitions=0
    for name in sorted(entries):
        if not name.endswith('.json'): continue
        path=(rel/'solve_results/validation'/name).as_posix()
        data=read(path); row=json.loads(data)
        assert row['execution_config_id']==config['execution_config_id'] and row['locked_test'] is False
        assert row['source_sha256']==config['source_sha256'] and row['budget']==1024
        assert row['method'] in ('HRS','S-CEM','MH-CEM') and row['seed'] in (6311,6312,6313,6314,6315)
        for key in ('execution_device','gpu_model','precision','batch_size','checkpoint_sha256','state_storage','bucket_strategy'):
            assert row[key]==config[key], f'Wrong execution identity: {key}'
        save(Path(path),data,immutable=True)
        actual_transitions+=row['outcome']['budget_receipt']['N_unique_transition_evals']
        inventory.append({'path':path,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    save(control/('snapshots/'+stamp+'_runtime.json'),(json.dumps(status,indent=2,sort_keys=True)+'\n').encode())
    for name in ('01_launch_preflight_receipt.json','02_runtime_status.json','formal_validation_stage_a_3080ti.log','monitor_attachment_receipt.json','attached_monitor.log','supervise.py','attached_monitor.py'):
        save(control/name,read((control/name).as_posix()))
    for name in ('07_validation_stage_a_primary_comparison_receipt.json','08_selected_method.json'):
        try: data=read((rel/name).as_posix())
        except FileNotFoundError: continue
        save(rel/name,data,immutable=True)
    backup={'snapshot_utc':stamp,'remote_repo':remote,'local_persistent_storage':str(root/rel),'file_count':len(inventory),'files':inventory,'status':status,'locked_test':False}
    save(control/('snapshots/'+stamp+'_backup_inventory.json'),(json.dumps(backup,indent=2,sort_keys=True)+'\n').encode())
    _,stdout,stderr=client.exec_command('nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader')
    gpu=stdout.read().decode().strip()
    print(json.dumps({'state':status['state'],'backed_up_completed_cases':len(inventory),'monitor_snapshot_completed_cases':status['completed_cases'],'elapsed_seconds':status['elapsed_seconds'],'cases_per_hour':status['cases_per_hour'],'eta_seconds':status['eta_seconds'],'backed_up_actual_unique_transitions':actual_transitions,'gpu':gpu,'blocker':status['blocker'],'snapshot_utc':stamp,'local_SHA_backup':'PASS','locked_test':False,'stage_b':'NOT_STARTED'}))
finally:
    client.close()
