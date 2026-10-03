"""Read-only SSH hardware/source inspection; credentials only in getpass memory."""
import getpass
import json
import sys
from pathlib import Path
import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.nmb2.seetacloud.com', port=18448, username=sys.argv[1],
               password=getpass.getpass('SSH password: '), look_for_keys=False,
               allow_agent=False, timeout=15)
commands = {
    'gpu': 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader',
    'git': 'git -C /root/autodl-tmp/pi-jwm-step6-3d rev-parse HEAD',
    'tracked': 'git -C /root/autodl-tmp/pi-jwm-step6-3d status --porcelain --untracked-files=no',
    'checkpoint': 'sha256sum /root/autodl-tmp/pi-jwm-step6-3d/code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt',
}
receipt = {'read_only': True, 'gpu_search_started': False}
for key, command in commands.items():
    _, stdout, stderr = client.exec_command(command)
    receipt[key] = {'exit_code': stdout.channel.recv_exit_status(),
                    'stdout': stdout.read().decode().strip(),
                    'stderr': stderr.read().decode().strip()}
client.close()
path = Path(__file__).resolve().parents[2]/'code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003/06_remote_readonly_preflight.json'
path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(receipt))
