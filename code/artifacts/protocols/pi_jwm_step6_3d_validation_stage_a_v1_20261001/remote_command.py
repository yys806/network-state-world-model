import getpass
import sys
import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(sys.argv[1], port=int(sys.argv[2]), username=sys.argv[3], password=getpass.getpass('SSH password: '), look_for_keys=False, allow_agent=False, timeout=15)
try:
    command = open(sys.argv[4], encoding='utf-8').read()
    _, stdout, stderr = client.exec_command(command)
    print(stdout.read().decode(), end='')
    print(stderr.read().decode(), end='', file=sys.stderr)
    sys.exit(stdout.channel.recv_exit_status())
finally:
    client.close()
