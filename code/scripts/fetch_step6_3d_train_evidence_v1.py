"""Read-only SFTP backup of the finished STEP 6.3D TRAIN evidence.

The password is prompted interactively and is never written to disk.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile

import paramiko


ROOT = Path(__file__).resolve().parents[2]
REL = Path("code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929")
CONFIG = Path("code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/05_selected_execution_config.json")
EXTRA = ("formal_train_tuning_3080ti_20260930.log",
         "05_train_hyperparameter_tuning_receipt.json",
         "06_frozen_selected_cem_configs.json")


def digest(stream) -> str:
    h = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        h.update(block)
    return h.hexdigest()


def remote_bytes(sftp: paramiko.SFTPClient, path: str) -> bytes:
    with sftp.open(path, "rb") as stream:
        return stream.read()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--remote-repo", required=True)
    args = p.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, port=args.port, username=args.user,
                   password=getpass.getpass("SSH password: "),
                   look_for_keys=False, allow_agent=False, timeout=15)
    try:
        sftp = client.open_sftp()
        remote_root = args.remote_repo.rstrip("/")
        remote_config = remote_bytes(sftp, f"{remote_root}/{CONFIG.as_posix()}")
        local_config = (ROOT / CONFIG).read_bytes()
        if hashlib.sha256(remote_config).digest() != hashlib.sha256(local_config).digest():
            raise RuntimeError("frozen execution config differs between local and remote")
        frozen = json.loads(local_config)
        sources = frozen["source_sha256"]
        for rel, expected in sources.items():
            local = ROOT / rel
            with local.open("rb") as stream:
                local_sha = digest(stream)
            with sftp.open(f"{remote_root}/{rel}", "rb") as stream:
                remote_sha = digest(stream)
            if local_sha != expected or remote_sha != expected:
                raise RuntimeError(f"source identity mismatch: {rel}")
        print(f"SOURCE_IDENTITY_PASS files={len(sources)}", flush=True)
        remote_dir = f"{remote_root}/{REL.as_posix()}"
        attrs = {x.filename: x.st_size for x in sftp.listdir_attr(remote_dir + "/solve_results/train")
                 if x.filename.endswith(".json")}
        names = sorted(attrs)
        if len(names) != 768 or len(names) != len(set(names)):
            raise RuntimeError(f"unexpected TRAIN result count: {len(names)}")
        wanted = [f"solve_results/train/{name}" for name in names] + list(EXTRA)
        command = (f"cd {shlex.quote(remote_dir)} && sha256sum solve_results/train/*.json " +
                   " ".join(shlex.quote(name) for name in EXTRA))
        _, stdout, stderr = client.exec_command(command)
        inventory_raw = stdout.read()
        inventory_error = stderr.read().decode()
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError("remote inventory failed: " + inventory_error)
        remote_shas = dict(line.split("  ", 1)[::-1]
                           for line in inventory_raw.decode().splitlines())
        if sorted(remote_shas) != sorted(wanted):
            raise RuntimeError("remote inventory file set changed")
        remote_inventory = [
            {"path": rel, "size_bytes": (attrs[Path(rel).name] if rel.startswith("solve_results/")
                                          else sftp.stat(f"{remote_dir}/{rel}").st_size),
             "sha256": remote_shas[rel]} for rel in wanted]
        tar_path = ROOT / REL / "train_remote_stream.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        partial = tar_path.with_suffix(".tar.part")
        tar_command = "tar -C " + shlex.quote(remote_dir) + " -cf - " + " ".join(
            shlex.quote(name) for name in ("solve_results/train", *EXTRA))
        _, stdout, stderr = client.exec_command(tar_command)
        with partial.open("wb") as stream:
            while data := stdout.channel.recv(1024 * 1024):
                stream.write(data)
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError("remote tar failed: " + stderr.read().decode())
        os.replace(partial, tar_path)
        print(f"TRANSFERRED_TAR bytes={tar_path.stat().st_size}", flush=True)
        expected = {row["path"]: row for row in remote_inventory}
        seen = set()
        with tarfile.open(tar_path, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                rel = member.name.removeprefix("./")
                if rel not in expected or rel in seen:
                    raise RuntimeError(f"unexpected tar member: {rel}")
                seen.add(rel)
                local = ROOT / REL / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                assert source is not None
                with source, local.open("wb") as dest:
                    while block := source.read(1024 * 1024):
                        dest.write(block)
                with local.open("rb") as stream:
                    local_sha = digest(stream)
                if local_sha != expected[rel]["sha256"] or local.stat().st_size != expected[rel]["size_bytes"]:
                    raise RuntimeError(f"download SHA/size mismatch: {rel}")
        if seen != set(wanted):
            raise RuntimeError(f"tar missing {len(set(wanted) - seen)} expected files")
        inventory = remote_inventory
        out = ROOT / REL / "train_local_backup_inventory.json"
        out.write_bytes((json.dumps({"source_identity_pass": True,
                                     "file_count": len(inventory), "files": inventory},
                                    ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        print(f"BACKUP_PASS inventory_sha256={hashlib.sha256(out.read_bytes()).hexdigest()}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
