"""Launch and aggregate the resumable 18-run PI-JWM R6 GPU matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_gpu_training_protocol import (  # noqa: E402
    build_default_gpu_training_protocol,
    validate_formal_run_records,
)


DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_gpu_training_v2"
TRAINING_SCRIPT = Path(__file__).resolve().with_name("run_r6_joint_policy_gpu_training.py")
WORKER_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class FormalCommand:
    run_id: str
    sumo_port: int
    argv: tuple[str, ...]


def build_worker_environment(
    *,
    cpu_threads: int,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved = int(cpu_threads)
    if resolved <= 0:
        raise ValueError("worker CPU threads must be positive")
    environment = dict(os.environ if base_environment is None else base_environment)
    for name in WORKER_THREAD_VARIABLES:
        environment[name] = str(resolved)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def build_formal_commands(
    *,
    python_executable: str,
    output_dir: Path,
    device: str,
    target_environment_steps: int,
    sumo_port_base: int,
) -> tuple[FormalCommand, ...]:
    protocol = build_default_gpu_training_protocol()
    return tuple(
        FormalCommand(
            run_id=run.run_id,
            sumo_port=int(sumo_port_base) + index,
            argv=(
                str(python_executable),
                str(TRAINING_SCRIPT),
                "--run-id",
                run.run_id,
                "--device",
                str(device),
                "--output-dir",
                str(Path(output_dir).resolve()),
                "--max-environment-steps",
                str(int(target_environment_steps)),
                "--sumo-port",
                str(int(sumo_port_base) + index),
            ),
        )
        for index, run in enumerate(protocol.formal_runs())
    )


def is_complete_summary(summary: dict[str, Any], *, target_steps: int) -> bool:
    return bool(
        summary.get("formal") is True
        and summary.get("status") == "complete"
        and int(summary.get("environment_steps", -1)) >= int(target_steps)
        and summary.get("state_source") == "online_airfogsim_strict_dual_graph"
        and summary.get("locked_test_accessed") is False
        and summary.get("checkpoint_reload_verified") is True
    )


def _read_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--target-environment-steps", type=int, default=100000)
    parser.add_argument("--sumo-port-base", type=int, default=18813)
    parser.add_argument("--worker-cpu-threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if int(args.max_parallel) <= 0:
        raise ValueError("max_parallel must be positive")
    if int(args.target_environment_steps) <= 0:
        raise ValueError("target_environment_steps must be positive")
    if int(args.worker_cpu_threads) <= 0:
        raise ValueError("worker_cpu_threads must be positive")
    protocol = build_default_gpu_training_protocol()
    if not 1024 <= int(args.sumo_port_base):
        raise ValueError("sumo_port_base must be at least 1024")
    if int(args.sumo_port_base) + len(protocol.formal_runs()) - 1 > 65535:
        raise ValueError("SUMO port range exceeds 65535")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = output_dir / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    commands = list(
        build_formal_commands(
            python_executable=sys.executable,
            output_dir=output_dir,
            device=args.device,
            target_environment_steps=args.target_environment_steps,
            sumo_port_base=args.sumo_port_base,
        )
    )
    _write_json_atomic(
        output_dir / "launch_manifest.json",
        {
            "schema_version": protocol.schema_version,
            "formal_run_count": len(commands),
            "max_parallel": int(args.max_parallel),
            "device": str(args.device),
            "state_source": protocol.state_source,
            "atomic_resume_checkpoint": protocol.atomic_resume_checkpoint,
            "target_environment_steps": int(args.target_environment_steps),
            "sumo_port_base": int(args.sumo_port_base),
            "worker_cpu_threads": int(args.worker_cpu_threads),
            "formal_budget_complete": (
                int(args.target_environment_steps) >= protocol.max_environment_steps
            ),
            "locked_test_accessed": False,
            "commands": [
                {
                    "run_id": row.run_id,
                    "sumo_port": row.sumo_port,
                    "argv": list(row.argv),
                }
                for row in commands
            ],
        },
    )
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "formal_run_count": len(commands)}))
        return 0

    records: dict[str, dict[str, Any]] = {}
    pending: list[FormalCommand] = []
    for command in commands:
        summary = _read_summary(output_dir / command.run_id / "summary.json")
        if summary is not None and is_complete_summary(
            summary,
            target_steps=int(args.target_environment_steps),
        ):
            records[command.run_id] = summary
        else:
            pending.append(command)

    running: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    environment = build_worker_environment(cpu_threads=int(args.worker_cpu_threads))
    while pending or running:
        while pending and len(running) < int(args.max_parallel):
            command = pending.pop(0)
            log_handle = (log_root / f"{command.run_id}.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                command.argv,
                cwd=Path(__file__).resolve().parent,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[command.run_id] = (process, log_handle)
        finished: list[str] = []
        for run_id, (process, log_handle) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            summary = _read_summary(output_dir / run_id / "summary.json")
            if return_code == 0 and summary is not None:
                records[run_id] = summary
            else:
                records[run_id] = {
                    "run_id": run_id,
                    "formal": True,
                    "status": "failed",
                    "return_code": int(return_code),
                    "locked_test_accessed": False,
                    "log": str((log_root / f"{run_id}.log").resolve()),
                }
            finished.append(run_id)
        for run_id in finished:
            del running[run_id]
        if pending or running:
            time.sleep(max(float(args.poll_seconds), 0.1))

    ordered = [records[run.run_id] for run in protocol.formal_runs()]
    validation = validate_formal_run_records(protocol, ordered)
    _write_json_atomic(output_dir / "run_records.json", ordered)
    _write_json_atomic(
        output_dir / "matrix_summary.json",
        {
            "schema_version": protocol.schema_version,
            "expected_count": validation.expected_count,
            "complete_count": validation.complete_count,
            "failed_count": validation.failed_count,
            "target_environment_steps": int(args.target_environment_steps),
            "formal_budget_complete": (
                int(args.target_environment_steps) >= protocol.max_environment_steps
            ),
            "locked_test_accessed": False,
        },
    )
    return 0 if validation.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
