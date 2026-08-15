from __future__ import annotations

"""Build the auditable PI-JWM main-experiment readiness bundle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.main_experiment_contract import (
    CONTRACT_VERSION,
    build_frozen_contract,
    build_readiness_report,
)


DEFAULT_EXP03 = (
    CODE_ROOT
    / "artifacts"
    / "small_experiments"
    / "exp03_airfogsim_cross_graph_evidence"
    / "evidence_v1"
    / "validation_report.json"
)
DEFAULT_EXP04 = (
    CODE_ROOT
    / "artifacts"
    / "small_experiments"
    / "exp04_task_resource_conservation"
    / "conservation_v1"
    / "validation_report.json"
)
DEFAULT_EXP05 = (
    CODE_ROOT
    / "artifacts"
    / "small_experiments"
    / "exp05_paired_action_sensitivity"
    / "paired_v1"
    / "validation_report.json"
)
DEFAULT_OUTPUT = CODE_ROOT / "artifacts" / "main_experiment_readiness" / "readiness_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_text(readiness: dict[str, Any]) -> str:
    failed = readiness["blocking_checks"]
    failed_text = "、".join(failed) if failed else "无"
    return "\n".join(
        [
            "# PI-JWM 主实验收口与就绪报告",
            "",
            f"- 契约版本：`{readiness['contract_version']}`",
            f"- `contract_ready`: `{str(readiness['contract_ready']).lower()}`",
            f"- `simulator_preflight_ready`: `{str(readiness['simulator_preflight_ready']).lower()}`",
            f"- `simulation_training_ready`: `{str(readiness['simulation_training_ready']).lower()}`",
            f"- `formal_dataset_ready`: `{str(readiness['formal_dataset_ready']).lower()}`",
            f"- `external_validation_ready`: `{str(readiness['external_validation_ready']).lower()}`",
            f"- 阻塞检查：{failed_text}",
            "",
            "## 状态解释",
            "",
            "`simulation_training_ready=true`仅表示可以启动正式仿真数据生成和训练smoke；不表示正式规模数据集已经生成，也不表示真实数据外部验证已经完成。",
            "",
            "历史实验01仍然证明旧60-seed数据不满足新契约；本报告只使用修正后实验03—05的验证结果，不篡改历史数据。",
            "",
        ]
    )


def build_main_experiment_readiness(
    output_dir: Path,
    exp03_validation: Path,
    exp04_validation: Path,
    exp05_validation: Path,
    *,
    formal_dataset_manifest: Path | None = None,
    external_validation_manifest: Path | None = None,
) -> dict[str, Any]:
    input_paths = [Path(exp03_validation), Path(exp04_validation), Path(exp05_validation)]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    formal = _read_json(formal_dataset_manifest) if formal_dataset_manifest else None
    external = _read_json(external_validation_manifest) if external_validation_manifest else None
    contract = build_frozen_contract()
    readiness = build_readiness_report(
        _read_json(input_paths[0]),
        _read_json(input_paths[1]),
        _read_json(input_paths[2]),
        formal_dataset_manifest=formal,
        external_validation_manifest=external,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    readiness_path = output_dir / "readiness_report.json"
    report_path = output_dir / "REPORT.md"
    _write_json(contract_path, contract)
    _write_json(readiness_path, readiness)
    report_path.write_text(_report_text(readiness), encoding="utf-8")

    output_paths = [contract_path, readiness_path, report_path]
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "input_files": {str(path.resolve()): _sha256(path) for path in input_paths},
        "output_files": {path.name: _sha256(path) for path in output_paths},
        "source_files": {
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
            str((SRC_ROOT / "pi_jwm" / "main_experiment_contract.py").resolve()): _sha256(
                SRC_ROOT / "pi_jwm" / "main_experiment_contract.py"
            ),
            str((SRC_ROOT / "pi_jwm" / "airfogsim_contract_adapter.py").resolve()): _sha256(
                SRC_ROOT / "pi_jwm" / "airfogsim_contract_adapter.py"
            ),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp03-validation", type=Path, default=DEFAULT_EXP03)
    parser.add_argument("--exp04-validation", type=Path, default=DEFAULT_EXP04)
    parser.add_argument("--exp05-validation", type=Path, default=DEFAULT_EXP05)
    parser.add_argument("--formal-dataset-manifest", type=Path)
    parser.add_argument("--external-validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_main_experiment_readiness(
        args.output_dir,
        args.exp03_validation,
        args.exp04_validation,
        args.exp05_validation,
        formal_dataset_manifest=args.formal_dataset_manifest,
        external_validation_manifest=args.external_validation_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
