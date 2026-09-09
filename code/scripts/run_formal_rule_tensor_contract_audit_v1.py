from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_rule_tensor_contract_audit_v1 import audit_rule_tensor_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_rule_tensor_contract(args.tensor_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rule_tensor_contract_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["rule_layer_tensor_contract_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
