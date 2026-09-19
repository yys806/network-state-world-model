"""Build the CPU fixed-shape tensor/collation evidence for Step 3.3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pi_jwm.step3_3_model_input_tensor_v1 import TENSOR_INPUT_GAP_TABLE, build_tensor_batch, save_tensor_batch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "code/artifacts/protocols/pi_jwm_step3_2_raw_to_dataset_batch_v1_20260919"
DEFAULT_OUTPUT = ROOT / "code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    batch = json.loads((args.input_dir / "batch.json").read_text(encoding="utf-8"))
    normalized = json.loads((args.input_dir / "normalized_samples.json").read_text(encoding="utf-8"))
    stats = json.loads((args.input_dir / "train_normalization_stats.json").read_text(encoding="utf-8"))
    tensor = build_tensor_batch(normalized, stats=stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / "tensor.npz"
    save_tensor_batch(tensor, tensor_path)
    (args.output_dir / "tensor_schema.json").write_text(json.dumps(tensor["contract"].to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": tensor["schema_version"],
        "sample_contract_version": tensor["contract"].sample_contract_version,
        "input_batch_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "files": {"tensor.npz": sha256(tensor_path), "tensor_schema.json": sha256(args.output_dir / "tensor_schema.json")},
        "sample_count": len(tensor["sample_ids"]),
        "scope": {"locked_test": False, "training": False, "gpu": False, "formal_dataset": False},
        "capacity_semantics": "development observed maxima only; not final formal research capacity",
        "tensor_input_gap_table": list(TENSOR_INPUT_GAP_TABLE),
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in tensor.items() if hasattr(value, "shape")},
        "mask_presence_counts": {name: int(value.sum()) for name, value in tensor.items() if hasattr(value, "dtype") and value.dtype == bool},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "sample_count": len(tensor["sample_ids"]), "contract": tensor["contract"].to_dict()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
