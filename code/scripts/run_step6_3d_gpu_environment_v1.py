"""Record the CUDA host and frozen checkpoint identity without training."""
from __future__ import annotations

import argparse

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
import torch
CHECKPOINT = ROOT / "code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt"
EXPECTED_SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("STEP_6_3D_GPU_EXECUTION=BLOCKED_GPU_UNAVAILABLE")
    index = 0
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    checkpoint_hash = sha(CHECKPOINT)
    if checkpoint_hash != EXPECTED_SHA:
        raise ValueError("frozen checkpoint SHA mismatch")
    source_files = ("code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py",
        "code/src/pi_jwm/step6_1_trained_candidate_rollout_v1.py",
        "code/src/pi_jwm/step6_3c_search_protocol_v1.py",
        "code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py",
        "code/scripts/run_step6_3d_one_cpu_solve_v1.py")
    result = {"cuda_available": True, "gpu_index": index, "name": props.name,
        "vram_bytes": props.total_memory, "cuda_mem_get_info_free_bytes": free,
        "cuda_mem_get_info_total_bytes": total,
        "allocated_vram_bytes": torch.cuda.memory_allocated(index),
        "reserved_vram_bytes": torch.cuda.memory_reserved(index),
        "nvidia_smi_row": nvidia,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "precision_policy_changed": False, "checkpoint_sha256": checkpoint_hash,
        "source_sha256": {name: sha(ROOT/name) for name in source_files},
        "training": False, "formal_method_comparison": False,
        "locked_test": False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "02_gpu_environment.json").write_text(json.dumps(result,
        indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
        ("name", "vram_bytes", "torch_version", "checkpoint_sha256")}))


if __name__ == "__main__":
    main()
