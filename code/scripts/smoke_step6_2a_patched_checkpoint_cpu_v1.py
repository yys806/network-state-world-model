"""One-window CPU inference check using the frozen formal best checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
CHECKPOINT = ROOT / "code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt"
FROZEN_SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"


def main() -> None:
    torch.set_num_threads(1)
    before_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if before_sha != FROZEN_SHA:
        raise ValueError("frozen best.pt SHA-256 mismatch")
    interface = FormalTrainingInterface.from_manifest(MANIFEST)
    protocol = formal_training_config_v1(MANIFEST)
    trainer = FullFormalTrainer.from_interface(interface, replace(protocol.training, device="cpu"))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if trainer.data.identity != payload["data_identity"]:
        raise ValueError("checkpoint dataset identity mismatch")
    keys_and_shapes = {}
    for name, module in (("encoder", trainer.encoder), ("rssm", trainer.model),
                         ("target_encoder", trainer.target_encoder),
                         ("future_posterior", trainer.future_posterior)):
        frozen = payload["model_state"][name]
        current = module.state_dict()
        if set(current) != set(frozen) or any(current[k].shape != frozen[k].shape for k in current):
            raise ValueError(f"{name} parameter interface changed")
        module.load_state_dict(frozen, strict=True)
        keys_and_shapes[name] = len(current)
    parameter_before = trainer.parameter_digest()
    selected = interface.validation_indices[0]
    trainer.eval()
    with torch.inference_mode():
        result = trainer.validate_indices([selected])
    parameter_after = trainer.parameter_digest()
    after_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    output = {"checkpoint_sha256_before": before_sha, "checkpoint_sha256_after": after_sha,
              "state_dict_keys_and_shapes_strict_load": keys_and_shapes,
              "parameter_digest_before": parameter_before, "parameter_digest_after": parameter_after,
              "window": selected, "horizon_rows": result["horizon_rows"]}
    if before_sha != after_sha or parameter_before != parameter_after:
        raise ValueError("frozen parameter identity changed")
    print(json.dumps(output, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
