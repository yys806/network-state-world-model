"""Frozen-checkpoint legacy/patched CPU paired rollout; no parameter update."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from itertools import groupby
from dataclasses import replace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1
from smoke_step6_2a_patched_checkpoint_cpu_v1 import ROOT, MANIFEST, CHECKPOINT, FROZEN_SHA


def _different(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return not torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) != set(b) or any(_different(a[k], b[k]) for k in a)
    return a != b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 audits all 5520 windows")
    parser.add_argument("--potential-only", action="store_true", help="audit every anchor Flow/Route overlap; prove other windows by branch reachability")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sha_before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if sha_before != FROZEN_SHA:
        raise ValueError("frozen best.pt identity mismatch")
    interface = FormalTrainingInterface.from_manifest(MANIFEST)
    trainer = FullFormalTrainer.from_interface(interface, replace(formal_training_config_v1(MANIFEST).training, device="cpu"))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if trainer.data.identity != payload["data_identity"]:
        raise ValueError("checkpoint data identity mismatch")
    for name, module in (("encoder", trainer.encoder), ("rssm", trainer.model),
                         ("target_encoder", trainer.target_encoder),
                         ("future_posterior", trainer.future_posterior)):
        module.load_state_dict(payload["model_state"][name], strict=True)
    trainer.eval()
    digest_before = trainer.parameter_digest()
    report = {split: Counter() for split in ("dev_train", "dev_validation")}
    potential = json.loads((ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928/static_audit_full.json").read_text(encoding="utf-8"))["sample"] if args.potential_only else None
    errors = []
    started = time.perf_counter()
    with torch.inference_mode():
        for split, indices in (("dev_train", interface.train_indices), ("dev_validation", interface.validation_indices)):
            if potential is not None:
                ids = set(potential[split]["potential_route_overlap_sample_ids"])
                indices = [i for i in indices if interface.samples[i]["metadata"]["sample_id"] in ids]
            if args.limit:
                indices = indices[:args.limit]
            groups = [[i for _, i in items] for _, items in groupby(
                ((interface.samples[i]["metadata"]["trajectory_id"], i) for i in indices),
                key=lambda item: item[0])]
            for group in groups:
              for offset in range(0, len(group), trainer.config.batch_size):
                selected = group[offset:offset + trainer.config.batch_size]
                batch = trainer.shards.load_batch(selected)
                original = trainer.data
                try:
                    trainer.data = batch
                    trainer._move_data_to_device()
                    local = list(range(len(selected)))
                    legacy = trainer._recursive_rollout(local, 4, stage="prior_dominant_recursive", route_rule_mode="LEGACY_ROUTE_RULE")
                    patched = trainer._recursive_rollout(local, 4, stage="prior_dominant_recursive", route_rule_mode="PATCHED_ROUTE_RULE")
                    c = report[split]
                    c["windows"] += len(selected)
                    for h in range(4):
                        if _different(legacy["actions"][h], patched["actions"][h]):
                            c["action_tensor_mismatch_batches"] += 1
                        for row in range(len(selected)):
                            state_delta = any(not torch.equal(legacy["states"][h][k][row], patched["states"][h][k][row]) for k in legacy["states"][h])
                            graph_delta = any(not torch.equal(legacy["graphs"][h][k][row], patched["graphs"][h][k][row]) for k in legacy["graphs"][h] if isinstance(legacy["graphs"][h][k], torch.Tensor) and legacy["graphs"][h][k].shape[0] == len(selected))
                            learned_delta = any(not torch.equal(legacy["traces"][h]["learned"][k][row], patched["traces"][h]["learned"][k][row]) for k in ("vehicle_motion", "csi"))
                            prior_delta = any(not torch.equal(legacy["latents"][h]["prior"][family][key][row], patched["latents"][h]["prior"][family][key][row]) for family in ("physical", "communication") for key in ("mean", "log_std"))
                            c["state_different_window_horizons"] += state_delta
                            c["graph_different_window_horizons"] += graph_delta
                            c["decoder_different_window_horizons"] += learned_delta
                            c["prior_different_window_horizons"] += prior_delta
                            for family in ("physical", "agent", "communication", "flow", "task"):
                                old = legacy["latents"][h]["h"][family][row]
                                new = patched["latents"][h]["h"][family][row]
                                c[f"max_abs_delta_h_{family}"] = max(c[f"max_abs_delta_h_{family}"], float((old - new).abs().max()))
                            for family in ("physical", "communication"):
                                for key in ("mean", "log_std"):
                                    old = legacy["latents"][h]["prior"][family][key][row]
                                    new = patched["latents"][h]["prior"][family][key][row]
                                    c[f"max_abs_delta_prior_{family}_{key}"] = max(c[f"max_abs_delta_prior_{family}_{key}"], float((old - new).abs().max()))
                            for key, label in (("vehicle_motion", "motion"), ("csi", "csi")):
                                old = legacy["traces"][h]["learned"][key][row]
                                new = patched["traces"][h]["learned"][key][row]
                                c[f"max_abs_delta_{label}"] = max(c[f"max_abs_delta_{label}"], float((old - new).abs().max()))
                            if state_delta or graph_delta or learned_delta or prior_delta:
                                c["affected_window_horizons"] += 1
                        c[f"H{h+1}_window_count"] += len(selected)
                except Exception as exc:
                    errors.append({"split": split, "first_index": selected[0], "error": str(exc)})
                    raise
                finally:
                    trainer.data = original
                print(json.dumps({"split": split, "windows": report[split]["windows"], "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    sha_after = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    digest_after = trainer.parameter_digest()
    if sha_after != sha_before or digest_after != digest_before:
        raise ValueError("frozen checkpoint or parameter identity changed")
    result = {"splits": {k: dict(v) for k, v in report.items()}, "errors": errors,
              "checkpoint_sha256": sha_after, "parameter_digest": digest_after,
              "elapsed_seconds": time.perf_counter() - started, "cpu_only": True,
              "prior_mode": "mean", "service_mode": "expectation", "horizons": [1, 2, 3, 4]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
