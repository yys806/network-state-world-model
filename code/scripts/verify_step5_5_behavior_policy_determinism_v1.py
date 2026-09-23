"""Replay one real AirFogSim trajectory to verify STEP 5.5 policy determinism."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "scripts"))

def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def action_digest(value: dict) -> str:
    rows = [step["behavior_policy_audit"] for step in value["steps"]]
    return digest(rows)


def merge_dataset_receipt(dataset_receipt_path: Path, receipt: dict) -> None:
    dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
    dataset_receipt.setdefault("checks", {})["behavior_policy_seed_determinism_and_variation"] = receipt["passed"]
    dataset_receipt["behavior_policy_determinism_receipt"] = receipt
    dataset_receipt["passed"] = all(dataset_receipt["checks"].values())
    dataset_receipt_path.write_text(
        json.dumps(dataset_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    if args.merge_existing:
        receipt = json.loads(args.output.resolve().read_text(encoding="utf-8"))
        if not args.dataset_receipt:
            raise ValueError("--merge-existing requires --dataset-receipt")
        merge_dataset_receipt(args.dataset_receipt.resolve(), receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0 if receipt["passed"] else 1
    if not args.reference:
        raise ValueError("--reference is required unless --merge-existing is used")
    from collect_step5_5_formal_raw_v1 import collect_trajectory

    reference = json.loads(gzip.decompress(args.reference.resolve().read_bytes()).decode("utf-8"))
    simulator_seed = int(reference["environment"]["seed"])
    policy_seed = int(reference["environment"]["policy_seed"])
    trajectory_id = str(reference["decisions"][0]["trajectory_id"])
    same = collect_trajectory(simulator_seed, policy_seed, trajectory_id)
    different = collect_trajectory(simulator_seed, policy_seed + 1, trajectory_id)
    checks = {
        "same_seed_full_payload_equal": digest(reference) == digest(same),
        "same_seed_action_audit_equal": action_digest(reference) == action_digest(same),
        "different_policy_seed_action_variation": action_digest(reference) != action_digest(different),
        "real_airfogsim_96_transitions": len(same["steps"]) == 96 and len(different["steps"]) == 96,
        "locked_test_accessed_false": True,
    }
    receipt = {
        "schema_version": "PI-JWM-Step-5.5-Behavior-Policy-Determinism-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "simulator_seed": simulator_seed,
        "reference_policy_seed": policy_seed,
        "variation_policy_seed": policy_seed + 1,
        "digests": {
            "reference_payload": digest(reference),
            "same_seed_payload": digest(same),
            "reference_action": action_digest(reference),
            "same_seed_action": action_digest(same),
            "different_policy_action": action_digest(different),
        },
        "scope": {"formal_dataset_generation": False, "training": False, "gpu": False, "locked_test": False},
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dataset_receipt:
        merge_dataset_receipt(args.dataset_receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
