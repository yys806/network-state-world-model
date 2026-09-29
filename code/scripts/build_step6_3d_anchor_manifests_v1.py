"""Freeze deterministic TRAIN/Validation selections from accepted 6.3C receipts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from pi_jwm.step6_3d_anchor_selection_v1 import select_stratified_anchors

SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split, filename, count in (
        ("dev_train", "01_formal_train_domain_feasibility.json", 32),
        ("dev_validation", "02_formal_validation_domain_descriptive.json", 64),
    ):
        source = SOURCE / filename
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload["split"] != split or len(payload["anchors"]) != payload["anchor_count"]:
            raise ValueError("source split or anchor count mismatch")
        selected = select_stratified_anchors(payload["anchors"], count)
        if len(selected["empty_sample_ids"]) != payload["empty_domain_count"]:
            raise ValueError("empty-domain count mismatch")
        selected.update({"split": split, "source_receipt": str(source.relative_to(ROOT)).replace("\\", "/"),
                         "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "future_target_used": False, "locked_test": False,
                         "validation_used_for_tuning": False})
        manifests[split] = selected
        name = "01_train_anchor_manifest.json" if split == "dev_train" else "02_validation_anchor_manifest.json"
        (OUT / name).write_text(json.dumps(selected, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8", newline="\n")
    train_ids = {row["sample_id"] for row in manifests["dev_train"]["selected"]}
    validation_ids = {row["sample_id"] for row in manifests["dev_validation"]["selected"]}
    if train_ids & validation_ids or len(manifests["dev_validation"]["empty_sample_ids"]) != 6:
        raise ValueError("split overlap or Validation empty-domain boundary changed")
    print(json.dumps({"train_selected": len(train_ids), "validation_selected": len(validation_ids),
                      "validation_empty_separate": 6,
                      "train_strata": manifests["dev_train"]["strata"],
                      "validation_strata": manifests["dev_validation"]["strata"]}, sort_keys=True))


if __name__ == "__main__":
    main()
