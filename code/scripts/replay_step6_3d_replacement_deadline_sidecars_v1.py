"""Replay only eight replacement anchors, then compose the formal 96 sidecars."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/src")]
from replay_step6_3d_selected_deadline_sidecars_v1 import OUT, replay_selected

MANIFESTS = ("15_train_anchor_manifest_objective_eligible.json",
             "16_validation_anchor_manifest_objective_eligible.json")


def main() -> None:
    manifests = [json.loads((OUT / name).read_text(encoding="utf-8")) for name in MANIFESTS]
    replacements = {row["new_sample_id"] for manifest in manifests
                    for row in manifest["replacements"]}
    if len(replacements) != 8:
        raise ValueError("expected exactly eight distinct replacement IDs")
    replay_selected(manifest_names=MANIFESTS, selected_ids=replacements,
                    expected_count=8, sidecar_name="17_replacement_deadline_sidecars.json",
                    evidence_name="18_replacement_deadline_alignment_receipt.json")
    old_path = OUT / "03_selected_deadline_sidecars.json"
    new_path = OUT / "17_replacement_deadline_sidecars.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    selected = {row["sample_id"] for manifest in manifests for row in manifest["selected"]}
    if len(selected) != 96 or set(new) != replacements or not selected.issubset(set(old) | set(new)):
        raise ValueError("formal sidecar selection identity mismatch")
    merged = {sample_id: new[sample_id] if sample_id in new else old[sample_id]
              for sample_id in sorted(selected)}
    if any(not row["alignment_passed"] for row in merged.values()):
        raise ValueError("unaligned selected sidecar")
    (OUT / "19_formal_selected_deadline_sidecars.json").write_text(
        json.dumps(merged, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"selected_count": len(merged),
                      "retained_prior_aligned": len(merged) - len(new),
                      "new_exact_replay_aligned": len(new),
                      "prior_sidecar_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
                      "replacement_sidecar_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
