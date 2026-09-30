"""Package only the frozen non-locked TRAIN fixtures required by the GPU gate."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1")
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930"
CHECKPOINT = Path("code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt")
EXPECTED_CHECKPOINT_SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"
SELECTED = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/15_train_anchor_manifest_objective_eligible.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))["selected"]
    fixture_ids = [selected[index]["sample_id"] for index in (0, 2)]
    sample_index = json.loads((ROOT / BASE / "packages/samples/index.json").read_text(encoding="utf-8"))
    rows = [next(row for row in sample_index if row["metadata"]["sample_id"] == sample_id)
            for sample_id in fixture_ids]
    trajectory_ids = {row["metadata"]["trajectory_id"] for row in rows}
    paths = [BASE / name for name in (
        "formal_dataset_manifest.json", "split_manifest.json",
        "packages/samples/index.json", "packages/tensor/index.json",
    )]
    paths.extend(BASE / "packages/normalization" / p.relative_to(ROOT / BASE / "packages/normalization")
                 for p in (ROOT / BASE / "packages/normalization").rglob("*") if p.is_file())
    paths.extend(BASE / "packages/graph" / p.relative_to(ROOT / BASE / "packages/graph")
                 for p in (ROOT / BASE / "packages/graph").rglob("*") if p.is_file())
    for trajectory_id in sorted(trajectory_ids):
        paths += [BASE / "packages/samples" / f"{trajectory_id}.json.gz",
                  BASE / "packages/tensor" / f"{trajectory_id}.npz",
                  BASE / "raw" / f"{trajectory_id}.json.gz"]
    paths.append(CHECKPOINT)
    if sha(ROOT / CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("frozen checkpoint mismatch")
    entries = []
    for relative in sorted(set(paths)):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                        "sha256": sha(path)})
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "gpu_fixture_bundle.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as target:
        for entry in entries:
            target.add(ROOT / entry["path"], arcname=entry["path"], recursive=False)
    result = {"fixture_sample_ids": fixture_ids,
              "fixture_trajectory_ids": sorted(trajectory_ids),
              "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
              "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha(archive), "files": entries,
              "locked_test": False, "training": False,
              "validation_search": False}
    (OUT / "01_fixture_transfer_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"files": len(entries), "archive_bytes": archive.stat().st_size,
                      "archive_sha256": result["archive_sha256"]}))


if __name__ == "__main__":
    main()
