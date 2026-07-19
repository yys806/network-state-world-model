import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class PhysicalBenefitBuilderContractTest(unittest.TestCase):
    def _args(self, root: Path, dry_run: bool = True) -> Namespace:
        paths = {}
        for name in (
            "physical_train_csv",
            "physical_calibration_csv",
            "train_cache",
            "calibration_cache",
            "validation_cache",
            "sample_index_csv",
        ):
            path = root / f"{name}.dat"
            path.write_text(name, encoding="utf-8")
            paths[name] = path
        return Namespace(
            **paths,
            output_dir=root / "output",
            dry_run=dry_run,
        )

    def test_cli_requires_all_alignment_and_cache_paths(self):
        from build_v11_physical_benefit_bridge import parse_args

        with mock.patch.object(sys, "argv", ["builder"]):
            with self.assertRaises(SystemExit):
                parse_args()

        argv = [
            "builder",
            "--physical-train-csv", "train.csv",
            "--physical-calibration-csv", "calibration.csv",
            "--train-cache", "train.npz",
            "--calibration-cache", "calibration.npz",
            "--validation-cache", "validation.npz",
            "--sample-index-csv", "sample_index.csv",
            "--output-dir", "report",
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.dry_run)
        self.assertEqual(args.sample_index_csv, Path("sample_index.csv"))

    def test_dry_run_writes_input_hashes_without_building_caches(self):
        from build_v11_physical_benefit_bridge import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            summary = run(args)

            saved = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary, saved)
        self.assertEqual(set(summary["input_sha256"]), {
            "physical_train_csv",
            "physical_calibration_csv",
            "train_cache",
            "calibration_cache",
            "validation_cache",
            "sample_index_csv",
        })
        self.assertEqual(summary["augmented_caches"], {})
        self.assertFalse(summary["matched_test_accessed"])
        self.assertFalse(summary["external_holdout_accessed"])

    def test_augmented_cache_feature_order_is_identical_and_physical_last(self):
        from build_v11_physical_benefit_bridge import validate_augmented_feature_order
        from pi_jwm.v11_physical_benefit import PHYSICAL_PREDICTION_FEATURES

        names = ["base_a", "base_b", *PHYSICAL_PREDICTION_FEATURES]
        manifests = {
            split: {"feature_names": names}
            for split in ("train", "calibration", "validation")
        }

        self.assertEqual(
            validate_augmented_feature_order(manifests),
            tuple(names),
        )
        manifests["validation"] = {"feature_names": list(reversed(names))}
        with self.assertRaisesRegex(ValueError, "feature order"):
            validate_augmented_feature_order(manifests)


class PhysicalBenefitSelectorGuardTest(unittest.TestCase):
    def _manifest(self, root: Path, gate_passed: bool = True):
        from pi_jwm.v11_selector import canonical_sha256, file_sha256

        cache_paths = {}
        augmented = {}
        for split in ("train", "calibration", "validation"):
            path = root / f"{split}.npz"
            path.write_bytes(f"cache-{split}".encode("ascii"))
            cache_paths[split] = path
            augmented[split] = {"path": path.name, "sha256": file_sha256(path)}
        payload = {
            "result_kind": "diagnostic_only",
            "bridge_gate_passed": gate_passed,
            "matched_test_accessed": False,
            "external_holdout_accessed": False,
            "actual_outcome_feature_count": 0,
            "augmented_caches": augmented,
        }
        manifest = {**payload, "bridge_manifest_digest": canonical_sha256(payload)}
        manifest_path = root / "bridge_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, cache_paths

    def test_selector_guard_accepts_only_bound_passed_bridge(self):
        from train_v11_candidate_set_selector import validate_physical_bridge_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, cache_paths = self._manifest(root)
            digest = validate_physical_bridge_manifest(manifest_path, cache_paths)

            self.assertEqual(len(digest), 64)
            failed_path, failed_caches = self._manifest(root, gate_passed=False)
            with self.assertRaisesRegex(ValueError, "gate"):
                validate_physical_bridge_manifest(failed_path, failed_caches)

    def test_selector_guard_rejects_digest_lock_and_cache_tampering(self):
        from train_v11_candidate_set_selector import validate_physical_bridge_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, cache_paths = self._manifest(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["matched_test_accessed"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_physical_bridge_manifest(manifest_path, cache_paths)

            manifest_path, cache_paths = self._manifest(root)
            cache_paths["train"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "cache hash"):
                validate_physical_bridge_manifest(manifest_path, cache_paths)

    def test_physical_cache_requires_manifest_but_legacy_cache_does_not(self):
        from train_v11_candidate_set_selector import validate_physical_cache_presence

        legacy = {
            split: {"feature_names": ["predicted_task_delta_8"]}
            for split in ("train", "calibration", "validation")
        }
        physical = {
            split: {"feature_names": ["predicted_task_delta_8", "physical_task_delta_lcb"]}
            for split in ("train", "calibration", "validation")
        }

        self.assertFalse(validate_physical_cache_presence(legacy, manifest_supplied=False))
        self.assertTrue(validate_physical_cache_presence(physical, manifest_supplied=True))
        with self.assertRaisesRegex(ValueError, "requires --physical-bridge-manifest"):
            validate_physical_cache_presence(physical, manifest_supplied=False)
        with self.assertRaisesRegex(ValueError, "does not contain physical"):
            validate_physical_cache_presence(legacy, manifest_supplied=True)


if __name__ == "__main__":
    unittest.main()
