from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_ROOT / "scripts" / "audit_information_edge_contract_v4.py"
LEGACY_INFORMATION_EDGE_FEATURES = [
    "pre.interface_available",
    "pre.csi_mean",
    "pre.channel_gain",
    "pre.path_loss",
    "pre.noise",
    "pre.historical_interference",
    "pre.historical_sinr",
    "pre.historical_rate",
    "action.allocated_rb_count",
    "action.tx_power",
    "action.mcs",
    "outcome.active_task_count",
    "outcome.rate_sum",
    "outcome.actual_interference",
    "outcome.actual_sinr",
    "outcome.outage",
    "outcome.throughput",
    "outcome.served_data",
]


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "audit_information_edge_contract_v4", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load P1 audit script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v3_fixture(root: Path, *, invalid_nonzero: bool = False) -> Path:
    root.mkdir(parents=True)
    fieldnames = [
        "trajectory_id",
        "scenario_id",
        "seed",
        "split",
        "v3_seed_dir",
    ]
    rows = [
        {
            "trajectory_id": "train_0",
            "scenario_id": "load_low__density_sparse",
            "seed": 0,
            "split": "train",
            "v3_seed_dir": "seed_000",
        },
        {
            "trajectory_id": "validation_0",
            "scenario_id": "load_low__density_sparse",
            "seed": 1,
            "split": "validation",
            "v3_seed_dir": "seed_001",
        },
        {
            "trajectory_id": "calibration_0",
            "scenario_id": "load_low__density_sparse",
            "seed": 2,
            "split": "calibration",
            "v3_seed_dir": "seed_002",
        },
        {
            "trajectory_id": "locked_0",
            "scenario_id": "load_low__density_sparse",
            "seed": 9,
            "split": "locked_test",
            "v3_seed_dir": "seed_009",
        },
    ]
    with (root / "trajectory_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows[:3]:
        seed_dir = root / str(row["v3_seed_dir"])
        seed_dir.mkdir()
        values = np.zeros((2, 3, 18), np.float32)
        masks = np.zeros_like(values, dtype=bool)
        masks[..., [0, 1, 8, 11, 12]] = True
        masks[:, 2, :] = False
        values[..., 0] = 1.0
        values[..., 1] = 91.0
        if invalid_nonzero and row["split"] == "train":
            values[..., 2] = 7.0
        tensor_path = seed_dir / "trajectory_tensors.npz"
        np.savez_compressed(
            tensor_path,
            information_edge_state=values,
            information_edge_feature_mask=masks,
            information_edge_present=np.asarray(
                [[True, True, False], [True, True, False]], dtype=bool
            ),
            information_edge_kind_index=np.asarray([2, 2, -1], dtype=np.int16),
        )
        (seed_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "PIJWM-DG-Contract-v3-dataset",
                    "trajectory_id": row["trajectory_id"],
                    "seed": row["seed"],
                    "split": row["split"],
                    "files": {
                        "trajectory_tensors.npz": {
                            "sha256": _sha256(tensor_path),
                            "size_bytes": tensor_path.stat().st_size,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    locked_dir = root / "seed_009"
    locked_dir.mkdir()
    (locked_dir / "DO_NOT_OPEN").write_text("locked sentinel", encoding="utf-8")
    (root / "tensor_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "PIJWM-DG-Contract-v3-tensor",
                "information_edge_features": LEGACY_INFORMATION_EDGE_FEATURES,
                "information_edge_types": ["V2V", "V2U", "V2I"],
            }
        ),
        encoding="utf-8",
    )
    return root


class P1AuditSplitGuardTests(unittest.TestCase):
    def test_locked_rows_are_filtered_before_np_load(self):
        subject = load_subject()
        rows = [
            {
                "seed": "0",
                "trajectory_id": "train_0",
                "split": "train",
                "v3_seed_dir": "seed_000",
            },
            {
                "seed": "9",
                "trajectory_id": "locked_0",
                "split": "locked_test",
                "v3_seed_dir": "seed_009",
            },
        ]
        with patch.object(
            subject.np,
            "load",
            side_effect=AssertionError("np.load must not be reached"),
        ):
            selected, locked_metadata = subject.partition_trajectory_rows(rows)

        self.assertEqual(["train_0"], [row["trajectory_id"] for row in selected])
        self.assertEqual(
            [{"trajectory_id": "locked_0", "seed": 9, "split": "locked_test"}],
            locked_metadata,
        )

    def test_end_to_end_audit_never_calls_file_io_for_locked_seed(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = build_v3_fixture(root / "dataset")
            output = root / "output"
            original_load = subject.np.load
            original_manifest = subject._read_manifest
            original_hash = subject.sha256_file

            def guard(function):
                def guarded(path, *args, **kwargs):
                    self.assertNotIn("seed_009", str(path))
                    return function(path, *args, **kwargs)

                return guarded

            with (
                patch.object(subject.np, "load", side_effect=guard(original_load)),
                patch.object(
                    subject, "_read_manifest", side_effect=guard(original_manifest)
                ),
                patch.object(subject, "sha256_file", side_effect=guard(original_hash)),
            ):
                result = subject.run_audit(
                    v3_dataset_root=dataset, output_dir=output
                )
            self.assertFalse(result["locked_test_accessed"])

    def test_unknown_split_and_seed_path_escape_are_rejected(self):
        subject = load_subject()
        with self.assertRaisesRegex(ValueError, "unknown split"):
            subject.partition_trajectory_rows(
                [
                    {
                        "seed": "0",
                        "trajectory_id": "bad",
                        "split": "mystery",
                        "v3_seed_dir": "seed_000",
                    }
                ]
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                subject.resolve_seed_dir(
                    Path(temporary_directory),
                    {
                        "seed": "0",
                        "trajectory_id": "bad",
                        "split": "train",
                        "v3_seed_dir": "../outside",
                    },
                )


class P1AuditEvidenceTests(unittest.TestCase):
    def test_legacy_coverage_preserves_five_of_eighteen_fact(self):
        subject = load_subject()
        values = np.zeros((2, 3, 18), np.float32)
        masks = np.zeros_like(values, dtype=bool)
        masks[..., [0, 1, 8, 11, 12]] = True
        masks[:, 2, :] = False

        rows = subject.summarize_legacy_coverage(
            trajectory_id="train_0",
            split="train",
            scenario_id="load_low__density_sparse",
            edge_type="V2I",
            values=values,
            masks=masks,
            presence_mask=np.asarray(
                [[True, True, False], [True, True, False]], dtype=bool
            ),
        )

        by_index = {row["legacy_index"]: row for row in rows}
        self.assertEqual(4, by_index[0]["valid_count"])
        self.assertEqual(4, by_index[0]["total_count"])
        self.assertEqual(0, by_index[2]["valid_count"])
        self.assertEqual("load_low__density_sparse", by_index[0]["scenario_id"])
        self.assertEqual("V2I", by_index[0]["edge_type"])
        self.assertEqual("legacy_observation_only", by_index[12]["evidence_scope"])
        self.assertFalse(by_index[12]["v4_field_implemented"])

    def test_micro_sample_separates_fixture_and_observed_origins(self):
        subject = load_subject()
        sample = subject.build_micro_sample(
            observed_candidates=[
                {
                    "trajectory_id": "train_0",
                    "split": "train",
                    "seed": 0,
                    "time_index": 0,
                    "edge_index": 0,
                    "legacy_index": 1,
                    "value": 91.0,
                    "valid": True,
                }
            ]
        )

        origins = sample["sample_origin"]
        self.assertIn("observed_nonlocked", origins.tolist())
        self.assertIn("contract_fixture", origins.tolist())
        self.assertFalse(sample["training_eligible"][origins == "contract_fixture"].any())
        self.assertFalse(sample["training_eligible"].any())
        observed = origins == "observed_nonlocked"
        self.assertEqual(
            ["pre.csi_mean"], sample["field_name"][observed].tolist()
        )
        self.assertEqual(
            ["pre_link.channel_attenuation_mean_db"],
            sample["candidate_v4_target"][observed].tolist(),
        )
        self.assertFalse(sample["v4_field_implemented"].any())


class P1AuditBundleTests(unittest.TestCase):
    def test_reordered_contract_and_wrong_manifest_version_are_rejected(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = build_v3_fixture(root / "dataset")
            contract_path = dataset / "tensor_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["information_edge_features"] = list(
                reversed(contract["information_edge_features"])
            )
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with patch.object(
                subject.np,
                "load",
                side_effect=AssertionError("tensor load must follow contract validation"),
            ):
                with self.assertRaisesRegex(ValueError, "feature order mismatch"):
                    subject.run_audit(
                        v3_dataset_root=dataset, output_dir=root / "bad_contract"
                    )

            dataset = build_v3_fixture(root / "dataset_bad_manifest")
            manifest_path = dataset / "seed_000" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "wrong-version"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest schema mismatch"):
                subject.run_audit(
                    v3_dataset_root=dataset,
                    output_dir=root / "bad_manifest",
                )

    def test_numeric_legacy_masks_are_rejected(self):
        subject = load_subject()
        values = np.zeros((1, 1, 18), np.float32)
        with self.assertRaisesRegex(ValueError, "mask must have bool dtype"):
            subject.summarize_legacy_coverage(
                trajectory_id="train_0",
                split="train",
                scenario_id="s",
                edge_type="V2I",
                values=values,
                masks=np.zeros_like(values, dtype=np.uint8),
                presence_mask=np.asarray([[True]]),
            )

    def test_audit_writes_complete_hashed_bundle_without_locked_content(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = build_v3_fixture(root / "dataset")
            output = root / "output"

            result = subject.run_audit(v3_dataset_root=dataset, output_dir=output)

            expected = {
                "field_registry.json",
                "legacy_18_slot_mapping.csv",
                "micro_sample.npz",
                "field_coverage.csv",
                "rejected_records.csv",
                "audit_summary.json",
                "manifest.json",
            }
            self.assertEqual(expected, {path.name for path in output.iterdir()})
            self.assertTrue(result["p1_mvs_complete"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertFalse(result["gpu_started"])
            self.assertFalse(result["v4_dataset_complete"])
            self.assertEqual("self_audited_p1_mvs", result["evidence_scope"])
            self.assertEqual(5, result["legacy_valid_feature_count"])
            self.assertEqual(13, result["legacy_missing_feature_count"])
            self.assertEqual(1, result["scenario_count"])
            self.assertEqual(["V2I"], result["observed_edge_types"])

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(expected - {"manifest.json"}, set(manifest["files"]))
            self.assertEqual(2, len(manifest["code_files"]))
            self.assertEqual(
                manifest["files"]["field_registry.json"]["sha256"],
                manifest["protocol_sha256"],
            )
            for name, metadata in manifest["files"].items():
                self.assertEqual(
                    metadata["sha256"], subject.sha256_file(output / name)
                )
                self.assertEqual(metadata["size_bytes"], (output / name).stat().st_size)

    def test_failure_does_not_publish_success_and_retains_rejection_evidence(self):
        subject = load_subject()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = build_v3_fixture(root / "dataset", invalid_nonzero=True)
            output = root / "output"

            with self.assertRaisesRegex(ValueError, "invalid nonzero"):
                subject.run_audit(v3_dataset_root=dataset, output_dir=output)

            self.assertFalse(output.exists())
            failure = root / "output_failed"
            self.assertTrue(failure.is_dir())
            rejected = subject.read_csv(failure / "rejected_records.csv")
            self.assertEqual(1, len(rejected))
            self.assertIn("invalid nonzero", rejected[0]["detail"])
            summary = json.loads(
                (failure / "audit_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["p1_mvs_complete"])


if __name__ == "__main__":
    unittest.main()
