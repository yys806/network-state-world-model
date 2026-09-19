import copy
import json
import tempfile
import unittest
from pathlib import Path

from pi_jwm.step3_2_batch_preprocessing_v1 import (
    RawSource,
    apply_normalization,
    audit_batch_future_action_references,
    build_batch,
    collate_samples,
    fit_train_normalization_stats,
    load_batch_bundle,
    write_batch_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"


class Step32BatchPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))

    def _write_raw(self, directory: Path, trajectory_id: str, seed: int) -> Path:
        payload = copy.deepcopy(self.raw)
        for decision in payload["decisions"]:
            decision["trajectory_id"] = trajectory_id
        payload["environment"]["seed"] = seed
        payload["environment"]["config_hash"] = f"config-{seed}"
        path = directory / f"{trajectory_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_split_is_trajectory_level_and_windows_are_reverse_traceable(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train_a = self._write_raw(directory, "dev-train-a", 101)
            train_b = self._write_raw(directory, "dev-train-b", 102)
            validation = self._write_raw(directory, "dev-validation-a", 201)
            bundle = build_batch([
                RawSource(train_a, "dev_train"),
                RawSource(train_b, "dev_train"),
                RawSource(validation, "dev_validation"),
            ])
        samples = bundle["samples"]
        self.assertGreater(len(samples), 0)
        self.assertEqual(
            {"dev-train-a", "dev-train-b"},
            {row["metadata"]["trajectory_id"] for row in samples if row["metadata"]["split"] == "dev_train"},
        )
        self.assertEqual(
            {"dev-validation-a"},
            {row["metadata"]["trajectory_id"] for row in samples if row["metadata"]["split"] == "dev_validation"},
        )
        self.assertEqual(len(samples), len({row["metadata"]["sample_id"] for row in samples}))
        for sample in samples:
            metadata = sample["metadata"]
            self.assertIn("trajectory_id", metadata)
            self.assertIn("anchor_decision_frame", metadata)
            self.assertEqual(metadata["split"], metadata["source_split"])
            self.assertEqual(metadata["history_frame_indices"][-1], metadata["anchor_decision_frame"])

    def test_duplicate_trajectory_id_is_rejected_even_when_source_version_differs(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = self._write_raw(directory, "same-run", 101)
            second = self._write_raw(directory, "same-run", 102)
            with self.assertRaisesRegex(ValueError, "duplicate trajectory_id"):
                build_batch([RawSource(first, "dev_train"), RawSource(second, "dev_validation")])

    def test_train_only_mask_aware_statistics_and_counterfactuals(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train = self._write_raw(directory, "train", 101)
            validation = self._write_raw(directory, "validation", 201)
            bundle = build_batch([RawSource(train, "dev_train"), RawSource(validation, "dev_validation")])
        samples = bundle["samples"]
        stats = fit_train_normalization_stats(samples)
        validation_mutated = copy.deepcopy(samples)
        validation_row = next(row for row in validation_mutated if row["metadata"]["split"] == "dev_validation")
        validation_row["history"][0]["entities"][0]["speed_mps"]["value"] = 1e12
        validation_stats = fit_train_normalization_stats(validation_mutated)
        self.assertEqual(stats, validation_stats)

        train_mutated = copy.deepcopy(samples)
        train_row = next(row for row in train_mutated if row["metadata"]["split"] == "dev_train")
        train_row["history"][0]["entities"][0]["speed_mps"]["value"] += 1e6
        changed_stats = fit_train_normalization_stats(train_mutated)
        self.assertNotEqual(stats, changed_stats)

        masked_mutated = copy.deepcopy(samples)
        masked_row = masked_mutated[0]["history"][0]["entities"][0]["canonical_acceleration_mps2"]
        masked_row["feature_mask"] = False
        masked_row["value"] = 1e12
        masked_stats = fit_train_normalization_stats(masked_mutated)
        self.assertEqual(stats, masked_stats)

    def test_normalization_apply_collate_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train = self._write_raw(directory, "train", 101)
            validation = self._write_raw(directory, "validation", 201)
            bundle = build_batch([RawSource(train, "dev_train"), RawSource(validation, "dev_validation")])
            stats = fit_train_normalization_stats(bundle["samples"])
            normalized = apply_normalization(bundle["samples"], stats)
            validation_rows = [row for row in normalized if row["metadata"]["split"] == "dev_validation"]
            batch = collate_samples(validation_rows)
            self.assertEqual(len(validation_rows), len(batch["sample_ids"]))
            self.assertEqual(len(batch["sample_ids"]), len(set(batch["sample_ids"])))
            self.assertEqual("dev_validation", batch["split"])
            self.assertTrue(any(
                "normalized_value" in feature
                for sample in validation_rows
                for frame in sample["history"]
                for row in frame["entities"]
                for feature in (row["speed_mps"], row["canonical_acceleration_mps2"])
            ))

            output = directory / "bundle"
            write_batch_bundle(bundle, stats, normalized, output)
            loaded = load_batch_bundle(output)
            self.assertEqual(bundle["samples"], loaded["bundle"]["samples"])
            self.assertEqual(stats, loaded["stats"])
            self.assertEqual(normalized, loaded["normalized"])

    def test_non_contiguous_frames_are_rejected_before_windowing(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            path = self._write_raw(directory, "broken", 101)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["steps"][2]["frame_index"] = 7
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                build_batch([RawSource(path, "dev_train")])

    def test_provenance_time_grid_and_units_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train = self._write_raw(directory, "train", 101)
            validation = self._write_raw(directory, "validation", 201)
            bundle = build_batch([RawSource(train, "dev_train"), RawSource(validation, "dev_validation")])
        self.assertEqual({"train", "validation"}, {row["trajectory_id"] for row in bundle["provenance"]})
        for row in bundle["provenance"]:
            self.assertIn("source_path", row)
            self.assertIn("source_sha256", row)
            self.assertEqual(row["seed"], 101 if row["trajectory_id"] == "train" else 201)
            self.assertEqual(row["config_hash"], f"config-{row['seed']}")
            self.assertEqual(row["decision_frame_range"], [0, 6])
            self.assertEqual(row["step_frame_range"], [0, 5])
            self.assertEqual(row["slot_duration_s"], 0.1)
            self.assertEqual(row["model_ready_sample_contract_version"], "PI-JWM-Model-Ready-Sample-Contract-v3-step3.1F")
        self.assertEqual(bundle["normalization_units"], {
            "entity.speed_mps": "m/s",
            "entity.canonical_acceleration_mps2": "m/s^2",
            "task.task_size": "AirFogSim data-unit",
        })

    def test_time_gap_is_rejected_before_windowing(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            path = self._write_raw(directory, "gap", 101)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["decisions"][2]["simulation_time_s"] += 0.25
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "time grid"):
                build_batch([RawSource(path, "dev_train")])

    def test_presence_false_padding_does_not_change_statistics(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train = self._write_raw(directory, "train", 101)
            bundle = build_batch([RawSource(train, "dev_train")])
        stats = fit_train_normalization_stats(bundle["samples"])
        mutated = copy.deepcopy(bundle["samples"])
        mutated[0]["history"][0]["entities"].append({
            "entity_id": "padding",
            "presence": False,
            "speed_mps": {"value": 1e12, "presence": True, "feature_mask": True},
            "canonical_acceleration_mps2": {"value": 1e12, "presence": True, "feature_mask": True},
            "feature_mask": {"speed_mps": True, "canonical_acceleration_mps2": True},
        })
        self.assertEqual(stats, fit_train_normalization_stats(mutated))

    def test_batch_future_reference_audit_is_observation_only(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train = self._write_raw(directory, "train", 101)
            validation = self._write_raw(directory, "validation", 201)
            bundle = build_batch([RawSource(train, "dev_train"), RawSource(validation, "dev_validation")])
            audit = audit_batch_future_action_references(
                [RawSource(train, "dev_train"), RawSource(validation, "dev_validation")],
                bundle=bundle,
            )
        self.assertEqual(audit["total_candidate_windows"], 8)
        self.assertEqual(audit["successfully_constructed_windows"], 8)
        self.assertEqual(audit["unresolved_reference_windows"], 0)
        self.assertEqual(audit["unresolved_reference_count"], 0)
        self.assertTrue(audit["observation_only"])


if __name__ == "__main__":
    unittest.main()
