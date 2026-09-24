from __future__ import annotations

import copy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from pi_jwm.step5_5_lifecycle_repair_v1 import TASK_COLLECTION_PRIORITY, repair_duplicate_task_references
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_fixed_support_audit_v1 import detect_future_return_birth
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FormalTrajectorySampler
from pi_jwm.step5_1a_motion_csi_target_contract_v1 import extend_future_motion_csi_targets, build_future_target_tensor_batch, load_future_target_tensor_batch
from build_step5_1d_unified_model_chain_v1 import build_action


MANIFEST = Path(__file__).parents[1] / "artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"


class FakeTask:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.transmitted = 2.5
        self.computed = 1.5
        self.returned = 0.5
        self.done = True

    def getTaskId(self) -> str:
        return self.task_id


class FakeManager:
    def __init__(self):
        for name in TASK_COLLECTION_PRIORITY:
            setattr(self, name, {})


class Step55PatchTests(unittest.TestCase):
    def test_wrong_normalization_package_hash_rejected(self) -> None:
        interface = FormalTrainingInterface.from_manifest(MANIFEST)
        forged = replace(interface, manifest={**interface.manifest, "hashes": {**interface.manifest["hashes"], "normalization": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "normalization package hash mismatch"):
            FullFormalShardDataset(forged)

    def test_formal_batch_needs_no_local_raw_path(self) -> None:
        loader = FullFormalShardDataset(FormalTrainingInterface.from_manifest(MANIFEST))
        with mock.patch("pi_jwm.step5_5_full_sharded_loader_v1.ROOT", Path("Z:/absent-pijwm-root")):
            batch = loader.load_batch([loader.interface.train_indices[0]])
        self.assertEqual(1, len(batch.states))
        self.assertEqual(1, len(batch.samples))

    def test_full_index_and_cross_shard_batch(self) -> None:
        loader = FullFormalShardDataset(FormalTrainingInterface.from_manifest(MANIFEST))
        self.assertEqual((4416, 1104), (len(loader.interface.train_indices), len(loader.interface.validation_indices)))
        self.assertEqual(5520, len({row["metadata"]["sample_id"] for row in loader.index}))
        for split, indices in (("dev_train", loader.interface.train_indices), ("dev_validation", loader.interface.validation_indices)):
            self.assertEqual(len(indices), sum(row["metadata"]["split"] == split for row in loader.index))
            self.assertEqual(len(indices), len({(loader.index[i]["shard"], loader.index[i]["shard_index"]) for i in indices}))
        for indices in ((0, 92), (4416, 4508)):
            batch = loader.load_batch(indices)
            self.assertEqual([loader.index[i]["metadata"]["sample_id"] for i in indices], [sample["metadata"]["sample_id"] for sample in batch.samples])
            self.assertEqual(2, batch.tensor["entity_presence"].shape[0])
            self.assertEqual(2, batch.graph["blocks"]["physical_nodes"]["presence"].shape[0])
            self.assertEqual(2, batch.target_tensors["target_vehicle_motion_mask"].shape[0])
        self.assertEqual(2, loader.peak_loaded_shards)

    def test_formal_trajectory_sampler_epoch_exact_once_and_reproducible(self) -> None:
        loader = FullFormalShardDataset(FormalTrainingInterface.from_manifest(MANIFEST))
        sampler = FormalTrajectorySampler(loader.index, loader.interface.train_indices, seed=5201)
        order0 = sampler.order_for_epoch(0)
        order0_repeat = sampler.order_for_epoch(0)
        order1 = sampler.order_for_epoch(1)
        self.assertEqual(order0, order0_repeat)
        self.assertNotEqual(order0, order1)
        self.assertEqual(set(order0), set(loader.interface.train_indices))
        self.assertEqual(len(order0), len(set(order0)))
        self.assertTrue(all(loader.index[i]["metadata"]["split"] == "dev_train" for i in order0))
        batches = []
        for step in range((len(order0) + 7) // 8):
            batches.extend(sampler.batch_for_global_step(step, 8))
        self.assertEqual(list(order0), batches)
        self.assertEqual(sampler.state_dict()["train_trajectories"], 48)
        resumed = FormalTrajectorySampler(loader.index, loader.interface.train_indices, seed=sampler.state_dict()["seed"])
        self.assertEqual(sampler.batch_for_global_step((len(order0) + 7) // 8 + 3, 8),
                         resumed.batch_for_global_step((len(order0) + 7) // 8 + 3, 8))

    def test_future_return_comm_action_keeps_current_support_fixed(self) -> None:
        loader = FullFormalShardDataset(FormalTrainingInterface.from_manifest(MANIFEST))
        sample_id = "formal-v1-sim-2026092307-policy-2026092407::anchor-0058"
        index = next(i for i, row in enumerate(loader.index) if row["metadata"]["sample_id"] == sample_id)
        batch = loader.load_batch([index])
        before = copy.deepcopy(batch.samples[0]["static"]["input_entity_index"]["logical_flow"])
        action, audit = build_action(batch.samples[0], batch.states[0], 1)
        blocked = [row for row in audit["comm"] if row["task_index"] == 22]
        self.assertEqual(2, len(blocked))
        self.assertTrue(all(row["mode"] == "future_return_birth_fixed_support_blocked" for row in blocked))
        self.assertEqual(before, batch.samples[0]["static"]["input_entity_index"]["logical_flow"])
        self.assertEqual(len(batch.samples[0]["future_action"][1]["comm"]["entries"]) - len(blocked), action["comm_relation_index"].numel())
        self.assertFalse(bool(action["comm_allocation_mask"].any()))
        older_input = {key: value.clone() for key, value in batch.states[0].items()}
        input_slot = batch.samples[0]["static"]["input_entity_index"]["logical_flow"]["flow::Task_34::Input::0"]
        older_input["flow_presence"][0, input_slot] = True
        _, with_input_audit = build_action(batch.samples[0], older_input, 1)
        self.assertTrue(all(row["mode"] == "future_return_birth_fixed_support_blocked"
                            for row in with_input_audit["comm"] if row["task_index"] == 22))

    def test_comm_action_rebinds_current_typed_support_after_predicted_completion(self) -> None:
        loader = FullFormalShardDataset(FormalTrainingInterface.from_manifest(MANIFEST))
        sample_id = "formal-v1-sim-2026092302-policy-2026092402::anchor-0022"
        index = next(i for i, row in enumerate(loader.index) if row["metadata"]["sample_id"] == sample_id)
        batch = loader.load_batch([index])
        state = {key: value.clone() for key, value in batch.states[0].items()}
        slots = np.flatnonzero((state["flow_task_index"][0] == 11).numpy() & state["flow_presence"][0].numpy())
        self.assertEqual(1, len(slots))
        state["flow_presence"][0, int(slots[0])] = False
        state["flow_comm_relation_index"][0, int(slots[0])] = -1
        action, audit = build_action(batch.samples[0], state, 1)
        rows = [row for row in audit["comm"] if row["task_index"] == 11]
        self.assertEqual(1, len(rows))
        self.assertEqual("typed_current_support_predicted_inactive", rows[0]["mode"])
        self.assertFalse(rows[0]["transition_applied"])
        self.assertGreaterEqual(int(rows[0]["relation_index"]), 0)
        self.assertTrue(bool(action["comm_allocation_mask"][0, rows[0]["relation_index"], 25]))
        # A completed Input and an active Return can share a task; the causal
        # current Return identity disambiguates the typed support.
        second_id = "formal-v1-sim-2026092302-policy-2026092402::anchor-0041"
        second_index = next(i for i, row in enumerate(loader.index) if row["metadata"]["sample_id"] == second_id)
        second = loader.load_batch([second_index])
        state = {key: value.clone() for key, value in second.states[0].items()}
        task_slots = np.flatnonzero((state["flow_task_index"][0] == 4).numpy())
        self.assertEqual(2, len(task_slots))
        for slot in task_slots:
            state["flow_presence"][0, int(slot)] = False
            state["flow_comm_relation_index"][0, int(slot)] = -1
        _, audit = build_action(second.samples[0], state, 1)
        rows = [row for row in audit["comm"] if row["task_index"] == 4]
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["mode"] == "typed_current_support_predicted_inactive" for row in rows))

    def test_shard_hash_tamper_rejected(self) -> None:
        loader = FullFormalShardDataset(FormalTrainingInterface.from_manifest(MANIFEST))
        trajectory_id = loader.index[0]["metadata"]["trajectory_id"]
        original = loader.shards[trajectory_id]["files"]["tensor"]["sha256"]
        try:
            loader.shards[trajectory_id]["files"]["tensor"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                loader.load_batch([0])
        finally:
            loader.shards[trajectory_id]["files"]["tensor"]["sha256"] = original

    def test_future_return_birth_positive_negative_and_component_retention(self) -> None:
        flow_id = "flow::Task_7::Return::0"
        future = {"logical_flows": [{"flow_type": "Return", "flow_id": flow_id, "task_id": "Task_7", "known": True, "presence": True}], "vehicle_motion_targets": [{"mask": [True] * 4}], "comm_csi_targets": [{"mask": [True, False]}]}
        no_return = {"static": {"input_entity_index": {"logical_flow": {"flow::Task_7::Input::0": 0}}}}
        before = copy.deepcopy(no_return)
        positive = detect_future_return_birth(no_return, future)
        self.assertEqual((1, 0, 1), (positive["unsupported_count"], positive["unresolved_count"], positive["fixed_support_blocked_count"]))
        self.assertEqual(before, no_return)
        self.assertTrue(positive["window_retained"])
        self.assertEqual([True] * 4, future["vehicle_motion_targets"][0]["mask"])
        self.assertEqual([True, False], future["comm_csi_targets"][0]["mask"])
        existing = {"static": {"input_entity_index": {"logical_flow": {flow_id: 0}}}}
        negative = detect_future_return_birth(existing, future)
        self.assertEqual((0, 0, 0), (negative["unsupported_count"], negative["unresolved_count"], negative["fixed_support_blocked_count"]))
        completed_future = copy.deepcopy(future)
        completed_future["logical_flows"][0].update({"status": "COMPLETED", "presence": False})
        self.assertEqual(1, detect_future_return_birth(no_return, completed_future)["unsupported_count"])
        self.assertEqual(0, detect_future_return_birth(existing, completed_future)["unsupported_count"])
        unknown_future = copy.deepcopy(future)
        unknown_future["logical_flows"][0].update({"known": False, "flow_id": None})
        self.assertEqual((0, 1, 0), tuple(detect_future_return_birth(no_return, unknown_future)[key] for key in ("unsupported_count", "unresolved_count", "fixed_support_blocked_count")))

    def test_real_future_birth_side_accounting_keeps_motion_csi_arrays(self) -> None:
        dataset = MANIFEST.parent
        name = "formal-v1-sim-2026092300-policy-2026092400"
        with gzip.open(dataset / "packages/samples" / f"{name}.json.gz", "rt", encoding="utf-8") as handle:
            sample = json.load(handle)[7]
        with gzip.open(dataset / "raw" / f"{name}.json.gz", "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        stats = json.loads((dataset / "packages/normalization/stats.json").read_text(encoding="utf-8"))["extension_step4_2a"]
        extended = extend_future_motion_csi_targets(sample, raw, stats)
        self.assertEqual([0, 0, 0, 1], [frame["future_target_side_metadata"]["unsupported_count"] for frame in extended["target"]])
        current_index = sample["static"]["input_entity_index"]["logical_flow"]
        self.assertNotIn("flow::Task_1::Return::0", current_index)
        target = build_future_target_tensor_batch([extended], stats)
        stored = load_future_target_tensor_batch(dataset / "packages/target" / f"{name}.npz")
        for key, value in target.items():
            if key.startswith("target_") and isinstance(value, np.ndarray):
                stored_slice = stored[key][(slice(7, 8), *[slice(0, width) for width in value.shape[1:]])]
                self.assertTrue(np.array_equal(value, stored_slice), key)

    def test_lifecycle_repair_same_object_and_distinct_object_rejection(self) -> None:
        task = FakeTask("Task_7")
        manager = FakeManager()
        manager._offloading_tasks["UAV_0"] = [task, task]
        manager._done_tasks["UAV_1"] = [task]
        snapshot = vars(task).copy()
        repairs = repair_duplicate_task_references(manager, 17)
        self.assertEqual(1, len(repairs))
        self.assertEqual("_done_tasks:UAV_1", repairs[0]["kept"])
        self.assertEqual([], manager._offloading_tasks["UAV_0"])
        self.assertIs(task, manager._done_tasks["UAV_1"][0])
        self.assertEqual(snapshot, vars(task))
        broken = FakeManager()
        broken._offloading_tasks["UAV_0"] = [FakeTask("Task_7")]
        broken._done_tasks["UAV_1"] = [FakeTask("Task_7")]
        with self.assertRaisesRegex(RuntimeError, "distinct Task objects"):
            repair_duplicate_task_references(broken, 17)


if __name__ == "__main__":
    unittest.main()
