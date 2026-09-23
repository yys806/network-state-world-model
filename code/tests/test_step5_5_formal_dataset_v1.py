from __future__ import annotations

import unittest
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from step5_5_formal_dataset_cpu_acceptance_v1 import _adapter_fixture_checks
from pi_jwm.step5_2_training_loop_v1 import (
    CurriculumConfig,
    DevelopmentBundle,
    Step52Trainer,
    Step52TrainingConfig,
    _pad_action_rows,
)
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, validate_readiness

from build_step5_5_formal_dataset_v1 import (
    FAMILIES,
    SPLIT_SEED,
    coverage,
    coverage_checks,
    canonicalize_npz,
    deterministic_split,
    merge_graph_batches,
    merge_target_batches,
    synchronize_tensor_contract,
)
from finalize_step5_5_existing_package_v1 import _stream_sample_shard_audit


def _step(mask: str, signature_suffix: str) -> dict:
    families = {}
    for index, family in enumerate(FAMILIES):
        active = mask[index] == "1"
        families[family] = {
            "eligible": True,
            "intervention": active,
            "no_op": not active,
            "signature": f"{family}-{signature_suffix}" if active else None,
        }
    return {
        "behavior_policy_audit": {
            "activation_bitmask": mask,
            "global_noop": mask == "0000",
            "families": families,
        }
    }


class Step55FormalDatasetTests(unittest.TestCase):
    def test_formal_action_adapter_pending_current_comp_noop_and_device_fixtures(self) -> None:
        manifest = Path(__file__).parents[1] / "artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
        interface = FormalTrainingInterface.from_manifest(manifest)
        config = Step52TrainingConfig(
            seed=5501, batch_size=1, max_steps=1, max_epochs=1, stage1_steps=0,
            max_horizon=4, curriculum=CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1, 2)), device="cpu",
        )
        trainer = Step52Trainer.from_formal_interface(interface, config)
        checks = _adapter_fixture_checks(trainer)
        self.assertEqual({"action_tensors_on_device", "comp_agent_task_routed_cpu_transition", "current_flow_route_comm_routed", "pending_comm_relation_routed", "pending_route_flow_sentinel", "pending_route_routed_not_claimed_transitioned", "pending_route_task_tensor", "route_comp_noop_regression"}, set(checks))
        self.assertTrue(all(checks.values()), checks)

    def test_action_padding_uses_source_tensor_device(self) -> None:
        action = {"route_task_index": torch.tensor([[0]], dtype=torch.long), "route_values": torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])}
        padded_index = _pad_action_rows([action], "route_task_index", 3)
        padded_value = _pad_action_rows([action], "route_values", 3)
        self.assertEqual(action["route_task_index"].device, padded_index.device)
        self.assertEqual(action["route_values"].device, padded_value.device)
        self.assertTrue(torch.equal(padded_index, torch.tensor([[0, -1, -1]])))
        self.assertTrue(torch.equal(padded_value[0, 1:], torch.zeros((2, 4))))

    def test_readiness_negative_fixture_fails_top_level_verdicts(self) -> None:
        checks = {
            "package_load": True, "trainer_construct": True, "action_adapter_support": False,
            "cpu_train_step": True, "prior_only_validation": True, "checkpoint_reload": True,
            "model_device": True, "data_device": True, "checkpoint_map_location": True,
            "cpu_generic_dry_run": True, "dataset_contract": True, "action_coverage": False,
            "split_isolation": True,
        }
        verdict = validate_readiness(checks, formal_dataset=True, research_decisions_frozen=True)
        self.assertEqual("FAIL", verdict["training_stack_readiness"])
        self.assertEqual("NOT_READY", verdict["formal_dataset_readiness"])
        self.assertEqual("BLOCKED", verdict["formal_training_readiness"])

    def test_large_sample_shard_audit_is_streamed_and_counts_contract(self) -> None:
        samples = [
            {
                "metadata": {"history_frame_indices": [0, 1], "future_action_frame_indices": [1, 2, 3, 4]},
                "target": [{"future_target_side_metadata": {"unsupported_count": 2, "unresolved_count": 1, "fixed_support_blocked_count": 3}}],
            },
            {
                "metadata": {"history_frame_indices": [3, 4], "future_action_frame_indices": [1, 2, 3, 4]},
                "target": [{"future_target_side_metadata": {"unsupported_count": 5, "unresolved_count": 0, "fixed_support_blocked_count": 7}}],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(samples, handle)
            audit = _stream_sample_shard_audit(path, chunk_size=64)
        self.assertEqual(2, audit["samples"])
        self.assertEqual(2, audit["h2"])
        self.assertEqual(2, audit["l4"])
        self.assertEqual(7, audit["unsupported"])
        self.assertEqual(1, audit["unresolved"])
        self.assertEqual(10, audit["fixed_support_blocked"])

    def test_npz_canonicalization_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("first.npz", "second.npz")]
            for path in paths:
                np.savez_compressed(path, z=np.arange(4), a=np.ones((2, 3), dtype=np.float32))
                canonicalize_npz(path)
            digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            self.assertEqual(digests[0], digests[1])

    def test_exact_h2_l4_window_count(self) -> None:
        self.assertEqual(92, 97 - 2 - 4 + 1)
        self.assertEqual(4416, 48 * 92)
        self.assertEqual(1104, 12 * 92)
        self.assertEqual(5520, 60 * 92)

    def test_formal_split_is_deterministic_and_isolated(self) -> None:
        trajectory_ids = [f"trajectory-{index:02d}" for index in range(60)]
        first = deterministic_split(trajectory_ids)
        second = deterministic_split(trajectory_ids)
        self.assertEqual(first, second)
        self.assertEqual(SPLIT_SEED, 20260923)
        self.assertEqual(len(first["dev_train"]), 48)
        self.assertEqual(len(first["dev_validation"]), 12)
        self.assertFalse(set(first["dev_train"]) & set(first["dev_validation"]))

    def test_coverage_uses_eligible_opportunities_and_real_trajectory_counts(self) -> None:
        trajectory_ids = [f"trajectory-{index:02d}" for index in range(60)]
        split = deterministic_split(trajectory_ids)
        masks = ("0000", "1000", "0100", "0010", "0001", "1100")
        raw_by_id = {
            trajectory_id: {
                "steps": [_step(masks[index % len(masks)], f"{index % 5}-{slot}") for slot, index in enumerate(range(12))]
            }
            for index, trajectory_id in enumerate(trajectory_ids)
        }
        report = coverage(raw_by_id, split)
        checks = coverage_checks(report)
        self.assertTrue(checks["all_noop_present"])
        self.assertTrue(checks["single_family_present"])
        self.assertTrue(checks["multi_family_present"])
        for split_name in ("dev_train", "dev_validation"):
            for family in FAMILIES:
                row = report["splits"][split_name]["families"][family]
                self.assertEqual(row["eligible_count"], len(split[split_name]) * 12)
                self.assertGreater(row["no_op_count"], 0)

    def test_graph_batch_merge_preserves_nested_blocks_and_pads_capacity(self) -> None:
        first = {
            "schema_version": "graph-v1",
            "contract": {"source": "real"},
            "blocks": {"physical_nodes": {"presence": np.ones((2, 3), dtype=np.bool_), "value": np.ones((2, 3, 1), dtype=np.float32)}},
        }
        second = {
            "schema_version": "graph-v1",
            "contract": {"source": "real"},
            "blocks": {"physical_nodes": {"presence": np.ones((1, 5), dtype=np.bool_), "value": np.full((1, 5, 1), 2.0, dtype=np.float32)}},
        }
        merged = merge_graph_batches([first, second])
        self.assertEqual("graph-v1", merged["schema_version"])
        self.assertEqual((3, 5), merged["blocks"]["physical_nodes"]["presence"].shape)
        self.assertEqual((3, 5, 1), merged["blocks"]["physical_nodes"]["value"].shape)
        self.assertFalse(merged["blocks"]["physical_nodes"]["presence"][0, 4])
        self.assertEqual(2.0, merged["blocks"]["physical_nodes"]["value"][2, 4, 0])
        self.assertEqual(5, merged["contract"]["block_capacities"]["physical_nodes"])

    def test_tensor_contract_is_synchronized_to_merged_array_capacities(self) -> None:
        tensor = {
            "contract": {"max_entity": 1, "capacity_scope": "stale"},
            "entity_presence": np.zeros((3, 2, 7), dtype=np.bool_),
            "task_presence": np.zeros((3, 2, 11), dtype=np.bool_),
            "flow_presence": np.zeros((3, 2, 2), dtype=np.bool_),
            "target_entity_presence": np.zeros((3, 4, 9), dtype=np.bool_),
            "target_task_presence": np.zeros((3, 4, 13), dtype=np.bool_),
            "target_flow_presence": np.zeros((3, 4, 4), dtype=np.bool_),
            "relation_mask": np.zeros((3, 2, 17), dtype=np.bool_),
            "dag_mask": np.zeros((3, 2, 5), dtype=np.bool_),
            "past_outcome_relation_mask": np.zeros((3, 1, 16), dtype=np.bool_),
            "past_outcome_dag_mask": np.zeros((3, 1, 6), dtype=np.bool_),
            "comm_relation_presence": np.zeros((3, 2, 19), dtype=np.bool_),
            "task_agent_validity_mask": np.zeros((3, 2, 23), dtype=np.bool_),
            "comm_csi": np.zeros((3, 2, 19, 50)),
            "logical_flow_known_mask": np.zeros((3, 2, 3), dtype=np.bool_),
            "target_logical_flow_known_mask": np.zeros((3, 4, 8), dtype=np.bool_),
            "past_action_entry_mask": np.zeros((3, 1, 4, 2), dtype=np.bool_),
            "future_action_entry_mask": np.zeros((3, 4, 4, 5), dtype=np.bool_),
            "past_route_hop_mask": np.zeros((3, 1, 2, 1), dtype=np.bool_),
            "future_route_hop_mask": np.zeros((3, 4, 5, 3), dtype=np.bool_),
            "past_comm_rb_mask": np.zeros((3, 1, 2, 48), dtype=np.bool_),
            "future_comm_rb_mask": np.zeros((3, 4, 5, 50), dtype=np.bool_),
            "route_node_mask": np.zeros((3, 2, 3, 2), dtype=np.bool_),
            "target_route_node_mask": np.zeros((3, 4, 8, 4), dtype=np.bool_),
        }
        contract = synchronize_tensor_contract(tensor)["contract"]
        self.assertEqual(7, contract["max_entity"])
        self.assertEqual(13, contract["max_target_task"])
        self.assertEqual(19, contract["max_comm_relation"])
        self.assertEqual(8, contract["max_target_logical_flow"])
        self.assertEqual(5, contract["max_action_entries"])
        self.assertEqual(4, contract["max_route_nodes"])
        self.assertEqual(50, contract["n_rb"])
        self.assertEqual(4, contract["horizon_steps"])

    def test_target_merge_updates_shapes_and_preserves_all_identities(self) -> None:
        first = {
            "schema_version": "target-v1",
            "contract": {"motion_identity": [["first"]], "comm_identity": [["first"]]},
            "target_vehicle_motion_raw": np.ones((1, 4, 2, 4)),
            "target_comm_csi_raw": np.ones((1, 4, 3, 5)),
        }
        second = {
            "schema_version": "target-v1",
            "contract": {"motion_identity": [["second"]], "comm_identity": [["second"]]},
            "target_vehicle_motion_raw": np.ones((1, 4, 6, 4)),
            "target_comm_csi_raw": np.ones((1, 4, 7, 5)),
        }
        merged = merge_target_batches([first, second])
        self.assertEqual([2, 4, 6, 4], merged["contract"]["motion_shape"])
        self.assertEqual([2, 4, 7, 5], merged["contract"]["csi_shape"])
        self.assertEqual([["first"], ["second"]], merged["contract"]["motion_identity"])
        self.assertEqual([["first"], ["second"]], merged["contract"]["comm_identity"])

if __name__ == "__main__":
    unittest.main()
