"""CPU-only Formal Dataset v1 interface and H=4 trainer acceptance."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))

from build_step5_1d_unified_model_chain_v1 import build_action
from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig, Step52Trainer, Step52TrainingConfig
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, validate_readiness
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMConfig, StructuredRSSMWorldModel


def _adapter_fixture_checks(trainer: Step52Trainer) -> dict[str, bool]:
    """Exercise all four action adapters and distinguish pending Flow routing from transition."""
    device = trainer.device
    base_state = {key: value[:1].clone() for key, value in trainer.data.states[trainer.data.train_indices[0]].items()}
    base_graph = {key: value[:1].clone() for key, value in trainer.data.graphs[trainer.data.train_indices[0]].items()}
    physical = trainer.data.samples[trainer.data.train_indices[0]]["static"]["input_entity_index"]["physical"]
    task_ids = trainer.data.samples[trainer.data.train_indices[0]]["static"]["input_entity_index"]["task"]
    task_id, task_index = next(iter(task_ids.items()))
    active_physical = torch.nonzero(base_state["entity_presence"][0], as_tuple=False).flatten().tolist()
    if len(active_physical) < 2:
        raise ValueError("four-action fixture requires at least two current physical entities")
    source, destination = active_physical[:2]
    relation_hits = torch.nonzero(
        (base_state["comm_source_index"][0] == source)
        & (base_state["comm_target_index"][0] == destination)
        & base_state["comm_presence"][0]
        & base_state["comm_validity"][0], as_tuple=False,
    ).flatten().tolist()
    if not relation_hits:
        # Reuse a real current valid communication relation and set the fixture
        # endpoints to its source and destination.
        valid = torch.nonzero(base_state["comm_presence"][0] & base_state["comm_validity"][0], as_tuple=False).flatten().tolist()
        if not valid:
            raise ValueError("four-action fixture has no current valid communication relation")
        relation = valid[0]
        source = int(base_state["comm_source_index"][0, relation])
        destination = int(base_state["comm_target_index"][0, relation])
    else:
        relation = relation_hits[0]

    # Bind a fixture task to the causal physical source without changing the
    # packaged sample, its target, or its package identity.
    fixture_sample = copy.deepcopy(trainer.data.samples[trainer.data.train_indices[0]])
    base_state["flow_presence"].zero_()
    base_state["flow_task_index"].fill_(-1)
    base_state["flow_comm_relation_index"].fill_(-1)
    route = {"task_id": task_id, "task_index": int(task_index), "task_node_index": source, "target_node_index": destination, "route_node_indices": [destination], "route_kind": "offload"}
    empty = lambda: {"entries": [], "empty": True, "missing": False}
    action_frame = {"mobility": empty(), "comm": empty(), "comp": empty(), "route": {"entries": [route], "empty": False, "missing": False}}

    def sample_for(frame: dict[str, object]) -> dict[str, object]:
        value = copy.deepcopy(fixture_sample)
        value["future_action"] = [frame]
        return value

    # Route and Comm arriving before the post-action Flow ledger row.
    pending_frame = copy.deepcopy(action_frame)
    pending_frame["comm"] = {"entries": [{"task_id": task_id, "task_index": int(task_index), "rb_indices": [0]}], "empty": False, "missing": False}
    pending_action, pending_mapping = build_action(sample_for(pending_frame), base_state, 0)
    model = StructuredRSSMWorldModel(StructuredRSSMConfig(n_comm_rb=base_state["csi"].shape[-1])).to(device)
    model_state = {key: value.to(device) for key, value in base_state.items()}
    model_graph = {key: value.to(device) for key, value in base_graph.items()}
    # The actual Route/Comm adapter tensors are consumed by model action routing.
    routed = model.route_actions(pending_action, model_state, model_graph)
    pending_no_fake_flow = pending_mapping["route"][0]["flow_index"] == -1 and not bool((base_state["flow_presence"] & (base_state["flow_task_index"] == int(task_index))).any())
    pending_not_claimed_applied = not pending_mapping["route"][0]["transition_applied"] and not pending_mapping["comm"][0]["transition_applied"]
    pending_route_routed = bool(torch.count_nonzero(routed["task"])) > 0
    pending_comm_routed = bool(torch.count_nonzero(routed["communication"])) > 0
    transition_action = {key: value.to(device) for key, value in pending_action.items()}
    learned_pending = {"vehicle_motion": torch.zeros((*model_state["position"].shape[:-1], 4), device=device), "csi": model_state["csi"].clone()}
    transitioned, _ = model.deterministic_transition(model_state, transition_action, learned_pending, graph=model_graph, service_mode="expectation", generator=None)
    pending_transition_preserves_no_flow_identity = torch.equal(transitioned["flow_identity_index"], model_state["flow_identity_index"])

    # Current Flow route uses existing Flow slot and reaches carrying rule.
    current_state = {key: value.clone() for key, value in base_state.items()}
    slot = int(torch.nonzero(current_state["task_presence"][0], as_tuple=False)[0]) if bool(current_state["task_presence"].any()) else int(task_index)
    current_state["flow_presence"][0, 0] = True
    current_state["flow_task_index"][0, 0] = slot
    current_state["flow_comm_relation_index"][0, 0] = relation
    current_state["carrying_hop_source_index"][0, 0] = source
    current_state["carrying_hop_destination_index"][0, 0] = destination
    current_state["flow_identity_index"][0, 0] = 12345
    current_frame = copy.deepcopy(action_frame)
    current_frame["route"]["entries"][0]["task_index"] = slot
    current_frame["route"]["entries"][0]["task_id"] = next(key for key, value in task_ids.items() if int(value) == slot)
    current_frame["comm"] = {"entries": [{"task_id": current_frame["route"]["entries"][0]["task_id"], "task_index": slot, "rb_indices": [0]}], "empty": False, "missing": False}
    current_sample = sample_for(current_frame)
    current_sample["static"]["input_entity_index"]["task"] = {current_frame["route"]["entries"][0]["task_id"]: slot}
    current_action, current_mapping = build_action(current_sample, current_state, 0)
    current_routed = model.route_actions(current_action, current_state, model_graph)
    current_route_applied = current_mapping["route"][0]["flow_index"] == 0 and current_mapping["route"][0]["transition_applied"]
    current_comm_applied = current_mapping["comm"][0]["mode"] == "current_flow" and current_mapping["comm"][0]["transition_applied"]

    # A non-empty Comp row routes to both its Agent and Task tensors and runs
    # through the same deterministic CPU/task transition function.
    comp_frame = {"mobility": empty(), "comm": empty(), "route": empty(), "comp": {"entries": [{"task_id": task_id, "node_id": next(key for key, value in physical.items() if int(value) == source), "allocated_cpu_per_s": 1.0}], "empty": False, "missing": False}}
    comp_action, _ = build_action(sample_for(comp_frame), base_state, 0)
    comp_routed = model.route_actions(comp_action, model_state, model_graph)
    cpu_state, comp_trace = model.deterministic_transition(model_state, comp_action, learned_pending, graph=model_graph, service_mode="expectation", generator=None)

    # Existing Route/Comp empty action remains the explicit absent sentinel.
    noop_action, noop_mapping = build_action(sample_for({"mobility": empty(), "comm": empty(), "route": empty(), "comp": empty()}), base_state, 0)
    tensors_on_device = all(value.device == device for value in (*pending_action.values(), *current_action.values(), *comp_action.values(), *noop_action.values()))
    return {
        "pending_route_task_tensor": int(pending_action["route_task_index"][0, 0]) == int(task_index),
        "pending_route_flow_sentinel": pending_no_fake_flow,
        "pending_route_routed_not_claimed_transitioned": pending_route_routed and pending_not_claimed_applied and pending_transition_preserves_no_flow_identity,
        "pending_comm_relation_routed": pending_comm_routed and pending_mapping["comm"][0]["relation_index"] == relation,
        "current_flow_route_comm_routed": current_route_applied and current_comm_applied and bool(torch.count_nonzero(current_routed["flow"])) and bool(torch.count_nonzero(current_routed["communication"])),
        "comp_agent_task_routed_cpu_transition": bool(torch.count_nonzero(comp_routed["agent"])) and bool(torch.count_nonzero(comp_routed["task"])) and bool(torch.isfinite(cpu_state["task_work_remaining"]).all()) and bool(torch.isfinite(comp_trace["cpu_service"]).all()),
        "route_comp_noop_regression": bool(noop_mapping["route_explicit_noop"] and noop_mapping["comp_explicit_noop"]) and int(noop_action["route_task_index"][0, 0]) == -1 and int(noop_action["route_flow_index"][0, 0]) == -1,
        "action_tensors_on_device": tensors_on_device,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rejects_checkpoint(trainer: Step52Trainer, path: Path) -> bool:
    try:
        trainer.load_checkpoint(path)
    except ValueError:
        return True
    return False


def run(manifest_path: Path, output_path: Path) -> dict[str, object]:
    interface = FormalTrainingInterface.from_manifest(manifest_path)
    packages = interface.verify_packages()
    runtime_packages = interface.verify_runtime_packages()
    dataset_receipt = json.loads((manifest_path.parent / "dataset_acceptance_receipt.json").read_text(encoding="utf-8"))
    coverage = json.loads((manifest_path.parent / "action_coverage_audit.json").read_text(encoding="utf-8"))
    config = Step52TrainingConfig(
        seed=5501,
        batch_size=1,
        max_steps=1,
        max_epochs=1,
        stage1_steps=0,
        max_horizon=4,
        curriculum=CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1, 2)),
        device="cpu",
    )
    trainer = Step52Trainer.from_formal_interface(interface, config)
    optimizer_parameter_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
    trainable_parameter_ids = {id(parameter) for parameter in trainer.parameters() if parameter.requires_grad}
    adapter_checks = _adapter_fixture_checks(trainer)
    train_row = trainer.train_step(global_step=2, epoch=0)
    validation = trainer.validate()

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        checkpoint = temporary_path / "step5_5_cpu_h4.pt"
        trainer.save_checkpoint(checkpoint, state=trainer.state_snapshot(global_step=3, curriculum_horizon=4))
        restored = Step52Trainer.from_formal_interface(interface, config)
        restored.load_checkpoint(checkpoint)
        compatible_digest = restored.deterministic_forward_digest() == trainer.deterministic_forward_digest()

        wrong_identity = torch.load(checkpoint, map_location="cpu", weights_only=False)
        wrong_identity["data_identity"] = {"tampered": True}
        wrong_identity_path = temporary_path / "wrong_identity.pt"
        torch.save(wrong_identity, wrong_identity_path)

        wrong_config = copy.deepcopy(torch.load(checkpoint, map_location="cpu", weights_only=False))
        wrong_config["architecture_identity"]["rssm"]["d_h"] += 1
        wrong_config_path = temporary_path / "wrong_config.pt"
        torch.save(wrong_config, wrong_config_path)
        wrong_identity_rejected = _rejects_checkpoint(restored, wrong_identity_path)
        wrong_config_rejected = _rejects_checkpoint(restored, wrong_config_path)

    checks = {
        "package_load": bool(packages["all_present_and_matching"] and runtime_packages["all_present_and_matching"]),
        "trainer_construct": bool(optimizer_parameter_ids == trainable_parameter_ids),
        "action_adapter_support": all(adapter_checks.values()),
        "cpu_train_step": bool(train_row["loss_finite"] and train_row["gradients_finite"] and train_row["parameter_update"]["any_changed"]),
        "prior_only_validation": bool(validation["prior_only_rollout"] and validation["validation_no_parameter_update"] and validation["future_posterior_teacher_calls"] == 0 and validation["future_target_encoder_calls"] == 0),
        "checkpoint_reload": bool(compatible_digest and wrong_identity_rejected and wrong_config_rejected),
        "model_device": all(parameter.device.type == "cpu" for parameter in trainer.parameters()),
        "data_device": all(value.device.type == "cpu" for value in trainer.data.target_tensors.values() if isinstance(value, torch.Tensor)),
        "checkpoint_map_location": "map_location=self.device" in (Path(__file__).resolve().parents[1] / "src/pi_jwm/step5_2_training_loop_v1.py").read_text(encoding="utf-8"),
        "cpu_generic_dry_run": bool(train_row["loss_finite"] and validation["l_val_finite"] and config.device == "cpu"),
        "dataset_contract": bool(dataset_receipt.get("passed") and all(dataset_receipt.get("checks", {}).values())),
        "action_coverage": bool(coverage.get("passed")),
        "split_isolation": bool(set(interface.train_indices).isdisjoint(interface.validation_indices)),
    }
    readiness = validate_readiness(checks, formal_dataset=True, research_decisions_frozen=False)
    receipt: dict[str, object] = {
        "schema_version": "PI-JWM-Step-5.5-Formal-Dataset-CPU-Acceptance-v1",
        "passed": bool(
            readiness["training_stack_readiness"] == "PASS"
            and readiness["formal_dataset_readiness"] == "READY"
            and readiness["gpu_codepath_readiness"] == "PREPARED"
            and readiness["formal_training_readiness"] == "BLOCKED"
        ),
        "checks": checks,
        "action_adapter_checks": adapter_checks,
        "runtime": {
            "device": "cpu",
            "optimizer_step_count": 1,
            "train_rollout_horizon": int(train_row["rollout_horizon"]),
            "train_stage": train_row["stage"],
            "train_loss_finite": bool(train_row["loss_finite"]),
            "prior_only_validation_horizons": [int(row["horizon"]) for row in validation["per_horizon"]],
            "l_gt_2_runtime_verified": int(train_row["rollout_horizon"]) == 4 and [int(row["horizon"]) for row in validation["per_horizon"]] == [1, 2, 3, 4],
            "optimizer_parameter_identity_exact": optimizer_parameter_ids == trainable_parameter_ids,
            "checkpoint_compatible_reload": compatible_digest,
            "wrong_dataset_identity_rejected": wrong_identity_rejected,
            "wrong_architecture_config_rejected": wrong_config_rejected,
        },
        "readiness": readiness,
        "package_verification": packages,
        "runtime_package_verification": runtime_packages,
        "scope": {
            "gpu": False,
            "formal_training": False,
            "locked_test_accessed": False,
            "baseline": False,
            "planner": False,
            "performance_claim": False,
        },
        "formal_training_blocker": "STEP 5.6A formal training configuration and GPU smoke are not frozen or executed",
    }
    receipt["passed"] = bool(receipt["passed"] and receipt["runtime"]["l_gt_2_runtime_verified"] and receipt["runtime"]["optimizer_parameter_identity_exact"])
    _write_json(output_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    receipt = run(args.manifest.resolve(), args.output.resolve())
    print(json.dumps({"passed": receipt["passed"], "readiness": receipt["readiness"]}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
