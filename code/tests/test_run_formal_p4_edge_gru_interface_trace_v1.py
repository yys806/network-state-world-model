from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def _allocations(mode: str) -> list[dict[str, int]]:
    rows = [{"incoming": 0, "downcross": 0, "tp": 0, "fp": 0, "tn": 1} for _ in range(20)]
    if mode == "incoming":
        rows[0].update(incoming=5836, tp=1902, fp=557)
        for row in rows[1:]:
            row["incoming"] = 1
    elif mode == "downcross":
        rows[0].update(downcross=5836, tp=1902, fp=557)
        for row in rows[1:]:
            row["downcross"] = 1
    elif mode == "mixed":
        rows[0].update(incoming=5836, tp=1902, fp=557)
        for row in rows[1:19]:
            row["incoming"] = 1
        rows[19]["downcross"] = 1
    elif mode == "exact_080":
        rows[0].update(incoming=80, downcross=20, tp=1902, fp=557)
        rows[1].update(incoming=4524, downcross=1131)
        rows[19].update(incoming=80, downcross=20)
    elif mode == "h20_no_fn":
        rows[0].update(incoming=5855, tp=1901, fp=557)
        rows[19]["tp"] = 1
    else:
        raise ValueError(mode)
    return rows


def _frozen_rows(mode: str = "incoming") -> tuple[list[torch.Tensor], ...]:
    pre, post, official, gates, labels, prior = [], [], [], [], [], []
    for allocation in _allocations(mode):
        incoming = allocation["incoming"]
        downcross = allocation["downcross"]
        tp = allocation["tp"]
        fp = allocation["fp"]
        tn = allocation["tn"]
        pre_row = torch.tensor(
            [-3.0] * incoming + [3.0] * downcross + [3.0] * tp + [3.0] * fp + [-3.0] * tn,
            dtype=torch.float32,
        )
        post_row = torch.tensor(
            [-3.0] * incoming + [-3.0] * downcross + [3.0] * tp + [3.0] * fp + [-3.0] * tn,
            dtype=torch.float32,
        )
        label_row = torch.tensor(
            [1] * (incoming + downcross + tp) + [0] * (fp + tn), dtype=torch.int64
        )
        prior_row = torch.tensor(
            [1, 0, -1] * ((label_row.numel() + 2) // 3), dtype=torch.int64
        )[: label_row.numel()]
        pre.append(pre_row)
        post.append(post_row)
        official.append(post_row.clone())
        gates.append(torch.full_like(pre_row, 0.2))
        labels.append(label_row)
        prior.append(prior_row)
    return pre, post, official, gates, labels, prior


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TraceModel(torch.nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.edge_transition = torch.nn.GRUCell(width, width)
        self.other_transition = torch.nn.GRUCell(width, width)
        self.link_activity_head = torch.nn.Linear(width, 1)
        with torch.no_grad():
            self.link_activity_head.weight.copy_(torch.tensor([[0.5, -0.25, 0.75]]))
            self.link_activity_head.bias.fill_(0.1)


def _batch(batch_id: int) -> dict[str, object]:
    target_activity = torch.zeros((1, 20, 3), dtype=torch.int64)
    target_activity[:, :, 0] = 1
    target_activity[:, 1::2, 1] = 1
    target_mask = torch.zeros((1, 20, 3), dtype=torch.bool)
    if batch_id == 0:
        target_mask[:, 0::2, :2] = True
        target_mask[:, 1::2, 1:] = True
    else:
        target_mask[:, 0::2, 1:] = True
        target_mask[:, 1::2, :2] = True
    return {
        "batch_id": batch_id,
        "history": {
            "aggregate_link_activity": torch.tensor([[[1, 0, 1], [0, 1, 0]]], dtype=torch.int64),
            "aggregate_link_activity_mask": torch.ones((1, 2, 3), dtype=torch.bool),
        },
        "target": {
            "aggregate_link_activity": target_activity,
            "aggregate_link_activity_mask": target_mask,
        },
        "static": {
            "physical_edge_endpoint_index": torch.tensor([[[0, 1], [1, 0], [1, 2]]], dtype=torch.int64)
        },
    }


def _prediction_factory(*, calls: int = 20, official_drift: float = 0.0):
    def fake_prediction(method, model, batch, stats):
        del stats
        if method != "coupled_dual_gnn_residual":
            raise AssertionError(method)
        batch_size = 1
        edges = batch["static"]["physical_edge_endpoint_index"].shape[1]
        width = 3
        hidden = torch.full((batch_size * edges, width), float(batch["batch_id"]) / 10.0)
        logits = []
        for horizon in range(calls):
            message = torch.full_like(hidden, float(horizon + 1) / 20.0)
            hidden = model.edge_transition(message, hidden)
            logits.append(model.link_activity_head(hidden.reshape(batch_size, edges, width)).squeeze(-1))
        while len(logits) < 20:
            logits.append(logits[-1].clone())
        official = torch.stack(logits[:20], dim=1)
        return {"link_activity_logits": official + official_drift}
    return fake_prediction


def _collector_result(*, hook_module_full_name: str | None = None) -> tuple[object, ...]:
    from run_formal_p4_edge_gru_interface_trace_v1 import FROZEN_HOOK_MODULE_FULL_NAME

    vectors = [[torch.zeros(1) for _ in range(20)] for _ in range(6)]
    counts = {**{f"h{index}": 1 for index in range(1, 21)}, "total": 20}
    fingerprints = {f"h{index}": "a" * 64 for index in range(1, 21)}
    return (*vectors, counts, fingerprints, 1280, 0.0, 0.0, hook_module_full_name or FROZEN_HOOK_MODULE_FULL_NAME)


def _runner_provenance(root: Path) -> dict[str, object]:
    from run_formal_p4_edge_gru_interface_trace_v1 import (
        CANONICAL_TENSOR_MANIFEST_SHA256,
        FROZEN_SENTINEL_CHECKPOINT_SHA256,
        FROZEN_SENTINEL_SAMPLE_IDS_SHA256,
    )

    paths = {
        "checkpoint_path": root / "run" / "checkpoints" / "coupled_dual_gnn_residual__best.pt",
        "sample_ids_path": root / "run" / "sample_ids.json",
        "class_weights_path": root / "run" / "class_weights.json",
        "tensor_manifest_path": root / "tensor" / "manifest.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    return {
        "config_sha256": "d" * 64,
        "checkpoint_sha256": FROZEN_SENTINEL_CHECKPOINT_SHA256,
        "sample_ids_sha256": FROZEN_SENTINEL_SAMPLE_IDS_SHA256,
        "class_weights_sha256": "e" * 64,
        "tensor_manifest_sha256": CANONICAL_TENSOR_MANIFEST_SHA256,
        **{name: str(path.resolve()) for name, path in paths.items()},
        "selected_validation_tensor_inputs": [
            {"relative_path": "seed_001/trajectory_tensors.npz", "size_bytes": 7, "sha256": "f" * 64}
        ],
        "class_weights_source_split": "train",
        "link_activity_pos_weight": 50.0,
        "checkpoint_reload": {"strict": True, "missing_keys": 0, "unexpected_keys": 0},
        "gpu_execution": False,
        "locked_test_accessed": False,
        "formal_performance_claim_ready": False,
    }


class EdgeGruHookTests(unittest.TestCase):
    def test_reconstructs_real_gru_cell_in_float32_and_float64(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import _reconstruct_gru_cell

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(17)
                cell = torch.nn.GRUCell(4, 3).to(dtype=dtype)
                message = torch.randn(5, 4, dtype=dtype)
                hidden = torch.randn(5, 3, dtype=dtype)
                actual = cell(message, hidden)
                reconstructed, update = _reconstruct_gru_cell(cell, message, hidden)
                self.assertLessEqual(float((reconstructed - actual).abs().max().detach().item()), 1e-6)
                self.assertEqual(update.shape, hidden.shape)

    def test_hook_is_readonly_edge_only_and_removed_after_normal_exit(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import edge_gru_readonly_trace

        torch.manual_seed(23)
        model = _TraceModel()
        message, hidden = torch.randn(4, 3), torch.randn(4, 3)
        expected = model.edge_transition(message, hidden).detach().clone()
        with edge_gru_readonly_trace(model) as audit:
            model.other_transition(message, hidden)
            actual = model.edge_transition(message, hidden)
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(audit["actual_hook_calls"], 1)
        self.assertTrue(audit["hook_removed"])
        self.assertFalse(model.edge_transition._forward_hooks)
        records_after_exit = len(audit["records"])
        model.edge_transition(message, hidden)
        self.assertEqual(len(audit["records"]), records_after_exit)

    def test_hook_is_removed_after_exception(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import edge_gru_readonly_trace

        model = _TraceModel()
        message, hidden = torch.randn(2, 3), torch.randn(2, 3)
        audit = None
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with edge_gru_readonly_trace(model) as audit:
                model.edge_transition(message, hidden)
                raise RuntimeError("forced")
        self.assertIsNotNone(audit)
        self.assertTrue(audit["hook_removed"])
        count = audit["actual_hook_calls"]
        model.edge_transition(message, hidden)
        self.assertEqual(audit["actual_hook_calls"], count)
        self.assertFalse(model.edge_transition._forward_hooks)


class EdgeGruStatisticsTests(unittest.TestCase):
    def test_fixed_counts_hit_all_three_registered_states(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import build_edge_gru_interface_trace

        expected = {
            "incoming": "incoming_readout_below_threshold_dominant",
            "downcross": "transition_downcrossing_dominant",
            "mixed": "mixed_no_single_dominant_pattern",
        }
        for mode, state in expected.items():
            with self.subTest(mode=mode):
                report = build_edge_gru_interface_trace(*_frozen_rows(mode), {"checkpoint_sha256": "a" * 64})
                self.assertEqual(report["candidate"], {"tp": 1902, "fp": 557, "fn": 5855})
                self.assertEqual(report["edge_gru_interface_trace"], state)

    def test_exact_point_eight_is_inclusive_and_fractions_use_fn_denominator(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import build_edge_gru_interface_trace

        report = build_edge_gru_interface_trace(*_frozen_rows("exact_080"), {"checkpoint_sha256": "a" * 64})
        self.assertEqual(report["edge_gru_interface_trace"], "incoming_readout_below_threshold_dominant")
        bucket = report["overall"]["all_positive"]
        self.assertEqual(bucket["false_negative_count"], 5855)
        self.assertEqual(bucket["incoming_below_count"] + bucket["downcross_count"], 5855)
        self.assertTrue(math.isclose(bucket["incoming_below_fraction"], 0.8))
        self.assertTrue(math.isclose(bucket["downcross_fraction"], 0.2))
        for horizon in ("h1", "h20"):
            self.assertTrue(math.isclose(report["by_horizon"][horizon]["all_positive"]["incoming_below_fraction"], 0.8))

    def test_four_transition_counts_are_mutually_exclusive_and_sum_to_positives(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import build_edge_gru_interface_trace

        rows = list(_frozen_rows("incoming"))
        pre = rows[0]
        first_tp = 5836
        pre[0] = pre[0].clone()
        pre[0][0] = 3.0
        pre[0][first_tp] = -3.0
        report = build_edge_gru_interface_trace(*rows, {"checkpoint_sha256": "a" * 64})
        bucket = report["by_horizon"]["h1"]["all_positive"]
        flow_sum = sum(bucket[name] for name in (
            "pre_positive_post_positive",
            "pre_positive_post_negative",
            "pre_negative_post_positive",
            "pre_negative_post_negative",
        ))
        self.assertEqual(flow_sum, bucket["positive_count"])
        self.assertGreater(bucket["pre_positive_post_positive"], 0)
        self.assertGreater(bucket["pre_positive_post_negative"], 0)
        self.assertGreater(bucket["pre_negative_post_positive"], 0)
        self.assertGreater(bucket["pre_negative_post_negative"], 0)
        self.assertEqual(bucket["incoming_below_count"] + bucket["downcross_count"], bucket["false_negative_count"])

    def test_float32_nextafter_threshold_uses_original_dtype(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import build_edge_gru_interface_trace

        rows = list(_frozen_rows("incoming"))
        pre, post, official, _, _, prior = rows
        threshold_logit = torch.logit(torch.tensor(0.9, dtype=torch.float32))
        values = torch.stack((
            torch.nextafter(threshold_logit, torch.tensor(float("-inf"), dtype=torch.float32)),
            threshold_logit,
            torch.nextafter(threshold_logit, torch.tensor(float("inf"), dtype=torch.float32)),
        ))
        first_tp = 5836
        boundary = slice(first_tp, first_tp + 3)
        pre[0] = pre[0].clone(); post[0] = post[0].clone(); official[0] = official[0].clone(); prior[0] = torch.ones_like(prior[0])
        pre[0][boundary] = values; post[0][boundary] = values; official[0][boundary] = values; prior[0][boundary] = 0
        expected_positive = int((torch.sigmoid(values) >= 0.9).sum().item())
        for index in range(3 - expected_positive):
            pre[0][index] = post[0][index] = official[0][index] = 3.0
        report = build_edge_gru_interface_trace(*rows, {"checkpoint_sha256": "a" * 64})
        bucket = report["by_horizon"]["h1"]["newly_active"]
        self.assertEqual(bucket["pre_positive_post_positive"], expected_positive)
        self.assertEqual(bucket["pre_negative_post_negative"], 3 - expected_positive)

    def test_empty_group_and_no_fn_are_not_computable(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import build_edge_gru_interface_trace

        rows = list(_frozen_rows("h20_no_fn"))
        rows[5] = [torch.ones_like(row) for row in rows[5]]
        report = build_edge_gru_interface_trace(*rows, {"checkpoint_sha256": "a" * 64})
        self.assertEqual(report["by_horizon"]["h20"]["all_positive"]["incoming_below_fraction"], "not_computable")
        self.assertEqual(report["by_horizon"]["h20"]["newly_active"], "not_computable")
        self.assertEqual(report["edge_gru_interface_trace"], "mixed_no_single_dominant_pattern")

    def test_dominance_rejects_nonfinite_nonreal_and_bool_fractions(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import _fraction_at_least

        for value in ("not_computable", True, False, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertFalse(_fraction_at_least(value, 0.8))
        self.assertTrue(_fraction_at_least(0.8, 0.8))

    def test_build_rejects_frozen_counts_and_official_drift(self):
        from run_formal_p4_edge_gru_interface_trace_v1 import build_edge_gru_interface_trace

        rows = list(_frozen_rows("incoming"))
        rows[2][0] = rows[2][0].clone()
        rows[2][0][0] += 1e-3
        with self.assertRaisesRegex(ValueError, "post.*official"):
            build_edge_gru_interface_trace(*rows, {"checkpoint_sha256": "a" * 64})
        rows = list(_frozen_rows("incoming"))
        rows[1][0] = rows[1][0].clone(); rows[2][0] = rows[2][0].clone()
        rows[1][0][0] = rows[2][0][0] = 3.0
        with self.assertRaisesRegex(ValueError, "1902/557/5855"):
            build_edge_gru_interface_trace(*rows, {"checkpoint_sha256": "a" * 64})


class EdgeGruCollectorTests(unittest.TestCase):
    def test_two_real_shape_batches_align_hidden_previous_masks_and_official_output(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        torch.manual_seed(31)
        model = _TraceModel()
        batches = [_batch(0), _batch(1)]
        with patch.object(module, "_prediction", side_effect=_prediction_factory()):
            result = module._collect_edge_gru_trace(model, batches)
        pre, post, official, gates, labels, previous, counts, fingerprints, calls, post_error, gru_error, hook_module_full_name = result
        self.assertEqual(len(pre), 20)
        self.assertEqual(pre[0].shape, torch.Size([4]))
        self.assertEqual(gates[0].shape, torch.Size([4]))
        self.assertEqual(counts, {**{f"h{index}": 4 for index in range(1, 21)}, "total": 80})
        self.assertEqual(calls, 40)
        self.assertEqual(previous[0].tolist(), [0, 1, 1, 0])
        self.assertEqual(previous[1].tolist(), [0, -1, -1, 0])
        self.assertNotEqual(fingerprints["h1"], fingerprints["h2"])
        self.assertLessEqual(post_error, 1e-6)
        self.assertLessEqual(gru_error, 1e-6)
        self.assertEqual(hook_module_full_name, f"{type(model).__module__}.{type(model).__qualname__}.edge_transition")
        for horizon in range(20):
            self.assertTrue(torch.equal(post[horizon], official[horizon]))
        self.assertFalse(model.edge_transition._forward_hooks)

    def test_collector_rejects_hook_call_count_drift(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        with patch.object(module, "_prediction", side_effect=_prediction_factory(calls=19)):
            with self.assertRaisesRegex(ValueError, "exactly 20"):
                module._collect_edge_gru_trace(_TraceModel(), [_batch(0)])

    def test_collector_rejects_post_official_drift(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        with patch.object(module, "_prediction", side_effect=_prediction_factory(official_drift=1e-3)):
            with self.assertRaisesRegex(ValueError, "numerical drift"):
                module._collect_edge_gru_trace(_TraceModel(), [_batch(0)])

    def test_collector_rejects_gru_reconstruction_drift(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        original = module._reconstruct_gru_cell
        def wrong_reconstruction(cell, message, hidden):
            output, gate = original(cell, message, hidden)
            return output + 1e-3, gate
        with patch.object(module, "_prediction", side_effect=_prediction_factory()), patch.object(module, "_reconstruct_gru_cell", side_effect=wrong_reconstruction):
            with self.assertRaisesRegex(ValueError, "numerical drift"):
                module._collect_edge_gru_trace(_TraceModel(), [_batch(0)])

    def test_collector_returns_actual_full_hook_module_identity(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        model = _TraceModel()
        expected = f"{type(model).__module__}.{type(model).__qualname__}.edge_transition"
        with patch.object(module, "_prediction", side_effect=_prediction_factory()):
            result = module._collect_edge_gru_trace(model, [_batch(0)])
        self.assertEqual(result[-1], expected)


class EdgeGruRunnerAndPublishTests(unittest.TestCase):
    def _make_roots(self, temporary: str) -> tuple[Path, Path, Path]:
        root = Path(temporary)
        run_root, tensor_root = root / "run", root / "tensor"
        run_root.mkdir(parents=True)
        tensor_root.mkdir(parents=True)
        (run_root / "config.json").write_text("{}", encoding="utf-8")
        (tensor_root / "tensor_contract.json").write_text(
            json.dumps({"history_steps": 8, "horizon_steps": 20}), encoding="utf-8"
        )
        return root, run_root, tensor_root

    def test_runner_freezes_prepare_inputs_and_publishes_complete_provenance(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        with tempfile.TemporaryDirectory() as temporary:
            root, run_root, tensor_root = self._make_roots(temporary)
            provenance = _runner_provenance(root)
            captured = {}
            def fake_build(*args):
                captured["provenance"] = args[-1]
                return {
                    "schema_version": module.SCHEMA_VERSION,
                    "edge_gru_interface_trace": "mixed_no_single_dominant_pattern",
                    "gpu_execution": False,
                    "locked_test_accessed": False,
                    "formal_performance_claim_ready": False,
                    "p4_status": "blocked",
                    "provenance": args[-1],
                }
            loader = [object()] * 64
            output = root / "output"
            with patch.object(module, "_prepare_frozen_link_validation", return_value=(_TraceModel(), loader, provenance)) as prepare, patch.object(module, "_collect_edge_gru_trace", return_value=_collector_result()), patch.object(module, "build_edge_gru_interface_trace", side_effect=fake_build):
                report = module.run_formal_p4_edge_gru_interface_trace(
                    run_root=run_root,
                    tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual",
                    output_dir=output,
                )
            prepare.assert_called_once_with(
                run_root,
                tensor_root,
                "coupled_dual_gnn_residual",
                module.CANONICAL_TENSOR_MANIFEST_SHA256,
                module.FROZEN_SENTINEL_CHECKPOINT_SHA256,
                module.FROZEN_SENTINEL_SAMPLE_IDS_SHA256,
            )
            self.assertEqual(report["p4_status"], "blocked")
            trace_provenance = captured["provenance"]
            self.assertEqual(trace_provenance["hook_calls"], {"actual": 1280, "expected": 1280})
            self.assertEqual(trace_provenance["config_sha256"], provenance["config_sha256"])
            for name, entry in trace_provenance["input_files"].items():
                with self.subTest(name=name):
                    self.assertTrue(Path(entry["path"]).is_absolute())
                    self.assertEqual(len(entry["sha256"]), 64)
                    self.assertTrue(all(character in "0123456789abcdef" for character in entry["sha256"].lower()))
            self.assertEqual(trace_provenance["input_files"]["config"]["path"], str((run_root / "config.json").resolve()))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            entry = manifest["files"]["edge_gru_interface_trace.json"]
            self.assertEqual(entry["size_bytes"], (output / "edge_gru_interface_trace.json").stat().st_size)
            self.assertEqual(entry["sha256"], _sha256(output / "edge_gru_interface_trace.json"))
            for key, value in (
                ("gpu_execution", False),
                ("locked_test_accessed", False),
                ("formal_performance_claim_ready", False),
                ("p4_status", "blocked"),
            ):
                self.assertEqual(manifest[key], value)

    def test_noncanonical_sha_overrides_fail_before_prepare(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        override_names = (
            "_expected_tensor_manifest_sha256",
            "_expected_checkpoint_sha256",
            "_expected_sample_ids_sha256",
        )
        for name in override_names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root, run_root, tensor_root = self._make_roots(temporary)
                with patch.object(module, "_prepare_frozen_link_validation") as prepare:
                    with self.assertRaises((TypeError, ValueError)):
                        module.run_formal_p4_edge_gru_interface_trace(
                            run_root=run_root,
                            tensor_root=tensor_root,
                            method="coupled_dual_gnn_residual",
                            output_dir=root / "output",
                            **{name: "0" * 64},
                        )
                prepare.assert_not_called()

    def test_runner_propagates_and_locks_full_hook_module_identity(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        with tempfile.TemporaryDirectory() as temporary:
            root, run_root, tensor_root = self._make_roots(temporary)
            captured = {}
            def fake_build(*args):
                captured["provenance"] = args[-1]
                return {
                    "schema_version": module.SCHEMA_VERSION,
                    "gpu_execution": False,
                    "locked_test_accessed": False,
                    "formal_performance_claim_ready": False,
                    "p4_status": "blocked",
                }
            valid_collector = _collector_result()
            with patch.object(module, "_prepare_frozen_link_validation", return_value=(_TraceModel(), [object()] * 64, _runner_provenance(root))), patch.object(module, "_collect_edge_gru_trace", return_value=valid_collector), patch.object(module, "build_edge_gru_interface_trace", side_effect=fake_build), patch.object(module, "_publish"):
                module.run_formal_p4_edge_gru_interface_trace(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "output")
            self.assertEqual(captured["provenance"]["hook_module"], "edge_transition")
            self.assertEqual(captured["provenance"]["hook_module_full_name"], module.FROZEN_HOOK_MODULE_FULL_NAME)

            drifted_collector = _collector_result(hook_module_full_name="example.OtherModel.edge_transition")
            with patch.object(module, "_prepare_frozen_link_validation", return_value=(_TraceModel(), [object()] * 64, _runner_provenance(root))), patch.object(module, "_collect_edge_gru_trace", return_value=drifted_collector), patch.object(module, "build_edge_gru_interface_trace"), patch.object(module, "_publish"):
                with self.assertRaisesRegex(ValueError, "hook module identity"):
                    module.run_formal_p4_edge_gru_interface_trace(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "output")

    def test_runner_rejects_every_frozen_contract_drift_before_publish(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        cases = {
            "checkpoint_sha": {"checkpoint_sha256": "0" * 64},
            "sample_ids_sha": {"sample_ids_sha256": "0" * 64},
            "tensor_manifest_sha": {"tensor_manifest_sha256": "0" * 64},
            "strict_reload": {"checkpoint_reload": {"strict": False, "missing_keys": 0, "unexpected_keys": 0}},
            "missing_key": {"checkpoint_reload": {"strict": True, "missing_keys": 1, "unexpected_keys": 0}},
            "unexpected_key": {"checkpoint_reload": {"strict": True, "missing_keys": 0, "unexpected_keys": 1}},
            "gpu": {"gpu_execution": True},
            "locked": {"locked_test_accessed": True},
            "formal": {"formal_performance_claim_ready": True},
            "selected_size": {"selected_validation_tensor_inputs": [{"relative_path": "a", "size_bytes": 0, "sha256": "f" * 64}]},
            "selected_sha": {"selected_validation_tensor_inputs": [{"relative_path": "a", "size_bytes": 1, "sha256": "z" * 64}]},
            "selected_path": {"selected_validation_tensor_inputs": [{"relative_path": "", "size_bytes": 1, "sha256": "f" * 64}]},
            "config_sha": {"config_sha256": "not-a-sha"},
            "relative_checkpoint": {"checkpoint_path": "checkpoint.pt"},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root, run_root, tensor_root = self._make_roots(temporary)
                provenance = _runner_provenance(root)
                provenance.update(overrides)
                output = root / "output"
                with patch.object(module, "_prepare_frozen_link_validation", return_value=(_TraceModel(), [object()] * 64, provenance)), patch.object(module, "_collect_edge_gru_trace", return_value=_collector_result()), patch.object(module, "build_edge_gru_interface_trace"):
                    with self.assertRaises(ValueError):
                        module.run_formal_p4_edge_gru_interface_trace(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=output)
                self.assertFalse(output.exists())

    def test_runner_rejects_loader_hook_error_and_tensor_contract_drift(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        cases = (
            ("loader", 63, _collector_result(), {"history_steps": 8, "horizon_steps": 20}),
            ("calls", 64, (*_collector_result()[:-4], 1279, 0.0, 0.0, module.FROZEN_HOOK_MODULE_FULL_NAME), {"history_steps": 8, "horizon_steps": 20}),
            ("post", 64, (*_collector_result()[:-4], 1280, 1.1e-6, 0.0, module.FROZEN_HOOK_MODULE_FULL_NAME), {"history_steps": 8, "horizon_steps": 20}),
            ("gru", 64, (*_collector_result()[:-4], 1280, 0.0, 1.1e-6, module.FROZEN_HOOK_MODULE_FULL_NAME), {"history_steps": 8, "horizon_steps": 20}),
            ("history", 64, _collector_result(), {"history_steps": 7, "horizon_steps": 20}),
            ("horizon", 64, _collector_result(), {"history_steps": 8, "horizon_steps": 19}),
        )
        for name, loader_size, collector, contract in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root, run_root, tensor_root = self._make_roots(temporary)
                (tensor_root / "tensor_contract.json").write_text(json.dumps(contract), encoding="utf-8")
                with patch.object(module, "_prepare_frozen_link_validation", return_value=(_TraceModel(), [object()] * loader_size, _runner_provenance(root))), patch.object(module, "_collect_edge_gru_trace", return_value=collector), patch.object(module, "build_edge_gru_interface_trace"):
                    with self.assertRaises(ValueError):
                        module.run_formal_p4_edge_gru_interface_trace(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "output")

    def test_publish_refuses_overwrite_and_cleans_only_its_own_staging(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        report = {"schema_version": module.SCHEMA_VERSION}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "output"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                module._publish(target, report)
            self.assertTrue(target.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "output"
            unrelated = root / ".output.staging-keep"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "forced"):
                module._publish(target, report, before_rename=lambda: (_ for _ in ()).throw(RuntimeError("forced")))
            self.assertFalse(target.exists())
            self.assertTrue((unrelated / "keep.txt").is_file())
            self.assertEqual(list(root.glob(".output.staging-*")), [unrelated])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "output"
            def create_target():
                target.mkdir()
                (target / "winner.txt").write_text("winner", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module._publish(target, report, before_rename=create_target)
            self.assertTrue((target / "winner.txt").is_file())
            self.assertFalse(list(root.glob(".output.staging-*")))

    def test_existing_target_is_rejected_before_prepare(self):
        import run_formal_p4_edge_gru_interface_trace_v1 as module

        with tempfile.TemporaryDirectory() as temporary:
            root, run_root, tensor_root = self._make_roots(temporary)
            output = root / "output"
            output.mkdir()
            with patch.object(module, "_prepare_frozen_link_validation") as prepare:
                with self.assertRaises(FileExistsError):
                    module.run_formal_p4_edge_gru_interface_trace(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=output)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
