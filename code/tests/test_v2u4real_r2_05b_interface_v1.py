"""R2-05B perception contract and AirFogSim ingress tests.

The real-frame test reads the existing audited train sample when it is present.
All tests are CPU-only and do not instantiate a detector or use ground truth
to build a prediction.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))

from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked
from pi_jwm.v2u4real_perception_interface_v1 import (
    BOX_FORMAT,
    BOX_DIMENSION_ORDER,
    DetectionObject,
    DetectionResult,
    MockPerception,
    PerceptionFrame,
    canonical_result_bytes,
    read_v2u4real_frame,
    validate_detection_result,
)
from pi_jwm.v2u4real_airfogsim_adapter_v1 import (
    PerceptionPayloadRecord,
    submit_payload_to_task_manager,
    task_record_dict,
)


class PerceptionContractTests(unittest.TestCase):
    def frame(self, points=None, annotations=None):
        return PerceptionFrame(
            frame_key="1752814590.2",
            sensor_id="agent-1/ouster",
            points=np.asarray(points if points is not None else [[0, 1, 2], [2, 3, 4]], dtype=np.float64),
            coordinate_frame="v2u4real::agent-1::ouster::native",
            coordinate_status="dataset_native_unverified",
            source_metadata={"source": "synthetic_test"},
            annotation_metadata=annotations or {"labels_used_by_mock_perception": False},
        )

    def test_input_rejects_shape_empty_and_nonfinite(self):
        for points in ([], np.zeros((2, 2)), np.array([[0, 1, np.nan]])):
            with self.subTest(points=repr(points)), self.assertRaises(InterfaceBlocked):
                self.frame(points)

    def test_mock_is_deterministic_and_does_not_consume_labels(self):
        detector = MockPerception()
        first = detector.predict(self.frame(annotations={"objects": {"1": {"x": 1}}}))
        changed = detector.predict(self.frame(annotations={"objects": {"1": {"x": 999}, "2": {"x": -1}}}))
        repeated = detector.predict(self.frame())
        self.assertEqual(first.to_dict(), changed.to_dict())
        self.assertEqual(first.to_dict(), repeated.to_dict())
        self.assertEqual(first.result_source, MockPerception.source)
        self.assertEqual(first.prediction_semantics, "predictions_are_not_ground_truth")

    def test_output_box_contract_and_roundtrip(self):
        result = MockPerception().predict(self.frame())
        self.assertEqual(result.to_dict()["box_format"], BOX_FORMAT)
        self.assertEqual(tuple(result.to_dict()["box_dimension_order"]), BOX_DIMENSION_ORDER)
        restored = validate_detection_result(result.to_dict())
        self.assertEqual(result.to_dict(), restored.to_dict())
        self.assertEqual(canonical_result_bytes(result), canonical_result_bytes(restored))

    def test_invalid_detection_is_rejected(self):
        with self.assertRaises(InterfaceBlocked):
            DetectionObject(
                category="vehicle", box3d=(0, 0, 0, 1, 0, 1, 0), confidence=0.5,
                frame_key="f", sensor_id="s", coordinate_frame="c",
                coordinate_status="unverified", result_source="test",
            )
        result = MockPerception().predict(self.frame()).to_dict()
        result["objects"][0]["illegal_field"] = 1
        with self.assertRaises(InterfaceBlocked):
            DetectionResult.from_dict(result)
        result = MockPerception().predict(self.frame()).to_dict()
        result["objects"][0]["box3d"][3] = float("nan")
        with self.assertRaises(InterfaceBlocked):
            DetectionResult.from_dict(result)

    def test_real_fixed_frame_is_read_without_coordinate_inference(self):
        root = ROOT / "code/artifacts/data/v2u4real_r2_01/selected_samples"
        if not (root / "train").is_dir():
            self.skipTest("audited V2U4Real sample is not present")
        bundle = read_v2u4real_frame(root, "2025-07-18-12-53_2", "1752814590.2", "1")
        self.assertGreater(bundle.frame.point_count, 0)
        self.assertEqual(bundle.frame.frame_key, "1752814590.2")
        self.assertEqual(bundle.frame.coordinate_status, "dataset_native_unverified")
        self.assertTrue(bundle.file_metadata["xyz_finite"])
        self.assertRegex(bundle.file_metadata["pcd_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(bundle.file_metadata["yaml_sha256"], r"^[0-9a-f]{64}$")


class _FakeTask:
    def __init__(self, task_id):
        self._task_id = task_id
        self.attributes = {"task_id": task_id}

    def setAttribute(self, key, value):
        self.attributes[key] = value

    def getTaskId(self):
        return self._task_id

    def to_dict(self):
        return copy.deepcopy(self.attributes)


class _FakeTaskManager:
    def __init__(self):
        self.tasks = []
        self.waiting = {}

    def generateTaskInfoOfMission(self, task_node_id, task_deadline, arrival_time, return_size, **kwargs):
        task = _FakeTask(f"Task_{len(self.tasks) + 1}")
        task.setAttribute("task_node_id", task_node_id)
        task.setAttribute("task_deadline", task_deadline)
        task.setAttribute("task_arrival_time", arrival_time)
        self.tasks.append(task)
        return task

    def getWaitingToOffloadTasksByNodeId(self, node_id):
        return self.waiting.get(node_id, [])

    def setWaitingToOffloadTasksByNodeId(self, node_id, task_list):
        self.waiting[node_id] = task_list


class AirFogSimIngressTests(unittest.TestCase):
    def test_payload_and_application_ingress_stop_before_wireless(self):
        frame = PerceptionFrame(
            frame_key="f", sensor_id="agent-1/ouster", points=np.ones((2, 3)),
            coordinate_frame="native", coordinate_status="unverified", source_metadata={},
        )
        result = MockPerception().predict(frame)
        payload = PerceptionPayloadRecord.from_result(result)
        manager = _FakeTaskManager()
        task, receipt = submit_payload_to_task_manager(
            manager, payload, source_node_id="vehicle_0", target_node_id="RSU_0", simulation_time=0.0,
        )
        self.assertEqual(receipt.software_interface_status, "PASS")
        self.assertEqual(receipt.simulation_task_record_status, "PASS_WAITING_TO_OFFLOAD")
        self.assertEqual(receipt.wireless_delivery_status, "NOT_ATTEMPTED_BLOCKED")
        self.assertEqual(len(manager.waiting["vehicle_0"]), 1)
        record = task_record_dict(task)
        self.assertEqual(record["perception_payload_id"], payload.payload_id)
        self.assertEqual(record["application_task_state"], "waiting_to_offload")
        self.assertEqual(record["native_lifecycle_interpretation"], "not_used_as_wireless_delivery_evidence")
        self.assertEqual(record["wireless_delivery_status"], "NOT_ATTEMPTED_BLOCKED")

    def test_payload_id_is_reproducible(self):
        frame = PerceptionFrame(
            frame_key="f", sensor_id="s", points=np.array([[1, 2, 3]], dtype=float),
            coordinate_frame="native", coordinate_status="unverified", source_metadata={},
        )
        result = MockPerception().predict(frame)
        self.assertEqual(PerceptionPayloadRecord.from_result(result), PerceptionPayloadRecord.from_result(result))


if __name__ == "__main__":
    unittest.main()
