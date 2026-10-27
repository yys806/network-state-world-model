"""R2-09A deterministic CPU service and task-lifecycle acceptance tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))

from pi_jwm.deterministic_cpu_service_lifecycle_v1 import (  # noqa: E402
    COMPLETED,
    DETECTION_FAILED,
    DETECTION_NOT_EXECUTED,
    FEATURE_PACKET_NOT_DELIVERED,
    REQUEST_NOT_DELIVERED,
    RSU_PROCESSING_TIMEOUT,
    DeterministicCpuServiceLifecycle,
    DeterministicLifecyclePolicy,
)
from pi_jwm.scenario_contract_v1 import CONFIG_DIR, record, validate_record  # noqa: E402
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked  # noqa: E402


def config():
    return json.loads((CONFIG_DIR / "pi_jwm_scenario_v1.synthetic_test.json").read_text(encoding="utf-8"))


def provenance(c):
    return c["parameters"]["sources"][0]["data_provenance"]


def make_task(c, task_id="task:0", packet_ids=("packet:0",), *, deadline=.35, target=.0, generated=.0):
    return record(
        "PerceptionTask", provenance(c), task_id=task_id, region_id="region_A",
        generated_at_s=generated, target_observation_time_s=target,
        allowed_total_latency_s=deadline - target, absolute_deadline_s=deadline,
        feature_packet_ids=list(packet_ids), status="waiting_for_features",
        actual_completion_time_s=None,
    )


def make_packet(c, task_id="task:0", packet_id="packet:0", *, byte_count=1000,
                ready=.0, source_index=0, observation_id="ob:0"):
    source = c["parameters"]["sources"][source_index]
    return record(
        "FeaturePacket", provenance(c), packet_id=packet_id, source_id=source["source_id"],
        task_id=task_id, observation_id=observation_id, byte_count=byte_count,
        preparation_started_at_s=ready, ready_at_s=ready,
        preparation_delay_source="configured_synthetic", preparation_evidence_ref=None,
        granularity="source_complete", block_id=None, spatial_block_contract_ref=None,
        integrity_requirement="complete_packet", payload_role="synthetic_size_only",
        payload_ref="R2-09A synthetic size only",
    )


def engine(c, *, detector=None, processing=.04, boundary="inclusive", order="processing_before_deadline"):
    policy = DeterministicLifecyclePolicy.from_scene_config(
        c, processing_duration_s=processing, deadline_boundary=boundary,
        same_time_event_order=order,
    )
    return DeterministicCpuServiceLifecycle(c, policy, detector=detector)


def complete_request(e, rate=.016, start=0., end=.05, rb=0):
    return e.service_request(
        "task:0", event_id=f"request:{start}", interval_start_s=start, interval_end_s=end,
        effective_rbs=[rb], rate_values=[rate],
    )


class DeterministicCpuLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.c = config()

    def add_one(self, *, task_id="task:0", packet_id="packet:0", packet_bytes=1000,
                deadline=.35, detector=None, processing=.04, boundary="inclusive",
                order="processing_before_deadline"):
        e = engine(self.c, detector=detector, processing=processing,
                   boundary=boundary, order=order)
        task = make_task(self.c, task_id, (packet_id,), deadline=deadline)
        packet = make_packet(self.c, task_id, packet_id, byte_count=packet_bytes)
        e.register_task(task, request_byte_count=100)
        e.register_packet(packet)
        return e

    def test_single_slot_complete_and_receipt_is_synthetic(self):
        e = self.add_one()
        complete_request(e)
        receipt = e.service_packet(
            "packet:0", event_id="packet:single", interval_start_s=.05,
            interval_end_s=.15, effective_rbs=[1], rate_values=[.08],
        )
        self.assertEqual(receipt["delivered_bytes"], 1000)
        self.assertEqual(receipt["result_kind"], "delivery_receipt")
        self.assertEqual(receipt["physical_service_status"], "SYNTHETIC_ONLY")
        self.assertEqual(e.packet_state("packet:0")["remaining_bytes"], 0)

    def test_multi_slot_fractional_capacity_and_partial_progress(self):
        e = self.add_one(packet_bytes=1000)
        complete_request(e)
        first = e.service_packet(
            "packet:0", event_id="packet:part1", interval_start_s=.05,
            interval_end_s=.15, effective_rbs=[1], rate_values=[.004],
        )
        second = e.service_packet(
            "packet:0", event_id="packet:part2", interval_start_s=.15,
            interval_end_s=.25, effective_rbs=[1], rate_values=[.004],
        )
        self.assertEqual((first["delivered_bytes"], second["delivered_bytes"]), (50, 50))
        self.assertEqual(e.packet_state("packet:0")["delivered_bytes"], 100)
        self.assertEqual(e.packet_state("packet:0")["status"], "partial")

    def test_bit_per_second_unit_and_deadline_clipping(self):
        e = self.add_one(packet_bytes=200)
        request = e.service_request("task:0", event_id="request:bits", interval_start_s=0.,
                                    interval_end_s=.05, effective_rbs=[0], rate_values=[16000.],
                                    rate_unit="bit/s")
        self.assertEqual(request["delivered_bytes"], 100)
        packet = e.service_packet("packet:0", event_id="packet:clip", interval_start_s=.05,
                                  interval_end_s=.5, effective_rbs=[1], rate_values=[16000.],
                                  rate_unit="bit/s")
        self.assertAlmostEqual(packet["interval_end_s"], .35)
        self.assertEqual(packet["delivered_bytes"], 200)

    def test_task_crosses_observation_boundary_without_reset(self):
        e = self.add_one(packet_bytes=2000, deadline=.55)
        complete_request(e)
        e.advance_to(.1)
        e.service_packet("packet:0", event_id="p1", interval_start_s=.1,
                         interval_end_s=.2, effective_rbs=[1], rate_values=[.008])
        state_before = e.task_state("task:0")
        e.advance_to(.2)
        self.assertEqual(e.task_state("task:0")["task_id"], "task:0")
        self.assertEqual(e.task_state("task:0")["record"]["absolute_deadline_s"], .55)
        self.assertEqual(e.task_state("task:0")["request"]["delivered_bytes"], 100)
        self.assertEqual(state_before["record"]["task_id"], e.task_state("task:0")["record"]["task_id"])

    def test_inclusive_completion_at_deadline(self):
        e = self.add_one(deadline=.35, processing=.1, detector=lambda *_: True)
        complete_request(e)
        e.service_packet("packet:0", event_id="p1", interval_start_s=.05,
                         interval_end_s=.25, effective_rbs=[1], rate_values=[.04])
        e.advance_to(.35)
        self.assertEqual(e.task_state("task:0")["terminal_reason"], COMPLETED)
        self.assertEqual(e.task_state("task:0")["record"]["status"], "completed")

    def test_exclusive_and_deadline_first_strategies_are_explicit(self):
        for boundary, order in (("exclusive", "processing_before_deadline"),
                                ("inclusive", "deadline_before_processing")):
            e = self.add_one(deadline=.35, processing=.1, boundary=boundary, order=order,
                             detector=lambda *_: True)
            complete_request(e)
            e.service_packet("packet:0", event_id="p1", interval_start_s=.05,
                             interval_end_s=.25, effective_rbs=[1], rate_values=[.04])
            e.advance_to(.35)
            self.assertEqual(e.task_state("task:0")["terminal_reason"], RSU_PROCESSING_TIMEOUT)

    def test_request_and_feature_packet_failures_are_distinct(self):
        e = self.add_one(packet_bytes=1000)
        e.service_request("task:0", event_id="request:partial", interval_start_s=0.,
                          interval_end_s=.05, effective_rbs=[0], rate_values=[.001])
        e.advance_to(.35)
        self.assertEqual(e.task_state("task:0")["terminal_reason"], REQUEST_NOT_DELIVERED)

        e = self.add_one(packet_bytes=10000)
        complete_request(e)
        e.service_packet("packet:0", event_id="packet:partial", interval_start_s=.05,
                         interval_end_s=.15, effective_rbs=[1], rate_values=[.004])
        e.advance_to(.35)
        self.assertEqual(e.task_state("task:0")["terminal_reason"], FEATURE_PACKET_NOT_DELIVERED)

    def test_processing_timeout_is_separate_from_packet_timeout(self):
        e = self.add_one(packet_bytes=1000, processing=.21)
        complete_request(e)
        e.service_packet("packet:0", event_id="packet:done", interval_start_s=.05,
                         interval_end_s=.15, effective_rbs=[1], rate_values=[.08])
        e.advance_to(.35)
        self.assertEqual(e.task_state("task:0")["terminal_reason"], RSU_PROCESSING_TIMEOUT)

    def test_detection_success_failure_and_not_executed_are_simulated(self):
        cases = [(lambda *_: True, COMPLETED), (lambda *_: False, DETECTION_FAILED),
                 (None, DETECTION_NOT_EXECUTED)]
        for detector, expected in cases:
            with self.subTest(expected=expected):
                e = self.add_one(detector=detector)
                complete_request(e)
                e.service_packet("packet:0", event_id="packet:det", interval_start_s=.05,
                                 interval_end_s=.15, effective_rbs=[1], rate_values=[.08])
                e.advance_to(.19)
                self.assertEqual(e.task_state("task:0")["terminal_reason"], expected)
                validate_record(e.task_state("task:0")["record"])
                self.assertEqual(e.results()["perception_results"][0]["output_role"],
                                 "synthetic_contract_fixture")

    def test_multiple_tasks_and_multiple_packets_are_independent(self):
        e = engine(self.c, detector=lambda *_: True)
        t0 = make_task(self.c, "task:0", ("packet:0", "packet:1"), deadline=.5)
        t1 = make_task(self.c, "task:1", ("packet:2",), deadline=.35)
        e.register_task(t0, request_byte_count=100)
        e.register_task(t1, request_byte_count=100)
        e.register_packet(make_packet(self.c, "task:0", "packet:0", byte_count=400))
        e.register_packet(make_packet(self.c, "task:0", "packet:1", byte_count=600, source_index=0))
        e.register_packet(make_packet(self.c, "task:1", "packet:2", byte_count=1000, source_index=2))
        e.service_request("task:0", event_id="r0", interval_start_s=0., interval_end_s=.05,
                          effective_rbs=[0], rate_values=[.016])
        e.service_request("task:1", event_id="r1", interval_start_s=0., interval_end_s=.05,
                          effective_rbs=[2], rate_values=[.016])
        e.service_packet("packet:0", event_id="p0", interval_start_s=.05, interval_end_s=.09,
                         effective_rbs=[1], rate_values=[.08])
        e.service_packet("packet:1", event_id="p1", interval_start_s=.05, interval_end_s=.11,
                         effective_rbs=[3], rate_values=[.08])
        e.service_packet("packet:2", event_id="p2", interval_start_s=.05, interval_end_s=.15,
                         effective_rbs=[4], rate_values=[.08])
        e.advance_to(.5)
        self.assertEqual(e.task_state("task:0")["terminal_reason"], COMPLETED)
        self.assertEqual(e.task_state("task:1")["terminal_reason"], COMPLETED)
        self.assertEqual(len(e.results()["perception_results"]), 2)

    def test_fixed_spatial_block_mode_keeps_packet_id_lifecycle(self):
        c = deepcopy(self.c)
        c["parameters"]["feature_packet_granularity"] = "fixed_spatial_blocks"
        c["parameters"]["spatial_block_contract_ref"] = "R2-09A-engineering-block-layout"
        c["engineering_assumptions"] = [
            a for a in c["engineering_assumptions"]
            if a["path"] not in ("parameters.feature_packet_granularity", "parameters.spatial_block_contract_ref")
        ]
        c["engineering_assumptions"].extend([
            {"path": "parameters.feature_packet_granularity", "value": "fixed_spatial_blocks",
             "reason": "ENGINEERING_FIXTURE_ONLY: block interface compatibility test"},
            {"path": "parameters.spatial_block_contract_ref", "value": "R2-09A-engineering-block-layout",
             "reason": "ENGINEERING_FIXTURE_ONLY: block interface compatibility test"},
        ])
        e = engine(c, detector=lambda *_: True)
        task = make_task(c, packet_ids=("block:0", "block:1"))
        e.register_task(task, request_byte_count=100)
        for pid, block in (("block:0", "b0"), ("block:1", "b1")):
            packet = make_packet(c, packet_id=pid, byte_count=500)
            packet["granularity"] = "fixed_spatial_block"
            packet["block_id"] = block
            packet["spatial_block_contract_ref"] = "R2-09A-engineering-block-layout"
            e.register_packet(packet)
        complete_request(e)
        e.service_packet("block:0", event_id="block0", interval_start_s=.05, interval_end_s=.1,
                         effective_rbs=[1], rate_values=[.08])
        e.service_packet("block:1", event_id="block1", interval_start_s=.05, interval_end_s=.1,
                         effective_rbs=[2], rate_values=[.08])
        e.advance_to(.2)
        self.assertEqual(e.task_state("task:0")["terminal_reason"], COMPLETED)
        self.assertEqual(set(e.task_state("task:0")["packets"]), {"block:0", "block:1"})

    def test_request_gates_feature_service(self):
        e = self.add_one()
        with self.assertRaises(InterfaceBlocked):
            e.service_packet("packet:0", event_id="premature", interval_start_s=0.,
                             interval_end_s=.1, effective_rbs=[0], rate_values=[.08])

    def test_rb_time_unit_and_duplicate_guards(self):
        e = self.add_one()
        with self.assertRaises(InterfaceBlocked):
            e.service_request("task:0", event_id="bad-rate", interval_start_s=0., interval_end_s=.1,
                              effective_rbs=[0], rate_values=[-1.])
        with self.assertRaises(InterfaceBlocked):
            e.service_request("task:0", event_id="bad-rb", interval_start_s=0., interval_end_s=.1,
                              effective_rbs=[50], rate_values=[1.])
        with self.assertRaises(InterfaceBlocked):
            e.service_request("task:0", event_id="bad-unit", interval_start_s=0., interval_end_s=.1,
                              effective_rbs=[0], rate_values=[1.], rate_unit="MB/s")
        e.service_request("task:0", event_id="r1", interval_start_s=0., interval_end_s=.05,
                          effective_rbs=[0], rate_values=[.016])
        e.service_packet("packet:0", event_id="dup", interval_start_s=.05, interval_end_s=.15,
                         effective_rbs=[0], rate_values=[.08])
        with self.assertRaises(InterfaceBlocked):
            e.service_packet("packet:0", event_id="dup", interval_start_s=.05, interval_end_s=.15,
                             effective_rbs=[1], rate_values=[.08])

    def test_overlapping_same_target_and_rb_are_rejected(self):
        e = self.add_one(packet_bytes=2000)
        complete_request(e)
        e.service_packet("packet:0", event_id="p1", interval_start_s=.05, interval_end_s=.15,
                         effective_rbs=[1], rate_values=[.04])
        with self.assertRaises(InterfaceBlocked):
            e.service_packet("packet:0", event_id="overlap-target", interval_start_s=.1,
                             interval_end_s=.2, effective_rbs=[2], rate_values=[.04])
        # Different packets may share time but not an RB.
        e2 = engine(self.c, detector=lambda *_: True)
        e2.register_task(make_task(self.c, "task:0", ("packet:0",), deadline=.5), request_byte_count=100)
        e2.register_task(make_task(self.c, "task:1", ("packet:1",), deadline=.5), request_byte_count=100)
        e2.register_packet(make_packet(self.c, "task:0", "packet:0", byte_count=1000))
        e2.register_packet(make_packet(self.c, "task:1", "packet:1", byte_count=1000, source_index=1))
        e2.service_request("task:0", event_id="a", interval_start_s=0., interval_end_s=.05,
                           effective_rbs=[0], rate_values=[.016])
        e2.service_request("task:1", event_id="b", interval_start_s=0., interval_end_s=.05,
                           effective_rbs=[2], rate_values=[.016])
        e2.service_packet("packet:0", event_id="c", interval_start_s=.05, interval_end_s=.15,
                          effective_rbs=[1], rate_values=[.08])
        with self.assertRaises(InterfaceBlocked):
            e2.service_packet("packet:1", event_id="d", interval_start_s=.05, interval_end_s=.15,
                              effective_rbs=[1], rate_values=[.08])

    def test_deterministic_replay(self):
        def run():
            e = self.add_one(detector=lambda *_: True)
            complete_request(e)
            e.service_packet("packet:0", event_id="p1", interval_start_s=.05, interval_end_s=.1,
                             effective_rbs=[1], rate_values=[.016])
            e.service_packet("packet:0", event_id="p2", interval_start_s=.1, interval_end_s=.2,
                             effective_rbs=[1], rate_values=[.064])
            e.advance_to(.3)
            return e.results()
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
