"""R2-08 pure contract tests; synthetic records only, no lifecycle execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from pi_jwm.scenario_contract_v1 import (
    CONFIG_DIR, record, validate_record, validate_scene_config, sample_region_tasks,
    validate_record_graph, validate_schedule_progress, packet_from_legacy_payload,
    service_from_legacy_capacity,
)
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked


def fixture_config():
    return json.loads((CONFIG_DIR / 'pi_jwm_scenario_v1.synthetic_test.json').read_text(encoding='utf-8'))


def set_parameter(c, key, value):
    c['parameters'][key] = deepcopy(value)
    path = 'parameters.' + key
    c['engineering_assumptions'] = [a for a in c['engineering_assumptions'] if a['path'] != path]
    if value is not None:
        c['engineering_assumptions'].append(dict(path=path, value=deepcopy(value), reason='ENGINEERING_FIXTURE_ONLY: test variation'))


def fixture_records():
    c = fixture_config()
    source = c['parameters']['sources'][0]
    p = source['data_provenance']
    ob = record('ObservationRecord', p, observation_id='ob:0:1', source_id=source['source_id'],
                sensor_id=source['sensor_id'], frame_key='synthetic-frame-1',
                target_observation_time_s=.1, available_at_s=.1, availability='available',
                position_m=[1, 2, 0], trajectory_origin=source['trajectory_origin'], trajectory_ref=source['trajectory_ref'])
    task = record('PerceptionTask', p, task_id='task:A:1', region_id='region_A', generated_at_s=.12,
                  target_observation_time_s=.1, allowed_total_latency_s=.35, absolute_deadline_s=.45,
                  feature_packet_ids=['packet:0:1'], status='waiting_for_features', actual_completion_time_s=None)
    packet = record('FeaturePacket', p, packet_id='packet:0:1', source_id=source['source_id'],
                    task_id=task['task_id'], observation_id=ob['observation_id'], byte_count=1000,
                    preparation_started_at_s=.1, ready_at_s=.14, preparation_delay_source='configured_synthetic',
                    preparation_evidence_ref=None, granularity='source_complete', block_id=None,
                    spatial_block_contract_ref=None, integrity_requirement='complete_packet',
                    payload_role='synthetic_size_only', payload_ref='fixture-size-only; no feature tensor')
    action = record('SchedulingAction', p, action_id='action:1', decision_time_s=.15,
                    information_cutoff_s=.15, effective_time_s=.15, selected_packet_ids=[packet['packet_id']],
                    allocations=[dict(packet_id=packet['packet_id'], source_id=source['source_id'], rsu_id='RSU_0', rb_ids=[0])],
                    information_refs=[ob['observation_id']], prediction_refs=[], trigger_event_id=None)
    service = record('ServiceResult', p, service_id='service:1', link_service_id='link-sample:1',
                     action_id=action['action_id'], packet_id=packet['packet_id'], source_id=source['source_id'],
                     rsu_id='RSU_0', interval_start_s=.15, interval_end_s=.18, available_at_s=.15,
                     effective_rbs=[0], rate_values=[8.], rate_unit='Mbps', rate_semantics='fixture_constant',
                     result_kind='capacity_diagnostic', theoretical_capacity_bytes=30000., delivered_bytes=None,
                     delivery_evidence_ref=None, physical_service_status='SYNTHETIC_ONLY')
    result = record('PerceptionResult', p, result_id='result:1', task_id=task['task_id'],
                    used_packet_ids=[packet['packet_id']], output_at_s=.2, actual_completion_time_s=.21,
                    input_receipts=[dict(packet_id=packet['packet_id'], received_byte_count=1000,
                                         available_at_s=.18, integrity_status='complete',
                                         evidence_ref='externally supplied synthetic receipt, not produced by service capacity',
                                         evidence_kind='synthetic_fixture')],
                    detection_output=None, evaluation_result=None, output_role='synthetic_contract_fixture')
    return c, [ob, task, packet, action, service, result]


class SceneContractTests(unittest.TestCase):
    def test_pending_profile_is_valid_but_cannot_execute(self):
        c = json.loads((CONFIG_DIR / 'pi_jwm_scenario_v1.pending.json').read_text(encoding='utf-8'))
        validate_scene_config(c)
        with self.assertRaises(InterfaceBlocked): validate_scene_config(c, require_complete=True)
        c['parameters']['allowed_total_latency_s'] = .35
        with self.assertRaises(InterfaceBlocked): validate_scene_config(c)

    def test_frozen_scope_and_extensions_reject_changes(self):
        for section, field, value in [('frozen_scope', 'observation_period_s', .2),
                                       ('frozen_scope', 'logical_rsu_count', 2), ('extensions', 'rgb', True),
                                       ('frozen_scope', 'uav_trajectory', 'optimized')]:
            c = fixture_config(); c[section][field] = value
            with self.subTest(field=field), self.assertRaises(InterfaceBlocked): validate_scene_config(c)

    def test_explicit_assumptions_and_complete_configuration(self):
        c = fixture_config(); validate_scene_config(c, require_complete=True)
        c['parameters']['rb_count'] = 60
        with self.assertRaises(InterfaceBlocked): validate_scene_config(c)
        c = fixture_config(); set_parameter(c, 'allowed_total_latency_s', None)
        with self.assertRaises(InterfaceBlocked): validate_scene_config(c, require_complete=True)

    def test_invalid_configuration(self):
        for key, value in [('rb_count', -1), ('vehicle_count', True), ('internal_scheduling_interval_s', 0),
                           ('rsu_position_m', [1, 2]), ('feature_preparation_delay_s', -1)]:
            c = fixture_config(); set_parameter(c, key, value)
            with self.subTest(key=key), self.assertRaises(InterfaceBlocked): validate_scene_config(c)
        c = fixture_config(); regions = c['parameters']['regions']; regions[0]['bounds_xy_m'] = [1, 1, 0, 0]
        set_parameter(c, 'regions', regions)
        with self.assertRaises(InterfaceBlocked): validate_scene_config(c)

    def test_100ms_grid_and_probability_endpoints(self):
        c = fixture_config(); regions = c['parameters']['regions']
        regions[0]['arrival_probability'] = 1.; regions[1]['arrival_probability'] = 0.
        set_parameter(c, 'regions', regions)
        for tick in range(5):
            r = sample_region_tasks(c, tick_index=tick, generated_at_s=tick*.1+.02, seed=7)
            self.assertAlmostEqual(r['target_observation_time_s'], tick*.1)
            self.assertEqual([t['region_id'] for t in r['tasks']], ['region_A'])

    def test_region_streams_independent_and_reproducible(self):
        c = fixture_config(); other = deepcopy(c); regions = other['parameters']['regions']
        regions.reverse(); regions[0]['arrival_probability'] = 1
        set_parameter(other, 'regions', regions)
        for tick in range(20):
            a = sample_region_tasks(c, tick_index=tick, generated_at_s=tick*.1, seed=19)
            self.assertEqual(a, sample_region_tasks(c, tick_index=tick, generated_at_s=tick*.1, seed=19))
            b = sample_region_tasks(other, tick_index=tick, generated_at_s=tick*.1, seed=19)
            self.assertEqual(a['trials'][0], b['trials'][0])
        self.assertNotEqual(sample_region_tasks(c, tick_index=1, generated_at_s=.1, seed=19)['trials'],
                            sample_region_tasks(c, tick_index=1, generated_at_s=.1, seed=20)['trials'])

    def test_invalid_tick_seed_and_generation_time(self):
        for tick, generated, seed in [(True, .1, 0), (-1, .1, 0), (1, .09, 0), (1, float('nan'), 0), (1, .1, True)]:
            with self.subTest(tick=tick, seed=seed), self.assertRaises(InterfaceBlocked):
                sample_region_tasks(fixture_config(), tick_index=tick, generated_at_s=generated, seed=seed)

    def test_separate_times_and_deadline_anchor(self):
        c, rows = fixture_records(); validate_record_graph(c, rows)
        ob, task, packet = rows[:3]
        self.assertEqual([ob['target_observation_time_s'], task['generated_at_s'], packet['ready_at_s'], task['absolute_deadline_s']], [.1, .12, .14, .45])
        task['absolute_deadline_s'] = task['generated_at_s'] + task['allowed_total_latency_s']
        with self.assertRaises(InterfaceBlocked): validate_record(task)

    def test_old_task_survives_observation_boundaries_without_transition(self):
        c, rows = fixture_records(); before = deepcopy(rows)
        for tick in [2, 3, 5]:
            sample_region_tasks(c, tick_index=tick, generated_at_s=tick*.1, seed=11)
            validate_record_graph(c, rows)
        self.assertEqual(rows, before)  # even after deadline: no timeout engine exists in R2-08

    def test_unknown_negative_nonfinite_bool_and_wrong_units_rejected(self):
        c, rows = fixture_records()
        for field, value in [('unexpected', 1), ('byte_count', -1), ('byte_count', True),
                             ('ready_at_s', float('inf')), ('ready_at_s', float('nan')),
                             ('units', dict(time='ms', size='byte'))]:
            p = deepcopy(rows[2]); p[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(InterfaceBlocked): validate_record(p)

    def test_invalid_time_and_missing_observation(self):
        _, rows = fixture_records()
        for index, key, val in [(0, 'target_observation_time_s', .11), (1, 'generated_at_s', .09),
                                (2, 'ready_at_s', .09), (4, 'interval_end_s', .15),
                                (0, 'availability', 'missing')]:
            r = deepcopy(rows[index]); r[key] = val
            with self.subTest(key=key), self.assertRaises(InterfaceBlocked): validate_record(r)

    def test_identity_source_and_ownership_guards(self):
        c, rows = fixture_records()
        for index, key, value in [(0, 'sensor_id', 'unknown'), (2, 'source_id', 'unknown'),
                                  (2, 'task_id', 'unknown'), (4, 'packet_id', 'unknown')]:
            bad = deepcopy(rows); bad[index][key] = value
            with self.subTest(key=key), self.assertRaises(InterfaceBlocked): validate_record_graph(c, bad)
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows + [deepcopy(rows[0])])

    def test_unprepared_and_unallocated_packets_rejected(self):
        c, rows = fixture_records(); rows[3]['information_cutoff_s'] = .13
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[4]['effective_rbs'] = [1]
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[3]['allocations'][0]['rb_ids'] = [50]
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)

    def test_future_observations_and_feedback_rejected_prediction_allowed(self):
        c, rows = fixture_records(); ob = deepcopy(rows[0]); ob.update(observation_id='future', target_observation_time_s=.2, available_at_s=.2)
        rows.append(ob); rows[3]['information_refs'].append('future')
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[3]['information_refs'].append(rows[4]['service_id']); rows[4]['available_at_s'] = .18
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[3]['prediction_refs'] = [dict(prediction_id='p:1', issued_at_s=.14, target_time_s=.3, model_ref='fixture-model-not-executed', kind='world_model_prediction')]
        validate_record_graph(c, rows)
        rows[3]['prediction_refs'][0]['issued_at_s'] = .16
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)

    def test_schedule_must_advance_time_or_consume_finite_fresh_event(self):
        _, rows = fixture_records(); a = rows[3]; b = deepcopy(a); b['action_id'] = 'action:2'
        with self.assertRaises(InterfaceBlocked): validate_schedule_progress([a, b])
        b['trigger_event_id'] = 'e:1'; events = [dict(event_id='e:1', time_s=.15, kind='timer')]
        validate_schedule_progress([a, b], available_events=events)
        d = deepcopy(b); d['action_id'] = 'action:3'
        with self.assertRaises(InterfaceBlocked): validate_schedule_progress([a, b, d], available_events=events)
        with self.assertRaises(InterfaceBlocked): validate_schedule_progress([a, a])
        b.update(trigger_event_id=None, decision_time_s=.18, effective_time_s=.18)
        validate_schedule_progress([a, b])
        with self.assertRaises(InterfaceBlocked): validate_schedule_progress([b, a])

    def test_capacity_is_not_delivery_and_units_are_explicit(self):
        _, rows = fixture_records(); s = rows[4]; validate_record(s)
        s['delivered_bytes'] = 1
        with self.assertRaises(InterfaceBlocked): validate_record(s)
        s['delivered_bytes'] = None; s['rate_values'] = [8e6]; s['rate_unit'] = 'bit/s'; validate_record(s)
        s['rate_unit'] = 'MB/s'
        with self.assertRaises(InterfaceBlocked): validate_record(s)

    def test_external_delivery_receipt_fields_not_capacity_substitution(self):
        _, rows = fixture_records(); s = rows[4]
        s.update(result_kind='delivery_receipt', delivered_bytes=1000, delivery_evidence_ref='synthetic supplied receipt', available_at_s=.18)
        validate_record(s)
        s['delivery_evidence_ref'] = None
        with self.assertRaises(InterfaceBlocked): validate_record(s)
        s['delivery_evidence_ref'] = 'fixture'; s['rate_semantics'] = 'instantaneous_sample_assumed_constant'
        with self.assertRaises(InterfaceBlocked): validate_record(s)

    def test_result_requires_complete_upstream_input_receipts(self):
        c, rows = fixture_records(); rows[5]['input_receipts'] = []
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[5]['input_receipts'][0]['received_byte_count'] = 999
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[5]['input_receipts'][0]['available_at_s'] = .22
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        c, rows = fixture_records(); rows[5]['input_receipts'][0]['evidence_ref'] = rows[4]['service_id']
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        rows[4].update(result_kind='delivery_receipt', delivered_bytes=999, delivery_evidence_ref='synthetic-test-only', available_at_s=.18)
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        rows[4]['delivered_bytes'] = 1000; validate_record_graph(c, rows)

    def test_prediction_cannot_masquerade_as_actual_feedback(self):
        c, rows = fixture_records()
        rows[4]['provenance'] = dict(kind='world_model_prediction', source_ref='fixture-model',
                                    time_basis='simulation_seconds', physical_sync_status='NOT_APPLICABLE', evidence_ref=None)
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)
        rows[4]['result_kind'] = 'predicted_service'; validate_record_graph(c, rows)
        rows[3]['information_refs'].append(rows[4]['service_id'])
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)

    def test_result_and_delivery_receipt_times_and_budgets(self):
        _, rows = fixture_records(); s = rows[4]
        s.update(result_kind='delivery_receipt', delivered_bytes=30001, delivery_evidence_ref='fixture', available_at_s=.18)
        with self.assertRaises(InterfaceBlocked): validate_record(s)
        s.update(delivered_bytes=1000, available_at_s=.17)
        with self.assertRaises(InterfaceBlocked): validate_record(s)
        r = rows[5]; r['actual_completion_time_s'] = .19
        with self.assertRaises(InterfaceBlocked): validate_record(r)

    def test_no_double_counting_link_service_or_rb_capacity(self):
        c, rows = fixture_records(); another = deepcopy(rows[4]); another['service_id'] = 'service:2'
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows + [another])
        another['link_service_id'] = 'link-sample:2'
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows + [another])
        c, rows = fixture_records(); p = deepcopy(rows[2]); p['packet_id'] = 'packet:other'; rows.append(p)
        rows[1]['feature_packet_ids'].append(p['packet_id']); rows[3]['selected_packet_ids'].append(p['packet_id'])
        a = deepcopy(rows[3]['allocations'][0]); a['packet_id'] = p['packet_id']; rows[3]['allocations'].append(a)
        with self.assertRaises(InterfaceBlocked): validate_record_graph(c, rows)

    def test_spatial_block_is_contract_only_not_generated_feature(self):
        c, rows = fixture_records(); set_parameter(c, 'feature_packet_granularity', 'fixed_spatial_blocks')
        with self.assertRaises(InterfaceBlocked): validate_scene_config(c)
        set_parameter(c, 'spatial_block_contract_ref', 'synthetic block layout fixture')
        rows[2].update(granularity='fixed_spatial_block', block_id='block:0', spatial_block_contract_ref='synthetic block layout fixture')
        validate_record_graph(c, rows)

    def test_synthetic_provenance_cannot_assert_physical_sync(self):
        _, rows = fixture_records(); rows[0]['provenance'].update(physical_sync_status='VERIFIED', evidence_ref='made-up')
        with self.assertRaises(InterfaceBlocked): validate_record(rows[0])

    def test_legacy_payload_byte_id_preserved_not_encoder_features(self):
        from pi_jwm.v2u4real_airfogsim_adapter_v1 import PerceptionPayloadRecord
        c, rows = fixture_records(); ob, task = rows[:2]
        payload = PerceptionPayloadRecord('legacy:1', ob['frame_key'], ob['sensor_id'], 123, 0,
                                          'native', 'unverified', 'MockPerception.fixture')
        packet = packet_from_legacy_payload(payload, task, ob, ready_at_s=.14)
        self.assertEqual((packet['packet_id'], packet['byte_count']), ('legacy:1', 123))
        self.assertEqual(packet['payload_role'], 'legacy_detection_result_json')
        task['feature_packet_ids'] = [packet['packet_id']]
        validate_record_graph(c, [ob, task, packet])

    def test_r2_07_capacity_adapter_explicit_time_no_delivery(self):
        import numpy as np
        from pi_jwm.airfogsim_service_interface_v1 import slot_service
        _, rows = fixture_records(); capacity = slot_service([8., 5.], np.array([True, False]), .03)
        r = service_from_legacy_capacity(capacity, rows[3], rows[2], service_id='legacy:s', sample_time_s=.15)
        self.assertEqual(r['delivered_bytes'], None); self.assertEqual(r['physical_service_status'], 'BLOCKED')
        self.assertAlmostEqual(r['theoretical_capacity_bytes'], 30000)
        capacity['sample_time_s'] = .18
        with self.assertRaises(InterfaceBlocked):
            service_from_legacy_capacity(capacity, rows[3], rows[2], service_id='legacy:s', sample_time_s=.15)


if __name__ == '__main__':
    unittest.main()
