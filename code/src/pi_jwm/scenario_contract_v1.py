"""R2-08 versioned JSON contracts and pure consistency checks.

This module does not run an environment, maintain a queue, accumulate service,
complete/expire tasks, evaluate detection, or access any dataset.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
import random

from .v2u4real_interface_v1 import InterfaceBlocked

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / 'code/scripts/configs'
VERSION = 'PIJWM-Scenario-Interfaces-v1'
PERIOD_S = .1
PENDING_DECISIONS = {
    'regions_and_bounds', 'real_synthetic_source_mapping', 'feature_packet_granularity',
    'per_region_arrival_probability', 'allowed_total_latency', 'feature_preparation_delay',
    'internal_scheduling_interval', 'rsu_position_and_physical_coverage',
    'rb_budget_and_sharing', 'same_time_event_order',
}
ID_FIELDS = dict(ObservationRecord='observation_id', PerceptionTask='task_id',
                 FeaturePacket='packet_id', SchedulingAction='action_id',
                 ServiceResult='service_id', PerceptionResult='result_id')


def _require(condition, reason):
    if not condition:
        raise InterfaceBlocked(reason)


def _json(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InterfaceBlocked('contract requires finite JSON values') from exc


@lru_cache(maxsize=2)
def _validator(name):
    # Installed dependency; no remote schema resolution or package installation.
    from jsonschema import Draft202012Validator
    schema = json.loads((CONFIG_DIR / name).read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema(value, name):
    value = _json(value)
    errors = list(_validator(name).iter_errors(value))
    _require(not errors, 'schema violation: ' + (errors[0].message[:220] if errors else ''))
    return value


def _same(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def _same_capacity(a, b):
    # Relative tolerance accounts for seconds subtraction; does not round bytes.
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _observation_boundary(t):
    _require(t >= 0 and _same(t, round(t / PERIOD_S) * PERIOD_S),
             'target observation must lie on the 100 ms simulation grid')


def _provenance(p):
    if p['physical_sync_status'] == 'VERIFIED' or p['time_basis'] == 'verified_physical_mapping':
        _require(p['physical_sync_status'] == 'VERIFIED' and bool(p['evidence_ref']),
                 'physical assertions require explicit evidence; field checks do not verify that evidence')
    if p['kind'] in ('synthetic_fixture', 'mock_output', 'world_model_prediction'):
        _require(p['physical_sync_status'] != 'VERIFIED', 'synthetic/model output cannot establish physical synchronization')
    if p['kind'] == 'real_train_sample':
        _require(p['source_ref'].startswith('v2u4real://train/'), 'real data reference must explicitly identify official train')


def validate_scene_config(value, *, require_complete=False):
    """Validate a pending specification or a separately labeled test fixture.

    Both v1 profiles retain pending researcher decisions. Neither admits formal
    research execution or training. Completeness is for synthetic tests only.
    """
    c = _schema(value, 'pi_jwm_scenario_v1.schema.json')
    _require(PENDING_DECISIONS <= set(c['pending_researcher_decisions']), 'pending decision list is incomplete')
    p = c['parameters']
    if c['profile_type'] == 'formal_pending':
        _require(not c['engineering_assumptions'], 'formal pending profile cannot import test assumptions')
        _require(all(v is None for k, v in p.items() if k != 'coverage_status'), 'unapproved formal numeric/structural choice')
        _require(p['coverage_status'] == 'BLOCKED_NOT_VERIFIED', 'logical service cannot establish coverage')
    else:
        assumptions = {x['path']: x for x in c['engineering_assumptions']}
        _require(len(assumptions) == len(c['engineering_assumptions']), 'duplicate assumption path')
        for k, v in p.items():
            if v is not None:
                a = assumptions.get('parameters.' + k)
                _require(a is not None and a['value'] == v and a['reason'].startswith('ENGINEERING_FIXTURE_ONLY'),
                         'each test value must carry its explicit engineering-only assumption')
        _require(p['coverage_status'] == 'SYNTHETIC_NO_PHYSICAL_CLAIM', 'test profile cannot imply physical coverage')
    if p['regions'] is not None:
        _require(len({x['region_id'] for x in p['regions']}) == len(p['regions']), 'duplicate region id')
        for region in p['regions']:
            x0, y0, x1, y1 = region['bounds_xy_m']
            _require(x0 < x1 and y0 < y1, 'invalid rectangular test region bounds')
    if p['sources'] is not None:
        sources = p['sources']
        _require(len({x['source_id'] for x in sources}) == len(sources), 'duplicate source id')
        for source in sources:
            _provenance(source['data_provenance'])
            if c['profile_type'] == 'synthetic_contract_test':
                _require(source['data_provenance']['kind'] == 'synthetic_fixture' and source['trajectory_origin'] == 'preset_synthetic',
                         'the test fixture cannot impersonate a real data source')
        for kind, field in (('vehicle', 'vehicle_count'), ('uav', 'uav_count')):
            _require(len({x['node_id'] for x in sources if x['node_type'] == kind}) == p[field], 'node count/catalog mismatch')
    if p['feature_packet_granularity'] == 'fixed_spatial_blocks':
        _require(bool(p['spatial_block_contract_ref']), 'spatial blocks require a declared block contract')
    if require_complete:
        _require(c['profile_type'] == 'synthetic_contract_test', 'formal configuration blocked pending researcher decisions')
        missing = [k for k, v in p.items() if v is None and k != 'spatial_block_contract_ref']
        _require(not missing, 'incomplete explicit test configuration: ' + ','.join(missing))
    return c


def record(kind, provenance, **fields):
    return validate_record(dict(schema_version=VERSION, record_type=kind,
                                units=dict(time='s', size='byte'), provenance=provenance, **fields))


def validate_record(value):
    r = _schema(value, 'pi_jwm_scenario_interfaces_v1.schema.json')
    _provenance(r['provenance'])
    kind = r['record_type']
    if kind == 'ObservationRecord':
        _observation_boundary(r['target_observation_time_s'])
        if r['availability'] == 'available':
            _require(r['available_at_s'] is not None and r['available_at_s'] >= r['target_observation_time_s'], 'observation availability precedes target time')
        else:
            _require(r['available_at_s'] is None, 'missing observation cannot have an availability receipt')
    elif kind == 'PerceptionTask':
        _observation_boundary(r['target_observation_time_s'])
        _require(r['generated_at_s'] >= r['target_observation_time_s'], 'task generation precedes target observation')
        _require(_same(r['absolute_deadline_s'], r['target_observation_time_s'] + r['allowed_total_latency_s']), 'deadline must be target observation plus total latency, not generation plus latency')
        if r['status'] == 'completed':
            _require(r['actual_completion_time_s'] is not None, 'completed record requires a completion receipt time')
        else:
            _require(r['actual_completion_time_s'] is None, 'non-completed task has a completion timestamp')
        if r['actual_completion_time_s'] is not None:
            _require(r['actual_completion_time_s'] >= r['generated_at_s'], 'completion precedes task generation')
    elif kind == 'FeaturePacket':
        _require(r['ready_at_s'] >= r['preparation_started_at_s'], 'feature ready time precedes preparation')
        if r['preparation_delay_source'] == 'measured_encoder':
            _require(bool(r['preparation_evidence_ref']), 'measured preparation needs measurement evidence')
        else:
            _require(r['provenance']['kind'] != 'real_train_sample', 'configured preparation is a simulation parameter, not a real timing measurement')
        if r['granularity'] == 'fixed_spatial_block':
            _require(bool(r['block_id']) and bool(r['spatial_block_contract_ref']), 'spatial packet needs explicit block identity and contract')
        else:
            _require(r['block_id'] is None and r['spatial_block_contract_ref'] is None, 'whole-source packet cannot hide block metadata')
        if r['payload_role'] == 'synthetic_size_only':
            _require(r['provenance']['kind'] == 'synthetic_fixture', 'synthetic bytes cannot masquerade as real encoded features')
    elif kind == 'SchedulingAction':
        _require(r['information_cutoff_s'] <= r['decision_time_s'] <= r['effective_time_s'], 'invalid information/decision/effect order')
        ids = [x['packet_id'] for x in r['allocations']]
        _require(len(ids) == len(set(ids)) and set(ids) == set(r['selected_packet_ids']), 'selected packet/allocation mismatch')
        for pred in r['prediction_refs']:
            _require(pred['issued_at_s'] <= r['information_cutoff_s'], 'prediction was not issued by information cutoff')
    elif kind == 'ServiceResult':
        if r['result_kind'] == 'predicted_service':
            _require(r['provenance']['kind'] == 'world_model_prediction', 'predicted service must declare model provenance')
        else:
            _require(r['provenance']['kind'] != 'world_model_prediction', 'prediction cannot masquerade as a measured service')
        start, end = r['interval_start_s'], r['interval_end_s']
        _require(end > start, 'service interval must advance time')
        _require(len(r['rate_values']) == len(r['effective_rbs']), 'one rate value is required per effective RB')
        factor = 1e6 if r['rate_unit'] == 'Mbps' else 1.
        capacity = sum(r['rate_values']) * factor * (end-start) / 8.
        _require(math.isfinite(capacity), 'capacity overflow')
        _require(r['theoretical_capacity_bytes'] is not None and _same_capacity(capacity, r['theoretical_capacity_bytes']), 'rate, interval and capacity units disagree')
        if r['result_kind'] == 'delivery_receipt':
            _require(r['delivered_bytes'] is not None and bool(r['delivery_evidence_ref']), 'delivery requires separate runtime evidence; capacity is insufficient')
            _require(r['available_at_s'] >= end, 'delivery feedback precedes actual interval end')
            _require(r['delivered_bytes'] <= capacity + 1e-9, 'receipt exceeds its declared single-interval service budget')
            _require(r['provenance']['kind'] != 'world_model_prediction', 'model predictions are not delivery receipts')
            _require(r['rate_semantics'] != 'instantaneous_sample_assumed_constant', 'hypothetical held rate cannot establish delivery')
        else:
            _require(r['delivered_bytes'] is None and r['delivery_evidence_ref'] is None, 'theoretical/predicted service cannot populate delivered bytes')
            _require(r['available_at_s'] >= start, 'capacity sample precedes sampling time')
        if r['physical_service_status'] == 'VERIFIED':
            _require(bool(r['provenance']['evidence_ref']) and r['provenance']['kind'] != 'synthetic_fixture', 'physical service assertion lacks non-synthetic evidence')
    elif kind == 'PerceptionResult':
        _require(r['actual_completion_time_s'] >= r['output_at_s'], 'result completion precedes detector output')
        if r['detection_output'] is not None:
            from .v2u4real_perception_interface_v1 import validate_detection_result
            validate_detection_result(r['detection_output'])
        if r['output_role'] == 'real_detector_output':
            _require(r['detection_output'] is not None and r['provenance']['kind'] not in ('mock_output', 'synthetic_fixture'), 'mock/test outputs cannot count as real detector results')
    return r


def sample_region_tasks(config, *, tick_index, generated_at_s, seed):
    """One Bernoulli trial per configured region at an explicit observation tick.

    A separate stable random stream for (seed, region ID, tick) prevents region
    order or another region's probability from changing this region's draw.
    Returns records/diagnostics only; repeated calls are not queue insertions.
    """
    c = validate_scene_config(config, require_complete=True)
    _require(isinstance(tick_index, int) and not isinstance(tick_index, bool) and tick_index >= 0, 'invalid tick index')
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, 'seed must be explicit nonnegative integer')
    _require(isinstance(generated_at_s, (int, float)) and not isinstance(generated_at_s, bool) and math.isfinite(generated_at_s), 'invalid generation time')
    target = tick_index * PERIOD_S
    _require(generated_at_s >= target, 'generation before observation boundary')
    provenance = dict(kind='simulation_generated', source_ref='regional Bernoulli contract fixture',
                      time_basis='simulation_seconds', physical_sync_status='NOT_APPLICABLE', evidence_ref=None)
    tasks, trials = [], []
    for region in sorted(c['parameters']['regions'], key=lambda x: x['region_id']):
        region_id = region['region_id']
        draw = random.Random(json.dumps([VERSION, seed, region_id, tick_index])).random()
        arrived = draw < region['arrival_probability']
        trials.append(dict(region_id=region_id, tick_index=tick_index, draw=draw, arrived=arrived))
        if arrived:
            latency = c['parameters']['allowed_total_latency_s']
            tasks.append(record('PerceptionTask', provenance, task_id=f'task:{region_id}:{tick_index}',
                                region_id=region_id, generated_at_s=generated_at_s,
                                target_observation_time_s=target, allowed_total_latency_s=latency,
                                absolute_deadline_s=target+latency, feature_packet_ids=[],
                                status='waiting_for_features', actual_completion_time_s=None))
    return dict(tasks=tasks, trials=trials, target_observation_time_s=target,
                scope='synthetic_contract_test_only; no queue/lifecycle update')


def validate_schedule_progress(actions, *, available_events=()):
    """Pure sequence check: advance time, or consume a distinct known event.

    Same-time event ordering is a testable interface option, not a frozen
    research ordering policy. A finite supplied event set prevents repeated
    same-clock actions manufactured from the same event.
    """
    events = {}
    for event in available_events:
        _require(set(event) == {'event_id', 'time_s', 'kind'}, 'invalid event fields')
        _require(event['kind'] in ('observation_available', 'service_feedback', 'timer'), 'predictions cannot manufacture actual scheduling events')
        _require(isinstance(event['time_s'], (int, float)) and not isinstance(event['time_s'], bool) and math.isfinite(event['time_s']) and event['time_s'] >= 0, 'invalid event time')
        _require(isinstance(event['event_id'], str) and bool(event['event_id']) and event['event_id'] not in events, 'invalid/duplicate event id')
        events[event['event_id']] = event
    used, action_ids, previous = set(), set(), None
    for value in actions:
        action = validate_record(value)
        _require(action['record_type'] == 'SchedulingAction', 'expected scheduling action')
        _require(action['action_id'] not in action_ids, 'duplicate scheduling action id')
        action_ids.add(action['action_id'])
        event_id = action['trigger_event_id']
        if event_id is not None:
            _require(event_id in events and event_id not in used, 'unknown or reused scheduling event')
            _require(events[event_id]['time_s'] <= action['decision_time_s'], 'event was not available at decision time')
            used.add(event_id)
        if previous is not None:
            _require(action['effective_time_s'] >= previous['effective_time_s'] and action['decision_time_s'] >= previous['decision_time_s'], 'schedule clock moved backwards')
            if _same(action['effective_time_s'], previous['effective_time_s']):
                _require(event_id is not None, 'same-clock decision requires a fresh known event')
        previous = action
    return True


def validate_record_graph(config, records):
    """Check identities, readiness and causality without mutating any state."""
    c = validate_scene_config(config, require_complete=True)
    rows = [validate_record(x) for x in records]
    groups = {k: {} for k in ID_FIELDS}
    all_ids = {}
    for r in rows:
        key = r[ID_FIELDS[r['record_type']]]
        _require(key not in all_ids, 'duplicate record identity')
        all_ids[key] = r
        groups[r['record_type']][key] = r
    sources = {x['source_id']: x for x in c['parameters']['sources']}
    regions = {x['region_id'] for x in c['parameters']['regions']}
    observations, tasks, packets, actions, services, results = (groups[k] for k in ID_FIELDS)
    for ob in observations.values():
        _require(ob['source_id'] in sources, 'unknown observation source')
        source = sources[ob['source_id']]
        _require(ob['sensor_id'] == source['sensor_id'] and ob['trajectory_origin'] == source['trajectory_origin'] and ob['trajectory_ref'] == source['trajectory_ref'], 'observation source/sensor/trajectory mismatch')
        _require(ob['provenance'] == source['data_provenance'], 'observation provenance differs from source catalog')
    for task in tasks.values():
        _require(task['region_id'] in regions, 'unknown task region')
        _require(_same(task['allowed_total_latency_s'], c['parameters']['allowed_total_latency_s']), 'task latency differs from explicit configuration')
        _require(set(task['feature_packet_ids']) == {p['packet_id'] for p in packets.values() if p['task_id'] == task['task_id']}, 'task packet references do not match packet ownership')
    for packet in packets.values():
        _require(packet['task_id'] in tasks and packet['observation_id'] in observations, 'missing task or observation reference')
        task, ob = tasks[packet['task_id']], observations[packet['observation_id']]
        _require(packet['source_id'] == ob['source_id'] and _same(ob['target_observation_time_s'], task['target_observation_time_s']), 'packet belongs to another source or observation tick')
        _require(ob['availability'] == 'available' and packet['preparation_started_at_s'] >= ob['available_at_s'], 'preparation before data availability')
        if packet['preparation_delay_source'] == 'configured_synthetic':
            _require(_same(packet['ready_at_s']-packet['preparation_started_at_s'], c['parameters']['feature_preparation_delay_s']), 'ready time differs from declared preparation parameter')
        expected = 'source_complete' if c['parameters']['feature_packet_granularity'] == 'source_complete' else 'fixed_spatial_block'
        _require(packet['granularity'] == expected, 'packet granularity differs from selected test profile')
        if expected == 'fixed_spatial_block':
            _require(packet['spatial_block_contract_ref'] == c['parameters']['spatial_block_contract_ref'], 'block layout differs from configuration')
    for action in actions.values():
        _require(action['decision_time_s'] >= 0, 'invalid action time')
        used_rbs = set()
        for allocation in action['allocations']:
            _require(allocation['packet_id'] in packets, 'selected packet does not exist')
            packet = packets[allocation['packet_id']]
            _require(packet['source_id'] == allocation['source_id'] and allocation['rsu_id'] == 'RSU_0', 'allocation endpoint mismatch')
            _require(packet['ready_at_s'] <= action['information_cutoff_s'] and tasks[packet['task_id']]['generated_at_s'] <= action['information_cutoff_s'], 'unprepared packet cannot enter transmission selection')
            _require(not used_rbs.intersection(allocation['rb_ids']), 'exclusive test policy forbids double allocation of RB')
            used_rbs.update(allocation['rb_ids'])
            _require(all(x < c['parameters']['rb_count'] for x in allocation['rb_ids']), 'RB index outside declared resource budget')
        for ref in action['information_refs']:
            _require(ref in all_ids, 'unknown information reference')
            fact = all_ids[ref]
            _require(fact['record_type'] in ('ObservationRecord', 'ServiceResult'), 'evaluation/labels cannot enter online information refs')
            if fact['record_type'] == 'ServiceResult':
                _require(fact['result_kind'] != 'predicted_service', 'model forecasts belong to prediction refs, not actual feedback')
            _require(fact['available_at_s'] is not None and fact['available_at_s'] <= action['information_cutoff_s'], 'future real observation or feedback at decision cutoff')
            if fact['record_type'] == 'ObservationRecord':
                _require(fact['target_observation_time_s'] <= action['information_cutoff_s'], 'future actual observation is not a prediction')
    link_service_ids = set()
    checked_services = []
    for service in services.values():
        _require(service['link_service_id'] not in link_service_ids, 'link capacity cannot be duplicated per packet')
        link_service_ids.add(service['link_service_id'])
        for earlier in checked_services:
            overlap = max(earlier['interval_start_s'], service['interval_start_s']) < min(earlier['interval_end_s'], service['interval_end_s'])
            _require(not overlap or not set(earlier['effective_rbs']).intersection(service['effective_rbs']), 'exclusive fixture service intervals reuse an RB')
        checked_services.append(service)
        _require(service['action_id'] in actions and service['packet_id'] in packets, 'service references missing action/packet')
        action, packet = actions[service['action_id']], packets[service['packet_id']]
        alloc = next((a for a in action['allocations'] if a['packet_id'] == service['packet_id']), None)
        _require(alloc is not None and service['source_id'] == packet['source_id'] == alloc['source_id'] and service['rsu_id'] == alloc['rsu_id'], 'service endpoint/packet allocation mismatch')
        _require(set(service['effective_rbs']) <= set(alloc['rb_ids']), 'service uses unallocated RB')
        _require(service['interval_start_s'] >= action['effective_time_s'], 'service precedes action effect')
    for result in results.values():
        _require(result['task_id'] in tasks, 'result task does not exist')
        task = tasks[result['task_id']]
        _require(set(result['used_packet_ids']) <= set(task['feature_packet_ids']), 'result consumes another task packet')
        _require(result['output_at_s'] >= task['generated_at_s'], 'detector output precedes task generation')
        receipts = {x['packet_id']: x for x in result['input_receipts']}
        _require(len(receipts) == len(result['input_receipts']) and set(receipts) == set(result['used_packet_ids']), 'each consumed packet needs a distinct upstream complete-input receipt')
        for pid in result['used_packet_ids']:
            receipt = receipts[pid]
            if receipt['evidence_ref'] in services:
                upstream = services[receipt['evidence_ref']]
                _require(upstream['result_kind'] == 'delivery_receipt', 'capacity diagnostic is not complete-input evidence')
                _require(upstream['packet_id'] == pid and upstream['delivered_bytes'] == packets[pid]['byte_count'], 'referenced single service does not deliver this whole packet')
                _require(upstream['available_at_s'] <= receipt['available_at_s'], 'input receipt predates upstream feedback')
            _require(receipt['received_byte_count'] == packets[pid]['byte_count'], 'incomplete input cannot be consumed as a complete packet')
            _require(packets[pid]['ready_at_s'] <= receipt['available_at_s'] <= result['output_at_s'], 'input receipt timing is inconsistent')
            if result['output_role'] == 'real_detector_output':
                _require(receipt['evidence_kind'] == 'runtime_receipt', 'fixture input is not runtime delivery evidence')
            _require(packets[pid]['ready_at_s'] <= result['output_at_s'], 'result uses unprepared feature')
            if result['output_role'] == 'real_detector_output':
                _require(packets[pid]['payload_role'] == 'encoded_lidar_feature', 'serialized mock detections are not real encoder features')
        if result['detection_output'] is not None:
            keys = {observations[packets[pid]['observation_id']]['frame_key'] for pid in result['used_packet_ids']}
            _require(result['detection_output']['frame_key'] in keys, 'legacy output frame identity mismatch')
    return dict(status='PASS', records=len(rows), lifecycle_executed=False, physical_sync_verified=False)


def packet_from_legacy_payload(payload, task, observation, *, ready_at_s):
    """Preserve R2-05B byte identity; serialized detections are not encoder features."""
    from .v2u4real_airfogsim_adapter_v1 import PerceptionPayloadRecord
    _require(isinstance(payload, PerceptionPayloadRecord), 'expected existing R2-05B payload')
    task, observation = validate_record(task), validate_record(observation)
    _require(task['record_type'] == 'PerceptionTask' and observation['record_type'] == 'ObservationRecord', 'legacy payload requires task and observation records')
    _require(payload.frame_key == observation['frame_key'] and payload.sensor_id == observation['sensor_id'], 'legacy payload frame/sensor mismatch')
    _require(payload.result_source.startswith('MockPerception.'), 'v1 legacy compatibility covers the audited mock payload only')
    provenance = dict(kind='mock_output', source_ref=payload.payload_id,
                      time_basis='simulation_seconds', physical_sync_status='NOT_APPLICABLE', evidence_ref=None)
    return record('FeaturePacket', provenance, packet_id=payload.payload_id, task_id=task['task_id'],
                  observation_id=observation['observation_id'], source_id=observation['source_id'],
                  byte_count=payload.byte_count, preparation_started_at_s=observation['available_at_s'],
                  ready_at_s=ready_at_s, preparation_delay_source='configured_synthetic', preparation_evidence_ref=None,
                  granularity='source_complete', block_id=None, spatial_block_contract_ref=None,
                  integrity_requirement='complete_packet', payload_role='legacy_detection_result_json',
                  payload_ref='canonical_detection_result_json_utf8:' + payload.payload_id)


def service_from_legacy_capacity(capacity, action, packet, *, service_id, sample_time_s):
    """R2-07 capacity adapter leaves delivered_bytes unset; no transmission runs."""
    action, packet = validate_record(action), validate_record(packet)
    _require(action['record_type'] == 'SchedulingAction' and packet['record_type'] == 'FeaturePacket', 'legacy service requires action and packet records')
    _require(capacity.get('quantity') == 'theoretical_single_slot_capacity' and capacity.get('delivery_status') == 'NOT_RUN', 'expected audited capacity diagnostic')
    _require(capacity.get('physical_validity_status') == 'BLOCKED' and capacity.get('rate_unit') == 'Mbps_decimal', 'legacy capacity physical/unit boundary changed')
    allocation = next((a for a in action['allocations'] if a['packet_id'] == packet['packet_id']), None)
    _require(allocation is not None, 'capacity packet is not allocated')
    active = capacity['active_rbs']
    effective = capacity['effective_rate_mbps']
    _require(set(active) == set(allocation['rb_ids']), 'legacy effective RBs differ from action')
    _require(isinstance(sample_time_s, (int, float)) and not isinstance(sample_time_s, bool) and math.isfinite(sample_time_s) and sample_time_s >= 0, 'explicit finite capacity sample time required')
    start = sample_time_s
    if 'sample_time_s' in capacity:
        _require(_same(start, capacity['sample_time_s']), 'explicit sample time differs from R2-07 receipt')
    _require(start >= action['effective_time_s'], 'sample precedes action effect')
    if 'source_id' in capacity:
        _require(capacity['source_id'] == allocation['source_id'] and capacity['target_id'] == allocation['rsu_id'], 'legacy capacity endpoint mismatch')
    provenance = dict(kind='simulation_generated', source_ref='R2-07 conditional single-slot capacity',
                      time_basis='simulation_seconds', physical_sync_status='NOT_APPLICABLE', evidence_ref=None)
    return record('ServiceResult', provenance, service_id=service_id, link_service_id=service_id,
                  action_id=action['action_id'], packet_id=packet['packet_id'], source_id=packet['source_id'],
                  rsu_id=allocation['rsu_id'], interval_start_s=start, interval_end_s=start+capacity['slot_s'],
                  available_at_s=start, effective_rbs=active, rate_values=[effective[i] for i in active],
                  rate_unit='Mbps', rate_semantics='instantaneous_sample_assumed_constant',
                  result_kind='capacity_diagnostic', theoretical_capacity_bytes=capacity['theoretical_bytes'],
                  delivered_bytes=None, delivery_evidence_ref=None, physical_service_status='BLOCKED')
