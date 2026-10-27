"""R2-07 synthetic post-step observations and single-slot service diagnostics.

No propagation correction, packet accumulation, deadline or delivery semantics.
Native entity timestamps remain visible rather than being overwritten.
"""
from __future__ import annotations

import math
import numpy as np

from .v2u4real_interface_v1 import InterfaceBlocked


def observe_post_step(env):
    """Return position sample time after an equal-interval synthetic env.step.

    Traffic snapshots are the time authority for positions, not the lagging
    native entity update label. Only the explicitly exercised contract is accepted.
    """
    t = float(env.traffic_manager.getCurrentTime())
    dt = float(env.traffic_interval)
    if not math.isfinite(t) or not math.isclose(dt, env.simulation_interval, rel_tol=0, abs_tol=1e-10):
        raise InterfaceBlocked('only equal traffic/simulation intervals audited')
    if not math.isclose(t, env.simulation_time, rel_tol=0, abs_tol=1e-10):
        raise InterfaceBlocked('observation requires aligned post-step clocks')
    records = {}
    for kind, entities, infos in (
        ('vehicle', env.vehicles, env.traffic_manager.getVehicleTrafficInfos()),
        ('uav', env.UAVs, env.traffic_manager.getUAVTrafficInfos()),
    ):
        if set(entities) != set(infos):
            raise InterfaceBlocked('entity membership differs from traffic snapshot')
        for key, entity in entities.items():
            p = np.asarray(entity.getPosition(), dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all() or not np.allclose(p, infos[key]['position'], rtol=0, atol=1e-10):
                raise InterfaceBlocked('entity position differs from traffic snapshot')
            grid = env.traffic_manager.getMapIndexByNodeId(key)
            if grid == (None, None):
                raise InterfaceBlocked('entity has no grid membership')
            native_t = float(entity._last_updated_time)
            if not math.isfinite(native_t) or not math.isclose(t - native_t, dt, rel_tol=0, abs_tol=1e-10):
                raise InterfaceBlocked('native timestamp behavior differs from audited one-step lag')
            records[key] = dict(kind=kind, position_m=p.tolist(), grid=list(grid),
                                position_time_s=t, native_last_updated_time_s=native_t,
                                native_timestamp_lag_s=t-native_t)
    return dict(status='PASS', scope='synthetic_post_step_position_observation',
                observation_time_s=t, traffic_time_s=t, env_time_s=float(env.simulation_time),
                native_timestamp_repaired=False, entities=records)


def slot_service(rate_mbps, active_mask, slot_s):
    """Masked theoretical single-slot capacity; never an observed delivery.

    1 Mbps = 1e6 bit/s; 8 bits = 1 byte. Fractional bytes are retained.
    The caller must supply same-sample rates and allocation masks.
    """
    rate = np.asarray(rate_mbps, dtype=float)
    mask = np.asarray(active_mask)
    if rate.ndim != 1 or mask.shape != rate.shape or mask.dtype.kind != 'b':
        raise InterfaceBlocked('rate/mask must be matching 1D vectors and mask boolean')
    if not np.isfinite(rate).all() or (rate < 0).any():
        raise InterfaceBlocked('rate must be finite and nonnegative')
    if isinstance(slot_s, bool) or not math.isfinite(slot_s) or slot_s <= 0:
        raise InterfaceBlocked('slot duration must be positive finite seconds')
    effective = np.where(mask, rate, 0.)
    total = float(effective.sum())
    bits = total * 1e6 * slot_s
    if not math.isfinite(bits):
        raise InterfaceBlocked('service amount overflow')
    return dict(status='PASS', quantity='theoretical_single_slot_capacity',
                rate_unit='Mbps_decimal', slot_s=float(slot_s),
                active_rbs=np.flatnonzero(mask).tolist(), raw_rate_mbps=rate.tolist(),
                effective_rate_mbps=effective.tolist(), sum_effective_mbps=total,
                theoretical_bits=bits, theoretical_bytes=bits/8.,
                inactive_positive_count=int(np.count_nonzero((rate > 0) & ~mask)),
                delivery_status='NOT_RUN', physical_validity_status='BLOCKED')


def read_post_step_link_service(env, channel_type, source_id, target_id, allocated_rbs):
    """Read a single V2I/U2I link after same-sample native computeRate.

    Requires the caller's expected RBs to exactly match the native link mask.
    This checks indexing/allocation/units, not physical model validity or packets.
    """
    observation = observe_post_step(env)
    if channel_type not in ('V2I', 'U2I'):
        raise InterfaceBlocked('only V2I/U2I audited')
    source_ids = env.vehicle_ids_as_index if channel_type == 'V2I' else env.uav_ids_as_index
    if source_id not in source_ids or target_id not in env.rsu_ids_as_index:
        raise InterfaceBlocked('link endpoint missing from current index')
    cm = env.channel_manager
    expected_shape = (len(source_ids), len(env.rsu_ids_as_index), cm.n_RB)
    rates = np.asarray(getattr(cm, channel_type + '_Rate'))
    masks = np.asarray(getattr(cm, channel_type + '_active_links'))
    if rates.shape != expected_shape or masks.shape != expected_shape:
        raise InterfaceBlocked('rate/mask dimensions differ from current node index')
    rbs = list(allocated_rbs)
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= cm.n_RB for i in rbs) or len(set(rbs)) != len(rbs):
        raise InterfaceBlocked('invalid or duplicate RB indices')
    tx, rx = source_ids.index(source_id), env.rsu_ids_as_index.index(target_id)
    expected = np.zeros(cm.n_RB, dtype=bool)
    expected[rbs] = True
    if masks.dtype.kind != 'b' or not np.array_equal(masks[tx, rx], expected):
        raise InterfaceBlocked('native mask differs from expected allocation')
    result = slot_service(rates[tx, rx], masks[tx, rx], env.simulation_interval)
    result.update(channel_type=channel_type, source_id=source_id, target_id=target_id,
                  tx_index=tx, rx_index=rx, rate_shape=list(rates.shape),
                  sample_time_s=observation['observation_time_s'],
                  temporal_scope='instantaneous post-step rate times one hypothetical slot; not completed env.step service')
    return result
