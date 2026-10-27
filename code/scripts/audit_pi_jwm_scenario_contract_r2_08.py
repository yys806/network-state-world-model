"""CPU-only R2-08 acceptance. No dataset, simulator run or packet lifecycle."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/tests')]
sys.dont_write_bytecode = True
from pi_jwm.scenario_contract_v1 import sample_region_tasks, validate_record_graph, validate_scene_config
from test_scenario_contract_v1 import fixture_config, fixture_records

DEFAULT_OUTPUT = ROOT / 'code/artifacts/audit/pi_jwm_r2_08_scenario_contract_20261027'
NEW_PYTHON = ['code/src/pi_jwm/scenario_contract_v1.py',
              'code/tests/test_scenario_contract_v1.py',
              'code/scripts/audit_pi_jwm_scenario_contract_r2_08.py']
CONFIGS = ['code/scripts/configs/' + f for f in (
    'pi_jwm_scenario_v1.schema.json', 'pi_jwm_scenario_v1.pending.json',
    'pi_jwm_scenario_v1.synthetic_test.json', 'pi_jwm_scenario_interfaces_v1.schema.json')]
UPSTREAM = ['code/src/pi_jwm/v2u4real_airfogsim_adapter_v1.py',
            'code/src/pi_jwm/v2u4real_perception_interface_v1.py',
            'code/src/pi_jwm/airfogsim_service_interface_v1.py',
            'code/src/pi_jwm/v2u4real_interface_v1.py']
NOTES = [Path('D:/shen/OB/科研/PKU/PIJWM/PIJWM/新场景下训练前任务书.md'),
         Path('D:/shen/OB/科研/PKU/PIJWM/组会笔记/1028组会.md')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(out, name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def protected_paths():
    # Only vendored source and earlier audit receipts. Never enumerate raw data.
    paths = [ROOT / p for p in git('ls-files', '--', 'code/reference/AirFogSim').splitlines() if p]
    prior = json.loads((ROOT / 'code/artifacts/audit/v2u4real_airfogsim_r2_07_20261027/source_identity.json').read_text(encoding='utf-8'))
    for name, expected in prior['vendor_files'].items():
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError('inherited vendor evidence changed: ' + name)
        paths.append(path)
    for folder in ('v2u4real_r2_05b_20261027', 'v2u4real_r2_06_20261027', 'v2u4real_airfogsim_r2_07_20261027'):
        paths.extend(p for p in (ROOT / 'code/artifacts/audit' / folder).rglob('*') if p.is_file())
    return sorted(set(p for p in paths if p.is_file()))


def run(out):
    out = out.resolve()
    if not out.is_relative_to(ROOT / 'code/artifacts/audit'):
        raise ValueError('output must be a fresh repository audit directory')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError('preserve prior evidence; use a new --output directory')
    protected = protected_paths()
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in protected}
    source_identity = dict(baseline_git_head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                           files={p:digest(ROOT / p) for p in NEW_PYTHON + CONFIGS + UPSTREAM},
                           read_only_authority={str(p):digest(p) for p in NOTES})
    save(out, 'source_identity.json', source_identity)
    save(out, 'runtime_environment.json', dict(python=sys.version, executable=sys.executable,
          platform=platform.platform(), jsonschema=importlib.metadata.version('jsonschema'),
          installed_dependencies_changed=False, device='cpu'))
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for target in ('test_scenario_contract_v1', 'test_airfogsim_service_interface_r2_07',
                   'test_v2u4real_r2_05b_interface_v1.AirFogSimIngressTests'):
        suite.addTests(loader.loadTestsFromName(target))
    stream = io.StringIO()
    test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out / 'unit_regression_tests.log').write_text(stream.getvalue(), encoding='utf-8', newline='\n')
    compile_rows = []
    for f in NEW_PYTHON:
        try:
            compile((ROOT / f).read_text(encoding='utf-8'), f, 'exec')
            compile_rows.append(dict(path=f, status='PASS'))
        except Exception as e:
            compile_rows.append(dict(path=f, status='BLOCKED', error=repr(e)))
    save(out, 'compile_check.json', dict(method='in-memory compile; no pycache writes', files=compile_rows))
    c, records = fixture_records()
    graph = validate_record_graph(c, records)
    original = deepcopy(records)
    trace = []
    for tick in range(6):
        arrival = sample_region_tasks(c, tick_index=tick, generated_at_s=tick*.1+.02, seed=20261027)
        trace.append(dict(observation_index=tick, **arrival,
                          old_task_snapshot=deepcopy(records[1]),
                          retention_scope='caller retains records; no lifecycle engine'))
    assert original == records
    pending = json.loads((ROOT / CONFIGS[1]).read_text(encoding='utf-8'))
    validate_scene_config(pending)
    from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked
    try:
        validate_scene_config(pending, require_complete=True)
        pending_gate = 'UNEXPECTED_ACCEPTANCE'
    except InterfaceBlocked as e:
        pending_gate = 'BLOCKED_AS_REQUIRED'
        save(out, 'pending_execution_rejection.json', dict(status=pending_gate, reason=str(e)))
    save(out, 'synthetic_contract_trace.json', dict(configuration=c, records=records, graph=graph,
                                                   arrivals=trace, old_task_records_unchanged=original == records,
                                                   delivery_receipts_are_supplied_fixtures_not_generated=True))
    after = {p.relative_to(ROOT).as_posix():digest(p) for p in protected}
    protection = dict(status='PASS' if before == after else 'BLOCKED', before=before, after=after,
                      raw_dataset_accessed=False, vendor_modified=False, prior_receipts_modified=False)
    save(out, 'protection_check.json', protection)
    passed = test_result.wasSuccessful() and all(x['status']=='PASS' for x in compile_rows) and before==after and pending_gate=='BLOCKED_AS_REQUIRED'
    result = dict(step='R2-08', contract_version='PIJWM-Scenario-Contract-v1',
                  status='PASS_CONTRACT_ONLY' if passed else 'BLOCKED',
                  checks=dict(unit_and_targeted_regression=dict(status='PASS' if test_result.wasSuccessful() else 'BLOCKED',
                        tests_run=test_result.testsRun, failures=len(test_result.failures), errors=len(test_result.errors),
                        log='unit_regression_tests.log'), configuration_and_record_graph=graph,
                        pending_formal_configuration=pending_gate, compile_check=compile_rows,
                        historical_and_vendor_preservation=protection['status']),
                  pending_researcher_decisions=c['pending_researcher_decisions'],
                  boundary=dict(dataset_accessed=False, gpu_used=False, training_started=False,
                      validation_test_locked_test_accessed=False, simulator_instantiated=False,
                      real_physical_synchronization='BLOCKED_INHERITED_R2_06',
                      physical_radio_service='BLOCKED_INHERITED_R2_07',
                      multi_slot_lifecycle='NOT_RUN', actual_delivery='NOT_RUN', real_detector='NOT_RUN',
                      research_parameter_freeze='BLOCKED_PENDING_RESEARCHER',
                      task4_authorized_or_started=False),
                  source_identity='source_identity.json', artifacts=['synthetic_contract_trace.json','protection_check.json',
                      'unit_regression_tests.log','compile_check.json','runtime_environment.json','pending_execution_rejection.json'])
    save(out, 'result.json', result)
    print(json.dumps(dict(status=result['status'], tests=test_result.testsRun,
                         protected_files=len(before), output=str(out)), ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    raise SystemExit(run(parser.parse_args().output))
