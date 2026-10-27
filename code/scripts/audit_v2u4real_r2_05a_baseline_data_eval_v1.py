"""Read-only CPU audit; no detector, label export, or val/test sample reads."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from shapely.geometry import Polygon
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, UniqueKeyLoader, field_inventory, read_pcd_xyz, sha256
from audit_v2u4real_r2_04a_single_frame_geometry_v1 import official_functions, strict_vector, verify_source
from audit_v2u4real_r2_04c_label_merge_policy_v1 import KEYS, SCENE, DATASET_VERSION, merge_official_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code/artifacts/audit/v2u4real_r2_05a_20261027'
COMMIT = 'e099a3a746c6092a614e18b74d1f5b2f3c8351e8'
SOURCE_MANIFEST = OUT / 'supplemental_source_manifest.json'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + chr(10), encoding='utf8', newline=chr(10))


def verify_sources():
    verify_source()
    m = json.loads(SOURCE_MANIFEST.read_text(encoding='utf8'))
    if m['repository'] != 'https://github.com/VjiaLi/V2U4Real' or m['commit'] != COMMIT or m['git_tree_sha'] != COMMIT:
        raise InterfaceBlocked('fixed supplemental source identity mismatch')
    paths = {}
    for row in m['files']:
        p = ROOT / row['local_path']
        if not p.resolve().is_relative_to(ROOT.resolve()):
            raise InterfaceBlocked('source escapes workspace')
        payload = p.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + bytes([0]) + payload).hexdigest()
        if sha256(p) != row['sha256'] or blob != row['git_blob']:
            raise InterfaceBlocked('fixed source changed: ' + row['official_path'])
        paths[row['official_path']] = p
    return m, paths


def source_locations(paths):
    result = []
    for rel, p in paths.items():
        if p.suffix != '.py':
            continue
        for n in ast.parse(p.read_text(encoding='utf8')).body:
            nodes = [(n.name, n)] if isinstance(n, ast.FunctionDef) else []
            if isinstance(n, ast.ClassDef):
                nodes += [(n.name + '.' + x.name, x) for x in n.body if isinstance(x, ast.FunctionDef)]
            result += [dict(official_path=rel, local_path=p.relative_to(ROOT).as_posix(), symbol=name,
                            start_line=x.lineno, end_line=x.end_lineno) for name, x in nodes]
    return result


def load_audit_functions(paths):
    ns = {'np': np, 'torch': torch, 'Polygon': Polygon}
    provenance = []

    def load(rel, names, owner=None):
        p = paths[rel]
        tree = ast.parse(p.read_text(encoding='utf8'))
        parent = tree if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        nodes = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise InterfaceBlocked('required function missing: ' + rel)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
        provenance.extend(dict(path=rel, function=n.name, start_line=n.lineno, end_line=n.end_lineno,
                               sha256=sha256(p), ast_body_modified=False) for n in nodes)

    helpers = ['compute_iou', 'convert_format', 'torch_tensor_to_numpy', 'check_numpy_to_torch', 'rotate_points_along_z']
    load('opencood/utils/common_utils.py', helpers)
    ns['common_utils'] = SimpleNamespace(**{k: ns[k] for k in helpers})
    load('opencood/utils/eval_utils.py', ['voc_ap', 'calculate_tp_fp', 'calculate_ap'])
    official, _ = official_functions()
    ns.update(x1_to_x2=official.x1_to_x2, create_bbx=official.create_bbx, corner_to_center=official.corner_to_center)
    load('opencood/utils/box_utils.py', ['boxes_to_corners_3d', 'mask_boxes_outside_range_numpy',
         'project_world_objects', 'nms_rotated', 'get_mask_for_boxes_within_range_torch'])
    ns['box_utils'] = SimpleNamespace(project_world_objects=ns['project_world_objects'])
    load('opencood/data_utils/post_processor/base_postprocessor.py', ['generate_object_center'], 'BasePostprocessor')
    return SimpleNamespace(**ns), official, provenance


def corner_box(cx=0., z=0.):
    bottom = [[cx-1, -1, z-1], [cx+1, -1, z-1], [cx+1, 1, z-1], [cx-1, 1, z-1]]
    return torch.tensor(bottom + [[x, y, z+1] for x, y, _ in bottom], dtype=torch.float64)


def empty_stat():
    return {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}


def evaluator_probes(f):
    same, gt = torch.stack([corner_box(z=100)]), torch.stack([corner_box()])
    elevated, duplicate = empty_stat(), empty_stat()
    f.calculate_tp_fp(same, torch.tensor([.9]), gt, elevated, .5)
    f.calculate_tp_fp(torch.stack([corner_box(), corner_box()]), torch.tensor([.9, .8]), gt, duplicate, .5)
    ordering = {0.5: {'tp': [0, 1], 'fp': [1, 0], 'gt': 1, 'score': [.1, .9]}}
    mutation = {0.5: {'tp': [0, 1, 1], 'fp': [1, 0, 0], 'gt': 2, 'score': [.9, .8, .7]}}
    before = copy.deepcopy(mutation)
    first = f.calculate_ap(mutation, .5, False)[0]
    after = copy.deepcopy(mutation)
    second = f.calculate_ap(mutation, .5, False)[0]
    zero_gt = 'NO_EXCEPTION'
    try:
        f.calculate_ap({0.5: {'tp': [0], 'fp': [1], 'gt': 0, 'score': [.9]}}, .5, False)
    except ZeroDivisionError:
        zero_gt = 'ZeroDivisionError'
    return {'evidence_type': 'SYNTHETIC_OFFICIAL_FUNCTION_PROBES_NOT_DETECTOR_RESULTS',
        'elevated_box': {'z_center_separation': 100, 'tp': elevated[.5]['tp'], 'fp': elevated[.5]['fp'],
            'official_polygon_iou': float(f.compute_iou(f.convert_format(same.numpy())[0], f.convert_format(gt.numpy()))[0])},
        'duplicate_prediction_one_gt': duplicate[.5],
        'ordering_probe': {'frame_concatenation_AP': f.calculate_ap(copy.deepcopy(ordering), .5, False)[0],
                           'global_score_sort_AP': f.calculate_ap(copy.deepcopy(ordering), .5, True)[0]},
        'mutable_default_AP': {'first_AP': first, 'second_AP_same_state': second, 'state_after_first': after,
                               'input_stat_mutated': before != after},
        'nonempty_prediction_zero_gt': zero_gt, 'real_detection_metrics': 'NOT_RUN'}


def fixed_input_paths(key, agent):
    if key not in KEYS or agent not in ('1', '2'):
        raise InterfaceBlocked('only the five fixed train keys and agents 1/2 are authorized')
    base = ROOT / 'code/artifacts/data/v2u4real_r2_01/selected_samples/train' / SCENE / agent
    paths = {'yaml': base / 'yaml/ouster' / (key + '.yaml'), 'pcd': base / 'ouster' / (key + '.pcd')}
    if any(not p.resolve().is_relative_to(base.resolve()) for p in paths.values()):
        raise InterfaceBlocked('fixed input symlink escape')
    return paths


def inspect_five_frames(f, official):
    previous = ROOT / 'code/artifacts/audit'
    reference = json.loads((previous / 'v2u4real_r2_04b_20261026/result.json').read_text(encoding='utf8'))
    policy = json.loads((previous / 'v2u4real_r2_04c_20261027/result.json').read_text(encoding='utf8'))
    if (reference['dataset_version'] != DATASET_VERSION or tuple(reference['fixed_keys']) != KEYS
            or policy['policy_decision'] != 'B' or policy['label_merge_status'] != 'PASS'):
        raise InterfaceBlocked('prior fixed evidence identity mismatch')
    expected = {r['frame_key']: r for r in reference['frames']}
    frames, hashes = [], []
    lidar_range = [-100.8, -80, -4, 100.8, 80, 4]
    native = SimpleNamespace(params={'GT_RANGE': lidar_range, 'anchor_args': {'cav_lidar_range': lidar_range},
                                    'order': 'hwl', 'max_num': 100})
    for key in KEYS:
        row = {'frame_key': key, 'sides': {}}
        annotations, converted = {}, {}
        for side, agent in (('vehicle', '1'), ('uav', '2')):
            paths = fixed_input_paths(key, agent)
            identities = {}
            for kind, path in paths.items():
                before = sha256(path)
                if before != expected[key]['inputs'][side][kind]['sha256_before']:
                    raise InterfaceBlocked('input differs from R2-04B: ' + str(path))
                identities[kind] = dict(path=path.relative_to(ROOT).as_posix(), sha256=before,
                                        bytes=path.stat().st_size, read_status='PASS')
                hashes.append(dict(path=path.relative_to(ROOT).as_posix(), sha256_before=before))
            xyz, header = read_pcd_xyz(paths['pcd'])
            raw = yaml.load(paths['yaml'].read_text(encoding='utf-8-sig'), Loader=UniqueKeyLoader)
            strict_vector(raw['lidar_pose'], 6, side + '/lidar_pose')
            for oid, obj in raw['vehicles'].items():
                if not isinstance(oid, str) or not isinstance(obj['obj_type'], str):
                    raise InterfaceBlocked('unexpected raw object ID/category type')
                for field in ('location', 'center', 'extent', 'angle'):
                    strict_vector(obj[field], 3, side + '/' + oid + '/' + field)
            annotations[side] = raw
            row['sides'][side] = dict(inputs=identities, pcd_header=header, xyz_shape=list(xyz.shape),
                xyz_finite=bool(np.isfinite(xyz).all()), root_keys=list(raw), field_inventory=field_inventory(raw),
                lidar_pose_raw=raw['lidar_pose'], raw_objects=raw['vehicles'], object_ids=list(raw['vehicles']),
                obj_type_counts={c: sum(o['obj_type'] == c for o in raw['vehicles'].values())
                                 for c in sorted({o['obj_type'] for o in raw['vehicles'].values()})})
        t = official.x1_to_x2(annotations['uav']['lidar_pose'], annotations['vehicle']['lidar_pose'])
        native_centers = {}
        for side, agent, transform in (('vehicle', '1', np.eye(4)), ('uav', '2', t)):
            converted[side] = official.load_annotation_objects(str(fixed_input_paths(key, agent)['yaml']), transform=transform)
            centers, mask, ids = f.generate_object_center(native, [{'params': annotations[side]}], transform)
            native_centers[side] = {'retained_ids_after_native_Car_filter_and_range': ids,
                                    'native_hwl_centers': centers[mask == 1].tolist()}
        merged, provenance = merge_official_policy(converted['vehicle'], converted['uav'])
        ids = list(converted['vehicle']) + list(converted['uav'])
        row['annotation_path_comparison'] = {
            'interpretation': 'official_code_interpretation_only', 'native_field_path': native_centers,
            'policy_B_ids': list(merged), 'policy_B_sources': {oid: p['final_source'] for oid, p in provenance.items()},
            'priority_comparison_scope': 'mapping_priority_simulation_before_class_and_range_filters_not_full_dataset_execution',
            'simulated_native_first_occurrence_sources_before_class_and_range_filters': {
                oid: ('vehicle' if oid in converted['vehicle'] else 'uav') for oid in sorted(set(ids))},
            'shared_ids_priority_conflicts': [oid for oid in converted['vehicle'] if oid in converted['uav']],
            'input_instance_count': len(ids), 'output_object_count': len(merged)}
        frames.append(row)
    for item in hashes:
        item['sha256_after'] = sha256(ROOT / item['path'])
        item['unchanged'] = item['sha256_before'] == item['sha256_after']
    if not all(h['unchanged'] for h in hashes):
        raise InterfaceBlocked('raw input changed during audit')
    return dict(dataset_version=DATASET_VERSION, scene=SCENE, fixed_keys=list(KEYS),
        scope='five train pairs only; no population statistics inferred', frames=frames,
        input_hashes=hashes, input_count=len(hashes), all_inputs_unchanged=True)


def inspect_configs(paths):
    result = []
    for rel, p in paths.items():
        if rel.startswith('opencood/hypes_yaml/') and p.suffix == '.yaml':
            d = yaml.safe_load(p.read_text(encoding='utf8'))
            result.append(dict(path=rel, source_path=p.relative_to(ROOT).as_posix(), sha256=sha256(p),
                train_root=d.get('root_dir'), validation_root=d.get('validate_dir'), fusion=d.get('fusion'),
                model_core=d.get('model', {}).get('core_method'), preprocess=d.get('preprocess'),
                postprocess=d.get('postprocess'), wild_setting=d.get('wild_setting'), train_params=d.get('train_params')))
    return result


def audited_contracts(paths):
    """Audit findings, not an executable training/evaluation policy approval."""
    def evidence(rel, symbol=None, lines=None):
        return dict(official_path=rel, local_path=paths[rel].relative_to(ROOT).as_posix(),
                    sha256=sha256(paths[rel]), symbol=symbol, lines=lines)

    evaluation = dict(
        status='DRAFT_AWAITING_RESEARCHER_REVIEW', official_commit=COMMIT,
        evidence_type='SOURCE_AUDIT_AND_SYNTHETIC_FUNCTION_PROBES_NOT_DETECTOR_PERFORMANCE',
        output_boxes=dict(model_order='hwl', center_format=['x', 'y', 'z', 'h', 'w', 'l', 'yaw_radians'],
                          evaluation_shape=['N', 8, 3], confidence='sigmoid anchor objectness; no class argument in evaluator'),
        iou=dict(dimension='BEV_XY_POLYGON_AREA', corners='first four XY corners', ignores_height=True,
                 evaluation_thresholds=[0.25, 0.5, 0.7], tp_comparison='IoU >= threshold',
                 paper_conflict='Paper calls table metrics 3D AP; fixed evaluator computes XY polygon IoU'),
        matching=dict(scope='per frame', order='descending prediction confidence',
                      rule='greedy maximum IoU among remaining GT; matched GT removed', class_aware=False),
        ap=dict(method='VOC2010 precision envelope integrated at all recall changes',
                recall='cumulative TP / total GT', precision='cumulative TP / (cumulative TP + cumulative FP)',
                global_sort_default=False, default_order='frame-local score sort followed by dataset iteration order',
                global_sort_option='--global_sort_detections', scope=['all loaded samples', 'each scenario'],
                scene_average=False, category_map='UNSPECIFIED_NOT_IMPLEMENTED',
                equal_score_tie_policy='UNSPECIFIED', default_mutates_tp_fp=True,
                repeated_evaluation_requires='fresh statistics or deep copy',
                zero_gt_with_predictions='ZeroDivisionError in fixed code; future empty-set policy requires approval'),
        other_metrics=dict(precision_recall='intermediate cumulative arrays, not one frozen operating-point result',
                           fn='not directly emitted; GT minus TP is a possible derived quantity',
                           f1='UNSPECIFIED_NOT_IMPLEMENTED', map='UNSPECIFIED_NOT_IMPLEMENTED'),
        point_pillar_postprocess=dict(score_threshold=0.2, score_comparison='>',
            nms=dict(type='rotated BEV polygon NMS', threshold=0.15, suppress_comparison='>', pre_max_size=1000),
            prediction_filters=['XY standup spans <= 6', 'z_min >= -3 and z_max <= 1',
                                'all eight corners inside GT_RANGE after NMS'],
            height_filter_caveat='remove_large_pred_bbx passes z_len as logical operand, not an explicit height limit',
            threshold_scope='point_pillar early/late/intermediate YAML only; other configs differ'),
        train_target_assignment=dict(order='hwl', iou='axis-aligned XY standup-box overlaps',
            positive_threshold=0.6, positive_comparison='>', negative_threshold=0.45, negative_comparison='< for every GT',
            forced_positive='per-GT highest-overlap anchor if overlap > 0',
            regression='normalized xyz, log dimension ratios, yaw difference; not evaluation matching'),
        ranges=dict(train_val_gt=[-100.8, -80, -4, 100.8, 80, 4], inference_test_gt=[-15, -80, -4, 100, 80, 4],
                    point_pillar_input=[-100.8, -80, -4, 100.8, 80, 4],
                    point_boundary='strict inequalities', box_boundary='inclusive, all eight corners'),
        splits=dict(paper_ratio='4:1:1 train:val:test', root_key='root_dir', evaluation_root_key='validate_dir',
                    sample_read_scope='five fixed train pairs only',
                    caveat='train=False selects validate_dir; mode=test selects GT_RANGE, not a separate test directory'),
        label_policy=dict(frozen='B: UAV overwrites shared IDs; retain both unique sides; no ID normalization',
                          native_conflict='early/intermediate and GT dedup keep first occurrence (ego first)',
                          native_class_filter='exact obj_type == Car', native_id_output_caveat='generate_gt_bbx_with_id casts output IDs to int'),
        task_quality_interface=dict(J_per='mean of Q_k(D_k); overdue task contributes zero only under candidate definition',
            official_direct_support=False, required_decisions=['task-to-frame/ROI/class association',
            'Qk definition and fixed matching/score/IoU thresholds', 'deadline and last-valid-result semantics',
            'empty GT/prediction cases', 'measured delivery/completion time or explicitly authorized simulation'],
            timestamp_policy='file keys are association keys, not independently verified physical timestamps'),
        unresolved=['paper/code angle order and complete field units', 'physical alignment NOT_ESTABLISHED',
                    'Car-only vs paper Vehicle/Cyclist classes', 'paper ego-priority vs frozen B/native priority',
                    'BEV vs volume-3D metric claim', 'AP sorting convention and empty-set behavior'],
        evidence=[evidence('opencood/utils/eval_utils.py', 'voc_ap/calculate_tp_fp/calculate_ap'),
                  evidence('opencood/utils/common_utils.py', 'compute_iou/convert_format', [189, 229]),
                  evidence('opencood/tools/inference.py', 'test_parser/main'),
                  evidence('opencood/data_utils/datasets/__init__.py', 'build_dataset', [19, 53]),
                  evidence('opencood/data_utils/post_processor/voxel_postprocessor.py', 'generate_label/post_process'),
                  evidence('opencood/data_utils/post_processor/base_postprocessor.py', 'generate_object_center/generate_gt_bbx_with_id')])
    common_blockers = ['researcher scope decision', 'same final policy-B GT for all methods',
                       'class/ROI/metric convention decision', 'field and coordinate limitations accepted explicitly',
                       'future approved adapter and CPU pipeline tests; no training authorization in this task']
    candidates = dict(status='DRAFT_AWAITING_RESEARCHER_REVIEW', official_commit=COMMIT,
        selection_frozen=False, formal_training_admission='BLOCKED',
        recommended_scope='LiDAR 3D-perception module only if researcher authorizes; not complete visual multi-car/RSU system',
        common_fairness='same GT provenance, class mapping, ROI, evaluator/sort/thresholds and split; only input availability differs',
        candidates=[
            dict(id='vehicle_only_point_pillar', input='vehicle Ouster only',
                official_model_exists=True, dedicated_vehicle_only_config_found=False,
                official_reproducibility='COMPONENTS_EXIST_NOT_READY_END_TO_END',
                reference_config='opencood/hypes_yaml/point_pillar_late_fusion.yaml',
                caveat='LateFusionDataset trains on a randomly selected agent and tests multiple agents; not a ready vehicle-only experiment',
                new_code_needed='approved input restriction and common policy-B GT adapter; not implemented', blockers=common_blockers),
            dict(id='early_fusion_point_pillar', input='vehicle plus transformed UAV raw point clouds',
                official_model_exists=True, official_config_exists=True,
                official_reproducibility='OFFICIAL_PIPELINE_PRESENT_NOT_EXECUTED_AND_POLICY_B_INCOMPATIBLE',
                reference_config='opencood/hypes_yaml/point_pillar_early_fusion.yaml',
                new_code_needed='approved policy-B/common-GT adapter and class decisions; not implemented', blockers=common_blockers),
            dict(id='intermediate_fusion_reference', input='LiDAR-derived feature tensors from each agent',
                official_config_exists=True, reference_config='opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml',
                official_reproducibility='REFERENCE_ONLY_NOT_EXECUTED',
                caveat='Feature fusion existence does not verify visual tokens, real radio delivery or multi-round scheduling',
                new_code_needed='policy-B/common-GT adapter plus future system interface; not implemented', blockers=common_blockers)],
        evidence=[evidence('opencood/models/point_pillar.py', 'PointPillar'),
                  evidence('opencood/data_utils/datasets/late_fusion_dataset.py', 'LateFusionDataset'),
                  evidence('opencood/data_utils/datasets/early_fusion_dataset.py', 'EarlyFusionDataset'),
                  evidence('opencood/tools/inference.py', 'main', [48, 128])])
    return evaluation, candidates


def run(output=OUT):
    if output.resolve().parent != OUT.parent.resolve() or not output.name.startswith('v2u4real_r2_05a_'):
        raise InterfaceBlocked('output must be a distinct R2-05A audit directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest, paths = verify_sources()
    f, official, executed = load_audit_functions(paths)
    probes = evaluator_probes(f)
    fields = inspect_five_frames(f, official)
    write_json(output / 'field_schema.json', fields)
    write_json(output / 'evaluator_probes.json', probes)
    write_json(output / 'configuration_inventory.json', inspect_configs(paths))
    write_json(output / 'source_locations.json', source_locations(paths))
    evaluation, candidates = audited_contracts(paths)
    write_json(output / 'evaluation_contract.json', evaluation)
    write_json(output / 'baseline_candidates.json', candidates)
    result = dict(schema_version='PIJWM-R2-05A-v1', official_commit=COMMIT,
        source_file_count_verified=len(manifest['files']), executed_official_functions=executed,
        scenario_alignment_status='AUDITED_CRITICAL_ADAPTATION_GAPS', source_audit_status='PASS_WITH_DOCUMENTED_CONFLICTS',
        field_schema_status='PASS_CODE_CONSUMPTION_ONLY', label_contract_status='B_PRESERVED_NATIVE_PATH_CONFLICT',
        evaluation_audit_status='PASS_WITH_DOCUMENTED_CONFLICTS', baseline_readiness_status='NEEDS_RESEARCHER_DECISION',
        formal_training_admission='BLOCKED', physical_alignment_status='NOT_ESTABLISHED',
        task_execution_status='COMPLETED_AUDIT_ONLY', raw_input_count=fields['input_count'],
        all_input_hashes_unchanged=fields['all_inputs_unchanged'], real_detector_metrics='NOT_RUN',
        evidence=['field_schema.json', 'evaluator_probes.json', 'configuration_inventory.json',
                  'source_locations.json', 'baseline_candidates.json', 'evaluation_contract.json'],
        runtime=dict(device='cpu', torch=torch.__version__, detector_instantiated=False, training_started=False,
            gpu_used=False, val_test_samples_read=False, locked_test_accessed=False,
            formal_labels_written=False, next_step_started=False))
    write_json(output / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k.endswith('status')}, ensure_ascii=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args().output)
