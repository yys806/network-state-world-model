"""Separate synthetic evaluator evidence from real train-only schema evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import audit_v2u4real_r2_05a_baseline_data_eval_v1 as audit


class OfficialEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.paths = audit.verify_sources()
        cls.f, cls.official, cls.provenance = audit.load_audit_functions(cls.paths)

    def test_verified_source_functions_are_unmodified(self):
        self.assertTrue(self.provenance)
        self.assertTrue(all(not p['ast_body_modified'] for p in self.provenance))

    def test_changed_source_fingerprint_is_rejected(self):
        payload = json.loads(audit.SOURCE_MANIFEST.read_text(encoding='utf8'))
        payload['files'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'source.json'
            p.write_text(json.dumps(payload), encoding='utf8')
            with patch.object(audit, 'SOURCE_MANIFEST', p), self.assertRaises(audit.InterfaceBlocked):
                audit.verify_sources()

    def test_output_cannot_overwrite_prior_step_or_data(self):
        for p in (audit.ROOT / 'code/artifacts/data', audit.OUT.parent / 'v2u4real_r2_04c_20261027'):
            with self.assertRaises(audit.InterfaceBlocked):
                audit.run(p)

    def test_unapproved_key_or_agent_is_rejected_before_read(self):
        for key, agent in (('../val/sample', '1'), ('1752814590.8', '1'), (audit.KEYS[0], '3')):
            with self.assertRaises(audit.InterfaceBlocked):
                audit.fixed_input_paths(key, agent)

    def test_xy_identical_nonoverlapping_height_is_still_tp(self):
        s = audit.empty_stat()
        self.f.calculate_tp_fp(torch.stack([audit.corner_box(z=100)]), torch.tensor([.9]),
                               torch.stack([audit.corner_box()]), s, .5)
        self.assertEqual(s[.5]['tp'], [1])
        self.assertEqual(s[.5]['fp'], [0])

    def test_duplicate_predictions_match_gt_once(self):
        s = audit.empty_stat()
        self.f.calculate_tp_fp(torch.stack([audit.corner_box(), audit.corner_box()]), torch.tensor([.9, .8]),
                               torch.stack([audit.corner_box()]), s, .5)
        self.assertEqual(s[.5]['tp'], [1, 0])
        self.assertEqual(s[.5]['fp'], [0, 1])
        self.assertEqual(s[.5]['gt'], 1)

    def test_iou_threshold_equality_matches(self):
        s = {1.: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        self.f.calculate_tp_fp(torch.stack([audit.corner_box()]), torch.tensor([.9]),
                               torch.stack([audit.corner_box()]), s, 1.)
        self.assertEqual(s[1.]['tp'], [1])

    def test_no_predictions_still_accumulates_ground_truth(self):
        s = audit.empty_stat()
        self.f.calculate_tp_fp(None, None, torch.stack([audit.corner_box()]), s, .5)
        self.assertEqual(s[.5]['gt'], 1)
        self.assertEqual(self.f.calculate_ap(s, .5, False)[0], 0.)

    def test_default_frame_order_differs_from_global_score_order(self):
        s = {.5: {'tp': [0, 1], 'fp': [1, 0], 'gt': 1, 'score': [.1, .9]}}
        self.assertEqual(self.f.calculate_ap(copy.deepcopy(s), .5, False)[0], .5)
        self.assertEqual(self.f.calculate_ap(copy.deepcopy(s), .5, True)[0], 1.)

    def test_default_ap_mutates_statistics_and_reuse_is_not_idempotent(self):
        s = {.5: {'tp': [0, 1, 1], 'fp': [1, 0, 0], 'gt': 2, 'score': [.9, .8, .7]}}
        before = copy.deepcopy(s)
        first = self.f.calculate_ap(s, .5, False)[0]
        self.assertNotEqual(s, before)
        second = self.f.calculate_ap(s, .5, False)[0]
        self.assertNotEqual(first, second)

    def test_nonempty_predictions_with_zero_gt_raise(self):
        with self.assertRaises(ZeroDivisionError):
            self.f.calculate_ap({.5: {'tp': [0], 'fp': [1], 'gt': 0, 'score': [.9]}}, .5, False)

    def test_nms_suppresses_strictly_above_threshold(self):
        b = torch.stack([audit.corner_box(), audit.corner_box()])
        self.assertEqual(self.f.nms_rotated(b, torch.tensor([.9, .8]), 1.).tolist(), [0, 1])
        self.assertEqual(self.f.nms_rotated(b, torch.tensor([.9, .8]), .99).tolist(), [0])

    def test_native_category_filter_is_exact_car_only(self):
        raw = {'location': [0., 0., 0.], 'center': [0., 0., 0.], 'extent': [4., 2., 2.], 'angle': [0., 0., 0.]}
        objects = {name: dict(raw, obj_type=name) for name in ('Car', 'Truck', 'Cyclist', 'car')}
        r = [-100., -100., -100., 100., 100., 100.]
        owner = SimpleNamespace(params={'GT_RANGE': r, 'anchor_args': {'cav_lidar_range': r}, 'order': 'hwl', 'max_num': 100})
        _, _, ids = self.f.generate_object_center(owner, [{'params': {'vehicles': objects}}], np.eye(4))
        self.assertEqual(ids, ['Car'])

    def test_extent_is_consumed_as_full_corner_span(self):
        p = self.official.create_bbx([4., 2., 2.])
        np.testing.assert_array_equal(p.max(0) - p.min(0), [4., 2., 2.])


class RealFiveFrameSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='v2u4real_r2_05a_', dir=audit.OUT.parent)
        cls.result = audit.run(Path(cls.temp.name))
        cls.fields = json.loads((Path(cls.temp.name) / 'field_schema.json').read_text(encoding='utf8'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_five_pairs_twenty_original_hashes(self):
        self.assertEqual([f['frame_key'] for f in self.fields['frames']], list(audit.KEYS))
        self.assertEqual(len(self.fields['input_hashes']), 20)
        self.assertTrue(all(h['unchanged'] for h in self.fields['input_hashes']))

    def test_all_inputs_stay_in_fixed_train_scene(self):
        for row in self.fields['input_hashes']:
            self.assertIn('/selected_samples/train/' + audit.SCENE + '/', row['path'])
            self.assertNotIn('/val/', row['path'])
            self.assertNotIn('/test/', row['path'])

    def test_actual_pcd_schema_and_xyz_finiteness(self):
        for frame in self.fields['frames']:
            for side in frame['sides'].values():
                self.assertEqual(side['pcd_header']['FIELDS'], ['x', 'y', 'z', 'rgb'])
                self.assertTrue(side['xyz_finite'])
                self.assertEqual(side['xyz_shape'][1], 3)
                self.assertGreater(side['xyz_shape'][0], 0)

    def test_policy_b_retained_without_changing_native_code(self):
        total = 0
        for frame in self.fields['frames']:
            row = frame['annotation_path_comparison']
            for oid in row['shared_ids_priority_conflicts']:
                self.assertEqual(row['policy_B_sources'][oid], 'uav')
                self.assertEqual(row['simulated_native_first_occurrence_sources_before_class_and_range_filters'][oid], 'vehicle')
                total += 1
        self.assertEqual(total, 8)

    def test_schema_success_never_authorizes_training_or_physical_claim(self):
        self.assertEqual(self.result['formal_training_admission'], 'BLOCKED')
        self.assertEqual(self.result['baseline_readiness_status'], 'NEEDS_RESEARCHER_DECISION')
        self.assertEqual(self.result['physical_alignment_status'], 'NOT_ESTABLISHED')
        self.assertEqual(self.result['real_detector_metrics'], 'NOT_RUN')
        self.assertFalse(self.result['runtime']['gpu_used'])
        self.assertFalse(self.result['runtime']['val_test_samples_read'])


if __name__ == '__main__':
    unittest.main()
