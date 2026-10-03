"""Acceptance gates and independent paired classification, without GPU search."""
import copy
import unittest
from close_step6_3d_validation_stage_a_v1 import validate_stage_a_cases, paired_oracle, runtime_clock_facts


class StageAAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.anchors=tuple(f'a{i:02d}' for i in range(64))
        self.execution={'execution_config_id':'fixture3080','execution_device':'cuda',
                        'gpu_model':'NVIDIA GeForce RTX 3080 Ti','batch_size':16,
                        'precision':'FP32','source_sha256':{'fixture':'sha'},'locked_test':False}
        self.rows=[dict(self.execution,sample_id=a,seed=s,method=m,budget=1024,
                        iterations=1 if m=='HRS' else 4,elite_ratio=None if m=='HRS' else .1,
                        split='dev_validation',training=False,closed_loop=False)
                   for a in self.anchors for s in range(6311,6316)
                   for m in ('HRS','S-CEM','MH-CEM')]

    def test_complete_identity_grid(self):
        self.assertEqual(len(validate_stage_a_cases(self.rows,self.anchors,self.execution)),960)

    def test_duplicate_cannot_replace_missing_case(self):
        rows=self.rows[:-1]+[copy.deepcopy(self.rows[0])]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            validate_stage_a_cases(rows,self.anchors,self.execution)

    def test_incomplete_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing|960'):
            validate_stage_a_cases(self.rows[:-1],self.anchors,self.execution)

    def test_wrong_execution_or_scientific_identity_is_rejected(self):
        for key,value in (('budget',512),('seed',6301),('sample_id','other'),
                          ('batch_size',32),('gpu_model','NVIDIA GeForce RTX 4090'),
                          ('source_sha256',{}),('locked_test',True),('elite_ratio',.2)):
            rows=copy.deepcopy(self.rows); rows[2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_stage_a_cases(rows,self.anchors,self.execution)

    def test_six_categories_and_lexicographic_priority(self):
        good=[0,999.,999.,999.,999.]; bad=[1,0.,0.,0.,0.]
        pairs=[(good,None),(None,good),(good,bad),(bad,good),(good,good),(None,None)]
        counts={}; outcomes=[]
        for left,right in pairs:
            category,value=paired_oracle(left,right)
            counts[category]=counts.get(category,0)+1; outcomes.append(value)
        self.assertEqual(set(counts),{'only_left_scoreable','only_right_scoreable',
            'both_scoreable_left_win','both_scoreable_right_win','both_scoreable_tie','both_unscoreable'})
        self.assertEqual(outcomes,[1,-1,1,-1,0,0])
        self.assertEqual(sum(counts.values()),6)

    def test_distinct_clock_intervals_are_preserved(self):
        facts=runtime_clock_facts({'invocation_start_utc':'2026-10-01T00:00:00+00:00',
            'invocation_finish_utc':'2026-10-01T00:00:10+00:00','invocation_elapsed_seconds':14.0})
        self.assertEqual(facts['actual_elapsed_seconds'],14.0)
        self.assertEqual(facts['utc_timestamp_interval_seconds'],10.0)
        self.assertEqual(facts['utc_minus_monotonic_seconds'],-4.0)


if __name__=='__main__': unittest.main()
