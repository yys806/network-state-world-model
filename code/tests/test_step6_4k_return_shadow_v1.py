"""Synthetic rule tests: never evidence of real simulator performance."""
import unittest
from pi_jwm.step6_4k_return_shadow_v1 import ReturnShadow, exact_completion


def task(**kw):
    return dict(task_id='T', current_node_id='A', return_destination_id='C',
                required_returned_size=2., task_cpu_work=1., computed_cpu_work=1.,
                lifecycle='waiting_to_return', **kw)


class ReturnShadowTests(unittest.TestCase):
    def setUp(self):
        self.shadow = ReturnShadow()
        self.nodes = {'A', 'B', 'C'}
        self.links = {('A', 'B'), ('B', 'C'), ('A', 'C')}

    def birth(self, t=None, route=('C',)):
        return self.shadow.establish(t or task(), route, now=1., source_time=1.,
                                     entities=self.nodes, links=self.links)

    def test_waiting_without_route_is_not_birth(self):
        self.assertEqual(self.shadow.snapshot()['logical_flow_rows'], [])
        with self.assertRaises(ValueError): self.birth(route=())

    def test_identity_idempotency_and_complete_retention(self):
        fid = self.birth()
        self.assertEqual(self.birth(), fid)
        self.shadow.transfer(fid, source='A', target='C', amount=2., hop_completed=True)
        row = self.shadow.snapshot()['logical_flow_rows'][0]
        self.assertEqual((row['flow_id'], row['epoch'], row['status'], row['presence']),
                         ('flow::T::Return::0', 0, 'COMPLETED', False))
        self.assertEqual(self.birth(), fid)
        self.assertEqual(len(self.shadow.snapshot()['logical_flow_rows']), 1)

    def test_multihop_conservation_and_reroute(self):
        fid = self.birth(route=('B', 'C'))
        self.shadow.transfer(fid, source='A', target='B', amount=2., hop_completed=True)
        self.assertEqual(self.shadow.snapshot()['logical_flow_rows'][0]['e2e_delivered'], 0.)
        self.shadow.transfer(fid, source='B', target='C', amount=2., hop_completed=True)
        self.assertEqual(self.shadow.snapshot()['logical_flow_rows'][0]['e2e_remaining'], 0.)

    def test_same_destination_reroute_keeps_epoch(self):
        fid = self.birth(route=('B', 'C'))
        self.assertEqual(self.birth(route=('C',)), fid)
        self.assertEqual(self.shadow.snapshot()['carrying_rows'][0]['route_revision'], 1)

    def test_reject_bad_route_future_terminal_and_unfinished(self):
        for route in [('B',), ('X',), ('A', 'C')]:
            with self.assertRaises(ValueError): self.birth(route=route)
        for lifecycle in ('computing', 'failed', 'done'):
            t=task(); t['lifecycle']=lifecycle
            with self.assertRaises(ValueError): self.birth(t)
        t=task(); t['computed_cpu_work']=1.-1e-16
        with self.assertRaises(ValueError): self.birth(t)
        with self.assertRaises(ValueError):
            self.shadow.establish(task(), ('C',), now=1., source_time=2., entities=self.nodes, links=self.links)

    def test_local_and_no_return_do_not_create(self):
        t=task(); t['return_destination_id']='A'
        self.assertIsNone(self.shadow.establish(t, ('A',), now=1., source_time=1., entities=self.nodes, links=self.links))
        t=task(); t['required_returned_size']=0.
        self.assertIsNone(self.birth(t))

    def test_overservice_and_missing_identity_fail_closed(self):
        fid=self.birth()
        with self.assertRaises(ValueError): self.shadow.transfer(fid, source='A', target='C', amount=3., hop_completed=True)
        with self.assertRaises(ValueError): self.shadow.transfer(fid, source='B', target='C', amount=1., hop_completed=False)
        with self.assertRaises(ValueError): ReturnShadow({'logical_flow_rows':[{'flow_id':'x'}], 'carrying_rows':[], 'flow_index':{}})

    def test_numeric_boundary_exact_not_epsilon(self):
        t=task(); t['computed_cpu_work']=1.-1e-16
        self.assertFalse(exact_completion(t))
        t['computed_cpu_work']=1.
        self.assertTrue(exact_completion(t))


if __name__ == '__main__': unittest.main()
