import unittest

import torch

from pi_jwm.step5_1b_posterior_loss_metric_v1 import (
    Step5_1BConfig,
    TargetEncoder,
    PosteriorTeacher,
    PriorPredictor,
    masked_family_mse,
    normalized_prediction_loss,
    diagonal_gaussian_kl,
    motion_metrics,
    csi_metrics,
    validation_loss,
)


class Step51BTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = Step5_1BConfig(target_dim=4, csi_dim=3, latent_dim=5, hidden_dim=8)

    def test_target_encoders_are_family_specific_and_posterior_has_mean_sample(self):
        motion = torch.randn(2, 2, 3, 4)
        mmask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
        csi = torch.randn(2, 2, 4, 3)
        cmask = torch.ones(2, 2, 4, 3, dtype=torch.bool)
        enc = TargetEncoder(self.cfg)
        phy = enc.motion(motion, mmask)
        comm = enc.csi(csi, cmask)
        self.assertEqual(tuple(phy.shape), (2, 3, 5))
        self.assertEqual(tuple(comm.shape), (2, 4, 5))
        teacher = PosteriorTeacher(self.cfg)
        out = teacher(torch.zeros(2, 3, 5), phy)
        self.assertEqual(tuple(out.mean.shape), (2, 3, 5))
        self.assertEqual(tuple(out.sample.shape), (2, 3, 5))

    def test_prior_is_invariant_to_target_tampering(self):
        prior = PriorPredictor(self.cfg)
        h = torch.randn(2, 3, 5)
        torch.manual_seed(11); a = prior(h).mean
        tampered_target = torch.randn(2, 2, 3, 4) * 1000
        torch.manual_seed(11); b = prior(h).mean
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(tampered_target.shape), (2, 2, 3, 4))

    def test_masked_mse_empty_mask_and_normalized_bridge(self):
        pred = torch.tensor([[[1.0, 3.0], [2.0, 5.0]]])
        target = torch.tensor([[[0.0, 1.0], [4.0, 5.0]]])
        mask = torch.tensor([[[True, False], [False, True]]])
        value, count = masked_family_mse(pred, target, mask)
        self.assertAlmostEqual(float(value), 0.5, places=6)
        self.assertEqual(int(count), 2)
        empty, empty_count = masked_family_mse(pred, target, torch.zeros_like(mask))
        self.assertEqual(float(empty), 0.0)
        self.assertEqual(int(empty_count), 0)
        raw = torch.tensor([[[3.0, 7.0]]])
        stats = {"mean": [1.0, 5.0], "std": [2.0, 2.0]}
        loss, _ = normalized_prediction_loss(raw, raw, torch.ones_like(raw, dtype=torch.bool), stats)
        self.assertEqual(float(loss), 0.0)

    def test_kl_free_bits_and_masks(self):
        qmu = torch.zeros(1, 2, 2)
        qls = torch.zeros_like(qmu)
        pmu = torch.ones_like(qmu)
        pls = torch.zeros_like(qmu)
        mask = torch.tensor([[True, False]])
        raw, adjusted, count = diagonal_gaussian_kl(qmu, qls, pmu, pls, mask, free_bits=0.5)
        self.assertGreater(float(raw), 0.0)
        self.assertGreaterEqual(float(adjusted), 0.5)
        self.assertEqual(int(count), 1)
        identity, identity_adjusted, _ = diagonal_gaussian_kl(qmu, qls, qmu, qls, mask, free_bits=0.0)
        self.assertEqual(float(identity), 0.0)
        self.assertEqual(float(identity_adjusted), 0.0)

    def test_metrics_and_validation_aggregate(self):
        pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        target = torch.tensor([[[[2.0, 2.0, 1.0, 6.0]]]])
        mask = torch.ones_like(pred, dtype=torch.bool)
        mm = motion_metrics(pred, target, mask, xyz_aggregate=True)
        self.assertAlmostEqual(float(mm["mae"][0, 0]), 1.25)
        cs = csi_metrics(torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2, dtype=torch.bool), mean=0.0, std=2.0)
        self.assertAlmostEqual(float(cs["rmse"][0, 0]), 2.0)
        self.assertAlmostEqual(float(validation_loss(torch.tensor([1.0, 3.0]))), 2.0)


if __name__ == "__main__":
    unittest.main()
