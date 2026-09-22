import unittest
import torch

from pi_jwm.step5_1b_posterior_loss_metric_v1 import (
    Step5_1BConfig, TargetEncoder, FuturePosterior, diagonal_gaussian_kl,
    complete_total_loss, motion_metrics, csi_metrics,
)


class Step51BTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = Step5_1BConfig(target_dim=4, csi_dim=3, latent_dim=4, hidden_dim=8, d_h=6, d_encoder=5, d_z=4)

    def test_per_horizon_and_mask_evidence(self):
        enc = TargetEncoder(self.cfg)
        value = torch.zeros(1, 2, 1, 4)
        a = enc.motion(value, torch.tensor([[[[True] * 4], [[False] * 4]]]))
        b = enc.motion(value, torch.tensor([[[[False] * 4], [[False] * 4]]]))
        self.assertEqual(tuple(a.shape), (1, 2, 1, 5))
        self.assertFalse(torch.equal(a[:, 0], b[:, 0]))
        self.assertTrue(torch.equal(a[:, 1], torch.zeros_like(a[:, 1])))

    def test_temporal_isolation_and_family_parameter_separation(self):
        enc, teacher = TargetEncoder(self.cfg), FuturePosterior(self.cfg)
        motion = torch.zeros(1, 2, 1, 4); mask = torch.ones_like(motion, dtype=torch.bool)
        base = enc.motion(motion, mask)
        csi = enc.csi(torch.zeros(1, 2, 1, 3), torch.ones(1, 2, 1, 3, dtype=torch.bool))
        q1 = teacher(torch.zeros(1, 2, 1, 6), torch.zeros(1, 2, 1, 6), base, csi)["physical"]
        motion[:, 1] = 9; changed = enc.motion(motion, mask)
        q2 = teacher(torch.zeros(1, 2, 1, 6), torch.zeros(1, 2, 1, 6), changed, csi)["physical"]
        self.assertTrue(torch.equal(base[:, 0], changed[:, 0])); self.assertTrue(torch.equal(q1.mean[:, 0], q2.mean[:, 0]))
        self.assertIsNot(teacher.phy_future_posterior.mean.weight, teacher.comm_future_posterior.mean.weight)

    def test_per_dimension_free_bits_and_horizon_eligibility(self):
        q = torch.zeros(1, 2, 1, 4); p = torch.ones_like(q); eligible = torch.tensor([[[True], [False]]])
        result = diagonal_gaussian_kl(q, q, p, q, eligible, free_bits=0.5)
        self.assertEqual(tuple(result.per_dim_raw.shape), (1, 2, 1, 4)); self.assertEqual(int(result.count), 1)
        zero = diagonal_gaussian_kl(q, q, q, q, eligible, free_bits=0.0)
        self.assertTrue(torch.equal(zero.raw, zero.adjusted))

    def test_total_loss_family_balance_and_empty_counts(self):
        shape = (1, 2, 1, 4); z = torch.zeros(shape); m = torch.ones(shape, dtype=torch.bool); empty = torch.zeros_like(m)
        q = (torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4)); p = (torch.ones(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
        out = complete_total_loss(z, torch.ones_like(z), m, z, torch.ones_like(z), empty, q, p, m[..., 0], q, p, m[..., 0], beta_kl=0.2)
        self.assertTrue(torch.all(out["motion_available"])); self.assertFalse(torch.any(out["csi_available"]))
        self.assertTrue(torch.allclose(out["L_Pred"], 0.5 * out["L_Mot"]))

    def test_raw_unit_metrics(self):
        pred = torch.zeros(1, 2, 1, 4); target = torch.ones_like(pred); mask = torch.ones_like(pred, dtype=torch.bool)
        mm = motion_metrics(pred, target, mask, xyz_aggregate=True)
        self.assertEqual(mm["units"], ("m", "m", "m", "m/s")); self.assertIn("delta_x_mae", mm); self.assertNotIn("overall_mae", mm)
        cm = csi_metrics(torch.zeros(1, 2, 1, 3), torch.ones(1, 2, 1, 3), torch.ones(1, 2, 1, 3, dtype=torch.bool))
        self.assertEqual(cm["unit"], "dB"); self.assertEqual(tuple(cm["valid_count"].shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
