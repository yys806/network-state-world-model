import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class V11RolloutValueCalibratorTest(unittest.TestCase):
    def test_hard_forward_returns_codebook_values_and_zeros_inactive_actions(self):
        from pi_jwm.v11_rollout_value_calibrator import RolloutAlignedValueCalibrator

        codebook = torch.tensor(
            [
                [[1.0, 2.0, 4.0], [5.0, 10.0, 20.0]],
                [[1.0, 3.0, 6.0], [4.0, 8.0, 16.0]],
            ]
        )
        model = RolloutAlignedValueCalibrator(horizon=2, action_dim=2, codebook_size=3, hidden_dim=8)
        base_value = torch.tensor([[[[2.2, 11.0]], [[5.5, 7.0]]]])
        activity_prob = torch.tensor([[[[0.9, 0.1]], [[0.8, 0.7]]]])
        active_mask = activity_prob >= 0.5

        output = model(base_value, activity_prob, codebook, active_mask, hard=True)

        self.assertEqual(tuple(output.shape), (1, 2, 1, 2))
        self.assertEqual(float(output[0, 0, 0, 1].detach()), 0.0)
        for step in range(2):
            for dim in range(2):
                value = float(output[0, step, 0, dim].detach())
                if bool(active_mask[0, step, 0, dim]):
                    self.assertIn(value, codebook[step, dim].tolist())

    def test_gradient_flows_only_to_calibrator_when_world_model_is_frozen(self):
        from pi_jwm.v11_rollout_value_calibrator import RolloutAlignedValueCalibrator, freeze_module

        calibrator = RolloutAlignedValueCalibrator(horizon=1, action_dim=2, codebook_size=3, hidden_dim=8)
        world_model = freeze_module(torch.nn.Linear(2, 1, bias=False))
        codebook = torch.tensor([[[1.0, 2.0, 4.0], [5.0, 10.0, 20.0]]])
        base_value = torch.tensor([[[[2.2, 11.0]]]])
        activity_prob = torch.tensor([[[[0.9, 0.8]]]])
        active_mask = torch.ones_like(activity_prob, dtype=torch.bool)

        calibrated = calibrator(base_value, activity_prob, codebook, active_mask, hard=True)
        loss = world_model(calibrated.reshape(-1, 2)).pow(2).mean()
        loss.backward()

        calibrator_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in calibrator.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(calibrator_grad, 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in world_model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in world_model.parameters()))

    def test_aggregate_loss_compares_edge_sums_for_selected_action_dimensions(self):
        from pi_jwm.v11_rollout_value_calibrator import compute_action_aggregate_loss

        predicted = torch.zeros((1, 1, 2, 6), dtype=torch.float32)
        target = torch.zeros_like(predicted)
        predicted[0, 0, :, 2] = torch.tensor([10.0, 20.0])
        target[0, 0, :, 2] = torch.tensor([15.0, 25.0])
        predicted[0, 0, :, 4] = torch.tensor([2.0, 4.0])
        target[0, 0, :, 4] = torch.tensor([1.0, 3.0])

        loss = compute_action_aggregate_loss(predicted, target, dims=(2, 4))

        expected = torch.tensor(((30.0 - 40.0) ** 2 + (6.0 - 4.0) ** 2) / 2.0)
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_calibrator_rejects_shape_mismatch(self):
        from pi_jwm.v11_rollout_value_calibrator import RolloutAlignedValueCalibrator

        model = RolloutAlignedValueCalibrator(horizon=2, action_dim=2, codebook_size=3, hidden_dim=8)
        with self.assertRaises(ValueError):
            model(
                torch.ones((1, 1, 1, 2)),
                torch.ones((1, 1, 1, 2)),
                torch.ones((2, 2, 3)),
                torch.ones((1, 1, 1, 2), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
