import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11DiffusionSupportGeneratorTest(unittest.TestCase):
    def test_diffusion_targets_include_positive_labels_and_top_rank(self):
        from compare_v11_diffusion_support_generator import make_diffusion_support_targets

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
        labels = np.array([0, 1, 0, 0, 0], dtype=np.int64)
        rank = np.array([0.9, 0.1, 0.8, 0.4, 0.2], dtype=np.float32)

        target = make_diffusion_support_targets(coords, labels, rank, target_top_k=1)

        self.assertEqual(target.tolist(), [1.0, 1.0, 0.0, 1.0, 0.0])

    def test_noisy_mask_respects_zero_noise(self):
        from compare_v11_diffusion_support_generator import make_noisy_mask

        target = torch.tensor([0.0, 1.0, 1.0, 0.0])
        noise = torch.zeros_like(target)

        noisy = make_noisy_mask(target, noise, false_positive_scale=0.25)

        self.assertTrue(torch.equal(noisy, target))

    def test_denoising_mlp_forward_shape(self):
        from compare_v11_diffusion_support_generator import DenoisingSupportMLP

        model = DenoisingSupportMLP(feature_dim=5, hidden_dim=8, dropout=0.0)
        features = torch.randn(7, 5)
        noisy = torch.zeros(7)
        noise_level = torch.full((7,), 0.5)

        logits = model(features, noisy, noise_level)

        self.assertEqual(tuple(logits.shape), (7,))


if __name__ == '__main__':
    unittest.main()
