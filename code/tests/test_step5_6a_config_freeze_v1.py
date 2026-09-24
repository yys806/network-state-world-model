"""CPU-only checks for the researcher-approved formal protocol and sampler."""
from __future__ import annotations

from pathlib import Path
import unittest

from pi_jwm.step5_2_training_loop_v1 import Step52TrainingConfig
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FormalTrajectorySampler
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1


MANIFEST = Path(__file__).parents[1] / "artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"


class FormalConfigFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = formal_training_config_v1(MANIFEST)
        cls.interface = FormalTrainingInterface.from_manifest(MANIFEST)

    def test_exact_budget_stage_and_horizon_boundaries(self) -> None:
        p = self.protocol
        c = p.training
        self.assertEqual((4416, 1104, 8, 552), (p.train_windows, p.validation_windows, c.batch_size, p.steps_per_epoch))
        self.assertEqual((10, 5520), (c.max_epochs, c.max_steps))
        for step, epoch, stage, horizon in (
            (0, 0, "posterior_assisted_warmup", 1),
            (551, 0, "posterior_assisted_warmup", 1),
            (552, 1, "prior_dominant_recursive", 1),
            (1103, 1, "prior_dominant_recursive", 1),
            (1104, 2, "prior_dominant_recursive", 2),
            (2207, 3, "prior_dominant_recursive", 2),
            (2208, 4, "prior_dominant_recursive", 4),
            (5519, 9, "prior_dominant_recursive", 4),
        ):
            self.assertEqual((p.epoch_for_step(step), p.stage_for_step(step), p.horizon_for_step(step)),
                             (epoch, stage, horizon))
        with self.assertRaises(ValueError):
            p.epoch_for_step(5520)

    def test_kl_intervals_and_optimizer_contract(self) -> None:
        p = self.protocol
        c = p.training
        self.assertEqual((5601, 3e-4, 0.0, 1.0, "cuda"),
                         (c.seed, c.learning_rate, c.weight_decay, c.gradient_clip_norm, c.device))
        self.assertEqual([0.0, 0.5, 1.0, 1.0], [c.kl_schedule.beta_at(s) for s in (0, 552, 1104, 2208)])
        self.assertEqual(0.1, c.kl_schedule.free_bits)
        self.assertEqual({"betas": [0.9, 0.999], "eps": 1e-8}, p.effective_adamw_defaults())
        self.assertEqual((False, True, False, True), tuple(p.validation_due(s) for s in (0, 1104, 1656, 5520)))
        self.assertEqual((False, True, True), tuple(p.latest_checkpoint_due(s) for s in (0, 552, 5520)))
        m = p.as_manifest()
        self.assertEqual("argmin L_Val", m["checkpoint"]["selector"])
        self.assertEqual("save_on_strict_L_Val_improvement", m["checkpoint"]["best_policy"])
        self.assertEqual(3, m["checkpoint"]["patience_full_validations"])
        self.assertEqual(("fp32", False, False, False),
                         (m["precision"], m["mixed_precision"], m["formal_training_started"], m["locked_test_accessed"]))
        self.assertEqual((5201, 1, 1e-3, "cpu"),
                         (Step52TrainingConfig().seed, Step52TrainingConfig().batch_size,
                          Step52TrainingConfig().learning_rate, Step52TrainingConfig().device))

    def test_sampler_exact_once_validation_isolation_and_resume(self) -> None:
        interface = self.interface
        sampler = FormalTrajectorySampler(interface.samples, interface.train_indices, seed=self.protocol.training.seed)
        first = sampler.order_for_epoch(0)
        self.assertEqual((4416, 4416), (len(first), len(set(first))))
        self.assertEqual(set(interface.train_indices), set(first))
        self.assertTrue(set(first).isdisjoint(interface.validation_indices))
        self.assertEqual(first, FormalTrajectorySampler(interface.samples, interface.train_indices, seed=5601).order_for_epoch(0))
        self.assertNotEqual(first, sampler.order_for_epoch(1))
        for step in (0, 551, 552, 1104, 5519):
            epoch, within = divmod(step, 552)
            expected = sampler.order_for_epoch(epoch)[within * 8:(within + 1) * 8]
            self.assertEqual(expected, tuple(sampler.batch_for_global_step(step, 8)))


if __name__ == "__main__":
    unittest.main()
