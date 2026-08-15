import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11BridgeCalibrationSweepTest(unittest.TestCase):
    def test_enumerate_sweep_configs_expands_checkpoint_decoder_and_scales(self):
        from sweep_v11_bridge_calibration import enumerate_sweep_configs

        configs = enumerate_sweep_configs(
            checkpoints=["best", "last"],
            action_decoders=["threshold"],
            value_decoders=["train_median", "policy"],
            value_scales=[0.75, 1.0],
            budget_quantiles=[0.5],
        )

        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0].checkpoint_name, "best")
        self.assertEqual(configs[0].action_decoder, "threshold")
        self.assertEqual(configs[0].value_decoder, "train_median")
        self.assertEqual(configs[0].value_scale, 0.75)

    def test_config_slug_is_stable_and_filesystem_safe(self):
        from sweep_v11_bridge_calibration import SweepConfig

        config = SweepConfig(
            checkpoint_name="val_bin_accuracy",
            action_decoder="val_quantile_topk",
            value_decoder="train_median_step_scaled",
            value_scale=1.25,
            budget_quantile=0.75,
        )

        self.assertEqual(
            config.slug(),
            "val_bin_accuracy__val_quantile_topk__train_median_step_scaled__scale_1p25__bq_0p75",
        )

    def test_extract_bridge_metrics_reads_validation_and_test_values(self):
        from sweep_v11_bridge_calibration import extract_bridge_metrics

        payload = {
            "val": {
                "active_rate": {"active_rmse": 10.0},
                "activity": {"f1": 0.2},
                "link_rate": {"rmse": 3.0},
            },
            "test": {
                "active_rate": {"active_rmse": 12.0},
                "activity": {"f1": 0.25},
                "link_rate": {"rmse": 4.0},
            },
        }

        metrics = extract_bridge_metrics(payload)

        self.assertEqual(metrics["val_active_rate_rmse"], 10.0)
        self.assertEqual(metrics["test_active_rate_rmse"], 12.0)
        self.assertEqual(metrics["val_activity_f1"], 0.2)
        self.assertEqual(metrics["test_link_rmse"], 4.0)

    def test_rank_results_prefers_lower_validation_rmse_then_test_rmse(self):
        from sweep_v11_bridge_calibration import rank_results

        rows = [
            {"name": "worse", "val_active_rate_rmse": 5.0, "test_active_rate_rmse": 1.0},
            {"name": "tie_better_test", "val_active_rate_rmse": 3.0, "test_active_rate_rmse": 4.0},
            {"name": "best", "val_active_rate_rmse": 3.0, "test_active_rate_rmse": 2.0},
        ]

        ranked = rank_results(rows)

        self.assertEqual([row["name"] for row in ranked], ["best", "tie_better_test", "worse"])

    def test_should_run_respects_existing_output_and_overwrite(self):
        from sweep_v11_bridge_calibration import should_run

        tmp = PROJECT_ROOT / "artifacts" / "tmp_test_sweep_existing.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"ok": True}), encoding="utf-8")
        try:
            self.assertFalse(should_run(tmp, overwrite=False))
            self.assertTrue(should_run(tmp, overwrite=True))
            self.assertTrue(should_run(tmp.with_name("missing.json"), overwrite=False))
        finally:
            tmp.unlink(missing_ok=True)

    def test_build_bridge_command_includes_scale_and_budget_quantile(self):
        from sweep_v11_bridge_calibration import SweepConfig, build_bridge_command

        command = build_bridge_command(
            python_executable="python",
            bridge_script=Path("scripts/evaluate_v10_policy_bridge.py"),
            world_experiment_dir=Path("world"),
            world_checkpoint=Path("world/checkpoints/best.pt"),
            policy_checkpoint=Path("policy.pt"),
            output_json=Path("out.json"),
            device="cpu",
            batch_size=16,
            config=SweepConfig(
                checkpoint_name="best",
                action_decoder="val_quantile_topk",
                value_decoder="train_q75",
                value_scale=1.25,
                budget_quantile=0.75,
            ),
        )

        self.assertIn("--value-scale", command)
        self.assertIn("1.25", command)
        self.assertIn("--budget-quantile", command)
        self.assertIn("0.75", command)


if __name__ == "__main__":
    unittest.main()
