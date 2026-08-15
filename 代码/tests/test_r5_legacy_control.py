from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r3_preflight_data import (  # noqa: E402
    load_r3_window,
    make_explicit_batch,
    read_trajectory_index,
    select_r3_windows,
)
from pi_jwm.r4_objective import compute_r4_objective  # noqa: E402
from pi_jwm.r5_legacy_control import LegacyDirectedResidualBackend  # noqa: E402
from pi_jwm.r5_module_confirmation import build_confirmation_model  # noqa: E402


class R5LegacyControlTest(unittest.TestCase):
    def test_j_uses_current_contract_and_backpropagates_on_real_window(self) -> None:
        dataset = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        evaluation = CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        normalization = json.loads(
            (evaluation / "evaluation_normalization_stats.json").read_text(encoding="utf-8")
        )
        rows = read_trajectory_index(dataset)
        window = select_r3_windows(
            dataset,
            rows,
            split="train",
            horizons=(5,),
            history_steps=8,
            per_horizon=1,
            seed=20260806,
        )[0]
        batch = make_explicit_batch(load_r3_window(window), normalization)
        model = build_confirmation_model("J", hidden_dim=4, history_steps=8)
        self.assertIsInstance(model.backend, LegacyDirectedResidualBackend)
        output = model(batch, rollout_steps=5)
        objective = compute_r4_objective(output, batch)
        objective.total.backward()
        tensors = [*output.predicted_explicit.values(), *output.predicted_logits.values()]
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in tensors))
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertTrue(any(bool(torch.count_nonzero(value)) for value in gradients))
        self.assertEqual(
            output.predicted_explicit["information_edge_state"].shape[1],
            5,
        )

        backend = model.backend
        belief = backend.infer_belief(batch)
        first, _ = backend._predict(belief)
        second, _ = backend._predict(belief)
        self.assertTrue(
            all(torch.equal(first[name], second[name]) for name in first),
            "legacy residual predictions must keep the last-observed anchor fixed",
        )


if __name__ == "__main__":
    unittest.main()
