# PI-JWM v11 Selector Progress Log

## 2026-07-20

- Resumed from completed RTX 4090 formal run.
- Confirmed repository HEAD is `f7cb651` and the worktree started clean.
- Confirmed formal output directory and checkpoint loading/selection APIs exist locally.
- Started systematic attribution of the 100% defer failure before changing any selector behavior.
- Confirmed all 36 checkpoint files and the formal summary are present locally.
- Confirmed the candidate gate passed while the selector gate failed only on RMSE and improved-seed count.
- First attribution launch stopped before loading data because `pi_jwm` was not on the piped Python process path; no experiment result was produced.
- A second launch with a literal Unicode path failed at the same pre-load import boundary; attribution data and checkpoints remain untouched.
- The environment probe proved `PYTHONPATH` works. A third launch then exposed stdin code-page corruption of the Chinese `代码` literal; it stopped before model loading. The launcher is being changed to an ASCII-only body with a full code-root environment variable.
- Completed the formal checkpoint attribution on validation only and wrote CSV/JSON artifacts under `selector_attribution_task_bridge_h10_f7cb651`.
- Attribution shows rank failure, not merely conservative defer: rank-only RMSE is 292.90-312.33 and all validation seeds worsen.
- Ran the existing schema-v6 benefit identifiability audit with HGB on selected-edge, pooled-interaction, and full interaction feature groups.
- Opportunity detection generalized strongly, but candidate ordering did not; the next architecture will separate the opportunity gate from token-level candidate ranking.
- Added the token-level ranker, opportunity-masked loss, deterministic mini-batch fit/predict, safe selection, and a train/calibration/validation-only runner using TDD.
- Local gate passed: 85 relevant selector tests, Python compilation, and diff checks.
