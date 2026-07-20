# PI-JWM v11 Selector Findings

## Frozen Evidence

- Local/remote source commit: `f7cb651c57262a1938e7b44ea43c3f3bbee12a44`.
- Formal run: 12 configurations x 3 training seeds x 20 epochs; 36 checkpoints.
- Ranked default validation active-rate RMSE: 233.7162005.
- Sample oracle validation RMSE: 105.3486359 (`sample_oracle`, headroom only).
- All CandidateSet configurations selected the default for every validation sample; nominal RMSE 233.7162005.
- Best classical deployable comparison was GB pairwise at 233.7057583, an immaterial 0.0104 improvement.
- Stage-only diagnostic policy selected on calibration reached validation RMSE 231.3377871 and improved 6/10 seeds. This is the current strongest interpretable diagnostic, not a frozen selector.
- Aggressive HGB pointwise rules overfit calibration and reached validation RMSE 234.5976.

## Current Question

Does the CandidateSet model fail because its candidate ordering is wrong, or because uncertainty calibration and/or the Pareto/defer rules suppress useful rankings?

The candidate-generation gate itself passed strongly on validation: sample-oracle RMSE 105.3486, nontrivial ratio 0.8369, identity oracle-win ratio 0.2575, and action-applied ratio 1.0. This isolates the immediate failure to candidate selection rather than candidate headroom.

## Required Attribution Policies

1. Formal z=1.64 plus Pareto.
2. z=0 plus Pareto.
3. Ensemble variance only, excluding predicted uncertainty.
4. Rank-only plus Pareto.
5. Rank-only without Pareto.
6. Improvement-only argmax.

Each policy must report global RMSE, execution/defer count, realized positive-benefit precision, negative-selection rate, per-seed RMSE, and improved-seed count.

## Formal Checkpoint Attribution

- The 12-config x 6-policy attribution completed on validation only.
- Rank-only plus Pareto produced RMSE 292.90-301.86; rank-only without Pareto produced RMSE 298.82-312.33. Every rank-only policy worsened all 10 validation seeds.
- Rank-only positive-benefit precision was only about 24%-29%, with roughly 47%-54% negative selections. The learned ordering itself is invalid.
- Removing aleatoric uncertainty or setting z=0 executed only a handful of candidates and improved at most 0.067 or 0.025 RMSE, respectively.
- The best improvement-head diagnostic was RMSE 233.5862, only 0.1300 better than default, with 16 active executions and 31.25% positive precision. A more aggressive improvement-head variant improved 7/10 seeds but only by 0.0939 global RMSE and had a 31.45% negative-selection rate.
- Root-cause decision: do not tune the defer threshold. The next method must replace the candidate ordering representation/objective.

## Schema-v6 Interaction Audit

- The formal schema-v6 cache contains 72 x 25 edge-step interaction tokens and 234 pooled interaction features for every sample-candidate pair.
- The completed CandidateSet ranker ignored both arrays and used only the 75 global candidate features plus context.
- A full train/calibration/validation HGB audit compared selected-edge, pooled-interaction-only, and full schema-v6 feature groups.
- Full schema-v6 learned opportunity detection well: validation opportunity ROC-AUC 0.8752 and PR-AUC 0.9495.
- Candidate ranking remained poor: sign PR-AUC 0.4754, sample rank Spearman 0.0832, top-1 positive ratio 0.2825, and no calibration threshold satisfied the safety gate.
- Root-cause hypothesis is now specific: opportunity detection is identifiable, while candidate benefit requires token-level local interaction encoding rather than global or hand-pooled statistics.

## Observable Episode Phase

- Every seed contains exactly 390 consecutive samples, and sample IDs follow `seed * 390 + local_step` on all unlocked splits.
- Current episode phase is therefore recoverable as `sample_id mod 390` without exposing seed identity or future state.
- Adding four phase terms (linear, squared, sine, cosine) to the full schema-v6 HGB audit changed validation RMSE from 233.7162 to 232.0230.
- The phase-aware audit executed six candidates, all six had positive realized benefit, and benefit Pearson correlation increased to 0.5775. This is evidence that task evolution position is a missing deployable context feature.

## Token Selector Probe

- A full-data CUDA smoke completed in 42 seconds and a 3-seed x 20-epoch probe completed in 205.8 seconds.
- The no-phase token model's loss decreased steadily, but calibration candidate sign probabilities did not separate safe actions: threshold 0.50 executed 41 candidates at 43.9% positive precision; threshold 0.65 executed none.
- The no-phase token probe therefore remains diagnostic-only and is retained as a controlled ablation.

## Phase-conditioned Benefit LCB

- Exact within-episode phase is substantially more stable than learned all-candidate ranking. A train-only phase table estimates each candidate's mean raw-SSE benefit, cross-seed positive direction rate, variance, and support count.
- Calibration alone selected the risk/defer rule. Validation selection never changed the calibrated thresholds.
- Enforcing deployable candidate masks and the observable task-energy Pareto gate produced validation active-rate RMSE 207.5399 versus 233.7162 default, a 26.1763 improvement.
- All 10 validation seeds improved. Of 65 active executions, 93.85% had positive realized benefit and 6.15% were negative; Pareto violations were zero.
- Link RMSE improved by 0.574%, and activity F1 dropped only 0.000086.
- The result is B-grade because it is in [200, 213.160874). It passes the general validation safety gate but does not meet the pre-registered <200 A-grade gate, so external seeds 60-69 remain locked.
- Token ranker, candidate-specific experts, learned residual routing, phase smoothing, estimator ensembles, and phase-restricted kNN all failed to improve over the exact phase table. These failed routes are retained as diagnostic evidence rather than hidden.
