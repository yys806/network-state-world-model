# STEP 6.4H S-CEM BUDGET QUALIFICATION

## Researcher authorization / Definition Basis

2026-10-08 researcher request: independent repaired-domain S-CEM K=4 rho=0.2 B512/B1024 qualification. This turn CPU only; stop before GPU. No algorithm/target/budget-rule changes. Parent baseline aea391f280e5933898cf1804d06179c0d9ec5f2f. Definition 06 Planner pure-search implementation boundary and 6.4F current Comm eligibility remain unchanged.

## Frozen contract

- 64 original Validation anchors, seeds6311–6315. 320 new S-CEM B512 cases; nominal163840 transitions. No new B1024, B256, method comparison or tuning.
- B1024: 320 read-only S-CEM K4 rho0.2 from accepted repaired 6.4G A; parent config a7c793ef7619e107fe7f3a5e3fdf16cffb8ae64af76b35422ee613fb69914db8. Per-file identity/SHA + ZIP content + acceptance lineage verified before freezing.
- Checkpoint SHA 941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9; train normalization/data/support and scientific source unchanged. H4 strict lexicographic (N_DDL,A_DDL,J_Delay,J_Burden,J_Effort). Route EXPLICIT_NOOP_ONLY, fallback CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1.
- GPU future launch: NVIDIA GeForce RTX 3080 Ti, CUDA FP32 batch16; no AMP/BF16/FP16/quantization/compile/bucketing; original CPU cache/prefix, prior mean and service expectation. Freeze commit HEAD=origin/main, tracked clean, Git LF blobs and executed bytes exact. 5GiB disk minimum recheck at launch; CPU disk measured, remote capacity/VRAM/device NOT MEASURED.
- Namespace code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2; 00 protocol, 01 cohort, 02 raw parent references, 03 config, 04 CPU receipt. Early uncommitted CPU draft in the unsuffixed namespace is not execution evidence, has no raw results, and is superseded by this r2 freeze.
- Linux flock plus old/duplicate formal-runner inspection; per-case identity hashed path, os.replace atomic JSON, resume validates every completed result before WM call. Wrong source/config/device/precision/batch/checkpoint/anchor/seed/method/budget refuses. STOP receipt prevents automatic restart. NaN/Inf/scorer/budget/source drift hard stop without retry.

## Preregistered statistics

320 strict pairs. Six mutually exclusive bins: only left, only right, both left better, both right better, both tie, neither. Unscoreable stays None. Win/tie/loss; 64 anchor clusters retain all5seed outcomes, bootstrap10000 seed6316, percentile95%CI. Independent oracle rechecks comparator, bins, first different component and bootstrap. Count first losses on N_DDL/A_DDL/J_Delay separately from Burden/Effort. Per-budget H4 complete/distinct/unscoreable, Return boundary, grammar, scorer, proposal/admission/rejection, actual unique/cache, mean/median/P90/P95/max/throughput. In-solve runtime from raw; case setup and matrix invocation elapsed separate.

## Qualification, not final research decision

PASS iff complete320 paired provenance/budget, scorer error/inconsistency0, identical scoreability sets, primary first-loss count0, mean and median in-solve B512 lower than B1024. Burden/Effort losses always reported. Failed gates mean FAIL, no threshold/sample/budget change. PASS recommends512; final budget always RESEARCHER_DECISION_PENDING. No equivalence/optimality/real closed-loop claim. 6.4G conditional stop remains immutable.

## Execution and backup

CPU: python code/scripts/step6_4h_s_cem_budget_v1.py (binding-only preflight). After separate user continuation and all remote gates: same script --execute --gate-commit <exact freeze commit>; --resume only strictly matching completed raw. Runner stops after320, produces raw inventory/ZIP/archive SHA and independent accepted summaries. Download all namespace files to D: preserving raw; run --verify-local to verify ZIP/raw/SHA and independently rebuild statistics. Never commit raw ZIP or large raw JSON. No cloud power action authorized in this turn.
