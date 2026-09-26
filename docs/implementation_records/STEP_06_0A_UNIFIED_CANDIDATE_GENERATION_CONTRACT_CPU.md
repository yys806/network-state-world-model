# STEP 6.0A — Unified Candidate Generation Contract & Static CPU Implementation

## 1. Research Problem / Step Goal

Provide one causal, typed candidate-sequence interface for three future generator families, and prove that it compiles to the formal training action tensors. This Step ends before candidate future prediction, scoring and selection.

## 2. Definition Basis, Initial State and Current Project Facts

Definition 06 sections 1–2/4, read-only SHA-256 `F20294BD8708076AE7583BE679A8182A0990ECF6777E05DF378AE4C2855431F1`; current action source `code/scripts/build_step5_1d_unified_model_chain_v1.py`; actual full formal Trainer imports `build_action` at `code/src/pi_jwm/step5_2_training_loop_v1.py`. Starting `origin/main=b25bcdaa6b1d570b834879576312e1f256b30273` after fetch. Historical `formal_candidate_rollout_planner_v1.py` uses obsolete `task_action*` fields and is not the current implementation. Running Line A STEP 5.6B has source SHA `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`, run ID `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`. It was not contacted.

## 3. External Design References

These are design references, **not PI-JWM experiment evidence**. No K, elite size, iterations, network dimension or other numeric setting was copied.

| Reference | Relevant precedent |
| --- | --- |
| [PETS, Chua et al.](https://arxiv.org/abs/1805.12114) | Probabilistic model with sampling-based model predictive planning / CEM. |
| [PlaNet, Hafner et al.](https://proceedings.mlr.press/v97/hafner19a.html) | Latent dynamics and online planning. |
| [POPLIN, Wang & Ba](https://arxiv.org/abs/1906.08649); [official code](https://github.com/WilsonWangTHU/POPLIN) | Policy proposal/initialization combined with online planning. |
| [iCEM, Pinneri et al.](https://martius-lab.github.io/iCEM/); [official code](https://github.com/martius-lab/iCEM) | Elite reuse, shifted elites and temporally correlated sampling. |
| [TD-MPC2, Hansen et al.](https://arxiv.org/abs/2310.16828); [official code](https://github.com/nicklashansen/tdmpc2) | Policy-generated trajectories and sampling MPC with previous-plan warm start. |
| [Henaff et al., Model-Based Planning with Discrete and Continuous Actions](https://arxiv.org/abs/1705.07177) | Mixed discrete/continuous action design motivation. |

## 4. Frozen Contract / Changes / Reuse

The contract and high-level semantic fields are in `docs/contracts/PIJWM_STEP_06_0A_UNIFIED_CANDIDATE_GENERATION_CONTRACT_V1.md`. Implementation is additive in `code/src/pi_jwm/step6_0a_candidate_generation_v1.py`. It wraps the current `build_action`, so the formal training source and World Model remain untouched. Four explicit action families, horizon 1–4, tri-state constraints, fixed current support, fingerprint/pool/source merge, a deterministic Search stub, Learned callable interface, Hybrid composition, warm-start shift and no-op rule fallback are implemented. No network or objective is constructed.

The compiler uses the exact current adapter; acceptance compares all 11 tensor values with `torch.equal`, mapping metadata equality, and independent Route/Comm/Comp/Mob semantic assertions. This is stronger than shape equality. The synthetic fixture includes simultaneous four-action rows. Existing Route/Comp no-op and pending-flow behavior remain in the original adapter. The historical P6 prototype is left untouched.

## 5. Constraints, Fixed Support and Limitations

Known violation rejects construction; UNKNOWN remains reported in pool and blocks compilation pending researcher policy. New physical/Task/Flow identities are forbidden. A future-only Return Flow birth is not created by a Planner action. Dynamic available CPU lacks a reliable causal source; static capability is not substituted for it. Current action numeric mobility bounds have not been frozen from reliable evidence. Rule fallback is structurally representable only and has no safety claim. The current adapter can map Comm to a current Flow relation or a pending Route endpoint; v1 adds a current relation validity check. Interference means global RB exclusivity is not asserted.

Nonempty Comp and Mob add automatic unresolved records for dynamic CPU availability and numeric mobility bounds. The exact four-family adapter fixture uses `synthetic_contract_test_only=True` solely to prove tensor semantics; normal compilation rejects these unknowns. This avoids treating an engineering equivalence test as proof of a legal real action.

## 6. CPU Validation / Expected vs Actual

Expected: deterministic contract fixture, exact current adapter equality, negative tests for horizon/tri-state/support/relation/RB/UAV, backend/pool/warm-start behavior, no future-target access. Actual: see `code/artifacts/protocols/pi_jwm_step6_0a_candidate_generation_contract_v1_20260926/` for `candidate_generation_contract.json`, `candidate_generation_acceptance_receipt.json` and `action_adapter_equivalence_receipt.json`. Evidence is `SYNTHETIC_CONTRACT_EVIDENCE`, not real simulator candidate coverage. Focused test command: `python -m unittest discover -s code/tests -p test_step6_0a_candidate_generation_v1.py -v`. Other gate commands and outputs are recorded in the final completion report and process logs. Static source audit excludes rollout, optimizer, CUDA, future targets and locked test.

## 7. Research Pending and STEP 5.6B Relationship

Search versus Learned versus Hybrid is undecided. Optimizer, proposal training approach, factorization, objective/risk/future constraint details, numeric budgets, UNKNOWN feasibility policy and fallback safety remain researcher decisions. Formal Dataset behavior actions are coverage-oriented, not optimal expert labels; `proposal_training=false`. Line A STEP 5.6B runs independently on its pinned remote source; this Step does not use SSH/GPU, change its process/config/dataset/training code, or read a checkpoint. **NO WORLD-MODEL ROLLOUT WAS PERFORMED.**

## 8. Results, Known Issues, Git and Next Step

Result scope: static CPU candidate infrastructure only. Known issues: no later causal states are generated for L>1 compilation; no real candidate coverage, dynamic CPU proof, numeric UAV bounds or fallback safety proof. Git commit/push identity is in the final report. Next single action: wait for STEP 5.6B final best checkpoint and separate researcher authorization for World-Model-Dependent Planner Rollout.
