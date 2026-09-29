# STEP 6.3B — Structured Candidate Grammar and Support Policy Closure v1

> 2026-09-29 supersession: STEP 6.3B/6.3C-PATCH researcher decision removed the all-eligible Comm Task coverage rule. For current Planner-v1 Comm selection, use `STEP_06_3BC_PATCH_COMM_TASK_SELECTION_TRAIN_SELF_REPLAY_AUDIT.md` and the amended contract. Counts and claims below describe the original 6.3B acceptance at its time.

## Step Goal and definition basis

Define a search-independent Comm/Comp/Mob candidate grammar, multidimensional Formal TRAIN support labels and formal pool admission. Researcher choices in this conversation authorize repeated Comm rows for one Task, shared-profile Mobility, TRAIN-observed joint structural signatures only, temporal labels without a hard transition gate, and nonempty Comp when current computing Tasks exist. The frozen Route no-op and five-part lexicographic Objective remain unchanged. This is a contract/CPU mechanism step, not a search algorithm or performance experiment.

## Initial state and evidence

Start: `43fc0a7f308c71d26cfb0eeddd4397ebc3b73b14` at local HEAD and origin/main. Existing STEP 6.3A and STEP 6.2B receipts said PASS. Source/config/receipts take precedence over prose. The prior 6.3A use of “no-op” referred to policy intervention flags, not empty actions: TRAIN raw decision slots with no-intervention flags but nonempty family action were Comm 674, Comp 913, Mob 2761; eligible-with-empty action was zero in all three. This is a correction of interpretation, not a rewrite of 6.3A evidence. Formal TRAIN includes 4608 raw decision slots and 4416 windows. Its 251 joint structural signatures are summaries, not concrete actions.

## Files and changes

- `code/scripts/run_step6_3b_comm_support_boundary_audit_v1.py` freezes a TRAIN-only structural catalog and separately scans validation for description.
- `code/src/pi_jwm/step6_3b_candidate_grammar_v1.py` binds one structural choice to the current causal/predicted state and validates supplied CandidateActionSequence objects from any backend.
- `code/src/pi_jwm/step6_3b_candidate_support_v1.py` loads the TRAIN catalog and labels family, joint, temporal, causal and fixed-support axes.
- `code/src/pi_jwm/step6_0a_candidate_generation_v1.py` now accepts repeated Comm Task rows; the existing adapter compiles both at their current relation.
- The STEP 6.3B contract, focused tests, machine receipts, context and process records carry the same boundaries.

## Reuse and evidence interpretation

Existing `allocate_work_conserving_cpu` is the Comp base; the grammar does not invent a CPU allocator. Existing 6.0A CandidateActionSequence and adapter remain the action contract; Route is explicit empty. Current Flow and Comm relation tensors provide wireless eligibility. The current Task-Agent `Exec` relation, remaining CPU work and static capacity provide Comp eligibility. Current UAV control side-state provides Mobility parameters. TRAIN future action frames supply structural evidence only; no future target or future truth enters binding or admission. Validation is descriptive and cannot alter the catalog.

Formal TRAIN has 293/4608 raw slots with repeated Comm rows for one Task, and 280/4416 H1 windows. There are 145 observed `(start,width)` pairs from 50 RB starts times widths 1–3, a maximum of four Comm rows per slot, and 251 joint structures in the H1–H4 TRAIN action-frame union. H4 structural sequence support is sparse (2719 unique prefixes). Thus the formal policy admits observed joint structures with state-conditioned concrete parameters, records temporal evidence, and does not replay fixed historical identities or require observed transitions. The separate descriptive Formal Validation scan found 46/1152 raw slots with a TRAIN-unseen joint structure and 2/479 Comm rows with a TRAIN-unseen `(start,width)` pair; neither changed the catalog or policy.

The 4.4 deterministic rule does not promote offloading to computing after Flow completion. Future Comp rows therefore need an explicitly predicted computing lifecycle and unique Exec relation; the grammar refuses to invent a base for uncertain states. TRAIN-only future-Return `H_sup` histogram is unavailable. This statistic is unnecessary to define action grammar: `h_sup` remains pending until an actual supported candidate rollout and the 6.2B scorer's `H_eff` remains unchanged.

The 6.0A all-empty `RULE_FALLBACK` is still a generic structural action. In a state with eligible Comm/Comp/Mob objects it is outside this Formal TRAIN-backed pool, because TRAIN policy “no intervention” still emitted explicit family rows. A later fallback protocol can address it; no exception was invented here.

## Validation, results, expected versus actual

Focused commands with `PYTHONPATH=code/src;code/scripts`: `python -m unittest discover -s code/tests -p 'test_step6_3b*.py'` (11/11), the corresponding STEP 6.0A test file (6/6), STEP 6.0C test file (15/15), and STEP 6.2B scorer test file (11/11). `python -m compileall -q code/src code/scripts code/tests` passed. `python code/scripts/build_project_knowledge_index_v1.py` and its `--check` both reported `mismatches=[]`, `passed=true`. `git diff --check` passed. Context Consistency Check inspected all `AI_CONTEXT/00`–`08` and synchronized current state, research boundary, architecture/data flow, module map, experiments, explicit researcher decisions, known limits and changelog. No full training or full repository unittest discovery is part of this step. Expected and actual result: a train-supported structural candidate can be rebound and admitted using the current state; unsupported identity/parameter/joint structure rejects; temporal unseen remains a label. The synthetic H2 test rebinds current Flow and recomputes CPU base after predicted-state change. These tests do not establish search quality or model accuracy.

## Known issues, Git and next step

The formal pool is limited to current learned support and does not prove candidate quality or World Model reliability under newly combined concrete parameters. Temporal unseen combinations are labeled but admitted when each step meets the other gates; their effect is untested. A fixed-support failure can shorten `H_sup` after rollout; no pre-rollout action signature guarantees it. The checkpoint is not read or modified in this Step. No optimizer, ranking, GPU, locked test, training, baseline or closed loop is run. Final Git SHA and push status belong in the acceptance receipt after verification. The only possible next research step is separately authorized STEP 6.3C candidate method comparison; it is not started here.
