# Skill Expert: lower-error skill drafts and confidence-adaptive bridge

## Fixed protocol (before execution)

- Frozen inputs: `run-01` rows, M1 codec, 6,583-row memory, b-hat prior, official pi05 checkpoint, and existing evaluation split. No input artifact is rewritten.
- Expert training changes only its training-time draft source. Seed 20261104, trainable filter, AdamW settings, maximum 4,000 updates, active7 loss, and 10-step bridge remain the same as `run.py`.
- Training draft mixture: 0.60 nearest active7 memory skill, 0.15 rank 2 or 3 nearest memory skill, 0.25 b-prior Top-1 retrieval; exclude the query's episode from all candidate pools. Decode each candidate base with the query's predicted residual, feature, and state. Save per-source draft errors.
- Select a checkpoint every 500 updates using fixed Dev rows, b-prior retrieval draft, and bridge start t0=.4. Calibration/Eval episodes are disjoint; Long Eval is never used for checkpoint selection or score-edge selection.
- Offline comparison: old and new experts × nearest-skill, b-prior retrieval, zero drafts × t0=.2/.4/.6, active7 normalized MSE. Evaluate calibration Eval (all, Long 548 rows, hard 63 rows).
- Confidence edges are b-prior retrieval cosine score quantiles q20/q60/q80 on fixed Dev rows only. Proposed C: `<q20` uses unchanged official pi05 normal generation; `[q20,q60)` uses new expert t0=.6; `[q60,q80)` t0=.4; `>=q80` t0=.2. A is b-prior retrieval at fixed .6. D is zero draft at fixed .6. B omits only C's fallback.
- Closed-loop smoke: two paired initial states per condition C/A/D. Assert task/init hashes and environment/query seed formulas match run-01; check normal fallback outputs exactly match official `sample_actions`; verify cached b-prior outputs against the frozen score-audit codec prior. This is a smoke only, not a success-rate claim.
- Runtime seeds: `env_seed=7+init_state_id`; `query_seed=(7+init_state_id)*100000+query_number`. Integrator 10, replan 5, wait 10, max control steps 520, environment horizon 1000.

## Predeclared interpretation

The new Expert uses skill content if nearest-skill drafts improve over zero drafts offline. Confidence-dependent C remains a policy hypothesis; its smoke success is not inferential evidence. If nearest-skill does not improve, report that the intended skill-correction effect was not demonstrated and interpret any closed-loop result accordingly.
