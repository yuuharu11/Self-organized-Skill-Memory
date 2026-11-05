# Skill Expert implementation and smoke

## Scope and artifacts

Implemented the approved skill-draft mixture and bridge training in `examples/memory_pilot/pi05_memory/skill_expert/run.py`; a confidence-adaptive server in `serve.py`; a killable LIBERO worker and paired runner in `trial_worker.py` and `evaluate.py`; plus `run_production.sh` for exp. `tests/test_pi05_skill_expert.py` covers mixture, confidence-boundary, and seed contracts.

Only the new Skill Expert checkpoint is trained. The run-01 codec, 6,583-row memory, b-prior, public pi05 checkpoint, and run-01 artifacts remain read-only. The best checkpoint is selected by fixed Dev rows with the b-prior draft at t0=.4, never by Long Eval. Old comparison loads `run-01/best_bridge.json`, not `latest_bridge.json`.

## Smoke results

- Draft preparation: 26,581 rows; 6,583 memory rows; exact active7 nearest-neighbor candidates exclude the query episode. Mean normalized active7 draft MSE across prepared rows: nearest skill .02135, rank 2/3 .02658, b-prior Top-1 .02716. Training mixture probabilities are .60/.15/.25.
- Training: 30 updates, batch 4, 45.25 s including checkpoint write; training loss step 1 .1440 and step 30 .1181. The saved smoke Expert is deliberately not the production model.
- Offline: 8 fixed Eval rows, both old and smoke Experts, all three drafts × t0 .2/.4/.6, plus official normal. On this tiny slice the smoke Expert nearest-draft MSE was .00127/.00097/.00088 for t0 .2/.4/.6 and zero-draft MSE .14316/.00862/.00807. These first eight rows are not a result estimate; run the full 1,318-row comparison before interpretation.
- Dev-only confidence edges from the 40 fixed Dev rows: q20=.94969045, q60=.97304760, q80=.98285078.
- Closed-loop smoke: C, A, D each completed 2/2 on task 0, initial states 0 and 1. All six environment seeds were 7/8 as specified; all per-query seeds matched `(7+init)*100000+query_number`; all initial-state hashes matched the corresponding run-01 `libero_10` normal trials. C had 4 low-confidence fallback queries across its two rollouts; each used the official action-expert weights and normal `sample_actions`. Server initialization asserted exact normal-path probe equality and score-audit b/r prior equality (cached b/r max absolute difference 0).

The first two closed-loop smoke attempts were preserved in separate output directories and exposed seed-parse and suite-key mistakes in the smoke harness. The final `smoke-closedloop-final` run passed; no run-01 output was changed.

Artifacts: `/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/skill-expert-01/`. The final smoke rollouts are in `smoke-closedloop-final/{C,A,D}/rollouts.jsonl`; the actual 30-step weights are under `smoke-model/`.

## Production command and estimate

After copying the approved source into the exp snapshot, run:

```bash
cd /work
bash skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_expert/run_production.sh
```

The script runs the 4,000-update bridge fit, full 1,318-row offline evaluation, Dev-only q20/q60/q80 fixation, then C/A/D in parallel (100 trials each, tasks 0–9 × initial states 0–9); it checks the paired run-01 state hashes and seed rules. Set `RUN_B=1` to run B after C/A/D. It uses `XLA_PYTHON_CLIENT_PREALLOCATE=false`, bounded allocator fractions, `timeout`, `nice -n 10`, and taskset. Each output directory refuses reuse when prior artifacts exist.

Estimated wall time on the H200: draft preparation <1 min; 4,000 updates about 15–25 min including Dev checkpoints; full offline comparison about 20–40 min; C/A/D parallel closed-loop about 25–45 min with two environment workers per condition. Total about 1–2 hours; optional B adds roughly 20–35 min. GPU sharing or simulator slowdowns can extend these estimates. The closed-loop smoke's 2/2 success per condition is only a plumbing check, not evidence of a success-rate change.
