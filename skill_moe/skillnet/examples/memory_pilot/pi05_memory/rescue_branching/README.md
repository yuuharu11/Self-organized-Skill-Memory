# LIBERO-Long rescue branching

This directory contains a separate fast grouped-MaxSim index, a policy server,
and a simulator-state capture/restart harness. It does not modify the existing
`adaptive/`, `pi05_memory/serve.py`, or `anchor_flow/run_case.py` code.

## Retrieval path

`FastTokenIndex` keeps the same fp16 key normalization, modality-matched
MaxSim, group averaging and stable first-row tie break as the adaptive server.
It selects the three query modality slices once per query, then processes the
memory bank in wider row blocks (default 256 rather than 32). This avoids
repeating query-side selection in each block and reduces launch/loop overhead.
`FastTokenIndex.search` is a drop-in `TokenIndex` interface. To use it in a
separate adaptive-server snapshot, import this module and assign
`serve_module.TokenIndex = FastTokenIndex` before constructing
`MemoryPolicy`; this repository's adaptive server remains untouched.

## Smoke

The smoke is deliberately one previously failed run-01 trajectory: LIBERO-10
task 6, public initial state 0, environment seed 7. It records simulator
snapshots before queries at 20%, 40% and 60% of the 520-step control budget.
At a snapshot it also records the exact Top-8 memory rows and scores. The
flattened `env.sim.get_state()` is saved. A direct `set_init_state(flat)`
restore was checked, but it changed the rendered observation and the next
normal action chunk: this robosuite `MjSimState` serializes only time/qpos/qvel,
not the controller/solver state. Therefore each branch deterministically
replays the recorded pre-branch action chunks from the same seeded initial
state, then asserts that the resulting flattened MuJoCo state and policy
observation exactly equal the saved checkpoint before branching. This replay
path retains controller and solver history and passed the action-sequence
reproduction check.

```bash
cd /work
source env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 \
BRANCH_TORCH_MEM_FRACTION=0.20 taskset -c 8-47 nice -n 10 python \
  skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/evaluate_smoke.py \
  --workers 4 --branch-fraction 0.6
```

The smoke checks same-seed normal action chunks against the original replay,
then runs eight independent normal-noise seeds, zero draft, current calibrated
retrieval, and each of the saved Top-8 memory rows at t0=0.4 and 0.2. Results,
states, query ranking and per-branch timing are stored below
`rescue-branching-01/trials/`; the controller writes `smoke_results.json`.

## Prepare exp's trial list and start the full run

The helper selects up to ten normal failures from the paired-breakdown list,
prioritizing tasks 6 and 8, and preserves the initial state index and expected
seed/hash for verification. Exp should inspect and approve that generated
list, then run the controller from a detached tmux session:

```bash
cd /work
source env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/make_trial_list.py \
  --output /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01/normal_failures.json

tmux new-session -d -s rescue-branching \
  "cd /work && source env/activate.sh && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 BRANCH_TORCH_MEM_FRACTION=0.20 taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/evaluate.py --output /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01/full --trials-json /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01/normal_failures.json --workers 4 --timeout 600"
```

The production controller records a baseline replay and three simulator
snapshots per failure. Each snapshot branches into eight normal-policy random
seeds, zero draft, current search, and the 16 Top-8-row × {0.4, 0.2} drafts
(26 continuations per branch point, approximately 780 total for ten failures).
Environment workers are capped at four; each has a 600-second wall limit.
Completed capture/branch JSON files are reused on restart; failed branch
attempts receive a new filename. The full run is for exp; the implementation
turn only performs the single-failure smoke.

## Expanded 44-condition protocol (current stage-3 implementation)

Use `serve_extended.py`, `rollout_extended.py`, and `evaluate_expanded.py` for
the approved four-key candidate union. This sibling path leaves the legacy
26-condition controller above intact. At each branchpoint it ranks memory rows
with current grouped-VLM MaxSim, wrist-only token MaxSim, memory-standardized
codec-prior b̂ cosine, and negative active-7 MSE from the official π0.5 action
prediction to recorded memory actions. Each key contributes two distinct rows;
overlaps advance to the next rank. `*.keys.json` stores the selected candidates'
rank and score under every key. Candidate branches use either the stored
normalized `rows/actions[row]` array in all 32 dimensions (D-raw, identical to
`memory_gap/run.py`) or `bases[row]` decoded with the query prior residual.
Zero and current-VLM drafts each run at fixed t0 .6 and .2; each of the eight
candidates runs in both formats at t0 .4 and .2. This gives 44 requested
conditions per point (8 alternate-noise normal + 2 zero + 2 current + 32
candidate branches). The separate same-RNG continuation is only a replay check.

After the implementation smoke is reviewed, exp can prepare a fresh trial list
and launch from a detached tmux session. Do not point this command at the
preflight-only `production/` directory; it is read-only evidence.

```bash
cd /work && source env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  python skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/make_trial_list.py \
  --output /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01/failure-trials-expanded-01.json

tmux new-session -d -s rescue-branching-expanded \
  "cd /work && source env/activate.sh && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.11 BRANCH_TORCH_MEM_FRACTION=0.15 taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/evaluate_expanded.py --output /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01/production-expanded-01 --trials-json /work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/rescue-branching-01/failure-trials-expanded-01.json --workers 4 --timeout 600"
```

The expanded path writes to the new `production-expanded-01/` tree, stores
failure replay attempts and per-branch JSON/logs separately, and resumes
completed branches without deleting error logs. The planning estimate uses
10 failures × 3 branchpoints × 44 conditions = 1,320 branches; use the measured
smoke wall time for the final runtime estimate.
