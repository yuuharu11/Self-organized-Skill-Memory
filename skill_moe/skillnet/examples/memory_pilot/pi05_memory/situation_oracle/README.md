# Situation-oracle retrieval audit

`run.py` reconstructs exact LIBERO simulator states from the source HDF5 demos,
forms anonymous E/R/G/H geometry/history blocks, selects only block weights on
the fixed non-Long Dev episodes, then evaluates the Long queries against the
fixed `raw-rows-noLong` memory pool and a separate same-task/different-episode
reference pool. It never trains a model or runs a policy rollout.

The source RLDS episodes have no-op frames removed. `source_episode_index` and
task identify the source episode; an exact monotone action subsequence maps its
filtered frame index to the original HDF5 frame index. Each selected 10-step
window is checked against the HDF5 raw actions and against the cached normalized
active7 actions before simulator states are used.

Run tests and a tiny state/action smoke first:

```bash
cd "$SKILLNET_REPO_ROOT" && source env/activate.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  nice -n 10 taskset -c 8-23 pytest -q tests/test_pi05_situation_oracle.py
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 nice -n 10 taskset -c 8-23 \
  python skill_moe/skillnet/examples/memory_pilot/pi05_memory/situation_oracle/run.py \
  --smoke --output /tmp/situation-oracle-smoke
```

Full run (CPU only; refuses to overwrite its output):

```bash
cd "$SKILLNET_REPO_ROOT" && source env/activate.sh
mkdir -p .skillnet-state/outputs/memory-pilot/pi05-memory40-20261104
nohup setsid env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 \
  nice -n 10 taskset -c 8-23 python \
  skill_moe/skillnet/examples/memory_pilot/pi05_memory/situation_oracle/run.py \
  > .skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/situation-oracle-01.log 2>&1 &
```

Artifacts: `simulator_features.npz`, `episode_mapping.json`, `weight_selection.json`,
`row_metrics.npz`, `results.json`, and `logs/progress.jsonl`. For the final
audited Top-8 recall and complete same-task reference metrics, run
`analyze.py` after feature extraction; it uses the saved simulator features and
writes `row_metrics_final.npz` / `results_final.json` without repeating state
reconstruction. Row-level retrieval arrays contain dataset row IDs, not offsets
in the compact candidate pool. In this run, `results.json` was superseded
because its first summary pass evaluated recall@8 from only five stored rows;
use `results_final.json`.
