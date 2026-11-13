# DN2 execution

The implementation is in `skill_expert_denoise.py`; the closed-loop server/runner
is a separate snapshot in `skill_expert_closed_loop.py`. Neither edits the INV1
or existing memory closed-loop code.

```bash
cd /work && source env/activate.sh
export SKILLNET_STATE_DIR=/work/.skillnet-state
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.25
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # lets the single-microbatch train fit under shared GPU pressure
nice -n 10 taskset -c 8-47 python /work/.skillnet-state/worktrees/query-invariance/skill_moe/skillnet/examples/memory_pilot/unseen_libero/skill_expert_denoise.py prepare --threads 8
nice -n 10 taskset -c 8-47 python /work/.skillnet-state/worktrees/query-invariance/skill_moe/skillnet/examples/memory_pilot/unseen_libero/skill_expert_denoise.py train --steps 4000 --threads 8
nice -n 10 taskset -c 8-47 python /work/.skillnet-state/worktrees/query-invariance/skill_moe/skillnet/examples/memory_pilot/unseen_libero/skill_expert_denoise.py evaluate --threads 8
```

After Dev selection and Eval are fixed, run the 100 paired LIBERO-90 trials in
a detached shell (three servers, each capped at 15% XLA memory):

```bash
cd /work && source env/activate.sh
export SKILLNET_STATE_DIR=/work/.skillnet-state
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.15
export MEM_POOL=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/cluster-pools-01/pool_unit_08.npz
export MEM_CKPT=/work/.skillnet-state/outputs/memory-pilot/unseen-libero/cluster-pools-01/query_unit_08.pt
export MEM_EXPERT="$(python -c 'import json;print(json.load(open("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-skill-01/best_bridge.json"))["path"])')"
export MEM_T0="$(python -c 'import json;print(json.load(open("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-skill-01/best_bridge.json"))["t0"])')"
setsid nohup nice -n 10 taskset -c 8-47 python /work/.skillnet-state/worktrees/query-invariance/skill_moe/skillnet/examples/memory_pilot/unseen_libero/skill_expert_closed_loop.py run --output /work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-skill-01/closedloop --checkpoint /work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero --suite libero_90 --conditions memory --servers 3 --inits 10 > /work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-skill-01/logs/closedloop.log 2>&1 < /dev/null &
```

The runner uses the same 10-task selection, trial indices, seed base 7, replan 5,
wait 10, max 520 steps, and baseline row keys as `memory_closed_loop.py`.
