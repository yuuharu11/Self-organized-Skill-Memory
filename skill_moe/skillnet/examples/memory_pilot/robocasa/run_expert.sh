#!/usr/bin/env bash
set -euo pipefail
cd /work
source env/activate.sh >/dev/null
WORKTREE=/work/.skillnet-state/worktrees/robocasa-memory
OUT=/work/.skillnet-state/outputs/robocasa-memory/expert-01
export PYTHONPATH="$WORKTREE/skill_moe/skillnet/src:$WORKTREE/skill_moe/skillnet/examples/memory_pilot/robocasa:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.35
PY=/work/.skillnet-state/envs/skillnet/bin/python
mkdir -p "$OUT/logs"
setsid nohup taskset -c 8-47 nice -n 10 timeout 12h bash -c "\
  '$PY' -u '$WORKTREE/skill_moe/skillnet/examples/memory_pilot/robocasa/expert.py' prepare --out '$OUT' && \
  '$PY' -u '$WORKTREE/skill_moe/skillnet/examples/memory_pilot/robocasa/expert.py' train --out '$OUT' --steps 4000 --batch 4 && \
  '$PY' -u '$WORKTREE/skill_moe/skillnet/examples/memory_pilot/robocasa/expert.py' evaluate --out '$OUT' && \
  '$PY' -u '$WORKTREE/skill_moe/skillnet/examples/memory_pilot/robocasa/expert.py' report --out '$OUT'" \
  > "$OUT/logs/full-run.log" 2>&1 < /dev/null &
echo "launched pid=$!; follow $OUT/logs/full-run.log"
