#!/usr/bin/env bash
# Unattended: latent-key draft + key-agnostic Expert, RoboCasa 6 tasks x 50 trials (same seeds as 10's baseline).
set -uo pipefail
source /work/.skillnet-state/env.sh
source /work/skill_moe/skillnet/examples/robocasa/scripts/common.sh
S=/work/.skillnet-state/outputs/robocasa-memory/latent-skill-01/closed-loop
C=/work/.skillnet-state/outputs/robocasa-memory/latent-skill-01/conf
TASKS=(PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove)
stamp(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
run_cond(){ # name gpu port server_cpus sim_cpus trials workers tasks_csv indices_csv -- server args...
  local name=$1 gpu=$2 port=$3 scpu=$4 ccpu=$5 trials=$6 workers=$7 tasks=$8 idx=$9; shift 10
  local O=$S/$name; [[ -e $O/eval/results.json ]] && { echo "$(stamp) $name already done"; return 0; }
  rm -rf "$O"; mkdir -p "$O"
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=.80 taskset -c $scpu "$SKILL_PY" "$SKILL_ROOT/examples/robocasa/serve_latent_policy.py" \
     --port $port --query-log "$O/queries.jsonl" "$@" > "$O/server.log" 2>&1 & local SV=$!
  for i in $(seq 1 600); do grep -q 'serving latent-key-expert on' "$O/server.log" && break; kill -0 $SV 2>/dev/null || { echo "$(stamp) $name server died"; tail -30 "$O/server.log"; return 1; }; sleep 3; done
  ( source "$REPO_ROOT/env/robocasa/activate.sh"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    PYTHONPATH="$SKILL_ROOT/packages/skillnet-client/src:$SKILL_ROOT/examples/robocasa" taskset -c $ccpu python "$SKILL_ROOT/examples/robocasa/main.py" \
      --host 127.0.0.1 --port $port --tasks ${tasks//,/ } --split pretrain --num-trials $trials --replan-steps 5 --seed 7 --seed-task-stride 50 \
      --seed-task-indices ${idx//,/ } --resize 224 --workers $workers --output "$O/eval" > "$O/client.log" 2>&1
    python "$SKILL_ROOT/examples/robocasa/get_eval_stats.py" "$O/eval/results.json" > "$O/stats.txt" 2>&1 )
  local rc=$?; kill -TERM $SV 2>/dev/null; wait $SV 2>/dev/null; echo "$(stamp) $name rc=$rc"; cat "$O/stats.txt" 2>/dev/null | tail -12; return $rc
}
echo "$(stamp) wait for the seed re-runs (GPUs)"
while pgrep -f "conf/seed_eval.py" >/dev/null; do sleep 60; done
while pgrep -f "smoke_baseline.sh" >/dev/null; do sleep 30; done
echo "$(stamp) smoke: OpenDrawer x2, fixed t0=.1"
run_cond smoke-latent 0 8811 0-7 16-31 2 2 OpenDrawer 2 -- --t0-policy fixed --t0 .1 || { echo "$(stamp) smoke failed; stop"; exit 1; }
ALL=$(IFS=,; echo "${TASKS[*]}")
echo "$(stamp) full runs: 3 conditions x 6 tasks x 50"
run_cond fixed-t0.1 0 8821 0-3 8-31 50 24 $ALL 0,1,2,3,4,5 -- --t0-policy fixed --t0 .1 > $S/fixed-t0.1.driver.log 2>&1 &
run_cond conf-dist 1 8822 4-7 32-55 50 24 $ALL 0,1,2,3,4,5 -- --t0-policy dist > $S/conf-dist.driver.log 2>&1 &
run_cond fixed-t0.5 2 8823 88-91 56-79 50 24 $ALL 0,1,2,3,4,5 -- --t0-policy fixed --t0 .5 > $S/fixed-t0.5.driver.log 2>&1 &
wait
echo "$(stamp) all closed-loop runs finished"
for n in fixed-t0.1 conf-dist fixed-t0.5; do echo "== $n"; tail -12 $S/$n/stats.txt 2>/dev/null; done
echo "closed-loop complete"
