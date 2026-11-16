#!/usr/bin/env bash
set -euo pipefail
cd /work
source env/activate.sh
export SKILLNET_STATE_DIR="${SKILLNET_STATE_DIR:-/work/.skillnet-state}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/key_shared_latent.py"
WORKER="$HERE/key_shared_latent_tune_worker.py"
BASELINES="$HERE/key_shared_latent_baselines.py"
COLLECTOR="$HERE/key_shared_latent_collect.py"
OUT="${1:?usage: key_shared_latent_launch.sh NEW_OUTPUT_ROOT}"
if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty output: $OUT" >&2; exit 2
fi
mkdir -p "$OUT/tuning" "$OUT/final" "$OUT/.dev-slots"
progress() { printf '%s\t%s\n' "$(date -Is)" "$*" >> "$OUT/progress.log"; }
progress 'started KEY2-F; attempt-25 superseded due to history and contrastive-temperature defects'
VJEPA_EVAL="$OUT/vjepa/libero90_eval"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.04 TORCH_MEM_FRACTION=.04
taskset -c 8-9 nice -n 10 python3 "$CODE" --output "$VJEPA_EVAL" --prepare-vjepa-cache >"$OUT/vjepa-build.log" 2>&1
progress 'exact Eval V-JEPA cache ready'
taskset -c 8-15 nice -n 10 python3 "$BASELINES" --output "$OUT/fixed-baselines" \
  --vjepa-eval-cache "$VJEPA_EVAL" \
  --parity-source "$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-25/final/e_vlm_h/results.json" \
  >"$OUT/fixed-baselines.log" 2>&1
progress 'fixed CPU references and hold-last gate ready'

run_dev() {
  local mode="$1" lr="$2" epochs="$3" tau="$4" tag="$5" objective="$6"
  local args=(--output "$OUT/tuning/$tag" --mode "$mode" --objective "$objective"
    --learning-rate "$lr" --epochs "$epochs" --teacher-temperature "$tau"
    --vjepa-eval-cache "$VJEPA_EVAL")
  mkdir -p "$(dirname "$OUT/tuning/$tag")"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.025 TORCH_MEM_FRACTION=.025
  local slot fd status
  while :; do
    for ((slot=0; slot<32; slot++)); do
      exec {fd}>"$OUT/.dev-slots/$slot.lock"
      if flock -n "$fd"; then
        taskset -c "$((8+slot))" nice -n 10 python3 "$WORKER" "${args[@]}" >"$OUT/tuning/$tag.log" 2>&1
        status=$?
        flock -u "$fd"; exec {fd}>&-
        return "$status"
      fi
      exec {fd}>&-
    done
    sleep 1
  done
}

# Dispatch all Dev configurations before one barrier. Thirty-two lock slots
# cap resident full-data copies, runnable CPU threads, and GPU allocations.
modes=(e_vlm e_vlm_h e_vj_h e_vj_vlm_h e_h separate p0 p5)
taus=(0.002 0.005 0.01 0.02)
declare -a pids=() labels=()
slot=0
for mode in "${modes[@]}"; do
  for lr in 0.0001 0.0003; do
    for epochs in 4 8 16; do
      for tau in "${taus[@]}"; do
        tag="$mode-lr${lr}-ep${epochs}-tau${tau}"
        run_dev "$mode" "$lr" "$epochs" "$tau" "$tag" soft &
        pids+=("$!"); labels+=("$tag"); slot=$((slot+1))
      done
    done
  done
done
for lr in 0.0001 0.0003; do
  for epochs in 4 8 16; do
    tag="e_vlm_h-hard-lr${lr}-ep${epochs}"
    run_dev e_vlm_h "$lr" "$epochs" 0.005 "$tag" hard &
    pids+=("$!"); labels+=("$tag"); slot=$((slot+1))
  done
done
progress "dispatched ${#pids[@]} Dev jobs across 32 CPU cores"
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then progress "Dev worker ${labels[$i]} complete"
  else progress "Dev worker ${labels[$i]} failed"; failed=$((failed+1)); fi
done
if (( failed )); then echo "$failed Dev workers failed; final evaluation not started" >&2; exit 1; fi

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
taskset -c 8-9 nice -n 10 python3 - "$OUT/tuning" "$OUT/selected_configs.json" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2])
modes=('e_vlm','e_vlm_h','e_vj_h','e_vj_vlm_h','e_h','separate','p0','p5')
best={}
def consider(label,path,mode,objective):
    f=path/'dev_results.json'
    if not f.exists():raise SystemExit(f'missing Dev output {f}')
    d=json.loads(f.read_text());m=d['models'][mode];blend=m['dev_selection']['blend_selection']['selected'];train=m['train']
    score=float(blend['mse_gripper_truth'])
    item={'mode':mode,'objective':objective,'learning_rate':float(train['learning_rate']),
          'epochs':int(train['epochs']),'teacher_temperature':d.get('training_teacher_temperature',.005),
          'checkpoint':m['checkpoint'],'dev_mse':score,'selected_blend':blend,
          'dev':m['dev_selection'],'tuning_dir':str(path)}
    tie=(score,item['learning_rate'],item['epochs'],item['teacher_temperature'])
    if label not in best or tie < best[label]['_tie']:item['_tie']=tie;best[label]=item
for mode in modes:
    for lr in ('0.0001','0.0003'):
        for ep in (4,8,16):
            for tau in ('0.002','0.005','0.01','0.02'):
                consider(mode,root/f'{mode}-lr{lr}-ep{ep}-tau{tau}',mode,'soft')
for lr in ('0.0001','0.0003'):
    for ep in (4,8,16):consider('e_vlm_h-hard',root/f'e_vlm_h-hard-lr{lr}-ep{ep}','e_vlm_h','hard')
expected=set(modes)|{'e_vlm_h-hard'}
if set(best)!=expected:raise SystemExit(f'incomplete Dev selection: {set(best)}')
for item in best.values():item.pop('_tie',None)
dest.write_text(json.dumps({'selection':'Dev-only top-k blend MSE; tie-break lower LR, epochs, then teacher tau',
    'training_teacher_tau_grid':[.002,.005,.01,.02], 'ranking_reference_tau_fixed':.01,
    'score':'exp(learned logit_scale) * similarity; logit_scale starts 1/.07, max 100',
    'selected':best},indent=2)+'\n')
PY

run_final() {
  local label="$1" cpu="$2" cfg="$OUT/selected_configs.json" mode objective lr epochs tau checkpoint dest
  read -r mode objective lr epochs tau checkpoint < <(python3 - "$cfg" "$label" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))['selected'][sys.argv[2]]
print(x['mode'],x['objective'],x['learning_rate'],x['epochs'],x['teacher_temperature'],x['checkpoint'])
PY
)
  dest="$OUT/final/$label"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.10 TORCH_MEM_FRACTION=.10
  taskset -c "$cpu" nice -n 10 python3 "$CODE" --output "$dest" --mode "$mode" --objective "$objective" \
    --seed 1 --learning-rate "$lr" --epochs "$epochs" --teacher-temperature "$tau" \
    --eval-only-checkpoint "$checkpoint" --baseline-json "$OUT/fixed-baselines/baselines.json" \
    --vjepa-eval-cache "$VJEPA_EVAL" >"$OUT/final/$label.log" 2>&1
}
final_labels=(e_vlm e_vlm_h e_vj_h e_vj_vlm_h e_h separate p0 p5 e_vlm_h-hard)
declare -a final_pids=()
slot=0
for label in "${final_labels[@]}"; do
  lo=$((8+2*slot));hi=$((lo+1));run_final "$label" "$lo-$hi" & final_pids+=("$!");slot=$((slot+1))
done
failed=0
for i in "${!final_pids[@]}"; do
  if wait "${final_pids[$i]}"; then progress "final worker ${final_labels[$i]} complete"
  else progress "final worker ${final_labels[$i]} failed"; failed=$((failed+1)); fi
done
if (( failed )); then echo "$failed final workers failed; inspect logs" >&2; exit 1; fi
taskset -c 8-9 nice -n 10 python3 "$COLLECTOR" --run-root "$OUT" --output "$OUT/aggregate" \
  --baselines "$OUT/fixed-baselines/baselines.json"
progress 'KEY2-F pipeline complete'
echo "KEY2-F complete under $OUT"
