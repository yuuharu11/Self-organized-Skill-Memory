#!/usr/bin/env bash
# Unattended R2/R3 run for docs/research/11_robocasa_latent_expert after R1 and the
# token-index stage. Waits for token shards, then: key grid (GPUs 0-2) in parallel
# with the key-agnostic Expert (GPU 3), key selection/Eval, key drafts, offline eval.
# Each stage refuses to overwrite finished outputs, so a rerun resumes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
: "${SKILLNET_STATE_DIR:?source .skillnet-state/env.sh first}"
PY="${SKILLNET_VENV}/bin/python"
export PYTHONPATH="${repo}/skill_moe/skillnet/src:${repo}/skill_moe/skillnet/packages/skillnet-client/src:${repo}/skill_moe/skillnet/examples/robocasa:${repo}/skill_moe/skillnet/examples/memory_pilot/robocasa${PYTHONPATH:+:$PYTHONPATH}"
M="${repo}/skill_moe/skillnet/examples/memory_pilot/robocasa"
OUT="${OUT:-$SKILLNET_STATE_DIR/outputs/robocasa-memory/latent-skill-01}"
LOG="$OUT/logs"; mkdir -p "$LOG"
SOURCES="${SOURCES:-e_vlm_h e_vlm e_h p5 p0}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(stamp) wait for token shards"
until [ "$(grep -l '"complete": true' "$OUT"/tokens/shard*_done.json 2>/dev/null | wc -l)" -eq 4 ]; do sleep 30; done
echo "$(stamp) wait for training spread drafts"
until [ -f "$OUT/expert/drafts/train_spread.npz" ]; do sleep 30; done

echo "$(stamp) start Expert training (GPU 3) and key grid (GPUs 0-2)"
EXPERT_RUN="$OUT/expert/train/key-agnostic-01"
if pgrep -f "latent_expert.py train --out $OUT" >/dev/null; then
  echo "$(stamp) Expert training already running; will wait for its summary.json"
elif [ ! -f "$EXPERT_RUN/summary.json" ]; then
  CUDA_VISIBLE_DEVICES=3 "$PY" "$M/latent_expert.py" train --out "$OUT" > "$LOG/expert-train.log" 2>&1 &
  expert_pid=$!
fi
# Token E modes hold the ~40 GB bank on the GPU: one per GPU. Pooled/E-H runs are light.
CUDA_VISIBLE_DEVICES='' "$PY" "$M/latent_key.py" grid --out "$OUT" --modes e_vlm_h e_vlm --gpus 0 1 2 --per-gpu 1 > "$LOG/key-grid-token.log" 2>&1 &
g1=$!
CUDA_VISIBLE_DEVICES='' "$PY" "$M/latent_key.py" grid --out "$OUT" --modes p5 p0 e_h --gpus 0 1 2 --per-gpu 2 > "$LOG/key-grid-light.log" 2>&1 &
g2=$!
wait "$g1"; wait "$g2"
"$PY" "$M/latent_key.py" select --out "$OUT" > "$LOG/key-select.log" 2>&1
[ -f "$OUT/key/eval/results.json" ] || CUDA_VISIBLE_DEVICES=0 "$PY" "$M/latent_key.py" eval --out "$OUT" > "$LOG/key-eval.log" 2>&1
echo "$(stamp) key stage done"

CUDA_VISIBLE_DEVICES=0 "$PY" "$M/latent_expert.py" drafts --out "$OUT" --splits dev --sources $SOURCES > "$LOG/drafts-dev.log" 2>&1 &
d1=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$M/latent_expert.py" drafts --out "$OUT" --splits eval --sources $SOURCES > "$LOG/drafts-eval.log" 2>&1 &
d2=$!
wait "$d1"; wait "$d2"
echo "$(stamp) key drafts done"

if [ -n "${expert_pid:-}" ]; then wait "$expert_pid"; fi
until [ -f "$EXPERT_RUN/summary.json" ]; do
  pgrep -f "latent_expert.py train --out $OUT" >/dev/null || { echo "$(stamp) Expert training ended without summary.json" >&2; exit 1; }
  sleep 60
done
CK="$(ls "$EXPERT_RUN"/expert_*.msgpack | sort | tail -1)"
echo "$(stamp) evaluate with $CK"
CUDA_VISIBLE_DEVICES=0 "$PY" "$M/latent_expert.py" eval --out "$OUT" --split dev --expert "$CK" --sources $SOURCES > "$LOG/expert-eval-dev.log" 2>&1 &
e1=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$M/latent_expert.py" eval --out "$OUT" --split eval --expert "$CK" --sources $SOURCES > "$LOG/expert-eval-eval.log" 2>&1 &
e2=$!
wait "$e1"; wait "$e2"
"$PY" "$M/latent_expert.py" report --out "$OUT" > "$LOG/report.log" 2>&1
echo "$(stamp) pipeline complete"
