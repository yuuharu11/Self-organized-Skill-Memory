# Compress/cluster oracle diagnostic

This diagnostic deliberately uses each query's true action to select one medoid
from a pool that includes the query row. It measures representation and
clustering ceilings, not retrieval or generalization.

After the existing b96 codec process has completed, build the CPU-only
pre-generation table:

```bash
cd /work && source env/activate.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
taskset -c 8-23 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/compress_cluster_oracle/experiment.py build --threads 8
```

The table is written before any Expert training. Expert runs use the frozen
public π0.5 checkpoint, the fair-expert batch row sequence and optimizer, and
should be launched with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and per-process
GPU fraction at or below 0.20. Output paths are append-only; do not remove or
reuse an existing run directory.

After `pre_generation.json` exists, run the ordered, GPU-aware queue. The
intended four-worker `.20` and five-worker `.17` settings were tried first, but
those allocations failed with XLA `RESOURCE_EXHAUSTED` (including when only
two workers were active). A single `.35` worker also failed during training.
The completed run therefore used one worker at `.50`; this allocated about
66 GB on the 143 GB H200 and left about 77 GB free. Each worker used 8 CPU
threads on cores 8–15 with `nice -n 10`. The queue checks GPU status before
launching work, preserves failed attempts, skips completed combinations, and
continues after a failed combination. All 21 requested combinations were
covered with no failures: the previously completed `b48 × mid` evaluation was
reused, and the queue ran the other 20 combinations. Add `--include-extra` to also run
raw/b12 fine and b24/b48/b96 mid after the priority list:

```bash
cd /work && source env/activate.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
taskset -c 8-15 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/pi05_memory/compress_cluster_oracle/run_queue.py --include-extra
```

The queue sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=.50` for its single worker. Do not increase
concurrency on this machine without a fresh GPU allocation check: per-process
XLA limits caused OOM even though aggregate `nvidia-smi` free memory looked
sufficient.
