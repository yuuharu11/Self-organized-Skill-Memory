#!/usr/bin/env python3
"""Run one real LIBERO-90 trial through the EX1 P5+Expert websocket policy."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero/expert-ex1-01')
PI05 = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--expert-checkpoint', type=Path, required=True)
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--trial-index', type=int, default=0)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    out = a.out
    if out.exists():
        raise FileExistsError(out)
    prep = json.loads((a.prepared_dir / 'prepare.json').read_text())
    if prep.get('dev_draft_selection_method') != 'key2f_retrieval':
        raise ValueError('closed-loop smoke requires an actual-retrieval Dev preparation')
    if not a.expert_checkpoint.exists():
        raise FileNotFoundError(a.expert_checkpoint)
    out.mkdir(parents=True)
    import run as harness

    from expert_ex1 import load_locked_key2_checkpoints
    p5_ckpt = Path(load_locked_key2_checkpoints()['p5']['checkpoint'])
    port = free_port()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.35')
    log = (out / 'server.log').open('w')
    proc = subprocess.Popen(['taskset', '-c', '8-15', 'nice', '-n', '10', sys.executable,
                             str(HERE / 'serve_ex1_smoke.py'), '--pi05-checkpoint', str(PI05),
                             '--p5-checkpoint', str(p5_ckpt), '--expert-checkpoint', str(a.expert_checkpoint),
                             '--port', str(port), '--t0', '.3'],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = None
    try:
        harness.ready(proc, port)
        job = {'host': '127.0.0.1', 'port': port, 'suite': 'libero_90', 'task_id': a.task_id,
               'trial_index': a.trial_index, 'seed_base': 7, 'condition': 'EX1_D1_P5_top1_expert_smoke',
               'memory_snapshot': 'KEY2-F locked P5 + 4,516 DCT3 units',
               'query_text_source': 'instruction_only', 'checkpoint': str(PI05),
               'wait_steps': 10, 'replan_steps': 5, 'max_steps': 520, 'record': True}
        attempt = out / 'trial' / 'attempt-1'
        result = harness.trial(job, attempt, a.timeout, 1)
        result['started_unix'] = time.time()
        result['policy_server_log'] = str(out / 'server.log')
        (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get('status') != 'complete':
            raise RuntimeError(f"EX1 closed-loop smoke did not complete: {result.get('status')} {result.get('error')}")
    finally:
        harness.terminate(proc)
        log.close()


if __name__ == '__main__':
    main()
