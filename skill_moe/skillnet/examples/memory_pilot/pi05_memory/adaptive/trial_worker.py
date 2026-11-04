"""Run one LIBERO rollout in an isolated, killable process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def main():
    threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
    import torch
    torch.set_num_threads(threads)
    pi05 = Path(__file__).resolve().parents[1]
    anchor_flow = pi05.parent / 'anchor_flow'
    sys.path.insert(0, str(anchor_flow))
    from run_case import _run_trial
    job_path, result_path = map(Path, sys.argv[1:3])
    import run_case
    request_seconds = 0.0
    original_infer = run_case.S2WebsocketClientPolicy.infer

    def timed_infer(client, observation):
        nonlocal request_seconds
        started = time.monotonic()
        try:
            return original_infer(client, observation)
        finally:
            request_seconds += time.monotonic() - started

    run_case.S2WebsocketClientPolicy.infer = timed_infer
    result = _run_trial(json.loads(job_path.read_text()))
    result['websocket_roundtrip_seconds'] = request_seconds
    result['environment_reset_and_step_seconds'] = max(
        0.0, float(result.get('elapsed_seconds', 0.0)) - request_seconds)
    result_path.write_text(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
