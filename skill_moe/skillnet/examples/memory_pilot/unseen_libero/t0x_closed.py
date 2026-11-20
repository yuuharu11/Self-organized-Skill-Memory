#!/usr/bin/env python3
"""T0X runner: KQ2-CL protocol, fresh T0X output tree, and resumable trials."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import kq2cl_common as C
import kq2cl_closed as K
from t0x_common import OUT, ROOT, progress, write_json

_REAL_POPEN = subprocess.Popen
BASE_CL_OUT = C.OUT


def _popen_with_t0x_server(args, *pos, **kw):
    if isinstance(args, (list, tuple)):
        args = list(args)
        for i, v in enumerate(args):
            if str(v).endswith('/kq2cl_serve.py') or str(v) == str(C.HERE / 'kq2cl_serve.py'):
                args[i] = str(C.HERE / 't0x_serve.py')
                break
    return _REAL_POPEN(args, *pos, **kw)


def build_specs():
    base = K.conditions()
    out = {}
    for name, q in (("Aadd", "iii"), ("Badd", "i")):
        old = base['A' if q == 'iii' else 'B']
        b = float(old['tpol']['b'])
        if not np.isfinite(b) or b <= 0:
            raise ValueError(f"invalid fallback b for {q}: {b}")
        out[name] = old | {"tpol": {"formula": "additive_radius", "c": (.35**2-.15**2)/b,
                                      "lo": .15, "hi": .35, "b": b}}
    # Diagnostic conditions reuse the existing A-const query, checkpoint and heads.
    aconst = base['Aconst']
    b = float(aconst['tpol']['b'])
    out['AconstPG'] = aconst | {"tpol": {"const": .15, "b": b, "gripper_mode": "pi05"}}
    out['AconstT25'] = aconst | {"tpol": {"const": .25, "b": b}}
    disp_path = OUT / 't2' / 'displacement.json'
    cal_path = OUT / 't2' / 'displacement-calibration.npz'
    if disp_path.exists() and cal_path.exists():
        d = json.loads(disp_path.read_text())
        dev = d.get('splits', {}).get('dev2', {}).get('iii', {})
        gains = {k: float(dev[k]['m_spearman'] - dev['ehat_spearman']) for k in ('t0.15','t0.30') if k in dev}
        if gains and max(gains.values()) >= .10:
            probe_key = max(gains, key=gains.get); base_a = base['A']; b = float(base_a['tpol']['b'])
            out['Aexp'] = base_a | {"tpol": {"formula": "expert_displacement", "probe": float(probe_key[1:]),
                                             "b": b, "calibration": str(cal_path)}}
    return out


def run_stage(stage: str, suite: str, names: list[str], *, workers: int, fraction: float, cpus: list[str], smoke: bool = False):
    global _REAL_POPEN
    if stage not in ('smoke', 't1', 't2-exp', 't3'):
        raise ValueError(stage)
    if (stage, suite) == ('t3', 'libero90'):
        raise ValueError('T3 conditions are specified for LIBERO-10 only')
    base_out = OUT / stage
    base_out.mkdir(parents=True, exist_ok=True)
    # Build specs against the original KQ2-CL assets, then route the runner's outputs to T0X.
    C.OUT = BASE_CL_OUT
    spec = build_specs()
    C.OUT = base_out
    spec_path = base_out / f'conditions-{suite}.json'
    write_json(spec_path, spec)
    # Add condition names to the existing resumable KQ2-CL plan without modifying its source.
    for s in ('libero90', 'libero10'):
        K.PLAN[s] = dict(K.PLAN[s])
        for n in ('Aadd', 'Badd', 'AconstPG', 'AconstT25', 'Aexp'):
            K.PLAN[s][n] = range(10)
    old_popen = K.subprocess.Popen
    K.subprocess.Popen = _popen_with_t0x_server
    prior_env = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    # The runner forks workers; keep aggregate worker threads <= 12*2, plus the two 2-thread servers.
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    try:
        progress(stage, 'closed-loop suite starting', suite=suite, conditions=names, smoke=smoke, fraction=fraction,
                 gpu=K.gpu_used())
        K.run(names, workers=workers, fraction=fraction, smoke=smoke, cpus=cpus, spec_path=str(spec_path), suite=suite)
    finally:
        K.subprocess.Popen = old_popen
        for k, v in prior_env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    progress(stage, 'closed-loop suite finished', suite=suite, conditions=names, gpu=K.gpu_used())


def read_trials(path: Path):
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: continue
    return rows


def summarize_stage(stage: str, suite: str):
    rel = (f"closed-loop-smoke{'-' + suite if suite != 'libero90' else ''}" if stage == 'smoke'
           else f'closed-loop/{suite}')
    root = OUT / stage / rel
    summary = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir() or d.name == 'servers': continue
        rs = read_trials(d / 'trials.jsonl')
        complete = [r for r in rs if r.get('status') == 'complete']
        succ = [bool(r['success']) for r in complete]
        summary[d.name] = {"completed": len(complete), "successes": int(sum(succ)),
                           "success_rate": float(np.mean(succ)) if succ else None,
                           "task_successes": {str(t): int(sum(bool(r['success']) for r in complete if int(r['task_id']) == t))
                                              for t in sorted({int(r['task_id']) for r in complete})},
                           "fallback_rate": float(np.mean([q.get('fallback_pi05', False) for r in complete
                                                             for q in r.get('selected_units_and_t0', [])])) if complete else None,
                           "t0_quantiles": _q([q.get('t0') for r in complete for q in r.get('selected_units_and_t0', [])]),
                           "ehat_quantiles": _q([q.get('confidence_error') for r in complete for q in r.get('selected_units_and_t0', [])])}
    return summary


def _q(vals):
    x = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return {} if not len(x) else {"p05": float(np.quantile(x,.05)), "median": float(np.median(x)),
                                  "p95": float(np.quantile(x,.95)), "mean": float(x.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run-t1', 'run-t3', 'summarize'))
    p.add_argument('--suite', choices=('libero90', 'libero10'), default='libero90')
    p.add_argument('--stage', choices=('smoke','t1','t3'), default='t1')
    p.add_argument('--conditions', default=None)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--fraction', type=float, default=.2)
    p.add_argument('--cpus', default='24-31,32-39')
    a = p.parse_args()
    if a.command == 'summarize':
        result = summarize_stage(a.stage, a.suite)
        write_json(OUT / a.stage / f'summary-{a.suite}.json', result)
        print(json.dumps(result, indent=2)); return
    if a.command == 'smoke':
        a.stage = 'smoke'
        names = ['Aadd']
    elif a.command == 'run-t1':
        names = ['Aadd', 'Badd']
        if a.suite == 'libero10': names = ['Aadd', 'Badd']
    else:
        names = ['AconstPG', 'AconstT25']
        a.stage = 't3'
    if a.conditions: names = a.conditions.split(',')
    run_stage(a.stage, a.suite, names, workers=a.workers, fraction=a.fraction, cpus=a.cpus.split(','),
              smoke=(a.command == 'smoke'))


if __name__ == '__main__':
    main()

