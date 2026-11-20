#!/usr/bin/env python3
"""Sequential T0X closed-loop suites; each suite is independently resumable."""
from __future__ import annotations

import argparse
import os

import t0x_closed as CL
from t0x_common import progress


def run_t1(*, workers=6, fraction=.20, cpus='24-31,32-39'):
    # Two conditions run concurrently; 6 workers x 2 threads each + two servers x 2 = 28 threads.
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    cpu_list = cpus.split(',')
    for suite in ('libero90', 'libero10'):
        progress('t1', 'starting additive-radius evaluation', suite=suite, conditions=['Aadd','Badd'],
                 workers_per_condition=workers, fraction=fraction)
        CL.run_stage('t1', suite, ['Aadd','Badd'], workers=workers, fraction=fraction, cpus=cpu_list)
        s = CL.summarize_stage('t1', suite)
        progress('t1', 'additive-radius suite summary', suite=suite, summary=s)
    progress('t1', 'both suites finished')


def run_t3(*, workers=6, fraction=.20, cpus='24-31,32-39'):
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    progress('t3', 'starting L10 diagnostic conditions', suite='libero10', conditions=['AconstPG','AconstT25'])
    CL.run_stage('t3', 'libero10', ['AconstPG','AconstT25'], workers=workers, fraction=fraction, cpus=cpus.split(','))
    s = CL.summarize_stage('t3', 'libero10')
    progress('t3', 'L10 diagnostic summary', suite='libero10', summary=s)


def run_t2(*, workers=6, fraction=.20, cpus='24-31,32-39', batch=1):
    import json
    from pathlib import Path
    from t0x_common import OUT, write_json
    from t0x_offline import evaluate
    # T2 repeatedly hit the allocator cap even at inference batch=1. Keep memory
    # preallocation disabled and honor the caller's observed-free-memory budget.
    os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
                      OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    progress('t2', 'starting Expert displacement offline comparison', batch=batch, query='iii and i')
    evaluate('cuda:0', ('dev2','unseen20','learned'), batch=batch, queries=('iii','i'))
    d=json.loads((OUT/'t2/displacement.json').read_text())
    dev=d['splits']['dev2']['iii']
    gains={k:float(dev[k]['m_spearman']-dev['ehat_spearman']) for k in ('t0.15','t0.30')}
    selected=max(gains,key=gains.get); passed=gains[selected]>=.10
    d['dev_gain_vs_ehat']=gains; d['Aexp_gate_pass']=bool(passed); d['selected_probe_for_Aexp']=selected if passed else None
    write_json(OUT/'t2/displacement.json',d)
    progress('t2', 'Dev confidence gate evaluated', gains=gains, pass_gate=passed, selected_probe=selected if passed else None)
    if passed:
        os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        for suite in ('libero90','libero10'):
            CL.run_stage('t2-exp', suite, ['Aexp'], workers=workers, fraction=fraction, cpus=cpus.split(','))
            progress('t2', 'A-exp closed-loop summary', suite=suite, summary=CL.summarize_stage('t2-exp',suite))
    else:
        progress('t2', 'A-exp closed-loop skipped because Dev improvement was below +0.10')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('t1','t2','t3'))
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--fraction', type=float, default=.20)
    p.add_argument('--cpus', default='24-31,32-39')
    p.add_argument('--batch', type=int, default=1)
    a = p.parse_args()
    if a.stage == 't1': run_t1(workers=a.workers, fraction=a.fraction, cpus=a.cpus)
    elif a.stage == 't2': run_t2(workers=a.workers, fraction=a.fraction, cpus=a.cpus, batch=a.batch)
    else: run_t3(workers=a.workers, fraction=a.fraction, cpus=a.cpus)


if __name__ == '__main__':
    main()

