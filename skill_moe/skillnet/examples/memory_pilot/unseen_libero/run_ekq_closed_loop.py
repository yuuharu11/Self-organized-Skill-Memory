#!/usr/bin/env python3
"""EKQ closed loop: KQ-final drafts + frozen Expert with the π0.5-free runtime (expert_kq_runtime.py).

Same setting as CL1 (run_cl1_closed_loop.py): LIBERO-90 Eval 10 tasks x 10 trials, seed_base 7,
public initial states 0..9, replan 5, wait 10, 520 steps, same Expert checkpoint (CL1 schedule.json).
Every finished trial is appended (fsync) to <cond>/trials.jsonl; a rerun resumes the missing trials.

Conditions
  E1a     C1a query, new confidence/t0 + gripper head (π0.5 only for the Dev-chosen fallback)
  E1b     C1b query, same runtime
  E1a_pg  C1a query, new confidence/t0, π0.5 gripper (ablation: separates the two changes)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cl1_closed_loop as cl1  # noqa: E402

ROOT = cl1.ROOT
EKQ = ROOT / 'expert-kq-01'
OUT = EKQ / 'closed-loop'
CONDITIONS = {
    'E1a': dict(mode='KQ_AFIX_top1', query=cl1.CONDITIONS['C1a']['query'], bundle=EKQ / 'heads/C1a_bundle.pt', gripper='head',
                label='C1a 検索＋新しい信頼度・t0＋グリッパーヘッド（π0.5 は退避のみ）', cl1='C1a'),
    'E1b': dict(mode='KQ_HMLP_top1', query=cl1.CONDITIONS['C1b']['query'], bundle=EKQ / 'heads/C1b_bundle.pt', gripper='head',
                label='C1b 検索＋新しい信頼度・t0＋グリッパーヘッド（π0.5 は退避のみ）', cl1='C1b'),
    'E1a_pg': dict(mode='KQ_AFIX_top1', query=cl1.CONDITIONS['C1a']['query'], bundle=EKQ / 'heads/C1a_bundle.pt', gripper='pi05',
                   label='C1a 検索＋新しい信頼度・t0＋π0.5 グリッパー（切り分け）', cl1='C1a'),
}


def run_condition(name, out, schedule, expert_ck, *, workers, fraction, timeout, trials_filter=None, debug=False, server_cpus='8-15'):
    import run_ex1_scope_changed as ex1run
    import expert_ex1
    sys.path.insert(0, str(HERE.parent))
    from anchor_flow.run_case import S2WebsocketClientPolicy, _run_trial
    c = CONDITIONS[name]
    tasks, baseline = ex1run.load_baseline('libero90')
    cdir = out / name; (cdir / 'trials').mkdir(parents=True, exist_ok=True); (out / 'servers').mkdir(exist_ok=True)
    existing = {(int(r['task_id']), int(r['trial_index'])): r for r in cl1.read_trials(cdir)}
    want = [(t, i) for t in tasks for i in range(10)]
    if trials_filter:
        want = [w for w in want if trials_filter(w[0]['task_id'], w[1])]
    pending = [(t, i) for t, i in want if existing.get((t['task_id'], i), {}).get('status') != 'complete']
    cl1.log(out, 'condition_start', condition=name, pending=len(pending), existing=len(existing))
    if not pending:
        return
    p5 = expert_ex1.load_locked_key2_checkpoints()['p5']['checkpoint']
    port = cl1.free_port(); env = os.environ.copy()
    env.update(SKILLNET_SUBTASK_GENERATE='0', XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
               TORCH_MEM_FRACTION=str(fraction), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CL1_DEBUG_FEATURES='1' if debug else '0')
    slog = (out / 'servers' / f'{name}-{port}.log').open('a')
    cmd = ['taskset', '-c', server_cpus, 'nice', '-n', '10', sys.executable, str(HERE / 'serve_expert_ex2.py'),
           '--mode', c['mode'], '--candidate-cache', str(cl1.CANDIDATES), '--retrieval-cache', str(cl1.EX2 / 'retrieval'),
           '--schedule', str(schedule), '--confidence', str(cl1.EX2 / 'confidence/confidence_head.pt'),
           '--pi05-checkpoint', str(cl1.PI05), '--expert-checkpoint', str(expert_ck), '--p5-checkpoint', str(p5), '--port', str(port),
           '--kq-query-checkpoint', str(c['query']), '--kq-keys', str(cl1.KQF / 'data/x1_keys.npy'),
           '--ekq-bundle', str(c['bundle']), '--ekq-gripper', c['gripper']]
    proc = subprocess.Popen(cmd, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
    done = 0
    try:
        deadline = time.monotonic() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'{name} server exited; see {slog.name}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'{name} server did not become ready')
            try:
                cl = S2WebsocketClientPolicy('127.0.0.1', port); cl.close(); break
            except (ConnectionError, OSError, TimeoutError):
                time.sleep(2)
        cl1.log(out, 'server_ready', condition=name, port=port)
        jobs = [{'host': '127.0.0.1', 'port': port, 'suite': 'libero_90', 'task_id': int(t['task_id']), 'task_key': t['name'],
                 'trial_index': int(i), 'seed_base': 7, 'max_steps': 520, 'wait_steps': 10, 'replan_steps': 5,
                 'condition': f'EKQ_{name}', 'checkpoint': str(expert_ck), 'timeout': timeout,
                 'memory_snapshot': 'LIBERO-40 4,516 variable-length DCT3 units (EX2 bank order)',
                 'query_text_source': 'instruction_only'} for t, i in pending]
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(_run_trial, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    cl1.log(out, 'trial_error', condition=name, task_id=j['task_id'], trial=j['trial_index'], error=repr(exc)[:500])
                    continue
                r['condition'] = f'EKQ_{name}'; r['ekq_condition'] = name
                r['baseline_success'] = bool(baseline[(int(r['task_id']), int(r['trial_index']))]['success'])
                with (cdir / 'trials.jsonl').open('a') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
                (cdir / 'trials' / f"task{int(r['task_id']):02d}-trial{int(r['trial_index']):02d}.json").write_text(json.dumps(r, ensure_ascii=False) + '\n')
                done += 1
                cl1.log(out, 'trial', condition=name, task_id=int(r['task_id']), trial=int(r['trial_index']), status=r.get('status'),
                        success=bool(r.get('success')), done=done, pending=len(pending))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 15)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
        slog.close()
    cl1.log(out, 'condition_end', condition=name, done=done)


def summarize(out: Path):
    import run_ex1_scope_changed as ex1run
    tasks, baseline = ex1run.load_baseline('libero90')
    keys = [(t['task_id'], i) for t in tasks for i in range(10)]
    succ = {}
    rows = {}
    for n in CONDITIONS:
        rows[n] = {(int(r['task_id']), int(r['trial_index'])): r for r in cl1.read_trials(out / n) if r.get('status') == 'complete'}
        succ[n] = {k: bool(rows[n][k]['success']) for k in keys if k in rows[n]}
    for n in ('C1a', 'C1b', 'C3'):
        succ[n] = {(int(r['task_id']), int(r['trial_index'])): bool(r['success']) for r in cl1.read_trials(cl1.OUT / n) if r.get('status') == 'complete'}
    succ['pi05'] = {k: bool(baseline[k]['success']) for k in keys}
    res = {'setting': 'CL1 と同じ（LIBERO-90 Eval 10 task × 10、seed_base 7、初期状態 0..9、replan 5、wait 10、520 step）',
           'conditions': {}, 'mcnemar': {}, 'diagnostics': {}}
    for n, s in succ.items():
        res['conditions'][n] = {'label': CONDITIONS[n]['label'] if n in CONDITIONS else n, 'completed': len(s), 'successes': int(sum(s.values())),
                                'tasks': {str(t['task_id']): sum(s.get((t['task_id'], i), False) for i in range(10)) for t in tasks}}
    for a, b in (('C1a', 'E1a'), ('C1b', 'E1b'), ('C1a', 'E1a_pg'), ('E1a_pg', 'E1a'), ('pi05', 'E1a'), ('pi05', 'E1b'), ('pi05', 'E1a_pg'), ('E1a', 'E1b')):
        common = [k for k in keys if k in succ.get(a, {}) and k in succ.get(b, {})]
        if common:
            res['mcnemar'][f'{b} - {a}'] = {'pairs': len(common), **cl1.mcnemar_exact([succ[a][k] for k in common], [succ[b][k] for k in common])}
    for n in CONDITIONS:
        qs = [q for r in rows[n].values() for q in r.get('selected_units_and_t0', []) if isinstance(q, dict) and 'unit_id' in q]
        if not qs:
            continue
        t0 = np.asarray([q['t0'] for q in qs]); eh = np.asarray([q['confidence_error'] for q in qs])
        fb = np.asarray([bool(q.get('fallback_pi05')) for q in qs])
        gsw = []
        for r in rows[n].values():
            g = [q.get('gripper_chunk') for q in r.get('selected_units_and_t0', []) if isinstance(q, dict) and q.get('gripper_chunk')]
            ex = [np.sign(x[0]) for x in g]
            gsw.append(int(sum(a != b for a, b in zip(ex, ex[1:]))))
        res['diagnostics'][n] = {'queries': int(len(qs)), 't0_quantiles': {str(q): float(np.quantile(t0, q)) for q in (.05, .5, .95)},
                                 'ehat_quantiles': {str(q): float(np.quantile(eh, q)) for q in (.05, .5, .95)},
                                 'fallback_rate': float(fb.mean()), 'gripper_first_step_changes_per_trial_mean': float(np.mean(gsw)) if gsw else None,
                                 'success_mean_t0': float(np.mean([np.mean([q['t0'] for q in r['selected_units_and_t0'] if 'unit_id' in q]) for r in rows[n].values() if r['success']] or [np.nan])),
                                 'failure_mean_t0': float(np.mean([np.mean([q['t0'] for q in r['selected_units_and_t0'] if 'unit_id' in q]) for r in rows[n].values() if not r['success']] or [np.nan]))}
    res['status'] = 'complete' if all(res['conditions'][n]['completed'] == 100 for n in CONDITIONS) else 'partial'
    cl1.OUT  # noqa: B018
    (out / 'results.json').write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float) + '\n')
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'summarize'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--conditions', default='E1a,E1b')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--fraction', type=float, default=.22)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    conds = [c for c in a.conditions.split(',') if c]
    if len(conds) > 2:
        raise SystemExit('at most 2 servers at once (GPU shared with the EX3 Expert retraining; 4 caused OOM in CL1)')
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == 'summarize':
        r = summarize(a.out); print(json.dumps({n: v['successes'] for n, v in r['conditions'].items()})); return
    schedule = cl1.OUT / 'schedule.json'
    ck = json.loads(schedule.read_text())['expert']['checkpoint']
    target = a.out / 'smoke' if a.command == 'smoke' else a.out
    filt = (lambda t, i: t == 10 and i == 0) if a.command == 'smoke' else None
    cpus = ['8-15', '16-23', '24-31']

    def one(n, cpu):
        try:
            run_condition(n, target, schedule, ck, workers=1 if a.command == 'smoke' else a.workers, fraction=a.fraction,
                          timeout=a.timeout, trials_filter=filt, debug=a.command == 'smoke', server_cpus=cpu)
        except Exception as exc:
            cl1.log(target, 'condition_failed', condition=n, error=repr(exc)[:500])
    ths = [threading.Thread(target=one, args=(n, cpus[i])) for i, n in enumerate(conds)]
    [t.start() for t in ths]; [t.join() for t in ths]
    if a.command == 'run':
        summarize(a.out)
    cl1.log(target, 'run_end')


if __name__ == '__main__':
    main()
