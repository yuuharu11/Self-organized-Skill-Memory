#!/usr/bin/env python3
"""CL1: KQ-final motion-key drafts + EX2 Expert, LIBERO-90 Eval closed loop.

Reuses the EX2 server (serve_expert_ex2.py, KQ_* modes) and the shared
anchor_flow trial runner with the lib-closed-01 / EX1 / EX2 setting
(seed_base 7, public init states 0..9, replan 5, wait 10, 520 steps).
Every finished trial is appended (fsync) to <cond>/trials.jsonl and written to
<cond>/trials/task<id>-trial<i>.json, so a rerun resumes from the missing trials.

Subcommands:
  parity   offline: server-path scoring on the KQ-final unseen rows == export top-1
  run      closed loop for one or more conditions (resumable)
  smoke    one trial per condition with feature logging, then in-loop parity
  summarize  tables, McNemar, t0/confidence, provenance -> results.json
"""
from __future__ import annotations
import argparse, json, math, os, socket, subprocess, sys, threading, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'closed-loop-kq-01'
KQF = ROOT / 'key-motion-01/final'
EX2 = ROOT / 'expert-ex2-01/production-05'
CANDIDATES = ROOT / 'expert-ex1-01/production-03/scope-change-01/eval-drafts/closedloop_p5_bank.npz'
PI05 = (STATE_DIR / 'checkpoints/openpi-assets/checkpoints/pi05_libero')
CONDITIONS = {
    'C1a': dict(mode='KQ_AFIX_top1', expert=True, query=KQF / 'best/Afix_x1_fixed_e_h_cteff.pt',
                export='Afix_x1_fixed_e_h_cteff', label='K-x1 × E-H 注意クエリ（最終行）＋Expert'),
    'C1b': dict(mode='KQ_HMLP_top1', expert=True, query=KQF / 'best/H_x1_mk1b_cteff_tau0.01_s1.pt',
                export='H_x1_mk1b_cteff_tau0.01_s1', label='K-x1 × 履歴＋状態MLP（CT版）＋Expert'),
    'C2': dict(mode='KQ_AFIX_top1', expert=False, query=KQF / 'best/Afix_x1_fixed_e_h_cteff.pt',
               export='Afix_x1_fixed_e_h_cteff', label='C1a の下書きのまま（Expert なし）'),
    'C3': dict(mode='D5_HS_top1', expert=True, query=None, export=None, label='D5 履歴＋状態MLP＋Expert'),
}


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(out: Path, event: str, **kw):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'progress.log').open('a') as f:
        f.write(json.dumps({'utc': utc(), 'event': event, **kw}, ensure_ascii=False) + '\n')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


# ---------------------------------------------------------------------------
# offline parity of the server scoring path
# ---------------------------------------------------------------------------

def kq_scores(model, keys, hist, mask, state, torch, device):
    """Exactly the server's KQ_* scoring (serve_expert_ex2.EX2Policy.infer)."""
    import key_final as kf
    feature = np.concatenate([hist.reshape(len(hist), -1), mask.astype(np.float32).reshape(len(hist), -1), state], 1)
    T = lambda a, dt=torch.float32: torch.as_tensor(a, device=device, dtype=dt)
    with torch.no_grad():
        q = model(history=T(hist), history_mask=T(mask, torch.bool), state=T(state), hs=T(feature))
        return kf.score(q, keys, 'dot', model.logit_scale).float().cpu().numpy()


def load_kq(path, device):
    import torch, key_final as kf
    blob = torch.load(path, map_location='cpu', weights_only=False)
    m = kf.make_query(blob['cfg']).to(device); m.load_state_dict(blob['query']); m.eval()
    return m


def parity(out: Path):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keys = torch.as_tensor(np.load(KQF / 'data/x1_keys.npy'), device=device, dtype=torch.float32)
    d = KQF / 'data'
    res = {}
    for name in ('C1a', 'C1b'):
        c = CONDITIONS[name]; m = load_kq(c['query'], device)
        for split in ('unseen', 'dev'):
            h = np.load(d / f'{split}__history.npy').astype(np.float32); mk = np.load(d / f'{split}__history_mask.npy').astype(bool)
            st = np.load(d / f'{split}__state.npy').astype(np.float32)
            sc = np.concatenate([kq_scores(m, keys, h[s:s + 64], mk[s:s + 64], st[s:s + 64], torch, device) for s in range(0, len(h), 64)])
            with np.load(KQF / 'export' / f"{c['export']}__{split}_scores.npz") as z:
                ref = z['top1_unit']; refscore = z['top1_score']; mse = z['top1_arm6_mse']
            top = sc.argmax(1)
            res[f'{name}:{split}'] = {'rows': int(len(top)), 'top1_agree': float((top == ref).mean()),
                                      'max_abs_top1_score_diff': float(np.abs(sc.max(1) - refscore).max()),
                                      'export_top1_arm6_mse_mean': float(mse.mean())}
    # Candidate-bank and feature conventions shared with the D5 server path.
    with np.load(CANDIDATES) as z:
        src = z['source']; win = z['unit_windows']
    res['bank_source_order_equal'] = bool(np.array_equal(np.load(d / 'bank__source.npy'), src))
    res['bank_window_max_abs_diff'] = float(np.abs(np.load(d / 'bank_actions.npy') - win).max())
    st = {k: np.load(d / f'start__{k}.npy') for k in ('history', 'history_mask', 'state')}
    hs = np.concatenate([st['history'].reshape(len(st['history']), -1), st['history_mask'].astype(np.float32), st['state']], -1)
    res['start_hs_vs_ex2_candidate_history_state_max_abs_diff'] = float(np.abs(hs - np.load(EX2 / 'retrieval/candidate_history_state.npy')).max())
    (out / 'parity_offline.json').write_text(json.dumps(res, indent=2) + '\n')
    log(out, 'parity_offline', **{k: v for k, v in res.items() if not isinstance(v, dict)})
    print(json.dumps(res, indent=2))
    return res


def inloop_parity(out: Path, conds):
    """Recompute top-1 from the logged query feature of each smoke query."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keys = torch.as_tensor(np.load(KQF / 'data/x1_keys.npy'), device=device, dtype=torch.float32)
    res = {}
    for name in conds:
        c = CONDITIONS[name]
        if c['query'] is None:
            continue
        m = load_kq(c['query'], device)
        rows = read_trials(out / 'smoke' / name)
        qs = [q for r in rows for q in r.get('selected_units_and_t0', []) if 'query_feature' in q]
        if not qs:
            res[name] = {'queries': 0}; continue
        f = np.asarray([q['query_feature'] for q in qs], np.float32)
        h, mk, st = f[:, :70].reshape(-1, 10, 7), f[:, 70:80] > .5, f[:, 80:88]
        sc = np.concatenate([kq_scores(m, keys, h[s:s + 64], mk[s:s + 64], st[s:s + 64], torch, device) for s in range(0, len(h), 64)])
        logged = np.asarray([q['unit_id'] for q in qs])
        # Value ranges vs the offline unseen rows (same normalization => similar ranges).
        ref_state = np.load(KQF / 'data/unseen__state.npy'); ref_hist = np.load(KQF / 'data/unseen__history.npy')
        res[name] = {'queries': int(len(qs)), 'top1_agree': float((sc.argmax(1) == logged).mean()),
                     'max_abs_top1_score_diff': float(np.abs(sc.max(1) - np.asarray([q['top1_score'] for q in qs])).max()),
                     'loop_state_mean': st.mean(0).round(3).tolist(), 'offline_state_mean': ref_state.mean(0).round(3).tolist(),
                     'loop_state_std': st.std(0).round(3).tolist(), 'offline_state_std': ref_state.std(0).round(3).tolist(),
                     'loop_hist_absmean': float(np.abs(h[mk]).mean()), 'offline_hist_absmean': float(np.abs(ref_hist[np.load(KQF / 'data/unseen__history_mask.npy').astype(bool)]).mean())}
    (out / 'parity_inloop.json').write_text(json.dumps(res, indent=2) + '\n')
    log(out, 'parity_inloop', **{k: v.get('top1_agree') for k, v in res.items()})
    print(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# closed loop
# ---------------------------------------------------------------------------

def read_trials(cdir: Path):
    rows = {}
    p = cdir / 'trials.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[(int(r['task_id']), int(r['trial_index']))] = r
    return list(rows.values())


def run_condition(name: str, out: Path, schedule: Path, expert_ck: str, *, workers: int, fraction: float,
                  timeout: int, trials_filter=None, debug_features=False, server_cpus='8-15'):
    import run_ex1_scope_changed as ex1run
    import expert_ex1
    sys.path.insert(0, str(HERE.parent))
    from anchor_flow.run_case import S2WebsocketClientPolicy, _run_trial
    c = CONDITIONS[name]
    tasks, baseline = ex1run.load_baseline('libero90')
    cdir = out / name; (cdir / 'trials').mkdir(parents=True, exist_ok=True); (out / 'servers').mkdir(exist_ok=True)
    existing = {(int(r['task_id']), int(r['trial_index'])): r for r in read_trials(cdir)}
    want = [(t, i) for t in tasks for i in range(10)]
    if trials_filter:
        want = [w for w in want if trials_filter(w[0]['task_id'], w[1])]
    pending = [(t, i) for t, i in want if existing.get((t['task_id'], i), {}).get('status') != 'complete']
    log(out, 'condition_start', condition=name, pending=len(pending), existing=len(existing))
    if not pending:
        return
    p5 = expert_ex1.load_locked_key2_checkpoints()['p5']['checkpoint']
    port = free_port(); env = os.environ.copy()
    env.update(SKILLNET_SUBTASK_GENERATE='0', XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
               TORCH_MEM_FRACTION=str(fraction), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CL1_DEBUG_FEATURES='1' if debug_features else '0')
    slog = (out / 'servers' / f'{cdir.name}-{name}-{port}.log').open('a')
    cmd = ['taskset', '-c', server_cpus, 'nice', '-n', '10', sys.executable, str(HERE / 'serve_expert_ex2.py'),
           '--mode', c['mode'], '--candidate-cache', str(CANDIDATES), '--retrieval-cache', str(EX2 / 'retrieval'),
           '--schedule', str(schedule), '--confidence', str(EX2 / 'confidence/confidence_head.pt'),
           '--pi05-checkpoint', str(PI05), '--expert-checkpoint', str(expert_ck), '--p5-checkpoint', str(p5), '--port', str(port)]
    if c['query'] is not None:
        cmd += ['--kq-query-checkpoint', str(c['query']), '--kq-keys', str(KQF / 'data/x1_keys.npy')]
    if not c['expert']:
        cmd += ['--no-expert']
    proc = subprocess.Popen(cmd, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
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
        log(out, 'server_ready', condition=name, port=port)
        jobs = [{'host': '127.0.0.1', 'port': port, 'suite': 'libero_90', 'task_id': int(t['task_id']), 'task_key': t['name'],
                 'trial_index': int(i), 'seed_base': 7, 'max_steps': 520, 'wait_steps': 10, 'replan_steps': 5,
                 'condition': f'CL1_{name}', 'checkpoint': str(expert_ck), 'timeout': timeout,
                 'memory_snapshot': 'LIBERO-40 4,516 variable-length DCT3 units (EX2 bank order)',
                 'query_text_source': 'instruction_only'} for t, i in pending]
        done = 0
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(_run_trial, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # keep going; the trial stays pending for a rerun
                    log(out, 'trial_error', condition=name, task_id=j['task_id'], trial=j['trial_index'], error=repr(exc)[:500])
                    continue
                r['condition'] = f'CL1_{name}'; r['cl1_condition'] = name
                r['baseline_success'] = bool(baseline[(int(r['task_id']), int(r['trial_index']))]['success'])
                r['expert_checkpoint'] = str(expert_ck); r['query_checkpoint'] = None if c['query'] is None else str(c['query'])
                with (cdir / 'trials.jsonl').open('a') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
                (cdir / 'trials' / f"task{int(r['task_id']):02d}-trial{int(r['trial_index']):02d}.json").write_text(json.dumps(r, ensure_ascii=False) + '\n')
                done += 1
                log(out, 'trial', condition=name, task_id=int(r['task_id']), trial=int(r['trial_index']), status=r.get('status'),
                    success=bool(r.get('success')), steps=r.get('env_steps'), done=done, pending=len(pending))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 15)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
        slog.close()
    log(out, 'condition_end', condition=name, done=done)


def write_schedule(out: Path) -> tuple[Path, str]:
    sel = json.loads((out / 'expert-select/dev-checkpoint-selection-repaired.json').read_text())
    a = dict(sel['selected_by_condition']['A']); a['checkpoint'] = str(Path(a['checkpoint']).resolve())
    sched = {'status': 'complete', 'expert': {'condition': 'A', 'step': a['step'], 'checkpoint': a['checkpoint'],
             'dev_mse6_by_scale': a['dev_mse6_by_scale'],
             'selection': 'EX2 repaired Dev re-scoring (live NNX state) of production-05 train-A checkpoints, D1 P5 Dev draft; Eval unused'},
             'schedule': {'scale_c': 1.0, 'fallback_error_threshold': None, 'clip': [0.05, 0.6],
                          'rule': 't0 = clip(sqrt(predicted arm6 error), .05, .6); no π0.5 fallback'}}
    p = out / 'schedule.json'; p.write_text(json.dumps(sched, indent=2) + '\n')
    return p, a['checkpoint']


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def mcnemar_exact(a, b):
    """a,b aligned booleans. Returns wins(b only), losses(a only), two-sided exact p."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    w = int((b & ~a).sum()); l = int((a & ~b).sum()); n = w + l
    if n == 0:
        return {'b_only': w, 'a_only': l, 'p': 1.0}
    k = min(w, l)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {'b_only': w, 'a_only': l, 'p': p}


def summarize(out: Path):
    import run_ex1_scope_changed as ex1run
    import transfer_analysis as ta
    tasks, baseline = ex1run.load_baseline('libero90')
    forty, ninety, _ = ta.catalog()
    rows = {n: {(int(r['task_id']), int(r['trial_index'])): r for r in read_trials(out / n) if r.get('status') == 'complete'} for n in CONDITIONS}
    keys = [(t['task_id'], i) for t in tasks for i in range(10)]
    succ = {n: {k: bool(rows[n][k]['success']) for k in keys if k in rows[n]} for n in CONDITIONS}
    succ['pi05'] = {k: bool(baseline[k]['success']) for k in keys}
    res = {'setting': {'suite': 'libero_90', 'tasks': [t['task_id'] for t in tasks], 'trials_per_task': 10, 'seed_base': 7,
                       'initial_states': 'public 0..9', 'replan_steps': 5, 'wait_steps': 10, 'max_steps': 520,
                       'pi05_baseline': str(ex1run.BASELINE90), 'gripper': 'π0.5 normal output (EX2)'},
           'schedule': json.loads((out / 'schedule.json').read_text()) if (out / 'schedule.json').exists() else None,
           'conditions': {}, 'mcnemar': {}}
    for n, s in succ.items():
        per = {str(t['task_id']): {'name': t['name'], 'completed': sum(1 for i in range(10) if (t['task_id'], i) in s),
                                   'successes': sum(s.get((t['task_id'], i), False) for i in range(10))} for t in tasks}
        res['conditions'][n] = {'label': CONDITIONS[n]['label'] if n in CONDITIONS else '通常 π0.5（baseline-01 再掲）',
                                'completed': len(s), 'successes': int(sum(s.values())), 'tasks': per}
    for a, b in (('C3', 'C1a'), ('C3', 'C1b'), ('pi05', 'C1a'), ('pi05', 'C1b'), ('pi05', 'C3'), ('pi05', 'C2'), ('C2', 'C1a'), ('C1a', 'C1b')):
        common = [k for k in keys if k in succ[a] and k in succ[b]]
        if common:
            res['mcnemar'][f'{b} - {a}'] = {'pairs': len(common), **mcnemar_exact([succ[a][k] for k in common], [succ[b][k] for k in common])}
    # per-query diagnostics
    diag = {}
    for n in CONDITIONS:
        per_trial = []
        for k, r in rows[n].items():
            qs = [q for q in r.get('selected_units_and_t0', []) if isinstance(q, dict) and 'unit_id' in q]
            if not qs:
                continue
            q9 = ninety.get(k[0]) or ninety.get(str(k[0]))
            cats = [ta.category(q9, forty.get(int(q['unit_task']))) if q9 else 'unknown' for q in qs]
            units = [q['unit_id'] for q in qs]
            per_trial.append({'task_id': k[0], 'trial': k[1], 'success': bool(r['success']), 'queries': len(qs),
                              'mean_t0': float(np.mean([q['t0'] for q in qs])),
                              'mean_pred_error': float(np.mean([q['confidence_error'] for q in qs])),
                              'switches': int(sum(u != v for u, v in zip(units, units[1:]))),
                              'distinct_units': len(set(units)),
                              'same_verb_fraction': float(np.mean([c != 'other_verb' and c != 'unknown' for c in cats])),
                              'other_verb_fraction': float(np.mean([c == 'other_verb' for c in cats])),
                              'categories': {c: cats.count(c) for c in set(cats)},
                              't0_all': [q['t0'] for q in qs], 'err_all': [q['confidence_error'] for q in qs]})
        if not per_trial:
            continue
        t0 = np.concatenate([p['t0_all'] for p in per_trial]); er = np.concatenate([p['err_all'] for p in per_trial])
        allcats = {}
        for p in per_trial:
            for c, v in p['categories'].items():
                allcats[c] = allcats.get(c, 0) + v
        tot = sum(allcats.values())
        def grp(flag):
            g = [p for p in per_trial if p['success'] == flag]
            if not g:
                return None
            return {'trials': len(g), 'mean_t0': float(np.mean([p['mean_t0'] for p in g])),
                    'mean_pred_error': float(np.mean([p['mean_pred_error'] for p in g])),
                    'switches_per_query': float(np.mean([p['switches'] / max(p['queries'] - 1, 1) for p in g])),
                    'switches': float(np.mean([p['switches'] for p in g])),
                    'distinct_units': float(np.mean([p['distinct_units'] for p in g])),
                    'same_verb_fraction': float(np.mean([p['same_verb_fraction'] for p in g])),
                    'other_verb_fraction': float(np.mean([p['other_verb_fraction'] for p in g]))}
        diag[n] = {'queries': int(len(t0)),
                   't0_quantiles': {q: float(np.quantile(t0, q)) for q in (.05, .25, .5, .75, .95)},
                   't0_at_floor_fraction': float(np.mean(t0 <= .0501)), 't0_at_ceiling_fraction': float(np.mean(t0 >= .5999)),
                   'pred_error_quantiles': {q: float(np.quantile(er, q)) for q in (.05, .25, .5, .75, .95)},
                   'source_categories': {c: v / tot for c, v in sorted(allcats.items())},
                   'success': grp(True), 'failure': grp(False),
                   'per_trial': [{k: v for k, v in p.items() if k not in ('t0_all', 'err_all')} for p in per_trial]}
    res['diagnostics'] = diag
    res['offline_unseen_top1_arm6_mse'] = {
        'C1a/C2 draft (K-x1 × E-H, seed1)': .0936, 'C1a/C2 draft 3 seed mean': .0945,
        'C1b draft (K-x1 × HS-MLP CT, seed1)': .0836, 'C1b draft 3 seed mean': .0863,
        'C3 draft (D5, KEYGRID 原型)': .1011, 'C3 D5+Expert(production-04)': .0806, 'pi05': .1150,
        'note': 'KQ-final report.md / EX2 report.md; production-05 Expert offline values for the new drafts were not computed'}
    res['status'] = 'complete' if all(res['conditions'][n]['completed'] == 100 for n in CONDITIONS) else 'partial'
    (out / 'results.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n')
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('parity', 'smoke', 'run', 'summarize', 'inloop-parity'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--conditions', default='C1a,C1b,C3,C2')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--fraction', type=float, default=.16)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--parallel', action='store_true', help='run the listed conditions concurrently')
    a = p.parse_args()
    conds = [c for c in a.conditions.split(',') if c]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == 'parity':
        parity(a.out); return
    if a.command == 'summarize':
        r = summarize(a.out); print(json.dumps({n: v['successes'] for n, v in r['conditions'].items()})); return
    if a.command == 'inloop-parity':
        inloop_parity(a.out, conds); return
    schedule, ck = write_schedule(a.out)
    if a.command == 'smoke':
        target = a.out / 'smoke'
        def one(n):
            run_condition(n, target, schedule, ck, workers=1, fraction=a.fraction, timeout=a.timeout,
                          trials_filter=lambda t, i: t == 10 and i == 0, debug_features=True)
        ths = [threading.Thread(target=one, args=(n,)) for n in conds]
        [t.start() for t in ths]; [t.join() for t in ths]
        inloop_parity(a.out, conds); return
    def one(n):
        try:
            run_condition(n, a.out, schedule, ck, workers=a.workers, fraction=a.fraction, timeout=a.timeout)
        except Exception as exc:
            log(a.out, 'condition_failed', condition=n, error=repr(exc)[:500])
    if a.parallel:
        ths = [threading.Thread(target=one, args=(n,)) for n in conds]
        [t.start() for t in ths]; [t.join() for t in ths]
    else:
        for n in conds:
            one(n)
    summarize(a.out)
    log(a.out, 'run_end')


if __name__ == '__main__':
    main()
