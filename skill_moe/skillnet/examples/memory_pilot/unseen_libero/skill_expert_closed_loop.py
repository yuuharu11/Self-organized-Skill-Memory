"""Light closed-loop check of the current memory pipeline on the 10 unseen LIBERO-90 tasks.

Per query: π0.5 proposes a chunk (normal path, same seeds as baseline-01). The K1 history query
(P5_hist: last 10 executed normalized actions + current normalized state) picks one memory shape
from the 1595-unit pool (holdout tasks excluded, same pool as the offline eval). The shape is posed
with the Procrustes rotation+scale that best matches π0.5's six axes (θπ0.5). Gripper = π0.5.
No unseen ground truth is used anywhere. Paired with baseline-01 (normal π0.5, 32/100).

  python memory_closed_loop.py serve --checkpoint CKPT --port P      (server)
  python memory_closed_loop.py run --output OUT --checkpoint CKPT     (runner)
  python memory_closed_loop.py check                                  (offline consistency check)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = (STATE_DIR / 'outputs/memory-pilot/unseen-libero')
K1 = ROOT / 'key-query-01' / 'production'
BASELINE = ROOT / 'baseline-01'
REPLAN = 5
# LIBERO-10 task id -> LIBERO-40 task index (matched by instruction text); used only for own-task exclusion.
LONG_TO_40 = {0: 5, 1: 7, 2: 3, 3: 8, 4: 0, 5: 9, 6: 1, 7: 4, 8: 6, 9: 2}


class MemorySelector:
    """K1 P5_hist query/key + pool; CPU torch."""
    def __init__(self):
        import torch
        import key_query_probe as kq
        import theta_regression as tr
        import variable_shape_memory as vs
        self.torch, self.tr, self.vs = torch, tr, vs
        sel = json.loads((K1 / 'selection_summary.json').read_text())
        ck = torch.load(sel['selected_by_mode']['P5_hist']['checkpoint'], map_location='cpu', weights_only=False)
        net = kq.ProbeNetwork('P5_hist').module; net.load_state_dict(ck['state_dict']); net.eval()
        self.net = net
        p0 = tr.candidate_shape_pool(rep='C_vm_none')
        keep = ~np.isin(p0['source_task'], sel['holdout_tasks'])
        self.shapes = p0['shape'][keep].astype(np.float32); self.src_task = p0['source_task'][keep]
        if os.environ.get('MEM_POOL'):  # alternative pool (e.g. whole / clustered units) with its own trained P5_hist query
            alt = np.load(os.environ['MEM_POOL']); self.shapes = alt['shape'].astype(np.float32); self.src_task = alt['source_task']
            net.load_state_dict(torch.load(os.environ['MEM_CKPT'], map_location='cpu', weights_only=False)['state_dict']); net.eval()
        with torch.no_grad():
            k = net.key(torch.from_numpy(self.shapes.reshape(len(self.shapes), -1)))
            self.keys = torch.nn.functional.normalize(k, dim=-1)

    def pick(self, history, state, exclude_task=None):
        """history [N,10,7] normalized (zeros if unavailable), state [N,8] normalized -> indices [N]."""
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(np.concatenate([history.reshape(len(history), -1), state], -1).astype(np.float32))
            q = t.nn.functional.normalize(self.net.query(x), dim=-1)
            sc = q @ self.keys.T
            if exclude_task is not None: sc[:, t.from_numpy(self.src_task == exclude_task)] = -1e9
            self.last_scores = sc.numpy()
            return sc.argmax(-1).numpy()

    def confidence(self, scores, temperature=.05):
        """scores [M] cosine (excluded = -1e9). Selection-time confidence under the K1-selected temperature."""
        v = scores[scores > -1e8]; z = (v - v.max()) / temperature; p = np.exp(z); p /= p.sum()
        top = np.sort(v)[::-1]
        return {'p_top1': float(p.max()), 'margin': float(top[0] - top[1]), 'eff_candidates': float(np.exp(-(p * np.log(p + 1e-30)).sum())),
                'cos_top1': float(top[0])}

    def pose_to(self, idx, ref_norm):
        """Pose shapes[idx] with Procrustes onto ref_norm [N,10,>=6] (normalized). Returns normalized [N,10,7]."""
        import theta_flow as tf
        sh = self.shapes[idx]
        theta = tf.batched_procrustes(sh[:, :, :6], self.vs.to_physical(ref_norm[:, :, :6]))[0]
        return tf.decode_many(sh, theta, self.vs)


# ------------------------------------------------------------------ server
def make_policy(checkpoint):
    sys.path.insert(0, str(HERE.parent / 'pi05_memory'))
    from serve import MemoryPolicy

    class MemoryShapePolicy(MemoryPolicy):
        def __init__(self, ckpt):
            super().__init__(ckpt, ckpt, normal_only=True)
            self.sel = MemorySelector(); self.hist = {}
            self.t0 = float(os.environ.get('MEM_T0', '0'))  # >0: denoise the memory draft with the π0.5 Expert from t0
            if self.t0 > 0:
                from openpi.shared.nnx_utils import module_jit
                if os.environ.get('MEM_EXPERT'):  # optional fine-tuned Expert (compressed-denoise checkpoint)
                    import flax.nnx as nnx, orbax.checkpoint as ocp
                    from openpi.shared.nnx_utils import PathRegex
                    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
                    with ocp.PyTreeCheckpointer() as c: part = c.restore(Path(os.environ['MEM_EXPERT']))
                    st = nnx.state(self.model, filt); st.replace_by_pure_dict(self.jax.tree.map(self.jnp.asarray, part)); nnx.update(self.model, st)
                self.bridge = module_jit(self.model.sample_bridge, static_argnames=('num_steps',))

        def infer(self, observation):
            from openpi.models.model import Observation
            start = time.monotonic()
            raw = {k: observation[k] for k in ['observation/image', 'observation/wrist_image', 'observation/state', 'prompt']}
            row = self.transform(raw)
            obs = Observation(
                images={k: self.jnp.asarray(v)[None].astype(self.jnp.float32) / 255 * 2 - 1 for k, v in row['image'].items()},
                image_masks={k: self.jnp.asarray([v]) for k, v in row['image_mask'].items()},
                state=self.jnp.asarray(row['state'])[None],
                tokenized_prompt=self.jnp.asarray(row['tokenized_prompt'])[None],
                tokenized_prompt_mask=self.jnp.asarray(row['tokenized_prompt_mask'])[None])
            key = self.jax.random.key(int(observation['s2_rng_seed']))
            normal = np.asarray(self.normal(key, obs, num_steps=10))[0].astype(np.float32)  # [10,32] normalized
            if observation['s2_condition'] == 'normal':
                actions = self.unnormalize({'actions': normal, 'state': row['state']})['actions'][:, :7]
                return {'actions': actions.astype(np.float32), 's2_debug': {'condition': 'normal', 'seconds': time.monotonic() - start}}
            if observation['s2_condition'] == 'lowpass':  # ablation: π0.5's own chunk reduced to DCT-3 per axis, no memory
                from scipy.fft import dct, idct
                c = dct(normal[:, :6], type=2, norm='ortho', axis=0); c[3:] = 0
                out = normal.copy(); out[:, :6] = idct(c, type=2, norm='ortho', axis=0)
                actions = self.unnormalize({'actions': out, 'state': row['state']})['actions'][:, :7]
                return {'actions': actions.astype(np.float32), 's2_debug': {'condition': 'lowpass', 'seconds': time.monotonic() - start}}
            suite, task_id = observation['s2_episode_id'].split(':')[:2]
            excl = LONG_TO_40[int(task_id)] if suite == 'libero_10' else None
            ep = observation['s2_episode_id']; qid = int(observation['s2_query_id']) % 1000
            if qid == 0: self.hist[ep] = []
            executed = self.hist.setdefault(ep, [])
            history = np.stack(executed[-10:]) if len(executed) >= 10 else np.zeros((10, 7), np.float32)
            state = np.asarray(row['state'], np.float32)[:8]
            idx = int(self.sel.pick(history[None], state[None], excl)[0])
            mem = self.sel.pose_to(np.array([idx]), normal[None, :, :7])[0]
            out = normal.copy(); out[:, :6] = mem[:, :6]          # gripper (dim 6) stays π0.5
            draft = out.copy()
            if self.t0 > 0:
                nkey = self.jax.random.fold_in(self.jax.random.key(int(observation['s2_rng_seed'])), 7)
                eps = self.jax.random.normal(nkey, (1, 10, 32))
                out = np.asarray(self.bridge(self.jax.random.key(0), obs, self.jnp.asarray(draft[None]), self.jnp.asarray([self.t0]),
                                             num_steps=10, noise=eps))[0].astype(np.float32)
            executed.extend(list(out[:REPLAN, :7]))
            actions = self.unnormalize({'actions': out, 'state': row['state']})['actions'][:, :7]
            if not np.isfinite(actions).all(): raise FloatingPointError('nonfinite action')
            import theta_flow as tf
            ok = self.sel.last_scores[0] > -1e8
            res = tf.batched_procrustes(self.sel.shapes[ok][:, :, :6], np.repeat(self.sel.vs.to_physical(normal[None, :, :6]), ok.sum(), 0))[2]
            chosen = res[np.flatnonzero(ok) == idx][0]
            conf = self.sel.confidence(self.sel.last_scores[0])
            conf.update(fit_resid_chosen=float(chosen), fit_resid_best=float(res.min()), fit_rank_pct=float((res < chosen).mean()))
            dbg = {'condition': 'memory_k1_hist_theta_pi05', 'unit': idx, 'has_history': len(executed) - REPLAN >= 10, **conf,
                   'six_gap_to_pi05': float(((out[:, :6] - normal[:, :6]) ** 2).mean()), 't0': self.t0,
                   'six_gap_to_draft': float(((out[:, :6] - draft[:, :6]) ** 2).mean()),
                   'draft_gap_to_pi05': float(((draft[:, :6] - normal[:, :6]) ** 2).mean()), 'seconds': time.monotonic() - start}
            return {'actions': actions.astype(np.float32), 's2_debug': dbg}

    return MemoryShapePolicy(checkpoint)


def serve(a):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = make_policy(a.checkpoint)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=a.port, metadata={'policy': 'memory_k1_hist_theta_pi05'}).serve_forever()


# ------------------------------------------------------------------ runner
def run(a):
    import subprocess, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import run as base
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((BASELINE / 'plan.json').read_text())
    tasks = [t['task_id'] for t in plan['selected_tasks']] if a.suite == 'libero_90' else list(range(10))
    conds = a.conditions.split(',')
    servers, logs = [], []
    for i in range(a.servers):
        port = base.free_port(); folder = out / 'servers' / str(i); folder.mkdir(parents=True, exist_ok=True)
        log = (folder / 'server.log').open('a'); logs.append(log)
        env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                                            XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.15')
        p = subprocess.Popen(['taskset', '-c', '8-15', 'nice', '-n', '10', sys.executable, str(Path(__file__).resolve()), 'serve',
                              '--checkpoint', str(a.checkpoint), '--port', str(port)], env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        servers.append((p, port))
    try:
        for p, port in servers: base.ready(p, port)
        done = base.read_terminal(out / 'production' / 'rollouts.jsonl')
        jobs = [(c, t, i) for c in conds for t in tasks for i in range(a.inits) if (t, i) not in base.read_terminal(out / c / 'rollouts.jsonl')]
        total = len(tasks) * a.inits; lock = threading.Lock()

        def execute(cond, task, init, port):
            job = {'host': '127.0.0.1', 'port': port, 'suite': a.suite, 'task_id': task, 'trial_index': init, 'seed_base': 7,
                   'condition': cond, 'memory_snapshot': 'none', 'query_text_source': 'instruction_only',
                   'checkpoint': str(a.checkpoint), 'wait_steps': 10, 'replan_steps': REPLAN, 'max_steps': 520, 'record': init < 2}
            name = f'task{task:02d}-init{init:02d}'; root = out / cond / 'attempts' / name
            existing = list(root.glob('attempt-*')) if root.exists() else []
            first = max([int(x.name.split('-')[-1]) for x in existing], default=0) + 1
            r = None
            for attempt in range(first, first + 2):
                r = base.trial(job, root / f'attempt-{attempt}', a.timeout, attempt)
                base.append(out / cond / 'attempts.jsonl', r)
                if r['status'] == 'complete': break
            r.update(task_id=task, trial_index=init, task_suite=a.suite)
            base.append(out / cond / 'rollouts.jsonl', r)
            with lock:
                d = base.read_terminal(out / cond / 'rollouts.jsonl')
                base.write(out / cond / 'progress.json', {'processed': len(d), 'scheduled': total,
                           'successes': sum(x.get('status') == 'complete' and x.get('success', False) for x in d.values()), 'updated_unix': time.time()})

        def lane(slot):
            p, port = servers[slot % a.servers]
            for cond, task, init in jobs[slot::a.servers * 2]:
                if p.poll() is not None: raise RuntimeError('server died')
                execute(cond, task, init, port)
        for c in conds: (out / c).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=a.servers * 2) as pool:
            for f in as_completed([pool.submit(lane, k) for k in range(a.servers * 2)]): f.result()
        report(out, a.inits, a.suite)
        paired_report(out, a.inits, a.suite)
    finally:
        for p, _ in servers: base.terminate(p)
        for l in logs: l.close()


def report(out, inits, suite='libero_90'):
    import run as base
    mem = base.read_terminal(out / 'memory' / 'rollouts.jsonl')
    ref = BASELINE / 'production' / 'rollouts.jsonl' if suite == 'libero_90' else out / 'normal' / 'rollouts.jsonl'
    bl = base.read_terminal(ref)
    keys = sorted(k for k in mem if k in bl and k[1] < inits)
    s = lambda r: r.get('status') == 'complete' and bool(r.get('success'))
    rows = []; L = ['# Closed-loop: memory shape (K1 history query) posed to π0.5, gripper π0.5', '',
                    '| task | memory | π0.5 (same seeds) |', '|---|---:|---:|']
    for t in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == t]
        L.append(f'| {t} | {sum(s(mem[k]) for k in ks)}/{len(ks)} | {sum(s(bl[k]) for k in ks)}/{len(ks)} |')
    m = sum(s(mem[k]) for k in keys); b = sum(s(bl[k]) for k in keys)
    both = sum(s(mem[k]) and s(bl[k]) for k in keys); only_m = sum(s(mem[k]) and not s(bl[k]) for k in keys); only_b = sum(s(bl[k]) and not s(mem[k]) for k in keys)
    errs = sum(mem[k].get('status') != 'complete' for k in keys)
    L += ['', f'Total memory {m}/{len(keys)}, π0.5 {b}/{len(keys)}; both {both}, memory-only {only_m}, π0.5-only {only_b}; technical errors {errs}.']
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    (out / 'summary.json').write_text(json.dumps({'memory': m, 'pi05': b, 'n': len(keys), 'memory_only': only_m, 'pi05_only': only_b, 'errors': errs}))
    print('\n'.join(L))


def paired_report(out, inits, suite='libero_90'):
    """Pair new skill-Expert rollouts with frozen baseline/memory/t0=.4 results."""
    import run as base
    from scipy.stats import binomtest
    here = base.read_terminal(out / 'memory' / 'rollouts.jsonl')
    root = ROOT
    refs = {
        'baseline_pi05': BASELINE / 'production' / 'rollouts.jsonl',
        'unit_08': ROOT / 'closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl',
        'old_den04': ROOT / 'closed-loop-memory-01/libero90_unit_08_dn04/memory/rollouts.jsonl',
    }
    success = lambda x: bool(x.get('status') == 'complete' and x.get('success'))
    result={}
    for name,path in refs.items():
        ref=base.read_terminal(path);keys=sorted(k for k in here if k in ref and k[1] < inits)
        b=sum(success(here[k]) and not success(ref[k]) for k in keys)
        c=sum(success(ref[k]) and not success(here[k]) for k in keys)
        result[name]={'paired_n':len(keys),'skill_success':sum(success(here[k]) for k in keys),
            'reference_success':sum(success(ref[k]) for k in keys),'both_success':sum(success(here[k]) and success(ref[k]) for k in keys),
            'skill_only':b,'reference_only':c,'mcnemar_exact_p':float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.0}
    p=out/'paired_comparison.json';p.write_text(json.dumps(result,indent=2)+'\n')
    with (out/'report.md').open('a') as f:
        f.write('\n## Paired same-trial comparison\n\n')
        f.write('| reference | n | skill Expert | reference | both | skill-only | reference-only | exact McNemar p |\n|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for name,v in result.items():
            f.write(f"| {name} | {v['paired_n']} | {v['skill_success']} | {v['reference_success']} | {v['both_success']} | {v['skill_only']} | {v['reference_only']} | {v['mcnemar_exact_p']:.6g} |\n")
    summary_path=out.parent/'report.md'
    if summary_path.exists():
        with summary_path.open('a') as f:
            f.write('\n## LIBERO-90 閉ループ（100試行、同一seed対応）\n\n')
            f.write('| 基準 | skill Expert | 基準 | 両方成功 | skillのみ | 基準のみ | exact McNemar p |\n|---|---:|---:|---:|---:|---:|---:|\n')
            for name,v in result.items():
                f.write(f"| {name} | {v['skill_success']}/{v['paired_n']} | {v['reference_success']}/{v['paired_n']} | {v['both_success']} | {v['skill_only']} | {v['reference_only']} | {v['mcnemar_exact_p']:.6g} |\n")


def check():
    """Offline: runtime selector reproduces K1 rank_P5_hist top-1; θπ0.5 posing six-axis error."""
    import theta_flow as tf
    sel = MemorySelector()
    z = np.load(K1 / 'evaluation' / 'rows.npz'); c = np.load(tf.OUT / 'eval_cond.npz')
    assert np.array_equal(z['row_ids'], c['row_ids'])
    pick = sel.pick(c['history'].astype(np.float32), c['state'].astype(np.float32))
    print('top-1 agreement with offline K1 P5_hist:', float((pick == z['rank_P5_hist'][:, 0]).mean()))
    mem = sel.pose_to(pick, z['query_pi05'][:, :, :7].astype(np.float32))
    y = z['target'][:, :, :6]
    print('six-axis: memory posed to π0.5 %.4f, π0.5 %.4f' % (((mem[:, :, :6] - y) ** 2).mean() * 6 / 7, ((z['query_pi05'][:, :, :6] - y) ** 2).mean() * 6 / 7))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('serve'); s.add_argument('--checkpoint', type=Path, required=True); s.add_argument('--port', type=int, required=True)
    r = sp.add_parser('run'); r.add_argument('--output', type=Path, required=True); r.add_argument('--checkpoint', type=Path, required=True)
    r.add_argument('--suite', default='libero_90'); r.add_argument('--conditions', default='memory')
    r.add_argument('--servers', type=int, default=4); r.add_argument('--timeout', type=int, default=900); r.add_argument('--inits', type=int, default=10)
    sp.add_parser('check'); rp = sp.add_parser('report'); rp.add_argument('--output', type=Path, required=True); rp.add_argument('--suite', default='libero_90')
    a = p.parse_args()
    if a.cmd == 'serve': serve(a)
    elif a.cmd == 'run': run(a)
    elif a.cmd == 'check': check()
    else: report(a.output, 10, a.suite)
