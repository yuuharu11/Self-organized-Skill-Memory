#!/usr/bin/env python3
"""KQ2-CL closed loop: LIBERO-90 Eval, the CL1 10 tasks x 20 trials (trials 0..9 = CL1's initial
states / env seeds; 10..19 = the next public initial states, env seed 7 + trial).
Same protocol as CL1 / EX3 / EKQ: seed_base 7, wait 10 (dummy open-gripper steps), replan 5,
max 520 steps, images flipped [::-1, ::-1] and resized with padding to 224.

The trial loop is anchor_flow/run_case._run_trial (read-only) with one addition: the client also
sends the executed history (last 10 raw states s_{t-9..t} incl. the current one, last 10 raw
executed actions, and the step index t counted from the first query) so the server builds the
query tokens from what was really executed.

Conditions
  A       (iii) query, KQ2-CL Expert, t0 head + Dev map, gripper decoder, pi0.5 fallback over b
  B       (i)   query, same Expert, heads of (i), the same kind of map / fallback chosen for (i)
  Aconst  A with t0 = the Dev-chosen constant (same head ê and fallback b)
  pi05    plain pi0.5 (same 200 trials)

  python kq2cl_closed.py run --conditions A,B --smoke     (task 10 x 2 trials, dumps inputs)
  python kq2cl_closed.py run --conditions A,B,Aconst,pi05
  python kq2cl_closed.py parity                          (in-loop input parity from the smoke dumps)
  python kq2cl_closed.py summarize
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

import kq2cl_common as C

TRIALS = 10
SUITE_API = {'libero90': 'libero_90', 'libero10': 'libero_10'}
# job sets: condition -> trial indices per suite (coordinator change 2026-11-20: 10 trials per task,
# LIBERO-10 added; pi05 only on LIBERO-10 trials 5..9 plus a parity re-run of trial 0 of every task)
PLAN = {'libero90': {'A': range(10), 'B': range(10), 'Aconst': range(10), 'Bconst': range(10)},
        'libero10': {'A': range(10), 'B': range(10), 'Aconst': range(10), 'pi05': range(5, 10), 'pi05_parity': range(0, 1)}}
SMOKE_TASK, SMOKE_TRIALS = 10, 2


def gpu_used() -> str:
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


def conditions() -> dict:
    pol = C.read_json(C.OUT / 'policy.json'); sel = C.read_json(C.OUT / 'expert/select.json')
    ck = sel['selected']['checkpoint']
    out = {}
    for name, q in (('A', 'iii'), ('B', 'i')):
        p = pol['queries'][q]['selected']
        out[name] = {'mode': 'memory', 'query': q, 'expert': ck, 'bundle': str(C.OUT / 'heads' / f'bundle_{q}.pt'),
                     'tpol': {'c': p['c'], 'lo': p['lo'], 'hi': p['hi'], 'b': p['b'] if np.isfinite(p['b']) else None}}
    p = pol['queries']['iii']['selected']
    out['Aconst'] = {'mode': 'memory', 'query': 'iii', 'expert': ck, 'bundle': str(C.OUT / 'heads' / 'bundle_iii.pt'),
                     'tpol': {'const': pol['queries']['iii']['const_t0_selected'], 'b': p['b'] if np.isfinite(p['b']) else None}}
    pb = pol['queries']['i']['selected']   # diagnostic (added after A/B/Aconst): B with the A-const constant t0
    out['Bconst'] = {'mode': 'memory', 'query': 'i', 'expert': ck, 'bundle': str(C.OUT / 'heads' / 'bundle_i.pt'),
                     'tpol': {'const': pol['queries']['iii']['const_t0_selected'], 'b': pb['b'] if np.isfinite(pb['b']) else None}}
    out['pi05'] = {'mode': 'pi05'}
    out['pi05_parity'] = {'mode': 'pi05'}
    return out


def run_trial(job: dict) -> dict:
    """anchor_flow.run_case._run_trial + executed-history fields (kq2/*)."""
    sys.path.insert(0, str(C.HERE.parent))
    from anchor_flow import run_case as rc
    import hashlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from skillnet_client import image_tools
    suite_name = rc._suite_name(job['suite'])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(job['task_id'])
    language = rc._task_language(suite_name, task)
    initial_states = suite.get_task_init_states(job['task_id'])
    state_index = rc.initial_state_index(job['trial_index'], len(initial_states))
    initial_state = np.asarray(initial_states[state_index])
    init_digest = hashlib.sha256(np.ascontiguousarray(initial_state).view(np.uint8)).hexdigest()
    trial_seed = job['seed_base'] + job['trial_index']
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    horizon = rc.environment_horizon(job['max_steps'], job['wait_steps'])
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, horizon=horizon)
    env_seed = rc.environment_seed(suite_name, job['seed_base'], job['trial_index'])
    env.seed(env_seed)
    client = None; traces = []; started = time.perf_counter(); done = False; success = False; steps = 0; query_id = 0
    action_plan = []; states, actions = [], []
    base = {'trial_index': job['trial_index'], 'trial_seed': trial_seed, 'environment_seed': env_seed,
            'max_env_steps': job['max_steps'], 'stabilization_steps': job['wait_steps'], 'replan_steps': job['replan_steps'],
            'initial_state_index': state_index, 'initial_state_sha256': init_digest, 'task_suite': suite_name,
            'task_id': job['task_id'], 'task': language, 'condition': job['condition'], 'environment_horizon': horizon}

    def state_of(o):
        return np.concatenate((o['robot0_eef_pos'], rc._quat2axisangle(o['robot0_eef_quat']), o['robot0_gripper_qpos'])).astype(np.float32)
    try:
        client = rc.S2WebsocketClientPolicy(job['host'], job['port'])
        env.reset()
        obs = env.set_init_state(initial_state)
        dummy = [0.0] * 6 + [-1.0]
        for _ in range(job['wait_steps']):
            obs, _, returned_done, _ = env.step(dummy)
            success, done = rc.episode_status(env, returned_done)
            if done:
                break
        states.append(state_of(obs))
        while steps < job['max_steps'] and not done:
            if not action_plan:
                image = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
                image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                element = {
                    'observation/image': image, 'observation/wrist_image': wrist,
                    'observation/state': states[-1].copy(), 'prompt': language,
                    'kq2/state_hist': np.stack(states[-10:]).astype(np.float32),
                    'kq2/action_hist': (np.stack(actions[-10:]).astype(np.float32) if actions else np.zeros((0, 7), np.float32)),
                    'kq2/t': steps,
                    's2_episode_id': f"{suite_name}:{job['task_id']}:{job['trial_index']}",
                    's2_query_id': job['trial_index'] * 1000 + query_id,
                    's2_rng_seed': trial_seed * 100000 + query_id,
                }
                response = client.infer(element)
                chunk = np.asarray(response['actions'], dtype=np.float32)[:, :7]
                if chunk.shape[0] < job['replan_steps']:
                    raise ValueError(f'action chunk too short for replan: {chunk.shape}')
                action_plan.extend(chunk[:job['replan_steps']])
                traces.append(response.get('s2_debug', {'query_id': query_id, 'missing_debug': True}))
                query_id += 1
            action = np.asarray(action_plan.pop(0), np.float32)
            obs, _, returned_done, _ = env.step(action.tolist())
            actions.append(action); states.append(state_of(obs))
            success, done = rc.episode_status(env, returned_done)
            steps += 1
        return base | {'success': success, 'env_steps': steps, 'num_policy_queries': query_id,
                       'termination_reason': 'success' if success else 'environment_terminated' if done else 'control_budget',
                       'selected_units_and_t0': traces, 'elapsed_seconds': time.perf_counter() - started, 'status': 'complete'}
    except Exception as exc:
        return base | {'success': False, 'env_steps': steps, 'num_policy_queries': query_id, 'selected_units_and_t0': traces,
                       'status': 'error', 'error': repr(exc), 'traceback': traceback.format_exc(),
                       'elapsed_seconds': time.perf_counter() - started}
    finally:
        if client is not None:
            client.close()
        env.close()


def read_trials(cdir: Path) -> dict:
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
    return rows


def tasks(suite: str = 'libero90'):
    import run_ex1_scope_changed as ex1run
    t, _ = ex1run.load_baseline(suite)
    return t


def run(conds: list[str], *, workers: int, fraction: float, smoke: bool, cpus: list[str], spec_path: str | None = None,
        suite: str = 'libero90') -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    sys.path.insert(0, str(C.HERE.parent))
    from anchor_flow.run_case import S2WebsocketClientPolicy
    root = C.OUT / (('closed-loop-smoke' + ('' if suite == 'libero90' else '-' + suite)) if smoke else f'closed-loop/{suite}')
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(spec_path).read_text()) if spec_path else conditions()
    C.write_json(root / 'conditions.json', spec)
    tlist = tasks(suite)

    def one(name, cpu):
        c = spec[name]; cdir = root / name; cdir.mkdir(parents=True, exist_ok=True)
        existing = read_trials(cdir)
        want = [(t, i) for t in tlist for i in PLAN[suite][name]]
        if smoke:
            st = next(t for t in tlist if t['task_id'] == (SMOKE_TASK if suite == 'libero90' else tlist[0]['task_id']))
            want = [(st, i) for i in range(SMOKE_TRIALS)]
        pending = [(t, i) for t, i in want if existing.get((t['task_id'], i), {}).get('status') != 'complete']
        C.log('cl_condition_start', condition=name, pending=len(pending), smoke=smoke)
        if not pending:
            return
        port = free_port(); env = os.environ.copy()
        env.update(SKILLNET_SUBTASK_GENERATE='0', XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
                   TORCH_MEM_FRACTION='.05', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        if smoke:
            env['KQ2CL_DUMP'] = str(root / name / 'dump')
        (root / 'servers').mkdir(exist_ok=True)
        slog = (root / 'servers' / f'{name}-{port}.log').open('a')
        cmd = ['taskset', '-c', cpu, 'nice', '-n', '10', sys.executable, str(C.HERE / 'kq2cl_serve.py'), '--mode', c['mode'], '--port', str(port)]
        if c['mode'] == 'memory':
            cmd += ['--query', c['query'], '--expert-checkpoint', c['expert'], '--bundle', c['bundle'], '--tpol', json.dumps(c['tpol'])]
        C.log('gpu_before_server', condition=name, suite=suite, fraction=fraction, nvidia_smi=gpu_used())
        proc = subprocess.Popen(cmd, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'{name} server exited; see {slog.name}')
                if time.monotonic() > deadline:
                    raise TimeoutError(name)
                try:
                    cl = S2WebsocketClientPolicy('127.0.0.1', port); cl.close(); break
                except (ConnectionError, OSError, TimeoutError):
                    time.sleep(2)
            C.log('cl_server_ready', condition=name, port=port, suite=suite, nvidia_smi=gpu_used())
            jobs = [{'host': '127.0.0.1', 'port': port, 'suite': SUITE_API[suite], 'task_id': int(t['task_id']), 'trial_index': int(i),
                     'seed_base': 7, 'max_steps': 520, 'wait_steps': 10, 'replan_steps': 5, 'condition': f'KQ2CL_{name}'}
                    for t, i in pending]
            with ProcessPoolExecutor(max_workers=1 if smoke else workers) as pool:
                futs = {pool.submit(run_trial, j): j for j in jobs}
                for fut in as_completed(futs):
                    j = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:
                        C.log('cl_trial_error', condition=name, task_id=j['task_id'], trial=j['trial_index'], error=repr(exc)[:400]); continue
                    r['condition_spec'] = c
                    with (cdir / 'trials.jsonl').open('a') as f:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
                    C.log('cl_trial', condition=name, task_id=int(r['task_id']), trial=int(r['trial_index']), status=r.get('status'),
                          success=bool(r.get('success')), **({'error': r.get('error', '')[:300]} if r.get('status') != 'complete' else {}))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, 15)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
            slog.close()
        C.log('cl_condition_end', condition=name)

    ths = [threading.Thread(target=one, args=(n, cpus[i % len(cpus)])) for i, n in enumerate(conds)]
    [t.start() for t in ths]; [t.join() for t in ths]


# ---------------------------------------------------------------------------
# in-loop parity: recompute the query from the dumped inputs with the OFFLINE code path
# ---------------------------------------------------------------------------

def parity() -> dict:
    """For every smoke query: (1) tokens rebuilt with the offline GPUSplit.window rule from the dumped
    executed sequences, (2) VLM tokens recomputed by the offline extractor (kq2cl_obs transforms +
    vq_sources._vq_features) from the dumped raw images/state/prompt, (3) q / top-1 / ê recomputed."""
    import torch, types, jax
    import kq2_data as kd
    import kq2cl_obs as O
    import skill_expert_refine as exp3
    import vq_sources as vs
    from openpi.shared.nnx_utils import module_jit
    root = C.OUT / 'closed-loop-smoke'
    res = {}
    base = exp3._module(C.HERE.parent / 'pi05_memory/run.py', 'kq2cl_parity')
    model, _ = base.load_policy(C.PI05)
    feat = module_jit(types.MethodType(vs._vq_features, model)); trs = O.transforms()
    keys = torch.as_tensor(np.load(C.KEYS), device='cuda:0')
    for name, q in (('A', 'iii'), ('B', 'i')):
        files = sorted((root / name / 'dump').glob('*.npz'))
        if not files:
            continue
        qnet, cfg = C.load_query(q, 'cuda:0')
        b = torch.load(C.OUT / 'heads' / f'bundle_{q}.pt', map_location='cpu', weights_only=False)
        conf = C.make_head(b['confidence']['kind'], 1); conf.load_state_dict(b['confidence']['state_dict']); conf.eval()
        scale = float(qnet.logit_scale.exp().clamp(max=100.))
        dq, dtok, dv, top_same, de = [], [], [], [], []
        for p in files:
            z = np.load(p, allow_pickle=True)
            t = int(z['t']); sh = z['state_hist']; ah = z['action_hist']
            # offline rule: seq arrays -> kq2_run.GPUSplit.window equivalent (torch implementation reused)
            import kq2_run as kr
            seq_s = kd.qnorm(sh, 'state'); seq_a = kd.qnorm(ah, 'actions') if len(ah) else np.zeros((0, 7), np.float32)
            T0 = t - (len(seq_s) - 1)   # first sent state index
            # build a 1-episode GPUSplit: states for tau=T0..t, actions for tau=t-len(a)..t-1, pad earlier with zeros
            S = np.zeros((t + 1, 8), np.float32); S[T0:] = seq_s
            A = np.zeros((t + 1, 7), np.float32)
            if len(seq_a):
                A[t - len(seq_a):t] = seq_a
            sp = {'seq_actions': A, 'seq_states': S, 'ep_offset': np.array([0]), 'ep_slot': np.array([0]), 'start': np.array([t]),
                  'key': np.zeros((1, 3)), 'allowed': np.ones((1, 4516), bool), 'task': np.array([0]), 'episode': np.array([0])}
            gs = kr.GPUSplit(sp, torch.device('cuda:0'), False, errors=False)
            xo, vo = gs.window(torch.as_tensor([0], device='cuda:0'))
            dtok.append(float(np.abs(xo[0].cpu().numpy() - z['x']).max()))
            if q == 'iii':
                raw = {'observation/image': z['image'], 'observation/wrist_image': z['wrist'],
                       'observation/state': z['raw_state'], 'prompt': str(z['prompt'])}
                r = O.apply_transforms(trs, raw)
                rows = {'image_base': r['image']['base_0_rgb'][None], 'image_wrist': r['image']['left_wrist_0_rgb'][None],
                        'state': r['state'][None], 'prompt': r['tokenized_prompt'][None], 'prompt_mask': r['tokenized_prompt_mask'][None]}
                f = {k: np.asarray(v) for k, v in feat(base.obs_batch(rows, np.arange(1))).items()}
                v = np.zeros((1, 34, 2048), np.float16); v[0, :32] = f['vlm_img'][0]; v[0, 32] = f['vlm_txt_mean'][0]; v[0, 33] = f['vlm_txt_last'][0]
                dv.append(float(np.abs(v.astype(np.float32) - z['vlm'].astype(np.float32)).max()))
                gs.vlm = torch.as_tensor(v, device='cuda:0')
            with torch.no_grad():
                qq = qnet(gs, torch.as_tensor([0], device='cuda:0')); lg = qnet.score(qq, keys).double().cpu().numpy()
            dq.append(float(np.abs(qq.float().cpu().numpy() - z['q']).max()))
            s = C.summarize_scores(lg, np.ones_like(lg, bool))
            top_same.append(bool(s['ids'][0, 0] == z['top_ids'][0, 0]))
            fe = C.head_features(qq.float().cpu().numpy(), s['ids'], s['logits'], s['entropy'], s['top1_prob'], s['top5_mass'],
                                 np.load(C.KEYS).astype(np.float32), scale)
            with torch.no_grad():
                e = float(np.exp(conf(torch.as_tensor(C.head_input(fe, b['confidence']['kind']))).numpy()[0, 0]))
            de.append(abs(e - float(z['ehat'])) / max(float(z['ehat']), 1e-9))
        res[name] = {'queries': len(files), 'token_max_abs_diff': max(dtok), 'vlm_max_abs_diff': max(dv) if dv else None,
                     'q_max_abs_diff': max(dq), 'top1_same_fraction': float(np.mean(top_same)), 'ehat_rel_diff_max': max(de)}
        C.log('parity_inloop', condition=name, **res[name])
    C.write_json(C.OUT / 'parity_inloop.json', res)
    return res


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _diag(rows: dict, tlist) -> dict:
    qs = [q for r in rows.values() for q in r.get('selected_units_and_t0', []) if isinstance(q, dict)]
    if not qs or 'unit_id' not in qs[0]:
        return {'gripper_switches_per_trial': float(np.mean([_gripper_switches(r) for r in rows.values()])) if rows else None}
    fb = np.asarray([bool(q.get('fallback_pi05')) for q in qs])
    t0 = np.asarray([q['t0'] for q in qs]); eh = np.asarray([q['confidence_error'] for q in qs])
    pt = []
    for r in rows.values():
        qq = [q for q in r.get('selected_units_and_t0', []) if isinstance(q, dict)]
        u = [q.get('unit_id') for q in qq]; f = [bool(q.get('fallback_pi05')) for q in qq]
        ch = [a != b for a, b in zip(u, u[1:])]
        pt.append({'success': bool(r['success']), 'fallback_rate': float(np.mean(f)) if f else 0.,
                   'unit_switch_rate': float(np.mean(ch)) if ch else 0., 'gripper_switches': _gripper_switches(r), 'queries': len(u)})

    def grp(key, ok):
        v = [p[key] for p in pt if p['success'] == ok]
        return float(np.mean(v)) if v else None
    by_task = {}
    for t in tlist:
        f = [bool(q.get('fallback_pi05')) for r in rows.values() if r['task_id'] == t['task_id'] for q in r.get('selected_units_and_t0', []) if isinstance(q, dict)]
        by_task[str(t['task_id'])] = float(np.mean(f)) if f else None
    return {'queries': int(len(qs)), 'fallback_rate': float(fb.mean()),
            't0_non_fallback': C.quantiles(t0[~fb]) if (~fb).any() else {}, 'ehat': C.quantiles(eh),
            'unit_switch_rate': float(np.mean([p['unit_switch_rate'] for p in pt])),
            'gripper_switches_per_trial': float(np.mean([p['gripper_switches'] for p in pt])),
            'success_vs_failure': {k: {'success': grp(k, True), 'failure': grp(k, False)}
                                   for k in ('fallback_rate', 'unit_switch_rate', 'gripper_switches', 'queries')},
            'fallback_rate_by_task': by_task}


def _succ(rows: dict) -> dict:
    return {k: bool(v['success']) for k, v in rows.items() if v.get('status') == 'complete'}


def summarize() -> dict:
    import run_cl1_closed_loop as cl1
    import run_ex1_scope_changed as ex1run
    res = {'setting': 'CL1 protocol: seed_base 7 (env seed 7+trial), public initial state = trial index, wait 10, replan 5, 520 steps; '
                      '10 trials per task (coordinator change). pi05: LIBERO-90 = baseline-01 (CL1/EX2, 10x10); '
                      'LIBERO-10 = closed-loop-memory-01 normal trials 0..4 (EX2) + this run trials 5..9.',
           'suites': {}}
    for suite in ('libero90', 'libero10'):
        tlist, baseline = ex1run.load_baseline(suite)
        keys = [(t['task_id'], i) for t in tlist for i in range(TRIALS)]
        rows = {n: {k: v for k, v in read_trials(C.OUT / 'closed-loop' / suite / n).items() if v.get('status') == 'complete'}
                for n in ('A', 'B', 'Aconst', 'Bconst', 'pi05', 'pi05_parity')}
        succ = {n: _succ(rows[n]) for n in ('A', 'B', 'Aconst', 'Bconst') if rows[n]}
        pi = {k: bool(v['success']) for k, v in baseline.items()}
        if suite == 'libero10':
            pi.update(_succ(rows['pi05']))
        succ['pi05'] = pi
        ref = {}
        if suite == 'libero90':
            for n in ('C1a', 'C1b', 'C2', 'C3'):
                ref[n] = {(int(r['task_id']), int(r['trial_index'])): bool(r['success']) for r in cl1.read_trials(C.ROOT / 'closed-loop-kq-01' / n) if r.get('status') == 'complete'}
            for n in ('E1a', 'E1b'):
                ref[n] = {(int(r['task_id']), int(r['trial_index'])): bool(r['success']) for r in cl1.read_trials(C.ROOT / 'expert-kq-01/closed-loop' / n) if r.get('status') == 'complete'}
            for n in ('X3a', 'X3b'):
                ref[n] = {(int(r['task_id']), int(r['trial_index'])): bool(r['success']) for r in cl1.read_trials(C.ROOT / 'expert-ex3-01/closed-loop' / n) if r.get('status') == 'complete'}
        S = {'tasks': [{'task_id': t['task_id'], 'name': t['name']} for t in tlist], 'conditions': {}, 'reference': {}, 'mcnemar': {}, 'diagnostics': {}}
        for n, sc in succ.items():
            S['conditions'][n] = {'completed': len([k for k in keys if k in sc]), 'successes': int(sum(sc.get(k, False) for k in keys)),
                                  'tasks': {str(t['task_id']): int(sum(sc.get((t['task_id'], i), False) for i in range(TRIALS))) for t in tlist}}
        for n, sc in ref.items():
            S['reference'][n] = {'completed': len(sc), 'successes': int(sum(sc.values())),
                                 'tasks': {str(t['task_id']): int(sum(sc.get((t['task_id'], i), False) for i in range(TRIALS))) for t in tlist}}
        for a, b in (('pi05', 'A'), ('pi05', 'B'), ('pi05', 'Aconst'), ('B', 'A'), ('Aconst', 'A'), ('pi05', 'Bconst'), ('Bconst', 'Aconst'), ('B', 'Bconst')) + tuple((r, 'A') for r in ('C1a', 'C1b', 'X3a', 'E1a')) + tuple((r, 'B') for r in ('C1b',)):
            sa = succ.get(a, ref.get(a)); sb = succ.get(b, ref.get(b))
            if not sa or not sb:
                continue
            common = [k for k in keys if k in sa and k in sb]
            if common:
                S['mcnemar'][f'{b} - {a}'] = {'pairs': len(common), **C.mcnemar_exact([sa[k] for k in common], [sb[k] for k in common])}
        for n in ('A', 'B', 'Aconst', 'Bconst', 'pi05'):
            if rows[n]:
                S['diagnostics'][n] = _diag(rows[n], tlist)
        # pi05 parity: our pi05 server path vs the existing baseline on the same (task, trial)
        par = {}
        for n, src in (('pi05_parity', rows['pi05_parity']),):
            same = [(k, bool(v['success']) == bool(baseline[k]['success']), int(v['env_steps']) == int(baseline[k]['env_steps']))
                    for k, v in src.items() if k in baseline]
            if same:
                par[n] = {'pairs': len(same), 'success_equal': int(sum(x[1] for x in same)), 'env_steps_equal': int(sum(x[2] for x in same))}
        if suite == 'libero90':
            ab = read_trials(C.OUT / 'closed-loop-pi05-aborted' / 'pi05')
            same = [(bool(v['success']) == bool(baseline[k]['success']), int(v['env_steps']) == int(baseline[k]['env_steps']))
                    for k, v in ab.items() if v.get('status') == 'complete' and k in baseline]
            if same:
                par['aborted_pi05_trials0_9'] = {'pairs': len(same), 'success_equal': int(sum(x[0] for x in same)),
                                                 'env_steps_equal': int(sum(x[1] for x in same))}
        S['pi05_parity'] = par
        errs = {}
        for n in ('A', 'B', 'Aconst', 'Bconst', 'pi05', 'pi05_parity'):
            p = C.OUT / 'closed-loop' / suite / n / 'trials.jsonl'
            if p.exists():
                L = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
                bad = [x for x in L if x.get('status') != 'complete']
                errs[n] = {'error_rows': len(bad), 'oom_rows': sum('RESOURCE_EXHAUSTED' in str(x.get('error', '')) or 'out of memory' in str(x.get('error', '')).lower() for x in bad)}
        S['error_rows_rerun'] = errs
        S['status'] = 'complete' if all(S['conditions'][n]['completed'] == len(keys) for n in ('A', 'B', 'Aconst', 'pi05')) else 'partial'
        res['suites'][suite] = S
    C.write_json(C.OUT / 'closed_loop.json', res)
    return res


def _gripper_switches(r) -> int:
    """Executed gripper sign changes over the episode (first 5 steps of each chunk are executed)."""
    g = []
    for q in r.get('selected_units_and_t0', []):
        if isinstance(q, dict) and q.get('gripper_chunk'):
            g += [np.sign(x) if x != 0 else -1. for x in q['gripper_chunk'][:5]]
    return int(sum(a != b for a, b in zip(g, g[1:])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('run', 'parity', 'summarize'))
    ap.add_argument('--conditions', default='A,B,Aconst,pi05'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--fraction', type=float, default=.2)
    ap.add_argument('--cpus', default='24-31,32-39,40-47,16-23'); ap.add_argument('--spec', default=None)
    ap.add_argument('--suite', default='libero90', choices=('libero90', 'libero10'))
    a = ap.parse_args()
    C.log('start', stage='closed.' + a.cmd, conditions=a.conditions, smoke=a.smoke)
    if a.cmd == 'run':
        run(a.conditions.split(','), workers=a.workers, fraction=a.fraction, smoke=a.smoke, cpus=a.cpus.split(','), spec_path=a.spec, suite=a.suite)
        if not a.smoke:
            summarize()
    elif a.cmd == 'parity':
        parity()
    else:
        print(json.dumps({s: {k: v['successes'] for k, v in r['conditions'].items()} for s, r in summarize()['suites'].items()}))
    C.log('end', stage='closed.' + a.cmd)


if __name__ == '__main__':
    main()
