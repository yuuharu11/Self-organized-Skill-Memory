#!/usr/bin/env python3
"""MAG2-L: Dev-selected magnitude-only rules in LIBERO-90 closed loop.

This module imports ``memory_closed_loop`` and leaves it unchanged. It wraps
its selector pose operation, preserving the K1 history query, unit_08 pool,
5-step replanning, and π0.5 gripper. Run from /work after activating the
SkillNet environment. The full command uses three GPU servers capped at .15:

    taskset -c 8-47 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      OPENBLAS_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=.15 timeout 12h \
      python skill_moe/skillnet/examples/memory_pilot/unseen_libero/magnitude_closed_loop.py run

Outputs are write-once under ``outputs/memory-pilot/unseen-libero/magnitude-02``.
The offline selector uses only the 95-unit MAG1-L Dev table; Eval contributes
no selection or tuning. A2 uses previous executed arm motion. At episode start,
before any arm action exists, it falls back to the unit's source RMS size; the
fallback is counted and recorded. Only π0.5 gripper actions are retained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memory_closed_loop as legacy
import magnitude_bounds as mag1
import scale_only_query as sq
import variable_shape_memory as vs

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
OUT = ROOT / 'magnitude-02'
MAG1_OUT = ROOT / 'magnitude-01'
POOL = ROOT / 'cluster-pools-01/pool_unit_08.npz'
POOL_CKPT = ROOT / 'cluster-pools-01/query_unit_08.pt'
CHECKPOINT = Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero')
SCALE = ROOT / 'scale-only-02/production'
MODE_NAMES = ('mem-scale', 'mem-scale-hist')


def _atomic_json(path: Path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _episode_keys(task, episode):
    return mag1.episode_cluster(task, episode)


def choose_dev_rule(report_path=MAG1_OUT / 'report.json'):
    """Select among deployable (non-oracle) rules using the Dev 6-axis window means."""
    report = json.loads(Path(report_path).read_text())
    means = {}
    for row in report['all_combinations']:
        if row['split'] != 'dev' or row['axes'] != '6axis':
            continue
        means[(row['amp'], row['duration'], row['window'])] = float(row['mean'])
    amp_rules = ('A2_history', 'A3_pi05', 'A4_stored', 'A5_expert_t02', 'A5_expert_t04', 'A6_procrustes')
    time_rules = ('T2_stored', 'T3_ten', 'T4_history')
    options = []
    for amp in amp_rules:
        for duration in time_rules:
            m5 = means[(amp, duration, 'first5')]
            m59 = means[(amp, duration, 'continuation5_9')]
            options.append({'amp': amp, 'duration': duration, 'score': (m5 + m59) / 2,
                            'first5_6axis': m5, 'continuation5_9_6axis': m59})
    options.sort(key=lambda x: (x['score'], x['amp'], x['duration']))
    return {'selected': options[0], 'selection_metric': 'unweighted mean of Dev 6-axis first5 and continuation5_9 MSE',
            'candidate_count': len(options), 'candidates': options,
            'split_sizes': {'dev_windows': 113, 'eval_windows': 619},
            'eval_used_for_selection': False,
            'note': 'Oracle A1 and T1 are excluded from deployment selection. MAG2 uses the selected non-oracle rule.'}


def _segment_targets(split):
    inputs = mag1._load_rows(split)
    traces = mag1._traces()
    seg = mag1._segments_by_row(inputs, traces)
    return inputs, seg


def scale_only_1595(out: Path, *, include_a5=True):
    """All amplitude x duration combinations, oracle type, 1,595 W candidates, 6 axes."""
    out.mkdir(parents=True, exist_ok=True)
    wpool = np.load(SCALE / 'prep/W_pool.npz', allow_pickle=False)
    pool = np.asarray(wpool['phys'], np.float32)
    all_results = []
    row_caches = {}
    for split in ('dev', 'eval'):
        inp, segments = _segment_targets(split)
        rows = np.asarray(inp['row'], np.int64)
        action_all = np.load(mag1.ROWS / 'actions.npy', mmap_mode='r', allow_pickle=False)
        target = np.asarray(action_all[rows, :, :7], np.float32)
        target_phys = vs.to_physical(target[:, :, :6]).astype(np.float32)
        pick, rot, a1, type_err = mag1.best_oracle_types(target_phys, pool,
            pool_task=wpool['task'], query_task=inp['task'])
        selected = mag1.apply_rotation(pool[pick], rot)
        stored = np.sqrt(np.mean(np.sum(selected ** 2, axis=2), axis=1))
        canonical = selected / np.maximum(stored[:, None, None], 1e-8)
        hist_phys = vs.to_physical(np.asarray(inp['hist'])[:, :, :6])
        hist_speed = np.sqrt(np.mean(np.sum(hist_phys ** 2, axis=2), axis=1))
        canonical_speed = np.sqrt(np.mean(np.sum(canonical ** 2, axis=2), axis=1))
        a2 = hist_speed / np.maximum(canonical_speed, 1e-8)
        pi_phys = vs.to_physical(np.asarray(inp['pi05'])[:, :, :6])
        a3 = np.maximum(np.sum(canonical * pi_phys, axis=(1, 2)), 0) / np.maximum(np.sum(canonical ** 2, axis=(1, 2)), 1e-12)
        a4 = stored
        a6, a6rot = np.empty(len(rows)), np.empty((len(rows), 3, 3))
        for i in range(len(rows)):
            a6rot[i], a6[i] = sq.procrustes_rs(canonical[i], pi_phys[i])
        template = vs.to_normalized(canonical).astype(np.float32)
        template7 = np.concatenate([template, np.zeros((len(rows), 10, 1), np.float32)], axis=2)
        a1 = a1 * stored
        amp = {'A1_oracle': a1, 'A2_history': a2, 'A3_pi05': a3, 'A4_stored': a4, 'A6_procrustes': a6}
        rotations = {'A6_procrustes': a6rot}
        if include_a5:
            draft = np.zeros((len(rows), 10, 32), np.float32)
            for i in range(len(rows)):
                draft[i, :, :7] = mag1.rollout(template7[i], a2[i], 10)
            exp = mag1._expert_outputs(split, draft)
            for t0, name in ((.2, 'A5_expert_t02'), (.4, 'A5_expert_t04')):
                exphys = vs.to_physical(exp[t0][:, :, :6])
                amp[name] = np.maximum(np.sum(canonical * exphys, axis=(1, 2)), 0) / np.maximum(np.sum(canonical ** 2, axis=(1, 2)), 1e-12)
        t1 = np.asarray([v[1] for v in segments], np.int64)
        t2 = np.full(len(rows), 10, np.int64)
        type_distance = np.sqrt(np.sum(canonical ** 2, axis=(1, 2)))
        durations = mag1.duration_values(t1, t2, hist_speed, type_distance)
        future = [v[0][:max(int(t1[i]), 10)] for i, v in enumerate(segments)]
        episode_key = _episode_keys(inp['task'], inp['episode'])
        for amp_name, av in amp.items():
            for duration_name, dv in durations.items():
                pred = np.zeros((len(rows), 10, 7), np.float32)
                truth = np.zeros_like(pred)
                for i, horizon in enumerate(dv):
                    p = mag1.rollout(template7[i], av[i], horizon,
                        rotation=rotations.get(amp_name, [None] * len(rows))[i])
                    source = future[i]
                    take = min(10, len(p), len(source))
                    pred[i, :take] = p[:take]
                    truth[i, :take] = source[:take]
                    if take < 10:
                        pred[i, take:] = pred[i, max(take - 1, 0)]
                        truth[i, take:] = truth[i, max(take - 1, 0)]
                for window, sl in (('first5', slice(0, 5)), ('continuation5_9', slice(5, 10))):
                    mse = np.mean((pred[:, sl, :6] - truth[:, sl, :6]) ** 2, axis=(1, 2))
                    all_results.append({'split': split, 'amp': amp_name, 'duration': duration_name,
                        'window': window, **mag1.paired_bootstrap(mse, episode_key)})
        row_caches[split] = {'rows': rows, 'pick': pick, 'type_error': type_err, 'amp': amp,
                             'duration_t1': t1}
    _atomic_json(out / 'scale_only_1595_all_combinations.json', {'candidate_count': 1595,
        'type_selection': 'target-oracle minimum Procrustes residual, same-task candidates excluded',
        'metric': '6-axis normalized action MSE; episode bootstrap; Eval not used for selection',
        'combinations': all_results})
    for split, z in row_caches.items():
        np.savez_compressed(out / f'scale_only_1595_{split}_rows.npz', row=z['rows'],
            selected_window=z['pick'], oracle_type_error=z['type_error'], duration_t1=z['duration_t1'],
            **{f'amp_{k}': v for k, v in z['amp'].items()})
    return all_results


def make_policy(checkpoint: Path, selected_amp: str):
    """Wrap only MemorySelector.pose_to; keep legacy query, environment, and action protocol."""
    policy = legacy.make_policy(checkpoint)
    orig_pick = policy.sel.pick
    orig_infer = policy.infer
    policy._mag_selected_amp = selected_amp
    policy._mag_source_scales = np.load(os.environ['MAG_SOURCE_SCALES'], allow_pickle=False)
    import threading
    context = threading.local()

    def pick(history, state, exclude_task=None):
        context.last_history = np.asarray(history, np.float32).copy()
        return orig_pick(history, state, exclude_task)

    def pose_to(idx, ref_norm):
        shape = np.asarray(policy.sel.shapes[idx], np.float64)
        source = shape[:, :, :6]
        amp_rule = getattr(context, 'amp_rule', selected_amp)
        hist_all = policy.hist.get(getattr(context, 'episode', ''), [])
        hist = np.asarray(hist_all[-10:], np.float32) if hist_all else np.zeros((0, 7), np.float32)
        if amp_rule in ('A3_pi05', 'A6_procrustes'):
            ref = vs.to_physical(np.asarray(ref_norm, np.float64)[:, :, :6])
            if policy._mag_selected_amp == 'A3_pi05':
                den = np.sum(source * source, axis=(1, 2))
                scale = np.maximum(np.sum(source * ref, axis=(1, 2)), 0) / np.maximum(den, 1e-12)
                posed = source * scale[:, None, None]
            else:
                out = []
                scales = []
                for i in range(len(source)):
                    r, s = sq.procrustes_rs(source[i], ref[i]); out.append(mag1.apply_rotation(source[i:i+1], r)[0] * s); scales.append(s)
                posed = np.stack(out); scale = np.asarray(scales)
            context.fallback = False
        else:
            # A2: match physical RMS speed of up to the ten most recent executed arm actions.
            if len(hist):
                hist_phys = vs.to_physical(hist[:, :6])
                speed = float(np.sqrt(np.mean(np.sum(hist_phys ** 2, axis=1))))
                unit_speed = np.sqrt(np.mean(np.sum(source[0] ** 2, axis=1)))
                s = speed / max(float(unit_speed), 1e-8)
                context.fallback = False
            else:
                # A2 has no observed speed before the first arm command. Use the stored
                # source unit size once, then switch to measured history speed.
                s = float(policy._mag_source_scales[int(np.asarray(idx).reshape(-1)[0])])
                context.fallback = True
            scale = np.full(len(source), s, np.float64)
            posed = source * scale[:, None, None]
        context.scale = float(np.mean(scale))
        mem = vs.to_normalized(posed).astype(np.float32)
        return np.concatenate([mem, shape[:, :, 6:7].astype(np.float32)], axis=2)

    def infer(observation):
        context.episode = str(observation['s2_episode_id'])
        context.condition = str(observation['s2_condition'])
        context.amp_rule = 'A2_history' if context.condition == 'mem-scale-hist' else selected_amp
        result = orig_infer(observation)
        dbg = result.get('s2_debug', {})
        dbg.update({'magnitude_rule': context.amp_rule, 'magnitude_scale': getattr(context, 'scale', 1.0),
                    'magnitude_fallback_to_source_at_episode_start': bool(getattr(context, 'fallback', False)),
                    'pi05_action_axes_used_by_arm': ['xyz+rotation magnitude'] if context.amp_rule == 'A3_pi05' else [],
                    'pi05_action_axes_used_by_gripper': ['gripper'],
                    'pose_rotation_from_pi05': context.amp_rule == 'A6_procrustes'})
        result['s2_debug'] = dbg
        return result

    policy.sel.pick = pick
    policy.sel.pose_to = pose_to
    policy.infer = infer
    return policy


def serve(args):
    import torch
    torch.set_num_threads(2)
    os.environ['MEM_POOL'] = str(POOL)
    os.environ['MEM_CKPT'] = str(POOL_CKPT)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p = make_policy(args.checkpoint, args.amp)
    WebsocketPolicyServer(p, host='127.0.0.1', port=args.port,
        metadata={'policy': 'MAG2-L magnitude-only memory', 'amp_rule': args.amp}).serve_forever()


def _read_report(path):
    import run as harness
    return harness.read_terminal(path)


def _mcnemar(a, b, keys):
    from scipy.stats import binomtest
    success = lambda r: r.get('status') == 'complete' and bool(r.get('success'))
    wins = sum(success(a[k]) and not success(b[k]) for k in keys)
    losses = sum(success(b[k]) and not success(a[k]) for k in keys)
    n = wins + losses
    return {'left_only_success': int(wins), 'right_only_success': int(losses),
            'discordant': int(n), 'p_two_sided_exact': float(binomtest(min(wins, losses), n, .5).pvalue) if n else 1.0}


def _diagnostics(records):
    scales, unique_per_trial, same_repick, fallbacks = [], [], [], 0
    for row in records.values():
        events = [x for x in row.get('selected_units_and_t0', []) if x.get('condition') == 'memory_k1_hist_theta_pi05']
        units = [int(x['unit']) for x in events if 'unit' in x]
        ss = [float(x['magnitude_scale']) for x in events if 'magnitude_scale' in x]
        scales.extend(ss)
        unique_per_trial.append(len(set(units)))
        if len(units) > 1:
            same_repick.append(float(np.mean(np.asarray(units[1:]) == np.asarray(units[:-1]))))
        fallbacks += sum(bool(x.get('magnitude_fallback_to_source_at_episode_start', False)) for x in events)
    return {'n_trials': len(records), 'mean_unique_units_per_trial': float(np.mean(unique_per_trial)) if unique_per_trial else None,
        'median_unique_units_per_trial': float(np.median(unique_per_trial)) if unique_per_trial else None,
        'same_unit_reselection_fraction': float(np.mean(same_repick)) if same_repick else None,
        'reselection_trial_count': len(same_repick), 'magnitude_scale_count': len(scales),
        'magnitude_scale_quantiles': {str(q): float(np.quantile(scales, q)) for q in (0, .1, .5, .9, 1)} if scales else {},
        'episode_start_source_scale_fallbacks': int(fallbacks)}


def _summary_table(out, *, suite='libero_90', inits=10):
    import run as harness
    result = {}
    selected_tasks = [int(x['task_id']) for x in json.loads((ROOT / 'baseline-01/plan.json').read_text())['selected_tasks']]
    base_path = ROOT / 'baseline-01/production/rollouts.jsonl'
    old_mem = ROOT / 'closed-loop-memory-01/libero90_unit_08/memory/rollouts.jsonl'
    baseline = harness.read_terminal(base_path)
    old = harness.read_terminal(old_mem)
    pairs = {}
    for cond in MODE_NAMES:
        recs = harness.read_terminal(out / cond / 'rollouts.jsonl')
        keys = sorted((t, i) for t in selected_tasks for i in range(inits) if (t, i) in recs and (t, i) in baseline and (t, i) in old)
        success = lambda r: r.get('status') == 'complete' and bool(r.get('success'))
        summary = {'n_paired': len(keys), 'successes': sum(success(recs[k]) for k in keys),
            'baseline01_successes': sum(success(baseline[k]) for k in keys),
            'unit08_successes': sum(success(old[k]) for k in keys),
            'paired_vs_baseline01': _mcnemar(recs, baseline, keys),
            'paired_vs_unit08': _mcnemar(recs, old, keys),
            'diagnostics': _diagnostics(recs)}
        result[cond] = summary
        pairs[cond] = (recs, keys)
    report = {'experiment': 'MAG2-L', 'suite': suite, 'inits': inits,
        'dev_rule': json.loads((out / 'dev_selection.json').read_text()),
        'task_ids': selected_tasks, 'seed_protocol': 'same run.py harness seed_base=7, same trial_index/init and public initial state as baseline-01',
        'conditions': result,
        'policy_axes': {'arm': 'memory unit shape, magnitude rule only; no current π0.5 arm output',
                        'gripper': 'π0.5 action dim 6', 'query': 'same K1 P5 history+state selector',
                        'replan_steps': 5, 'temporal_horizon': 'T3 fixed 10 action samples'},
        'unit08_reference': {'path': str(old_mem), 'reported_successes': 43},
        'baseline_reference': {'path': str(base_path), 'reported_successes': 32},
        'scale_only_1595': json.loads((out / 'offline/scale_only_1595_all_combinations.json').read_text())}
    _atomic_json(out / 'report.json', report)
    lines = ['# MAG2-L: LIBERO-90 magnitude-only closed loop', '',
        f"Dev-selected rule: **{report['dev_rule']['selected']['amp']} + {report['dev_rule']['selected']['duration']}**; Dev selection metric is the mean of 6-axis first-5 and continuation 5–9 MSE.", '',
        '| Condition | Successes | Baseline-01 | Unit_08 | McNemar vs baseline (new-only / baseline-only, p) | McNemar vs unit_08 (new-only / old-only, p) |',
        '|---|---:|---:|---:|---:|---:|']
    for cond, r in result.items():
        b, u = r['paired_vs_baseline01'], r['paired_vs_unit08']
        lines.append(f"| {cond} | {r['successes']}/{r['n_paired']} | {r['baseline01_successes']} | {r['unit08_successes']} | {b['left_only_success']}/{b['right_only_success']}, {b['p_two_sided_exact']:.4g} | {u['left_only_success']}/{u['right_only_success']}, {u['p_two_sided_exact']:.4g} |")
    lines += ['', '## Runtime diagnostics', '', '```json', json.dumps({k: v['diagnostics'] for k, v in result.items()}, indent=2), '```', '',
        '## 1,595-candidate scale-only pool: all 6-axis combinations', '',
        '| Split | Amplitude | Duration | Window | MSE | 95% episode bootstrap CI |', '|---|---|---|---|---:|---:|']
    for row in report['scale_only_1595']['combinations']:
        lines.append(f"| {row['split']} | {row['amp']} | {row['duration']} | {row['window']} | {row['mean']:.6f} | [{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}] |")
    lines += ['', 'Only the Dev split selected a rule. Eval rows are reporting-only.', '']
    (out / 'report.md').write_text('\n'.join(lines))
    return report


def run(args):
    out = args.output
    if out.exists():
        if not (out / 'run_manifest.json').exists():
            raise FileExistsError(f'refusing to use unrelated existing output: {out}')
    else:
        out.mkdir(parents=True)
        _atomic_json(out / 'run_manifest.json', {'experiment': 'MAG2-L', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'write_once_output': True, 'suite': 'libero_90'})
    dev_path = out / 'dev_selection.json'
    if dev_path.exists():
        dev = json.loads(dev_path.read_text())
    else:
        dev = choose_dev_rule()
        _atomic_json(dev_path, dev)
    offline_path = out / 'offline/scale_only_1595_all_combinations.json'
    if not offline_path.exists():
        scale_only_1595(out / 'offline', include_a5=True)
    # The cluster-pool NPZ stores canonical shapes without source magnitude;
    # rebuild the aligned source magnitudes from the frozen LIBERO-40 units.
    scale_file = out / 'unit_source_scales.npy'
    if not scale_file.exists():
        source_scales = mag1._stored_scales()
        np.save(scale_file, source_scales, allow_pickle=False)
    # The two requested online arms both use A2 if Dev selects A2. This is
    # expected to make them identical; retain separate labels for auditability.
    amp_rule = dev['selected']['amp']
    if amp_rule not in ('A2_history', 'A3_pi05', 'A4_stored', 'A6_procrustes'):
        raise ValueError(f'MAG2 runtime does not support Dev-selected amplitude rule: {amp_rule}')
    if dev['selected']['duration'] != 'T3_ten':
        raise ValueError(f'MAG2 runtime currently expects Dev-selected T3_ten, got {dev["selected"]["duration"]}')
    tasks = [int(t['task_id']) for t in json.loads((ROOT / 'baseline-01/plan.json').read_text())['selected_tasks']]
    import run as harness
    servers, logs = [], []
    for i in range(args.servers):
        port = harness.free_port()
        folder = out / 'servers' / str(i); folder.mkdir(parents=True, exist_ok=True)
        log = (folder / 'server.log').open('a'); logs.append(log)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.15',
            MEM_POOL=str(POOL), MEM_CKPT=str(POOL_CKPT), MAG_SOURCE_SCALES=str(scale_file))
        proc = __import__('subprocess').Popen(['taskset', '-c', '8-15', 'nice', '-n', '10', sys.executable,
            str(Path(__file__).resolve()), 'serve', '--checkpoint', str(args.checkpoint), '--port', str(port), '--amp', amp_rule],
            env=env, stdout=log, stderr=__import__('subprocess').STDOUT, start_new_session=True)
        servers.append((proc, port))
    try:
        for proc, port in servers:
            harness.ready(proc, port)
        done_by_condition = {c: harness.read_terminal(out / c / 'rollouts.jsonl') for c in MODE_NAMES}
        jobs = [(cond, task, init) for cond in MODE_NAMES for task in tasks for init in range(args.inits)
                if (task, init) not in done_by_condition[cond]]
        total = len(tasks) * args.inits
        import threading
        lock = threading.Lock()

        def execute(cond, task, init, port):
            job = {'host': '127.0.0.1', 'port': port, 'suite': 'libero_90', 'task_id': task,
                'trial_index': init, 'seed_base': 7, 'condition': cond, 'memory_snapshot': 'none',
                'query_text_source': 'instruction_only', 'checkpoint': str(args.checkpoint),
                'wait_steps': 10, 'replan_steps': 5, 'max_steps': 520, 'record': init < 2}
            root = out / cond / 'attempts' / f'task{task:02d}-init{init:02d}'
            existing = list(root.glob('attempt-*')) if root.exists() else []
            first = max([int(x.name.split('-')[-1]) for x in existing], default=0) + 1
            row = None
            for attempt in range(first, first + 2):
                row = harness.trial(job, root / f'attempt-{attempt}', args.timeout, attempt)
                harness.append(out / cond / 'attempts.jsonl', row)
                if row['status'] == 'complete':
                    break
            row.update(task_id=task, trial_index=init, task_suite='libero_90')
            harness.append(out / cond / 'rollouts.jsonl', row)
            with lock:
                done = harness.read_terminal(out / cond / 'rollouts.jsonl')
                harness.write(out / cond / 'progress.json', {'processed': len(done), 'scheduled': total,
                    'successes': sum(x.get('status') == 'complete' and bool(x.get('success')) for x in done.values()),
                    'updated_unix': time.time()})

        def lane(slot):
            proc, port = servers[slot % args.servers]
            for cond, task, init in jobs[slot::args.servers]:
                if proc.poll() is not None:
                    raise RuntimeError(f'MAG2 server {slot % args.servers} exited')
                execute(cond, task, init, port)

        for cond in MODE_NAMES:
            (out / cond).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.servers) as pool:
            for future in as_completed([pool.submit(lane, k) for k in range(args.servers)]):
                future.result()
        _summary_table(out, inits=args.inits)
        (out / 'DONE').write_text('complete\n')
    finally:
        for proc, _ in servers:
            harness.terminate(proc)
        for handle in logs:
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run'); r.add_argument('--output', type=Path, default=OUT)
    r.add_argument('--checkpoint', type=Path, default=CHECKPOINT); r.add_argument('--servers', type=int, default=3)
    r.add_argument('--inits', type=int, default=10); r.add_argument('--timeout', type=int, default=900)
    s = sp.add_parser('serve'); s.add_argument('--checkpoint', type=Path, required=True); s.add_argument('--port', type=int, required=True)
    s.add_argument('--amp', required=True)
    d = sp.add_parser('offline'); d.add_argument('--output', type=Path, default=OUT / 'offline')
    a = p.parse_args()
    if a.cmd == 'run':
        if not 1 <= a.servers <= 3: raise ValueError('MAG2 allows at most three GPU servers')
        run(a)
    elif a.cmd == 'serve': serve(a)
    else:
        choose = choose_dev_rule(); print(json.dumps(choose, indent=2))
        scale_only_1595(a.output, include_a5=True)


if __name__ == '__main__': main()
