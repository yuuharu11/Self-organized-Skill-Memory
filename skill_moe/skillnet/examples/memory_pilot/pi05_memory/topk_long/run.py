#!/usr/bin/env python3
"""Stage 4-C: raw-action Top-K candidates through the frozen bridge expert."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
RUN = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
GAP = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/memory-gap-01')
RETRIEVAL = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/retrieval-4b-01')
DATA10 = Path('/work/data/LIBERO/libero_10')
DEFAULT_OUT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/topk-long-01')
KEYS = ('wrist_state_contrastive', 'current', 'codec_prior_b', 'pi05_pred_to_memory_action')
EXCLUSIONS = ('task', 'episode')
T0S = (.2, .4)
KS = (1, 4, 8, 16)
NOISE0 = 4101
BOOTSTRAP_REPS = 5000
sys.path.insert(0, str(PI05))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_inputs():
    rows = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (RUN / 'rows').glob('*.npy')}
    gap = np.load(GAP / 'predictions_and_errors.npz', allow_pickle=False)
    eval_rows = np.asarray(gap['row_ids'], np.int64)
    if len(eval_rows) != 1318:
        raise ValueError(f'expected memory-gap Eval 1,318 rows, got {len(eval_rows)}')
    long_rows = eval_rows[np.asarray(rows['task'][eval_rows]) < 10]
    if not len(long_rows) or set(np.asarray(rows['task'][long_rows]).tolist()) != set(range(10)):
        raise ValueError('Long Eval rows do not cover exactly manifest task ids 0..9')
    return rows, gap, eval_rows, long_rows


def verify_long_task_mapping(rows: dict) -> dict:
    """Cross-check task 0..9 instruction strings against the ten Long demo filenames."""
    manifest = json.loads((RUN / 'data_manifest.json').read_text())['tasks']
    files = sorted(DATA10.glob('*_demo.hdf5'))
    if len(files) != 10:
        raise ValueError(f'expected 10 LIBERO-10 demo files, found {len(files)}')
    def norm(s):
        return re.sub(r'[^a-z0-9]+', '', s.lower())
    by_instruction = {}
    for path in files:
        stem = re.sub(r'^.*?scene\d+_', '', path.name, flags=re.I)
        stem = re.sub(r'_demo\.hdf5$', '', stem, flags=re.I).replace('_', ' ')
        by_instruction[norm(stem)] = path.name
    mapping = {}
    for task_id in range(10):
        text = str(manifest[str(task_id)])
        key = norm(text)
        found = by_instruction.get(key)
        if found is None:
            raise AssertionError(f'Long task mapping mismatch for {task_id}: {text}')
        mapping[str(task_id)] = {'instruction': text, 'demo_file': found,
                                 'eval_rows': int(np.sum((np.asarray(rows['task']) == task_id) &
                                                         np.asarray(rows['validation'], bool)))}
    return mapping


def select_medoid(outputs: np.ndarray) -> np.ndarray:
    """Return medoid index for each row; outputs [N,K,T,D]."""
    x = np.asarray(outputs, np.float32)
    if x.ndim != 4 or x.shape[1] < 1:
        raise ValueError(f'expected [N,K,T,D], got {x.shape}')
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    delta = flat[:, :, None, :] - flat[:, None, :, :]
    dist = np.mean(delta * delta, axis=-1)
    return np.argmin(dist.mean(axis=-1), axis=1)


def select_min_edit(outputs: np.ndarray, drafts: np.ndarray) -> np.ndarray:
    x, d = np.asarray(outputs, np.float32), np.asarray(drafts, np.float32)
    if x.shape != d.shape:
        raise ValueError(f'output/draft shape mismatch {x.shape} != {d.shape}')
    return np.argmin(np.mean((x - d) ** 2, axis=(-1, -2)), axis=1)


def select_near(outputs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x, r = np.asarray(outputs, np.float32), np.asarray(reference, np.float32)
    if x.ndim != 4 or r.shape != (x.shape[0], x.shape[2], x.shape[3]):
        raise ValueError('invalid outputs/reference shapes')
    return np.argmin(np.mean((x - r[:, None]) ** 2, axis=(-1, -2)), axis=1)


def validate_candidate_rows(rows: dict, candidates: np.ndarray, query_task: np.ndarray,
                            query_episode: np.ndarray, exclusion: str) -> None:
    """Fail closed if a retrieval list leaves memory or violates its split."""
    candidates = np.asarray(candidates, np.int64)
    query_task = np.asarray(query_task)
    query_episode = np.asarray(query_episode)
    if exclusion not in EXCLUSIONS:
        raise ValueError(f'unsupported exclusion mode: {exclusion}')
    if candidates.ndim != 2 or candidates.shape[0] != len(query_task) or query_episode.shape != query_task.shape:
        raise ValueError('candidate rows and query metadata are not aligned')
    if np.any((candidates < 0) | (candidates >= len(rows['memory']))):
        raise AssertionError('candidate row is outside the dataset')
    if not np.all(np.asarray(rows['memory'], bool)[candidates]):
        raise AssertionError('retrieval candidate is not in the fixed memory set')
    if any(len(np.unique(row)) != len(row) for row in candidates):
        raise AssertionError('retrieval list contains duplicate candidate rows')
    mem_task = np.asarray(rows['task'])[candidates]
    mem_episode = np.asarray(rows['episode'])[candidates]
    if exclusion == 'task' and np.any(mem_task == query_task[:, None]):
        raise AssertionError('same-task candidate violates task exclusion')
    if exclusion == 'episode' and np.any(
        (mem_task == query_task[:, None]) & (mem_episode == query_episode[:, None])
    ):
        raise AssertionError('same-episode candidate violates episode exclusion')


def episode_bootstrap_ci(a: np.ndarray, b: np.ndarray, episodes: np.ndarray,
                         selected: np.ndarray, *, reps: int = BOOTSTRAP_REPS,
                         seed: int = 4101) -> list[float]:
    """Cluster bootstrap, resampling episodes and retaining all selected rows."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    episodes, selected = np.asarray(episodes), np.asarray(selected, bool)
    if a.shape != b.shape or a.shape != episodes.shape or selected.shape != a.shape:
        raise ValueError('bootstrap inputs must be aligned row vectors')
    chosen = np.flatnonzero(selected)
    groups = np.unique(episodes[chosen])
    if not len(groups):
        return [float('nan'), float('nan')]
    d = a - b
    sums, counts = [], []
    for e in groups:
        vals = d[chosen[episodes[chosen] == e]]
        sums.append(vals.sum()); counts.append(len(vals))
    sums, counts = np.asarray(sums), np.asarray(counts)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    # Batch resamples to keep peak memory small and preserve row-weighted estimates.
    for start in range(0, reps, 256):
        end = min(start + 256, reps)
        ix = rng.integers(0, len(groups), size=(end-start, len(groups)))
        draws[start:end] = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def _noise(jax, seed: int, row_id: int):
    return np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(seed), int(row_id)),
                                       (10, 32)), np.float32)


def _gpu_used_mib():
    import subprocess
    try:
        return int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True, timeout=5).splitlines()[0].strip())
    except Exception:
        return None


def _sample_bridge(bridge, jax, jnp, policy, rows, qrows, drafts, t0, seed, batch_start,
                   *, batch_size: int, condition_batch: int):
    """Exact Stage-4-B repeat/order convention; drafts [B,C,10,32]."""
    b, c = drafts.shape[:2]
    obs = policy.obs_batch(rows, qrows)
    obs_rep = jax.tree.map(lambda x: jnp.repeat(x, condition_batch, axis=0), obs)
    if c < condition_batch:
        pad = np.repeat(drafts[:, -1:], condition_batch-c, axis=1)
        drafts = np.concatenate([drafts, pad], axis=1)
    # Observation repeat order is [q0×C, q1×C, ...], so keep candidate rows in [B,C] order.
    d = jnp.asarray(drafts.reshape(b*condition_batch, 10, 32))
    seeds = [int(seed)] * c if np.isscalar(seed) else [int(s) for s in seed]
    if len(seeds) != c:
        raise ValueError('one noise seed is required per draft condition')
    eps0 = np.stack([[_noise(jax, seeds[ci], int(q)) for ci in range(c)] for q in qrows])
    if c < condition_batch:
        eps0 = np.concatenate([eps0, np.repeat(eps0[:, -1:], condition_batch-c, axis=1)], axis=1)
    eps = jnp.asarray(eps0.reshape(b*condition_batch, 10, 32))
    starts = jnp.full((b*condition_batch,), t0, jnp.float32)
    key_seed = int(seed) if np.isscalar(seed) else int(seeds[0])
    key = jax.random.fold_in(jax.random.key(key_seed), int(batch_start))
    pred = np.asarray(bridge(key, obs_rep, d, starts, num_steps=10, noise=eps), np.float32)[:, :, :7]
    return pred.reshape(b, condition_batch, 10, 7)[:, :c]


def _load_runtime():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.shared.nnx_utils import PathRegex, module_jit
    policy = _load_module('pi05_topk_long_policy_run', PI05 / 'run.py')
    base, _ = policy.load_policy(Path('/work/.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'))
    baseline = module_jit(base.sample_actions, static_argnames=('num_steps',))
    probe_rows = np.arange(0, 4, dtype=np.int64)
    # Overlay the already-selected, immutable bridge checkpoint.
    best = json.loads((RUN / 'best_bridge.json').read_text())
    if Path(best['path']) != RUN / 'bridge/001500' or int(best['steps']) != 1500:
        raise ValueError(f'unexpected bridge checkpoint: {best}')
    filt = nnx.All(nnx.Param, PathRegex(r'.*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*'))
    with ocp.PyTreeCheckpointer() as ckpt:
        partial = ckpt.restore(Path(best['path']))
    state = nnx.state(base, filt)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, partial)); nnx.update(base, state)
    return jax, jnp, policy, base, baseline, module_jit(base.sample_bridge, static_argnames=('num_steps',))


def run(out: Path, *, smoke_rows: int | None = None, batch_size: int = 4,
        condition_batch: int = 8, bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    if out.exists():
        occupied=[p for p in out.iterdir() if not (p.is_dir() and p.name.startswith('smoke'))]
        if occupied:
            raise FileExistsError(f'refusing to overwrite non-smoke output: {occupied[0]}')
    out.mkdir(parents=True, exist_ok=True)
    rows, gap, eval_rows, long_rows = load_inputs()
    task_mapping = verify_long_task_mapping(rows)
    full_pos = {int(q): i for i, q in enumerate(eval_rows)}
    hard_source = np.load(RETRIEVAL / 'raw_draft_final_mse.npz', allow_pickle=False)
    if not np.array_equal(hard_source['eval_rows'], eval_rows):
        raise AssertionError('retrieval-4b Eval row order differs from memory-gap')
    threshold = float(hard_source['hard_threshold'])
    if not np.isclose(threshold, .01549410422449, atol=1e-10, rtol=0):
        raise AssertionError(f'hard threshold changed: {threshold}')
    hard_all = np.asarray(hard_source['hard_mask'], bool)
    hard_map = {int(q): bool(h) for q, h in zip(eval_rows, hard_all, strict=True)}
    hard = np.asarray([hard_map[int(q)] for q in long_rows], bool)
    if smoke_rows is not None:
        # Deterministic task-balanced sample of hard Long queries, then fill with Long rows.
        chosen = []
        for task in range(10):
            ix = np.flatnonzero((np.asarray(rows['task'][long_rows]) == task) & hard)
            if len(ix): chosen.append(int(ix[0]))
        if len(chosen) < smoke_rows:
            chosen_set = set(chosen)
            chosen.extend(i for i in np.flatnonzero(hard) if int(i) not in chosen_set)
        if len(chosen) < smoke_rows:
            chosen_set = set(chosen)
            chosen.extend(i for i in range(len(long_rows)) if i not in chosen_set)
        take = np.asarray(chosen[:smoke_rows], np.int64)
        long_rows, hard = long_rows[take], hard[take]

    query_task = np.asarray(rows['task'][long_rows], np.int32)
    query_episode = np.asarray(rows['episode'][long_rows], np.int32)
    episodes = np.asarray([f'{t}:{e}' for t, e in zip(query_task, query_episode, strict=True)])
    truth = np.asarray(rows['actions'][long_rows, :, :7], np.float32)
    n = len(long_rows)
    memory_count = int(np.sum(np.asarray(rows['memory'], bool)))
    if memory_count != 6583:
        raise AssertionError(f'expected fixed 6,583-row memory, got {memory_count}')
    candidate_rows = np.empty((len(KEYS), len(EXCLUSIONS), n, 16), np.int64)
    candidate_scores = np.empty((len(KEYS), len(EXCLUSIONS), n, 16), np.float32)
    key_files = {}
    for ki, keyname in enumerate(KEYS):
        for xi, exclusion in enumerate(EXCLUSIONS):
            path = RETRIEVAL / 'keys' / f'{keyname}_{exclusion}.npz'
            z = np.load(path, allow_pickle=False)
            qmap = {int(q): i for i, q in enumerate(z['query_rows'])}
            try: qix = np.asarray([qmap[int(q)] for q in long_rows], np.int64)
            except KeyError as e: raise AssertionError(f'key file missing Long query row {e}')
            cand = np.asarray(z['cand_rows'][qix, :16], np.int64)
            scores = np.asarray(z['score'][qix, :16], np.float32)
            validate_candidate_rows(rows, cand, query_task, query_episode, exclusion)
            candidate_rows[ki, xi], candidate_scores[ki, xi] = cand, scores
            key_files[f'{keyname}|{exclusion}'] = str(path)

    # Match memory-gap's fixed source-row shuffle, while re-applying the current query's task exclusion.
    gap_protocol = json.loads((GAP / 'protocol.json').read_text())
    source_rows_all = np.asarray(gap_protocol['shuffle_source_rows'], np.int64)
    if len(source_rows_all) != len(eval_rows):
        raise AssertionError('memory-gap shuffle source list is not aligned with all Eval rows')
    task_key_file = np.load(RETRIEVAL / 'keys/wrist_state_contrastive_task.npz', allow_pickle=False)
    source_qmap = {int(q): i for i, q in enumerate(task_key_file['query_rows'])}
    shuffled_source_rows, shuffled_cand_rows = np.empty(n, np.int64), np.empty(n, np.int64)
    for j, q in enumerate(long_rows):
        src = int(source_rows_all[full_pos[int(q)]])
        shuffled_source_rows[j] = src
        source_ix = source_qmap[src]
        options = np.asarray(task_key_file['cand_rows'][source_ix], np.int64)
        allowed = options[np.asarray(rows['task'][options]) != query_task[j]]
        if not len(allowed):
            raise AssertionError(f'no shuffled source candidate obeys task exclusion for query {q}')
        shuffled_cand_rows[j] = allowed[0]
    validate_candidate_rows(rows, shuffled_cand_rows[:, None], query_task, query_episode, 'task')

    # Preserve protocol early so a full run is auditable if interrupted.
    protocol = {
        'stage': '4-C', 'source_run': str(RUN), 'source_memory_gap': str(GAP),
        'source_retrieval': str(RETRIEVAL), 'rows_total_eval': 1318,
        'long_task_ids': list(range(10)), 'long_task_mapping_checked_against': str(DATA10),
        'long_task_mapping': task_mapping, 'query_rows': long_rows.tolist(),
        'query_task': query_task.tolist(), 'query_episode': query_episode.tolist(),
        'memory_rows': memory_count,
        'hard_threshold_dev_q80': threshold, 'hard_rule': 'normal-policy active7 row MSE >= Dev episode-mean q80; no Long retuning',
        'hard_rows': int(hard.sum()), 'key_files': key_files, 'keys': list(KEYS),
        'exclusions': list(EXCLUSIONS), 'candidate_count_generated': 16, 'K_prefixes': list(KS),
        'draft': 'raw rows/actions candidate chunk [10,32], codec not used',
        'expert': str(RUN / 'bridge/001500'), 'expert_steps': 1500,
        'integrator_steps': 10, 't0': list(T0S), 'memory_candidate_noise_seed': NOISE0,
        'noise_rule': 'jax.random.normal(fold_in(key(seed), row_id), [10,32])',
        'bridge_key_rule': 'fold_in(key(seed), batch_start), batch_size=4, matching retrieval-4b',
        'zero_K_seeds': [NOISE0 + 1009*i for i in range(16)],
        'normal_K_seeds': [NOISE0 + 1009*i for i in range(16)],
        'shuffle_rule': 'memory-gap shuffle_source_rows (seed 20261104), source row top-ranked primary-key candidate, re-filtered against current query task',
        'shuffled_source_rows': shuffled_source_rows.tolist(), 'shuffled_candidate_rows': shuffled_cand_rows.tolist(),
        'bootstrap': {'unit': 'episode', 'reps': int(bootstrap_reps), 'seed': NOISE0},
        'smoke_rows': smoke_rows, 'batch_size': batch_size, 'condition_batch': condition_batch,
        'gpu_mem_fraction_env': '.20', 'policy_dependent_reference_keys': ['codec_prior_b','pi05_pred_to_memory_action'],
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    (out / 'progress.jsonl').write_text('')

    jax, jnp, policy, base, baseline, bridge = _load_runtime()
    # Normal π0.5 single seed and K-noise controls.
    normal_single = np.empty((n, 10, 7), np.float32)
    normal_k = np.empty((16, n, 10, 7), np.float32)
    zero_preds = np.empty((len(T0S), 16, n, 10, 7), np.float32)
    mem_preds = np.empty((len(KEYS), len(EXCLUSIONS), len(T0S), 16, n, 10, 7), np.float32)
    shuffled_preds = np.empty((len(T0S), n, 10, 7), np.float32)
    started = time.monotonic()
    for start in range(0, n, batch_size):
        end = min(start+batch_size, n); q = long_rows[start:end]
        obs = policy.obs_batch(rows, q)
        for sstart in range(0, 16, condition_batch):
            seeds = [NOISE0 + 1009*si for si in range(sstart, min(sstart+condition_batch, 16))]
            c, b = len(seeds), end-start
            obs_rep = jax.tree.map(lambda x: jnp.repeat(x, c, axis=0), obs)
            eps = np.stack([[_noise(jax, seed, int(qi)) for seed in seeds] for qi in q])
            pred = np.asarray(baseline(jax.random.key(0), obs_rep, num_steps=10,
                noise=jnp.asarray(eps.reshape(b*c,10,32))), np.float32)[:, :, :7]
            pred = pred.reshape(b,c,10,7).transpose(1,0,2,3)
            normal_k[sstart:sstart+c,start:end] = pred
            if sstart == 0: normal_single[start:end] = pred[0]
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps({'stage':'normal_K','rows_done':end,'rows_total':n,'elapsed_s':time.monotonic()-started,'gpu_used_mib':_gpu_used_mib()})+'\n')

    # Seed 4101 is run in the original single-condition shape to match memory-gap B exactly.
    for ti, t0 in enumerate(T0S):
        for start in range(0, n, batch_size):
            end=min(start+batch_size,n); q=long_rows[start:end]
            d=np.zeros((end-start,1,10,32),np.float32)
            pred=_sample_bridge(bridge,jax,jnp,policy,rows,q,d,t0,NOISE0,start,
                                batch_size=batch_size,condition_batch=1)
            zero_preds[ti,0,start:end]=pred[:,0]
        for sstart in range(1, 16, condition_batch):
            seeds=[NOISE0+1009*si for si in range(sstart,min(sstart+condition_batch,16))]
            for start in range(0,n,batch_size):
                end=min(start+batch_size,n);q=long_rows[start:end];b=end-start
                d=np.zeros((b,len(seeds),10,32),np.float32)
                pred=_sample_bridge(bridge,jax,jnp,policy,rows,q,d,t0,seeds,start,
                                    batch_size=batch_size,condition_batch=condition_batch)
                zero_preds[ti,sstart:sstart+len(seeds),start:end]=pred.transpose(1,0,2,3)

    # Memory candidates: the same fixed noise seed for all candidates of a given query.
    for ki in range(len(KEYS)):
        for xi in range(len(EXCLUSIONS)):
            for ti, t0 in enumerate(T0S):
                for cstart in range(0, 16, condition_batch):
                    cend = min(cstart+condition_batch, 16)
                    for start in range(0, n, batch_size):
                        end = min(start+batch_size, n); q = long_rows[start:end]
                        crows = candidate_rows[ki, xi, start:end, cstart:cend]
                        d = np.asarray(rows['actions'][crows], np.float32)
                        pred = _sample_bridge(bridge, jax, jnp, policy, rows, q, d, t0, NOISE0, start,
                                              batch_size=batch_size, condition_batch=condition_batch)
                        mem_preds[ki, xi, ti, cstart:cend, start:end] = pred.transpose(1,0,2,3)
                    with (out/'progress.jsonl').open('a') as f:
                        f.write(json.dumps({'stage':'memory_candidates','key':KEYS[ki],'exclusion':EXCLUSIONS[xi],
                                            't0':t0,'candidate_start':cstart,'rows_total':n,
                                            'elapsed_s':time.monotonic()-started,'gpu_used_mib':_gpu_used_mib()})+'\n')

    for ti, t0 in enumerate(T0S):
        for start in range(0, n, batch_size):
            end = min(start+batch_size,n); q=long_rows[start:end]
            d = np.asarray(rows['actions'][shuffled_cand_rows[start:end]], np.float32)[:,None]
            pred = _sample_bridge(bridge,jax,jnp,policy,rows,q,d,t0,NOISE0,start,
                                  batch_size=batch_size,condition_batch=condition_batch)
            shuffled_preds[ti,start:end]=pred[:,0]

    # Required parity: K=1 primary rank1 against Stage-4-B cached MSE; first zero seed against Stage 2 B.
    ref_names = np.asarray(hard_source['condition_names']).astype(str)
    zero_ref = np.load(GAP / 'predictions_and_errors.npz', allow_pickle=False)
    gap_names = np.asarray(zero_ref['condition_names']).astype(str)
    for ti,t0 in enumerate(T0S):
        for xi,exclusion in enumerate(EXCLUSIONS):
            cond = f'wrist_state_contrastive|{exclusion}|t0={t0:.1f}'
            ci = int(np.flatnonzero(ref_names == cond)[0])
            row_lookup = {int(q):i for i,q in enumerate(hard_source['eval_rows'])}
            cached = hard_source['final_mse'][ci,0,[row_lookup[int(q)] for q in long_rows]]
            got = np.mean((mem_preds[0,xi,ti,0]-truth)**2,axis=(-1,-2))
            if not np.allclose(got,cached,rtol=2e-5,atol=2e-7):
                raise AssertionError(f'K=1 rank1 parity failed for exclusion={exclusion}, t0={t0}: maxdiff={np.max(np.abs(got-cached))}')
        gci=int(np.flatnonzero(gap_names==f'zero_t0_{t0:.1f}')[0])
        gi={int(q):i for i,q in enumerate(zero_ref['row_ids'])}
        cached0=zero_ref['predictions'][gci,0,[gi[int(q)] for q in long_rows]]
        got0=zero_preds[ti,0]
        if not np.allclose(got0,cached0,rtol=2e-5,atol=2e-7):
            raise AssertionError(f'zero_K first seed parity failed at t0={t0}: maxdiff={np.max(np.abs(got0-cached0))}')

    # Row-level errors and selection errors.
    mem_actions = np.asarray(rows['actions'][candidate_rows][..., :7], np.float32)
    mem_pre = np.mean((mem_actions-truth[None,None,:,None])**2,axis=(-1,-2))
    mem_post = np.mean((mem_preds-truth[None,None,None,None])**2,axis=(-1,-2))
    zero_err = np.mean((zero_preds-truth[None,None])**2,axis=(-1,-2))
    normal_k_err = np.mean((normal_k-truth[None])**2,axis=(-1,-2))
    normal_err = np.mean((normal_single-truth)**2,axis=(-1,-2))
    shuffled_err = np.mean((shuffled_preds-truth[None])**2,axis=(-1,-2))
    selector_names=('rank1','consensus','min_edit','near_normal')
    sel_err=np.empty((len(KEYS),len(EXCLUSIONS),len(T0S),len(selector_names),len(KS),n),np.float32)
    zero_sel=np.empty((len(T0S),len(selector_names),len(KS),n),np.float32)
    normal_sel=np.empty((len(selector_names),len(KS),n),np.float32); normal_sel[:]=np.nan
    def selected_error(x, idx):
        return np.mean((x[np.arange(len(x)),idx]-truth)**2,axis=(-1,-2))
    for ki in range(len(KEYS)):
      for xi in range(len(EXCLUSIONS)):
       for ti in range(len(T0S)):
        for kk,k in enumerate(KS):
         x=mem_preds[ki,xi,ti,:k].transpose(1,0,2,3)
         d=np.asarray(rows['actions'][candidate_rows[ki,xi,:,:k]][...,:7],np.float32)
         indices=[np.zeros(n,np.int64),select_medoid(x),select_min_edit(x,d),select_near(x,normal_single)]
         for si,ix in enumerate(indices): sel_err[ki,xi,ti,si,kk]=selected_error(x,ix)
    for ti in range(len(T0S)):
      for kk,k in enumerate(KS):
        x=zero_preds[ti,:k].transpose(1,0,2,3)
        d=np.zeros_like(x)
        refs=[np.zeros(n,np.int64),select_medoid(x),select_min_edit(x,d),select_near(x,normal_single)]
        for si,ix in enumerate(refs): zero_sel[ti,si,kk]=selected_error(x,ix)
        nx=normal_k[:k].transpose(1,0,2,3)
        normal_sel[0,kk]=np.mean((nx[:,0]-truth)**2,axis=(-1,-2))
        normal_sel[1,kk]=selected_error(nx,select_medoid(nx))

    # Aggregate result table with episode-cluster paired intervals for Long and hard Long.
    subsets={'all_long':np.ones(n,bool),'hard_long':hard}
    summary={'query_rows':n,'hard_rows':int(hard.sum()),'hard_threshold':threshold,
             'task_ids':sorted(set(query_task.tolist())),'selectors':{},'output_sensitivity':{},
             'shuffled':{},'parity':{'primary_K1_rank1_retrieval4b':True,'zero_first_seed_memory_gap':True},
             'decision':{},'bootstrap_reps':bootstrap_reps}
    pair_reports={}
    for ki,keyname in enumerate(KEYS):
      for xi,exclusion in enumerate(EXCLUSIONS):
       for ti,t0 in enumerate(T0S):
        for kk,k in enumerate(KS):
         tag=f'{keyname}|{exclusion}|t0={t0:.1f}|K={k}'
         record={'pre_best@K':{},'post_best@K':{},'selectors':{}}
         for subset_name,mask in subsets.items():
          record['pre_best@K'][subset_name]=float(mem_pre[ki,xi,:, :k].min(axis=1)[mask].mean())
          record['post_best@K'][subset_name]=float(mem_post[ki,xi,ti,:k].min(axis=0)[mask].mean())
          for si,sname in enumerate(selector_names):
           value=sel_err[ki,xi,ti,si,kk]
           record['selectors'].setdefault(sname,{})[subset_name]=float(value[mask].mean())
          # Paired episode bootstrap contrasts.
          zero_best=zero_err[ti,:k].min(axis=0)
          deltas={'post_best_vs_zero_K':(mem_post[ki,xi,ti,:k].min(axis=0),zero_best),
                  **{f'{s}_vs_zero_K':(sel_err[ki,xi,ti,si,kk],zero_sel[ti,si,kk]) for si,s in enumerate(selector_names)},
                  **{f'{s}_vs_normal_single':(sel_err[ki,xi,ti,si,kk],normal_err) for si,s in enumerate(selector_names)}}
          for cname,(a,b) in deltas.items():
            record.setdefault('paired_delta',{}).setdefault(cname,{})[subset_name]={
              'mean':float(np.asarray(a)[mask].mean()-np.asarray(b)[mask].mean()),
              'ci95':episode_bootstrap_ci(a,b,episodes,mask,reps=bootstrap_reps,
                    seed=NOISE0+ki*101+xi*13+ti*7+kk+(0 if subset_name=='all_long' else 50000))}
          pair_reports.setdefault(tag,{})[subset_name]=record
    summary['comparison_table']=pair_reports

    # Normal-K controls and shuffled diagnostics are reported separately from memory selectors.
    shuffled_comparison = {}
    for ti,t0 in enumerate(T0S):
      summary['selectors'][f't0={t0:.1f}']={'zero_K':{},'normal_K':{}}
      for kk,k in enumerate(KS):
       for subset_name,mask in subsets.items():
        summary['selectors'][f't0={t0:.1f}']['zero_K'][str(k)]={
          'post_best':float(zero_err[ti,:k].min(axis=0)[mask].mean()),
          **{s:float(zero_sel[ti,si,kk][mask].mean()) for si,s in enumerate(selector_names)}}
        summary['selectors'][f't0={t0:.1f}']['normal_K'][str(k)]={
          'post_best':float(normal_k_err[:k].min(axis=0)[mask].mean()),
          'rank1':float(normal_sel[0,kk][mask].mean()),'consensus':float(normal_sel[1,kk][mask].mean())}
      for subset_name,mask in subsets.items():
       summary['shuffled'].setdefault(subset_name,{})[f't0={t0:.1f}']=float(shuffled_err[ti,mask].mean())
       # Compare shuffled draft against the deployment-like primary-key rank1
       # output (K=1), paired by query and bootstrapped by episode.
       primary_rank1 = sel_err[0, 0, ti, 0, 0]
       delta = shuffled_err[ti] - primary_rank1
       ci = episode_bootstrap_ci(
           shuffled_err[ti], primary_rank1, episodes, mask, reps=bootstrap_reps,
           seed=NOISE0 + 8000 + ti * 100 + (0 if subset_name == 'all_long' else 50000),
       )
       shuffled_comparison.setdefault(subset_name, {})[f't0={t0:.1f}'] = {
           'shuffled_minus_primary_rank1_mean': float(delta[mask].mean()),
           'ci95': ci,
           'distinguishable_from_zero': bool(ci[0] > 0 or ci[1] < 0),
       }
    summary['shuffled_comparison_vs_primary_rank1'] = shuffled_comparison
    # Sensitivity ratio: average off-diagonal pairwise distance, generated / raw drafts.
    pri=0; pxi=EXCLUSIONS.index('task'); ti=0
    outx=mem_preds[pri,pxi,ti].transpose(1,0,2,3).reshape(n,16,-1)
    dx=np.asarray(rows['actions'][candidate_rows[pri,pxi]][...,:7],np.float32).reshape(n,16,-1)
    def offdiag_mean(x):
      d=np.mean((x[:,:,None]-x[:,None,:])**2,axis=-1); k=d.shape[1]
      return d.sum(axis=(1,2))/max(k*(k-1),1)
    out_dist=offdiag_mean(outx); draft_dist=offdiag_mean(dx)
    sens=out_dist/np.maximum(draft_dist,1e-12)
    for subset_name,mask in subsets.items():
      summary['output_sensitivity'][subset_name]={'ratio_mean':float(sens[mask].mean()),
         'output_pair_mse':float(out_dist[mask].mean()),'draft_pair_mse':float(draft_dist[mask].mean())}
    # G1/G2 are evaluated only on primary key/task exclusion/hard rows, per fixed plan.
    g1=[]; g2=[]
    for ti,t0 in enumerate(T0S):
      for kk,k in enumerate(KS[:3]):
        a=mem_post[0,0,ti,:k].min(axis=0); b=zero_err[ti,:k].min(axis=0)
        ci=episode_bootstrap_ci(a,b,episodes,hard,reps=bootstrap_reps,seed=NOISE0+ti*100+kk)
        g1.append({'t0':t0,'K':k,'delta_mean':float((a[hard]-b[hard]).mean()),'ci95':ci})
        for si,sname in enumerate(selector_names):
          x=sel_err[0,0,ti,si,kk]; z=zero_sel[ti,si,kk]
          c=episode_bootstrap_ci(x,z,episodes,hard,reps=bootstrap_reps,seed=NOISE0+1000+ti*100+kk*10+si)
          g2.append({'t0':t0,'K':k,'selector':sname,'delta_mean':float((x[hard]-z[hard]).mean()),'ci95':c})
    g1_pass=any(x['ci95'][1]<0 for x in g1)
    g2_pass=g1_pass and any(x['ci95'][1]<0 for x in g2)
    shuffled_hard_indistinguishable = any(
        not row['distinguishable_from_zero']
        for row in shuffled_comparison.get('hard_long', {}).values()
    )
    summary['decision']={'G1_generator_can_use_candidates':g1_pass,'G1_details':g1,
       'G2_runtime_selector_promise':g2_pass,'G2_details':g2,
       'shuffled_indistinguishable_warning':shuffled_hard_indistinguishable,
       'draft_ignored_warning':bool(summary['output_sensitivity']['hard_long']['ratio_mean']<1e-3 or
                                    shuffled_hard_indistinguishable)}

    out_file=out/'predictions_and_errors.npz'
    if out_file.exists(): raise FileExistsError(out_file)
    np.savez_compressed(out_file, query_rows=long_rows, query_task=query_task, query_episode=query_episode,
      hard_mask=hard, candidate_rows=candidate_rows, candidate_scores=candidate_scores,
      truth=truth, memory_predictions=mem_preds, memory_pre_mse=mem_pre, memory_post_mse=mem_post,
      selector_mse=sel_err, zero_predictions=zero_preds, zero_mse=zero_err,
      zero_selector_mse=zero_sel, normal_predictions=normal_k, normal_single=normal_single,
      normal_k_mse=normal_k_err, normal_selector_mse=normal_sel, normal_single_mse=normal_err,
      shuffled_source_rows=shuffled_source_rows, shuffled_candidate_rows=shuffled_cand_rows,
      shuffled_predictions=shuffled_preds, shuffled_mse=shuffled_err,
      key_names=np.asarray(KEYS), exclusion_names=np.asarray(EXCLUSIONS), t0_values=np.asarray(T0S),
      K_values=np.asarray(KS), selector_names=np.asarray(selector_names))
    summary.update({'seconds':time.monotonic()-started,'output_file':str(out_file),
                    'gpu_used_mib_end':_gpu_used_mib()})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    with (out/'progress.jsonl').open('a') as f:
      f.write(json.dumps({'stage':'complete','seconds':summary['seconds'],'rows':n,'hard_rows':int(hard.sum())})+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--smoke-rows',type=int,default=None)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--condition-batch',type=int,default=8)
    ap.add_argument('--bootstrap-reps',type=int,default=BOOTSTRAP_REPS)
    args=ap.parse_args()
    if args.batch_size != 4 or args.condition_batch != 8:
      raise ValueError('fixed bridge parity requires --batch-size 4 --condition-batch 8')
    result=run(args.output,smoke_rows=args.smoke_rows,batch_size=args.batch_size,
               condition_batch=args.condition_batch,bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({'rows':result['query_rows'],'hard_rows':result['hard_rows'],
                      'seconds':result['seconds'],'decision':result['decision']},ensure_ascii=False))


if __name__=='__main__': main()
