"""Inference-available retrieval keys for the PI05 memory score audit.

All scores use the convention "larger is better".  Search candidates are
ranked directly from the whole memory bank; the returned key files contain
the top 64 rows for the requested exclusion rule.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = HERE.parents[5]
STATE = Path(os.environ.get('SKILLNET_STATE_DIR', REPO / '.skillnet-state')).expanduser()
PI05_ROOT = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser()
RUN = PI05_ROOT / 'run-01'
OUT = PI05_ROOT / 'score-audit-01'
TOP_K = 64


def load_rows(run: Path = RUN) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (run / 'rows').glob('*.npy')}


def history_features(rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return previous executed action chunk (70D) and its availability flag.

    The prepared rows have stride=10 and each action row covers ten frames, so
    frame-10 is exactly the preceding executed chunk whenever it is present.
    """
    n = len(rows['frame'])
    lookup = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(rows['episode'], rows['frame']))}
    values = np.zeros((n, 70), np.float32)
    present = np.zeros(n, np.float32)
    actions = rows['actions'][:, :, :7]
    for i, (episode, frame) in enumerate(zip(rows['episode'], rows['frame'])):
        previous = lookup.get((int(episode), int(frame) - 10))
        if previous is not None:
            values[i] = np.asarray(actions[previous], np.float32).reshape(-1)
            present[i] = 1.
    return values, present


def state_views(state: np.ndarray) -> dict[str, np.ndarray]:
    s = np.asarray(state, np.float32)[:, :8]
    # OpenPI LIBERO state contract: xyz, quaternion, gripper opening command.
    return {'state_all': s, 'state_pose_grip': s[:, 3:8], 'state_height_grip': s[:, [2, 7]]}


def _standardize(x: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, ref = np.asarray(x, np.float32), np.asarray(reference, np.float32)
    mean, std = ref.mean(0), ref.std(0)
    std = np.where(std < 1e-5, 1., std)
    return (x - mean) / std, mean, std


def _eligible(memory_ids: np.ndarray, rows: dict[str, np.ndarray], query: int, exclusion: str) -> np.ndarray:
    if exclusion == 'task':
        return memory_ids[np.asarray(rows['task'][memory_ids]) != int(rows['task'][query])]
    if exclusion == 'episode':
        return memory_ids[np.asarray(rows['episode'][memory_ids]) != int(rows['episode'][query])]
    raise ValueError(f'unsupported exclusion: {exclusion}')


def top64(scores: np.ndarray, ids: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    scores, ids = np.asarray(scores, np.float32), np.asarray(ids, np.int64)
    if len(ids) < k:
        raise ValueError(f'only {len(ids)} candidates available, need {k}')
    # Stable row-id tie-break makes the files reproducible.
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def write_key(name: str, query_rows: np.ndarray, rows: dict[str, np.ndarray], memory_ids: np.ndarray,
              scorer, exclusion: str, description: str, out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'keys' / f'{name}_{exclusion}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    qids = np.asarray(query_rows, np.int64)
    candidates = np.empty((len(qids), TOP_K), np.int64)
    scores_out = np.empty((len(qids), TOP_K), np.float32)
    started = time.monotonic()
    for qi, q in enumerate(qids):
        eligible = _eligible(memory_ids, rows, int(q), exclusion)
        all_scores = np.asarray(scorer(int(q), eligible), np.float32).reshape(-1)
        if len(all_scores) != len(eligible) or not np.isfinite(all_scores).all():
            raise ValueError(f'invalid candidate scores for query {q}')
        candidates[qi], scores_out[qi] = top64(all_scores, eligible)
    np.savez_compressed(path, query_rows=qids, cand_rows=candidates, score=scores_out,
                        exclusion=np.asarray(exclusion), description=np.asarray(description))
    return {'name': name, 'exclusion': exclusion, 'rows': len(qids), 'seconds': time.monotonic()-started,
            'path': str(path), 'description': description}


def generate_tabular(run: Path = RUN, out: Path = OUT, limit_queries: int | None = None) -> list[dict]:
    rows = load_rows(run)
    query_rows = np.flatnonzero(rows['validation'])
    if limit_queries is not None:
        query_rows = query_rows[:limit_queries]
    memory_ids = np.flatnonzero(rows['memory'])
    if len(memory_ids) != 6583:
        raise ValueError(f'expected 6,583 memory rows, found {len(memory_ids)}')

    results = []
    for view_name, raw in state_views(rows['state']).items():
        standardized, _, _ = _standardize(raw, raw[memory_ids])
        norm = np.linalg.norm(standardized, axis=1).clip(1e-8)
        unit = standardized / norm[:, None]
        scorer = lambda q, ids, unit=unit: unit[ids] @ unit[q]
        for exclusion in ('task', 'episode'):
            results.append(write_key(view_name, query_rows, rows, memory_ids, scorer, exclusion,
                'Cosine of memory-standardized proprioception; indices xyz/quaternion/gripper as documented in run.py.', out))

    hist, present = history_features(rows)
    hist_z, _, _ = _standardize(hist, hist[memory_ids[present[memory_ids] > 0]])
    hist_vec = np.concatenate([hist_z, present[:, None]], axis=1)
    hist_vec /= np.linalg.norm(hist_vec, axis=1, keepdims=True).clip(1e-8)
    scorer = lambda q, ids: hist_vec[ids] @ hist_vec[q]
    for exclusion in ('task', 'episode'):
        results.append(write_key('history_previous_action', query_rows, rows, memory_ids, scorer, exclusion,
            'Cosine over standardized preceding executed 10x7 action chunk plus history-present bit; frame 0 uses zeros and flag 0.', out))

    # Joint, standardized non-VLM inputs. This remains an unlearned distance baseline.
    state, _, _ = _standardize(np.asarray(rows['state'])[:, :8], np.asarray(rows['state'])[memory_ids, :8])
    combo = np.concatenate([state, hist_z, present[:, None]], axis=1)
    combo /= np.linalg.norm(combo, axis=1, keepdims=True).clip(1e-8)
    scorer = lambda q, ids: combo[ids] @ combo[q]
    for exclusion in ('task', 'episode'):
        results.append(write_key('state_plus_history', query_rows, rows, memory_ids, scorer, exclusion,
            'Cosine over concatenated memory-standardized 8D proprioception and preceding executed action history.', out))
    return results


def generate_codec(run: Path = RUN, out: Path = OUT, limit_queries: int | None = None) -> list[dict]:
    """Rank using M1 prior outputs; CPU inference over already cached features."""
    sys.path.insert(0, str(PI05))
    from run import restore_codec
    rows = load_rows(run)
    qids = np.flatnonzero(rows['validation'])
    if limit_queries is not None:
        qids = qids[:limit_queries]
    memory_ids = np.flatnonzero(rows['memory'])
    features = np.load(run / 'pooled.npy', mmap_mode='r')
    model = restore_codec(run)
    b, r = model.prior(np.asarray(features), np.asarray(rows['state']))
    b = np.asarray(b, np.float32); r = np.asarray(r, np.float32)
    outputs = {'codec_prior_b': b, 'codec_prior_r': r, 'codec_prior_br': np.concatenate([b, r], axis=1)}
    results = []
    for name, value in outputs.items():
        standardized, _, _ = _standardize(value, value[memory_ids])
        norms = np.linalg.norm(standardized, axis=1).clip(1e-8)
        unit = standardized / norms[:, None]
        scorer = lambda q, ids, unit=unit: unit[ids] @ unit[q]
        for exclusion in ('task', 'episode'):
            results.append(write_key(name, qids, rows, memory_ids, scorer, exclusion,
                'Cosine over memory-standardized output of the frozen M1 prior_net; inputs are cached pooled VLM feature and current proprioception.', out))
    return results


def generate_pi05(run: Path = RUN, out: Path = OUT, checkpoint: Path | None = None,
                  limit_rows: int | None = None, batch_size: int = 4) -> dict:
    """Generate official LIBERO pi0.5 predictions with row-index fixed noise.

    This explicit function is kept separate because it needs the shared GPU.
    `limit_rows` is for timing/shape smoke only; full keys require every query
    and memory row to be included in the prediction cache.
    """
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.22')
    import jax
    import jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    sys.path.insert(0, str(PI05))
    from run import load_policy, obs_batch
    rows = load_rows(run)
    checkpoint = checkpoint or REPO / '.skillnet-state/checkpoints/openpi-assets/checkpoints/pi05_libero'
    model, _ = load_policy(checkpoint)
    sample = module_jit(model.sample_actions, static_argnames=('num_steps',))
    qids = np.flatnonzero(rows['validation'])
    memory_ids = np.flatnonzero(rows['memory'])
    if limit_rows is None:
        needed = np.unique(np.concatenate([qids, memory_ids]))
    else:
        # Timing smoke covers both sides of the intended query/candidate use.
        n_query = max(1, limit_rows // 2)
        needed = np.unique(np.concatenate([qids[:n_query], memory_ids[:limit_rows-n_query]]))
    pred = np.zeros((len(rows['actions']), 10, 7), np.float32)
    root = jax.random.key(20261104)
    started = time.monotonic()
    for start in range(0, len(needed), batch_size):
        ids = needed[start:start+batch_size]
        eps = np.stack([np.asarray(jax.random.normal(jax.random.fold_in(root, int(i)), (10, 32))) for i in ids])
        value = sample(jax.random.key(0), obs_batch(rows, ids), num_steps=10, noise=jnp.asarray(eps))
        pred[ids] = np.asarray(value)[:, :, :7]
    seconds = time.monotonic() - started
    return {'row_ids': needed, 'predicted_actions': pred, 'seconds': seconds,
            'seconds_per_row': seconds/max(len(needed), 1), 'batch_size': batch_size,
            'checkpoint': str(checkpoint), 'fixed_noise': 'jax.random.fold_in(key(20261104), row_id)'}


def generate_policy_keys(predicted: np.ndarray, run: Path = RUN, out: Path = OUT,
                         limit_queries: int | None = None) -> list[dict]:
    rows = load_rows(run)
    qids = np.flatnonzero(rows['validation'])
    if limit_queries is not None:
        qids = qids[:limit_queries]
    memory_ids = np.flatnonzero(rows['memory'])
    pred = np.asarray(predicted, np.float32)
    if pred.shape != (len(rows['actions']), 10, 7):
        raise ValueError('predicted action cache must be [all_rows,10,7]')
    true_actions = np.asarray(rows['actions'][:, :, :7], np.float32)
    variants = {'pi05_pred_to_pred': pred,
                'pi05_pred_to_memory_action': true_actions}
    results = []
    for name, values in variants.items():
        for exclusion in ('task', 'episode'):
            results.append(write_key(name, qids, rows, memory_ids,
                lambda q, ids, values=values: -np.mean((values[ids]-pred[q])**2, axis=(1,2)), exclusion,
                'Negative normalized active-7 action MSE; query representation is official pi05_libero fixed-noise prediction; candidate is predicted action or stored demonstration action.', out))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['tabular', 'codec', 'pi05-smoke', 'pi05-full'])
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'tabular':
        result = generate_tabular(args.run, args.out, args.limit)
    elif args.stage == 'codec':
        result = generate_codec(args.run, args.out, args.limit)
    else:
        # One file per invocation; do not silently overwrite a prediction cache.
        pred_path = args.out / ('pi05_pred_actions_smoke.npy' if args.stage == 'pi05-smoke' else 'pi05_pred_actions.npy')
        if pred_path.exists():
            raise FileExistsError(pred_path)
        sampled = generate_pi05(args.run, args.out, limit_rows=128 if args.stage == 'pi05-smoke' else None,
                                batch_size=args.batch_size)
        np.save(pred_path, sampled.pop('predicted_actions'), allow_pickle=False)
        sampled.pop('row_ids', None)
        result = {'prediction_path': str(pred_path), **sampled}
        if args.stage == 'pi05-full':
            result['keys'] = generate_policy_keys(np.load(pred_path, mmap_mode='r', allow_pickle=False), args.run, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
