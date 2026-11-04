"""Compact cross-task action-contrastive retrieval key (CPU by default)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PI05))
import keys


def raw_inputs(rows, run: Path):
    pooled = np.load(run / 'pooled.npy', mmap_mode='r')
    history, flag = keys.history_features(rows)
    return np.concatenate([np.asarray(pooled, np.float32), np.asarray(rows['state'], np.float32)[:, :8],
                           history, flag[:, None]], axis=1)


def pair_pool(train_ids, rows, actions, *, count=250_000, seed=261104):
    rng = np.random.default_rng(seed)
    left_parts, right_parts, mse_parts = [], [], []
    remaining = int(count)
    while remaining:
        size = min(250_000, remaining)
        left = rng.choice(train_ids, size=size, replace=True)
        right = rng.choice(train_ids, size=size, replace=True)
        different_task = np.asarray(rows['task'][left]) != np.asarray(rows['task'][right])
        delta = actions[left] - actions[right]
        mse = np.mean(delta * delta, axis=(1, 2))
        left_parts.append(left[different_task]); right_parts.append(right[different_task]); mse_parts.append(mse[different_task])
        remaining -= size
    return np.concatenate(left_parts), np.concatenate(right_parts), np.concatenate(mse_parts)


def fit_and_select(run: Path = keys.RUN, out: Path = keys.OUT, *, epochs=4, pair_count=10_000_000,
                   seed=20261104, threads=8):
    import torch
    from torch import nn
    from run import calibration_episodes
    torch.set_num_threads(int(threads))
    rows = keys.load_rows(run)
    memory_ids = np.flatnonzero(rows['memory'])
    train_ids = np.flatnonzero(~rows['memory'] & ~rows['validation'])
    dev_episodes, eval_episodes = calibration_episodes(rows)
    dev_ids = np.flatnonzero(rows['validation'] & np.isin(rows['episode'], dev_episodes))
    x = raw_inputs(rows, run)
    mean, std = x[train_ids].mean(0), x[train_ids].std(0)
    std[std < 1e-5] = 1.
    x = ((x - mean) / std).astype(np.float32)
    action = np.asarray(rows['actions'][:, :, :7], np.float32)
    # Action standardization is already defined by the prepared dataset; the
    # requested thresholds therefore use its active-7 normalized MSE directly.
    pair_l, pair_r, pair_mse = pair_pool(train_ids, rows, action, count=pair_count, seed=seed)
    thresholds = (.005, .01, .02)
    rng = np.random.default_rng(seed + 9)
    device = torch.device('cpu')
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    mem = torch.as_tensor(memory_ids, dtype=torch.long)
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / 'learned_key.pt'
    result_file = out / 'learned_key_fit.json'
    if model_file.exists() or result_file.exists():
        raise FileExistsError('learned key output already exists; refusing overwrite')
    results = []
    models = []
    for threshold in thresholds:
        pos = np.flatnonzero(pair_mse <= threshold)
        neg = np.flatnonzero(pair_mse > max(threshold, .03))
        if len(pos) < 256 or len(neg) < 256:
            raise ValueError(f'not enough cross-task positive/negative pairs at {threshold}: {len(pos)}/{len(neg)}')
        # The pairwise objective always includes positives from distinct tasks.
        sample_count = min(8_192, max(2_048, len(pos)))
        selected_pos = rng.choice(pos, size=sample_count, replace=len(pos) < sample_count)
        selected_neg = rng.choice(neg, size=sample_count, replace=len(neg) < sample_count)
        left = np.concatenate([pair_l[selected_pos], pair_l[selected_neg]])
        right = np.concatenate([pair_r[selected_pos], pair_r[selected_neg]])
        label = np.concatenate([np.ones(len(selected_pos), np.float32), np.zeros(len(selected_neg), np.float32)])
        order = rng.permutation(len(label)); left, right, label = left[order], right[order], label[order]
        net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 64)).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        started = time.monotonic(); losses = []
        batch_size = 256
        for epoch in range(int(epochs)):
            perm = rng.permutation(len(label)); total = 0.; seen = 0
            net.train()
            for start in range(0, len(perm), batch_size):
                ix = perm[start:start+batch_size]
                a = torch.nn.functional.normalize(net(tx[left[ix]]), dim=-1)
                b = torch.nn.functional.normalize(net(tx[right[ix]]), dim=-1)
                similarity = (a*b).sum(-1)
                y = torch.as_tensor(label[ix], device=device)
                # Cross-task positives are pulled together; all sampled distant
                # pairs are pushed below a cosine margin.
                loss = (y * (1-similarity).square() + (1-y) * torch.relu(similarity-.35).square()).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(ix); seen += len(ix)
            losses.append(total / max(seen, 1))
        net.eval()
        with torch.no_grad():
            emb = torch.nn.functional.normalize(net(tx), dim=-1).cpu().numpy()
        # Dev-only model selection: task-excluded nearest key, judged by the
        # same normalized active-7 action error used by the common evaluator.
        dev_errors = []
        for q in dev_ids:
            eligible = keys._eligible(memory_ids, rows, int(q), 'task')
            best = eligible[np.argmax(emb[eligible] @ emb[q])]
            dev_errors.append(float(np.mean((action[q] - action[best])**2)))
        record = {'positive_action_mse_max': threshold, 'cross_task_positive_pairs': int(len(pos)),
                  'distant_negative_pairs': int(len(neg)), 'pair_examples_each_class': int(len(selected_pos)),
                  'epoch_pair_loss': losses, 'dev_queries': int(len(dev_ids)),
                  'dev_task_excluded_top1_action_mse': float(np.mean(dev_errors)),
                  'seconds': time.monotonic()-started}
        results.append(record); models.append((record['dev_task_excluded_top1_action_mse'], threshold, net, emb))
    _, selected_threshold, selected_model, embeddings = min(models, key=lambda item: item[0])
    torch.save({'state_dict': selected_model.state_dict(), 'input_mean': mean, 'input_std': std,
                'input_dim': int(x.shape[1]), 'hidden_dim': 128, 'embedding_dim': 64,
                'positive_action_mse_max': selected_threshold, 'seed': seed}, model_file)
    summary = {'selected_positive_action_mse_max': selected_threshold, 'selection': 'minimum Dev task-excluded top-1 action MSE',
               'dev_episodes': np.asarray(dev_episodes).tolist(), 'eval_episodes': np.asarray(eval_episodes).tolist(),
               'train_rows': int(len(train_ids)), 'memory_rows': int(len(memory_ids)), 'results': results,
               'architecture': 'shared MLP 2127->128 GELU->64; L2-normalized output',
               'objective': 'pairwise contrastive; action-MSE-positive pairs are always cross-task; negatives > max(threshold,0.03)',
               'candidate_selection': 'held-out validation Dev episodes only; Eval episodes were not scored for selection'}
    result_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    del selected_model
    return summary, embeddings


def generate_keys(embeddings: np.ndarray, run: Path = keys.RUN, out: Path = keys.OUT):
    rows = keys.load_rows(run); qids = np.flatnonzero(rows['validation']); mids = np.flatnonzero(rows['memory'])
    z = np.asarray(embeddings, np.float32)
    result = []
    for exclusion in ('task', 'episode'):
        result.append(keys.write_key('learned_cross_task_contrastive', qids, rows, mids,
            lambda q, ids: z[ids] @ z[q], exclusion,
            '64D L2-normalized learned key from pooled VLM token feature + state[0:8] + preceding executed action history; trained on 16,721 non-memory/non-validation rows with cross-task action-near positives; selected on Dev only.', out))
    return result


def extract_wrist_token_mean(run: Path, rows: dict[str, np.ndarray], out_file: Path,
                             *, tokens_file: Path | None = None, batch_rows: int = 64) -> np.ndarray:
    """Cache the mean wrist-camera prefix token per row, without VLM inference.

    `tokens.npy` is memory-mapped and read in row blocks. The wrist group is
    identified by the saved `groups.npy` labels (1), not by annotation.
    """
    out_file = Path(out_file)
    if out_file.exists():
        raise FileExistsError(f'refusing to overwrite {out_file}')
    tokens_file = tokens_file or run / 'tokens.npy'
    tokens = np.load(tokens_file, mmap_mode='r', allow_pickle=False)
    groups = np.load(run / 'groups.npy', mmap_mode='r', allow_pickle=False)
    if tokens.ndim != 3 or tokens.shape[0] != len(rows['task']) or groups.shape != tokens.shape[:2]:
        raise ValueError('token/group cache does not align with prepared rows')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wrist_indices = np.flatnonzero(np.asarray(groups[0]) == 1)
    if not len(wrist_indices):
        raise ValueError('first row has no wrist tokens')
    means = np.lib.format.open_memmap(out_file, mode='w+', dtype='float32',
                                      shape=(len(tokens), tokens.shape[-1]))
    for start in range(0, len(tokens), batch_rows):
        end = min(start + batch_rows, len(tokens))
        block_groups = np.asarray(groups[start:end])
        if not np.all(block_groups[:, wrist_indices] == 1) or np.any(np.sum(block_groups == 1, axis=1) != len(wrist_indices)):
            raise ValueError(f'wrist-token positions vary within rows {start}:{end}')
        # Read only the contiguous wrist token slice; avoid faulting base and
        # instruction token pages into RSS on the large shared cache.
        block_tokens = np.asarray(tokens[start:end, wrist_indices], dtype=np.float32)
        means[start:end] = block_tokens.mean(axis=1)
    means.flush()
    return means


def wrist_key_inputs(wrist: np.ndarray, rows: dict[str, np.ndarray], *, include_history: bool) -> np.ndarray:
    """Construct inference-available wrist/proprio inputs, with optional past."""
    current = [np.asarray(wrist, np.float32), np.asarray(rows['state'], np.float32)[:, :8]]
    if not include_history:
        return np.concatenate(current, axis=1)
    prev_action, present = keys.history_features(rows)
    n = len(rows['task'])
    lookup = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(rows['episode'], rows['frame']))}
    prev_wrist = np.zeros_like(np.asarray(wrist, np.float32))
    prev_state = np.zeros((n, 8), np.float32)
    prev_present = np.zeros(n, np.float32)
    state = np.asarray(rows['state'], np.float32)[:, :8]
    for i, (episode, frame) in enumerate(zip(rows['episode'], rows['frame'])):
        previous = lookup.get((int(episode), int(frame) - 10))
        if previous is not None:
            prev_wrist[i] = wrist[previous]
            prev_state[i] = state[previous]
            prev_present[i] = 1.
    return np.concatenate([*current, prev_wrist, prev_state,
                           np.asarray(prev_action, np.float32), prev_present[:, None]], axis=1)


def fit_wrist_contrastive(x: np.ndarray, rows: dict[str, np.ndarray], *, out: Path,
                          run: Path = keys.RUN, epochs: int = 4, pair_count: int = 10_000_000,
                          seed: int = 20261104, threads: int = 8) -> tuple[dict, np.ndarray]:
    """Fit an action-neighborhood contrastive wrist key using non-memory train rows.

    Three positive-distance thresholds are fit; selection is by Dev-only,
    task-excluded Top-1 raw action MSE. Eval rows are never used for selection.
    """
    import torch
    from torch import nn
    from run import calibration_episodes
    torch.set_num_threads(int(threads))
    x = np.asarray(x, np.float32)
    memory_ids = np.flatnonzero(rows['memory'])
    train_ids = np.flatnonzero(~rows['memory'] & ~rows['validation'])
    dev_episodes, eval_episodes = calibration_episodes(rows)
    dev_ids = np.flatnonzero(rows['validation'] & np.isin(rows['episode'], dev_episodes))
    mean, std = x[train_ids].mean(0), x[train_ids].std(0)
    std = np.where(std < 1e-5, 1., std)
    x = ((x - mean) / std).astype(np.float32)
    action = np.asarray(rows['actions'][:, :, :7], np.float32)
    pair_l, pair_r, pair_mse = pair_pool(train_ids, rows, action, count=pair_count, seed=seed)
    thresholds = (.005, .01, .02)
    rng = np.random.default_rng(seed + 19)
    device = torch.device('cpu')
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model_file = out / 'wrist_contrastive.pt'
    result_file = out / 'wrist_contrastive_fit.json'
    if model_file.exists() or result_file.exists():
        raise FileExistsError(f'refusing to overwrite learned-key outputs in {out}')
    results = []; fitted = []
    for threshold in thresholds:
        pos = np.flatnonzero(pair_mse <= threshold)
        neg = np.flatnonzero(pair_mse > max(threshold, .03))
        if len(pos) < 256 or len(neg) < 256:
            raise ValueError(f'not enough cross-task pairs at {threshold}: {len(pos)}/{len(neg)}')
        count = min(8_192, max(2_048, len(pos)))
        pi = rng.choice(pos, count, replace=len(pos) < count)
        ni = rng.choice(neg, count, replace=len(neg) < count)
        left = np.concatenate([pair_l[pi], pair_l[ni]])
        right = np.concatenate([pair_r[pi], pair_r[ni]])
        labels = np.concatenate([np.ones(count, np.float32), np.zeros(count, np.float32)])
        order = rng.permutation(len(labels)); left, right, labels = left[order], right[order], labels[order]
        net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 64)).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []; batch_size = 256
        for _ in range(int(epochs)):
            perm = rng.permutation(len(labels)); total = 0.; seen = 0
            net.train()
            for start in range(0, len(perm), batch_size):
                ix = perm[start:start + batch_size]
                a = torch.nn.functional.normalize(net(tx[left[ix]]), dim=-1)
                b = torch.nn.functional.normalize(net(tx[right[ix]]), dim=-1)
                sim = (a * b).sum(-1); y = torch.as_tensor(labels[ix], device=device)
                loss = (y * (1-sim).square() + (1-y) * torch.relu(sim-.35).square()).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(ix); seen += len(ix)
            losses.append(total / max(seen, 1))
        net.eval()
        with torch.no_grad():
            embedding = torch.nn.functional.normalize(net(tx), dim=-1).cpu().numpy()
        errs = []
        for q in dev_ids:
            eligible = keys._eligible(memory_ids, rows, int(q), 'task')
            chosen = eligible[np.argmax(embedding[eligible] @ embedding[q])]
            errs.append(float(np.mean((action[q] - action[chosen]) ** 2)))
        rec = {'positive_action_mse_max': threshold, 'positive_pairs': int(len(pos),),
               'negative_pairs': int(len(neg)), 'pair_examples_per_class': int(count),
               'epoch_pair_loss': losses, 'dev_queries': int(len(dev_ids)),
               'dev_task_excluded_top1_action_mse': float(np.mean(errs))}
        results.append(rec); fitted.append((rec['dev_task_excluded_top1_action_mse'], threshold, net, embedding))
    _, chosen_threshold, chosen_model, embeddings = min(fitted, key=lambda z: z[0])
    torch.save({'state_dict': chosen_model.state_dict(), 'input_mean': mean, 'input_std': std,
                'input_dim': int(x.shape[1]), 'hidden_dim': 128, 'embedding_dim': 64,
                'positive_action_mse_max': chosen_threshold, 'seed': seed}, model_file)
    summary = {'selected_positive_action_mse_max': chosen_threshold,
               'selection': 'minimum Dev task-excluded Top-1 raw action MSE',
               'train_rows': int(len(train_ids)), 'memory_rows': int(len(memory_ids)),
               'dev_rows': int(len(dev_ids)), 'dev_episodes': np.asarray(dev_episodes).tolist(),
               'eval_episodes': np.asarray(eval_episodes).tolist(), 'input_dim': int(x.shape[1]),
               'architecture': f'{x.shape[1]}->128 GELU->64 L2 normalized',
               'positive_definition': 'action MSE <= threshold and pair tasks differ',
               'negative_definition': 'action MSE > max(threshold, 0.03)', 'results': results}
    result_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary, embeddings


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=keys.RUN); parser.add_argument('--out',type=Path,default=keys.OUT)
    parser.add_argument('--epochs',type=int,default=4); parser.add_argument('--pair-count',type=int,default=10_000_000)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    summary, embedding=fit_and_select(args.run,args.out,epochs=args.epochs,pair_count=args.pair_count,threads=args.threads)
    files=generate_keys(embedding,args.run,args.out)
    print(json.dumps({'fit':summary,'keys':files},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
