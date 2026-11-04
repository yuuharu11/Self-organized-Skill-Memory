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
