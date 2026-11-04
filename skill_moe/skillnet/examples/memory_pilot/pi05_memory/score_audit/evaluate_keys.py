#!/usr/bin/env python3
"""Common action-oracle evaluation for all cached retrieval keys."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
DEFAULT_OUT = Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/score-audit-01')


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    label = np.asarray(label, bool)
    n1 = int(label.sum()); n0 = len(label) - n1
    if not n1 or not n0:
        return float('nan')
    ranks = _rankdata(np.asarray(score, np.float64))
    return float((ranks[label].sum() - n1*(n1+1)/2) / (n1*n0))


def _episode_bootstrap(values: np.ndarray, episodes: np.ndarray, *, seed=84021, n=2000):
    if n <= 0:
        return [float('nan'), float('nan')]
    values = np.asarray(values, np.float64)
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    means = np.empty(n, np.float64)
    members = {ep: np.flatnonzero(episodes == ep) for ep in unique}
    for i in range(n):
        chosen = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([members[x] for x in chosen])
        means[i] = np.nanmean(values[ix])
    return [float(np.nanquantile(means, .025)), float(np.nanquantile(means, .975))]


def _metrics(key: dict, oracle: np.ndarray, q_rows: np.ndarray, qtask: np.ndarray,
             qep: np.ndarray, mem_rows: np.ndarray, mtask: np.ndarray, mep: np.ndarray,
             *, tau: float, bootstraps=2000):
    exclusion = str(key['exclusion'].item())
    cands = np.asarray(key['cand_rows'], np.int64)
    scores = np.asarray(key['score'], np.float32)
    action_dist = np.empty_like(scores, np.float32)
    present = np.zeros(len(q_rows), bool)
    hit = np.zeros(len(q_rows), bool)
    recall10 = np.zeros(len(q_rows), np.float32)
    recall64 = np.zeros(len(q_rows), np.float32)
    best64 = np.empty(len(q_rows), np.float32)
    median64 = np.empty(len(q_rows), np.float32)
    random64 = np.empty(len(q_rows), np.float32)
    oracle_rank = np.empty(len(q_rows), np.float32)
    qindex = {int(row): i for i, row in enumerate(q_rows)}
    mindex = {int(row): i for i, row in enumerate(mem_rows)}
    for qi, row in enumerate(np.asarray(key['query_rows'], np.int64)):
        oi = qindex[int(row)]
        cols = np.asarray([mindex[int(x)] for x in cands[qi]], np.int64)
        all_d = oracle[oi]
        eligible = (mtask != qtask[oi]) if exclusion == 'task' else (mep != qep[oi])
        valid_dist = all_d[eligible]
        present[qi] = np.any(valid_dist <= tau)
        action_dist[qi] = all_d[cols]
        best64[qi] = float(np.min(action_dist[qi]))
        median64[qi] = float(np.median(action_dist[qi]))
        random64[qi] = float(np.mean(action_dist[qi]))
        oracle_rank[qi] = float(np.argmin(action_dist[qi]) + 1)
        positives = valid_dist <= tau
        if positives.any():
            eligible_rows = mem_rows[eligible]
            posrows = set(map(int, eligible_rows[positives]))
            top = list(map(int, cands[qi]))
            hit[qi] = top[0] in posrows
            recall10[qi] = len(posrows.intersection(top[:10])) / len(posrows)
            recall64[qi] = len(posrows.intersection(top[:64])) / len(posrows)
    corr = np.asarray([_spearman(scores[i], action_dist[i]) for i in range(len(q_rows))])
    top_action = action_dist[:, 0]
    result = {
        'queries': int(len(q_rows)), 'tau': tau,
        'oracle_presence_rate': float(present.mean()),
        'top1_hit_rate_all_queries': float(hit.mean()),
        'top1_hit_rate_given_oracle_exists': float(hit[present].mean()) if present.any() else None,
        'recall_at_10_given_oracle_exists': float(recall10[present].mean()) if present.any() else None,
        'recall_at_64_given_oracle_exists': float(recall64[present].mean()) if present.any() else None,
        'top1_action_mse_mean': float(np.mean(top_action)),
        'top1_action_mse_median': float(np.median(top_action)),
        'top64_oracle_action_mse_mean': float(np.mean(best64)),
        'top64_median_action_mse_mean': float(np.mean(median64)),
        'top64_random_candidate_action_mse_mean': float(np.mean(random64)),
        'oracle_best_candidate_rank_median': float(np.median(oracle_rank)),
        'oracle_best_candidate_rank_p90': float(np.quantile(oracle_rank, .9)),
        'top1_draft_mse_mean': float(np.mean(key['dist'][:, 0])) if 'dist' in key else None,
        'within_query_spearman_mean': float(np.nanmean(corr)),
        'within_query_spearman_median': float(np.nanmedian(corr)),
        'within_query_spearman_negative_fraction': float(np.mean(corr < 0)),
        'top1_hit_auc': _auc(scores[:, 0], hit),
        'bootstrap95': {
            'oracle_presence_rate': _episode_bootstrap(present.astype(float), qep, seed=84021, n=bootstraps),
            'top1_hit_rate': _episode_bootstrap(hit.astype(float), qep, seed=84022, n=bootstraps),
            'top1_action_mse_mean': _episode_bootstrap(top_action, qep, seed=84023, n=bootstraps),
            'top64_oracle_action_mse_mean': _episode_bootstrap(best64, qep, seed=84025, n=bootstraps),
            'top64_median_action_mse_mean': _episode_bootstrap(median64, qep, seed=84026, n=bootstraps),
            'top64_random_candidate_action_mse_mean': _episode_bootstrap(random64, qep, seed=84027, n=bootstraps),
            'within_query_spearman_mean': _episode_bootstrap(corr, qep, seed=84024, n=bootstraps),
        },
    }
    return result


def evaluate(out: Path, *, bootstraps=2000):
    if (out/'evaluation.json').exists() or (out/'report.md').exists():
        raise FileExistsError(f'refusing to overwrite evaluation in {out}')
    oracle_meta = np.load(out/'oracle_meta.npz', allow_pickle=False)
    q_rows = oracle_meta['query_rows']; mem_rows = oracle_meta['memory_rows']
    oracle = np.load(out/'oracle_action_mse.npy', mmap_mode='r', allow_pickle=False)
    qtask=oracle_meta['query_task'];qep=oracle_meta['query_episode'];splits=oracle_meta['query_split']
    mtask=oracle_meta['memory_task'];mep=oracle_meta['memory_episode']
    root=Path('/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01')
    manifest=json.loads((root/'data_manifest.json').read_text())
    task_names={int(k):v for k,v in manifest['tasks'].items()}
    results=[]
    for path in sorted((out/'keys').glob('*.npz')):
        key=np.load(path, allow_pickle=False)
        if not np.array_equal(key['query_rows'],q_rows):
            raise ValueError(f'{path.name}: query row ordering differs from oracle cache')
        for tau in (.005,.01,.02):
            metrics=_metrics(key,oracle,q_rows,qtask,qep,mem_rows,mtask,mep,tau=tau,bootstraps=bootstraps)
            results.append({'method':path.stem,'exclusion':str(key['exclusion'].item()),**metrics})
    per_split=[]
    for split in ('dev','eval'):
        ix=np.flatnonzero(splits==split)
        for path in sorted((out/'keys').glob('*.npz')):
            key0=np.load(path,allow_pickle=False)
            key={k:key0[k][ix] if getattr(key0[k],'ndim',0)>0 and len(key0[k])==len(q_rows) else key0[k]
                 for k in key0.files}
            for tau in (.005,.01,.02):
                met=_metrics(key,oracle[ix],q_rows[ix],qtask[ix],qep[ix],mem_rows,mtask,mep,tau=tau,
                             bootstraps=min(bootstraps,500))
                per_split.append({'split':split,'method':path.stem,'exclusion':str(key['exclusion'].item()),**met})
    # Task-wise Top-1 hit rate and action MSE at tau=.01.
    task_rows=[]
    for path in sorted((out/'keys').glob('*.npz')):
        key=np.load(path,allow_pickle=False)
        for task in np.unique(qtask):
            ix=np.flatnonzero(qtask==task)
            subset={k:key[k][ix] if getattr(key[k],'ndim',0)>0 and len(key[k])==len(q_rows) else key[k]
                    for k in key.files}
            m=_metrics(subset,oracle[ix],q_rows[ix],qtask[ix],qep[ix],mem_rows,mtask,mep,tau=.01,bootstraps=0)
            task_rows.append({'method':path.stem,'task':int(task),'task_name':task_names.get(int(task),str(task)),
                              'n':len(ix),'oracle_presence':m['oracle_presence_rate'],
                              'top1_hit_all':m['top1_hit_rate_all_queries'],'top1_action_mse':m['top1_action_mse_mean']})
    payload={'bootstrap_unit':'query episode; 2,000 resamples for pooled results, up to 500 per split; tasks are not treated as independent windows',
             'thresholds':[.005,.01,.02],'results':results,'split_results':per_split,'task_results':task_rows}
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2,allow_nan=True)+'\n')
    lines=['# PI05 memory retrieval key audit','',
           '正解は候補実演の正規化済み active-7 action chunk とクエリ実演の行動 MSE。codec 復元誤差は検索の正解に混ぜず、副指標として別記。',
           '同一タスク除外を主条件、同一エピソード除外を副条件とした。', '',
           '## 集計表 (τ=0.01)', '',
           '| 手法 / 除外 | oracle存在 | Top1 hit (全行) | Top1 hit (存在行) | Recall@10 | Recall@64 | Top1 action MSE | draft MSE | within-query ρ | 負のρ率 | AUROC |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['tau'] != .01: continue
        fmt=lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.4f}'
        lines.append('| '+r['method']+' / '+r['exclusion']+' | '+' | '.join(fmt(r[k]) for k in
            ['oracle_presence_rate','top1_hit_rate_all_queries','top1_hit_rate_given_oracle_exists',
             'recall_at_10_given_oracle_exists','recall_at_64_given_oracle_exists','top1_action_mse_mean',
             'top1_draft_mse_mean','within_query_spearman_mean','within_query_spearman_negative_fraction','top1_hit_auc'])+' |')
    lines+=['','## Dev / Eval','', '詳細は `evaluation.json` の `split_results` に保存。係数選択・調整は行っていない。', '',
            '## Top-K 行動誤差とオラクル順位','',
            'Top-1、top64内オラクル最良、top64中央値、top64一様ランダム候補の行動MSE、およびTop-64内の最良対応の順位分布は `evaluation.json` に保存。', '',
            '## タスク別','', 'タスク別の存在率、Top1 hit、action MSE は `evaluation.json` の `task_results` にある。', '',
            '## 手順（擬似コード）','', '```text',
            'for query q in validation rows:',
            '  split[q] = run.calibration_episodes(rows) による task 内 episode 分割',
            '  for memory m in 6,583 memory rows:',
            '    oracle_mse[q,m] = mean((actions[q,:,:7] - actions[m,:,:7])^2)',
            '  mask candidates by task != task[q] (primary) or episode != episode[q] (secondary)',
            '  score all memory keys; take top 64; compare oracle membership at τ=.005/.01/.02',
            '  separately compare codec-decoded draft error; rank(score, oracle action MSE) within q',
            'bootstrap whole query episodes, never individual candidate rows', '```', '',
            '## 限界','',
            '評価器は検索候補の収録分布に条件づける。Top-1 hit は許容距離内のどれかを正解とする。',
            'LIBERO-Long のタスク対応表はこの LIBERO-40 キャッシュのタスク文に明示されないため、task_results は task id/name をそのまま列挙する。',
            '“pooled” は全有効トークン平均、token whitening はメモリの固定256行から推定した group-wise diagonal whitening。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return payload


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args();payload=evaluate(args.output,bootstraps=args.bootstraps)
    print(json.dumps({'methods':len({x['method'] for x in payload['results']}),'rows':len(payload['results'])},indent=2))


if __name__=='__main__':main()
