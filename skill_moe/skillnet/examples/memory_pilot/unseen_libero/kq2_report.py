#!/usr/bin/env python3
"""KQ2 aggregation: seed means, paired episode bootstrap vs C1b (KQ-final 3 seeds), subsets,
VLM-attention diagnostics, layer tables.  Writes kq2-01/results.json and kq2-01/tables.md."""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kq2_data as kd  # noqa: E402

OUT = kd.OUT
EVAL = ('dev2', 'learned', 'unseen20')
SUBS = ('first_unit', 'post_switch', 'dir_change', 'middle')


def episodes(n: str) -> np.ndarray:
    d = OUT / 'data' / n
    return np.load(d / 'task.npy').astype(np.int64) * 10_000_000 + np.load(d / 'episode.npy').astype(np.int64)


def boot(a, b, ep, n=2000, seed=0):
    import vq_geometry as vg
    return vg.paired_bootstrap(a, b, ep, n=n, seed=seed)


def load_rows(sub: str) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(OUT / sub / '*.json'))):
        r = json.loads(Path(f).read_text())
        if 'evaluation' not in r:
            continue
        g.setdefault(re.sub(r'_s\d+$', '', r['name']), []).append(r)
    return g


def agg(rs: list[dict], ref: dict[str, np.ndarray] | None, eps: dict[str, np.ndarray]) -> dict:
    e = {'seeds': len(rs), 'epochs': [r.get('selected_epoch') for r in rs], 'params': rs[0].get('params')}
    for n in EVAL:
        per = np.stack([np.asarray(r['evaluation'][n]['per_query_top1']) for r in rs])
        v = per.mean(1)
        blk = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None, 'seeds': v.tolist()}
        subs = rs[0]['evaluation'][n].get('subsets', {})
        if subs:
            import kq2_run as R  # noqa
        blk['subsets'] = {}
        for s in SUBS:
            m = SUBMASK[n][s]
            blk['subsets'][s] = {'mean': float(per[:, m].mean()), 'rows': int(m.sum())}
        if ref is not None:
            blk['diff_vs_c1b'] = boot(per.mean(0), ref[n], eps[n])
            blk['diff_vs_c1b_subsets'] = {s: boot(per.mean(0)[SUBMASK[n][s]], ref[n][SUBMASK[n][s]], eps[n][SUBMASK[n][s]])
                                          for s in SUBS}
        diag = [r['evaluation'][n].get('diag') for r in rs if r['evaluation'][n].get('diag')]
        if diag:
            blk['diag'] = {k: float(np.mean([d[k] for d in diag])) for k in diag[0]}
        e[n] = blk
    e['dev_selected'] = [r.get('dev_selected') for r in rs]
    return e


SUBMASK: dict = {}


def main() -> dict:
    import kq2_run as R
    ctx_sub = {}
    for n in EVAL:
        d = OUT / 'data' / n
        sp = {k: np.load(d / f'{k}.npy') for k in ('key', 'history', 'history_mask', 'actions')}
        ctx_sub[n] = R.subsets_for(n, sp)
    SUBMASK.update(ctx_sub)
    eps = {n: episodes(n) for n in EVAL}
    base = json.loads((OUT / 'baselines.json').read_text())
    ref = {n: np.mean([np.asarray(base[f'C1b_s{i}'][n]['per_query_top1']) for i in (1, 2, 3)], 0) for n in EVAL}
    res = {'baselines': {}, 'stage1_keys': {}, 'stage1_keyq': {}, 'stage2': {}}
    for b in ('hold_last', 'oracle'):
        rs = [{'name': b, 'evaluation': base[b]}]
        res['baselines'][b] = agg(rs, ref, eps)
    res['baselines']['C1b'] = agg([{'name': f'C1b_s{i}', 'evaluation': base[f'C1b_s{i}']} for i in (1, 2, 3)], None, eps)
    ref_keys = json.loads((OUT / 'stage1' / 'reference.json').read_text())
    res['stage1_reference'] = ref_keys
    for f in sorted(glob.glob(str(OUT / 'stage1/keys/*.json'))):
        r = json.loads(Path(f).read_text())
        res['stage1_keys'][r['name']] = {k: r[k] for k in ('cfg', 'params', 'selected_epoch', 'a', 'diagnostics')}
        res['stage1_keys'][r['name']]['curve'] = [{'epoch': h['epoch'], 'loss': h['loss'], **({'a': h['a']} if 'a' in h else {})}
                                                  for h in r['history'] if 'a' in h or h['epoch'] in (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300)]
    for g, rs in load_rows('stage1/keyq').items():
        res['stage1_keyq'][g] = agg(rs, ref, eps)
    for g, rs in load_rows('stage2/rows').items():
        res['stage2'][g] = agg(rs, ref, eps)
    p = OUT / 'stage2' / 'vlm_shuffle.json'
    if p.exists():
        res['vlm_shuffle'] = json.loads(p.read_text())
    kd.write_json(OUT / 'results.json', res)
    (OUT / 'tables.md').write_text(render(res))
    return res


def f4(x):
    return '—' if x is None else f'{x:.4f}'.replace('0.', '.', 1) if x >= 0 else f'{x:.4f}'.replace('-0.', '−.', 1)


def pm(b):
    return f4(b['mean']) + (f' ± {f4(b["sd"])}' if b.get('sd') is not None else '')


def ci(t):
    s = lambda x: (f'{x:+.4f}').replace('+0.', '+.').replace('-0.', '−.')
    return f'{s(t[0])} [{s(t[1])}, {s(t[2])}]'


def render(res: dict) -> str:
    L = []
    L += ['## 第1段 (a) キー空間の最近傍（真の次10歩をキー化して最近傍、自 episode 除外）', '',
          '|キー|パラメータ|選んだ epoch|Dev v2|学習済み|未学習20Hz|Spearman|cos 平均|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    rk = res['stage1_reference']
    L.append(f"|K-x1（既存 MK1b をそのまま）|{rk['K-x1']['params']:,}|—|{f4(rk['K-x1']['a']['dev2'])}|{f4(rk['K-x1']['a']['learned'])}|{f4(rk['K-x1']['a']['unseen20'])}|{rk['K-x1']['diagnostics']['spearman_vs_neg_mse']:.3f}|{rk['K-x1']['diagnostics']['cos_mean']:.3f}|")
    for n, r in res['stage1_keys'].items():
        L.append(f"|{n}|{r['params']:,}|{r['selected_epoch']}|{f4(r['a']['dev2'])}|{f4(r['a']['learned'])}|{f4(r['a']['unseen20'])}|{r['diagnostics']['spearman_vs_neg_mse']:.3f}|{r['diagnostics']['cos_mean']:.3f}|")
    L.append(f"|DCT3（学習なし、窓どうし）|0|—|{f4(rk['DCT3']['a']['dev2'])}|{f4(rk['DCT3']['a']['learned'])}|{f4(rk['DCT3']['a']['unseen20'])}|—|—|")
    L.append(f"|oracle|—|—|{f4(rk['oracle']['dev2'])}|{f4(rk['oracle']['learned'])}|{f4(rk['oracle']['unseen20'])}|—|—|")
    L += ['', '### 学習曲線（(a) Dev v2 / 未学習20Hz、損失）', '', '|キー|' + '|'.join(f'ep{e}' for e in (4, 10, 25, 50, 100, 200, 300)) + '|',
          '|---|' + '---:|' * 7]
    for n, r in res['stage1_keys'].items():
        cells = {c['epoch']: c for c in r['curve'] if 'a' in c}
        L.append(f'|{n}|' + '|'.join(f"{f4(cells[e]['a']['dev2'])}/{f4(cells[e]['a']['unseen20'])} ({cells[e]['loss']:.3f})" if e in cells else '—'
                                       for e in (4, 10, 25, 50, 100, 200, 300)) + '|')
    L += ['', '## 第1段 (b) C1b クエリを各キーで学習（Dev v2 で epoch 選択、top-1 腕 MSE）', '',
          '|キー|seed|Dev v2|学習済み|未学習20Hz|対 C1b 未学習20Hz|対 C1b 学習済み|', '|---|---:|---:|---:|---:|---:|---:|']
    for n, r in sorted(res['stage1_keyq'].items(), key=lambda x: x[1]['dev2']['mean']):
        L.append(f"|{n}|{r['seeds']}|{pm(r['dev2'])}|{pm(r['learned'])}|{pm(r['unseen20'])}|{ci(r['unseen20']['diff_vs_c1b'])}|{ci(r['learned']['diff_vs_c1b'])}|")
    L += ['', '## 第2段 クエリ（Dev v2 で選択）', '',
          '|行|パラメータ|seed|epoch|Dev v2|学習済み|未学習20Hz|対 C1b 未学習20Hz|', '|---|---:|---:|---|---:|---:|---:|---:|']
    for n, r in sorted(res['stage2'].items(), key=lambda x: x[1]['dev2']['mean']):
        L.append(f"|{n}|{r['params']:,}|{r['seeds']}|{r['epochs']}|{pm(r['dev2'])}|{pm(r['learned'])}|{pm(r['unseen20'])}|{ci(r['unseen20']['diff_vs_c1b'])}|")
    L += ['', '## 部分集合（未学習20Hz）', '', '|行|最初の単位|切替直後|向き変化|途中|', '|---|---:|---:|---:|---:|']
    allrows = {**{f'基準:{k}': v for k, v in res['baselines'].items()}, **{f'K:{k}': v for k, v in res['stage1_keyq'].items()},
               **res['stage2']}
    for n, r in allrows.items():
        s = r['unseen20']['subsets']
        L.append(f"|{n}|" + '|'.join(f"{f4(s[k]['mean'])}" for k in SUBS) + '|')
    n0 = next(iter(allrows.values()))['unseen20']['subsets']
    L.append('|行数|' + '|'.join(str(n0[k]['rows']) for k in SUBS) + '|')
    L += ['', '## VLM 注意の診断（未学習20Hz / 学習済み、seed 平均）', '',
          '|行|エントロピー（頭平均、/log34）|‖Attn‖/‖h‖ 層1|層2|重み 画像base/手首/言語|‖h‖|', '|---|---:|---:|---:|---|---:|']
    for n, r in res['stage2'].items():
        for sp in ('unseen20', 'learned'):
            d = r[sp].get('diag')
            if not d or 'l0_entropy_headmean' not in d:
                continue
            L.append(f"|{n} ({sp})|{d['l0_entropy_headmean']:.3f}|{d['l0_attn_norm_ratio']:.3f}|{d.get('l1_attn_norm_ratio', float('nan')):.3f}|"
                     f"{d['l0_weight_image_base']:.2f}/{d['l0_weight_image_wrist']:.2f}/{d['l0_weight_text']:.2f}|{d['h_norm']:.2f}|")
    if 'vlm_shuffle' in res:
        L += ['', '## VLM を行間で入れ替えた評価（学習済みモデル、評価時のみ）', '', '|行|分割|そのまま|入れ替え|top-1 が変わった割合|', '|---|---|---:|---:|---:|']
        for n, r in res['vlm_shuffle'].items():
            for sp, v in r.items():
                L.append(f"|{n}|{sp}|{f4(v['intact'])}|{f4(v['vlm_shuffled'])}|{v['top1_changed_fraction']:.2f}|")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
    print((OUT / 'tables.md').read_text()[:6000])
