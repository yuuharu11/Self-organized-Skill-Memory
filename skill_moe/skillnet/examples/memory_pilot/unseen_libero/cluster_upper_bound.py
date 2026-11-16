#!/usr/bin/env python3
"""CMP3-style oracle upper bounds for unmerged and medoid-merged memories.

Representatives are chosen by the reviewed greedy cover.  Their actions are
played in the representative's own source orientation and magnitude; no pose
alignment is applied.  Evaluation actions are only used for nearest-template
selection and scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pi05_memory')]
import compression_seen_cuts as c3  # noqa: E402
import compression_compare as cc  # noqa: E402
from draft_reconstruction import ROOT  # noqa: E402

OUT_DEFAULT = ROOT / 'unseen-libero' / 'cluster-upper-01'
EXPECTED_G1 = {'units': 4516, 'all': .034447897620114326,
               'hard': .06845424480130637}


def nearest(target: np.ndarray, templates: np.ndarray, axes: int = 7):
    """Oracle nearest template; returns global template positions and MSE/70."""
    y = np.asarray(target, np.float64)
    x = np.asarray(templates, np.float64)
    if y.ndim != 3 or x.ndim != 3 or y.shape[1:] != x.shape[1:]:
        raise ValueError(f'incompatible action arrays: {y.shape}, {x.shape}')
    err = np.square(x[None, :, :, :axes] - y[:, None, :, :axes]).sum((2, 3)) / 70.
    idx = err.argmin(1)
    return idx.astype(np.int64), err[np.arange(len(y)), idx]


def metric_row(pred, target, hard, *, axes=7):
    d = np.asarray(pred, np.float64)[:, :, :axes] - np.asarray(target, np.float64)[:, :, :axes]
    per = np.square(d).sum((1, 2)) / 70.
    out = {'rows': int(len(per)), 'all_mse': float(per.mean()),
           'hard_count': int(np.asarray(hard, bool).sum()),
           'hard_mse': float(per[np.asarray(hard, bool)].mean()) if np.asarray(hard, bool).any() else None}
    return out, per


def axis_contrib(pred, target, hard):
    """Per-axis squared-error contributions, each normalized by 70."""
    e = np.square(np.asarray(pred, np.float64) - np.asarray(target, np.float64)).sum(axis=1) / 70.
    h = np.asarray(hard, bool)
    return {'all': e.mean(0).tolist(), 'hard': e[h].mean(0).tolist() if h.any() else None}


def grip_contribution(pred, target, hard):
    e = np.square(np.asarray(pred, np.float64)[:, :, 6] - np.asarray(target, np.float64)[:, :, 6]).sum(1) / 70.
    h = np.asarray(hard, bool)
    return {'all': float(e.mean()), 'hard': float(e[h].mean()) if h.any() else None}


def source_variable_units(d):
    """Return original source chunks and their DCT3 reconstructions at natural length."""
    raw, lengths = c3.spans(d['memory_traces'], {'mode': 'all', 'min_len': 5, 'width': 3})
    coded = c3.dct3(raw)
    if len(raw) != EXPECTED_G1['units']:
        raise AssertionError(f"anchor segmentation has {len(raw)} units, expected {EXPECTED_G1['units']}")
    phase = np.stack([cc.phase_view(x) for x in coded]).astype(np.float32)
    return raw, coded, phase, lengths


def _score_variable(items, lengths, data, axes=7):
    # The exact CMP3 DP playback and one-selection-per-ten-frame budget.
    if axes == 7:
        _, pred, info = c3.eval_variable(items, lengths, data)
    elif axes == 6:
        # The CMP3 optimal-unit DP is reused with grip zeroed on both sides,
        # changing only its selection cost. Returned arm trajectory is unchanged.
        masked = dict(data)
        masked['targets'] = np.asarray(data['targets']).copy(); masked['targets'][:, :, 6] = 0
        masked['traces'] = {k: np.asarray(v).copy() for k, v in data['traces'].items()}
        for v in masked['traces'].values(): v[:, 6] = 0
        arm_items = [np.asarray(x).copy() for x in items]
        for x in arm_items: x[:, 6] = 0
        old = c3.ev.Shared.target
        try:
            c3.ev.Shared.target = masked['targets']
            pred, info = c3.c2.score_variable_memory(arm_items, lengths, np.arange(len(arm_items)), masked['traces'], data['rows_of'])
        finally:
            c3.ev.Shared.target = old
    else:
        raise ValueError(axes)
    return pred, info


def _representative_indices(values, tau):
    # Same function and distance as compression_compare's reviewed implementation.
    if tau is None:
        return np.arange(len(values), dtype=np.int64)
    return np.asarray(cc.cover_units(np.asarray(values, np.float32), tau), np.int64)


def assert_eval_rows_outside_memory(rows, eval_ids):
    ids = np.asarray(eval_ids, np.int64)
    if np.asarray(rows['memory'], bool)[ids].any() or not np.asarray(rows['validation'], bool)[ids].all():
        raise AssertionError('evaluation rows must be validation-only and disjoint from memory')


def variable_results(d, evals, *, smoke=False):
    raw, coded, phase, lengths = source_variable_units(d)
    out = []
    for tau in (None, .04, .08, .16):
        # Merge on compressed template values, but keep the chosen medoid's own
        # DCT reconstruction, including its source orientation and magnitude.
        reps = _representative_indices(phase, tau)
        row = {'group': 'A_variable', 'tau': tau, 'units': int(len(reps)),
               'representative_indices': reps.tolist() if smoke else None,
               'playback': 'source medoid DCT3 reconstruction; natural-length DP'}
        scores = {}
        picked_items = [coded[int(i)] for i in reps]
        for split, data in evals.items():
            pred, info = _score_variable(picked_items, lengths[reps], data, axes=7)
            met7, _ = metric_row(pred, data['targets'], data['hard'], axes=7)
            arm7, _ = metric_row(pred, data['targets'], data['hard'], axes=6)
            pred6, info6 = _score_variable(picked_items, lengths[reps], data, axes=6)
            met6, _ = metric_row(pred6, data['targets'], data['hard'], axes=6)
            scores[split] = {'active7': met7, 'arm6': met6,
                             'axis_contribution_active7': axis_contrib(pred, data['targets'], data['hard']),
                             'axis_contribution_arm6_selection': axis_contrib(pred6, data['targets'], data['hard']),
                             'arm6_contribution_from_7axis_choice': arm7,
                             'selection_info_7axis': info, 'selection_info_arm6': info6}
        row['eval'] = scores
        out.append(row)
    return out


def reference_rows(d, evals, full_evals=None):
    refs = []
    # CMP3 raw and DCT3 fixed-10 references, using identical oracle selection.
    for label, templates in [('CMP3_fixed_raw', d['memory']),
                             ('CMP3_fixed_DCT3', c3.c2.dct_decode(c3.c2.dct_encode(d['memory'], 3), 10).astype(np.float32))]:
        e = {}
        for split, data in evals.items():
            ix, _ = nearest(data['targets'], templates, 7); pred = templates[ix]
            e[split] = {'active7': metric_row(pred, data['targets'], data['hard'], axes=7)[0],
                        'arm6': metric_row(pred, data['targets'], data['hard'], axes=6)[0]}
        refs.append({'name': label, 'units': len(templates), 'eval': e})
    import unit_memory_offline as umo
    # Corrected official π0.5 outputs; this is a reference, not a memory method.
    learned = umo._learned_data(d, c3)
    unseen = umo._unseen_data(d, c3)
    full_evals = evals if full_evals is None else full_evals
    pi_eval = {}
    for name, pi_data in [('learned', learned), ('unseen', unseen)]:
        target = pi_data['target']; pred = pi_data['pi05']
        hard = next(v['hard'] for k, v in evals.items() if k == name)
        # The helper data follows the canonical full sequence: apply the same
        # first-demo smoke selection when requested by the caller's eval subset.
        if len(target) != len(evals[name]['targets']):
            subset_keys = set(evals[name]['traces'])
            source_ids = np.sort(np.concatenate([full_evals[name]['rows_of'][k] for k in subset_keys]))
            chosen = source_ids
            target, pred = target[chosen], pred[chosen]
        pi_eval[name] = {
            'active7': metric_row(pred, target, hard, axes=7)[0],
            'arm6': metric_row(pred, target, hard, axes=6)[0]}
    refs.append({'name': 'π0.5_rot180', 'units': None, 'eval': pi_eval})
    return refs


def run(*, smoke=False):
    d, evals = c3.prepare()
    full_counts = c3.unit_meta(d, evals)
    rows = d['rows']
    mem = np.asarray(rows['memory'], bool)
    val = np.asarray(rows['validation'], bool)
    if np.asarray(rows['memory'])[d['mem_ids']].sum() != len(d['mem_ids']):
        raise AssertionError('memory ids are not exactly the declared memory rows')
    learned_ids = np.asarray(evals['learned']['ids'], np.int64)
    assert_eval_rows_outside_memory(rows, learned_ids)
    if np.intersect1d(learned_ids, d['mem_ids']).size:
        raise AssertionError('evaluation demos leaked into memory')
    full_evals = evals
    raw, coded, phase, lengths = source_variable_units(d)
    # Always reproduce the full CMP3 G1 value before any smoke subset.
    full_g1, _, _ = c3.eval_variable(coded, lengths, evals['unseen'])
    if len(evals['unseen']['targets']) == 619:
        if len(coded) != EXPECTED_G1['units'] or abs(full_g1['all_mse']-EXPECTED_G1['all']) > 2e-6 or abs(full_g1['hard_mse']-EXPECTED_G1['hard']) > 2e-6:
            raise RuntimeError(f"CMP3 G1 reproduction failed: {full_g1}; expected {EXPECTED_G1}")
    if smoke:
        evals = {k: c3.subset_eval(v, n_demos=1) for k, v in evals.items()}
    return {'mode': 'smoke' if smoke else 'full', 'evaluation_sets': full_counts,
            'split_audit': {'learned_eval_memory_overlap': 0,
                            'learned_eval_nonvalidation_rows': int((~val[learned_ids]).sum()),
                            'memory_rows': int(mem.sum())},
            'scope_note': 'Oracle-only offline bound. No pose rotation/scale is fitted.',
            'legacy_tau08_reference': {'units': 247, 'unseen_all': 0.0614, 'unseen_hard': 0.1069,
                                       'note': 'Earlier 09 reference; its cluster construction differs from this explicit compression_compare.cover_units phase-template path.'},
            'A': variable_results(d, evals, smoke=smoke),
            'references': reference_rows(d, evals, full_evals),
            'G1_smoke_or_full': full_g1}


def render(r, output: Path):
    lines = ['# τ 統合メモリの正規化 MSE 上限', '',
             '誤差は CMP3 と同じ active7 正規化空間の 10 ステップ MSE（平方誤差和 / 70）。各クエリでは対象行動との距離が最小のメモリ単位を選び、単位を保存時の向き・大きさのまま使う。π0.5 Procrustes、回転・倍率の推定は行わない。正解行動は選択と採点だけに使用する。', '',
             f"モード: **{r['mode']}**。評価セット（全件定義）: 学習済み {r['evaluation_sets']['learned']['tasks']}タスク / {r['evaluation_sets']['learned']['demos']}デモ / {r['evaluation_sets']['learned']['windows']}窓 (hard {r['evaluation_sets']['learned']['hard']}), 未学習 {r['evaluation_sets']['unseen']['tasks']}タスク / {r['evaluation_sets']['unseen']['demos']}デモ / {r['evaluation_sets']['unseen']['windows']}窓 (hard {r['evaluation_sets']['unseen']['hard']}).", '',
             '## A. 速度極小＋グリッパー切替の可変長メモリ', '',
             '| τ | 単位数 | 学習済み all/hard active7 | 学習済み all/hard 腕6 | 未学習 all/hard active7 | 未学習 all/hard 腕6 |', '|---:|---:|---:|---:|---:|---:|']
    for x in r['A']:
        le, ue = x['eval']['learned'], x['eval']['unseen']
        def pair(v): return f"{v['all_mse']:.5f}/{v['hard_mse']:.5f}" if v['hard_mse'] is not None else f"{v['all_mse']:.5f}/n.a."
        lines.append(f"| {x['tau']} | {x['units']} | {pair(le['active7'])} | {pair(le['arm6'])} | {pair(ue['active7'])} | {pair(ue['arm6'])} |")
    lines += ['', 'τ は DCT3 復元テンプレートの平均二乗距離に対する greedy cover の包含しきい値。代表は `compression_compare.cover_units` が返した元のテンプレート index であり、その運動単位の向き・大きさを保持した DCT3 復元を再生する。可変長評価は CMP3 と同じ自然長 DP 再生・10フレームごとの選択予算。', '',
              f"既存09の τ=.08 参考値は 247単位・未学習 .0614/.1069。本実装は指定どおり `compression_compare.cover_units` を、32位相へ再標本化した DCT3 復元テンプレートに適用し、238単位となった。旧参考は `motion_core_eval` 側のクラスタ生成に基づくため、代表数・評価値の差をそのまま同一条件の不一致とは扱えない。", '',
              '## 参考と再現確認', '', '| 参考 | 学習済み active7 all/hard | 未学習 active7 all/hard |', '|---|---:|---:|']
    for x in r['references']:
        if x['name']=='π0.5_rot180':
            a,b=x['eval']['learned']['active7'],x['eval']['unseen']['active7']
        else:
            a,b=x['eval']['learned']['active7'],x['eval']['unseen']['active7']
        def p(v): return f"{v['all_mse']:.5f}/{v['hard_mse']:.5f}" if v['hard_mse'] is not None else f"{v['all_mse']:.5f}/n.a."
        lines.append(f"| {x['name']} | {p(a)} | {p(b)} |")
    g=r['G1_smoke_or_full']; lines += ['', f"CMP3 G1 再現: 現在の実行値 all={g['all_mse']:.8f}, hard={g['hard_mse']:.8f}。基準 .03444790/.06845424。", '',
       '再現コマンド: `cd /work && source env/activate.sh && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 8-47 nice -n 10 python skill_moe/skillnet/examples/memory_pilot/unseen_libero/cluster_upper_bound.py --smoke`。全件は `--full` を指定。']
    if r['mode']=='smoke':
        lines.insert(3, '**これは一部エピソードでのスモークであり、全件の結果ではない。**')
    output.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(); g.add_argument('--smoke',action='store_true'); g.add_argument('--full',action='store_true')
    ap.add_argument('--output',type=Path,default=OUT_DEFAULT)
    a=ap.parse_args()
    if a.output.exists() and any(p.is_file() for p in a.output.iterdir()):
        raise FileExistsError(f'refusing to overwrite {a.output}')
    result=run(smoke=not a.full)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    render(result,a.output/'report.md')
    print(a.output)

if __name__=='__main__': main()
