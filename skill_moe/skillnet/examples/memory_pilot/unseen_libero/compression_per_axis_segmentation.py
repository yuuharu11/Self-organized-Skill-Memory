#!/usr/bin/env python3
"""DIM3: error distributions and memory diagnostics by segmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compression_seen_cuts as cmp3  # noqa: E402
import compression_matched as cmp2  # noqa: E402

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
CMP3_PROD = ROOT / 'compress-seen-cuts-01' / 'production' / 'results.json'
DIM1 = ROOT / 'compress-per-axis-01' / 'results.json'
OUTPUT = ROOT / 'compress-per-axis-01' / 'segmentation' / 'attempt-01'
AXES7 = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper')
AXES6 = AXES7[:6]
MASK_NAMES = ('all', 'raw_memory_q80', 'gripper_switch', 'speed_change_top20')
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261118
TOL = 2e-7


def per_row_errors(pred, target):
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    delta = pred - target
    return {
        'active7': np.mean(delta ** 2, axis=(1, 2)),
        # Keep active7 denominator, matching DIM1 and GRIP1's arm contribution.
        'arm6': np.sum(delta[:, :, :6] ** 2, axis=(1, 2)) / 70.,
    }


def distribution(values):
    v = np.asarray(values, np.float64)
    if not len(v):
        return {'n': 0, 'mean': None, 'std': None, 'median': None, 'q25': None, 'q75': None, 'p90': None}
    return {'n': int(len(v)), 'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'median': float(np.median(v)), 'q25': float(np.quantile(v, .25)),
            'q75': float(np.quantile(v, .75)), 'p90': float(np.quantile(v, .90))}


def _episode_index(data):
    groups = np.full(len(data['targets']), -1, np.int64)
    keys = sorted(data['traces'])
    for gid, key in enumerate(keys):
        groups[np.asarray(data['rows_of'][key], np.int64)] = gid
    if np.any(groups < 0):
        raise AssertionError('some evaluation rows have no episode key')
    return groups, [tuple(map(int, x)) for x in keys]


def paired_episode_bootstrap(delta, groups, mask, seed):
    delta, groups, mask = np.asarray(delta, np.float64), np.asarray(groups), np.asarray(mask, bool)
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {'n': 0, 'episodes': 0, 'mean': None, 'ci95': [None, None]}
    unique = np.unique(groups[ids])
    point = float(delta[ids].mean())
    rng = np.random.default_rng(seed)
    group_sum = np.asarray([delta[ids[groups[ids] == g]].sum() for g in unique], np.float64)
    group_count = np.asarray([np.sum(groups[ids] == g) for g in unique], np.int64)
    draws_idx = rng.integers(0, len(unique), size=(BOOTSTRAP_REPS, len(unique)))
    draws = group_sum[draws_idx].sum(axis=1) / group_count[draws_idx].sum(axis=1)
    return {'n': int(len(ids)), 'episodes': int(len(unique)), 'mean': point,
            'ci95': [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def _best_sequence_prediction(data, items, lengths, dims):
    import optimal_units_eval as opt
    grouped = {}
    for item, length in zip(items, lengths, strict=True):
        grouped.setdefault(int(length), []).append(np.asarray(item, np.float64))
    drafts = {length: np.stack(units) for length, units in grouped.items()}
    lengths_sorted = sorted(drafts)
    pred = np.empty_like(data['targets'], dtype=np.float64)
    for key, trace in data['traces'].items():
        table = {}
        for start in range(len(trace)):
            for length, candidates in drafts.items():
                covered = min(length, len(trace) - start)
                err = np.sum((candidates[:, :covered, :dims] -
                              trace[None, start:start + covered, :dims]) ** 2, axis=(1, 2))
                ix = int(np.argmin(err))
                table[start, length] = float(err[ix]), candidates[ix, :covered]
        budget = max(1, int(np.ceil(len(trace) / 10)))
        _, sequence = opt.best_sequence(table, lengths_sorted, len(trace), budget)
        if not sequence:
            raise AssertionError(f'no feasible DP sequence for {key}')
        joined = np.empty_like(trace, dtype=np.float64)
        for start, length in sequence:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[np.asarray(data['rows_of'][key])] = joined.reshape(-1, 10, 7)
    return pred


def _fixed_memory_units(traces, length):
    units = []
    for trace in traces.values():
        for start in range(0, len(trace) - length + 1, length):
            units.append(np.asarray(trace[start:start + length, :7], np.float32))
    return units


def candidate_pools(d):
    mem10 = np.asarray(d['memory'], np.float32)
    mem15 = _fixed_memory_units(d['memory_traces'], 15)
    mem_g1, g1_lengths = cmp3.spans(d['memory_traces'], {'mode': 'all', 'min_len': 5, 'width': 3})
    pools = {
        'fixed10_raw': (list(mem10), np.full(len(mem10), 10, np.int64), 'rowwise'),
        'fixed10_dct3': (cmp3.dct3(list(mem10)), np.full(len(mem10), 10, np.int64), 'rowwise'),
        'fixed15_raw': (mem15, np.full(len(mem15), 15, np.int64), 'sequence'),
        'fixed15_dct3': (cmp3.dct3(mem15), np.full(len(mem15), 15, np.int64), 'sequence'),
        'G1_variable_dct3': (cmp3.dct3(mem_g1), g1_lengths, 'sequence'),
    }
    return pools


def predictions_for_split(data, pools):
    outputs, indexes = {}, {}
    for method, (items, lengths, kind) in pools.items():
        outputs[method], indexes[method] = {}, {}
        pool = np.stack(items).astype(np.float32) if kind == 'rowwise' else None
        for split, eval_data in data.items():
            if kind == 'rowwise':
                selected7, _ = cmp2.nearest_templates(eval_data['targets'], pool)
                selected6 = np.empty(len(eval_data['targets']), np.int64)
                for i, target in enumerate(eval_data['targets']):
                    err = np.mean((pool[:, :, :6] - target[None, :, :6]) ** 2, axis=(1, 2))
                    selected6[i] = int(np.argmin(err))
                outputs[method][split] = {
                    'seven_axis': pool[selected7],
                    'arm6': pool[selected6],
                }
                indexes[method][split] = {
                    'seven_axis_sha256': hashlib.sha256(selected7.astype(np.int64).tobytes()).hexdigest(),
                    'arm6_sha256': hashlib.sha256(selected6.astype(np.int64).tobytes()).hexdigest(),
                }
            else:
                outputs[method][split] = {
                    'seven_axis': _best_sequence_prediction(eval_data, items, lengths, 7),
                    'arm6': _best_sequence_prediction(eval_data, items, lengths, 6),
                }
                indexes[method][split] = {'selection': 'DP, at most one new segment per 10 frames'}
    return outputs, indexes


def evaluation_masks(data):
    masks = {}
    for split, q in data.items():
        grip = np.sign(q['targets'][:, 1:, 6]) != np.sign(q['targets'][:, :-1, 6])
        switch = np.any(grip, axis=1)
        physical = cmp3.motion.vs.to_physical(q['targets'][:, :, :6])
        speed = np.linalg.norm(physical[:, :, :3], axis=2)
        speed_variation = np.sum(np.abs(np.diff(speed, axis=1)), axis=1)
        n_top = max(1, int(np.ceil(.2 * len(speed_variation))))
        order = np.argsort(speed_variation, kind='stable')
        speed_top = np.zeros(len(speed_variation), bool)
        speed_top[order[-n_top:]] = True
        masks[split] = {
            'all': np.ones(len(q['targets']), bool),
            'raw_memory_q80': np.asarray(q['hard'], bool),
            'gripper_switch': switch,
            'speed_change_top20': speed_top,
        }
    return masks


def memory_diagnostics(d):
    sets = {
        'fixed10': _fixed_memory_units(d['memory_traces'], 10),
        'fixed15': _fixed_memory_units(d['memory_traces'], 15),
        'G1_variable': cmp3.spans(d['memory_traces'], {'mode': 'all', 'min_len': 5, 'width': 3})[0],
    }
    n_equal = min(len(items) for items in sets.values())
    sampled = {}
    rng = np.random.default_rng(cmp2.SEED)
    for name, items in sets.items():
        idx = np.sort(rng.choice(len(items), n_equal, replace=False))
        selected = [items[int(i)] for i in idx]
        phase = np.stack([cmp2.phase_resample(x, 32) for x in selected])
        sampled[name] = {
            'units': len(items), 'equalized_units': n_equal,
            'low_frequency': cmp2.low_frequency_concentration(items),
            'between_within': {str(k): cmp2.between_within_ratio(phase, k, seed=cmp2.SEED) for k in (32, 95, 256)},
        }
    return sampled


def build_result():
    d, evals = cmp3.prepare()
    production = json.loads(CMP3_PROD.read_text())
    dim1 = json.loads(DIM1.read_text())
    counts = cmp3.unit_meta(d, evals)
    if counts != production['evaluation_sets'] or counts != dim1['evaluation_sets']:
        raise AssertionError(f'evaluation rows changed: {counts}')
    pools = candidate_pools(d)
    predictions, selection_hashes = predictions_for_split(evals, pools)
    masks = evaluation_masks(evals)
    condition_order = list(pools)
    method_metrics, row_errors, episodes = {}, {}, {}
    for method in condition_order:
        method_metrics[method], row_errors[method], episodes[method] = {}, {}, {}
        for split, data in evals.items():
            method_metrics[method][split], row_errors[method][split], episodes[method][split] = {}, {}, {}
            groups, group_keys = _episode_index(data)
            for choice in ('seven_axis', 'arm6'):
                pred = predictions[method][split][choice]
                errs = per_row_errors(pred, data['targets'])
                row_errors[method][split][choice] = errs
                method_metrics[method][split][choice] = {
                    mask: {metric: distribution(errs[metric][masks[split][mask]])
                           for metric in ('active7', 'arm6')}
                    for mask in MASK_NAMES
                }
                episodes[method][split][choice] = {'ids': groups, 'keys': group_keys}

    paired = {}
    comparisons = (
        ('G1_variable_dct3', 'fixed10_raw'), ('G1_variable_dct3', 'fixed10_dct3'),
        ('G1_variable_dct3', 'fixed15_raw'), ('G1_variable_dct3', 'fixed15_dct3'),
    )
    for split, data in evals.items():
        paired[split] = {}
        group_ids, _ = _episode_index(data)
        paired[split] = {mask: {} for mask in MASK_NAMES}
        for mask_name, mask in masks[split].items():
            for choice in ('seven_axis', 'arm6'):
                for var, fixed in comparisons:
                    key = f'{var}_minus_{fixed}_{choice}'
                    paired[split][mask_name][key] = {}
                    for metric in ('active7', 'arm6'):
                        delta = row_errors[var][split][choice][metric] - row_errors[fixed][split][choice][metric]
                        seed = BOOTSTRAP_SEED + (0 if split == 'learned' else 1000) + MASK_NAMES.index(mask_name) * 20 + (0 if choice == 'seven_axis' else 10)
                        paired[split][mask_name][key][metric] = paired_episode_bootstrap(delta, group_ids, mask, seed)

    # The ordinary population totals must reproduce CMP3/DIM1 aggregates.
    parity = []
    ref_map = {
        'fixed10_raw': ('fixed_raw70', 'raw', 70),
        'fixed10_dct3': ('fixed_dct21', 'dct', 21),
        'G1_variable_dct3': ('variable_dct22', 'anchor_variable_dct', 21),
    }
    for method, (dim_method, prod_method, v) in ref_map.items():
        prod_rows = [x for x in production['E1'] if x['method'] == prod_method and x['V'] == v]
        if len(prod_rows) != 1:
            raise AssertionError(f'missing CMP3 row {prod_method}, V={v}')
        for split in ('learned', 'unseen'):
            for choice in ('seven_axis', 'arm6'):
                got = method_metrics[method][split][choice]['all']['active7']['mean']
                hard_got = method_metrics[method][split][choice]['raw_memory_q80']['active7']['mean']
                if choice == 'seven_axis':
                    ref = prod_rows[0]['eval'][split]
                    da, dh = got - ref['all_mse'], hard_got - ref['hard_mse']
                    parity.append({'method': method, 'split': split, 'choice': choice,
                                   'all_delta': da, 'hard_delta': dh, 'match': abs(da) <= TOL and abs(dh) <= TOL})
                    if abs(da) > TOL or abs(dh) > TOL:
                        raise AssertionError(f'CMP3 E1 parity failed {method}/{split}: {da}, {dh}')
    fixedmean = next(x for x in production['E2']['fixed_references'] if x['condition'] == 'fixedmean_G1_current')
    for split in ('learned', 'unseen'):
        got = method_metrics['fixed15_dct3'][split]['seven_axis']
        da = got['all']['active7']['mean'] - fixedmean['upper_error'][split]['all_mse']
        dh = got['raw_memory_q80']['active7']['mean'] - fixedmean['upper_error'][split]['hard_mse']
        parity.append({'method':'fixed15_dct3','split':split,'choice':'seven_axis',
                       'all_delta':da,'hard_delta':dh,'match':abs(da)<=TOL and abs(dh)<=TOL})
        if abs(da)>TOL or abs(dh)>TOL:
            raise AssertionError(f'fixedmean_G1_current parity failed {split}: {da}, {dh}')

    # Arm6 sums from the seven-axis candidate choices must equal the DIM1 E1 decomposition.
    dim1_parity = []
    for method, dim1_name in (
        ('fixed10_raw','fixed_raw70'), ('fixed10_dct3','fixed_dct21'),
        ('G1_variable_dct3','variable_dct22'),
    ):
        for split in ('learned','unseen'):
            for mask_name, subset, key in (('all','all','arm_all'), ('raw_memory_q80','hard','arm_hard')):
                got_axes = method_metrics[method][split]['seven_axis'][mask_name]['arm6']
                refm = dim1['retrieval'][dim1_name][split]
                total = sum(refm['axis_all'][a] for a in AXES7[:6]) if subset == 'all' else sum(refm['axis_hard'][a] for a in AXES7[:6])
                delta = got_axes['mean'] - total
                dim1_parity.append({'method':method,'split':split,'subset':subset,'delta':delta,'match':abs(delta)<=TOL})
                if abs(delta)>TOL:
                    raise AssertionError(f'DIM1 arm contribution parity failed {method}/{split}/{subset}: {delta}')

    memory = memory_diagnostics(d)
    result = {
        'metric': {'active7':'mean squared error over 10x7','arm6':'sum of arm squared errors divided by 10x7 (same active7 denominator)',
                   'selection':'top-1/DP oracle selection; separate selection by seven axes and axes 0..5',
                   'speed_change':'top 20 percent of sum of absolute frame-to-frame differences in physical translational speed magnitude',
                   'gripper_switch':'any sign transition within the target 10-step action chunk'},
        'evaluation_sets':counts,'condition_order':condition_order,
        'conditions':{'fixed10_raw':{'units':6583,'length':10},'fixed10_dct3':{'units':6583,'length':10},
                      'fixed15_raw':{'units':4263,'length':15},'fixed15_dct3':{'units':4263,'length':15},
                      'G1_variable_dct3':{'units':4516,'segmentation':'speed minima + gripper switches, min length 5'}},
        'masks':{s:{k:int(v.sum()) for k,v in m.items()} for s,m in masks.items()},
        'distributions':method_metrics,'paired_differences':paired,
        'memory_diagnostics':memory,'selection_hashes':selection_hashes,
        'parity':{'cmp3':parity,'dim1_arm6':dim1_parity,
                  'all_cmp3_match':all(x['match'] for x in parity),
                  'all_dim1_arm_match':all(x['match'] for x in dim1_parity)},
        'bootstrap':{'unit':'episode', 'replicates':BOOTSTRAP_REPS, 'base_seed':BOOTSTRAP_SEED},
    }
    return result


def render(result):
    lines = [
        '# DIM3：区間分割ごとの誤差分布', '',
        'E1 と同じ評価窓・候補・選択規則。7軸選択と腕6軸選択を個別に実行。active7 は10×7要素のMSE、arm6は腕6軸二乗和/(10×7)。', '',
    ]
    for split in ('learned','unseen'):
        c=result['evaluation_sets'][split]
        lines.append(f"評価 {split}: {c['tasks']}タスク、{c['demos']}デモ、{c['windows']}窓、既存hard {c['hard']}。各方法で難しい行の作り方を変えず、固定メモリraw上位20%、gripper切替窓、速度変化上位20%も別々に集計。\n")
        for mask in MASK_NAMES:
            lines += [f'## {split}・{mask}（n={result["masks"][split][mask]}）','']
            for metric in ('active7','arm6'):
                lines += [f'### {metric}','',
                          '| 条件 | 選択 | 平均 | 標準偏差 | 中央値 | Q25 | Q75 | P90 |',
                          '|---|---|---:|---:|---:|---:|---:|---:|']
                for method in result['condition_order']:
                    for choice,label in (('seven_axis','7軸選択'),('arm6','腕6軸選択')):
                        s=result['distributions'][method][split][choice][mask][metric]
                        lines.append(f"| {method} | {label} | {s['mean']:.6f} | {s['std']:.6f} | {s['median']:.6f} | {s['q25']:.6f} | {s['q75']:.6f} | {s['p90']:.6f} |")
                lines.append('')
        for mask in MASK_NAMES:
            lines += [f'## {split}・対応差（{mask}）：G1 可変長 − 固定窓','',
                      'episode bootstrap 5,000回、差は可変長側−固定側。','',
                      '| 比較 | 選択 | active7 平均差 [95%区間] | arm6 平均差 [95%区間] |',
                      '|---|---|---:|---:|']
            for choice,label in (('seven_axis','7軸選択'),('arm6','腕6軸選択')):
                for fixed in ('fixed10_raw','fixed10_dct3','fixed15_raw','fixed15_dct3'):
                    key=f'G1_variable_dct3_minus_{fixed}_{choice}'
                    vals=result['paired_differences'][split][mask][key]
                    a=vals['active7']; b=vals['arm6']
                    lines.append(f"| G1 − {fixed} | {label} | {a['mean']:.6f} [{a['ci95'][0]:.6f},{a['ci95'][1]:.6f}] | {b['mean']:.6f} [{b['ci95'][0]:.6f},{b['ci95'][1]:.6f}] |")
            lines.append('')
    lines += ['## メモリだけの分散診断','',
              '低周波は各単位で平均を引いた各軸のDCTエネルギー中、先頭3係数の割合。between/within は32位相へ再標本化した同数単位で計算。','',
              '| 分割 | 単位数 | 等化数 | 低周波平均 [Q25,中央値,Q75] | between/within k32 | k95 | k256 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in result['memory_diagnostics'].items():
        low=m['low_frequency']; ratios=m['between_within']
        lines.append(f"| {name} | {m['units']} | {m['equalized_units']} | {low['mean']:.4f} [{low['q25']:.4f},{low['median']:.4f},{low['q75']:.4f}] | {ratios['32']['ratio']:.3f} | {ratios['95']['ratio']:.3f} | {ratios['256']['ratio']:.3f} |")
    lines += ['', '## 整合性', '',
              f"CMP3 E1／fixedmean production 値との合計再現：{result['parity']['all_cmp3_match']}。DIM1 の既存7軸選択から抽出した腕6軸値との一致：{result['parity']['all_dim1_arm_match']}。結果詳細は results.json。", '',
              '元の compress-per-axis-01/report.md は変更せず、この追補は segmentation/attempt-01/report.md に保存。', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    result=build_result()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (args.output/'report.md').write_text(render(result))
    print(json.dumps({'output':str(args.output),'counts':result['evaluation_sets'],
                      'parity':result['parity']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
