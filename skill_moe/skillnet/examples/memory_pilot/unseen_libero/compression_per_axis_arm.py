#!/usr/bin/env python3
"""CMP3 E1 arm-only candidate selection and scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compression_seen_cuts as cmp3  # noqa: E402
import compression_matched as cmp2  # noqa: E402

ROOT = Path('/work/.skillnet-state/outputs/memory-pilot/unseen-libero')
PARENT = ROOT / 'compress-per-axis-01'
DIM1_RESULTS = PARENT / 'results.json'
GRIP1_RESULTS = ROOT / 'gripper-split-01' / 'production-20261114' / 'results.json'
OUTPUT = PARENT / 'arm_only' / 'attempt-02'
AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz')
TOL = 1e-6


def contributions(pred, target, hard):
    pred, target, hard = np.asarray(pred, np.float64), np.asarray(target, np.float64), np.asarray(hard, bool)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (10, 7):
        raise ValueError('expected matching (N,10,7) predictions and targets')
    # Per-axis arm contribution keeps the active7 denominator (10*7).
    def one(p, t):
        return np.mean(np.sum((p[..., :6] - t[..., :6]) ** 2, axis=1) / 70., axis=0)
    all_axes, hard_axes = one(pred, target), one(pred[hard], target[hard])
    all_sum, hard_sum = float(all_axes.sum()), float(hard_axes.sum())
    if not np.isclose(all_sum, np.mean((pred[..., :6] - target[..., :6]) ** 2) * 6 / 7, atol=1e-12):
        raise AssertionError('arm axis contributions do not preserve active7 denominator')
    return {'n': len(target), 'hard_n': int(hard.sum()),
            'axis_all': dict(zip(AXES, all_axes.astype(float).tolist())), 'arm_all': all_sum,
            'axis_hard': dict(zip(AXES, hard_axes.astype(float).tolist())), 'arm_hard': hard_sum}


def arm_projection(metric):
    return {'n': metric['n'], 'hard_n': metric['hard_n'],
            'axis_all': {a: metric['axis_all'][a] for a in AXES},
            'arm_all': sum(metric['axis_all'][a] for a in AXES),
            'axis_hard': {a: metric['axis_hard'][a] for a in AXES},
            'arm_hard': sum(metric['axis_hard'][a] for a in AXES)}


def select_arm(pool, target):
    """Top-1 unit by arm-only MSE, ignoring gripper."""
    pool, target = np.asarray(pool, np.float32), np.asarray(target, np.float32)
    idx = np.empty(len(target), np.int64)
    for q in range(len(target)):
        err = np.mean((pool[:, :, :6] - target[q:q+1, :, :6]) ** 2, axis=(1, 2))
        idx[q] = np.argmin(err)
    return idx


def fixed_arm_predictions(d, evals):
    mem, fit = d['memory'], d['train']
    mean, basis = cmp2.pca_fit(fit, 21)
    pools = {
        'W_raw': mem,
        'W_DCT3': cmp2.dct_decode(cmp2.dct_encode(mem, 3), 10).astype(np.float32),
        'W_PCA21': cmp2.pca_restore(cmp2.pca_project(mem, mean, basis), mean, basis).astype(np.float32),
        'W_keyframe3': cmp2.keyframe_decode(cmp2.keyframe_encode(mem, 3), 10).astype(np.float32),
    }
    preds, meta = {}, {}
    for name, pool in pools.items():
        preds[name], meta[name] = {}, {}
        for split, data in evals.items():
            ix = select_arm(pool, data['targets'])
            preds[name][split] = pool[ix]
            meta[name][split] = {'selected_pool_index_sha256': __import__('hashlib').sha256(ix.tobytes()).hexdigest()}
    return preds, meta


def variable_pools(d):
    cfg = {'mode': 'all', 'min_len': 5, 'width': 3}
    memory_units, lengths = cmp3.spans(d['memory_traces'], cfg)
    train_units, _ = cmp3.spans(d['train_traces'], cfg)
    pools = {
        'U_DCT3': cmp2.compressed_variable(memory_units, 'dct', 3),
    }
    train_phase = np.stack([cmp2.phase_resample(x, 32) for x in train_units])
    mean, basis = cmp2.pca_fit(train_phase, 21)
    pools['U_PCA21'] = cmp2.compressed_variable(memory_units, 'pca', 21, mean, basis)
    return pools, lengths


def arm_best_cost_table(trace, drafts):
    """DP edge costs use only the 6 arm coordinates; output draft retains grip."""
    table = {}
    n = len(trace)
    for start in range(n):
        for length, candidates in drafts.items():
            covered = min(length, n - start)
            err = np.sum((candidates[:, :covered, :6] - trace[None, start:start+covered, :6]) ** 2, axis=(1, 2))
            best = int(np.argmin(err))
            table[start, length] = float(err[best]), candidates[best, :covered]
    return table


def variable_arm_prediction(items, lengths, data):
    import optimal_units_eval as opt
    groups = {}
    for i, length in enumerate(lengths):
        groups.setdefault(int(length), []).append(np.asarray(items[i], np.float64))
    drafts = {length: np.stack(values) for length, values in groups.items()}
    draft_lengths = sorted(drafts)
    pred = np.empty_like(data['targets'], dtype=np.float64)
    selections = frames = 0
    for key, trace in data['traces'].items():
        table = arm_best_cost_table(trace, drafts)
        budget = max(1, int(np.ceil(len(trace) / 10)))
        _, sequence = opt.best_sequence(table, draft_lengths, len(trace), budget)
        if not sequence:
            raise AssertionError(f'no feasible arm-only segmentation for {key}')
        joined = np.empty_like(trace, dtype=np.float64)
        for start, length in sequence:
            piece = table[start, length][1]
            joined[start:start + len(piece)] = piece
        pred[data['rows_of'][key]] = joined.reshape(-1, 10, 7)
        selections += len(sequence)
        frames += len(trace)
    return pred, {'selections': selections, 'frames_per_selection': frames / max(1, selections)}


def variable_predictions(d, evals):
    pools, lengths = variable_pools(d)
    result, meta = {}, {}
    for name, items in pools.items():
        result[name], meta[name] = {}, {}
        for split, data in evals.items():
            pred, info = variable_arm_prediction(items, lengths, data)
            result[name][split] = pred
            meta[name][split] = info
    return result, meta


def verify_grip1(seven_axis, grip):
    # GRIP1's U_DCT3 is a distinct implementation from CMP3's variable-DP E1:
    # it DCT-decodes arm motion in physical units, stores majority grip, and
    # resamples each natural-length candidate to ten steps before oracle choice.
    import gripper_split as gs
    rows = gs._load_rows()
    memory_ids = np.flatnonzero(np.asarray(rows['memory'], bool))
    units, _, _ = gs._variable_pool(rows, memory_ids)
    u10 = np.stack([gs._resample10(unit) for unit in units]).astype(np.float32)
    u10[:, :, :6] = gs._to_norm(u10[:, :, :6])
    _, evals = cmp3.prepare()
    target, hard = evals['unseen']['targets'], evals['unseen']['hard']
    uidx, _ = gs.nearest_candidates(target, u10, np.arange(7))
    grip_u = contributions(u10[uidx], target, hard)
    grip_u_ref = grip['B_all_four_pools']['U_DCT3']['eval']
    checks = []
    for ours, grip_name in (('W_raw', 'W_raw'), ('W_DCT3', 'W_DCT3'), ('U_DCT3', 'U_DCT3')):
        row = grip['B_all_four_pools'][grip_name]['eval']
        ours_metrics = grip_u if ours == 'U_DCT3' else seven_axis[ours]['unseen']
        for label, key in (('all', 'sum_all'), ('hard', 'sum_hard')):
            if ours == 'U_DCT3':
                got = ours_metrics['arm_all' if label == 'all' else 'arm_hard']
            else:
                axis_key = 'axis_all' if label == 'all' else 'axis_hard'
                got = sum(ours_metrics[axis_key][axis] for axis in AXES)
            expected = row['seven_axis_oracle'][label]['arm_mse']
            diff = got - expected
            checks.append({'method':ours,'subset':label,'dim1_seven_axis_arm':got,'grip1_arm_mse':expected,'delta':diff,
                           'match':abs(diff)<=TOL})
            if abs(diff)>TOL:
                raise AssertionError(f'GRIP1 B mismatch {ours}/{label}: {got} vs {expected}')
    dim1_u = seven_axis['U_DCT3']['unseen']
    dim1_u_all = sum(dim1_u['axis_all'][a] for a in AXES)
    dim1_u_hard = sum(dim1_u['axis_hard'][a] for a in AXES)
    return {'checks': checks, 'grip1_u_replica': grip_u,
            'dim1_u_protocol_difference': {
                'DIM1_E1_all': dim1_u_all, 'GRIP1_B_all': grip_u_ref['seven_axis_oracle']['all']['arm_mse'],
                'all_delta': dim1_u_all - grip_u_ref['seven_axis_oracle']['all']['arm_mse'],
                'DIM1_E1_hard': dim1_u_hard, 'GRIP1_B_hard': grip_u_ref['seven_axis_oracle']['hard']['arm_mse'],
                'hard_delta': dim1_u_hard - grip_u_ref['seven_axis_oracle']['hard']['arm_mse'],
                'reason': 'DIM1 E1 uses natural-length units in best-sequence DP; GRIP1 B decodes arm in physical space, uses majority-sign grip, then resamples candidates to 10 steps before per-window oracle choice.'}}


def render(result):
    lines = [
        '# DIM1 追加：グリッパーを除く候補選択', '',
        '候補選択は腕6軸のみで行った。採点は腕の各軸について二乗誤差和/(10×7)とし、グリッパーは選択にも採点にも使わない。比較列「7軸選択」は元の DIM1 E1 と同じ選択結果から腕寄与だけを取り出した値。', '',
        f"評価集合：学習済み {result['counts']['learned']['tasks']}タスク/{result['counts']['learned']['demos']}デモ/{result['counts']['learned']['windows']}窓（hard {result['counts']['learned']['hard']}）、未学習 {result['counts']['unseen']['tasks']}タスク/{result['counts']['unseen']['demos']}デモ/{result['counts']['unseen']['windows']}窓（hard {result['counts']['unseen']['hard']}）。", ''
    ]
    for split in ('learned','unseen'):
        for subset, akey, total in (('全行','axis_all','arm_all'),('難しい行','axis_hard','arm_hard')):
            lines += [f'## {split}・{subset}', '',
                      '| 条件 | 選択 | x | y | z | rx | ry | rz | 腕6軸合計 |',
                      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for method in result['conditions']:
                for choice in ('arm_only','seven_axis'):
                    m=result['results'][choice][method][split]
                    lines.append(f"| {method} | {choice} | " + ' | '.join(f"{m[akey][x]:.6f}" for x in AXES) + f" | {m[total]:.6f} |")
            lines.append('')
    lines += ['## 参考：通常 π0.5 と圧縮だけ', '',
              '通常π0.5は pi05-reference-01、圧縮だけは正解10ステップ窓の自己復元。いずれも腕のみ採点。', '',
              '| 評価集合 | 条件 | 選択 | x | y | z | rx | ry | rz | 腕6軸合計 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('learned','unseen'):
        for name,m in result['references'][split].items():
            for subset,axes_key,total_key in (('全行','axis_all','arm_all'),('難しい行','axis_hard','arm_hard')):
                lines.append(f"| {split} | {name} | {subset} | " + ' | '.join(f"{m[axes_key][x]:.6f}" for x in AXES) + f" | {m[total_key]:.6f} |")
    gv = result['grip1_validation']
    lines += ['', '## GRIP1 B 整合性', '',
              f"W_raw、W_DCT3 は DIM1 E1 の7軸選択腕寄与を照合。U_DCT3 は GRIP1 B の元処理を再実行して照合。6件一致：{all(x['match'] for x in gv['checks'])}、最大絶対差 {max(abs(x['delta']) for x in gv['checks']):.3g}。", '',
              f"ただし DIM1 E1 の U_DCT3 と GRIP1 B の U_DCT3 は候補の復元・再生定義が異なり、DIM1−GRIP1 の腕寄与差は全行 {gv['dim1_u_protocol_difference']['all_delta']:.6f}、hard {gv['dim1_u_protocol_difference']['hard_delta']:.6f}。詳細：{gv['dim1_u_protocol_difference']['reason']}", '',
              '元の親 report.md は変更せず、この追補を arm_only/report.md として保存した。', '']
    return '\n'.join(lines)


def run(output=OUTPUT):
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    if not DIM1_RESULTS.exists() or not GRIP1_RESULTS.exists():
        raise FileNotFoundError('DIM1 or GRIP1 reference results missing')
    base=json.loads(DIM1_RESULTS.read_text())
    grip=json.loads(GRIP1_RESULTS.read_text())
    d,evals=cmp3.prepare()
    counts=cmp3.unit_meta(d,evals)
    if counts!=base['evaluation_sets']:
        raise AssertionError('DIM1 evaluation row sets changed')
    fixed_preds, selection_meta=fixed_arm_predictions(d,evals)
    variable_preds, variable_meta=variable_predictions(d,evals)
    arm_preds={**fixed_preds,**variable_preds}
    methods={
        'W_raw':'fixed_raw70','W_DCT3':'fixed_dct21','W_PCA21':'fixed_pca21',
        'W_keyframe3':'fixed_keyframe21','U_DCT3':'variable_dct22','U_PCA21':'variable_pca22'
    }
    arm_result={method:{split:contributions(arm_preds[method][split],evals[split]['targets'],evals[split]['hard'])
                        for split in evals} for method in methods}
    seven_axis={method:base['retrieval'][dim] for method,dim in methods.items()}
    # Preserve the original DIM1 arm contributions for the same predictions.
    for method in methods:
        for split in evals:
            m=seven_axis[method][split]
            seven_axis[method][split]={'axis_all':{a:m['axis_all'][a] for a in AXES},
                                       'arm_all':sum(m['axis_all'][a] for a in AXES),
                                       'axis_hard':{a:m['axis_hard'][a] for a in AXES},
                                       'arm_hard':sum(m['axis_hard'][a] for a in AXES)}
    refs={}
    for split in evals:
        refs[split]={'pi05':arm_projection(base['pi05'][split])}
        for method in ('fixed_dct21','fixed_pca21','fixed_keyframe21','variable_dct22','variable_pca22'):
            refs[split][method]=arm_projection(base['self_reconstruction'][split][method])
    grip_validation=verify_grip1(seven_axis,grip)
    result={
        'metric':'arm axis contribution = sum_t squared error /(10*7); selection distance uses only axes 0..5',
        'counts':counts,'conditions':list(methods),'pool_mapping':methods,
        'results':{'arm_only':arm_result,'seven_axis':seven_axis},
        'references':refs,'selection_audit':selection_meta,'variable_selection_audit':variable_meta,
        'grip1_validation':grip_validation,
        'validation':{'grip1_all_match':all(x['match'] for x in grip_validation['checks']),
                      'grip1_max_abs_delta':max(abs(x['delta']) for x in grip_validation['checks'])},
        'source_paths':{'dim1':str(DIM1_RESULTS),'grip1':str(GRIP1_RESULTS)}
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (output/'report.md').write_text(render(result))
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args(); r=run(a.output)
    print(json.dumps({'output':str(a.output),'grip1_match':r['validation']['grip1_all_match'],
                      'max_abs_delta':r['validation']['grip1_max_abs_delta']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
