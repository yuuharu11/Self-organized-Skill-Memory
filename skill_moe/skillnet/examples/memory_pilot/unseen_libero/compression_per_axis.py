#!/usr/bin/env python3
"""Axis decomposition of CMP3 E1 with row and selection parity."""
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
CMP3_PRODUCTION = ROOT / 'compress-seen-cuts-01' / 'production'
PI05_REFERENCE = ROOT / 'pi05-reference-01' / 'predictions'
OUTPUT = ROOT / 'compress-per-axis-01'
AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper')
TOL = 1e-7


def axis_contributions(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-axis contribution: sum of 10 time-step errors divided by 10*7."""
    p, t = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if p.shape != t.shape or p.ndim != 3 or p.shape[1:] != (10, 7):
        raise ValueError(f'expected matching (N,10,7), got {p.shape} and {t.shape}')
    return np.mean(np.sum((p - t) ** 2, axis=1) / 70.0, axis=0)


def summarize(pred: np.ndarray, target: np.ndarray, hard: np.ndarray) -> dict:
    pred, target, hard = np.asarray(pred, np.float64), np.asarray(target, np.float64), np.asarray(hard, bool)
    axis_all = axis_contributions(pred, target)
    axis_hard = axis_contributions(pred[hard], target[hard])
    mse_all = float(np.mean((pred - target) ** 2))
    mse_hard = float(np.mean((pred[hard] - target[hard]) ** 2))
    if abs(float(axis_all.sum()) - mse_all) > 1e-12 or abs(float(axis_hard.sum()) - mse_hard) > 1e-12:
        raise AssertionError('axis contributions do not sum to active7 MSE')
    return {'n': int(len(target)), 'hard_n': int(hard.sum()),
            'axis_all': dict(zip(AXES, map(float, axis_all))), 'sum_all': mse_all,
            'axis_hard': dict(zip(AXES, map(float, axis_hard))), 'sum_hard': mse_hard}


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()


def fixed_retrieval(d: dict, evals: dict) -> tuple[dict, dict]:
    mem, fit = d['memory'], d['train']
    mean, basis = cmp2.pca_fit(fit, 21)
    templates = {
        'fixed_raw70': mem,
        'fixed_dct21': cmp2.dct_decode(cmp2.dct_encode(mem, 3), 10).astype(np.float32),
        'fixed_pca21': cmp2.pca_restore(cmp2.pca_project(mem, mean, basis), mean, basis).astype(np.float32),
        'fixed_keyframe21': cmp2.keyframe_decode(cmp2.keyframe_encode(mem, 3), 10).astype(np.float32),
    }
    result, selected_hash = {}, {}
    for method, pool in templates.items():
        result[method], selected_hash[method] = {}, {}
        for split, data in evals.items():
            selected, _ = cmp2.nearest_templates(data['targets'], pool)
            result[method][split] = summarize(pool[selected], data['targets'], data['hard'])
            selected_hash[method][split] = _indices_hash(selected)
    return result, {'fixed_selected_pool_index_sha256': selected_hash}


def variable_retrieval(d: dict, evals: dict) -> tuple[dict, dict]:
    cfg = {'mode': 'all', 'min_len': 5, 'width': 3}
    memory_units, lengths = cmp3.spans(d['memory_traces'], cfg)
    train_units, _ = cmp3.spans(d['train_traces'], cfg)
    pools = {'variable_dct22': cmp2.compressed_variable(memory_units, 'dct', 3)}
    phase_train = np.stack([cmp2.phase_resample(unit, 32) for unit in train_units])
    mean, basis = cmp2.pca_fit(phase_train, 21)
    pools['variable_pca22'] = cmp2.compressed_variable(memory_units, 'pca', 21, mean, basis)
    result, selection_info = {}, {}
    for method, pool in pools.items():
        result[method], selection_info[method] = {}, {}
        for split, data in evals.items():
            _, prediction, info = cmp3.eval_variable(pool, lengths, data)
            result[method][split] = summarize(prediction, data['targets'], data['hard'])
            selection_info[method][split] = {'frames_per_selection': float(info['frames_per_selection'])}
    return result, {'variable_selection': selection_info}


def pi05_reference(evals: dict) -> tuple[dict, dict]:
    result, row_hash = {}, {}
    for split, filename in (('learned', 'learned.npz'), ('unseen', 'unseen_cached.npz')):
        z = np.load(PI05_REFERENCE / filename, allow_pickle=False)
        data = evals[split]
        if z['target'].shape != data['targets'].shape or not np.array_equal(z['target'], data['targets']):
            raise AssertionError(f'pi05-reference target rows/order differ for {split}')
        if not np.array_equal(z['hard'], data['hard']):
            raise AssertionError(f'pi05-reference hard mask differs for {split}')
        if split == 'learned' and not np.array_equal(z['row'], data['ids']):
            raise AssertionError('learned pi05-reference row IDs differ from CMP3 selected rows')
        result[split] = summarize(z['actions'], data['targets'], data['hard'])
        row_hash[split] = _indices_hash(z['row'])
    return result, row_hash


def self_reconstruction(d: dict, evals: dict) -> dict:
    result = {}
    fixed_mean, fixed_basis = cmp2.pca_fit(d['train'], 21)
    cfg = {'mode': 'all', 'min_len': 5, 'width': 3}
    train_units, _ = cmp3.spans(d['train_traces'], cfg)
    var_mean, var_basis = cmp2.pca_fit(np.stack([cmp2.phase_resample(x, 32) for x in train_units]), 21)
    for split, data in evals.items():
        target, hard = data['targets'], data['hard']
        pred = {
            'fixed_dct21': cmp2.dct_decode(cmp2.dct_encode(target, 3), 10),
            'fixed_pca21': cmp2.pca_restore(cmp2.pca_project(target, fixed_mean, fixed_basis), fixed_mean, fixed_basis),
            'fixed_keyframe21': cmp2.keyframe_decode(cmp2.keyframe_encode(target, 3), 10),
            # A 10-step query is treated as a one-segment variable-length item.
            'variable_dct22': cmp2.dct_decode(cmp2.dct_encode(target, 3), 10),
            'variable_pca22': np.stack([cmp2.phase_pca_restore(row, var_mean, var_basis, 10) for row in target]),
        }
        result[split] = {method: summarize(reconstruction, target, hard) for method, reconstruction in pred.items()}
    return result


def _verify_against_cmp3(retrieval: dict, production: dict) -> list[dict]:
    mapping = (
        ('fixed_raw70', 'raw', 70), ('fixed_dct21', 'dct', 21), ('fixed_pca21', 'pca', 21),
        ('fixed_keyframe21', 'keyframe', 21), ('variable_dct22', 'anchor_variable_dct', 21),
        ('variable_pca22', 'anchor_variable_pca', 21),
    )
    checks = []
    for ours, method, v in mapping:
        matches = [r for r in production['E1'] if r['method'] == method and r['V'] == v]
        if len(matches) != 1:
            raise AssertionError(f'missing unique CMP3 E1 condition {method} V={v}')
        for split in ('learned', 'unseen'):
            got, ref = retrieval[ours][split], matches[0]['eval'][split]
            da, dh = got['sum_all'] - ref['all_mse'], got['sum_hard'] - ref['hard_mse']
            passed = abs(da) <= TOL and abs(dh) <= TOL
            checks.append({'method': ours, 'split': split, 'all_delta': da, 'hard_delta': dh, 'match': passed})
            if not passed:
                raise AssertionError(f'E1 total mismatch {ours}/{split}: all={da}, hard={dh}')
    return checks


def _table(lines: list[str], result: dict, split: str, subset: str, names: list[str]) -> None:
    lines += [f"### {split} / {subset}", '',
              '| 条件 | x | y | z | rx | ry | rz | グリッパー | 合計 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m = result[name][split]
        axes = m['axis_all'] if subset == '全行' else m['axis_hard']
        total = m['sum_all'] if subset == '全行' else m['sum_hard']
        lines.append('| ' + name + ' | ' + ' | '.join(f'{axes[a]:.6f}' for a in AXES) + f' | {total:.6f} |')
    lines.append('')


def render(result: dict) -> str:
    sets, names = result['evaluation_sets'], result['conditions']
    lines = [
        '# CMP3 E1 の軸別誤差', '',
        '各軸の寄与 = その軸の10ステップ二乗誤差和 / (10×7)。7軸寄与の合計が active7 正規化 MSE になる。選択と予測は CMP3 E1 と同じ（7軸距離で最近傍候補を選択）。', '',
        f"メモリ：{sets['memory']['tasks']}タスク・{sets['memory']['demos']}デモ・{sets['memory']['windows']}窓。学習済み：{sets['learned']['tasks']}タスク・{sets['learned']['demos']}デモ・{sets['learned']['windows']}窓（hard {sets['learned']['hard']}）。未学習：{sets['unseen']['tasks']}タスク・{sets['unseen']['demos']}デモ・{sets['unseen']['windows']}窓（hard {sets['unseen']['hard']}）。", '',
        '## E1 検索・選択後', ''
    ]
    _table(lines, result['retrieval'], 'learned', '全行', names)
    _table(lines, result['retrieval'], 'learned', 'hard', names)
    _table(lines, result['retrieval'], 'unseen', '全行', names)
    _table(lines, result['retrieval'], 'unseen', 'hard', names)
    lines += ['## 通常 π0.5', '', 'pi05-reference-01 の予測。行ID、正解窓、hard mask の順序を検証してから集計。', '',
              '| 集合 | 行 | x | y | z | rx | ry | rz | グリッパー | 合計 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('learned', 'unseen'):
        for label in ('全行', 'hard'):
            m = result['pi05'][split]
            axes = m['axis_all'] if label == '全行' else m['axis_hard']
            total = m['sum_all'] if label == '全行' else m['sum_hard']
            lines.append(f"| {split} | {label} | " + ' | '.join(f'{axes[a]:.6f}' for a in AXES) + f' | {total:.6f} |')
    lines += ['', '## 圧縮だけ（正解窓を自己復元）', '',
              '検索をせず、正解の10ステップ窓自身を符号化して復元した対照。variable PCA は可変長の学習単位から fit したPCA21で、各評価窓を32位相へ写してから10ステップへ戻す。', '',
              '| 集合 | 条件 | 行 | x | y | z | rx | ry | rz | グリッパー | 合計 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('learned', 'unseen'):
        for method, m in result['self_reconstruction'][split].items():
            for label, axes_key, total_key in (('全行', 'axis_all', 'sum_all'), ('hard', 'axis_hard', 'sum_hard')):
                lines.append(f"| {split} | {method} | {label} | " + ' | '.join(f'{m[axes_key][a]:.6f}' for a in AXES) + f' | {m[total_key]:.6f} |')
    lines += ['', '## 正解行動の分散（正規化後）', '', '| 集合 | x | y | z | rx | ry | rz | グリッパー |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, axes in result['target_variance'].items():
        lines.append(f"| {split} | " + ' | '.join(f'{axes[a]:.6f}' for a in AXES) + ' |')
    validation = result['validation']
    lines += ['', '## CMP3 E1 合計との一致', '',
              f"6条件×2評価集合で、軸寄与の合計を production E1 の全体/hard 値と照合。最大差={validation['max_abs_delta']:.3g}、許容差={TOL:g}、全条件一致={validation['all_match']}。", '']
    return '\n'.join(lines)


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    production_path = CMP3_PRODUCTION / 'results.json'
    if not production_path.exists():
        raise FileNotFoundError(f'CMP3 production missing: {production_path}')
    production = json.loads(production_path.read_text())
    if production.get('mode') != 'full':
        raise AssertionError('CMP3 reference is not full production')
    d, evals = cmp3.prepare()
    counts = cmp3.unit_meta(d, evals)
    if counts != production['evaluation_sets']:
        raise AssertionError(f'CMP3 evaluation counts differ: {counts}')
    selection_path = CMP3_PRODUCTION.parent / 'production-control' / 'evaluation-selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if [list(x) for x in evals['learned']['demos']] != selection['seen_demo_pairs']:
            raise AssertionError('learned demo selection differs from production')
    fixed, selection_hashes = fixed_retrieval(d, evals)
    variable, variable_info = variable_retrieval(d, evals)
    retrieval = {**fixed, **variable}
    checks = _verify_against_cmp3(retrieval, production)
    pi05, pi05_rows = pi05_reference(evals)
    self_rec = self_reconstruction(d, evals)
    variances = {split: dict(zip(AXES, np.var(data['targets'].reshape(-1, 7).astype(np.float64), axis=0).tolist()))
                 for split, data in evals.items()}
    max_delta = max(max(abs(x['all_delta']), abs(x['hard_delta'])) for x in checks)
    result = {
        'metric': 'axis contribution = sum over 10 steps of squared error / (10*7); seven axes sum to active7 MSE',
        'evaluation_sets': counts, 'axis_names': list(AXES),
        'conditions': list(retrieval), 'retrieval': retrieval, 'pi05': pi05,
        'self_reconstruction': self_rec, 'target_variance': variances,
        'selection_audit': selection_hashes, 'variable_selection_info': variable_info,
        'pi05_row_id_sha256': pi05_rows,
        'validation': {'cmp3_e1_checks': checks, 'max_abs_delta': max_delta,
                       'all_match': all(x['match'] for x in checks), 'tolerance': TOL,
                       'pi05_targets_and_hard_masks_match': True},
        'sources': {'cmp3_production': str(CMP3_PRODUCTION), 'pi05_reference': str(PI05_REFERENCE)},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (output / 'report.md').write_text(render(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    r = run(args.output)
    print(json.dumps({'output': str(args.output), 'max_abs_delta': r['validation']['max_abs_delta'],
                      'all_match': r['validation']['all_match'], 'counts': r['evaluation_sets']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
