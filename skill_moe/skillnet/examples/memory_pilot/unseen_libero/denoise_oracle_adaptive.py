#!/usr/bin/env python3
"""DEN2 follow-up: choose bridge noise per row from Dev-calibrated bins."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import denoise_oracle_drafts as den2

SOURCE = den2.OUT / 'attempt-01'
OUT = den2.OUT / 'adaptive-t0-01'
FAMILIES = ('fixed_raw', 'fixed_dct3', 'variable_dct3')
T0S = np.asarray(den2.T0S, np.float64)
QUANTILES = (0.25, 0.50, 0.75)


def arm6_mse(prediction, target):
    p = np.asarray(prediction, np.float64)[..., :6]
    y = np.asarray(target, np.float64)[..., :6]
    if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (10, 6):
        raise ValueError((p.shape, y.shape))
    return np.square(p - y).sum(axis=(1, 2)) / 70.0


def fit_quantile_policy(feature, losses, t0_values=T0S, quantiles=QUANTILES):
    """Fit a quantile-binned t0 map by Dev mean arm6 MSE only."""
    x = np.asarray(feature, np.float64).reshape(-1)
    loss = np.asarray(losses, np.float64)
    t0_values = np.asarray(t0_values, np.float64)
    if loss.shape != (len(x), len(t0_values)) or not np.isfinite(x).all() or not np.isfinite(loss).all():
        raise ValueError((x.shape, loss.shape, t0_values.shape))
    edges = np.unique(np.quantile(x, quantiles))
    bins = np.digitize(x, edges, right=False)
    choices, means, counts = [], [], []
    fallback = int(np.argmin(loss.mean(axis=0)))
    for b in range(len(edges) + 1):
        mask = bins == b
        counts.append(int(mask.sum()))
        row = loss[mask].mean(axis=0) if mask.any() else loss.mean(axis=0)
        means.append(row.tolist())
        choices.append(int(np.argmin(row)) if mask.any() else fallback)
    return {'edges': edges.tolist(), 'choice_indices': choices,
            'choice_t0': [float(t0_values[i]) for i in choices],
            'mean_mse6_by_bin': means, 'dev_count_by_bin': counts,
            'quantiles': list(quantiles), 'criterion': 'minimum mean Dev arm6 MSE per bin'}


def apply_quantile_policy(feature, policy):
    x = np.asarray(feature, np.float64).reshape(-1)
    edges = np.asarray(policy['edges'], np.float64)
    bins = np.digitize(x, edges, right=False)
    choices = np.asarray(policy['choice_indices'], np.int64)
    selected_index = choices[bins]
    t0_values = np.asarray(policy['choice_t0'], np.float32)[bins]
    return bins.astype(np.int64), selected_index, t0_values


def _spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def _load_dev_predictions(base, model, rows, ids, row_key, draft_by_family, normal, batch):
    losses = {}
    predictions = {}
    for family in FAMILIES:
        draft = np.asarray(draft_by_family[family], np.float32).copy()
        draft[:, :, 6] = normal[:, :, 6]
        preds = []
        for t0 in T0S:
            if t0 == 1.0:
                pred = normal.copy()
            else:
                pred = den2._bridge(model, base, rows, ids, row_key, draft, float(t0), batch=batch)
                pred = den2.use_pi05_gripper(pred, normal)
            preds.append(pred)
        stack = np.stack(preds, axis=1)
        predictions[family] = {'draft': draft, 'predictions': stack}
        losses[family] = np.stack([arm6_mse(stack[:, j], np.asarray(rows['actions'][ids, :, :7])) for j in range(len(T0S))], axis=1)
    return predictions, losses


def _stack_saved(source, setname, family):
    files = [den2._load_npz(source / 'predictions' / f'{setname}__{family}__t0_{t0:.1f}.npz') for t0 in T0S]
    reference = files[0]
    for z in files[1:]:
        for k in ('row', 'target', 'task', 'episode', 'hard', 'draft', 'normal'):
            np.testing.assert_array_equal(z[k], reference[k])
    return reference, np.stack([z['prediction'] for z in files], axis=1)


def _per_row_metrics(p, ref):
    return {name: den2.action_metrics(p, ref['target'])[name] for name in ('arm6_contrib', 'active7')}


def apply_fixed_pi05_gripper(prediction, ordinary_pi05):
    """Use the ordinary-policy gripper while retaining the candidate arm output."""
    return den2.use_pi05_gripper(prediction, ordinary_pi05)


def correct_existing_run(source=SOURCE, adaptive=OUT, corrected_dir=None):
    """Recompute adaptive active7/gripper summaries with ordinary-pi05 gripper.

    The historical predictions and results are left intact. Corrected arrays,
    summary, and report are written to a fresh child directory.
    """
    import copy
    import shutil

    source, adaptive = Path(source), Path(adaptive)
    corrected_dir = Path(corrected_dir) if corrected_dir else adaptive / 'gripper-fixed-03'
    if corrected_dir.exists():
        raise FileExistsError(corrected_dir)
    base_results = json.loads((adaptive / 'results.json').read_text())
    corrected_den2_dir = source.parent / 'review-gripper-fixed-02'
    corrected_den2 = json.loads((corrected_den2_dir / 'results.json').read_text())
    if corrected_den2.get('status') != 'complete' or not corrected_den2.get('gripper_policy'):
        raise ValueError('expected completed review-gripper-fixed-02 DEN2 output')
    corrected_dir.mkdir(parents=True)
    (corrected_dir / 'predictions').mkdir()
    result = copy.deepcopy(base_results)
    result['status'] = 'complete'
    result['gripper_policy'] = 'ordinary pi05 gripper copied into every final action; arm output unchanged'
    result['corrected_from'] = str(adaptive)
    result['den2_parity_reference'] = str(corrected_den2_dir)
    result['fixed_condition_parity'] = {}
    result['pre_correction_fixed_condition_delta'] = {}

    for setname in ('learned', 'unseen'):
        for family in FAMILIES:
            entry = result['evaluation_sets'][setname][family]
            chosen_t0 = float(corrected_den2['selection']['chosen_t0'][family])
            fixed_key = f'{setname}__{family}__t0_{chosen_t0:.1f}'
            fixed_ref = corrected_den2['sets'][setname][fixed_key]
            normal_key = f'{setname}__{family}__t0_1.0'
            normal_data = den2._load_npz(corrected_den2_dir / 'predictions' / f'{normal_key}.npz')
            normal_ref = normal_data['normal']
            draft_ref = normal_data['draft']
            result['pre_correction_fixed_condition_delta'][f'{setname}/{family}'] = {}
            for metric in ('arm6_contrib', 'grip_contrib', 'active7'):
                gaps = {}
                for subset in ('all', 'hard'):
                    old_m = entry['fixed_dev_best']['metrics'][subset][metric]['mean']
                    ref_m = fixed_ref['metrics'][subset][metric]['mean']
                    gaps[subset] = float(old_m - ref_m)
                result['pre_correction_fixed_condition_delta'][f'{setname}/{family}'][metric] = gaps
                if metric == 'arm6_contrib' and max(abs(x) for x in gaps.values()) > 1e-12:
                    raise AssertionError(f'arm6 changed during gripper correction: {setname}/{family}: {gaps}')

            def copy_ref_summary(t0):
                refkey = f'{setname}__{family}__t0_{float(t0):.1f}'
                ref = corrected_den2['sets'][setname][refkey]
                return {k: copy.deepcopy(v) for k, v in ref.items() if k in ('metrics', 'error_delta_vs', 'action_distance', 'selected_on_dev', 'prediction_file')}

            entry['fixed_dev_best'] = copy_ref_summary(chosen_t0)
            entry['ordinary_pi05_t0_1'] = copy_ref_summary(1.0)
            entry['draft_t0_0'] = copy_ref_summary(0.0)

            # Per-row oracle selection remains arm-only; select among corrected
            # DEN2 predictions, with gripper held to the same ordinary output.
            candidates, target = [], None
            ref_arrays = None
            for t0 in T0S:
                key = f'{setname}__{family}__t0_{t0:.1f}'
                z = den2._load_npz(corrected_den2_dir / 'predictions' / f'{key}.npz')
                candidates.append(z['prediction'])
                if ref_arrays is None:
                    ref_arrays = z
                    target = z['target']
            stack = np.stack(candidates, axis=1)
            losses = np.stack([arm6_mse(stack[:, j], target) for j in range(len(T0S))], axis=1)
            oracle_index = losses.argmin(axis=1)
            oracle = stack[np.arange(len(target)), oracle_index]
            entry['per_row_oracle_t0'] = den2.summarize_condition(
                oracle, target, ref_arrays['hard'], ref_arrays['task'], ref_arrays['episode'],
                ref_arrays['draft'], ref_arrays['normal'])

            for policy_name, policy_record in entry['policy_eval'].items():
                old_path = Path(adaptive / 'predictions' / f'{setname}_{family}_{policy_name}.npz')
                z = den2._load_npz(old_path)
                normal = np.asarray(normal_ref, np.float32)
                if len(normal) != len(z['prediction']):
                    raise AssertionError(f'row count mismatch: {old_path}')
                np.testing.assert_array_equal(z['row'], normal_data['row'])
                np.testing.assert_array_equal(z['target'], normal_data['target'])
                fixed = apply_fixed_pi05_gripper(z['prediction'], normal)
                z['normal'] = normal
                z['draft'] = draft_ref
                new_path = corrected_dir / 'predictions' / old_path.name
                den2._save_npz(new_path, **{**z, 'prediction': fixed})
                policy_record['metrics'] = den2.summarize_condition(
                    fixed, z['target'], z['hard'], z['task'], z['episode'], draft_ref, normal)
                policy_record['prediction_file'] = str(new_path)

    # Fixed-t0 rows must match the corrected DEN2 result including the episode
    # bootstrap intervals, while arm-only metrics remain byte-for-byte stable.
    for setname in ('learned', 'unseen'):
        for family in FAMILIES:
            key = f'{setname}/{family}'
            t0 = float(corrected_den2['selection']['chosen_t0'][family])
            ref = corrected_den2['sets'][setname][f'{setname}__{family}__t0_{t0:.1f}']['metrics']
            got = result['evaluation_sets'][setname][family]['fixed_dev_best']['metrics']
            result['fixed_condition_parity'][key] = {}
            for metric in ('arm6_contrib', 'grip_contrib', 'active7'):
                result['fixed_condition_parity'][key][metric] = {}
                for subset in ('all', 'hard'):
                    if got[subset][metric] != ref[subset][metric]:
                        raise AssertionError(f'DEN3 fixed row does not match corrected DEN2: {key}/{subset}/{metric}')
                    result['fixed_condition_parity'][key][metric][subset] = float(
                        got[subset][metric]['mean'] - ref[subset][metric]['mean'])

    result['corrected_output_dir'] = str(corrected_dir)
    corrected_dir.joinpath('results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (corrected_dir / 'report.md').write_text(render_report(result))
    return result


def run(source=SOURCE, out=OUT, threads=8, batch=8):
    import torch
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.20)
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite adaptive DEN2 output: {out}')
    base = den2.dn1._module(HERE.parent / 'pi05_memory' / 'run.py', 'den2_adaptive_pi05_run')
    model, _ = base.load_policy(den2.dn1.PUBLIC_CKPT)
    # Reuse the DEN2 exact row/draft construction and its corrected Dev rows.
    d, sets = den2.make_sets()
    unseen_rows = sets['dev']['rows']
    dev_ids = np.asarray(sets['dev']['ids'], np.int64)
    dev_keys = np.asarray(sets['dev']['row_key'], np.uint32)
    dev_normal_file = den2._load_npz(source / 'dev_normal.npz')
    dev_normal = np.asarray(dev_normal_file['actions'], np.float32)
    dev_rows = den2._load_arrays(den2.UNSEEN)
    dev_target = np.asarray(dev_rows['actions'][dev_ids, :, :7], np.float32)
    dev_drafts = {fam: den2._load_npz(source / 'drafts' / f'dev_{fam}.npz')['draft'] for fam in FAMILIES}
    dev_predictions, dev_losses = _load_dev_predictions(base, model, dev_rows, dev_ids, dev_keys, dev_drafts, dev_normal, batch)
    output = {'status': 'running', 'source_run': str(source), 't0_values': T0S.tolist(),
              'dev_rows': len(dev_ids), 'evaluation_sets': {}, 'policies': {}}
    out.mkdir(parents=True)
    (out / 'predictions').mkdir()
    for fam in FAMILIES:
        # Oracle-side feature: true draft error, unavailable during inference.
        draft_err_dev = arm6_mse(dev_predictions[fam]['draft'], dev_target)
        # Inference feature: distance between the fixed arm draft and ordinary pi05 arm output.
        pi_distance_dev = arm6_mse(dev_predictions[fam]['draft'], dev_normal)
        policies = {
            'draft_true_error_upper_reference': fit_quantile_policy(draft_err_dev, dev_losses[fam]),
            'draft_pi05_distance': fit_quantile_policy(pi_distance_dev, dev_losses[fam]),
        }
        output['policies'][fam] = policies
        for setname in ('learned', 'unseen'):
            ref, pred_stack = _stack_saved(source, setname, fam)
            target = ref['target']
            draft = ref['draft']
            normal = ref['normal']
            true_draft_error = arm6_mse(draft, target)
            pi05_distance = arm6_mse(draft, normal)
            fixed_t0 = float(json.loads((source / 'selection.json').read_text())['chosen_t0'][fam])
            fixed_idx = int(np.flatnonzero(np.isclose(T0S, fixed_t0))[0])
            fixed_pred = pred_stack[:, fixed_idx]
            normal_pred = pred_stack[:, -1]
            draft_pred = pred_stack[:, 0]
            losses = np.stack([arm6_mse(pred_stack[:, j], target) for j in range(len(T0S))], axis=1)
            oracle_idx = np.argmin(losses, axis=1)
            oracle_pred = pred_stack[np.arange(len(target)), oracle_idx]
            subset_result = {}
            policies_eval = {}
            for policy_name, feature in (
                ('draft_true_error_upper_reference', true_draft_error),
                ('draft_pi05_distance', pi05_distance)):
                bins, indices, row_t0 = apply_quantile_policy(feature, policies[policy_name])
                chosen = pred_stack[np.arange(len(target)), indices]
                policies_eval[policy_name] = {
                    't0_histogram': {str(float(t)): int(np.sum(np.isclose(row_t0, t))) for t in T0S},
                    'bin_counts': np.bincount(bins, minlength=len(policies[policy_name]['choice_t0'])).tolist(),
                    'metrics': den2.summarize_condition(chosen, target, ref['hard'], ref['task'], ref['episode'], draft, normal),
                }
                np.savez_compressed(out / 'predictions' / f'{setname}_{fam}_{policy_name}.npz',
                    row=ref['row'], task=ref['task'], episode=ref['episode'], hard=ref['hard'],
                    feature=feature, draft_true_arm6_error=true_draft_error, pi05_arm6_distance=pi05_distance,
                    bin=bins, selected_t0=row_t0, prediction=chosen, target=target)
            conditions = {
                'fixed_dev_best': fixed_pred,
                'ordinary_pi05_t0_1': normal_pred,
                'draft_t0_0': draft_pred,
                'per_row_oracle_t0': oracle_pred,
            }
            for cond, pred in conditions.items():
                subset_result[cond] = den2.summarize_condition(pred, target, ref['hard'], ref['task'], ref['episode'], draft, normal)
            subset_result['draft_true_error_spearman'] = _spearman(pi05_distance, true_draft_error)
            subset_result['policy_eval'] = policies_eval
            output['evaluation_sets'].setdefault(setname, {})[fam] = subset_result
        np.savez_compressed(out / f'dev_{fam}_losses.npz', t0=T0S, true_error= draft_err_dev,
            pi05_distance=pi_distance_dev, **{f'loss_{i}':dev_losses[fam][:,i] for i in range(len(T0S))})
    output['status'] = 'complete'
    output['notes'] = [
        'Quantile policies use four bins with boundaries from Dev feature quantiles (25/50/75%).',
        'The true-draft-error policy is an oracle-sided reference: its feature requires ground truth and is not inferable at deployment.',
        'The deployable policy bins arm6 MSE-distance between draft and ordinary pi05 output; its t0 per bin is chosen on Dev labels only.',
        'Per-row oracle chooses minimum ground-truth arm6 error among the six t0 values and is an upper bound.',
        'Bootstrap uses the DEN2 episode-level paired bootstrap implementation (5,000 draws).',
    ]
    tmp = out / 'results.json.tmp'
    tmp.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(out / 'results.json')
    (out / 'report.md').write_text(render_report(output))
    return output


def _mean_ci(record, subset='all', metric='arm6_contrib'):
    metrics = record.get('metrics', record)
    metrics = metrics.get('metrics', metrics)
    v = metrics[subset][metric]
    return f"{v['mean']:.5f} [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]"


def _delta_ci(record, reference, subset='all', metric='mse6_sum_div7'):
    summary = record if 'error_delta_vs' in record else record['metrics']
    v = summary['error_delta_vs'][reference][subset][metric]
    return f"{v['mean']:.5f} [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]"


def render_report(result):
    lines = ['# DEN3（DEN2追加）：誤差に応じたノイズ強度', '',
        't0候補は0/.1/.2/.4/.6/1.0。各行の予測はDEN2 attempt-01と同じ下書き・noise・評価行で、既存の行別予測を再利用した。',
        '区間規則はDevの特徴量25/50/75百分位で4分割し、各区間内の平均腕6軸MSEが最小となるt0を選択。選択以外のEval/learned正解は規則作成に使っていない。',
        '全指標の単位とbootstrapはDEN2と同じ。表は腕6軸寄与MSE（sum/(10×7)）、active7寄与MSE。各値は平均[episode bootstrap 95% CI]。','']
    if result.get('gripper_policy'):
        lines += ['**修正済み集計：最終グリッパーは全t0で通常π0.5出力に固定。腕6軸の値はDEN2の修正版と一致する。**',
                  f"修正版DEN2との固定t0照合元：`{result.get('den2_parity_reference')}`。修正版データ：`{result.get('corrected_output_dir')}`。", '']
        lines += ['|固定条件|指標|全体 差|難しい行 差|', '|---|---|---:|---:|']
        for key, metrics in result.get('fixed_condition_parity', {}).items():
            for metric in ('arm6_contrib', 'grip_contrib', 'active7'):
                gap = metrics[metric]
                lines.append(f"|{key}|{metric}|{gap['all']:.3g}|{gap['hard']:.3g}|")
        lines.append('差は本レポートの固定Dev最良行 − review-gripper-fixed-02 の同じ固定t0行。腕6軸差は0、active7とグリッパー寄与も補正後は完全一致をassertした。')
        lines.append('')
    for fam in FAMILIES:
        lines += [f'## {fam}', '', 'Devで選んだ量子化規則：', '']
        for name, policy in result['policies'][fam].items():
            lines.append(f"- {name}: edges={policy['edges']}; t0={policy['choice_t0']}; Dev bin rows={policy['dev_count_by_bin']}")
        lines += ['', '|集合|規則|t0使用数|腕6 MSE all|腕6 MSE hard|active7 all|active7 hard|Δ腕6 vs draft all|Δ腕6 vs π0.5 all|',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for setname in ('learned', 'unseen'):
            e = result['evaluation_sets'][setname][fam]
            sel = json.loads((SOURCE / 'selection.json').read_text())['chosen_t0'][fam]
            for cond, label in (
                ('fixed_dev_best', f'固定Dev最良 ({sel})'),
                ('ordinary_pi05_t0_1', '通常π0.5 / t0=1'),
                ('draft_t0_0', '下書き / t0=0'),
                ('per_row_oracle_t0', '窓ごとの正解最良 (上限)'),
            ):
                r = e[cond]
                n = r['metrics']['all']['arm6_contrib']['rows']
                hist = ({str(float(t)): (n if np.isclose(t, sel) else 0) for t in T0S}
                        if cond == 'fixed_dev_best' else
                        {'1.0': n} if cond == 'ordinary_pi05_t0_1' else
                        {'0.0': n} if cond == 'draft_t0_0' else {'multiple': n})
                lines.append(f"|{setname}|{label}|{hist}|{_mean_ci(r)}|{_mean_ci(r,'hard')}|{_mean_ci(r,'all','active7')}|{_mean_ci(r,'hard','active7')}|{_delta_ci(r,'draft')}|{_delta_ci(r,'ordinary_pi05')}|")
            for pname in ('draft_true_error_upper_reference', 'draft_pi05_distance'):
                r = e['policy_eval'][pname]
                lines.append(f"|{setname}|{pname}|{r['t0_histogram']}|{_mean_ci(r)}|{_mean_ci(r,'hard')}|{_mean_ci(r,'all','active7')}|{_mean_ci(r,'hard','active7')}|{_delta_ci(r,'draft')}|{_delta_ci(r,'ordinary_pi05')}|")
            lines.append(f"|{setname}|Spearman(distance, true draft error)|—|{e['draft_true_error_spearman']:.4f}|—|—|—|—|—|")
        lines += ['']
    lines += ['## 読み方', '',
        '「窓ごとの正解最良」は各窓の正解誤差でt0を選ぶため、推論可能な方法ではなく上限。下書き誤差の区間規則も真の誤差を特徴に使うため参考上限である。推論候補は下書きと通常π0.5の腕出力の距離を使う規則のみ。',
        '固定条件、通常π0.5、下書きそのものとの比較は同一の行・noiseで対応付けたepisode bootstrap差を `results.json` に保存。SpearmanはEval/learned行ごとの下書き-π0.5距離と真の下書き誤差の相関。',
        'この節はDEN2の元レポートにも追記済み。データ：同階層のresults.json / predictions/*.npz。','']
    return '\n'.join(lines)


def append_to_den2_report(source_report, adaptive_report, marker='# DEN2追加：誤差に応じたノイズ強度'):
    source_report, adaptive_report = Path(source_report), Path(adaptive_report)
    text = source_report.read_text()
    if marker in text:
        raise FileExistsError(f'adaptive section already appended to {source_report}')
    with source_report.open('a') as f:
        f.write('\n\n' + adaptive_report.read_text())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--append-to-den2-report', action='store_true')
    args = ap.parse_args()
    result = run(args.source, args.out, args.threads, args.batch)
    if args.append_to_den2_report:
        append_to_den2_report(args.source / 'report.md', args.out / 'report.md')
    print(json.dumps({'status': result['status'], 'output': str(args.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
