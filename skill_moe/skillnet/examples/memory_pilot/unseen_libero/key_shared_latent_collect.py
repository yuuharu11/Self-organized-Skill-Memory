#!/usr/bin/env python3
"""Aggregate locked KEY2-E runs with paired episode-bootstrap intervals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

INPUTS = ('e_vlm', 'e_vlm_h', 'e_vj_h', 'e_vj_vlm_h', 'e_h')
CONTROLS = ('separate', 'p0', 'p5')


def _load_run(run_dir: Path):
    result_path = run_dir / 'results.json'
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())
    mode = result['mode']
    objective = result.get('objective', 'soft')
    label = f'{mode}-hard' if objective == 'hard' else mode
    return label, result


def _action_error(pred, target):
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    arm = np.square(pred[..., :6] - target[..., :6]).sum((1, 2)) / (pred.shape[1] * 7.0)
    grip = np.square(pred[..., 6] - target[..., 6]).sum(1) / (pred.shape[1] * 7.0)
    return {'arm6': arm, 'mse_gripper_truth': arm, 'active7_saved_gripper': arm + grip}


def _episode_bootstrap(values, task, episode, *, reps=5000, seed=4101):
    values = np.asarray(values, np.float64)
    groups = {}
    for i, (t, e) in enumerate(zip(task, episode, strict=True)):
        groups.setdefault((int(t), int(e)), []).append(i)
    group_values = [values[ix] for ix in groups.values()]
    if not group_values:
        raise ValueError('no episodes for bootstrap')
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for r in range(reps):
        picked = rng.integers(0, len(group_values), len(group_values))
        draws[r] = np.concatenate([group_values[i] for i in picked]).mean()
    return {'mean': float(values.mean()), 'ci95_low': float(np.quantile(draws, .025)),
            'ci95_high': float(np.quantile(draws, .975)), 'episodes': len(group_values),
            'rows': len(values)}


def _fmt(v, n=5):
    try:
        return f'{float(v):.{n}f}' if np.isfinite(float(v)) else '—'
    except (TypeError, ValueError):
        return '—'


def _rank_line(split, label, subset, rank):
    f = _fmt
    return (f"|{split}|{label}|{subset}|{f(rank.get('spearman_positive'))}|{f(rank.get('cross_entropy'))}|"
            f"{f(rank.get('kl_pq'))}|{f(rank.get('recall@1'))}/{f(rank.get('recall@5'))}/{f(rank.get('recall@20'))}|"
            f"{f(rank.get('ndcg@20'))}|{f(rank.get('mrr'))}|{f(rank.get('top1_arm6_mse'))}/{f(rank.get('top20_arm6_mse'))}|"
            f"{f(rank.get('recovery'))}|")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baselines', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.run_root, args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    baseline_doc = json.loads(args.baselines.read_text())
    runs = {}
    for p in sorted((root / 'final').iterdir()):
        if not p.is_dir():
            continue
        try:
            label, result = _load_run(p)
        except FileNotFoundError:
            continue
        runs[label] = {'dir': p, 'result': result}
    expected = set(INPUTS) | set(CONTROLS) | {'e_vlm_h-hard'}
    missing = expected - runs.keys()
    if missing:
        raise RuntimeError(f'missing final outputs for {sorted(missing)}')

    split_tables = {}
    paired = {}
    rows_by_split = {}
    for split in ('learned', 'unseen'):
        arrays = {}
        keys = None
        for label in sorted(expected):
            run = runs[label]
            mode = run['result']['mode']
            pred_path = run['dir'] / f'{mode}_{split}_paired_predictions.npz'
            pred = np.load(pred_path, allow_pickle=False)
            row_identity = {k: pred[k] for k in ('key', 'target_actions', 'task', 'episode', 'hard')}
            if keys is None:
                keys = row_identity
            else:
                for k in keys:
                    if not np.array_equal(keys[k], row_identity[k]):
                        raise AssertionError(f'locked paired rows differ: {split}/{label}/{k}')
            arrays[label] = pred
        rows_by_split[split] = {'key': keys['key'].tolist(), 'n': len(keys['key']),
            'hard_n': int(np.asarray(keys['hard'], bool).sum())}
        task = np.asarray(keys['task']); episode = np.asarray(keys['episode']); hard = np.asarray(keys['hard'], bool)
        base_metrics = baseline_doc['metrics'][split]
        split_tables[split] = {}
        for label in sorted(expected):
            run = runs[label]
            metrics = run['result']['models'][run['result']['mode']]['splits'][split]
            split_tables[split][label] = {
                'seed': run['result']['seed'], 'objective': run['result'].get('objective', 'soft'),
                'n': metrics['n'], 'hard_n': metrics['hard_n'],
                'selected_blend': metrics['selected_blend'], 'blend_by_k': metrics['blend_by_k'],
                'actions_all': metrics['actions_all'], 'actions_hard': metrics['actions_hard'],
                'rank_all': metrics['rank_all'], 'rank_hard': metrics['rank_hard'],
                'training': run['result']['models'][run['result']['mode']]['train'],
                'checkpoint': run['result']['models'][run['result']['mode']]['checkpoint'],
            }
        # Paired differences in the submitted action, grouped at episode level.
        refs = ('e_vlm_h', 'p0', 'p5', 'separate', 'random-5')
        paired[split] = {}
        for label in sorted(expected):
            comp_refs = refs if label != 'e_vlm_h' else ('p0', 'p5', 'separate', 'random-5')
            for ref in comp_refs:
                if ref == 'random-5':
                    chosen = arrays[label]['selected_actions']
                    random_stack = arrays[label]['random_actions']
                    pred_by_ref = random_stack.mean(axis=0)
                else:
                    chosen = arrays[label]['selected_actions']
                    pred_by_ref = arrays[ref]['selected_actions']
                diff = _action_error(chosen, keys['target_actions'])['arm6'] - _action_error(pred_by_ref, keys['target_actions'])['arm6']
                item = {'all': _episode_bootstrap(diff, task, episode, seed=4101),
                        'hard': _episode_bootstrap(diff[hard], task[hard], episode[hard], seed=4102)}
                paired[split][f'{label}_minus_{ref}'] = item
        # Top-1 versus selected top-k is recorded explicitly; selected top-k remains the primary.
        for label in expected:
            pred = arrays[label]
            run = runs[label]
            mode = run['result']['mode']
            ranker = run['result']['models'][mode]['splits'][split]
            top1 = pred['top1_actions']
            selected = pred['selected_actions']
            diff = _action_error(selected, keys['target_actions'])['arm6'] - _action_error(top1, keys['target_actions'])['arm6']
            paired[split][f'{label}_selected_blend_minus_top1'] = {
                'all': _episode_bootstrap(diff, task, episode, seed=4103),
                'hard': _episode_bootstrap(diff[hard], task[hard], episode[hard], seed=4104),
                'selected_blend': ranker['selected_blend']}
        # Preserve fixed CPU baselines, including rank diagnostics and oracle.
        split_tables[split]['fixed_baselines'] = base_metrics

    audit = {label: runs[label]['result'].get('leakage_audit', {}) for label in sorted(expected)}
    selection = {
        label: runs[label]['result']['models'][runs[label]['result']['mode']]['dev_selection']
        for label in sorted(expected)}
    final = {'status': 'complete', 'fixed_rows': rows_by_split, 'methods': split_tables,
        'paired_episode_bootstrap_5000': paired, 'dev_selection': selection,
        'training_teacher_tau_grid': [.002,.005,.01,.02],
        'ranking_reference_tau_fixed': .01,
        'teacher_objective': 'q=softmax(exp(logit_scale)*similarity), p=softmax(-stored-unit arm6 MSE/training_tau), loss=-sum(p log q); logit_scale is learned, initialized at 1/.07 and capped at 100.',
        'hard_control': 'e_vlm_h-hard: one argmin-arm6-MSE positive per candidate set (hard InfoNCE)',
        'leakage_audit': audit, 'random_seeds': [4101, 4102, 4103, 4104, 4105]}
    (out / 'results.json').write_text(json.dumps(final, indent=2, allow_nan=False) + '\n')

    lines = ['# KEY2-F 結果集計', '', '## 主表：top-1（グリッパー正解）', '',
        '|データ|方式|行/難|MSE 全体|MSE 難|', '|---|---|---:|---:|---:|']
    for split in ('learned', 'unseen'):
        for label in (*INPUTS, *CONTROLS, 'e_vlm_h-hard'):
            x = split_tables[split][label]
            lines.append(f"|{split}|{label}|{x['n']}/{x['hard_n']}|"
                f"{_fmt(x['top1_actions_all'].get('mse_gripper_truth'))}|"
                f"{_fmt((x['top1_actions_hard'] or {}).get('mse_gripper_truth'))}|")
        b = split_tables[split]['fixed_baselines']
        for label, all_key, hard_key in (
            ('random-5','random5_actions_all','random5_actions_hard'),
            ('hold-last','baselines','baselines_hard'),
            ('zero','baselines','baselines_hard'),
            ('random-20 blend','baselines','baselines_hard'),
            ('π0.5','baselines','baselines_hard'),
            ('same-playback oracle','oracle_playback_all','oracle_playback_hard')):
            if label == 'random-5':
                av=b[all_key].get('mse_gripper_truth');hv=b[hard_key].get('mse_gripper_truth')
            elif label == 'same-playback oracle':
                av=b[all_key].get('mse_gripper_truth');hv=b[hard_key].get('mse_gripper_truth')
            else:
                key={'hold-last':'hold_last','π0.5':'pi05','random-20 blend':'random20_blend','zero':'zero'}[label]
                av=(b[all_key].get(key,{}).get('mse_gripper_truth'))
                hv=(b[hard_key].get(key,{}).get('mse_gripper_truth'))
            lines.append(f'|{split}|{label}|{rows_by_split[split]["n"]}/{rows_by_split[split]["hard_n"]}|{_fmt(av)}|{_fmt(hv)}|')
        dp=b['baselines'].get('DIM3-arm6-natural-DP-reference',{})
        dph=b['baselines_hard'].get('DIM3-arm6-natural-DP-reference',{})
        lines.append(f'|{split}|whole-column DP oracle (DIM3; reference)|{rows_by_split[split]["n"]}/{rows_by_split[split]["hard_n"]}|{_fmt(dp.get("arm6"))}|{_fmt(dph.get("arm6"))}|')

    lines += ['', '## Devで選んだ top-k blend（別途）', '',
        '|データ|方式|行/難|k / 温度|MSE 全体|MSE 難|','|---|---|---:|---:|---:|---:|']
    for split in ('learned','unseen'):
        for label in (*INPUTS,*CONTROLS,'e_vlm_h-hard'):
            x=split_tables[split][label];cfg=x['selected_blend']
            lines.append(f"|{split}|{label}|{x['n']}/{x['hard_n']}|{cfg['k']} / {_fmt(cfg['temperature'],3)}|{_fmt(x['actions_all'].get('mse_gripper_truth'))}|{_fmt((x['actions_hard'] or {}).get('mse_gripper_truth'))}|")
    lines += ['', '## 全候補の順位・分布指標（top-1 score、全体／難）', '',
        '|データ|方式|集合|Spearman|H(p,q)|KL(p‖q)|Recall@1/5/20|NDCG@20|MRR|Top1/Top20腕MSE|回収率|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('learned', 'unseen'):
        for label in (*INPUTS, *CONTROLS, 'e_vlm_h-hard'):
            x=split_tables[split][label]
            lines.append(_rank_line(split,label,'全体',x['rank_all']))
            if x['rank_hard']:
                lines.append(_rank_line(split,label,'難',x['rank_hard']))
        b=split_tables[split]['fixed_baselines']
        for label,key in (('random-5','random_rank'),('oracle','oracle_rank')):
            lines.append(_rank_line(split,label,'全体',b[key]))
            if b.get(key+'_hard'):
                lines.append(_rank_line(split,label,'難',b[key+'_hard']))

    lines += ['', '## top-k 付録：Devで選んだ温度を用いた k 別誤差', '', '|データ|方式|k|温度|MSE 全体|MSE 難|', '|---|---|---:|---:|---:|---:|']
    for split in ('learned', 'unseen'):
        for label in (*INPUTS, *CONTROLS, 'e_vlm_h-hard'):
            x=split_tables[split][label]
            for k,cfg in sorted(x['blend_by_k'].items(),key=lambda p:int(p[0])):
                lines.append(f"|{split}|{label}|{k}|{_fmt(cfg['temperature'],3)}|{_fmt(cfg['mse_gripper_truth'])}|{_fmt(cfg['hard_mse_gripper_truth'])}|")

    lines += ['', '## episode bootstrap 95%区間（予測 MSE 差、5,000回）', '', '差は左の条件 − 右の条件。負値は左が低誤差。']
    for split, comps in paired.items():
        lines.append(f'### {split}')
        for name, v in comps.items():
            allv=v['all'];hardv=v['hard']
            lines.append(f"- {name}: 全体 {allv['mean']:.5f} [{allv['ci95_low']:.5f}, {allv['ci95_high']:.5f}], 難 {hardv['mean']:.5f} [{hardv['ci95_low']:.5f}, {hardv['ci95_high']:.5f}]")
    lines += ['', '## 対照学習と入力・漏洩監査', '',
        'soft InfoNCE: 候補集合 C は同一ミニバッチの他 unit-start keys と一様抽出したメモリ単位。自エピソード候補を mask する。q_j=softmax(exp(logit_scale)·sim(E(q),E(k_j)))、p_j=softmax(−arm6_MSE_j/τ_train)、L=−Σ_j p_j log q_j。τ_train は Dev で {.002,.005,.01,.02} から選択。logit_scale は1/.07で初期化し100で上限。H/KL/NDCGの教師分布は全方式でτ=.01固定。hard controlは許容候補内の最小arm6 MSEのみを正例とするhard InfoNCE。',
        '', 'モデル入力は各入力条件ごとの現在VLM token/group、現在フレーム画像からのV-JEPA token/camera group、過去に実行済みの行動履歴とmask、現在stateの指定部分だけ。タスクID・エピソードID・未来行動・正解の続き・ラベル・正解境界は入力特徴に含めない。IDはメモリ側学習の同一エピソード候補除外と episode bootstrap のみ。評価窓の行動は順位教師、指標、oracleに限る。',
        '各方式で保存された leakage_audit を results.json に含めた。因果履歴テストは時刻 t 以後の行動を変更しても _episode_history の返す履歴・状態が不変で、将来教師だけが変わることを確認する。クエリパッカーの不変性テストは target actions、task/episode IDs、row/key、hard mask を書き換えてもクエリ埋め込みが不変であることを確認する。',
        '', '評価行は固定 learned 621 (難125) / unseen 619 (難134)。学習済みデモ行の選択は既存 CMP3 evaluation-selection.json / PI0REF selection.npz を再利用し、設定選択は Dev のみ。top-k は自然長10-step再生行動を softmax 平均する。正解を用いた閉ループ・評価は実施していない。']
    (out / 'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
