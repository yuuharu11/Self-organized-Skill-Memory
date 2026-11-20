#!/usr/bin/env python3
"""KQ2-CL report: report.md (prose from report_prose.md + generated tables) and results.json."""
from __future__ import annotations

import json

import numpy as np

import kq2cl_common as C

J = lambda n: C.read_json(C.OUT / n, {})


def f4(x):
    return '—' if x is None else f'{x:.4f}'


def ci(v):
    return f"[{v[0]:.4f}, {v[1]:.4f}]"


def main() -> None:
    meas, prep, sel, heads, pol, off, cl, par, obs_par = (J('measure.json'), J('expert/prepare.json'), J('expert/select.json'),
                                                          J('heads/heads.json'), J('policy.json'), J('offline.json'),
                                                          J('closed_loop.json'), J('parity_inloop.json'), J('obs/parity.json'))
    train = J('expert/train-K1/train.json')
    L = ['# KQ2-CL：(iii) GRU×VLM クエリと (i) GRU クエリの閉ループ比較（Expert・t0 ヘッド・グリッパーデコーダ込み）', '']
    prose = C.OUT / 'report_prose.md'
    if prose.exists():
        L += [prose.read_text().strip(), '']
    # ---------------- closed loop main tables (one per suite)
    if cl:
        desc = {'A': '(iii) 提案：下書き→Expert、t0 ヘッド、グリッパーデコーダ、ê>b は π0.5',
                'B': '(i) VLM なし GRU：同じ Expert・同じ方式のヘッド',
                'Aconst': 'A の t0 を定数に（同じ ê・同じフォールバック b）', 'pi05': 'π0.5 単独',
                'Bconst': '（追加の切り分け）B の t0 を A-const と同じ定数に（(i) の ê・(i) の b）'}
        title = {'libero90': 'LIBERO-90 Eval（未学習）10 タスク × 10 試行', 'libero10': 'LIBERO-10（学習済み）10 タスク × 10 試行'}
        pinote = {'libero90': 'π0.5 は baseline-01（CL1/EX2 と同じ 10×10、再掲）。',
                  'libero10': 'π0.5 は試行 0〜4 が既存（closed-loop-memory-01、EX2）、試行 5〜9 を今回同じ設定で追加。'}
        for suite, S in cl['suites'].items():
            cond = S['conditions']
            L += [f"## 主表：{title[suite]}", '', 'CL1 と同じ初期状態（試行 i = 公開初期状態 i、env seed 7+i）、wait 10、replan 5、最大 520 step。' + pinote[suite] +
                  ' McNemar は同じ試行の対の正確二項検定（B だけ成功／A だけ成功）。', '',
                  '|条件|内容|成功/100|対 π0.5（条件だけ/π0.5 だけ, p）|', '|---|---|---:|---|']
            for n in ('A', 'B', 'Aconst', 'Bconst', 'pi05'):
                if n in cond:
                    c = cond[n]; m = S['mcnemar'].get(f'{n} - pi05')
                    ms = f"{m['b_only']}/{m['a_only']}, p={m['p']:.3f}" if m else '—'
                    L.append(f"|{n}|{desc[n]}|{c['successes']}/{c['completed']}|{ms}|")
            if S.get('reference'):
                L += ['', '参考（既存の閉ループ、同じ 100 試行）：' + '、'.join(f"{k} {v['successes']}/{v['completed']}" for k, v in S['reference'].items())]
            names = [n for n in ('A', 'B', 'Aconst', 'Bconst', 'pi05') if n in cond] + list(S.get('reference', {}))
            L += ['', '|task|名前|' + '|'.join(names) + '|', '|---|---|' + '---:|' * len(names)]
            for t in S['tasks']:
                k = str(t['task_id'])
                L.append(f"|{k}|{t['name'][:48]}|" + '|'.join(str((cond.get(n) or S['reference'][n])['tasks'][k]) for n in names) + '|')
            L += ['', '|比較 B − A|対|B だけ成功|A だけ成功|p|', '|---|---:|---:|---:|---:|']
            for k, v in S['mcnemar'].items():
                L.append(f"|{k}|{v['pairs']}|{v['b_only']}|{v['a_only']}|{v['p']:.3f}|")
            L += ['', '|条件|問い合わせ|フォールバック率|t0（非フォールバック）中央値 [5%, 95%]|ê 中央値|単位の切替率|試行あたりグリッパー切替|', '|---|---:|---:|---|---:|---:|---:|']
            for n, d in S.get('diagnostics', {}).items():
                if 'queries' not in d:
                    L.append(f"|{n}|—|—|—|—|—|{d['gripper_switches_per_trial']:.2f}|"); continue
                t = d['t0_non_fallback']
                ts = f"{t['0.5']:.3f} [{t['0.05']:.3f}, {t['0.95']:.3f}]" if t else '—'
                L.append(f"|{n}|{d['queries']}|{d['fallback_rate']:.3f}|{ts}|{d['ehat']['0.5']:.4f}|{d['unit_switch_rate']:.3f}|{d['gripper_switches_per_trial']:.2f}|")
            L += ['']
            for n, d in S.get('diagnostics', {}).items():
                if 'success_vs_failure' in d:
                    L.append(f"- {n} 成功／失敗の試行平均：" + '、'.join(f"{k} {f4(v['success'])}／{f4(v['failure'])}" for k, v in d['success_vs_failure'].items()))
                    L.append('  - タスク別フォールバック率：' + '、'.join(f"t{k} {v:.2f}" for k, v in d['fallback_rate_by_task'].items() if v is not None))
            if S.get('pi05_parity'):
                L.append('- π0.5 経路のパリティ：' + '、'.join(f"{k} {v['pairs']} 試行で成功一致 {v['success_equal']}・env step 一致 {v['env_steps_equal']}" for k, v in S['pi05_parity'].items()))
            if S.get('error_rows_rerun'):
                L.append('- エラーで再実行した試行行：' + '、'.join(f"{k} {v['error_rows']}（OOM {v['oom_rows']}）" for k, v in S['error_rows_rerun'].items()))
            L.append('')
    # ---------------- 1. rank law
    if meas:
        L += ['## 1. 検索の崩れ方（top-1 の真の順位と誤差）', '',
              '真の順位＝候補（各行の除外規則のもとで）を腕6軸 MSE で並べたときの top-1 の順位。insE/insT＝本番クエリで自エピソード/自タスク除外、oofE/oofT＝タスク5分割で学習し直したクエリ（そのタスクを見ていない）で自エピソード/自タスク除外。', '',
              '|クエリ|集合|行|順位 中央値 [25%, 75%, 90%]|recall@1/10/100|top-1 腕 MSE 平均|oracle|Dev v2 との KS（log順位/MSE）|', '|---|---|---:|---|---|---:|---:|---|']
        for k, v in meas.items():
            q, s = k.split(':'); r = v['rank']; ks = v.get('ks_vs_dev2')
            L.append(f"|{q}|{s}|{v['rows']}|{r['0.5']:.0f} [{r['0.25']:.0f}, {r['0.75']:.0f}, {r['0.9']:.0f}]|{v['recall@1']:.3f}/{v['recall@10']:.3f}/{v['recall@100']:.3f}|"
                     f"{v['top1_arm6_mean']:.4f}|{v['oracle_arm6_mean']:.4f}|{'—' if not ks else '{:.2f}/{:.2f}'.format(ks['log_rank'], ks['arm6'])}|")
        L.append('')
    if prep:
        lw = prep['law']
        L += ['## 2. スキル Expert の再学習（EX3 方式、キー非依存、腕6軸のみ）', '',
              f"- 下書きの順位則：{lw['pools'][0]}（重み {lw['weights'][0]}）と {lw['pools'][1]}（重み {lw['weights'][1]}）の混合を {1 - lw['near_fraction'] - lw['uniform_fraction']:.0%}、上位50の幾何（p=.16）を {lw['near_fraction']:.0%}、全候補一様を {lw['uniform_fraction']:.0%}。順位は自タスクを除いた単位の腕 MSE 順（EX3 の order をそのまま使用）。",
              f"- 抽選した下書きの順位 中央値 {prep['preview_rank']['0.5']:.0f}（Dev v2 の (iii) は {prep['measured_rank_dev2']['0.5']:.0f}）、腕 MSE 中央値 {prep['preview_arm6']['0.5']:.4f}（Dev v2 {prep['measured_arm6_dev2']['0.5']:.4f}）、KS（log順位/MSE）{prep['ks_preview_vs_dev2']['log_rank']:.2f}/{prep['ks_preview_vs_dev2']['arm6']:.2f}。学習時 t0 中央値 {prep['preview_t0_train']['0.5']:.3f}、上限 .6 の割合 {prep['preview_t0_at_cap']:.2f}。",
              f"- 学習：{train.get('updates')} 更新、batch {train.get('batch')}、{train.get('optimizer')}、t0 = {train.get('t0_rule_train')}、状態 {train.get('status')}。", '']
    if sel.get('scores'):
        sc = list(next(iter(sel['scores'].values()))['dev_arm6_by_scale'])
        L += ['### Dev v2 20 Hz での checkpoint 選択（(iii)+(i) の top-1 下書き、t0 = clip(c·√真の誤差, .05, .6)）', '',
              '|checkpoint|' + '|'.join(f'c={c}' for c in sc) + '|', '|---|' + '---:|' * len(sc)]
        for k, v in sel['scores'].items():
            L.append(f'|{k}|' + '|'.join(f4(v['dev_arm6_by_scale'][c]) for c in sc) + '|')
        if 'selected' in sel:
            L += ['', f"選択：{sel['selected']['label']}（c={sel['selected']['scale']}、Dev {sel['selected']['dev_arm6']:.4f}）。EX3 G2:4000 は c={sel['ex3_reference']['scale']} で {sel['ex3_reference']['dev_arm6']:.4f}。Dev 下書き：" +
                  '、'.join(f"{q} {v:.4f}" for q, v in sel['draft_dev_arm6'].items()) + '。', '']
    # ---------------- 3. heads
    if heads:
        L += ['## 3. t0 ヘッド（キーとクエリ由来の量だけ）', '',
              '入力：scalar＝top-1 logit、top-2/5/10/50 との差、正規化エントロピー、top-1 確率、top-5 質量、内積（top-1、差）、top-5 キーの広がり（12 次元）。full＝scalar＋クエリ q＋top-1 キー k。目標は top-1 下書きの log 腕 MSE。選択は Dev v2 の Spearman。', '',
              '|クエリ|学習行×入力|Dev v2 Spearman|学習済み|未学習20Hz|Dev 予測/真 中央値|未学習 予測/真 中央値|', '|---|---|---:|---:|---:|---|---|']
        for q, rec in heads.items():
            for name, r in rec['confidence'].items():
                mark = ' **（選択）**' if name == rec.get('selected', {}).get('confidence') else ''
                L.append(f"|{q}|{name}{mark}|{r['dev2']['spearman']:.3f}|{r['learned']['spearman']:.3f}|{r['unseen20']['spearman']:.3f}|"
                         f"{r['dev2']['pred_median']:.4f}/{r['dev2']['true_median']:.4f}|{r['unseen20']['pred_median']:.4f}/{r['unseen20']['true_median']:.4f}|")
        L.append('')
    if pol:
        L += ['### t0 写像とフォールバック閾値 b（Dev v2）', '',
              f"t0 = clip(c·√ê, lo, hi)、ê > b は π0.5。b は「タスクを抜いて学習したクエリで LIBERO-40 行を検索したときの ê（eh_oofT）の 90% 点」（検索器が学習分布の 9 割より自信がない）とした。オフライン MSE だけで b を選ぶと、π0.5 の Dev 腕 MSE（{pol['dev_pi05_arm6']:.4f}）が下書きよりずっと悪いため常に b=∞（フォールバックなし）になる（下の表）。(c, lo, hi) はその b のもとで Dev 腕 MSE 最小。", '',
              '|クエリ|規則|Dev フォールバック率|b|c|lo|hi|Dev 腕 MSE|定数 .15（同じ b）|定数 .2（同じ b）|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for q, d in pol['queries'].items():
            r = d['selected']
            L.append(f"|{q}|**採用：ê の 90% 点**|{r['dev_fallback_rate']:.3f}|{r['b']:.4f}|{r['c']}|{r['lo']}|{r['hi']}|{r['dev_arm6']:.4f}|{r['const']['0.15']:.4f}|{r['const']['0.2']:.4f}|")
            for r in d['by_fallback_rate']:
                L.append(f"|{q}|参考：Dev 率 {r['fallback_rate_target']:.2f} 固定|{r['dev_fallback_rate']:.3f}|{r['b']:.4f}|{r['c']}|{r['lo']}|{r['hi']}|{r['dev_arm6']:.4f}|{r['const']['0.15']:.4f}|{r['const']['0.2']:.4f}|")
            L.append(f"|{q}|A-const／B-const の定数 t0（.15 と .2 のうち同じ b で Dev が良い方）|—|—|—|—|—|{d['const_t0_selected']}|—|—|")
        L.append('')
    # ---------------- 4. gripper
    if heads:
        L += ['## 4. グリッパーデコーダ', '',
              '入力は t0 ヘッドと同じ量＋実行中のグリッパー（直前に実行した指令、履歴の有無、グリッパー qpos）。出力は次の 10 ステップの閉確率。直前の実行値から、反対側の確率が閾値を超えた最初のステップで1回だけ切り替える。選択は Dev v2 の（見逃し＋誤切替）行数。', '']
        for q, rec in heads.items():
            s = rec.get('selected', {})
            if s:
                L.append(f"- {q}：{s['gripper']}、閾値 {s['gripper_th']}（Dev の切替行 {s['dev_gt_switch_rows']}）")
        L.append('')
    if off:
        L += ['### 切替時刻の評価（直前に実行したグリッパーからの最初の符号変化、10 ステップ）', '',
              '|分割|方式|切替行|見逃し率|誤切替率|時刻ずれ（step）|±1 step 以内|', '|---|---|---:|---:|---:|---:|---:|']
        for split, sp in off['splits'].items():
            for k, m in sp['gripper'].items():
                L.append(f"|{split}|{k}|{m['gt_switch_rows']}|{f4(m['missed_switch_rate'])}|{f4(m['false_switch_rate'])}|{f4(m['timing_error_mean_steps'])}|{f4(m['within1_step_rate'])}|")
        L += ['', 'EX3_expert@t0.2＝EX3 Expert（G2:4000）にそのクエリの下書きを t0=.2 で入れたときのグリッパー出力の符号。unit_stored＝top-1 単位に保存されたグリッパー値。', '']
    # ---------------- 5. offline
    if off:
        L += ['## 5. オフライン確認（腕6軸 MSE、[ ] は episode bootstrap 95%）', '',
              'K＝今回の Expert、EX3＝EX3 の Expert（G2:4000）。head_t0＝t0 ヘッドの ê から Dev で選んだ写像、+fallback＝ê > b の行は π0.5。true_err_t0＝真の下書き誤差から（Dev で選んだ c）。best_t0_per_row＝t0 格子で行ごとの最良（上限）。', '',
              '|行|Dev v2（667）|学習済み（621）|未学習 20Hz（619）|', '|---|---|---|---|']
        names = list(off['splits']['unseen20']['arm6'])
        for k in names:
            L.append(f'|{k}|' + '|'.join(f"{off['splits'][s]['arm6'][k]['arm6']:.4f} {ci(off['splits'][s]['arm6'][k]['ci95'])}" for s in ('dev2', 'learned', 'unseen20')) + '|')
        L += ['', '対の差（未学習 20 Hz）：', '']
        for k, v in off['splits']['unseen20']['paired'].items():
            L.append(f"- {k}: {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        L += ['', '信頼度（ê と真の下書き誤差）と t0：', '', '|分割|クエリ|Spearman|予測中央値/真の中央値|幾何平均の比|フォールバック率|t0 中央値 [5%, 95%]（非フォールバック）|', '|---|---|---:|---|---:|---:|---|']
        for split, sp in off['splits'].items():
            for q in ('iii', 'i'):
                e = sp['extra'][f'{q}:confidence']; h = sp['extra'][f'{q}:K:head_t0+fallback']; t = h['t0']
                L.append(f"|{split}|{q}|{e['spearman']:.3f}|{e['pred_median']:.4f}/{e['true_median']:.4f}|{e['geo_ratio']:.2f}|{h['fallback_rate']:.3f}|"
                         f"{(t['0.5'] if t else float('nan')):.3f} [{(t['0.05'] if t else float('nan')):.3f}, {(t['0.95'] if t else float('nan')):.3f}]|")
        L.append('')
    # ---------------- parity
    L += ['## 6. 入力パリティ', '']
    if obs_par:
        L.append('- オフラインの観測行（kq2cl_obs）から π0.5 の VLM トークンを計算し直すと、KQ2 のトークン（vq-b-01/feats）と一致：' +
                 '、'.join(f"{k} 最小 cos {v['vlm_img_min_cos']:.6f}・最大差 {v['vlm_img_max_abs']:.2g}" for k, v in obs_par.items()))
    if par:
        for k, v in par.items():
            L.append(f"- 閉ループ smoke の {k}（{v['queries']} 問い合わせ）：サーバーが作った履歴トークンと、オフラインの GPUSplit.window で作り直したトークンの最大差 {v['token_max_abs_diff']:.2g}、"
                     f"VLM トークン最大差 {v['vlm_max_abs_diff'] if v['vlm_max_abs_diff'] is not None else '—'}、q の最大差 {v['q_max_abs_diff']:.2g}、top-1 一致率 {v['top1_same_fraction']:.3f}、ê 相対差 最大 {v['ehat_rel_diff_max']:.2g}")
    L.append('')
    text = '\n'.join(L) + '\n'
    (C.OUT / 'report.md').write_text(text)
    C.write_json(C.OUT / 'results.json', {'measure': meas, 'expert_prepare': prep, 'expert_train': train,
                                          'expert_select': {k: v for k, v in sel.items() if k != 'scores'} | {'scores': {k: v['dev_arm6_by_scale'] for k, v in sel.get('scores', {}).items()}},
                                          'heads': heads, 'policy': pol, 'offline': {s: {k: v for k, v in sp.items()} for s, sp in off.get('splits', {}).items()},
                                          'closed_loop': cl, 'parity_inloop': par, 'parity_obs': obs_par})
    print(text[:3000])


if __name__ == '__main__':
    main()
