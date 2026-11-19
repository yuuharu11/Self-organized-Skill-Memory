#!/usr/bin/env python3
"""Write closed-loop-kq-01/report.md from results.json (CL1)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[5]
STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', REPO_ROOT / '.skillnet-state')).expanduser()

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else STATE_DIR / 'outputs/memory-pilot/unseen-libero/closed-loop-kq-01')
r = json.loads((OUT / 'results.json').read_text())
C = ['C1a', 'C1b', 'C2', 'C3', 'pi05']
cond = r['conditions']
L = []
L.append('# CL1：運動キー＋Expert の閉ループ（LIBERO-90 Eval 10タスク×10試行）\n')
L.append(f"状態：**{r['status']}**（完了試行数 " + '、'.join(f"{c} {cond[c]['completed']}" for c in C if c in cond) + '）\n')
L.append('## 主表：成功数\n')
L.append('|条件|内容|成功/完了|')
L.append('|---|---|---:|')
for c in C:
    if c in cond:
        L.append(f"|{c}|{cond[c]['label']}|{cond[c]['successes']}/{cond[c]['completed']}|")
L.append('\n## タスク別成功数（/10）\n')
tasks = list(cond['pi05']['tasks'].keys())
L.append('|task|名前|' + '|'.join(c for c in C if c in cond) + '|')
L.append('|---|---|' + '|'.join('---:' for c in C if c in cond) + '|')
for t in tasks:
    L.append(f"|{t}|{cond['pi05']['tasks'][t]['name'][:60]}|" + '|'.join(
        f"{cond[c]['tasks'][t]['successes']}" + ('' if cond[c]['tasks'][t]['completed'] == 10 else f"({cond[c]['tasks'][t]['completed']})")
        for c in C if c in cond) + '|')
L.append('\n## McNemar（同じ試行の対、正確二項検定）\n')
L.append('|比較 B − A|対|B だけ成功|A だけ成功|p|')
L.append('|---|---:|---:|---:|---:|')
for k, v in r['mcnemar'].items():
    L.append(f"|{k}|{v['pairs']}|{v['b_only']}|{v['a_only']}|{v['p']:.3f}|")
L.append('\n## 信頼度・t0・単位の出どころ\n')
L.append('t0 = clip(√ê, .05, .6)（ê＝信頼度器の予測腕誤差）。C2 は Expert を適用しないが、同じ ê と t0 を記録した。出どころは TR1 の分類（未学習タスクの指示と LIBERO-40 の元タスクの指示で動詞が重なるか）。\n')
L.append('|条件|問い合わせ数|t0 中央値 [5%, 95%]|t0=下限 .05 の割合|t0=上限 .6 の割合|ê 中央値|同動詞の単位|別動詞の単位|')
L.append('|---|---:|---|---:|---:|---:|---:|---:|')
D = r['diagnostics']
for c in ['C1a', 'C1b', 'C2', 'C3']:
    if c not in D:
        continue
    d = D[c]; q = d['t0_quantiles']; sc = d['source_categories']
    same = sum(v for k, v in sc.items() if k not in ('other_verb', 'unknown'))
    L.append(f"|{c}|{d['queries']}|{q['0.5']:.3f} [{q['0.05']:.3f}, {q['0.95']:.3f}]|{d['t0_at_floor_fraction']:.3f}|{d['t0_at_ceiling_fraction']:.3f}|"
             f"{d['pred_error_quantiles']['0.5']:.4f}|{same:.3f}|{sc.get('other_verb', 0):.3f}|")
L.append('\n成功試行と失敗試行の比較（試行平均）\n')
L.append('|条件|群|試行|平均 t0|平均 ê|切り替え回数|切り替え/問い合わせ|異なる単位数|同動詞率|別動詞率|')
L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for c in ['C1a', 'C1b', 'C2', 'C3']:
    if c not in D:
        continue
    for g, name in (('success', '成功'), ('failure', '失敗')):
        x = D[c][g]
        if x:
            L.append(f"|{c}|{name}|{x['trials']}|{x['mean_t0']:.3f}|{x['mean_pred_error']:.4f}|{x['switches']:.1f}|{x['switches_per_query']:.3f}|"
                     f"{x['distinct_units']:.1f}|{x['same_verb_fraction']:.3f}|{x['other_verb_fraction']:.3f}|")
L.append('\n## オフライン未学習 MSE との順位\n')
for k, v in r['offline_unseen_top1_arm6_mse'].items():
    L.append(f'- {k}: {v}')
L.append('\n## 設定\n')
s = r['setting']; sch = r.get('schedule') or {}
L.append(f"- LIBERO-90 Eval {s['tasks']}、各10試行、seed_base {s['seed_base']}、公開初期状態 0..9、replan {s['replan_steps']}、wait {s['wait_steps']}、最大 {s['max_steps']} step（lib-closed-01 / EX1 / EX2 と同じ）。π0.5 は baseline-01 の 32/100 を再掲（同じ runner・同じ seed・同じ初期状態）。")
if sch:
    e = sch['expert']
    L.append(f"- Expert：EX2 修正版 production-05 の A、checkpoint step {e['step']}（{e['checkpoint']}）。学習時の Dev 曲線は module_jit で重みが固定されたまま計算されており（全 step 同値）、EX2 の修正済み再採点（repair_dev_checkpoint_selection、Dev 217 行・D1 下書き、Eval 不使用）で選び直した。再学習はしていない。")
    L.append(f"- t0 規則：{sch['schedule']['rule']}。グリッパーは π0.5 の通常出力。")
L.append('- 下書き：C1a/C2 は best/Afix_x1_fixed_e_h_cteff.pt（seed 1）、C1b は best/H_x1_mk1b_cteff_tau0.01_s1.pt、キーは final/data/x1_keys.npy。C3 は KEYGRID production-04 の履歴＋状態 MLP（EX2 の D5）。候補は LIBERO-40 の 4,516 単位（EX2 と同じ並び）。')
L.append('- パリティ：オフライン（parity_offline.json）でサーバーと同じ経路のスコアが KQ-final export の top-1 と未学習 619 行・Dev 217 行で 100% 一致。閉ループ smoke（parity_inloop.json）で、ループ内に記録した入力から再計算した top-1 が全問い合わせで一致。')
L.append('- コード：unseen_libero/run_cl1_closed_loop.py（実行・集計）、serve_expert_ex2.py（KQ_* モードと --no-expert を追加）、report_cl1.py。')
(OUT / 'report.md').write_text('\n'.join(L) + '\n')
print('wrote', OUT / 'report.md')
