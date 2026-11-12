"""Render the RM-EXP3 experiment record from saved stage artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/work/.skillnet-state/outputs/robocasa-memory/expert-03')


def main():
    prep_path = ROOT / 'prep/drafts/summary.json'
    train_path = ROOT / 'train/expert/train_summary.json'
    val_path = ROOT / 'train/expert/validation_grid.json'
    for p in (prep_path, train_path, val_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    prep, train, val = [json.loads(p.read_text()) for p in (prep_path, train_path, val_path)]
    if train['status'] != 'complete' or train['latest_step'] != 3000:
        raise RuntimeError(f"training is not complete: {train['status']} at {train['latest_step']}")
    if val['status'] != 'complete':
        raise RuntimeError(f"validation is not complete: {val['status']}")

    lines = [
        '# RM-EXP3: magnitude_v2 下書き＋可変 t0 Expert', '',
        '実装・学習・固定検証を完了。閉ループ評価は未実施。', '',
        '## 条件', '',
        f"- 下書き: {prep['rows']:,} 行（学習対象 {prep['train_rows']:,}、expert検証用に確保された区間 {prep['validation_rows']:,}）。固定検証300窓はこの確保分から各タスク50窓ずつ選択。",
        f"- magnitude head: `{prep['magnitude_head']}`。履歴を使わない prefix＋state＋選択単位メタデータ。選択器は Context+history Top-1、同一エピソード除外を維持。",
        '- 下書きは各50-step区間を逐次連結。生成した行動を次の検索履歴へ入れる。scale は exp(log ratio) を [.25, 4] に制限し、action storage channels 5:11 に適用。',
        f"- 初期検索の固定ランカーとの一致: `{prep['initial_selection_matches_fixed_ranker']}`。",
        '- 学習初期値: π0.5 RoboCasa 75k checkpoint。VLM凍結、action Expert部分のみ更新。',
        '- 更新: 3,000、AdamW lr=2.5e-5 (b1=.9, b2=.95, wd=1e-5)、global-norm clip 1。実効batch 32。',
        '- t0: 各学習例ごと独立に U[.2,.5) でサンプル。bridge 時刻は t=t0×progress、x=(1-progress)×target+progress×((1-t0)×draft+t0×noise)。',
        f"- 実測学習更新時間: {train['training_seconds']/60:.1f} 分。チェックポイントは500更新ごと。GPU割当上限: {train['memory_fraction']:.2f}。", '',
        '## 固定300窓でのチェックポイント比較', '',
        f"評価指標は各窓の先頭5 action steps × active6 の正規化MSE。noise seed=4101をrow_idでfold-inし、10-step積分。検証窓数={val['rows']}。選択規則は各t0で先頭5 MSE6最小。", '',
        '| checkpoint | 更新 | t0 | 先頭5 MSE6 | unit MSE6 |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in val['results']:
        step = int(r['checkpoint'].split('_')[1].split('.')[0])
        lines.append(f"| `{r['checkpoint']}` | {step} | {r['t0']:.1f} | {r['first5_mse6']:.6f} | {r['unit_mse6']:.6f} |")
    lines += ['', '### 各 t0 の最良checkpoint', '', '| t0 | checkpoint | 先頭5 MSE6 | unit MSE6 |', '|---:|---|---:|---:|']
    for t0 in (.2, .3, .4, .5):
        r = val['best_by_t0'][str(t0)]
        lines.append(f"| {t0:.1f} | `{r['checkpoint']}` | {r['first5_mse6']:.6f} | {r['unit_mse6']:.6f} |")
    lines += [
        '', '## 実行上の注記', '',
        '- 最初の microbatch=8 起動は初期コンパイルでGPUメモリ不足となり、500更新前に停止したため重みは採用していない。保存済みの前処理データを再利用し、初期75k重みからmicrobatch=4で再実行した。実効batch 32、optimizer、学習例順、seedは維持。',
        '- microbatch=4 実行では500更新ごとにモデルとoptimizer stateを保存し、再開用状態も残した。',
        '- validation は学習後 checkpoint をすべて各t0で評価した。モデル選択は固定300検証窓だけに基づく。',
        '- magnitude_v2 は既存 head を再学習せず利用。入力互換adapterは旧生成関数の履歴欄を identity-normalized にして除去し、同一サンプルで直接正規化入力との最大絶対差0を確認。',
        '- 本結果はオフライン検証であり、成功率や閉ループでの改善を主張しない。',
        '',
    ]
    out = ROOT / 'report.md'
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
