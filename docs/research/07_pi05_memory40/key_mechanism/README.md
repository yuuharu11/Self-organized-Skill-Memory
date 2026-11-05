# Key mechanism: attention-based memory selection

## 問い

局所的な視覚トークンへ状態・履歴に応じて注目する選択器が、Long以外の実演メモリからLongの次の10-step行動を単一候補として選べるかを評価する。

## 結論

M2の学習attention+MaxSimは局所キーよりTop-1 MSEを下げ、K1を満たした。一方、直接回帰より誤差が高く、全Longでhold-lastにも勝たず、凍結Expert後にzero draftを有意に上回る条件はなかった（K2–K4不合格）。ここでのLongはπ0.5の事前学習では既知であり、キー・メモリ学習に対する未見タスク条件である。オフライン誤差を閉ループ成功率とは解釈しない。詳細は [results.md](results.md)。

## ファイル

- [plan.md](plan.md): データ分割、手法、指標、K1–K4の事前基準
- [results.md](results.md): 本番の手法別比較と結論
- コード: [`key_mechanism/`](../../../../skill_moe/skillnet/examples/memory_pilot/pi05_memory/key_mechanism/)
- 実行結果: `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/key-mechanism-01/`
