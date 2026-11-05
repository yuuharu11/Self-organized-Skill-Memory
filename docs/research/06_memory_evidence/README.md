# 06 発表用のスキルメモリ検証

## 問い

03 のスキルメモリ実験（軽量 FP）の結果を発表に使えるか。足りない対照を補い、検索がどこで外れているかを示す。新しい方策・検索器の学習はしていない（既存の重みとキャッシュから再計算）。

## 結論

- **発表に必要な材料は揃った。ただし「スキルメモリで実際のタスク成功率を改善した」とは言えない。**
- 行動 MSE（保留 spatial）：通常 0.2350 → 同じ FP で運動から開始 0.1774 → 運動の経路も学習 0.1666。既知 goal でも 0.0970 → 0.0552。複数タスク・seed の集計で、1 タスクの結果ではない。
- 検索：保留 spatial で厳密な top-1 一致 8.8%、top-20 包含 71.4%。既知タスクより類似度も重みの集中度も低く、最上位と次点の差も小さい。
- VLM 検索特徴は、行動の近さに十分対応した距離になっていない。運動単位ごとのキーを平均する前から、ずれが残る。
- 通常 FP とメモリ経路を信頼度で切り替えるルーターは、常にメモリを使う場合を上回らなかった。

## ファイル

| ファイル | 中身 |
|---|---|
| [readiness.md](readiness.md) | **最初に読む**。発表に向けた充足性の判断と最終補完（スライドの流れ、対照、MSE の意味） |
| [offline_diagnostics.md](offline_diagnostics.md) | goal 単独の集計と、二経路ルーターの評価 |
| [task_significance.md](task_significance.md) | メモリ比較のタスク数と探索的 p 値 |
| [retrieval_confidence.md](retrieval_confidence.md) | 検索類似度・重みの集中度・候補間の差の分布（図あり） |
| [vlm_action_geometry.md](vlm_action_geometry.md) | 同じ行動を必要とする入力は VLM 空間でも近いか（図あり） |
| [evidence/](evidence/) | 集計 JSON と図（PNG / SVG） |

集計スクリプトは `scripts/analyze_memory_*.py`、`scripts/plot_memory_retrieval_confidence.py`、`scripts/analyze_vlm_action_geometry.py`。
