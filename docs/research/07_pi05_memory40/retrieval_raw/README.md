# 生行動に近いメモリ行を検索できるか

## 問い

実行時に得られる画像・状態・短い履歴から、π0.5が難しい query に合う生行動の実演を検索できるか。

## 結論

手首・状態のキーでTop-64内の生行動候補を見つける率は現行VLMより上がったが、hard行の生成後誤差は多くの方式でゼロ下書きを下回らなかった。詳細は `results.md`。

## ファイル

- [plan.md](plan.md)：固定条件、入力、分割、指標。
- [results.md](results.md)：Eval結果と解釈。
- `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/retrieval-4b-01/`：Top-8候補、学習キー、最終行動予測、機械可読集計。
