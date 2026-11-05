# π0.5 memory retrieval correspondence audit

記録開始: 2026-11-04

## 問い

正規化された実演 action chunk の近さを基準に、観測時に利用可能な検索キーが対応する経験を Top-K へ入れられるかを測る。検索キーの順位付けと、M1 codec による復元下書きの誤差を分ける。公開 SkillNet はデータに既習のため、未知タスク汎化を主張しない。

背景となる先行調査は [VLM と行動近傍の対応](../../06_memory_evidence/vlm_action_geometry.md)。

## データと分割

- 入力: `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01`。読み取り専用。
- クエリ: `rows['validation']` の3,277行。Dev/Eval episode は `pi05_memory/run.py::calibration_episodes` に従う。
- メモリ: `memory_row_ids.npy` の6,583行。
- 行動対応: `rows['actions'][q, :, :7]` と候補メモリの同じテンソルの normalized active-7 MSE。
- 正解集合: 候補除外後に action MSE ≤ τ（τ=.005/.01/.02）。同一タスク除外を主条件、同一episode除外を副条件とする。
- 検索時に真の行動・oracle距離は使わない。行動はoracle評価と、別欄のcodec draft MSEのみに使う。

## キー比較

VLM由来: 現行 grouped token MaxSim（base/wrist/instruction）、各group単独、camera-only、上位10/25% token mean、75 percentile、memory-centroid rarity weighting、memory-only centered token、sampled groupwise diagonal whitening、pooled reference。

非VLM由来: state、直前行動、codec latent/priors、π0.5予測行動、学習済み対照キー。作成・選択・評価の実装担当は担当間で分担する。

## 指標

Oracle候補存在率、Top-1 hit、候補存在時のRecall@10/@64、Top-1 action MSE（平均・中央値）、Top-1 codec-draft MSE、query内 Spearman(score, action MSE)、AUROC(score → Top-1 oracle hit at τ=.01)。τ別・Dev/Eval別・task別を保存し、query episode単位 bootstrap 95%区間を付ける。Top-64内のbest/median/uniform-random action MSEと最良oracle候補の順位も記録する。

## 共通キャッシュ

キー配列は `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/score-audit-01/keys/<method>.npz` の共通形式。VLM smoke は専用の `smoke-block256/keys/` に置き、全件データと混同しない。形式・再現コマンドは同 output の README に記載。

## 実行状況と完了条件

- 全3,277 queryの action-only oracle は作成済み。
- 非VLM側は `score-audit-01/nonvlm-full/` に全件の評価表がある。
- VLM側は全3,277 query、block=256で12キー方式×task/episode除外を作成した。既存 `retrieved_row.npy` と現行キーのTop-1は同一episode条件で全行一致し、スコア差は0。CUDA peak allocatedは約18.1 GiB。
- 共通 `_metrics` でEval 1,318行を再集計し、既存 `nonvlm-full` の重複22手法×除外条件はτ=.01のTop-1 action MSEで完全一致した。Dev由来のπ0.5予測誤差q75/q90に基づくEval部分集合も追加した。
- 統合結果、検索なし基準、Eval episode bootstrap CI、高誤差部分集合は [results.md](results.md) と `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/score-audit-01/integrated-full/` に記録した。Q=64 smoke値と全件値は分けて扱う。
