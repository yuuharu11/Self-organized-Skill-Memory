# 段階4-B：生行動に近い候補を実行時情報で検索できるか

実施日：2026-11-04。入力キャッシュは `run-01/` と `score-audit-01/` の読み取り専用。結果は固定されたLIBERO-40 validation queryに限り、閉ループ成功率への主張ではない。

## 結論

VLMの現行MaxSimは生行動の近さを十分に上位へ運べなかった。一方、手首画像単独は現行VLMよりRecallが高く、手首＋状態から学習した対照キーはRecall@8/@64をさらに上げた。ただし「候補がTop-64にある」ことは「Top-1が実行に良い」ことを意味せず、最終MSEは多くの方式でゼロ下書きを下回らなかった。π0.5予測行動キーとcodec prior b̂は高いRecallだったが、前者は方策予測依存、後者は既存圧縮器の予測を使う参考キーである。

## 固定条件

- query 3,277 行：Dev 1,959 / Eval 1,318。メモリ 6,583 行。候補正解はoracle action MSE ≤ τ=0.005,0.01,0.02。主条件は同task除外、副条件は同episode除外。
- Recall@kは、oracle正例が1件以上あるqueryについて、Top-kに入った正例数 / 許可された正例総数を平均。Top-1 hit（正例のどれかが1位）は別に記載。
- hard subsetはmemory-gapと同一のDev通常π0.5 episode平均誤差 q80 = 0.015494をEvalに適用し、141行。
- wrist特徴は保存済みトークンのwrist group平均。状態はstate[:8]。履歴は同episodeの直前strideのwrist・state・実行済み10×7 actionと有無flag。閉ループではaction履歴を実際に実行した指令に置き換える必要がある。
- 学習キーはmemory=Falseかつvalidation=Falseの行だけで学習。別taskかつ行動MSE近傍を正例にする対照目的、距離閾値{.005,.01,.02}はDev episodeのtask除外Top-1 action MSEで選択。2方式とも閾値 .02 を選択。
- 最終MSEはmemory-gapの修正Expert・同じ10積分・seed 4101/4102・生の候補action chunkを下書きに使用。ゼロ下書き比較も同じseed/query。

## Evalの検索性能（τ=.01、同task除外）

| キー | 情報/方式 | Recall@1 | Recall@8 | Recall@64 | Top-1 hit※ | Top-1 raw action MSE |
|---|---|---:|---:|---:|---:|---:|
| `cameras_only` | VLM base+wrist, instruction excluded | 0.0360 | 0.1546 | 0.5075 | 0.1973 | 0.1116 |
| `centered_current` | VLM centered grouped MaxSim | 0.0154 | 0.0792 | 0.2328 | 0.1068 | 0.2391 |
| `codec_prior_b` | frozen codec prior b̂ (VLM+state; reference) | 0.1313 | 0.4453 | 0.8809 | 0.4027 | 0.0444 |
| `current` | VLM grouped MaxSim, base+wrist+instruction | 0.0239 | 0.0800 | 0.2473 | 0.1315 | 0.2355 |
| `idf_weighted` | VLM IDF-like token weights | 0.0186 | 0.0779 | 0.2768 | 0.1397 | 0.2127 |
| `pi05_pred_to_memory_action` | π0.5 predicted action vs stored action (policy-dependent reference) | 0.3726 | 0.7735 | 0.9976 | 0.8795 | 0.0280 |
| `pooled` | VLM valid-token mean | 0.0172 | 0.0788 | 0.2830 | 0.1260 | 0.1771 |
| `q75` | VLM token 75th percentile | 0.0049 | 0.0203 | 0.1057 | 0.0493 | 0.4015 |
| `top10pct` | VLM top 10% token aggregation | 0.0065 | 0.0192 | 0.0859 | 0.0356 | 0.3988 |
| `top25pct` | VLM top 25% token aggregation | 0.0063 | 0.0243 | 0.0972 | 0.0548 | 0.3828 |
| `whitened_current` | VLM token diagonal whitening | 0.0259 | 0.1220 | 0.3833 | 0.1781 | 0.1591 |
| `wrist_only` | VLM wrist tokens only | 0.0432 | 0.1692 | 0.5083 | 0.2110 | 0.0989 |
| `wrist_state_contrastive` | wrist/state contrastive learned key | 0.0369 | 0.2198 | 0.6610 | 0.1945 | 0.0894 |
| `wrist_state_cos` | wrist tokens + state; train-standardized cosine | 0.0402 | 0.1616 | 0.4596 | 0.2027 | 0.1266 |
| `wrist_state_history_contrastive` | wrist/state/history contrastive learned key | 0.0274 | 0.1816 | 0.6592 | 0.1452 | 0.0945 |
| `wrist_state_history_cos` | current wrist/state + previous wrist/state/action; cosine | 0.0336 | 0.1519 | 0.4580 | 0.2110 | 0.1241 |
| `wrist_state_history_std_l2` | current wrist/state + previous wrist/state/action; L2 | 0.0368 | 0.1412 | 0.4355 | 0.1918 | 0.1303 |
| `wrist_state_std_l2` | wrist tokens + state; train-standardized L2 | 0.0393 | 0.1443 | 0.4290 | 0.2055 | 0.1335 |

※Top-1 hitはoracle正例が存在するquery内の割合。Recallの分母は正例集合サイズ。Eval全体とDev、τ=.005/.02、episode除外は `results.json` に保存。

## 生行動下書きの生成後MSE

各セルは平均MSE。`all`は1,318 Eval行、`hard`は141難行。ゼロ下書き基準は t0=.2: all 0.01065 / hard 0.03453、t0=.4: all 0.00549 / hard 0.02893。`better`は同じrow・noise seedでゼロ下書きより誤差が小さい割合。

| キー | t0=.2 all | t0=.2 hard | better all | better hard | t0=.4 all | t0=.4 hard | better all | better hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cameras_only` | 0.0176 | 0.0746 | 0.6574 | 0.2943 | 0.0060 | 0.0352 | 0.6123 | 0.4255 |
| `centered_current` | 0.0360 | 0.0916 | 0.5011 | 0.2199 | 0.0061 | 0.0331 | 0.5596 | 0.4113 |
| `codec_prior_b` | 0.0104 | 0.0562 | 0.8232 | 0.4645 | 0.0058 | 0.0343 | 0.6954 | 0.4716 |
| `current` | 0.0285 | 0.0812 | 0.5277 | 0.2766 | 0.0055 | 0.0287 | 0.5800 | 0.4574 |
| `idf_weighted` | 0.0262 | 0.0738 | 0.5353 | 0.3014 | 0.0059 | 0.0323 | 0.5800 | 0.4716 |
| `pi05_pred_to_memory_action` | 0.0071 | 0.0325 | 0.8877 | 0.6312 | 0.0049 | 0.0259 | 0.7166 | 0.4929 |
| `pooled` | 0.0292 | 0.0673 | 0.5967 | 0.3333 | 0.0056 | 0.0296 | 0.5933 | 0.4681 |
| `q75` | 0.0505 | 0.0824 | 0.3137 | 0.2624 | 0.0059 | 0.0302 | 0.4473 | 0.4291 |
| `top10pct` | 0.0451 | 0.0799 | 0.3103 | 0.2837 | 0.0058 | 0.0281 | 0.4317 | 0.3972 |
| `top25pct` | 0.0521 | 0.0849 | 0.3335 | 0.2730 | 0.0058 | 0.0293 | 0.4598 | 0.4326 |
| `whitened_current` | 0.0257 | 0.0800 | 0.6093 | 0.2801 | 0.0061 | 0.0354 | 0.6112 | 0.4326 |
| `wrist_only` | 0.0182 | 0.0762 | 0.6563 | 0.2801 | 0.0060 | 0.0327 | 0.6195 | 0.4610 |
| `wrist_state_contrastive` | 0.0159 | 0.0658 | 0.7299 | 0.3759 | 0.0055 | 0.0289 | 0.6491 | 0.4894 |
| `wrist_state_cos` | 0.0230 | 0.0782 | 0.6627 | 0.3440 | 0.0061 | 0.0339 | 0.6316 | 0.4255 |
| `wrist_state_history_contrastive` | 0.0177 | 0.0747 | 0.7003 | 0.3546 | 0.0056 | 0.0285 | 0.6404 | 0.4965 |
| `wrist_state_history_cos` | 0.0227 | 0.0833 | 0.6635 | 0.3121 | 0.0057 | 0.0319 | 0.6335 | 0.4610 |
| `wrist_state_history_std_l2` | 0.0248 | 0.0814 | 0.6582 | 0.2660 | 0.0057 | 0.0304 | 0.6366 | 0.4220 |
| `wrist_state_std_l2` | 0.0242 | 0.0769 | 0.6555 | 0.2837 | 0.0060 | 0.0331 | 0.6256 | 0.3936 |
Task除外以外のepisode除外結果、各方式の全行/難行の生行動MSE、t0別の中央値とゼロ比較割合は `retrieval-4b-01/results.json` にある。

## 難しい行の検索

同task除外・τ=.01・oracle正例が存在するhard queryに限る。Top-1 hitは正例が1位に来たqueryの割合。

| キー | Top-1 hit | Recall@1 | Recall@8 | Recall@64 |
|---|---:|---:|---:|---:|
| `current` | 0.0000 | 0.0000 | 0.0000 | 0.0171 |
| `wrist_only` | 0.0526 | 0.0066 | 0.0697 | 0.3747 |
| `wrist_state_cos` | 0.0000 | 0.0000 | 0.1680 | 0.3421 |
| `wrist_state_contrastive` | 0.1053 | 0.0171 | 0.1609 | 0.6762 |
| `wrist_state_history_contrastive` | 0.0000 | 0.0000 | 0.0612 | 0.5814 |
| `codec_prior_b` | 0.1053 | 0.0154 | 0.1147 | 0.7101 |
| `pi05_pred_to_memory_action` | 0.5263 | 0.2439 | 0.5162 | 1.0000 |

## 読み取り

- 現行VLMのEval task-excluded Recall@8は0.080。wrist-onlyは0.169、wrist＋状態の固定cosは0.162、対照キーは0.220（Recall@64は0.661）。短い履歴を加えた対照キーはRecall@8が0.182で、履歴なしを上回らなかった。
- hard subsetでは現行VLMのRecall@8は0、wrist＋状態cosは0.168、対照キーは0.161。Top-64 coverageが増えても、Top-1で選べる率と生成後誤差は別に見る必要がある。
- hard subsetのt0=.2ではゼロ下書きMSE 0.03453に対し、π0.5予測行動参照が0.03253で最小。通常画像/状態入力の対照キーは0.06577で、ゼロ基準を下回らない。t0=.4ではπ0.5予測参照0.02592、top10pct 0.02813、wrist-history contrastive 0.02851。差の統計区間は算出していない。
- したがって「良い生行動候補は上位に引けるか」は入力依存で、手首/状態の候補被覆は現行VLMより改善するが、今回の評価では生成後の難行改善を安定して確認できない。π0.5予測キーの高Recallは方策予測依存の参考値であり、独立した観測キーの成功とは扱わない。

## 成果物・再現

- 全候補・Recall・最終MSE・Dev選択記録はGit管理外の `/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/retrieval-4b-01/results.json`。
- `keys/*_task.npz` / `keys/*_episode.npz` は全query×Top-64とそのTop-8の`cand_actions[Q,8,10,32]`を格納。
- `wrist_token_mean.npy`、正規化統計、`*_embedding.npy`、`*_learned/wrist_contrastive.pt`、`learned_fit.json`を保存。
- `raw_draft_final_mse.npz` はEvalの2 seed×72条件の最終row MSE、hard mask、候補生行動誤差を保存。
- 単体テスト: `tests/test_pi05_memory_retrieval_4b.py`。出力作成はCPU 8 thread・taskset 8-47・nice 10、JAX推論はXLA preallocation false・memory fraction .20。
- 実時間：キー生成/学習/Recall集計 116.6秒、最終推論 3,571.9秒。GPUの共有使用量は他プロセスを含むため、プロセス別ピークとしては報告しない。
