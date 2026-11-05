# Fair Expert: 検索動作の内容を公平な学習条件で比較

計画は [plan.md](plan.md) に実行前固定。出力はオフライン行動誤差であり、閉ループ成功率ではない。

## データ漏洩・学習監査

- 固定分割: 学習 11,712 行（Long task 0–9、保留6 task、validation を除外）、Dev 213 行（保留6 taskのみ）、Long Eval 548 行・hard 63 行。raw-rows-noLong メモリは 3,857 行。Eval task/episode は学習・Dev に入らず、候補も別 task のメモリ row ID から `actions[row_id]` を直接参照した。
- M2 候補 Top-64 を使い、E-M2 学習では各 query ごとに rank 1 が 70%、rank 2–5 が各 7.5%（実現値は `training_drafts.npz` に保存）。rank counts: rank1=8189, rank2=895, rank3=880, rank4=834, rank5=914.
- 下書き active7 MSE: E-M2 学習行の固定混合 .08860、Dev 同じ混合 .06616（Dev Top-1 固定なら .06339）。M2 元のfit task上の検索エラーが heldout より小さいという楽観偏りは見えず、今回の検索はむしろ heldout Devで低い。
- E-M2/E-zero は同じ初期値・seed・batch row ID列（SHA256 `a5503e8aa72609cfad795105262c87c13106d615a4cab68f51740c1895df416d`）・4,000 updates・batch8。実行時間は E-M2 893.7s、E-zero 900.8s。GPU の観測最大は約 34.5 GiB。XLA preallocation off、fraction .25、platform allocator。
- E-M2 の Dev 選択で初版がTop-1固定になっていたため、保存済み全チェックポイントを、事前指定分布の固定Dev候補（seed 20263006）で再評価した。最終選択は 3000 updates, t0=0.6, MSE 0.003095。E-zero は 3500 updates, t0=0.6, MSE 0.003239。

## Long Eval: active7 正規化 MSE（全行 / hard 63行）

同一 noise `fold_in(key(4101), row_id)`、10 Euler steps。各セルは `全Long / hard`。

独立レビューで、初回の `evaluation-final` のshuffle下書きがTop-1ではなくrank 1〜5混合から選ばれており、M2 Top-1との分布が一致していないことを確認した。F2と以下のshuffle行は、M2 Top-1候補の**同じ548行分の集合**をクエリ間で置換した訂正版で再評価した。訂正版の予測は `corrected-top1-shuffle-01/` に別保存し、初回出力は監査用に残した。初回shuffle予測はF2に使わない。

| 条件 | t0=.2 | t0=.4 | t0=.6 |
|---|---:|---:|---:|
| 公式π0.5（通常推論、t0=1） | 0.00683 / 0.04557 | 0.00683 / 0.04557 | 0.00683 / 0.04557 |
| E-M2＋M2_top1 | 0.01302 / 0.06553 | 0.00541 / 0.03108 | 0.00480 / 0.02791 |
| E-M2＋shuffle（Top-1集合を置換） | 0.04802 / 0.07326 | 0.00533 / 0.02895 | 0.00478 / 0.02772 |
| E-M2＋zero | 0.00812 / 0.03126 | 0.00498 / 0.02810 | 0.00466 / 0.02661 |
| E-M2＋direct_regression | 0.00754 / 0.04231 | 0.00566 / 0.03273 | 0.00471 / 0.02699 |
| E-zero＋zero | 0.00934 / 0.05668 | 0.00682 / 0.04265 | 0.00637 / 0.03841 |
| E-zero＋direct_regression | 0.01965 / 0.11296 | 0.00910 / 0.05647 | 0.00662 / 0.03936 |
| 旧Expert＋M2_top1 | 0.02515 / 0.10193 | 0.00643 / 0.03737 | 0.00529 / 0.03042 |
| 旧Expert＋shuffle（Top-1集合を置換） | 0.08189 / 0.07768 | 0.00632 / 0.03368 | 0.00565 / 0.03314 |
| 旧Expert＋zero | 0.01003 / 0.03627 | 0.00590 / 0.03249 | 0.00547 / 0.03224 |
| 旧Expert＋direct_regression | 0.00837 / 0.04787 | 0.00542 / 0.03212 | 0.00527 / 0.03213 |

## 事前判定（選択 t0=.6）

- **F1: 成立** — (E-M2＋M2 Top-1) − (E-zero＋zero), 全Long Δ=-0.00157 [ -0.00253, -0.00069 ]; hard Δ=-0.01050 [ -0.01670, -0.00423 ]。両方でCI上端<0。
- **F2: 不成立** — 同じTop-1候補集合の置換との比較で、全Long Δ=+0.000026 [ -0.000694, +0.000780 ]、hard Δ=+0.000197 [ -0.005805, +0.007100 ]。両CI上端とも0未満ではない。
- 同じ候補順位分布を両側に使う追加対照（rank1 .7 / rank2–5 .3）: t0=.2 M2混合/shuffle=.01278/.04709、t0=.4=.00541/.00530、t0=.6=.00500/.00477。t0=.6 の対応差は Δ=+0.00024 [-0.00039, +0.00095] で、F2は同様に成立しない。
- **F3参考** — E-M2＋M2−公式π0.5: 全Long Δ=-0.00203 [-0.00335, -0.00078]; 因果判定には使わない。
- 出力感度比（候補間の出力距離/下書き距離、Top-1候補集合を置換）E-M2: t0=.2/.4/.6で .206/.049/.018。旧Expertは .292/.052/.020。選択した動作の内容を使う証拠は得られなかった。
- 直接回帰参照の下書きMSE 0.05396。M2 Top-1・ゼロ等の初回条件は `evaluation-final/offline_predictions.npz`、訂正版Top-1 shuffle予測とF2は `corrected-top1-shuffle-01/top1_shuffle_predictions.npz`・`decision.json`、学習rank分布を揃えた追加対照は `evaluation-rankmix/rankmix_results.npz`。
- 5,000回 episode bootstrap、20 episodes。E-zero はM2行動を見ずゼロ下書き専用、E-M2はM2下書き混合専用。統計的差なしを同等とは解釈しない。
