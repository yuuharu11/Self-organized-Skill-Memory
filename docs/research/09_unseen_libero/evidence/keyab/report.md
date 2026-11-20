# KEYAB：キー入力・位置埋め込み・Self-Attention 比較

## 結論

初回 raw の MSE 異常は、raw 配列として物理 DCT 戻り値を誤って使ったことが原因だった。raw_pool に直して再実行すると raw absolute の unseen20 は .0232〜.0249 となり、既存 K-x1 .0245 と同程度になった。全40キー設定と10下流 query を CPU・seed 4101 で完了。 追加教師3種の48設定・12構成も完了し、各構成はDev MSEでτ/epochを選択した。追加群で最小Devは raw_window10 / absolute / SA1 (Dev 0.02449; unseen20 0.02163)。

## 初回異常の原因と訂正監査

- `_variable_pool` の返り値は `(physical_DCT_pool, metadata, raw_pool)`。初回は第1要素を raw として使っていた。修正版は第3要素を使い、初回の全成果を `initial-invalid/` に退避して、再実行結果に混ぜていない。
- 元 bank の単位数 4516。`motion_core_eval.anchor_spans(min_length=5)` を全メモリ trace で独立再生成し、(task, episode, start, end) の順序・境界・length の不一致は 0。raw と source 行動の最大絶対差は 0.0e+00。
- 正規化は元の q01/q99 を [-1,1] に写した行動値。未クリップ外れ値を含む観測範囲は [-2.1731247901916504, 2.557753801345825]。raw を物理空間で DCT3 再構成して正規化へ戻した結果と `natural_actions` の最大差は 1.17e-07。
- 初回 raw absolute / SAなし / window10 / τ=.01 は ep4・ep25 とも Dev 上で epoch1 が選ばれ、同じ epoch1 checkpoint を評価したため最終値が同じだった。これは stale ep4/ep25 評価ではなく、実際に各 epoch の Dev 値が計算されていた（初回 long の Dev は epoch1 .12977 から epoch25 .20633へ悪化）。修正後は ep4 設定が epoch4 .02760、ep25 設定が epoch23 .02678 を選び、評価もそれぞれ .02491 と .02395 になった。
- 同じ runner の DCT3×absolute×SAなし X1 checkpoint control は K-x1 に対するキー最大差 1.64e-07、unseen20 MSE 0.02450。DCT3 学習条件 τ=.01/ep4 は 0.02481。

## キー空間最近傍（上位単位選択）

| 入力 | 位置 | SA | 教師 | τ | epoch | Dev top1腕MSE | learned | unseen20 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-x1 (既存) | absolute | no | 距離蒸留 | .01 | — | — | 0.02122 | 0.02450 |
| DCT3/X1 checkpoint control | absolute | no | 既存X1重み | — | — | 0.02706 | 0.02122 | 0.02450 |
| raw | absolute | 0 | window10 | 0.01 | 4 | 0.02760 | 0.02168 | 0.02491 |
| raw | absolute | 0 | window10 | 0.01 | 23 | 0.02678 | 0.02061 | 0.02395 |
| raw | absolute | 0 | window10 | 0.05 | 4 | 0.02692 | 0.02064 | 0.02433 |
| raw | absolute | 0 | window10 | 0.05 | 23 | 0.02619 | 0.02026 | 0.02321 |
| raw | absolute | 1 | window10 | 0.01 | 4 | 0.02771 | 0.02147 | 0.02481 |
| raw | absolute | 1 | window10 | 0.01 | 20 | 0.02678 | 0.02073 | 0.02433 |
| raw | absolute | 1 | window10 | 0.05 | 4 | 0.02702 | 0.02101 | 0.02443 |
| raw | absolute | 1 | window10 | 0.05 | 23 | 0.02605 | 0.02042 | 0.02310 |
| raw | none | 0 | window10 | 0.01 | 3 | 0.03334 | 0.02846 | 0.03016 |
| raw | none | 0 | window10 | 0.01 | 24 | 0.03248 | 0.02740 | 0.02949 |
| raw | none | 0 | window10 | 0.05 | 4 | 0.03323 | 0.02753 | 0.03064 |
| raw | none | 0 | window10 | 0.05 | 19 | 0.03216 | 0.02669 | 0.02944 |
| raw | none | 1 | window10 | 0.01 | 4 | 0.03362 | 0.02768 | 0.03099 |
| raw | none | 1 | window10 | 0.01 | 21 | 0.03249 | 0.02720 | 0.02867 |
| raw | none | 1 | window10 | 0.05 | 4 | 0.03372 | 0.02833 | 0.03077 |
| raw | none | 1 | window10 | 0.05 | 24 | 0.03226 | 0.02670 | 0.02858 |
| dct3 | absolute | 0 | window10 | 0.01 | 4 | 0.02732 | 0.02144 | 0.02481 |
| dct3 | absolute | 0 | window10 | 0.01 | 22 | 0.02669 | 0.02099 | 0.02421 |
| dct3 | absolute | 0 | window10 | 0.05 | 4 | 0.02650 | 0.02046 | 0.02390 |
| dct3 | absolute | 0 | window10 | 0.05 | 23 | 0.02590 | 0.02005 | 0.02292 |
| dct3 | absolute | 1 | window10 | 0.01 | 4 | 0.02741 | 0.02116 | 0.02497 |
| dct3 | absolute | 1 | window10 | 0.01 | 20 | 0.02658 | 0.02102 | 0.02436 |
| dct3 | absolute | 1 | window10 | 0.05 | 4 | 0.02694 | 0.02086 | 0.02424 |
| dct3 | absolute | 1 | window10 | 0.05 | 23 | 0.02593 | 0.02032 | 0.02311 |
| dct3 | none | 0 | window10 | 0.01 | 3 | 0.03365 | 0.02905 | 0.03079 |
| dct3 | none | 0 | window10 | 0.01 | 16 | 0.03232 | 0.02743 | 0.02932 |
| dct3 | none | 0 | window10 | 0.05 | 4 | 0.03335 | 0.02823 | 0.03041 |
| dct3 | none | 0 | window10 | 0.05 | 25 | 0.03250 | 0.02735 | 0.02925 |
| dct3 | none | 1 | window10 | 0.01 | 4 | 0.03357 | 0.02818 | 0.03028 |
| dct3 | none | 1 | window10 | 0.01 | 15 | 0.03228 | 0.02774 | 0.02908 |
| dct3 | none | 1 | window10 | 0.05 | 4 | 0.03416 | 0.02849 | 0.03082 |
| dct3 | none | 1 | window10 | 0.05 | 20 | 0.03229 | 0.02741 | 0.02897 |
| raw | absolute | 0 | whole16 | 0.01 | 4 | 0.02918 | 0.02349 | 0.02699 |
| raw | absolute | 0 | whole16 | 0.01 | 6 | 0.02906 | 0.02357 | 0.02646 |
| raw | absolute | 0 | whole16 | 0.05 | 4 | 0.02891 | 0.02279 | 0.02605 |
| raw | absolute | 0 | whole16 | 0.05 | 7 | 0.02867 | 0.02268 | 0.02600 |
| raw | absolute | 1 | whole16 | 0.01 | 4 | 0.02941 | 0.02402 | 0.02707 |
| raw | absolute | 1 | whole16 | 0.01 | 6 | 0.02913 | 0.02383 | 0.02708 |
| raw | absolute | 1 | whole16 | 0.05 | 4 | 0.02855 | 0.02298 | 0.02617 |
| raw | absolute | 1 | whole16 | 0.05 | 13 | 0.02848 | 0.02263 | 0.02635 |

## 追加：生アクション教師距離（raw DCT3 / DCT8 / window10）

入力は3教師すべて raw 正規化行動。位置埋め込み・SA・τ・epoch を変え、各 teacher×position×SA ごとに Dev v2 top-1 MSE で τ/epoch を選んだ。Eval の正解は選択に使っていない。

### τの固定方法

LIBERO-40 memory bank 内で、異なるエピソードの単位対から400,000組を seed 4101 で一様サンプルし、教師距離分布の q25/q50 を2つの τ とした。評価行・Dev/Eval の行動は τ 推定に使っていない。raw_dctK は DCT-II ortho の低周波 K 係数を各単位の √L で割り、係数×6軸の平均二乗距離を教師にする。raw_window10 は生行動の先頭10点、短い区間は末尾値保持で10×6平均二乗距離。

| 教師 | 次元 | τ(q25) | τ(q50) | 距離標本 p10 / p50 / p90 |
|---|---:|---:|---:|---|
| raw_dct3 | 18 | 0.0488068 | 0.0810914 | 0.0272508 / 0.0810914 / 0.175026 |
| raw_dct8 | 48 | 0.0187226 | 0.0309114 | 0.0105221 / 0.0309114 / 0.0663064 |
| raw_window10 | 60 | 0.131849 | 0.225877 | 0.0720438 / 0.225877 / 0.514514 |

### 各教師の Dev 選択キー結果

| 教師 | 位置 | SA | τ | 選択epoch | Dev top1腕MSE | learned | unseen20 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw_dct3 | absolute | 0 | 0.0810914 | 17 | 0.02867 | 0.02306 | 0.02651 |
| raw_dct3 | absolute | 1 | 0.0488068 | 16 | 0.02868 | 0.02290 | 0.02658 |
| raw_dct3 | none | 0 | 0.0810914 | 15 | 0.03278 | 0.02815 | 0.03008 |
| raw_dct3 | none | 1 | 0.0488068 | 24 | 0.03180 | 0.02672 | 0.02965 |
| raw_dct8 | absolute | 0 | 0.0309114 | 14 | 0.02849 | 0.02290 | 0.02643 |
| raw_dct8 | absolute | 1 | 0.0187226 | 20 | 0.02843 | 0.02287 | 0.02633 |
| raw_dct8 | none | 0 | 0.0187226 | 25 | 0.03253 | 0.02782 | 0.03005 |
| raw_dct8 | none | 1 | 0.0187226 | 20 | 0.03176 | 0.02653 | 0.02973 |
| raw_window10 | absolute | 0 | 0.131849 | 17 | 0.02452 | 0.01902 | 0.02172 |
| raw_window10 | absolute | 1 | 0.131849 | 24 | 0.02449 | 0.01878 | 0.02163 |
| raw_window10 | none | 0 | 0.131849 | 15 | 0.03233 | 0.02753 | 0.02938 |
| raw_window10 | none | 1 | 0.225877 | 25 | 0.03286 | 0.02717 | 0.02944 |

### 同じ構成で学習した C1b query 下流

| 教師キー | Dev選択MSE | learned | unseen20 | epoch |
|---|---:|---:|---:|---:|
| key_raw_absolute_sa0_raw_dct3_tau0.0810914_ep25 | 0.06836 | 0.05367 | 0.06043 | 16 |
| key_raw_absolute_sa1_raw_dct3_tau0.0488068_ep25 | 0.06370 | 0.05225 | 0.05656 | 14 |
| key_raw_none_sa0_raw_dct3_tau0.0810914_ep25 | 0.07057 | 0.05811 | 0.06036 | 16 |
| key_raw_none_sa1_raw_dct3_tau0.0488068_ep25 | 0.06840 | 0.05100 | 0.06030 | 14 |
| key_raw_absolute_sa0_raw_dct8_tau0.0309114_ep25 | 0.06829 | 0.05394 | 0.05928 | 16 |
| key_raw_absolute_sa1_raw_dct8_tau0.0187226_ep25 | 0.06311 | 0.05143 | 0.05614 | 14 |
| key_raw_none_sa0_raw_dct8_tau0.0187226_ep25 | 0.06643 | 0.05177 | 0.05763 | 16 |
| key_raw_none_sa1_raw_dct8_tau0.0187226_ep25 | 0.06820 | 0.05161 | 0.06067 | 14 |
| key_raw_absolute_sa0_raw_window10_tau0.131849_ep25 | 0.05829 | 0.04441 | 0.05185 | 14 |
| key_raw_absolute_sa1_raw_window10_tau0.131849_ep25 | 0.05724 | 0.04438 | 0.05130 | 14 |
| key_raw_none_sa0_raw_window10_tau0.131849_ep25 | 0.06389 | 0.04925 | 0.05665 | 14 |
| key_raw_none_sa1_raw_window10_tau0.225877_ep25 | 0.06584 | 0.05267 | 0.05699 | 16 |

### 既存参照との比較（unseen20）

| 参照 | Devでの選択MSE | unseen20 key-space腕MSE | unseen20 C1b downstream腕MSE |
|---|---:|---:|---:|
| K-x1 | — | 0.02450 | — |
| 学習なし DCT3 | — | 0.02088 | — |
| 既存40構成のDev最良 (dct3_absolute_sa0_window10_tau0.05_ep25) | 0.02590 | 0.02292 | 0.05097903311252594 |

## C1b query downstream

| キー | Dev選択MSE | learned | unseen20 | epoch |
|---|---:|---:|---:|---:|
| key_raw_absolute_sa0_window10_tau0.05_ep25 | 0.05863 | 0.04276 | 0.05133 | 14 |
| key_raw_absolute_sa1_window10_tau0.05_ep25 | 0.05827 | 0.04253 | 0.05129 | 14 |
| key_raw_none_sa0_window10_tau0.05_ep25 | 0.06312 | 0.05142 | 0.05765 | 8 |
| key_raw_none_sa1_window10_tau0.05_ep25 | 0.06139 | 0.05063 | 0.05745 | 8 |
| key_dct3_absolute_sa0_window10_tau0.05_ep25 | 0.05701 | 0.04226 | 0.05098 | 14 |
| key_dct3_absolute_sa1_window10_tau0.05_ep25 | 0.05819 | 0.04253 | 0.05128 | 14 |
| key_dct3_none_sa0_window10_tau0.01_ep25 | 0.06364 | 0.04501 | 0.05644 | 14 |
| key_dct3_none_sa1_window10_tau0.01_ep25 | 0.06484 | 0.04505 | 0.06329 | 14 |
| key_raw_absolute_sa0_whole16_tau0.05_ep25 | 0.06357 | 0.04782 | 0.05549 | 16 |
| key_raw_absolute_sa1_whole16_tau0.05_ep25 | 0.06251 | 0.05088 | 0.05721 | 8 |

## プロトコル・制約

- 固定候補バンク: 4516 可変長単位。raw は元の LeRobot 行動を同一 `anchor_spans` 境界で切り出し、DCT3 は KQ2 bank の natural reconstruction を使用。
- 学習教師は episode 除外付きの腕6 MSE soft-listwise。window10 は10 step 対応、whole16 は両区間を16点へ同長 resample して全体比較。
- 下流 query は KQ2 C1b HSMLP の設定、モデル・epoch は Dev v2 20Hz で選択。
- CPUのみ。PyTorch/BLAS/OMP の thread 数は進捗ログと環境情報に記録。
- 参照（unseen20）：直前保持 .0470、C1b .0522。DCT3: 0.02087775617837906
- 総時間: 4809.0 sec
