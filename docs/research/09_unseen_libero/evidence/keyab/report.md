# KEYAB：キー入力・位置埋め込み・Self-Attention 比較

## 結論

主評価は Dev v2 20Hz でキー設定を選び、学習済み／unseen20 は選択後に報告した。K-x1 を既存参照とし、全条件は CPU の seed 4101 で実行した。

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
- 総時間: 5.432021155022085 sec
