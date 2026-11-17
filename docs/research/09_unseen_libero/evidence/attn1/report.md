# ATTN1：潜在トークンの注意配分（学習なし）

行数：キー 4516、学習済み 621、未学習 Eval 619、Dev 217。

## 潜在表現と注意

| 条件 | 分割 | 平均 token cosine | 有効ランク | VLM画像 | VLM言語 | V-JEPA | 履歴 | 状態 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| e_vlm_h | learned | 0.8583 | 3.112 | 0.8416 | 0.0987 | 0.0000 | 0.0340 | 0.0256 |
| e_vlm_h | unseen_eval | 0.8375 | 3.221 | 0.8493 | 0.0904 | 0.0000 | 0.0313 | 0.0290 |
| e_vlm_h | dev | 0.8405 | 3.267 | 0.8635 | 0.0815 | 0.0000 | 0.0344 | 0.0206 |
| e_vlm_h | key | 0.8626 | 3.084 | 0.8431 | 0.0990 | 0.0000 | 0.0316 | 0.0263 |
| e_vj_vlm_h | learned | 0.7990 | 3.613 | 0.7117 | 0.0622 | 0.1889 | 0.0313 | 0.0060 |
| e_vj_vlm_h | unseen_eval | 0.7358 | 3.790 | 0.7194 | 0.0523 | 0.1952 | 0.0273 | 0.0059 |
| e_vj_vlm_h | dev | 0.6624 | 4.027 | 0.7253 | 0.0428 | 0.1988 | 0.0292 | 0.0039 |
| e_vj_vlm_h | key | 0.8135 | 3.555 | 0.7121 | 0.0624 | 0.1885 | 0.0309 | 0.0062 |
| e_h | learned | 0.9732 | 2.068 | 0.0000 | 0.0000 | 0.0000 | 0.6549 | 0.3451 |
| e_h | unseen_eval | 0.9723 | 2.002 | 0.0000 | 0.0000 | 0.0000 | 0.6042 | 0.3958 |
| e_h | dev | 0.9730 | 2.055 | 0.0000 | 0.0000 | 0.0000 | 0.5701 | 0.4299 |
| e_h | key | 0.9768 | 1.995 | 0.0000 | 0.0000 | 0.0000 | 0.6488 | 0.3512 |
| keygrid_e_vlm_h | learned | 0.8542 | 3.199 | 0.8418 | 0.0852 | 0.0000 | 0.0494 | 0.0235 |
| keygrid_e_vlm_h | unseen_eval | 0.8207 | 3.365 | 0.8511 | 0.0796 | 0.0000 | 0.0439 | 0.0254 |
| keygrid_e_vlm_h | dev | 0.8300 | 3.360 | 0.8622 | 0.0705 | 0.0000 | 0.0506 | 0.0166 |
| keygrid_e_vlm_h | key | 0.8570 | 3.192 | 0.8416 | 0.0861 | 0.0000 | 0.0480 | 0.0243 |
| keygrid_e_vj_vlm_h | learned | 0.9135 | 3.010 | 0.7460 | 0.1147 | 0.1046 | 0.0273 | 0.0073 |
| keygrid_e_vj_vlm_h | unseen_eval | 0.8801 | 3.218 | 0.7575 | 0.1032 | 0.1066 | 0.0240 | 0.0088 |
| keygrid_e_vj_vlm_h | dev | 0.8157 | 3.579 | 0.7709 | 0.0908 | 0.1043 | 0.0259 | 0.0082 |
| keygrid_e_vj_vlm_h | key | 0.9199 | 2.959 | 0.7459 | 0.1152 | 0.1044 | 0.0267 | 0.0078 |
| keygrid_e_h | learned | 0.9427 | 2.445 | 0.0000 | 0.0000 | 0.0000 | 0.6464 | 0.3536 |
| keygrid_e_h | unseen_eval | 0.9397 | 2.384 | 0.0000 | 0.0000 | 0.0000 | 0.6039 | 0.3961 |
| keygrid_e_h | dev | 0.9411 | 2.449 | 0.0000 | 0.0000 | 0.0000 | 0.6109 | 0.3891 |
| keygrid_e_h | key | 0.9500 | 2.365 | 0.0000 | 0.0000 | 0.0000 | 0.6381 | 0.3619 |

## 基準検索の top-1 腕 MSE

| 条件 | 分割 | 全体 | hard | forward 最大差 |
|---|---|---:|---:|---:|
| e_vlm_h | learned | 0.0350 | 0.0664 | 0.0 |
| e_vlm_h | unseen_eval | 0.1250 | 0.2561 | 0.0 |
| e_vj_vlm_h | learned | 0.0405 | 0.0779 | 0.0 |
| e_vj_vlm_h | unseen_eval | 0.1191 | 0.2247 | 0.0 |
| e_h | learned | 0.0552 | 0.0891 | 0.0 |
| e_h | unseen_eval | 0.1455 | 0.2711 | 0.0 |
| keygrid_e_vlm_h | learned | 0.0365 | 0.0703 | 0.0 |
| keygrid_e_vlm_h | unseen_eval | 0.1266 | 0.2498 | 0.0 |
| keygrid_e_vj_vlm_h | learned | 0.0316 | 0.0575 | 0.0 |
| keygrid_e_vj_vlm_h | unseen_eval | 0.1203 | 0.2474 | 0.0 |
| keygrid_e_h | learned | 0.0604 | 0.1023 | 0.0 |
| keygrid_e_h | unseen_eval | 0.1309 | 0.2519 | 0.0 |

## 入力除去時の top-1

| 条件 | 分割 | 入力 | zero 一致率 / MSE | swap 一致率 / MSE |
|---|---|---|---:|---:|
| e_vlm_h | learned | vlm | 0.010 / 0.1810 | 0.058 / 0.1028 |
| e_vlm_h | learned | history | 0.406 / 0.0419 | 0.501 / 0.0407 |
| e_vlm_h | learned | state | 0.417 / 0.0381 | 0.775 / 0.0360 |
| e_vlm_h | unseen_eval | vlm | 0.016 / 0.1840 | 0.005 / 0.2186 |
| e_vlm_h | unseen_eval | history | 0.430 / 0.1310 | 0.415 / 0.1345 |
| e_vlm_h | unseen_eval | state | 0.304 / 0.1289 | 0.349 / 0.1252 |
| e_vj_vlm_h | learned | vlm | 0.024 / 0.1177 | 0.113 / 0.0880 |
| e_vj_vlm_h | learned | vjepa | 0.055 / 0.0694 | 0.340 / 0.0507 |
| e_vj_vlm_h | learned | history | 0.338 / 0.0490 | 0.451 / 0.0468 |
| e_vj_vlm_h | learned | state | 0.823 / 0.0405 | 0.923 / 0.0408 |
| e_vj_vlm_h | unseen_eval | vlm | 0.024 / 0.1349 | 0.013 / 0.1715 |
| e_vj_vlm_h | unseen_eval | vjepa | 0.052 / 0.1484 | 0.079 / 0.1383 |
| e_vj_vlm_h | unseen_eval | history | 0.351 / 0.1281 | 0.262 / 0.1357 |
| e_vj_vlm_h | unseen_eval | state | 0.859 / 0.1168 | 0.837 / 0.1178 |
| e_h | learned | history | 0.079 / 0.1186 | 0.100 / 0.1099 |
| e_h | learned | state | 0.002 / 0.2097 | 0.332 / 0.0832 |
| e_h | unseen_eval | history | 0.173 / 0.1820 | 0.063 / 0.2166 |
| e_h | unseen_eval | state | 0.002 / 0.2092 | 0.019 / 0.2219 |
| keygrid_e_vlm_h | learned | vlm | 0.002 / 0.1657 | 0.058 / 0.1061 |
| keygrid_e_vlm_h | learned | history | 0.412 / 0.0419 | 0.530 / 0.0417 |
| keygrid_e_vlm_h | learned | state | 0.422 / 0.0399 | 0.770 / 0.0367 |
| keygrid_e_vlm_h | unseen_eval | vlm | 0.002 / 0.1512 | 0.006 / 0.2165 |
| keygrid_e_vlm_h | unseen_eval | history | 0.380 / 0.1280 | 0.312 / 0.1369 |
| keygrid_e_vlm_h | unseen_eval | state | 0.241 / 0.1233 | 0.263 / 0.1254 |
| keygrid_e_vj_vlm_h | learned | vlm | 0.005 / 0.1075 | 0.074 / 0.0895 |
| keygrid_e_vj_vlm_h | learned | vjepa | 0.163 / 0.0419 | 0.311 / 0.0375 |
| keygrid_e_vj_vlm_h | learned | history | 0.641 / 0.0342 | 0.670 / 0.0342 |
| keygrid_e_vj_vlm_h | learned | state | 0.844 / 0.0322 | 0.921 / 0.0317 |
| keygrid_e_vj_vlm_h | unseen_eval | vlm | 0.019 / 0.1214 | 0.010 / 0.1898 |
| keygrid_e_vj_vlm_h | unseen_eval | vjepa | 0.071 / 0.1170 | 0.108 / 0.1250 |
| keygrid_e_vj_vlm_h | unseen_eval | history | 0.645 / 0.1257 | 0.577 / 0.1281 |
| keygrid_e_vj_vlm_h | unseen_eval | state | 0.848 / 0.1185 | 0.774 / 0.1168 |
| keygrid_e_h | learned | history | 0.074 / 0.1712 | 0.079 / 0.1329 |
| keygrid_e_h | learned | state | 0.013 / 0.1607 | 0.475 / 0.0762 |
| keygrid_e_h | unseen_eval | history | 0.165 / 0.1781 | 0.037 / 0.2182 |
| keygrid_e_h | unseen_eval | state | 0.002 / 0.1907 | 0.047 / 0.1711 |

## キー側のタスクまとまり

| 条件 | 5-NN task accuracy | cosine silhouette |
|---|---:|---:|
| e_vlm_h | 0.3623 | -0.1774 |
| e_vj_vlm_h | 0.4300 | -0.2050 |
| e_h | 0.5279 | -0.2029 |
| keygrid_e_vlm_h | 0.3609 | -0.1649 |
| keygrid_e_vj_vlm_h | 0.3508 | -0.1913 |
| keygrid_e_h | 0.4942 | -0.1808 |

注意は各クロスアテンション層（本モデルは1層）・ヘッド・潜在トークンごとの平均質量で記録し、詳細配列は `results.json` に保存した。`per_input_token_attention` は各モダリティの有効トークン数で割った値。
入力 ablation は推論時のクエリ側だけを変更し、キー側の4,516行は固定した。Eval 正解は行動誤差の採点にのみ用いた。
観測機能は既存 forward の外側に置いたため、チェックポイント出力の再計算差は全条件で 0。学習・重み更新は行っていない。
KEYGRID で top-1 Dev 選択された同方式の E チェックポイントも別条件として評価した。これらは KEYGRID 専用に新規学習した E ではなく、既存 tuning の候補からの再選択である。
