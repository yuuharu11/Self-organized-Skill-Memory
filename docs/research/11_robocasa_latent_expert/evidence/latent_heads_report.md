# RCX6：潜在トークンからグリッパーと t0 を予測

## 結果の要点

E-VLM+H の固定潜在表現から予測誤差を出し、Expert の学習時 `T0Map` を通して t0 を決めた。非fallback 時の t0 は Eval で平均 .397、中央値 .396、範囲 .102–.701 と広がり、.1 への張り付きは解消した。ただし Eval の腕6 MSEは新方式 .578、下書き .562、dist版 .547、π0.5 .637 で、新方式は下書きと dist版を上回らなかった。

グリッパーの符号一致率は高い（Eval .958）が、主指標の切替時刻 recall@±2 は .136 で、RCX5 MLP .295 と π0.5 .261 より低い。閉ループは関門の PickPlaceSinkToCounter で 9/10 を通過したものの、新規2タスクで 11/60、6タスク trial 0–29 で 72/180 となり、対応する π0.5 と dist E-VLM+H を下回った。今回のグリッパー・t0 ヘッドを閉ループに採用する根拠は得られなかった。

## 手順1：12次元行動の tail と固定方法

RCX5 `conf-fix/attempt2/gripper_training.json` の6タスク学習デモ115,257行を読み取り、policy channel 7–12 を `[gripper, base0, base1, base2, base3, mode]` と確認した。

| channel | 意味 | 学習デモで最頻値 | 最頻値率 | RCX6非fallbackの扱い |
|---:|---|---:|---:|---|
| 7 | gripper | -1 | 70.15% | 潜在トークンの分類ヘッド |
| 8 | base0 | 0 | 99.93% | 0固定 |
| 9 | base1 | 0 | 99.94% | 0固定 |
| 10 | base2 | 0 | 99.96% | 0固定 |
| 11 | base3 | 0 | 100.00% | 0固定 |
| 12 | mode | -1 | 99.91% | -1固定 |

固定値は `[base0..base3, mode]=[0,0,0,0,-1]`。fallback 時は π0.5 の12次元出力を使う。gripper は5個の固定 E-VLM+H query token（5×256）を入力する50ステップMLPで別に予測した。

## 手順2：Devでのヘッド選択

### グリッパー切替

候補は hidden 64 / 8 epochs と hidden 128 / 10 epochs（lr .001）。Dev の切替 recall@±2 を優先し、次に50step balanced accuracyを比較して後者を選んだ。Devでは新ヘッドの符号一致 .967、切替 recall@±2 .281、平均最近傍時刻誤差6.47 steps。直前保持は符号一致 .980だが切替 recallは0。

| Evalの予測方法 | 符号一致率（最初10step） | 真の切替数 | 予測切替数 | 切替 recall@±2 | 平均最近傍時刻誤差 |
|---|---:|---:|---:|---:|---:|
| RCX6 潜在トークンヘッド | .958 | 88 | 82 | .136 | 7.76 step |
| RCX5 固定MLP（attempt2） | .973 | 88 | 106 | .295 | 6.17 step |
| 直前保持 | .976 | 88 | 0 | .000 | 10.00 step |
| 保存単位 | .867 | 88 | 290 | .148 | 7.90 step |
| π0.5 | .972 | 88 | 81 | .261 | 6.43 step |

符号一致率だけでは直前保持が最良に見える一方、切替時刻は全て見逃す。RCX6ヘッドは直前保持より切替を拾うが、Evalの切替時刻はRCX5 MLP・π0.5より悪かった。RCX5 MLPのEval指標は既存の固定チェックポイントを同じ2151行で再採点したもの。

### 誤差推定と t0 / fallback

誤差ヘッドの入力はクエリ潜在、選択された単位のkey潜在、差・積。追加信号版はさらに7信号（Eのtop score・gap・entropy、top-10単位の動きの散らばり、P5とEの選択差、draftの連続性、単位の繰り返し）を連結した。π0.5出力・正解行動・task IDは特徴入力に含めていない。Devで `latent_plus_7` が潜在のみより良く、誤差logのSpearman .785 対 .778、log-RMSE .817 対 .854 だったため追加版を選んだ。

| fallback目標 | Devの実fallback率 | 閾値 b（推定腕誤差） | Dev腕6 MSE | 非fallback t0平均 | 非fallback t0 P10–P90 |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.0% | ∞ | .5342 | .490 | .189–.795 |
| 10% | 10.1% | 1.1508 | .5457 | .449 | .181–.714 |
| **20%（既定）** | **20.1%** | **.7129** | .5541 | .411 | .175–.633 |
| 30% | 30.1% | .5123 | .5585 | .376 | .165–.573 |

既定は依頼どおり20%。t0は固定Expertの `config.json` にある `t0_map_quantiles` を経験CDFとして `[.1,.9]` に写像し、ê≥bでfallbackする。Devでは候補を表にし、Evalでは20%設定だけを一度評価した。

## 手順3：オフライン Eval

腕誤差は正規化した最初10ステップの腕6軸二乗誤差和を70で割った値。E-VLM+H key、Expert、既定20%閾値は既存の固定checkpoint。gripperヘッドを適用し、fallback時は通常π0.5出力を使用した。

| 条件 | Eval腕6 MSE |
|---|---:|
| E-VLM+Hの下書き | .5619 |
| dist選択 t0 + Expert | .5466 |
| 潜在 t0 + 潜在gripper + fallback（RCX6） | .5776 |
| 通常π0.5 | .6375 |

RCX6は通常π0.5より低いMSEだが、下書き・dist Expertより高い。これはオフライン誤差であり、閉ループ成功率の代替指標ではない。

## 手順4–5：サーバーと閉ループ

サーバーに `--t0-policy latent`、`--latent-heads`、`--latent-fallback-rate` を追加した。各query logにはê、t0（fallback時はnull）、fallback、gripperの出どころ、base/modeの出どころを残す。dist版 `serve_latent_policy_dist_snap.py` は変更していない。JAX memory fraction .25 の初回はExpertを同時ロードできずOOMとなったため、同一条件を .35 で再起動して完走した。失敗ログも保存してある。

### 関門：PickPlaceSinkToCounter trial 0–9

| 条件 | 成功 | discordant（RCX6のみ / 参照のみ） | McNemar exact p |
|---|---:|---:|---:|
| RCX6 latent | 9/10 | — | — |
| π0.5 | 9/10 | 1 / 1 | 1.000 |
| dist P5（既存 closed/p5） | 10/10 | 0 / 1 | 1.000 |
| dist E-VLM+H（既存 closed/e_vlm_h） | 9/10 | 1 / 1 | 1.000 |

9/10で事前の8/10関門を通過したため、次の評価へ進んだ。少数10試行なので検定力は低い。

### 新規2タスク：trial 0–29

| タスク | RCX6 latent | π0.5 | dist E-VLM+H |
|---|---:|---:|---:|
| OpenCabinet | 1/30 | 10/30 | 10/30 |
| PickPlaceCounterToStove | 10/30 | 14/30 | 16/30 |
| **合計** | **11/60** | **24/60** | **26/60** |

同一seedの対比較：π0.5に対しdiscordant 6（RCX6のみ成功）/19（π0.5のみ成功）、McNemar p=.0146。dist E-VLM+Hに対し5/20、p=.0041。

### 6タスク：trial 0–29

| タスク | RCX6 latent | π0.5 | dist E-VLM+H |
|---|---:|---:|---:|
| PickPlaceCounterToCabinet | 12/30 | 15/30 | 14/30 |
| PickPlaceSinkToCounter | 23/30 | 28/30 | 26/30 |
| OpenDrawer | 20/30 | 19/30 | 23/30 |
| CloseFridge | 7/30 | 11/30 | 12/30 |
| TurnOnSinkFaucet | 9/30 | 20/30 | 16/30 |
| TurnOffStove | 1/30 | 2/30 | 2/30 |
| **合計** | **72/180** | **95/180** | **93/180** |

同じ task/trial/seed を全て照合した。π0.5に対するdiscordantは13（RCX6のみ）/36（π0.5のみ）、McNemar p=.0014。dist E-VLM+Hに対して13/34、p=.0031。いずれもRCX6 latentの成功数が少ない。

| タスク | fallback率 | 平均t0（非fallback） | query数 |
|---|---:|---:|---:|
| PickPlaceCounterToCabinet | 8.32% | .359 | 3713 |
| PickPlaceSinkToCounter | 3.58% | .355 | 3015 |
| OpenDrawer | 3.22% | .335 | 3011 |
| CloseFridge | 5.93% | .302 | 4992 |
| TurnOnSinkFaucet | 0.31% | .202 | 3264 |
| TurnOffStove | 0.45% | .222 | 4426 |
| **全体** | **3.75%** | **.291** | **22421** |

閉ループの実fallback率はDev目標20%を下回った（3.75%）。推定誤差の分布がDevからずれたことを示す。平均サーバー処理は約.282秒/query。12 workerでの全180試行のwall timeは約109分（試行当たり換算36.5秒、並列スループット）。

## 結論と限界

1. t0分布は段階的に広がり、Devのfallback率も目標に合わせられたが、閉ループではfallback率が3.75%にずれた。
2. 切替時刻では潜在ヘッドはRCX5固定MLP・π0.5に届かず、符号一致だけでは評価できない。
3. 2タスク・6タスクともRCX6条件はπ0.5およびdist E-VLM+Hより成功数が少なく、現状のヘッドで閉ループを改善できなかった。
4. これは6タスク由来メモリと固定key/Expertを用いたRoboCasaの結果であり、別タスクへの一般化を主張しない。
5. 1シード・限られたDevでhead/閾値を選んでおり、特に関門10試行の推定は不確か。

## 再現・成果物

- frozen E特徴、head、閾値、オフライン結果、閉ループquery/resultは `latent-heads/` 以下。
- 対応表とfallback/t0集計は [`closed_summary.json`](closed_summary.json)。
- サーバー起動例は [`run_condition.sh`](run_condition.sh)。6タスクrunは `run_condition.sh six 8846 'PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove' 30 '0 1 2 3 4 5' 12`。
- `tests/test_latent_heads.py`: 4 tests passed. `serve_latent_policy.py` と `latent_heads.py` のpy_compileも成功。
