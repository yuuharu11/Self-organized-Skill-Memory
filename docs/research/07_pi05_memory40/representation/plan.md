# 表現4-A：圧縮・座標系のオフライン診断（事前設定）

承認済み `plan_memory_gap.md` の表現4-Aを、小さなオフライン比較として実施する。保存内容が再利用時の行動精度を保つかを調べ、閉ループ性能の主張はしない。

## 固定データと候補

- run: `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01/`。Evalは同じ1,318行（40 task・80 episode）。難しい行は段階2でDev episode通常π0.5誤差 q80=0.015494を使って選ばれた141行を再利用する。
- 新規 codec の学習行は `~validation & ~memory` の16,721行に限定する。メモリ6,583行とDev/Eval 3,277行はfitに使わない。M1内部のvalidation分割も既定どおり維持する。
- 候補は `memory-gap-01` の `chosen_candidate_rows` を再利用し、同じepisode除外と同じtask除外の2通りを固定する。選択基準は同じ `oracle_action_mse` の生行動 active7 MSE。表現ごとに候補を選び直さない。
- Evalノイズは前段と同じseed 4101/4102、各行 `fold_in(key(seed), row_id)`、bridge key `fold_in(key(seed), batch_start)`、batch 4、10 step。t0=.2/.4。

## 表現

1. raw 行動列：候補の元 `[10,32]` をそのまま使う。保存数70 active値。
2. current b12+r4：既存 `run-01/codec.pt` と `bases.npy` のモデル・表現を読み取り専用で再利用。既存条件の参照値。
3. b12+r4 (matched fit)、b24+r4、b48+r4：M1。共通設定は現行 `run.py::codec` の epochs=30, batch=128, width=128, independence_weight=.01, beta_r=1, base_use_weight=5, margin=.05, decoder_obs=False, seed=20261104。fit rowsだけを `~validation & ~memory` に変える。base dimension以外は同じ。
4. EEF相対行動 b12+r4：上記と同じM1。LIBERO stateの先頭6次元はEEF位置とaxis-angle、action先頭6次元はワールド座標のΔ位置・Δ回転指令。公式q01/q99で物理尺度に戻してから、開始EEF回転 `R` で平行移動・回転ベクトルを `R.T` に写し、正規化し直す。復元時はqueryのEEF回転で `R` を掛け戻す。グリッパーとpad次元は変えない。
5. 非圧縮の状態差分：fit rowsだけで、正規化stateからactive7の10×7行動を予測するridge回帰（alpha=1.0、state列はfit集合の平均・標準偏差で標準化）。保存値は候補行動からそのstate予測を引いた70値。query stateの予測を足して復元する。保存値数はrawと同じ70。

M1は現行の `emergent/light/methods/m1.py` を使う。新規M1はtorch CPUで8 threadsでfitし、推論用pi05 Expertは再学習しない。現行codecは歴史的に `~validation`（memory行を含む）でfit済みの保存表現なので、新規fit variantsとの学習範囲の差を結果で明記する。追加でmatched-fit b12を置き、fitデータ範囲を合わせた基準も出す。

## 評価・記録

- 各表現・除外方式の最良候補をencodeまたは差分表現で保存し、queryの状態・featureからのpredicted residual（M1）でdecode。EE相対表現のみ、復元後にquery座標系へ戻す。
- codecの再構成MSEはメモリ6,583行で、各行のposterior b/rから自己復元したactive7 MSE。新規codecではfit外のmemory全行で測る。raw/ridge差分は復元定義上の誤差も記録する。
- 最終MSEは全Evalおよび段階2の難しい141行で平均・中央値を出す。表現間の差は同じquery/noise対応でepisode bootstrap 5,000回。
- 保存量は区間ごとの連続数値の個数（モデル重み・metadataは除く）。現行b12+r4=16、b24+r4=28、b48+r4=52、raw/状態差分=70。
- 事前に「成功」の数値閾値は追加しない。rawに比べて圧縮がどの程度行動誤差を増やすか、relativeが現行b12より減らすかを数値で報告する。Expertはb12既存下書き分布で学習済みなので、他表現の差はcodec復元とExpert入力分布の双方を含む。

## 予測可能な限界

raw候補は行動距離で選んでいるoracle上限で、実行時の検索性能ではない。相対変換はLIBEROのEEF基準・OSC delta指令に基づく物理座標の回転であり、タスクラベル・サブタスクラベルは使わない。

## 実装確認の追記

M1の実運用経路を確認すると、各メモリ項目に保存するのはbase `b` で、residual `r` はその時のqueryの `prior(feature,state)` が予測していた。したがって計画本文の「b+rの合計値数」はメモリ項目の容量としては過大計数だった。結果では実際に保存するb次元（12/24/48）を容量として数える。モデル重みは全方式で別管理。これは結果を見た後の性能設定変更ではなく、保存契約の数え方の訂正である。
