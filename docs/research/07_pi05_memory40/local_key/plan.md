# 局所キーによる Long 未知タスク検索

実施日: 2026-11-05

## 問いと判定

手先周辺の局所的な視覚・状態キーは、Long以外の30タスクだけから作った生行動メモリを、Long Evalの未知タスクで適切に検索できるかを測る。対象は生成前の行動距離であり、閉ループ成功率は扱わない。

- **L1:** 局所キーのTop-1 active7 MSEが全体キーGより低い。hardまたは全Longの少なくとも一方で、episode bootstrap差（局所−G）の95%区間上端が0未満。
- **L2（核心）:** 局所キーのTop-1 MSEが、同一入力情報の直接行動回帰より低い。hardまたは全Longの少なくとも一方で95%区間上端が0未満。
- 区間推定はLong episodeを再標本化する。予測器・キー選択・Dev上の領域選択にLongを使わない。CIは各クエリを一単位として再標本化せずepisode単位で算出する。

## 固定データ・分割

- 入力スナップショット: `pi05-memory40-20261104/run-01`。トークンは `[26581,712,2048]` float16 memmap、カメラ画像 `[224,224,3]`。
- メモリは `skill-memories-01/raw-rows-noLong` と同じ `memory` 行のうちtask 0–9を除いた3,857行。保存値は正規化active7行動 `[10,7]`。
- クエリは memory-gap固定Long Eval 548行、hard 63行。hard閾値はDev q80 `0.01549410422449`。
- すべての学習・標準化・射影・領域選択・閾値選択は task 10–39のみ。π0.5本体はLong学習済みだが、ここでの判定対象は未知タスクでのキー／メモリ検索である。
- Dev領域選択は、30タスクのvalidation行から既存 `calibration_episodes` が返すDev episodeだけを使う。直接行動回帰と不変射影はLong以外の非memory・非validation行でfitする。長さ10・stride10の同一episode近接窓を同一foldに保つ。

## 比較条件

1. **G:** base+wrist 512画像トークン平均。現行のグループ別token MaxSimも参照条件。
2. **W:** wrist 256トークン平均。
3. **W-local:** wristの中央4×4、中央8×8、下半分パッチ平均。Dev上のTop-1 MSEで1種類を固定してLongに適用する。
4. **B-EEF:** base cameraに投影したEEF点の周囲3×3、5×5 patch平均。16×16 token gridは画像とのoverlayで配置確認。入力画像は記録前に180度回転しているため、投影座標も反転後の224×224へ写す。
5. **Att:** action-expertから画像tokenへの最初のdenoise stepの層・head平均attentionで重み付けする。重み抽出にGPUが必要な場合は、GPU空きと所要時間を確認し、1–4の結果を先に確定した後にのみ追加する。
6. **Projection:** W-localまたはB-EEFのDev最良特徴を小さなMLPで写像し、別タスクで未来10step行動MSEがDev選定τ以下の対を正例として対照学習。τ・射影の停止点はDevのみで決定する。特徴入力は局所キーのみ。

各キーに `+state[:8]` および `+previous_action[10,7]` を加えた版を作る。Top-1 MSE、Top-5最良MSE、Recall@8（行動MSE≤.01）、Top-64 Spearman(score, action MSE)を、全Long・hardで集計する。Gと局所キーとの差、および同一情報のdirect regressorとの差にはepisode bootstrap 5,000回の95%区間を付ける。oracleは全3,857行の行動MSE最小、ゼロと通常π0.5は生成後参照として分けて記載する。

## 実装・資源

全件 token 読み込みはmemmapで必要な行・領域のみとする。CPUは8 threads、`nice -n 10 taskset -c 8-47`。VLM注意重み取得のみGPU候補とし、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、使用量・ピーク制約を確認する。訂正版の結果と図は `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/local-key-02/` に新規作成する。local-key-01は初回の誤ったMSE集計の記録として保持する。

## 訂正版の実行プロトコル（local-key-02）

L1/L2の上記定義・閾値は変更しない。初回実装のプール位置と全データ行IDの混同を修正し、Top-K行動誤差は `actions[memory_row_ids[pool_position]]` から計算する。回帰テストではプール順と行IDを意図的に異ならせる。

状態・履歴を追加するキーは各ブロックをLong以外の非memory学習行でz-score標準化し、ブロックごとにL2正規化した後、重みの平方根を掛けて連結し、最終L2正規化する。視覚:状態:履歴の候補重みは事前に固定した少数格子から、Long以外のDevだけで同一タスク除外を適用して選ぶ。領域選択も同じタスク除外で行う。評価時には同じ重み・統計を固定する。

生成後の追加評価では、Long Evalの正解行動やタスクラベルでキーを選ばない。Devで選んだ局所キー、Devで選ばれたEEFパッチ、射影局所キーのTop-1生行動を凍結`best_bridge`へ渡す。選択済みの局所キーと同一入力の直接回帰出力（active7を残り25次元ゼロ埋め）、zero下書きも、t0=.4/.6・noise seed 4101で生成する。通常π0.5単発と併記する。これは生成後のオフライン行動誤差であり閉ループ評価ではない。
