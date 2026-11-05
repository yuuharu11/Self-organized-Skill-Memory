# Attention-key mechanism: 実演メモリの選択

## 問い・事前判定

未知Long場面の入力から、単一の実演チャンクを混ぜずに選べるかを、学習した視覚アテンション付きキーと候補行動判定器で測る。結果は生成前・生成後オフライン行動誤差であり、閉ループ成功率ではない。

- **K1:** key-Devで選んだ1モデルが、wrist_lowerhalf+prevよりTop-1 MSEを改善し、全Longまたはhardの少なくとも一方で episode-bootstrap差の95%区間上端が0未満。
- **K2:** 同モデルがhold-lastを全Long・hardの両方で改善し、両95%区間上端が0未満。
- **K3:** 凍結Expertの生成後MSEがzero下書きより低く、t0=.4または.6の全Longまたはhardで差の95%区間上端が0未満。
- **K4:** 信頼度と誤差の関係が十分強く、信頼度上位20%の誤差が全体より低い（95%区間上端<0）。信頼度は大きいほど採用候補への確信が高いので、Spearman(confidence, error) は負方向を良い関係として記録し、反転したSpearman(confidence, -error)も報告する。

直接回帰は判定に含めず、同じ学習行・同じM1表現情報の参考比較とする。全ての差は同一行のpaired episode bootstrap 5,000回。

## 分割

- 元データ: `pi05-memory40-20261104/run-01`。保存済みVLM token cacheをmemmapで読み、VLMを再抽出しない。
- 長期テスト: Long Eval 548行、hard 63行、固定Dev q80閾値 `0.01549410422449`。Long task ID 0–9は学習・選択・メモリ構築から除外する。
- 生メモリ: `skill-memories-01/raw-rows-noLong` と同じ3,857行、正規化行動10×7。
- key-Dev holdout: 乱数seed 20261105で suiteごとに2タスクずつ固定し、task IDs **17, 19 (goal), 28, 29 (object), 35, 38 (spatial)**。その全taskのvalidation行を選択に使用する。これらのタスクは勾配学習に使わず、key-Dev候補プールからも全rowを除外する。
- 学習: それ以外の24 nonLong tasksの非memory・非validation rows。key-DevとLongはfitに使わない。key-Devでモデルを選んだ後に再fitしない。
- queryと候補の同一taskは常に除外する。

## 比較機構

1. **M1, attention pooling:** 同じネットワークをqueryとmemoryに用い、状態[:8]+直前実行action[10×7]で条件付けた複数learned queryが、base+wristの512画像token（row-major patch位置を加算）をcross-attend。出力embeddingでsoft contrastive retrieval。4/8 slotsをDev選択。言語tokenなしを主条件、text tokenを足す4-slot版は参考条件。
2. **M2, weighted MaxSim:** tokenごとのsalienceを状態/履歴で学習し、低次元tokenのweighted max-simを評価する。全候補への計算を絞るため、同じ学習表現のpooled keyで候補を粗く絞り、MaxSimでTop-64を再順位づけする。学習損失はcross-task soft InfoNCE。
3. **M3, candidate judge:** M1 Top-64上で、query image tokenへのcross-attentionを候補の生action[10×7]+候補初期状態[:8]で条件付け、queryとの行動MSEの対数を予測する。教師はtrain rowsの行動距離だけで、入力にqueryの正解行動を含めない。出力は再順位後の1候補。

M1のsoft InfoNCE positiveは別taskかつ未来10-step action MSEがDev-selected threshold以下のpair。threshold候補は`.005/.01/.02`、Dev上cross-task positive比率が事前指定目標に最も近い値を採用する。学習時に複数行動を混ぜず、推論はTop-1の単一実演を返す。

## 指標・対照

生成前: Top-1 MSE、Top-5中最良、Recall@8 (MSE≤.01)、query内Spearman(score, action MSE)、xyz合成方向cosine、gripper符号一致率。信頼度についてSpearman(confidence, Top-1 error)、反転符号、上位20/40/60/80% subsetの誤差と上位20%差CIを出す。

同じLong rowsにG、wrist_lowerhalf+prev、invariant_projected_local、hold-last（直前chunkの終端actionを10step保持）、前chunkコピー、random one, action oracle、direct regression、zero draft、normal π0.5を並べる。Zero / normalの生成前の手法固有動作ではなく、保存済みTopK Long生成出力の単一noise baselineを使う。

生成後: key-DevでTop-2となったM1/M2/M3候補、wrist_lowerhalf+prev、直接回帰、zeroを凍結`run-01/bridge/001500`へ入力。t0=.4/.6、noise seed4101、10 integration steps。同一queryでpaired比較し、通常π0.5も基準に含める。

## 可視化・制約

Longのhard/easyから各1 episodeを選び、episode時刻の前半/中間/後半のbase・wrist画像にM1 attentionを重畳する。タスク・計画文・成功ラベルは学習にもモデル選択にも使わず、絵の解釈時だけ参照する。π0.5 VLMはLong学習済みであり、ここで未知なのはキー学習・メモリ側のタスクである。

CPUはOMP/MKL/OpenBLAS=16、torch threads=16、`nice -n 10 taskset -c 8-47`。GPUは`XLA_PYTHON_CLIENT_PREALLOCATE=false`、JAX memory fraction .30以下。計算と出力は `key-mechanism-01/` に保存し、学習・評価はtmuxから実行する。
