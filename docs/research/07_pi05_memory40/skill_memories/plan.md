# スキルメモリの複数表現：事前設定

直前の固定個数クラスタ案（k-means）は取り消し、個数を指定しないM6型の逐次登録に切り替える。区間は `run-01` の10-step/stride10。入力行・重み・特徴は読み取り専用で扱う。Expert学習とGPU推論はしない。

## 分割と漏洩防止

- 行データは `.skillnet-state/.../run-01/rows`、Long行はtask id 0–9（`topk-long-01` の task mapping 検査と同じ）とする。
- memory pool は `rows['memory']`。noLongの全構築では pool/query/代表/評価選択にLong task 0–9を含めず、実行時にassertする。Long Evalは `memory-gap-01` の固定Eval行からtask 0–9だけを選び、既存 `topk-long-01/production` と query row順およびhard maskを一致assertする。
- κ選択用Devは Long以外のvalidation episodeから taskごとに固定seedで3 episodeを選ぶ。候補κは `.0125,.025,.05,.1,.2,.4,.8`。Dev actionだけでM6型登録を試し、κ=.1を中央候補として固定し、κ<.1からDev登録数が300に最も近い値、κ>.1から30に最も近い値を選ぶ（同距離なら小さいκ）。この3値のDev登録数が単調で、少なくとも2つ異なることを確認し、Long Evalを選択に使わない。数十〜数百に届かない場合も値を結果を見て調整せず、Dev計測を報告する。
- rawの機能距離はnormalized active7の10×7 MSE。M6既定の `epsilon = kappa × median(random-pair functional distance)`、min_support=10、min_episodes=3、pending batch=200、max age=1000、seeded random stream、unsupported pending登録/mergeを使う。単位重心は所属判定・merge用でのみ使い、保存代表は単位内medoid（所属実区間の平均機能距離最小）とする。
- b版は `run-01/bases.npy` を値にし、距離はM6既定のfunctional distance（Longを含まない当該memory poolからseed固定で選ぶ64 probeの `codec.decode(b, prior_r, feature, state)` 出力を全次元・全probeで正規化した埋め込みの二乗L2＝平均復元行動MSE）。`_nearest` と同じbase-L2上位10前置きも適用する。b版のmedoidも同機能距離で選ぶ。
- run-01のM1重み/basesは再学習せず利用する。したがってnoLong poolにLong行は入りませんが、固定M1 codec自身はrun-01の非validation学習データ（Longを含む）でfit済みであることを明記する。

## 作るメモリ

1. `raw-rows-noLong`: Long以外のmemory行を1項目ずつ生行動保存。
2. `raw-rows-all40`: 全40タスクのmemory行を1項目ずつ生行動保存。
3. `raw-m6-noLong-k*`: noLong memory poolをκ3値でM6登録。
4. `raw-m6-all40`: 全40 memory poolを中央κでM6登録。
5. `b12-m6-noLong`: noLong poolをb12でM6登録、既存M6 functional distance。

単位ID/medoid row ID/action、所属row IDとoffset、単位ごとのepisode/task数、row-to-unit（未割当は-1）、κ/ε/未割当率/input hash/seedを保存する。各medoidのキー用wrist token meanとstate[:8]も、`retrieval-4b`の定義で保存する。

## 診断

Long Eval 548行・固定hard 63行で、各メモリの正解行動から最近傍medoidまでの距離と、上位5medoid内の最良距離を計算する。b版は各queryの `prior` residualとfeature/stateで代表bをdecodeしてから距離を取る。通常π0.5 singleとzero draftは保存済みtopk-long値を同じrow orderで参照し、再推論しない。単位task-count/長さ分布と代表actionのgrip端点・平均Δxyzの方向例も記録する。

## 事前所見の扱い

この診断のraw行動最近傍は保存・代表単位のカバレッジ上限で、実行時に見つかる距離ではない。ルーターは学習しない。行動誤差の改善を閉ループ成功率の改善とは記述しない。「統計的な差がない」を「同等」と書かない。
