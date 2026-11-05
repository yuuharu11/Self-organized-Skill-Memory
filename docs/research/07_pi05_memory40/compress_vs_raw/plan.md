# Compress vs raw:公平なM2検索下書きの圧縮比較

実行前固定。目的は、実際に検索した同一M2候補をb12で復元すると、raw候補より生成後のLong誤差が改善するかを調べること。オフライン行動誤差のみで、閉ループ成功率を主張しない。

## 学習

- E-M2の学習成果・設定を `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/fair-expert-01/` から読み取り専用で使う。新しく学習するのはE-b12のみ。
- seed `20261104`、固定 `batch_row_ids.npy`（4,000×8）、同一初期公開π0.5、optimizer・損失・500更新チェックポイント・Dev評価（6 holdout task、noise fold-in 88261、t0 `.2/.4/.6`）をE-M2と揃える。
- E-M2 `training_drafts.npz` の `selected_candidate_rows` をqueryごとそのまま使う。圧縮側のみ、候補row IDの `bases.npy` をM1 codecで query prior residual・pooled feature・state と共にdecodeする。`candidate_row_id` はデータセット行IDでありpool内offsetではない。
- Devの候補・順位混合は候補順位seed `20263006` とE-M2の同じサンプルを使う。Longはcheckpoint/t0/model selectionに使わない。
- codecはrun-01で全40タスク（Longを含む）からfit済み。圧縮表現の生成はこの制約を継承し、独立なLong未見codecとは扱わない。

## 評価

- 固定Long Eval 548行、hard 63行（Dev q80 `0.01549410422449`）、同じrow ID順、10積分、noise `fold_in(key(4101), row_id)`。
- E-M2(raw)とE-b12の各選択checkpoint/t0で、それぞれの通常M2 top1、正解行動oracle候補（全3,857 noLong行から生行動距離最小）、fair-expertのTop-1 source permutation shuffle、zero下書きを比較。raw/decodedのM2・oracle・shuffleは各pairで同じ候補IDを使う。
- E-zero+zeroと公式π0.5はfair-expert既存予測を読み取り専用で再利用し、row ID・target・noiseの同一性をassertする。
- 生成前にM2 top1/oracleのraw距離、b12 decode距離、raw対decodeの圧縮誤差を出す。Expertごとの候補raw対shuffle出力距離/下書き距離を感度とする。
- 差のCIはepisode cluster bootstrap 5,000回。各ExpertでDev選択されたt0に対してC1–C3を一度だけ判定する。

## 事前判定

- **C1:** `(E-raw + M2) - (E-b12 + M2)`。全Long/hardとも、95% CI上端<0ならraw優位、下端>0なら圧縮優位、それ以外は差を確認できない。
- **C2:** 同じ差を同一oracle候補のraw/decode下書きで計算。正しい候補を得た場合のcodec損失を評価する。
- **C3:** `(E-b12 + M2) - (E-b12 + shuffle)`。95% CI上端<0なら圧縮下書きも検索内容を使っている。
- `decision.json` にall/hard各判定とDev-selected checkpoint/t0を記録する。1 seedの学習であり、CI重複を同等性とは表現しない。

## 容量記録

rawは70 active値/row、b12は12 base値/row（decoderとpriorは共有モデルとして別計上）。3,857行のfloat32でのraw/b12値本体サイズとrun-01 token key cache (`tokens.npy`) のサイズを記録する。対象メモリは実際は32 action次元を保存するため、格納ファイル実サイズも別記する。

## 拡張：次元 b12m / b24 / b48 / b96（実行前固定）

- 比較用codecのfit行を `~validation & ~memory`（16,721行）にそろえる。Long taskはこの集合から除かないため codec fitには含まれ、結果の限界として明記する。
- representation-4a-01 の `fresh_b12r4.pt`（ここではb12m）、`b24r4.pt`、`b48r4.pt` を設定・fit行が一致することを確認して再利用する。b96は同じM1設定（epochs=30, batch=128, width=128, HSIC=.01, beta_r=1, base_use=5, margin=.05, decoder_obs=False, seed=20261104, residual=4）でCPU fitする。その他のfitが30 epoch未満で早期停止した場合も、patienceを30 epoch以上にした版を追加学習し、後述の保留行MSEで選ぶ。
- b48について早期停止14 epoch版と早期停止なし30 epoch版を比較する。codec選択は`validation & task>=10`だけの自己復元active7 MSEで行い、Long Evalは選択に使わない。両方の値を保存・報告する。他の次元も同じ非Long保留validation行で自己復元を報告する。
- メモリ／候補値はそれぞれのcodecの `encode` で作ったb、queryの `prior(feature,state)` が返すr̂を用いてdecodeする。候補・メモリpoolは既存noLong 3,857行。E-M2の学習候補row ID、4,000バッチ列、seed、optimizer、checkpoint頻度、Dev候補・ノイズを共通にし、下書きのみ次元ごとに変える。Expertはb12m/b24/b48/b96各1 seed。
- 既存の初回compress-vs-raw出力のshuffleはレビューで訂正済み。評価には訂正版 `compress-vs-raw-review-20261105-01` のTop-1置換だけを使い、初回出力は読み取り専用で保持する。
- Long Eval 548/hard 63、t0=.2/.4/.6について各次元のM2/oracle/shuffle/zero生成後MSE、自己復元、候補下書き誤差、rawとの差分、出力感度、保存値数・byteを出す。E-rawと既存run-01-codec E-b12を同表に載せる。
- C1/C2/C3は各次元でDev-selected t0を使う。CIはepisode bootstrap 5,000回。各C×subset（全Long/hard）内の4次元をHolm補正する。方向判定はCI上端<0で左(raw)優位、下端>0で右(compressed)優位、それ以外は差を確認できず。Holm補正後p<.05も別記し、未補正CIの方向と混同しない。
- 結果は `compress-vs-raw-01/dimension-extension-01/` とし、既存E-M2/E-b12および訂正版レビュー出力を上書きしない。
