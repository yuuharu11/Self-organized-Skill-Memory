# 未学習LIBEROタスクでの表現上限比較：事前計画

## 固定する比較

LIBERO-90 の未学習タスク実演を query として、LIBERO-40 の固定メモリ 6,583 窓から正解行動に最も近い1候補を選ぶ。これは検索器の評価ではなく、正解を用いる候補選択上限であり、閉ループでも汎化性能の主張でもない。候補下書きは raw、M1 b12、PCA24、PCA12、時間DCT k=2/3（14/21係数）、時間キーフレーム k=3/5（21/35値）、ゼロのみに限定する。クラスタリングはなし／M6既定の2条件。クラスタの代表は実在するメモリ窓のmedoidとする。評価窓の行動を直接参照して最小有効7次元MSEの候補を選ぶため、選択誤差と表現・復元誤差を混同しない。

## データ分割と学習

未学習 query タスクは既存 `unseen-libero/baseline-01/decision.json` の10タスクを固定し、各タスク10エピソードをEvalに使う。難しい行のしきい値は、別の未学習保留タスクである公式LIBERO-90 task id 6と7（それぞれ10エピソード）のDev行における、raw候補の最小行動MSEの80パーセンタイルとして固定する。Dev行はExpertの選択（保存更新数とt0）にだけ使い、Expertの勾配更新には入れない。Eval行はメモリ、PCA適合、Expert学習、Dev選択に使わない。両タスク群はLIBERO-40の40指示文との正規化後の完全一致がなく、Longと同じ場面を除外した未学習候補から機械的に選び、成功率は選択に使わない。

実演源はLIBERO-90公式デモHDF5。各選択タスクからdemo_0..demo_9を使用し、実データ時系列を10 Hzに揃えてから10フレーム窓・stride 10に切る。pi05_liberoのasset norm stats、LIBERO入力変換、224×224画像変換、VLM prefix token抽出を用いる。LIBERO-40のrun-01のtrain（validation=false）でPCAをfitし、同じtrain行ID列・seed 20261104・batch 8・AdamW 2.5e-5・4,000 update・500 updateごとのcheckpoint・t0一様[.2,.6]を全条件で共有する。初期重みは通常Expert。下書き表現以外の学習条件は変えない。Dev生成後MSEでcheckpointとt0∈{.2,.4,.6}を選び、Evalにはその組を一度だけ適用する。

## 評価と比較

Evalノイズはseed 4101とrow IDからfold_inし、10 Euler積分。難しい行はDev q80の固定しきい値で選ぶ。比較は生成前MSE、復元誤差、MSE≤.04率、軌道方向cos、通常π0.5、同一Expertのt0=1、別タスク置換シャッフル。主な対比較は各表現−raw、各表現−同一Expert t0=1、各表現−shuffleで、episode単位5,000回bootstrap 95%区間を出す。時間DCTとキーフレームの候補次元はどちらも事前に定めた全値を報告し、Devで選択しない。
