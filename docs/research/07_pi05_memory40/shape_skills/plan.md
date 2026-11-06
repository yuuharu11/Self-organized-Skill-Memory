# shape_skills A：型表現・可変長区間の比較計画

## 問いと事前判定

正規化行動のMSEを最終指標にし、姿勢と大きさを物理空間で分離した場合に、型だけの共有が固定長rawを超えるかを調べる。閉ループ評価ではなく、10-step行動窓に対するoracle診断である。差が有意でない場合に同等とは書かない。

fit/辞書構築から Long task 0–9 と保留6 task `[17, 19, 28, 29, 35, 38]` を完全に除外する。保留6 task の validation episode だけで区間しきい値と M6 の粒度候補を選ぶ。Long Eval の固定548 query / hard63 mask は最後の一回の評価にだけ使う。

## 固定入力・物理変換

入力は run-01 の episode/task/frame 行、および manifest に記された unsegmented parquet の actions/state を読む。行の正規化 active7 と一致するか、各 episode の先頭窓で検査する。q01/q99 を使い、回転・スケール操作は物理量へ戻した Δ位置と Δ回転の6次元にだけ適用し、同じ proper rotation を両者に掛ける。gripper は回転せず別チャンネルとして保持する。変換後に公式 q01/q99 で正規化する。

「型」は開始点を原点にした累積Δ位置・Δ回転軌跡を位相長32に再標本化したもの。方向は総Δ位置を +X に合わせる最小回転、スケールは総Δ位置ノルムで除く。総移動がほぼ0の場合は6次元軌跡RMS、なお0なら単位スケール・identity回転を使う。逆変換は保存poseから行う。Procrustes条件は proper-rotation Kabsch で参照平均軌跡へ整列する。これは実行時には未知の正解poseを使う上限診断である。

## 条件と実装上の定義

- S0: 10 raw frames 固定。S1: 10-frame線形軌跡再構成誤差がε以下となる貪欲最大延長と、同制約でセグメント数を最小化するDP（ε=.005/.01/.02/.04）。どの最短窓もεを満たさないepisodeはε適合を偽装せず、固定10step窓へfallbackする。S2: gripper sign切替。S2v: gripper切替＋平滑速度局所極小。速度極小のprominenceはholdout-Devの区間長中央値が10–40 frameに近づくよう、候補prominenceから選ぶ。S3: 10-frameの局所型 descriptor（physical Δ6の単位方向系列）変化ピーク。全て最短10 frame、episode境界で切る。
- R1: phase 32点へ揃えたraw action PCA、素/score白色化/gripper分離、k=8/12/16/24/32。R1固定S0も同じPCA family。
- R2: phase 32点の累積軌跡をproper Procrustes整列後、PCA k=8/12/16/24/32。pose(scale, rotation, original length, endpoint)を別保存。
- R3: 6次元軌跡に対するGaussian-basis DMP forcing weights（B=5/10/20）、goal/tauはposeに分離。
- R4: canonical 6次元軌跡を16制御点の線形スプラインで近似し、制御点ベクトルをTrain-fit PCAで実コード次元8/16/24へ射影してdecode時に逆射影する。
- R5: impl-g2 が同じ出力ルート内に作成した masked EMA VQ-VAE bundle を読み込む。実行前の探索で raw/Procrustes・K256/residual={0,4,8} のうち、当方の全segmentキー集合と完全一致する組だけDevの同じM6/medoid指標で評価し、R5を1 familyとして機械選択に含める。K1024はCPU予算外。Procrustes条件は新しい外部queryを再符号化できる参照軌跡がbundleに無いが、本評価は固定queryを含む事前符号化コード・poseを行キー完全一致で再利用する。外部bundleは読み取り専用。
- M1: run-01設定のM1 b12/b96を、学習可能行動rowで各一回fit。可変長区間は区間内最初のglobal 10-step rowをcodec入力・復元窓に使い、そのrowが無い区間では同episodeの直前最寄りrowを条件に使う。M1固定codec対照として解釈する。pose sidecarにposterior residual/state/featureを保存するため、bだけの軽量表現ではない。

単位化はk-meansなしのM6 assign/pending/register/merge方式（min_support=10、min_episodes=3）。表現codeの全次元MSEを距離とし、κ={.025,.05,.1,.2,.4,.8}を振ってデータ依存の単位数曲線を得る。medoidはcode距離の単位内平均が最小の実区間。min_support=10のため、3857件以下のmemory poolで1000単位以上は定義上到達できない。Dev選択は各configurationのDev queryで、正解poseを使ったmedoidの最終10-step active7 MSEと、誤差≤.04の割合を測る。固定S0-R1-PCA24とM1-b12を必ず含め、その他は `score = rank(aligned_MSE ascending) + rank(coverage_≤.04 descending)` が小さい順に最大4件を追加する。同じrepresentation familyからは1件まで。Longは選択に使わない。

## 出力と診断

各representationには segments.npz(task,episode,start,length,row_ids), codes.npy, 必要ならpose.npz, reconstruct.py::decode(codes,pose,length), manifest.json を置く。row_idsは区間内で最初に始まるrun-01の10-step窓ID（なければ-1）。行動復元と型距離は、Long各窓・hard maskで、型のみidentity/unit scale、oracle pose alignment、無分離raw、および無関係に選んだ型をqueryと同じscale/rotationで整列した距離を出す。ランダム型との差を必ず主値として示す。距離≤.04/.02率と、タスクをまたぐmedoid単位率を報告する。Longを使うのはこの最後の評価のみ。

## 実装訂正と再評価

Longの仮評価で一般角度の `rotation_to_x` が非直交になる不具合を検出した。Rodrigues式で sin(theta) の係数が欠けていたためで、一般方向の回転・往復テストを追加して修正した。影響しうる正準化コード・pose・型合わせの数値は使わず、同じ事前選択規則で新しい `shape-skills-01-corrected` 出力を作る。旧 `shape-skills-01` は監査用に保持する。既存VQ束はraw入力だけを読み取り専用で再利用する（rawの符号化はSO(3)正準化を通らない）。Procrustes VQ束は修正前回転の影響を排除できないため使わない。選択規則・しきい値・Long maskは変更しない。

## 明示的な非圧縮型対照

実装監査で、初版の表現グリッドには「圧縮せずに状態姿勢・移動量を分離した型を保存する」対照が独立条件としてなかったため、Longの追加集計より前に `canonical_raw` を追加した。これは物理指令の平行移動Δ位置・Δ回転を物理空間で正準化し、位相32点の6次元軌跡（192 float）をそのまま保存する。スケール・SO(3)回転・元の長さ・グリッパー系列は pose に保持し、回転・スケールは q01/q99 で物理空間へ戻した後にだけ適用する。通常のDev選択規則（各切り方でκ候補を選び、全設定のDev順位から必須2件を含む最大6件）に加え、他の表現と同様に扱う。Longを見た選択は行わない。

この追加条件を含む最終出力は `shape-skills-01-corrected2`。その前の `shape-skills-01-corrected` は回転修正後・非圧縮対照追加前の監査用出力として残す。

## 再実行計画（shape-skills-02）

旧出力の `selected_configs.json` とLong指標は再利用しない。固定窓PCAは各Train行の公式q01/q99正規化済み70値を直接fitし、Long/heldout Devはfitから除く。最初に保存行動と行IDから、固定raw最近傍（全/hard .0290/.0351）およびTrain-fit固定PCA24復元（全/hard .0009/.0021）を独立再計算し、許容差0.001以内で照合する。不一致時は原因を解消するまでグリッドとLong評価を開始しない。

各表現×切り方のDev M6曲線は、単位数が約30、300、3000となるκをデータから探索する。κはしきい値を決めるだけで単位数を強制しない。通常min_support=10/min_episodes=3で候補点を探し、300以上の粒度に届かない場合に限ってmin_support=1/min_episodes=1の全所属版も探索する。届かない水準は未到達として記録する。Devは保留6タスクのvalidation episodeから固定間引きした105行で、粒度比較に使う全曲線の予測が有効であることを確認する。

設定の選択はLongを見ずに行う。30/300/3000各水準内で、Devのoracle-aligned MSEと≤.04被覆率の順位和が最小の設定を最大2つずつ（計最大6）選ぶ。同じ表現ファミリーは各水準で重複させない。固定raw PCA24、M1 b12、canonical PCAの全候補中最良、修正後回転で再学習したVQのDev最良を必須追加する（必須枠は6枠上限の外）。VQのraw入力と標準形入力は各々の候補をDevだけで比較し、各1設定をBへ渡す。Procrustes VQおよびAEは新しいproper rotation実装で再生成する。S1 DPが最小長制約を満たせず固定10窓へfallbackしたエピソードは、可変長DPとしての成功区間に数えず、fallback件数を別記する。

選択規則は事後のLong成績を見て変えない。図・表・計算記録は新規出力 `shape-skills-02/` に保存する。

masked AE（raw / procrustes_shape / raw_invariant、全latent次元）も、再生成束を共通M6/Dev粒度曲線へ取り込む。必須選択枠ではないが、Longを見る前に同じ30/300/3000のDev grain評価へ含める。AEの自己再構成値は、Medoid選択性能と区別して報告する。
