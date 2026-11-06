# 未学習LIBEROタスクの通常方策：実行前計画

## 段階1：メモリ表現の検索なし上限比較（事前固定）

LIBERO-90 の未学習タスク実演を query として、LIBERO-40 の固定メモリ 6,583 窓から正解行動に最も近い1候補を選ぶ。これは検索器の評価ではなく、正解を用いる候補選択上限であり、閉ループでも汎化性能の主張でもない。候補下書きは raw、M1 b12、PCA24、PCA12、時間DCT k=2/3（14/21係数）、時間キーフレーム k=3/5（21/35値）、ゼロのみに限定する。クラスタリングはなし／M6既定の2条件。クラスタの代表は実在するメモリ窓のmedoidとする。評価窓の行動を直接参照して最小有効7次元MSEの候補を選ぶため、選択誤差と表現・復元誤差を混同しない。

未学習 query タスクは既存 `unseen-libero/baseline-01/decision.json` の10タスクを固定し、各タスク10エピソードをEvalに使う。難しい行のしきい値は、別の未学習保留タスクである公式LIBERO-90 task id 6と7（それぞれ10エピソード）のDev行における、raw候補の最小行動MSEの80パーセンタイルとして固定する。Dev行はExpertの選択（保存更新数とt0）にだけ使い、Expertの勾配更新には入れない。Eval行はメモリ、PCA適合、Expert学習、Dev選択に使わない。両タスク群はLIBERO-40の40指示文との正規化後の完全一致がなく、Longと同じ場面を除外した未学習候補から機械的に選び、成功率は選択に使わない。

実演源はLIBERO-90公式デモHDF5。各選択タスクからdemo_0..demo_9を使用し、実データ時系列を10 Hzに揃えてから10フレーム窓・stride 10に切る。pi05_liberoのasset norm stats、LIBERO入力変換、224×224画像変換、VLM prefix token抽出を用いる。LIBERO-40のrun-01のtrain（validation=false）でPCAをfitし、同じtrain行ID列・seed 20261104・batch 8・AdamW 2.5e-5・4,000 update・500 updateごとのcheckpoint・t0一様[.2,.6]を全条件で共有する。初期重みは通常Expert。下書き表現以外の学習条件は変えない。Dev生成後MSEでcheckpointとt0∈{.2,.4,.6}を選び、Evalにはその組を一度だけ適用する。

Evalノイズはseed 4101とrow IDからfold_inし、10 Euler積分。難しい行はDev q80の固定しきい値で選ぶ。比較は生成前MSE、復元誤差、MSE≤.04率、軌道方向cos、通常π0.5、同一Expertのt0=1、別タスク置換シャッフル。主な対比較は各表現−raw、各表現−同一Expert t0=1、各表現−shuffleで、episode単位5,000回bootstrap 95%区間を出す。時間DCTとキーフレームの候補次元はどちらも事前に定めた全値を報告し、Devで選択しない。

## 問いと学習データの確認

LIBERO-40で作った運動を未学習タスクで試す前に、公開pi05_liberoの通常生成に改善の余地があるかを測る。学習・メモリ・Expert overlayは一切使わない。

[公開評価README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) は配布pi05_libero重みと同名configを対応付けている。[公開config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) は repo_id=physical-intelligence/libero、pi05=True、action_horizon=10、discrete_state_input=False、extra_delta_transform=False、pi05_baseからの微調整を指定する。[公開データカード](https://huggingface.co/datasets/physical-intelligence/libero) と保存したtasks/infoはSpatial・Object・Goal・Longの40タスク、1693 episodeを示す。公開tasksの40指示文と、インストール済み4 suiteの指示文集合は完全一致した。

ローカルのtraining/config.pyはSkillNet configへの互換aliasであり、このdense公開configを保持していない。そのため公開openpi configも取得して照合した。チェックポイントのassets/physical-intelligence/libero/norm_stats.jsonをそのまま使う。asset名はrepoとの対応を補強するが、norm statsだけで学習履歴を証明するものではない。

**未学習の範囲は「公開LIBERO微調整データの40タスクに含まれない」まで。** pi05_baseなど全事前学習のタスク履歴は公開資料から網羅確認できず、全履歴での未学習は未確定。LIBERO-90全体には40と指示文が同じ3タスクがあるため、下記規則で除外する。選定タスクが公開微調整に含まれる根拠は見つからなかった。公開データ・config・README・API情報の取得内容とSHA256は出力のprovenance/に保存する。

## 成功率を見る前に固定した選択規則

1. LIBERO-90の公式task_order_index=0の90タスクを列挙する。
2. 指示文を小文字化、underscoreを空白化、連続空白を1個にする。同じ正規化指示文が40タスクにあれば除外する。該当する明示名は KITCHEN_SCENE3_turn_on_the_stove、KITCHEN_SCENE9_turn_on_the_stove、STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy。
3. Longと同じ場面を除外：KITCHEN 3/4/6/8、LIVING_ROOM 1/2/5/6、STUDY 1。場面IDは正規表現で抽出し、SCENE1とSCENE10を区別する。
4. 残り53タスク・11場面を、場面名とtask名の辞書順で固定する。NumPy default_rng(seed=20261106)で場面名をpermutationし、順に各場面からrng.integersで1タスクずつ選ぶ。10場面に達したら終了する。選定後は公式task ID順で表示する。成功率による選び直しは禁止。

| 公式ID | 場面 | タスク名 | 元の指示文 | 初期状態数 |
|---|---|---|---|---:|
| 0 | KITCHEN_SCENE10 | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet | close the top drawer of the cabinet | 50 |
| 10 | KITCHEN_SCENE1 | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | put the black bowl on top of the cabinet | 50 |
| 13 | KITCHEN_SCENE2 | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate | put the black bowl at the front on the plate | 50 |
| 31 | KITCHEN_SCENE5 | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet | put the black bowl on top of the cabinet | 50 |
| 35 | KITCHEN_SCENE7 | KITCHEN_SCENE7_open_the_microwave | open the microwave | 50 |
| 43 | KITCHEN_SCENE9 | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet | put the white bowl on top of the cabinet | 50 |
| 57 | LIVING_ROOM_SCENE3 | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray | pick up the cream cheese and put it in the tray | 50 |
| 61 | LIVING_ROOM_SCENE4 | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray | pick up the chocolate pudding and put it in the tray | 50 |
| 84 | STUDY_SCENE3 | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy | pick up the red mug and place it to the right of the caddy | 50 |
| 88 | STUDY_SCENE4 | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf | pick up the book on the right and place it on the cabinet shelf | 50 |

## 固定プロトコル

通常pi05だけ、10タスク×公開初期状態0–9の100試行。seed_base=7、環境seed=7+初期状態、問い合わせseed=(7+初期状態)×100000+問い合わせ番号。積分10回、replan5、待機10、制御上限520、horizon1000、試行壁時計timeout600秒。通常の失敗は再試行せず、技術エラーのみ1回再試行する。最終エラーも失敗として分母に含め、未完了と完了を区別する。同じ試行を二重計上しない。

[openpi公開クライアント](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py) と既存SkillNet公開評価はLIBERO-90に400制御ステップ、Longに520を指定する。本評価は依頼どおりLongとの共通設定520を使う。公式90の400より長い評価であることを明記し、論文成功率の直接再現とは扱わない。

既存anchor_flow/run_case.pyの試行ループをそのまま再利用する。両カメラagentview_image / robot0_eye_in_hand_imageを180度回転、224四方にpad resize、stateは手先xyz＋quaternionからaxis-angle3＋gripper qpos2。LIBERO-90とLongで同じPanda・カメラ・行動7次元の入力規約。dense通常方策は画像・状態・元のpromptのみを読み、注釈は使わない。注釈読み込みはこのworker内だけ空リストに置き換える（通常方策では参照しない）。指示文は元のlanguageをそのまま使う。

先にLongの保存済み通常結果に照合する4組を実行する。task0/init0成功、task8/init1成功、task8/init0失敗、task6/init0失敗。既存の100組は95成功。成否と初期状態hash・seedを必須照合し、制御ステップ数も記録する。成否が一致しなければ本番を開始せず報告する。

固定snapshotからdetach実行。2〜4サーバー（初期設定4、各2環境worker）、GPU共有、PREALLOCATE=false。サーバー2threads・worker1thread・controller2threads・監視1thread、最大19threads、cores8–47、nice10以上。空きポートを自動選択する。初期状態0/1/2を機械的に各タスクの動画・初期/最終フレーム保存対象とする。受動記録で環境のstep・actionには介入しない。

## 記録・判断

出力は .skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/。試行ログは追記し、attempt・エラーを保全する。集計は全100試行を終えて成功数とWilson95%を報告する。decision.jsonにはsuccess、trials、選定tasks、policy_dataを保存する。成功率を見て追加タスクや新しい判定閾値を選ばない。最終フレームと動画を各タスク数本確認し、失敗像を短く記述する。
