# 公開通常pi05：未学習LIBEROタスクの基準

状態：完了。事前計画は [plan.md](plan.md)。

## 学習データの確認

[公開評価README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) は公開重みがpi05_libero configで学習されたことを示す。[公開config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) はrepo_id=physical-intelligence/liberoを指定し、[公開データカード](https://huggingface.co/datasets/physical-intelligence/libero) とtasks/infoはSpatial・Object・Goal・Longの40タスクを示す。保存した公開40指示文とローカル4 suiteの集合は完全一致。ローカルnorm statsもassets/physical-intelligence/libero配下の公開pi05_liberoのものを使用する。

選定10タスクは、その公開40指示文にもLongの場面にも重ならない。ただしpi05_baseなど基盤モデルの全事前学習履歴での非包含は未確定。「公開LIBERO微調整データ外」という範囲での評価であり、全事前学習に対する未学習を断定しない。norm statsのasset名だけを学習履歴の証明とは扱わない。

## Long既知試行の照合

既存通常方策95/100の同じharnessと照合。成否・初期状態hash・環境/試行seedの照合結果：全組一致。

| task | 初期状態 | 既存成功 | 今回成功 | 既存steps | 今回steps | 一致 |
|---|---|---|---|---|---|---|
| 0 | 0 | True | True | 277 | 277 | True |
| 8 | 1 | True | True | 368 | 368 | True |
| 8 | 0 | False | False | 520 | 520 | True |
| 6 | 0 | False | False | 520 | 520 | True |

## 固定条件

機械的な場面層化、seed20261106、候補53タスク・11場面から10場面各1タスク。選び直しなし。指示文は元のものを使用。公開初期状態は各50個あり、0–9を使う。

通常pi05のみ、積分10回、replan5、待機10、制御上限520、horizon1000。環境seed=7+初期状態、問い合わせseed=(7+初期状態)×100000+問い合わせ番号。timeout600秒、技術エラーだけ1回再試行。最終技術エラーも失敗として分母に含める。途中の未処理ケースは分母に含めない。

[公開評価](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py) のLIBERO-90上限は400、Longは520。本評価は依頼どおりLong共通の520であり、公式90の成功率の直接再現ではない。Pandaのカメラ2視点、180度回転、224 pad resize、手先xyz・axis-angle3・gripper qpos2という入力規約はLongと同じ。注釈やスキルはモデルに入力せず、画像・状態・元のpromptのみを使う。

## 成功率

| 対象 | 成功/処理済み | 成功率 | Wilson95% | 終端技術エラー |
|---|---|---|---|---|
| 選定10タスク | 32/100 | 32.0% | 23.7–41.7% | 0 |

参考：保存した最初の成功ステップで400上限に再採点すると、32/100。追加ロールアウトは行っていない。主結果は固定した520上限のまま。

## タスク別

| 公式ID | タスク名 | 成功/処理済み | Wilson95% | 技術エラー |
|---|---|---|---|---|
| 0 | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet | 0/10 | 0.0–27.8% | 0 |
| 10 | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 10/10 | 72.2–100.0% | 0 |
| 13 | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate | 0/10 | 0.0–27.8% | 0 |
| 31 | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet | 0/10 | 0.0–27.8% | 0 |
| 35 | KITCHEN_SCENE7_open_the_microwave | 1/10 | 1.8–40.4% | 0 |
| 43 | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet | 1/10 | 1.8–40.4% | 0 |
| 57 | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray | 10/10 | 72.2–100.0% | 0 |
| 61 | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray | 10/10 | 72.2–100.0% | 0 |
| 84 | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy | 0/10 | 0.0–27.8% | 0 |
| 88 | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf | 0/10 | 0.0–27.8% | 0 |

## 失敗を確認する記録

各タスクの初期状態0/1/2について、初期・最終フレームと2視点動画を受動的に保存する。画像記録のために追加のenv.stepやaction変更は行わない。

| task | 初期状態 | 成功 | 動画 | 最終フレーム |
|---|---|---|---|---|
| 0 | 0 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task00-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task00-init00/attempt-1/final.png |
| 0 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task00-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task00-init01/attempt-1/final.png |
| 0 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task00-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task00-init02/attempt-1/final.png |
| 10 | 0 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task10-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task10-init00/attempt-1/final.png |
| 10 | 1 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task10-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task10-init01/attempt-1/final.png |
| 10 | 2 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task10-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task10-init02/attempt-1/final.png |
| 13 | 0 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task13-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task13-init00/attempt-1/final.png |
| 13 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task13-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task13-init01/attempt-1/final.png |
| 13 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task13-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task13-init02/attempt-1/final.png |
| 31 | 0 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task31-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task31-init00/attempt-1/final.png |
| 31 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task31-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task31-init01/attempt-1/final.png |
| 31 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task31-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task31-init02/attempt-1/final.png |
| 35 | 0 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task35-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task35-init00/attempt-1/final.png |
| 35 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task35-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task35-init01/attempt-1/final.png |
| 35 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task35-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task35-init02/attempt-1/final.png |
| 43 | 0 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task43-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task43-init00/attempt-1/final.png |
| 43 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task43-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task43-init01/attempt-1/final.png |
| 43 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task43-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task43-init02/attempt-1/final.png |
| 57 | 0 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task57-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task57-init00/attempt-1/final.png |
| 57 | 1 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task57-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task57-init01/attempt-1/final.png |
| 57 | 2 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task57-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task57-init02/attempt-1/final.png |
| 61 | 0 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task61-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task61-init00/attempt-1/final.png |
| 61 | 1 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task61-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task61-init01/attempt-1/final.png |
| 61 | 2 | True | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task61-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task61-init02/attempt-1/final.png |
| 84 | 0 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task84-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task84-init00/attempt-1/final.png |
| 84 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task84-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task84-init01/attempt-1/final.png |
| 84 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task84-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task84-init02/attempt-1/final.png |
| 88 | 0 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task88-init00/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task88-init00/attempt-1/final.png |
| 88 | 1 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task88-init01/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task88-init01/attempt-1/final.png |
| 88 | 2 | False | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task88-init02/attempt-1/two_views.mp4 | /work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/production/attempts/task88-init02/attempt-1/final.png |

task 0: 引き出しが開いたまま、手先はボウル付近にある。閉める対象と異なる動作の可能性。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 10: 成功例。ボウルをキャビネット上付近まで運んだことと環境の成功判定を確認。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 13: 皿付近にボウルはあるが成功判定は未達。手前の対象ボウルの取り違え、または置き位置のずれが候補であり、静止画だけでは確定しない。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 31: 手先がキャビネット手前側へ下がり、ボウルを上面へ置いた達成状態は確認できない。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 35: 扉の位置は変わって見えるがopen述語の成功判定は未達。画像だけでは開角度や対象の違いを確定できない。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 43: 白いボウルは台上に残り、手先はフライパン付近にある。対象を取り違えている可能性。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 57: 成功例。クリームチーズの箱が木製トレイ内にある様子を確認。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 61: 成功例。チョコレートプディングが木製トレイ内にある様子を確認。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 84: 赤いマグは初期位置付近で、手先はマグ付近にある。caddy右側への配置は未達。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

task 88: 本は持ち上げたように見えるが、指定したキャビネット棚上への配置は未達。 所見は保存した初期・最終フレームの比較であり、最初の失敗局面の因果分類ではない。

![初期と最終フレームの比較・前半](/work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/visual-panel-1.png)

![初期と最終フレームの比較・後半](/work/.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/visual-panel-2.png)

## 実行記録

固定snapshot：`.skillnet-state/snapshots/unseen-libero-baseline-01/`。source diff・全source hash・plan hash・norm stats hashはcontrol/manifest.json。import先・Torch threads・CPU affinityはcontrol/imports/に記録。初期状態assetの所在・SHA256はplan.json。選定テスト4件はprovenance/tests.log。公開情報の原文とSHA256はprovenance/。

4サーバー・各2環境（Long照合は1サーバー・2環境）。server2threads、worker1thread、controller2threads、監視1thread、cores8–47、nice10以上、PREALLOCATE=false、JAX fraction .20/server。GPU/進捗はcontrol/health.jsonl、全attemptとエラーはsmoke/・production/に保全する。

出力：`.skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/`。コード：`skill_moe/skillnet/examples/memory_pilot/unseen_libero/`。

全100試行を処理。今回固定したタスク集合で32%成功。タスクを成功率で選び直していない。未学習の範囲と400対520の違いを踏まえ、既存Longの95%と難易度差を同一視しない。
