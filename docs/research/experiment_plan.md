# 実験計画：スキルの選び方・スキル Expert・RoboCasa

別の GPU で続きを回すための計画です。どの実験も、問い、入力、手順、出力、判定の基準を書いています。
数値の根拠は [09 experiments.md](09_unseen_libero/experiments.md) の KEY2・LA1 の節と、[09 evidence/](09_unseen_libero/evidence/) にあります。

## 今わかっていること

未学習 LIBERO-90 Eval（619 窓、難しい行 134）で、腕6軸の MSE を比べました。グリッパーは正解の値で採点しています。選んだ単位は保存したまま再生しています。

| 選び方（1位の単位） | 学習済み 全体 | 未学習 全体／難 |
|---|---:|---:|
| 正解に一番近い単位（上限） | .018 | .031 ／ .067 |
| P5：VLM 平均＋直前10ステップの行動＋状態を連結して MLP | .041 | **.105 ／ .184** |
| P0：VLM 平均だけを MLP | .036 | .132 ／ .275 |
| 共有の潜在トークン E（VLM＋履歴＋状態をクロスアテンションで読む） | **.035** | .125 ／ .256 |
| 直前の行動を保つ | .056 | .114 ／ .201 |
| 通常 π0.5 | .002 | .115 ／ .253 |
| ランダム | .247 | .248 ／ .355 |

- E は、学習済みでは最良ですが、未学習では P5 に負けます。E − P5 の差は +.020 で、95% 区間は［.011, .029］です。学習した 40 タスクの場面に合わせ込んでいると考えています。
- 上限の .031 と大きな差が残っています。メモリには良い単位があるのに、選べていません。
- 上位20個を平均すると、MSE は下がります（P5 で .081）。ただし大半は、平均を取って値が縮む効果です。主張は1位の単位で立てます。
- この表の設定は、Dev で上位20個の平均の誤差によって選んでいます。1位の誤差で選び直すのが KEYGRID（実験 3）です。

## 優先順位

| 順 | 実験 | 問い | 状態 |
|---|---|---|---|
| 1 | **EX1 スキル Expert** | 検索した単位を Expert で直すと、学習済み・未学習の閉ループで π0.5 を超えるか | 下書きまで完了。Expert の学習は GPU 待ち |
| 2 | **JP1 JEPA の学習機構** | E を「次の潜在を予測する」目的で学習すると、未学習で汎化するか | Dev 探索の途中 |
| 3 | **KEYGRID キーの比較表** | 同じ入力で、連結 MLP と潜在トークンのどちらが良いか | 実装済み、本番はこれから |
| 4 | **GH1 グリッパーのヘッド** | 潜在トークンから開閉を出して、π0.5 のグリッパーを超えるか | 実装済み、本番は途中 |
| 5 | **LT1 別の潜在トークン** | π0.5 の行動生成部の内部表現や点の追跡を読むと汎化するか | 抽出は smoke のみ。全件で回す経路が未実装 |
| 6 | **RoboCasa** | スキルメモリの構築とスキル Expert の学習を、LIBERO で決めた形で作り直す | 計画のみ |

## 共通の決まり

- **メモリ：** LIBERO-40 のメモリ側デモ 400 本から作った可変長単位 4,516 個です。グリッパーの開閉と手先の速さの極小で切り、最短 5 ステップにしています。腕6軸を DCT 3係数で保存し、クラスタリングはしません。
- **評価する行：**
  - 学習済み：LIBERO-40 のメモリ外デモの 621 窓（難しい行 125）
  - 未学習：LIBERO-90 Eval の 619 窓（難しい行 134）
  - 行の一覧は `compress-seen-cuts-01/production-control/evaluation-selection.json` と `unseen-rows-02/` にあります。
- **採点：** 腕6軸の二乗誤差の和を (10×7) で割った値です。グリッパーは別に評価します。
- **設定の選び方：** Dev（LIBERO-90 のタスク 6 と 7）だけで、**1位の腕 MSE** を基準に選びます。Eval の採点は最後に1回だけです。
- **入力：** クエリには、その時点までの観測だけを入れます。タスク ID、未来、正解の続きは入れません。正解の行動は、単位の選択（上限の行）と採点にだけ使います。
- **学習データ：** 学習には、メモリ側の LIBERO-40 と、Dev・Eval を除いた LIBERO-90（L90X）だけを使います。未学習 Eval の観測は、行動ラベルなしの形でも使いません。
- **閉ループ：**
  - 未学習：LIBERO-90 Eval の 10 タスク × 初期状態 0〜9。seed_base=7、5 ステップごとに選び直し、上限 520 ステップ。
  - 学習済み：LIBERO-10 の 10 タスク × 初期状態 0〜4。
  - 同じ seed の通常 π0.5 と、McNemar の正確検定で比べます。

## 別の GPU での準備

1. clone して、[reproduction.md](../reproduction.md) の 2 節に従って `env/setup.sh` を実行します。sudo は要りません。
2. `SKILLNET_STATE_DIR` を決めて、元の保存先から次をコピーします。パスは `$SKILLNET_STATE_DIR` 以下です。コードには `/work/.skillnet-state` を直接書いた箇所があります。移し先でも同じパスにするか、シンボリックリンクを張ってください。

| 目的 | コピーするもの | サイズの目安 |
|---|---|---:|
| どの実験でも | `checkpoints/openpi-assets/checkpoints/pi05_libero/` と、段階 0 の `outputs/memory-pilot/pi05-memory40-20261104/run-01/`（VLM トークン） | 12 GB ＋ run-01 |
| 行と単位 | `unseen-libero/{unseen-rows-02,compress-seen-cuts-01}/` | 2.4 GB |
| 画像・履歴・状態の行 | `unseen-libero/latent-probe-01/data/` | 24 GB |
| KEY2-F の学習済みの重み（EX1・GH1・KEYGRID が読む） | `unseen-libero/key-shared-01/attempt-27-production/{final,tuning,selected_configs.json}` と `key-shared-01/vjepa/` | 3.2 GB |
| EX1 の準備済みデータ | `unseen-libero/expert-refine-exp3b-01/`（L90X の行と単位選択）と `unseen-libero/expert-ex1-01/production-02/prepared/` | 3.3 GB |
| 閉ループ | `unseen-libero/baseline-01/`（通常 π0.5 の対応試行） | [reproduction.md](../reproduction.md) 7 節 |

- 足りない元データが1つあります。L90X の task 51（`LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket`）の HDF5 です。EX1 の B は、このタスクを除いた 77 タスクで学習します。
- コードは `skill_moe/skillnet/examples/memory_pilot/unseen_libero/` にあります。どのスクリプトも `--help` で引数を確認できます。

## 実験 1：EX1 スキル Expert（最優先）

**問い：** 検索した単位を下書きにして Expert で弱く直すと、学習済み・未学習の閉ループで通常 π0.5 を超えるか。超えるなら、どの選び方の下書きが一番良いか。

**手順**

1. **下書きを作ります（完了済み）。** KEY2-F の固定チェックポイントで、各行の1位の単位を出します。
   - D0：P0
   - D1：P5
   - D3：E-VLM+H
   - 参考：D2（P5 の上位20個の平均）、D4（E-VJ+H の上位20個の平均）

   値は [evidence/ex1/draft_metrics.json](09_unseen_libero/evidence/ex1/draft_metrics.json) にあります。

2. **Expert を学習します。** `skill_expert_refine.py train` を使います。
   - pi05_libero の行動生成部を、(1−t0)·下書き＋t0·ノイズ から正解の行動へ戻すように学習します。VLM は固定です。
   - 学習時の下書きは、正解に一番近い保存単位を .6、2〜3位を .15、P5 の検索結果を .25 の割合で混ぜます。同じタスクの単位は除きます。見た目が似た区間から学習させないためです。
   - A は LIBERO-40 だけ、B は LIBERO-40 に L90X（77 タスク）を足します。
   - 5,000 更新で、重みと t0 は Dev で選びます。
   - メモリの目安：H200 の 32% 程度（約 46 GB）を、microbatch 1 で使います。

   ```bash
   U=skill_moe/skillnet/examples/memory_pilot/unseen_libero
   PREP=$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero/expert-ex1-01/production-02/prepared
   OUT=$SKILLNET_STATE_DIR/outputs/memory-pilot/unseen-libero/expert-ex1-01/<新しい出力先>
   python $U/skill_expert_refine.py train --prepared-dir $PREP --out $OUT/A --condition A --steps 5000 --batch 8 --microbatch 1 --mem-fraction .32
   python $U/skill_expert_refine.py train --prepared-dir $PREP --out $OUT/B --condition B --steps 5000 --batch 8 --microbatch 1 --mem-fraction .32
   ```

3. **オフラインで評価します。** {下書きのまま, Expert A, Expert B} × {D0, D1, D3} を、学習済み 621 / 未学習 619 の全体と難しい行で比べます。参考の行として、通常 π0.5、直前保持、上限の下書き＋Expert も並べます。グリッパーは π0.5 の値を使います。

4. **閉ループで評価します。** 未学習 LIBERO-90（100 試行）と学習済み LIBERO-10（50 試行）で、次の4条件を比べます。
   - 通常 π0.5
   - D0＋Expert
   - D1＋Expert
   - D3＋Expert

   余裕があれば「D1 下書きのまま」も足します。

**判定**

- 未学習の閉ループで、π0.5（32/100）と、これまでの最良（単位メモリの 43/100）を超えるか。
- 学習済みの LIBERO-10 で、π0.5（46/50）からどれだけ落ちるか。

**注意**

- 3〜4 の評価と閉ループのコード（`expert_ex1.py`、`run_ex1_production.py` など）は、この PR を作った時点で修正の途中でした。そのため含めていません。仕上がったら追加します。
- 以前、閉ループ用キャッシュで、キーと再生する単位の並びがずれる不具合がありました。修正済みで、オフラインの下書きとの最大差 0 を確かめています。移し先で作り直す場合も、この一致を最初に確認してください。

## 実験 2：JP1 JEPA の学習機構で E を学ぶ

V-JEPA の特徴を入力に使うのではなく、JEPA の学び方そのものを E に使います。

```
z_t   = E(時刻 t の観測)                    # KEY2-F と同じ E
z̄_t'  = E_ema(次の単位が始まる時点の観測)    # 目標。EMA で追従し、勾配は流さない
ẑ_t'  = P(z_t [, 実行した単位])              # 予測器
L_pred = ‖ẑ_t' − z̄_t'‖² ＋ 分散・共分散の正則化（潰れ防止）
```

**比べる行**

| 行 | 学び方 |
|---|---|
| a | 検索の損失だけ（KEY2-F と同じ。比較の基準） |
| b | L_pred で事前学習してから、検索の損失で微調整 |
| c | L_pred と検索の損失を同時に学習 |
| d | L_pred だけ |
| e | b の事前学習に L90X の観測を足す |

- 入力は E-VLM と E-VLM+H の2通りです。
- コード：`key_latent_predict.py`。本番のランナーはスナップショット `snapshots/jp1-20261115T103600Z/jp1_production.py` にあります。
- 出力：`unseen-libero/latent-pred-01/production/`

**判定**

- 未学習の1位の誤差で、b または e が a（E-VLM .132）と P5（.105）を下回るか。
- 診断として、タスク ID を当てる線形プローブの正解率が下がるか（場面を覚え込まなくなったか）。

**状態：** Dev 探索の途中です。GPU を EX1 に譲るため、dev 43〜49 の6設定を止めました。再実行が必要です。

## 実験 3：KEYGRID キーの比較表

行を入力、列を結合の仕方にした表を、同じ評価で埋めます。

| 入力 | 連結＋MLP | 連結＋MLP（キーとクエリで網を共有） | 潜在トークン E |
|---|---|---|---|
| VLM | P0 | 新規 | E-VLM |
| 履歴＋状態 | 新規 | 新規 | E-H |
| VLM＋履歴＋状態 | P5 | 新規 | E-VLM+H |
| V-JEPA＋履歴＋状態 | 新規 | 新規 | E-VJ+H |
| V-JEPA＋VLM＋履歴＋状態 | 新規 | 新規 | E-VJ+VLM+H |

- **選び方の統一：** 既存の行も、KEY2-F の `tuning/` にある Dev の結果から、1位の誤差で選び直します。学習はし直しません。
- **埋める指標：** 未学習と学習済みの MSE、Spearman、recall@1、回収率です。同じ入力どうしの「E − 連結 MLP」の差を、episode bootstrap の 95% 区間で出します。
- **指標の意味：**
  - Spearman：全 4,516 候補について、予測の順位と正解の誤差の順位の相関
  - recall@1：1位の単位が、誤差最小の単位と一致した割合
  - 回収率：ランダムから上限までの差を何割埋めたか
- **V-JEPA の注意：** 今は base と wrist を1トークンずつに平均しています。公平に比べるには、平均しないパッチトークンの行を足します。
- コード：`key_grid.py`。出力：`unseen-libero/key-grid-01/`

## 実験 4：GH1 潜在トークンからグリッパー

- **方式：** 固定した E の潜在トークン5本と、今のグリッパーの状態から、未来10ステップの開閉を出すヘッドを学習します。π0.5 の出力は入力に使いません。
- **比べる行：**
  - E-VLM+H または E-VJ+H を固定してヘッドだけを学習
  - 検索の損失と同時に学習
  - VLM 平均＋状態の MLP（対照）
  - L90X を足して学習
- **超えるべき値：** 未学習 Eval のグリッパー誤差（÷7）で、π0.5 の .026。ほかの規則は、分類器 .049、直前保持 .075、保存値 .194 でした（GRIP1）。
- コード：`gripper_from_latent.py --production`。出力：`unseen-libero/gripper-latent-01/`
- **次の段：** π0.5 を明確に下回る行が出たら、EX1 の閉ループで、グリッパーをこのヘッドに替えた条件を足します。

## 実験 5：LT1 別の潜在トークン

- **AX：** pi05_libero の行動生成部の、各層の内部表現です。今は V-cache で取っています。仕様にある残差の隠れ状態を取るなら、フックを足します。
- **PT：** CoTracker3 で、評価時点までの過去16フレームに格子の点を追跡し、その動きをトークンにします。未来のフレームは使いません。
- どちらも、同じ E のクロスアテンションで読みます。
- `key_latent_sources.py` に抽出と、KEY2 に合わせる部分があります。**全行のキャッシュを作り、学習・Dev 選択・Eval まで通す CLI は未実装です。** ここから作ります。

## 実験 6：RoboCasa のスキルメモリとスキル Expert

**背景：** これまでの結果は [10_robocasa365](10_robocasa365/README.md) にあります。6 タスク×50 試行で、メモリ＋Expert は 118/300 で、通常 π0.5 の 173/300 を下回りました。そのときのメモリは、固定窓と古い大きさヘッドで作っていました。LIBERO の予備実験で決まった形に作り直します。

1. **R1 メモリの再構築：** 6 タスク（PickPlaceCounterToCabinet、PickPlaceSinkToCounter、OpenDrawer、CloseFridge、TurnOnSinkFaucet、TurnOffStove）の pretrain human デモのメモリ側 80% から作ります。
   - 可変長の単位にします（グリッパーの開閉＋手先の速さの極小、最短 5）。
   - 腕は DCT 3係数で保存し、グリッパーは別に扱います。クラスタリングはしません。
   - LIBERO と同じ上限評価（正解に一番近い単位）で、固定窓の古いメモリと比べます。
2. **R2 キー：** 保留デモ 20% の窓で、P0・P5・E-VLM+H を LIBERO と同じ学習・評価で比べます。RoboCasa では、履歴だけのキーは閉ループで全滅しました。画像を含むキーを主にします。
3. **R3 スキル Expert：** EX1 と同じ学び方にします。学習時の下書きは、正解に一番近い単位を主にし、キーの検索結果を混ぜます。expert-03（正解の下書きを崩さない版）の設定を引き継ぎます。
4. **R4 閉ループ：** まず 6 タスク×50 試行を、ベースラインと同じ seed で回します（超えるべき値 173/300）。決めた構成のまま、18 タスク×50 試行に広げます。
5. **注意：** フォーク設定は `pi05_pretrain_human300` です。`discrete_state_input=True`、`SKILLNET_SUBTASK_GENERATE=0`、状態と行動の並べ替えを守ります（[robocasa365.md](../robocasa365.md)）。

## 結果の共有

- **数値の表：** [09 experiments.md](09_unseen_libero/experiments.md) の該当する節に足します。
- **元の集計：** 数 MB 以内のものを、`docs/research/<番号>_<内容>/evidence/` に置きます。今回の KEY2-F・LA1・EX1 の下書きは [09 evidence/](09_unseen_libero/evidence/) の `key2f/`、`la1/`、`ex1/` にあります。
- **重みや特徴量：** Git には入れません。保存先のどこにあるかを、experiments.md の各節に書きます。
- **発表の図と表：** 発表デッキ（claude.ai の Artifact、非公開）にあります。共有するときは、デッキの Share メニューから相手を追加してください。
