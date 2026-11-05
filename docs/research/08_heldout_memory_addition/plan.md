# 重み固定での保留タスク実演追加：準備計画

記録日：2026-11-04。状態：準備のみ。モデル学習、特徴抽出、評価rolloutは実行していない。

## 目的と主張の範囲

検証したい主張は「LIBERO-90で学習したモデルに対して、重みを更新せずLong実演をメモリに追加すると、今回の追加学習に含めていないLong評価タスクの成績が変わるか」。LIBERO-90 checkpointがLongを含まないことは、そのcheckpointの名前付き学習データに関する範囲だけで述べる。公開基盤モデルの事前学習全体でLongタスクが未見だったとは主張しない。

条件はメモリの有無・Long例の追加・無関係なGoal例の追加で分ける。実演追加後もモデル、行動圧縮器、写像、検索キー、修正Expert、ノイズ選択規則を固定する。変更が許されるのはメモリ項目の追加だけ。しきい値や検索設定はLIBERO-90内で決め、Long/Goal追加データでは調整しない。

前提となる調査計画は[メモリ効果の分離計画](../07_pi05_memory40/summary.md)と、`plan_memory_gap.md`。段階2・3で救える候補があると確認できるまで、本実験の本実行に進まない。

## 1. SkillNet-90への互換性

### いま確認できた構造

| 項目 | 現行 `pi05_memory/core.py` | SkillNet-90 checkpoint |
|---|---|---|
| モデル実装 | `openpi.models.pi0.Pi0` の `MemoryPi05` 派生 | `openpi.models.pi0_moe_skill.Pi0` |
| 設定 | 通常 `pi0_config.Pi0Config`。通常のdense action expert | `config_moe_skill.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_moe_4")` |
| Skill設定 | 入力・レイヤーともSkillNet MoE用でない | configに `skill_num`, `skill_embed_dim` の明示指定はなく、クラス既定値は6、64。MoE action expertは4-expert設定。 |
| 観測 | 画像、状態、tokenized promptを生成 | `Observation` に加えて `skills` と `skill_mask` を用い、SkillNetのprefix・action側のforwardに渡す |
| checkpoint復元 | `run.py::load_policy` はパラメータ木とshapeを厳密比較 | 4-expert MoEのparameter tree。dense Pi0の木とは一致しない |

対象params: `.skillnet-state/checkpoints/pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999/params`。training configの該当runはLIBERO90 repo、batch size 32、20,000 train steps、MoE checkpoint loaderを指定する。

したがって、現状のままでは載らない。`load_policy` の厳密チェックで復元を拒否するのが期待される安全な挙動であり、missing leavesをランダム初期化で埋めて進めてはいけない。checkpointのparamsディレクトリは約15 GB。

### 必要な変更点（未実装）

1. `MemoryPi05` の基底を `openpi.models.pi0_moe_skill.Pi0` にし、SkillNet学習時と同じ `pi0_config_moe_skill.Pi0Config` を使う。対象checkpointの正確なconfig値とparams木を一致させる。出力行動長は10、action次元は32のうちactive 7を使う。
2. `prefix_features`、prefix cache、bridgeの各velocity呼び出しに、SkillNet本来の `skill_embeds`・`skill_mask` を通す。対象configは位置埋め込み無効だが、条件付き呼び出しの扱いをコードとcheckpointで確認する。検索キーをSkillNet条件付きprefixから作るのか、視覚・言語だけのprefixから作るのかは評価前に固定する。
3. `obs_batch` と前処理を拡張し、実行時に利用できるSkillNet入力を `Observation` に含める。学習・評価用の `skills` を得る経路が実行時にも使えるのかを先に決める。正解デモのskill/plan注釈を評価時の選択器へ渡さない。必要な条件を実行時に生成できなければ、このcheckpointを使った方策条件は未定義のままである。
4. `sample_bridge` のprefix forwardと各denoise stepをSkillNetの `sample_actions` と同じexpert/routing条件にする。現在の実装はskill embeddings/maskを渡していないため、たとえparameter treeを復元できても同じ方策の補正にはならない。
5. trainable filterをMoE構造で再点検する。通常Pi0での既存filterは約430.1M parameterを選んだ。MoE側で意図したaction expertとそのrouterのみが選ばれるか、PaliGemma視覚・言語幹と不要expertが固定かをshape-only確認後、破棄する1 batchのgradient smokeで確認する。本番学習前に、固定重み・optimizer state・activationのVRAMを計測する。
6. `run.py::prepare` はタスク数40を固定条件として検査し、既定splitも10 memory episode/taskである。90タスクを受け入れられるよう条件化し、タスクごとのepisode分割をchunk生成より先に決める。HDF5→LeRobot変換で保存するtask/episode IDと、memory/query分割のepisode IDが一貫することを検査する。
7. LIBERO-90 checkpointに付随するnormalization assetと、データ側action/stateのスケールを使う。LIBERO-40 checkpointのnorm statsや通常Pi0 configを暗黙に再利用しない。

最小の互換確認は、コードを隔離したコピーで、(a) MoE model configからabstract treeを作る、(b) checkpointの全leaf名・shapeを照合する、(c) 実際の `Observation` を用いてprefix・bridge velocity shapeを通す、の順で行う。ここでは実施していない。

## 2. データ量と実行資源の見積もり

### 実データ数

読み取り専用でHDF5のgroup/shapeを集計した。

| データ | ファイル/タスク | episode | frame/action step | stride10の完全10-step窓 |
|---|---:|---:|---:|---:|
| `data/LIBERO/libero_90` | 90タスクファイル | 4,500（50/task） | 669,043 | 64,865 |
| `data/LIBERO/libero_10`（Long候補） | 10タスクファイル | 500（50/task） | 138,090 | 13,570 |

Longは1 episode内を分割前に割り当て、そこから10-step固定strideで窓を作る。隣接窓を別splitに分けない。

### run-01の実測とLIBERO-90外挿

run-01はLIBERO-40の1,693 episode、273,465 framesから26,581窓を処理した。extractの `features.json` は375.3秒。retrieveは26,581 query × 6,583 memory rowsで1,755.7秒（約66 ms/query）。trainはbridge 4,000更新で741.8秒。prepareログは17:39:09のepisode 0から17:46:25のepisode 1,650まで7分16秒で、最終fileは17:47:10。codec.logは空のため、codec.ptのmtime区間（features出力後から約45秒）をrun-01の粗い目安にした。

90版はrun.pyと同じ10-memory episodes/taskを仮定し、memory窓数をおよそ全窓の20%（約13,000）とする。retrieve時間はquery数×memory候補数に比例する一次近似。MoE化・同時GPU負荷・IO差を含まない。

| 工程 | run-01実測 | 90版の見積もり | 根拠・注意 |
|---|---:|---:|---|
| HDF5→LeRobot変換 | 未計測 | 見積もり保留。先に5タスクで計測 | 提供converterはHDF5 readerではなく `tfds.load("libero_90_openvla_processed", data_dir=...)` を呼ぶ。ローカル `data/LIBERO/libero_90` は90個のHDF5が直下にあり、TFDSのディレクトリ構成ではない。入力adapter/既存TFDS rootを確定しないと変換時間を外挿できない。 |
| prepare・画像/行動窓作成 | 約8分 / 273k frames | 約18–22分 / 669k frames | frame数比2.45倍で外挿。 |
| frozen特徴抽出 | 375秒 / 26,581窓 | 約15–18分 / 64,865窓 | 375.3×64,865/26,581。SkillNet条件付き入力が増える場合は追加余裕。 |
| 行動codec | 約45秒（artifact mtime差。codec.logは空） | 約1.5–3分 | 28 epoch。窓数比例の粗い外挿で、直接測定ではない。 |
| exact token retrieval | 1,755.7秒 / 26,581×6,583 | 約2時間20分 | 66ms/queryを候補数約2倍・query数2.44倍へ比例外挿。最も長い測定工程。 |
| 修正Expert学習 | 741.8秒 / 4,000更新 | 約15–30分 | dense run-01の4,000更新に対し、MoE側action expert・routingで増える可能性を含む幅。計測前予測。 |
| **LeRobot化済み入力からの合計** | — | **約3時間15分〜3時間45分** | 起動・checkpoint書込み等を含む概算。TFDS/HDF5入力変換は含まない。 |

検索memory bankはfp16 token key `[N,712,2048]`。LIBERO-90で約13,000窓なら単一bankだけで約38 GB（約35 GiB）をGPUに置く。上記条件を50例/taskへ拡張するとbankは約189 GBとなり、単一GPUに載せられない。現在方式の10例/task上限を保つか、CPU/chunked候補bank・pooled key等を別途決める必要がある。

Longの平均は27.14窓/demo。追加支援例のtoken keyと90-base bankを同時にGPUへ置く場合の概算は次の通り。配列本体だけの下限で、allocator/workspaceは含まない。

| Long追加 | Long窓概算 | Long key単体 | 90 base約13k窓と合算 |
|---:|---:|---:|---:|
| 5 episode/task（50件） | 1,357 | 約4.0 GB | 約42 GB |
| 20 episode/task（200件） | 5,428 | 約15.8 GB | 約54 GB |
| 40 episode/task（400件） | 10,856 | 約31.7 GB | 約70 GB |
| 50 episode/task（500件、holdoutなし） | 13,570 | 約39.6 GB | 約78 GB |

VRAMログには注意が必要。run-01 health記録の最大値は72,736 MiBだが、共有GPU全体の利用量で、各工程・各プロセスのpeakではない。SkillNet-90 checkpointのdiskサイズは約15 GB。bridge trainのMoE action expert/optimizerを含むピークVRAMは既存ログから分離できないため、**実行計画では学習用に60–80 GB、token retrieval用に少なくとも45–50 GBの空きを仮確保し、1 batch smokeで測ってから決める**。これは計測結果ではなく保守的な割当案である。

ディスクはrun-01 outputが約92 G（`du -sh`）/26,581窓。単純比例では90窓64,865件で約224 G、うちtokens.npy相当が約189 GB。LIBERO-40 LeRobot cache実測33 Gをframe比で外挿したLIBERO-90 cacheは約81 G。Long全500 demoの特徴を一度抽出・保持する場合は別にtoken key約40 GBと、画像/行/派生配列が加わる。既存Long HDF5は約13 G、90 HDF5は約63 G（すでに入力側に存在）。90 LeRobot cache＋90 run outputだけで約305 G、Long特徴も保存するなら少なくとも約350 Gを追加作業領域として見込む。checkpointはすでに配置されている約15 Gで、追加複製を除く。画像圧縮率・変換実装が変われば上下する。

## 3. Long追加用・保留用のepisode分割とGoal対照

### IDとsplit

1. 元のデモepisode ID（task名＋HDF5 demo番号またはconverterの`source_episode_index`）を固定し、窓を作る前に割り当てる。manifestに元ID、split、seed、taskを保存する。
2. Longの10 taskそれぞれ50 episodeから、まず5 episode/task（計50）をheld-out demoとして保留。これらはパラメータ、キー変換、しきい値、ノイズ選択に使わない。使う場合も選択後の記述的な圧縮・復元診断のみ。
3. 残る45 episode/taskの中で、taskごとにseed固定のnested supportを作る：5、20、40 episode/task。すべて同じ評価対象・同じbase memoryに対する累積集合とする。support追加時には重みを一切更新しない。
4. 50 episode/taskは利用可能全デモを使い、held-out demoを残せない。従って「50/task」は主比較のholdout条件から外し、全量上限（保留なし）として別表示するか、追加Longデモを調達してsplitを保つ。50条件を5/20/40のholdout付き比較と同列に扱わない。
5. 公開LIBERO初期状態によるシミュレーション評価は、デモepisodeと独立な状態・seedを固定し、全条件を同じ `(task, initial, seed)` で対にする。保留デモを閉ループevalに変換したり、eval成績で設定を選んだりしない。

### 90側の固定学習・memory split

HDF5で確認したLIBERO-90は各task 50 episode。run.pyの現在の方式を引き継ぐ場合、taskごとに5 episodeを90側dev/校正、別に10 episodeをbase memory、残る35 episodeを主train poolとし、すべてepisode単位で先に固定する。既存run.pyではmemory episodeもtrain poolの対象から除外されない点を明記し、再現時はこの挙動を維持するか変更するかを事前決定する。90側devだけで、codec、bridge、特徴/キー、noise・confidence規則を選ぶ。

base memoryはrun.pyの10例/task、計900 episode（窓約13k）を基準にする。90全4,500デモをtoken bankに置く設定ではない。全デモ保持が必要ならchunked searchへ設計を変え、VRAMと時間の再見積もりを行う。

### 同数の無関係なGoal memory

現在の `libero_40_unsegmented` にあるGoal task（正しいtask index 10–19）のデモは428 episode、task別33–50件。Long 5/task・20/task追加には、Goalも10 taskから同じ件数ずつ選び、追加総数をそれぞれ50・200件に一致させられる。Long 40/task（400件）はGoal pool総数428以内なので、全件数400に一致させてtask数をできるだけ均等に配分する。ただしtaskごと40件の完全均衡は33件・36件しかない2タスクで不可能で、5件/taskずつGoal holdoutを別に残すと利用可能量378件となり400件に届かない。よって400件対照は (a) count-matchを優先しholdout均衡を主張しない、または (b) Long/Goalとも25件/taskの250件に上限を揃える、のどちらかを実行前に固定する。

GoalはLIBERO-40 LeRobot cache由来で、90 checkpointの学習データとは別のtask-suite対照として扱う。Lang/task名、skill注釈は追加内容の選択や検索学習には入力しない。無関係条件はLongと同じmemory項目数・圧縮・検索・重みを使い、差は追加行動データの出所だけにする。

## 4. 比較・評価条件

### 主比較

| 条件 | base memory | 追加データ | 学習更新 |
|---|---|---|---|
| no-memory / zero-draft | なし | なし | 90側で学習済み修正Expertは固定 |
| 90-only | LIBERO-90の固定subset | なし | なし |
| 90 + Long | 同じLIBERO-90 | Long support 5/20/40 episode/task（50は全量上限） | なし |
| 90 + Goal control | 同じLIBERO-90 | count-matched Goal support | なし |

すべて同一の90専用codec・key・bridge checkpoint・threshold/noise設定を使う。Memory追加は既存codecでencodeして保存し、実際に推論時に算出できるキーで引く。Long行動をcodec/bridgeの再学習、confidenceの校正、noise選択に使わない。

### 指標・条件固定

- 主：Long上のシミュレーション閉ループ成功率と対応差。成功数だけでなく、救済・悪化、95%区間、paired testを出す。
- 副：LIBERO-90の一部タスク保持、Long追加デモの圧縮/復元MSE、memory検索時間、denoise/policy時間、CPU/GPU常駐量、diskサイズ。
- Longの保留demoは学習・選択から分離する。評価は公開初期状態を使い、デモそのものを閉ループtestと呼ばない。
- 最初に、SkillNet-90原文条件の再現ベースラインを、同じ対象タスク・公開初期状態・seed・制御上限で実行する。そこで全追加条件と対にする。既存L0の成績を直接基準差としない。
- 対象はLIBERO-90の名前付き学習範囲に含まれていないLongタスクに限定する。一般の「未見タスク」や事前学習全体で未見という表現は使わない。

## 5. SkillNet-90 L0と今回の閉ループ条件の差

| 条件 | SkillNet-90 L0（既存記録） | run-01閉ループ（直近の既存結果） |
|---|---|---|
| checkpoint | `pi05_libero_moe_skill_4_90`（MoE SkillNet） | 公開 `pi05_libero`（LIBERO-40適応checkpoint、Longを含む） |
| Long評価 | L0原文、28/60（46.7%）、9タスクの評価記録 | `libero_10`、通常459/500（91.8%）、10タスク×50 initial |
| 制御予算 | 最大800 control steps | rollout設定 `max_env_steps=520`、environment horizon=1000 |
| 再計画・積分 | replan=5、n=10 | replan=5、nfe=10 |
| 対応関係 | 今回の100組とは初期状態・分母を合わせた比較ではない | 500組baselineと、追加条件は共通初期状態0–9の100組で対応比較 |

L0の28/60とrun-01の459/500は、checkpoint、評価task数、分母、制御上限が違うため成功率差を手法効果として解釈しない。今回の本実験では、SkillNet-90 L0を同一seed・同一初期状態で取り直して追加条件の基準にする。800-step上限を採る場合も、全条件で揃え、別の520-step結果と混ぜない。

## 6. 実行前の必須確認と停止条件

1. HDF5データから想定task/demo/action shapeが取れる小さなCPU変換確認。提供converterのTFDS入力契約とHDF5実ファイルの不一致を解消する。
2. converter出力を90 tasks、4,500 episodes、669,043 framesと照合。`actions[7]`, `state[8]`, 2 camera image、source episode IDを検証する。
3. SkillNet-90 model configでabstract treeとcheckpoint leaf/shapeを完全照合。SkillNet入力のsourceと実行時利用可能性を確定。
4. 同じ一試行・同じseedのSkillNet-90 L0再現が既存L0の主要条件（800 control steps、replan5、n10）に合うことを確認。その後、固定checkpointをsnapshotし、各条件を対応実行する。
5. 検索bank/bridge trainの個別GPU peakが記録されない、あるいは予定空きVRAMを超える場合は、full runに進まずchunked retrieval・dataset縮小ではなく、まず測定と設計更新を行う。

以上は実行準備であり、この文書作成時点で学習・評価は未実施。
