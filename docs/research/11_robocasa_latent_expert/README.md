# RoboCasa：潜在トークンで取り出した単位からスキル Expert を学ぶ

**問い：** LIBERO で決めた運動の形（可変長の単位、腕6軸は DCT 3係数、グリッパーは別）で RoboCasa のスキルメモリを作り直し、潜在トークンのキー E で単位を取り出して、それを下書きにスキル Expert を学習すると、RoboCasa の保留デモでの行動誤差（MSE）は π0.5 や下書きのままと比べてどうなるか。

**位置づけ：** [実験計画](../experiment_plan.md)（`exp/key2-expert-plan` ブランチ）の実験6 を、オフラインの MSE までに絞って進める。前の RoboCasa の結果は [10](../10_robocasa365/README.md)。キーの設計は 09 の KEY2-F、Expert の学び方は 09 の EX1 に合わせる。

**現在地：** R1〜R3 のオフライン評価（2026-11-15）に続き、閉ループを回した（2026-11-17）。メモリにある6タスクでは、どのキーも π0.5 と同じ程度だった。メモリに無い2タスクでは、VLM 平均を使う2つのキー（P0・P5）が π0.5 を約10回上回った（p = .06〜.08、60試行）。潜在トークン（E-VLM+H）は、オフラインの誤差は最良だが閉ループでは伸びない。π0.5 を切り替え時だけ使い、グリッパーと t0 を潜在トークンのヘッドから出す版は、6タスク 72/180・2タスク 11/60 と悪化した。詳しくは下の「閉ループ」。

| 閉ループ（成功数、π0.5 と同じ seed） | 6タスク（メモリあり、各30試行） | 2タスク（メモリなし、各30試行） |
|---|---:|---:|
| 通常 π0.5 | 95/180 | 24/60 |
| P0（VLM 平均）＋Expert | 95/180（p 1.0） | **33/60（p .078）** |
| P5（VLM 平均＋履歴＋状態）＋Expert | 92/180（p .77） | **34/60（p .064）** |
| E-VLM+H（潜在トークン）＋Expert | 93/180（p .87） | 26/60（p .83） |
| E-VLM+H＋ヘッド（π0.5 は切り替え時だけ） | 72/180（p .0014） | 11/60 |

## 設計

| 段 | 中身 | LIBERO から引き継ぐもの | RoboCasa で変えたもの |
|---|---|---|---|
| R1 メモリ | 6タスク×100デモ。80/20 の分割は `memory-01` と同じ seed。保留20を Dev 5 / Eval 15 に分ける。グリッパーの符号の切り替えと、手先の速さ（3フレーム平滑化）の極小で切る。最短5ステップ | CMP3 の切り方、DCT-3 を自然長で符号化・復号、クラスタリングなし | 旧メモリは最短10・型を大きさで正規化していた。今回は保存値をそのまま再生する |
| 評価行 | 保留デモのフレーム 0,10,20,… の連続する窓。採点は次の10ステップ | KEY2 の採点（腕6軸の二乗和 ÷ (10×7)）、難しい行は上限の誤差の上位20% | 正規化はチェックポイントの z-score（行動生成部が扱う空間） |
| R2 キー | 5個の潜在トークン E が、π0.5 最終層の prefix トークン（カメラ3台＋プロンプト）、直前10ステップの実行行動、現在の状態を読む。クエリとキーで同じ E を使い、トークンごとの内積の平均で類似を取る | 単位の開始をクエリにする学習、8クエリ＋ランダム256候補、自エピソード除外、履歴の破損、soft listwise の教師、logit scale の上限100。比較に P0・P5・E-VLM・E-H | 状態16次元、履歴は policy 順の7次元 |
| R3 Expert | RoboCasa で微調整した π0.5 の行動生成部を、(1−t0)·下書き＋t0·ノイズ から正解の50行動へ戻すように学習。VLM は固定。**キーによらない部品として1回だけ学習**する | Expert 後のグリッパーは通常 π0.5 | 下書きは EX1 の混合ではなく、正解に近い順位 r（0〜500 の対数一様）の単位をつないだもの。t0 は下書きの誤差の学習分布での順位を [.1, .9] に写したもの（±.05）。除くのは自エピソードの単位だけ。50ステップの下書きは単位を自然長で再生し、同じ選び方で次をつなぐ |
| 評価 | Dev/Eval の各行で、通常 π0.5、直前保持、下書きのまま、学習前 π0.5 でのブリッジ、学習した Expert（t0 ∈ {.1,…,.9}）。t0 の決め方は fixed（Dev で1つ）、conf（π0.5 との距離とキーの点差から誤差を予測、Dev で当てはめ）、上限として true-err と行ごとの最良 | t0 は Dev で選び、Eval は1回だけ採点 | 先頭5ステップの腕誤差も並べる（10 の §11 との対応のため） |

## 判定

- R1：正解に一番近い単位（上限）が、直前保持と通常 π0.5 をはっきり下回るか。下回らなければメモリに救える動きがない。
- R2：E-VLM+H の1位の腕 MSE が、P5・P0・ランダムを下回るか。上限との差をどれだけ埋めるか（回収率）。
- R3：Expert の出力が、下書きのまま・通常 π0.5 を Eval で下回るか（エピソード単位の bootstrap の95%区間）。

**注意：** 10 の §11 で、オフラインの先頭5ステップの誤差は方策の良し悪しを順位づけできなかった（直前を繰り返すだけで π0.5 を上回る）。この段の結論はオフラインの誤差についてだけ述べ、閉ループの改善とは言わない。

## 結果（latent-skill-01、2026-11-15）

指標はすべて腕6軸の10ステップ誤差（z-score、÷(10×7)）。Eval は 15デモ/タスク×6タスク、エピソード単位の bootstrap 95%区間。

### R1 メモリ

単位 9897 個（長さの中央値 10）。Eval で、上限（正解に一番近い単位）.174 [.147, .204]、直前保持 .475、ゼロ 1.49、ランダム単位 2.78。上限は直前保持の約37%で、メモリには救える動きがある。

### R2 キー（Dev で選び、Eval を1回）

| キー | 選んだ設定 | Dev 1位 | Eval 1位 [95%] | 難しい行 | 回収率 | recall@1 |
|---|---|---:|---:|---:|---:|---:|
| E-VLM+H | lr 3e-4, τ .02, 16 ep | .529 | **.568** [.477, .662] | 1.53 | .858 | .039 |
| E-H | lr 1e-4, τ .02, 16 ep | .697 | .649 [.554, .746] | 1.61 | .820 | .044 |
| P5 | lr 1e-4, τ .02, 16 ep | .683 | .695 [.590, .808] | 1.69 | .794 | .022 |
| E-VLM | lr 3e-4, τ .02, 16 ep | .776 | .813 [.680, .959] | 2.09 | .775 | .013 |
| P0 | lr 1e-4, τ .002, 16 ep | .911 | .877 [.728, 1.03] | 2.40 | .760 | .008 |
| 参考：上限 / 直前保持 / 通常 π0.5 | | .192 / .498 / – | .174 / .475 / .637 | | | |

- E-VLM+H が一番良く、P5・P0・E-VLM より低い。判定 R2 の前半は満たす。
- ただしどのキーも直前保持（.475）に届かない。上限との差はほとんど埋まっていない。
- VLM トークンだけ（E-VLM）は履歴だけ（E-H）より悪い。履歴を足すと VLM トークンが効く。
- 選ばれた設定はほぼすべて 16 エポックと τ .02 で、探索範囲の端にある。範囲を広げる余地がある。

### R3 キーによらない Expert（5000 ステップ、Eval、− π0.5 は対応のある差）

| 下書き | 下書きのまま | Expert fixed（Dev の t0） | Expert conf | 上限 true-err | 上限 行ごと最良 |
|---|---:|---:|---:|---:|---:|
| oracle | .176 | .173 (t0 .2) | .211 | .226 | .145 |
| E-VLM+H | .566 (−.072 [−.113, −.030]) | .568 (t0 .1; −.070 [−.111, −.028]) | .613 (−.024 [−.052, +.001]) | .582 | .399 |
| E-H | .656 (+.019) | .615 (t0 .8; −.023 [−.059, +.013]) | .599 (−.039 [−.064, −.014]) | .565 | .383 |
| P5 | .699 (+.061) | .647 (t0 .7; +.010) | .648 (+.011) | .619 | .451 |
| E-VLM | .823 (+.185) | .720 (t0 .8; +.083) | .683 (+.046) | .654 | .503 |
| P0 | .874 (+.237) | .698 (t0 .8; +.061) | .657 (+.020) | .627 | .479 |

通常 π0.5 は .637 [.535, .749]、直前保持は .475。

- **E-VLM+H の下書きはそのままで π0.5 を下回る**（−.072、区間が0をまたがない）。Expert は Dev で t0 .1 が選ばれ、ほぼ下書きのままになった。上積みはない。
- 悪い下書き（P0・E-VLM）では、Expert が大きく直す（P0：.874 → .657）。それでも π0.5 には届かない。学習前の π0.5 のブリッジより、学習後の Expert のほうが一貫して良い（例：E-H conf .619 → .599）。
- conf による t0 は、下書きが悪いキーでは fixed より良い。E-VLM+H と oracle では逆に悪くなる。誤差の予測が粗いため。
- 行ごと最良の t0（上限）は全キーで .38〜.50。t0 の選び方に大きな余地がある。それでも直前保持 .475 を確実に下回るのは E-H・E-VLM+H の上限だけ。
- 判定 R3：Expert の出力が下書きのままと π0.5 の両方を Eval ではっきり下回る条件はない。E-H conf は π0.5 を下回る（−.039）が、E-VLM+H の下書きのまま（.566）より悪い。

### 追加の切り分け：信頼度と生成ノイズ（2026-11-16、Dev）

**誤差→t0 の対応がずれていた。** 真の誤差が同じ水準でも、学習時の T0Map が出す t0 は、行ごとに最良の t0 より .1〜.3 高かった（ノイズの入れすぎ）。真の誤差を使う上限が fixed より悪い条件があったのはこのため。Dev で「π0.5 との距離から各 t0 の誤差を予測し、最小の t0 を選ぶ」形に当てはめ直すと、Eval で E-H .614→.575、E-VLM+H .568→.562 になった。Expert の t0=.1 と t0=.9 の出力の差も特徴に加えると、それぞれ .567 と .554 になった。π0.5 との距離と下書き誤差の順位相関は .72〜.80 で、信頼度の信号としては十分に効いている。

**オフラインの誤差の大部分は生成ノイズ。** π0.5 と Expert を、ノイズの seed を4通り変えて出し直した（Dev 741行）。

| | 1回の生成 | 5回の平均 |
|---|---:|---:|
| 通常 π0.5 | .633 | **.452** |
| Expert、E-VLM+H（最良の t0） | .532（t0 .4） | .477（t0 .7） |
| Expert、E-H（最良の t0） | .641（t0 .7） | .518（t0 .8） |
| Expert、oracle（最良の t0） | .185（t0 .2） | .170（t0 .3） |
| 参考：直前保持 | .498 | |

- π0.5 を5回平均すると .452 で、直前保持（.498）を下回る。10 §11 と今回の「π0.5 は直前保持より悪い」は、主に1回の生成のばらつきによる。
- 行ごと最良の t0（上限）は、同じ seed で選んで測ると .394（E-VLM+H）。別の seed で測ると .444 になる。上限の見かけの余地の約3分の1は、偶然の当たり。
- MSE は平均の予測を有利に扱う。そのため、オフラインの誤差で t0 や条件を比べるときは、複数回の生成を平均するか、seed をまたいで評価する必要がある。

### 最初の閉ループの試み（中止）

2026-11-16 に別マシンで3条件×24シミュレーターを同時に起動したところ、描画がすべて GPU 0 に集中して NVIDIA ドライバーが止まり、試行の結果は出なかった。下の「閉ループ」は、このマシンで作り直してシミュレーター数を絞って回したもの。

### まとめ

1. 取り出しが律速。上限 .174 に対して最良のキーは .568 で、直前保持 .475 にも届かない。
2. キーによらない Expert は、悪い下書きを π0.5 の近くまで戻せる。一方、良い下書きを上限側へ寄せることはできていない。
3. 10 の §11 と同じく、通常 π0.5 は直前保持より悪い（.637 対 .475）。オフラインの10ステップ誤差は閉ループの良し悪しを表さない可能性がある。

小さな根拠は [evidence/](evidence/)（キーの Eval 結果、選んだ設定、Expert の report と学習設定）。出力：`$OUT/key/eval/results.json`、`$OUT/expert/report.md`、`$OUT/expert/{dev,eval}/per_row.npz`。

## 閉ループ（2026-11-17、H200 1枚で作り直し）

別マシンの重みが使えなくなったため、同じ手順・同じ分割 seed でメモリ・トークン・キー・Expert をこのマシンで作り直した。単位 9,897、Dev 741、Eval 2,151 は一致し、キーの Eval 1位は P0 .883・P5 .689・E-VLM+H .561（上の表と同程度）。

### 閉ループでの動かし方

1. 5ステップごとに、その時点の観測（VLM トークン・直前10ステップの実行行動・状態）でキーを引き、1位の単位を自然長でつないで50ステップの下書きを作る。
2. 通常 π0.5 も1回生成し、下書きとの距離から t0 を選ぶ（Dev で当てはめた選択器、`evidence/selector_*.json` と同じ形）。
3. Expert が出発点 (1−t0)·下書き＋t0·ε から10ステップで積分する。
4. 腕6軸は Expert の出力、グリッパー・台車・モードは π0.5 の出力を使う。

seed は `7 + task_index×50 + trial`、replan 5、split pretrain、上限は各タスクの horizon。π0.5 は同じ seed の記録（6タスクは `baseline-01/v6-nosubtask`）と対にし、McNemar（両側 exact）で比べた。サーバー1台では問い合わせ（約 .35 秒）が律速だったため、途中からサーバー2台・各シミュレーター8台で回した（[evidence/run_closed_loop.sh](evidence/run_closed_loop.sh)）。

### 6タスク（メモリにあるタスク、trial 0–29）

| 条件 | Cab | Sink | Drawer | Fridge | Faucet | Stove | 合計 | 条件だけ成功 / π0.5 だけ成功 | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 通常 π0.5 | 15 | 28 | 19 | 11 | 20 | 2 | 95/180 | — | — |
| P0＋Expert | 15 | 27 | 22 | **21** | **9** | 1 | 95/180 | 27 / 27 | 1.0 |
| P5＋Expert | 10 | 24 | 24 | **23** | **10** | 1 | 92/180 | 22 / 25 | .77 |
| E-VLM+H＋Expert | 14 | 26 | 23 | 12 | 16 | 2 | 93/180 | 19 / 21 | .87 |

（Cab = PickPlaceCounterToCabinet、Sink = PickPlaceSinkToCounter、Drawer = OpenDrawer、Fridge = CloseFridge、Faucet = TurnOnSinkFaucet、Stove = TurnOffStove）

- 合計はどれも π0.5 と同じ程度。VLM 平均の2キーは CloseFridge で π0.5 の約2倍、TurnOnSinkFaucet で約半分になり、打ち消し合っている。
- E-VLM+H は trial 0–49 まで回すと 157/300 対 173/300（p .072）。
- 試行ごとにスキルと π0.5 の良い方を選べた場合の上限は P5 117/180、E-VLM+H 114/180。切り替えの判断に余地がある。

### メモリに無い2タスク（trial 0–29、task_index 6・7）

メモリ・キー・Expert・選択器は6タスクだけで作ったものをそのまま使い、何も当てはめ直していない。π0.5 はこの2タスクも学習済み。候補の TurnOnMicrowave と CloseToasterOvenDoor は π0.5 が 2/10 ずつで低すぎたので使っていない。

| 条件 | OpenCabinet | PickPlaceCounterToStove | 合計 | 条件だけ成功 / π0.5 だけ成功 | p |
|---|---:|---:|---:|---:|---:|
| 通常 π0.5 | 10 | 14 | 24/60 | — | — |
| P0＋Expert | 14 | 19 | **33/60** | 15 / 6 | .078 |
| P5＋Expert | 16 | 18 | **34/60** | 17 / 7 | .064 |
| E-VLM+H＋Expert | 10 | 16 | 26/60 | 12 / 10 | .83 |

- VLM 平均の2キーは両タスクで π0.5 を上回った。60試行なので確定ではない。
- E-VLM+H はオフラインの誤差が最良（.561）なのに、閉ループでは6タスク・2タスクとも伸びない。オフラインの10ステップ誤差と閉ループの順位が合っていない。

集計は [evidence/closed_loop_summary.json](evidence/closed_loop_summary.json)。出力：`$OUT/closed/{p0,p5,e_vlm_h}/`（6タスク）、`$OUT/unseen2/{pi05,dist/*}/`（2タスク）。

### π0.5 を使わない版

閉ループでは t0 とグリッパーを π0.5 に頼っている。π0.5 を普段は使わず、困ったときだけ切り替える版を試している。

- π0.5 を使わない信号（キーのスコア、上位10単位の動きの散らばり、直前の行動から下書きへの跳びなど7つ）の組み合わせは、下書き誤差との Spearman が Eval で .83〜.85 で、π0.5 との距離（.77〜.78）より高かった。キーのスコアだけでは弱い（.24〜.52）。
- 最初の実装は、グリッパー・台車・モードを下書き（保存値と0埋め）から取り、t0 も .1 に張り付いて壊れた（Sink 0/6）。この結果は使わない。
- 次に、潜在トークン（E-VLM+H、固定）の上にヘッドを足し、グリッパーと下書き誤差を出す版（RCX6）を回した。t0 は予測誤差を Expert 学習時の対応（誤差の分位→[.1, .9]）に通して決め、Dev で20%になる閾値を超えたら π0.5 に切り替える。腕は Expert、グリッパーはヘッド、台車・モードは学習デモの代表値。π0.5 は切り替え時だけ使う。

| | 6タスク（trial 0–29） | メモリに無い2タスク（trial 0–29） |
|---|---:|---:|
| 通常 π0.5 | 95/180 | 24/60 |
| E-VLM+H＋Expert（t0 は π0.5 との距離、グリッパーは π0.5） | 93/180 | 26/60 |
| **E-VLM+H＋ヘッド（π0.5 は切り替え時だけ）** | **72/180**（π0.5 比 p .0014） | **11/60**（OpenCabinet 1/30） |

6タスクのタスク別（Cab / Sink / Drawer / Fridge / Faucet / Stove）は `12/23/20/7/9/1`。

- **悪化した。** 関門（Sink trial 0–9）は 9/10 で通ったが、6タスク・2タスクとも π0.5 と距離版をはっきり下回った。
- **グリッパー**：ヘッドは開閉の符号一致率 .958 だが、切り替えを ±2 ステップ以内に当てる割合は .136（π0.5 .261、観測から当てる小さな MLP .295）。つかむ・放す時刻がずれる。
- **切り替えが少ない**：π0.5 への切り替えは、閉ループで新2タスク 7.4%、6タスク 3.8%。Dev で決めた20%より少なく、知らないタスクで「自信がない」と出ていない。
- t0 は .1 に張り付かず広がった（Eval で平均 .40、範囲 .10〜.70）。
- どちらが効いたか（グリッパーか t0 か）は、まだ切り分けていない。

根拠：[evidence/latent_heads_report.md](evidence/latent_heads_report.md)、[evidence/latent_heads_closed_summary.json](evidence/latent_heads_closed_summary.json)、[evidence/latent_heads_selection.json](evidence/latent_heads_selection.json)。π0.5 を使わない信号の分析は [evidence/confidence_signals_report.md](evidence/confidence_signals_report.md)。出力：`$OUT/latent-heads/`（`closed/{gate,unseen2,six}/`）。

## ノイズの強さ t0 の決め方と、1回の問い合わせで回す回数

どの版も、Expert は出発点 (1−t0)·下書き＋t0·ε（ε は等方ガウス）から時刻 0 まで **10ステップ**で積分する（刻みは t0/10）。t0 = 1 なら π0.5 と同じくノイズから作り直す。違うのは t0 の決め方だけ。

| 版 | t0 の決め方 | t0 の実際（閉ループ） | 1回の問い合わせのネットワーク評価 |
|---|---|---|---|
| 通常 π0.5 | （ノイズから） | 1 | 10 |
| LIBERO EX1 | 固定 .3 | .30 | π0.5 10 ＋ Expert 10 |
| LIBERO EX2 | 信頼度 MLP（π0.5 と下書きの差、キーの点差など）で誤差を予測し clip(c·√誤差, .05, .6) | 平均 .13〜.18（p10–p90 .07〜.24） | π0.5 10 ＋ Expert 10 |
| RoboCasa 閉ループ（上の表） | π0.5 との距離から、Dev で当てはめた選択器で格子 .1〜.9 から選ぶ | 平均 P0 .61・P5 .49・E-VLM+H .24 | π0.5 10 ＋ Expert 10 |
| RoboCasa 潜在トークンのヘッド（RCX6） | 潜在トークンで下書き誤差を予測し、Expert 学習時の対応（誤差の分位→[.1, .9]）で t0。閾値を超えたら π0.5 | Eval で中央値 .40、平均 .40、範囲 .10〜.70（π0.5 への切り替えは Dev で20%、閉ループでは3.8〜7.4%） | Expert 10（切り替え時だけ π0.5 10） |

RoboCasa の閉ループでは、VLM 平均の2キーは t0 が .1〜.3 と .7〜.9 に二分された（中間は数%）。TurnOnSinkFaucet はどのキーでも t0 が最も小さく（.14〜.37）、ほぼ下書きのまま実行していたのに、スキル側が負けたタスクだった。

## 実行

コードは `skill_moe/skillnet/examples/memory_pilot/robocasa/` の `latent_skill.py`（R1・トークン）、`latent_key.py`（R2）、`latent_expert.py`（R3）。出力は `$SKILLNET_STATE_DIR/outputs/robocasa-memory/latent-skill-01/`。

```bash
source .skillnet-state/env.sh        # このマシンの設定（Git 管理外）
source env/activate.sh
M=skill_moe/skillnet/examples/memory_pilot/robocasa
python $M/latent_skill.py r1                                   # 準備・単位・評価行・上限
python $M/latent_skill.py token-index --nshards 4
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python $M/latent_skill.py tokens --shard $g --nshards 4 & done; wait
python $M/latent_key.py grid --gpus 0 1 2 3 --per-gpu 2        # Dev で設定を探す
python $M/latent_key.py select && python $M/latent_key.py eval
python $M/latent_expert.py drafts --sources e_vlm_h p5
CUDA_VISIBLE_DEVICES=0 python $M/latent_expert.py train --source e_vlm_h &
CUDA_VISIBLE_DEVICES=1 python $M/latent_expert.py train --source p5 &
wait
python $M/latent_expert.py eval --split dev  --experts <学習した重み> --sources e_vlm_h p5
python $M/latent_expert.py eval --split eval --experts <学習した重み> --sources e_vlm_h p5
python $M/latent_expert.py report
```

閉ループ（サーバーは `skill_moe/skillnet/examples/robocasa/serve_latent_policy.py`、`--t0-policy` で t0 の決め方を切り替える）：

```bash
# t0 は π0.5 との距離（6タスク／2タスクの表）。レーン名・サーバーのCPU・クライアントのCPU・ポート・「モード:試行数」
bash docs/research/11_robocasa_latent_expert/evidence/run_closed_loop.sh A 0-7 8-23 8810 p0:30 p5:30
OUT=$SKILLNET_STATE_DIR/outputs/robocasa-memory/latent-skill-01/unseen2/dist \
TASKS="OpenCabinet PickPlaceCounterToStove" INDICES="6 7" \
bash docs/research/11_robocasa_latent_expert/evidence/run_closed_loop.sh B 24-31 32-47 8820 e_vlm_h:30

# π0.5 を使わない版（RCX6）：ヘッドを学習してから閉ループ
python $M/latent_heads.py all                       # extract → fit → evaluate → dev_thresholds → report
bash docs/research/11_robocasa_latent_expert/evidence/run_latent_heads.sh six 8830 "PickPlaceCounterToCabinet PickPlaceSinkToCounter OpenDrawer CloseFridge TurnOnSinkFaucet TurnOffStove" 30 "0 1 2 3 4 5"
```

`confidence_signal_audit.py` は π0.5 を使わない信号の分析（RCX3-A）。`confidence_closed_loop.py` と `--t0-policy conf` は、信頼度で t0 を決める途中の版（最初はグリッパー等を下書きから取って壊れた版、その後に直したが RCX6 へ切り替えて止めた版）で、閉ループの結果は使っていない。`confidence_gripper.py` は観測からグリッパーの切り替えを当てる小さな MLP（RCX5、`--gripper-checkpoint`）。

## このマシンでの準備（記録）

- GPU は A100 80GB ×4。`/work/data` は共有データの読み取り専用マウントで、書き込まない（`latent_skill._guard_output` で拒否する）。
- 環境・データ・重み・出力はすべて `/work/.skillnet-state/`（Git 管理外）。uv のキャッシュは NFS 上だとビルドの一時ディレクトリで `getcwd` が失敗したため、ローカルディスクに置いた。
- RoboCasa で微調整した π0.5 は Hugging Face の `robocasa/robocasa365_checkpoints` の `pi05_pretrain_human300/multitask_learning/75000`。
- コンテナに GLVND（`libEGL.so.1`、`libGL.so.1`）と glib がなかった。`apt-get download` で取った .deb を `.skillnet-state/sysroot/` に展開し、`LD_LIBRARY_PATH` に足して NVIDIA EGL で描画できるようにした（システムには入れていない）。
