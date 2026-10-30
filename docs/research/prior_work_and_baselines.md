# 先行研究と直接の比較対象（2026-10-28）

対象は [anchor_flow_skill_memory.md](anchor_flow_skill_memory.md) §1 の主張 1〜5。研究全体の先行研究の入口として、
(1) 主張ごとに重なる先行研究を対応付け、(2) 既存の二つの調査で扱っていない系統を補い、(3) 実際に走らせる比較対象を一つの表にまとめる。

精読済みの論文の詳細は既存の調査にある。この文書では内容を繰り返さず、節へのリンクで示す。

- [related_work_retrieval_memory.md](related_work_retrieval_memory.md)（2026-10-25）— 検索メモリ・in-context・継続学習・MoE／スキル創発。Optimus-R の精読（§3.1）
- [novelty_anchor_flow.md](novelty_anchor_flow.md)（2026-10-28）— デノイズの開始点・開始分布（RTS、LAFM、OptimusVLA、SL-FM、MOSS ほか）
- [skill_representation_review.md](skill_representation_review.md) — 追加可能なスキル表現の学習

この文書で新しく加えた文献のうち、2026 年のプレプリント（§3.1）は arXiv の要旨ページまでしか確認していない。本文は精読しておらず、数値は著者の報告である。それ以前の文献は書誌と手法の概要を示すだけで、本案との重なりは手法の形から判断した。

## 1. 結論

- **個々の要素はどれも先行例がある。** 開始点・開始分布を検索や運動の潜在から作る研究（RTS、LAFM、OptimusVLA）、ラベルなしの運動単位を作る研究（潜在行動モデル、VQ-BeT、QueST、LOTUS）、勾配なしの経験追加（RTS、TraceFlow、RA-VLA）がある。
- **今回の調査で新たに見つかった直接の競合は二つ。**
  - **REFACTOR-VLA**（2026-09）：ラベルなしで運動プログラムのライブラリを作り、rectified flow のデコーダをライブラリで条件付けて、LIBERO で評価している。主張 1 の「ラベルなしに自己組織化する運動単位」と正面から重なる。
  - **TraceFlow**（2026-09）：凍結した flow-matching 方策に、成功・失敗の軌跡バンクを重み更新なしで足していく。主張 2 と重なる。
- **残る差の候補は、これまでの調査と同じ一点に絞られる。** 可変個数の運動単位を経験から作り、表現・方策・検索器を固定したまま単位を追記するだけで、未学習の場面・タスクへの汎化（RoboCasa365 の Composite-Unseen、実機）が改善すること。raw chunk の記憶、固定個数の学習済み prior、ラベル由来の単位と、同じ条件で比べて差を示す必要がある（§4）。
- **BeT 系（k-means の中心＋残差）は、「単位の中心から始めて状況に合わせて仕上げる」という主張 4・5 の構造の古い先例である。** 査読で必ず指摘されるので、関連研究の節と比較対象（§4 の B8）に入れる。

## 2. 主張ごとの先行研究

| 主張 | 重なる先行研究 | 本案に残る差の候補 | 詳細 |
| --- | --- | --- | --- |
| 1. スキル = 場面に依存しない運動の型。ラベルなしに自己組織化し、個数は固定しない | 潜在行動モデル（Genie、LAPA、UniVLA、villa-X、Moto）、離散スキル（VQ-BeT、QueST、PRISE）、スキル発見（LOTUS、BUDS、XSkill、EXTRACT）、**REFACTOR-VLA** | 実行した行動チャンクと状態から作る運動コードであること。単位数を M6 の機能的距離で決め、固定のコードブックを持たないこと。外観を変えても同じ単位が選ばれるか（転用）を示せるか | §3.1、§3.2、[related §10](related_work_retrieval_memory.md#10-エキスパートプリミティブの創発) |
| 2. 新しい運動は、勾配による更新なしに単位として追加される | RTS、TraceFlow、RA-VLA、RICL、REGENT、IsCiL、非パラメトリックメモリ一般（NEC、DP-means、Prototypical Networks） | raw な軌跡ではなく、圧縮した運動単位を足すこと。同じ容量で raw 記憶を上回り、旧タスクを保持できるか | §3.1、§3.4、[related §3–§6](related_work_retrieval_memory.md#3-最優先で比較すべきメモリvla研究) |
| 3. スキルを外部の条件として与えない（条件は画像・サブタスク・状態の埋め込みだけ） | SL-FM、MOSS（開始分布だけで行動を誘導する）。逆の設計は Optimus-R、SkillNet、LISA、SkillDiffuser（スキルをトークンや条件として入れる） | 経路の違いだけでは新規性にならない（[novelty §5](novelty_anchor_flow.md#5-optimus-rとの差-経路は違うがそれだけでは足りない)）。同じ単位・同じ query で、トークン経路と開始点経路を比べる必要がある | §3.3、[novelty §4.2](novelty_anchor_flow.md#42-lafmsl-fmmoss-source-onlyも無ラベルlatentも既存) |
| 4. デノイズは、根本的な運動を状況に合わせて最適化する過程（取り出した型から始める） | SDEdit、RAGDP、BRIDGeR、R2-Diff、RTS、OptimusVLA、STEP。開始ノイズを学習するもの（DSRL、NSL、UniSteer）。中心＋残差（BeT、VQ-BeT） | 開始点が、自己組織化した単位の復号であること。確信度で t0 と反復回数を決める部分は RTS が既にやっているので、それ自体は差にしない | §3.3、[novelty §3・§6](novelty_anchor_flow.md#3-比較表-開始点ラベル追加時更新確信度) |
| 5. 似た行動を集めると集合ができ、その集合をデノイズの出発点として表せる | LAFM（コードごとの N(μ_k, Σ_k)）、MM-FM（GMM の開始分布）、OptimusVLA（moment matching した単一 Gaussian）、BeT（k-means の中心） | 平均をとらずに成分を 1 つ選ぶ（混合分布を保つ）ことの利点を、多峰的な場面で示せるか | [novelty §4.3・§7.4](novelty_anchor_flow.md#43-optimusvla-複数検索の平均分散とadaptive-noiseまで重なる) |

## 3. 既存の調査に入っていなかった先行研究

### 3.1 2026 年の直接の競合（要旨のみ確認）

| 研究 | 要旨で確認した内容 | 本案との関係 |
| --- | --- | --- |
| **REFACTOR-VLA**: Unsupervised Library Learning of Typed Motor Programs（Shaik, Venkataraman。[arXiv:2609.01215](https://arxiv.org/abs/2609.01215) v1、2026-09-01） | wake/sleep の構成。sleep では、学習した潜在の世界モデルで rollout して求めた行動等価カーネルで運動プログラムの断片をクラスタリングする。MDL と報酬保存の基準を通った抽象化だけをライブラリに入れる。デコーダはライブラリで条件付けた rectified flow。LIBERO 4 suite でクラスタと正解の NMI を報告（例：spatial 0.867、LIBERO-10 0.754、3 seeds）。世界モデルのウォームアップでは supervised contrastive（InfoNCE）を使う | **主張 1 の直接の競合。** ラベルなしの運動単位＋flow デコーダ＋LIBERO という組合せまで重なる。違いの候補は、(a) 本案は単位を開始点にしてトークンとしては渡さない、(b) 単位を勾配なしに追加する、(c) 評価軸が NMI ではなく汎化の成功率であること。条件付けの経路と、追加時に勾配を使うかは本文で確認が必要 |
| **TraceFlow**: Guiding Frozen Flow-Matching Robot Policies with Success and Failure Traces（J. Zhang ほか。[arXiv:2609.20646](https://arxiv.org/abs/2609.20646) v2、2026-09-28） | 凍結した flow-matching の action expert に、検索した成功・失敗の rollout の行動密度から作った有界の補正をかける。TraceBank は rollout ごとに成否 1 bit を持ち、配備後の rollout も追加する。実機の順序付き梱包では 21/50 → 39/50（1 ラウンド追加後 47/50）。シミュでは 26 タスク集計で変化なし（改善は一部の suite だけ） | **主張 2 の競合。** 重み更新なしで経験バンクを育てる設定が同じ。違いは、raw な軌跡か運動単位か、成否ラベルを使うかどうか。補正をかける場所（velocity か開始点か）は本文で確認が必要 |
| **NSL**: Noise Space Learning for VLA Adaptation（Lu ほか。[arXiv:2610.13943](https://arxiv.org/abs/2610.13943) v1、2026-10-09） | VLA を固定し、文脈で条件付けた入力ノイズの分布を PPO／SAC／AWR で学習して適応する。LIBERO、RoboTwin、実機 4 タスク | 主張 4 の隣。開始ノイズを変えて凍結 VLA を適応させる点は同じだが、ノイズ分布を勾配で学習する。本案は勾配なしの記憶から作る |
| **UniSteer**: Unified Noise Steering for Efficient Human-Guided VLA Adaptation（Lu ほか。[arXiv:2605.10821](https://arxiv.org/abs/2605.10821) v2、2026-07-16） | 凍結した flow デコーダを逆に解いて、人の修正行動をノイズの目標に変え、ノイズ actor を RL と一緒に教師する。実機 4 タスクで平均 20% → 90%（平均 66 分） | 同上。「行動 → ノイズ」の逆写像を使う点は、本案のアンカーを開始点に置く操作の代わりとして比べられる |

### 3.2 潜在行動モデル（動画・行動からラベルなしに作る運動の潜在）

主張 1 の「ラベルなしの運動の型」に最も近い大規模研究の系統である。いずれも潜在を **VLA の事前学習の目標や条件** に使い、個数は固定したコードブックで決める。

- **Genie**（Bruce ほか、ICML 2024、[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)）— 動画から VQ の潜在行動を学習する。後続研究の起点
- **LAPA**（Ye ほか、ICLR 2025、[arXiv:2410.11758](https://arxiv.org/abs/2410.11758)）— 潜在行動で VLA を事前学習し、少量の行動ラベルで本物の行動へ写す
- **UniVLA**（Bu ほか、RSS 2025、[arXiv:2505.06111](https://arxiv.org/abs/2505.06111)）— タスクに関係する潜在行動を DINOv2 特徴の上で学習し、言語で外観の変化を分離する
- **villa-X**（Chen ほか、ICLR 2026、[arXiv:2507.23682](https://arxiv.org/abs/2507.23682)）— 潜在行動を本物の行動・状態で接地し、VLA の中で潜在行動の計画と行動生成を一緒に行う
- **Moto**（Chen ほか、[arXiv:2412.04445](https://arxiv.org/abs/2412.04445)）— 潜在の運動トークンを、動画と行動をつなぐ「言語」として使う
- **LAOM**（Nikulin ほか、ICML 2025、[arXiv:2502.00379](https://arxiv.org/abs/2502.00379)）— 外観の妨害があると、ラベルなしの潜在行動は運動よりも外観の変化を捉えてしまうと示した。少量の行動ラベルで改善する

本案との差：本案の運動コードは**実行した行動チャンクと状態から作る**（動画のフレーム差ではない）ので、LAOM が指摘した外観の混入は起きにくいはずだが、まだ検証していない。また、潜在行動モデルは個数を固定したコードブックを使う。追加時に勾配を使わない設計は持たない。

### 3.3 スキルの事前分布・スキルで条件付ける生成

主張 3（スキルを外部の条件として与えない）の対極にある設計である。比較の文脈を書くときに必要になる。

- **SPiRL**（Pertsch ほか、CoRL 2020、[arXiv:2010.11944](https://arxiv.org/abs/2010.11944)）、**OPAL**（Ajay ほか、ICLR 2021、[arXiv:2010.13611](https://arxiv.org/abs/2010.13611)）— ラベルなしのオフラインデータから連続のスキル潜在を学習し、状態に依存するスキル事前分布を持つ
- **LISA**（Garg ほか、NeurIPS 2022、[arXiv:2203.00054](https://arxiv.org/abs/2203.00054)）— 言語の指示から離散スキルを選び、方策をスキルで条件付ける
- **SkillDiffuser**（Liang ほか、CVPR 2024、[arXiv:2312.11598](https://arxiv.org/abs/2312.11598)）— 学習した離散スキルで拡散モデルを条件付けて軌跡を生成する
- **EXTRACT**（Zhang ほか、CoRL 2024、[arXiv:2406.17768](https://arxiv.org/abs/2406.17768)）— 事前学習した視覚言語モデルの特徴をクラスタリングして、ラベルなしにスキルを切り出す

### 3.4 中心＋残差、成長するメモリ

主張 2・4・5 の数学的な形の先例である。ロボットの VLA ではないが、関連研究の節に必ず入れる。

- **BeT**（Shafiullah ほか、NeurIPS 2022、[arXiv:2206.11251](https://arxiv.org/abs/2206.11251)）、**VQ-BeT**（Lee ほか、ICML 2024、[arXiv:2403.03181](https://arxiv.org/abs/2403.03181)）— 行動を k-means／残差 VQ の中心で表し、中心を選んでから残差（offset）を予測する。「単位の中心を選び、状況に合わせて仕上げる」という構造が同じ。違いは、中心の個数を固定すること、仕上げが 1 段の回帰で、デノイズではないこと
- **DP-means**（Kulis, Jordan、ICML 2012、[arXiv:1111.0352](https://arxiv.org/abs/1111.0352)）、**Growing Neural Gas**（Fritzke、NIPS 1994）— 距離の閾値でクラスタを追加し、個数を事前に決めない。M6 の「機能的距離 D と閾値 ε で追加・統合」はこの系統にあたる
- **Prototypical Networks**（Snell ほか、NeurIPS 2017、[arXiv:1703.05175](https://arxiv.org/abs/1703.05175)）— 埋め込みの平均をプロトタイプにし、新しいクラスは勾配なしで足す
- **Neural Episodic Control**（Pritzel ほか、ICML 2017、[arXiv:1703.01988](https://arxiv.org/abs/1703.01988)）— 鍵と値のメモリに経験を勾配なしで追記し、近傍の重み付きで読み出す
- **Modern Hopfield Networks**（Ramsauer ほか、ICLR 2021、[arXiv:2008.02217](https://arxiv.org/abs/2008.02217)）— 取り出しの `w = softmax(β · κᵀ q)` は、連続 Hopfield の 1 回の更新と同じ形である。主張の式を説明するときに引用する
- **DSRL**（Wagenmaker ほか、[arXiv:2506.15799](https://arxiv.org/abs/2506.15799)）— 拡散方策を固定し、入力ノイズ空間で RL を行って誘導する。§3.1 の NSL・UniSteer の前身にあたる

### 3.5 評価の土台

- **π0**（Black ほか、[arXiv:2410.24164](https://arxiv.org/abs/2410.24164)）、**π0.5**（Physical Intelligence、[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)）— ベースの方策。openpi の規約（t = 1 がノイズ）は anchor_flow の式と同じ
- **SkillNet**（Xie ほか、ICML 2026）— このリポジトリのベース。ラベルで監督したスキル MoE
- **RoboCasa365**（Nasiriany ほか、ICLR 2026、[arXiv:2603.04356](https://arxiv.org/abs/2603.04356)）— 365 の家庭内タスク、2,500 のキッチン環境。マルチタスク学習・基盤モデル・生涯学習の評価を想定する。本案の主評価は atomic で単位を作り、Composite-Unseen で汎化を測る。第三者の集計では π0 が全体 14.8%、GR00T N1.6 が 21.9% と低く、改善の余地が大きい（値は公式の表で確認すること）

## 4. 直接の比較対象（実行する baseline）

既存の二つの調査（[related §9.1](related_work_retrieval_memory.md#91-比較対象と公平な予算)、[novelty §10](novelty_anchor_flow.md#10-反論を潰す最小の実験)）の比較対象を、現在の手法（anchor flow）と評価計画に合わせて一つにまとめた。

共通の条件：

- 凍結した π0.5 の VLM、同じアクションヘッドの構造、同じ学習予算（更新回数・データ）、同じ NFE の上限
- メモリに入れる経験は同じにする。保存容量は総 bytes で揃える
- 原著をそのまま再現するのではなく、**同じ土台に機構だけを載せた近似**として比べる。原著の実装を使う比較は別に記す

| ID | 比較対象 | 何を確かめるか（対応する主張） | 優先 |
| --- | --- | --- | --- |
| B0 | 通常の π0.5（t0 = 1、純粋なノイズから開始） | メモリの効果そのもの | 必須 |
| B1 | 固定 t0 のアンカー（保留データで選んだ最良の t0）、t0 を query 間でシャッフル | 確信度で t0 を変える規則が効いているか（4） | 必須 |
| B2 | raw chunk メモリ＋確信度 t0（RTS／RAGDP 型）。同じ経験、同じ bytes | 運動単位に圧縮する意味があるか（1, 2） | 必須 |
| B3 | 検索結果を moment matching した単一 Gaussian から開始（OptimusVLA 型） | 混合分布を保つ（成分を 1 つ選ぶ）利点（5） | 必須 |
| B4 | 同じ単位・同じ query で、単位をトークンとして入れる（Optimus-R 型の経路） | 開始点経路とトークン経路の差（3） | 必須 |
| B5 | ラベル由来の単位（サブタスク注釈ごとに中心を作る）。同じ個数 | ラベルなしで自己組織化した単位に意味があるか（1） | 必須 |
| B6 | 新しい経験で LoRA／アダプタを更新（＋旧データのリプレイ） | 勾配なしの追加で、性能をどれだけ失うか。忘却との比較（2） | 必須 |
| B7 | 固定個数 K の学習済み prior（LAFM 型。K は本案の単位数に合わせる） | 個数を固定しないこと、勾配なしで追加できることの利点（1, 2） | 推奨 |
| B8 | k-means の中心＋残差の回帰（BeT／VQ-BeT 型。デノイズなし） | デノイズで仕上げる必要があるか（4） | 推奨 |
| B9 | 入力ノイズ分布を勾配で学習（DSRL／NSL 型） | 勾配なしの記憶から作る開始点と、学習した開始ノイズの差（2, 4） | 推奨 |
| B10 | SkillNet（ラベルで監督したスキル MoE。このリポジトリのベース） | ラベル付きスキルとの差（1, 3） | 推奨 |
| B11 | 近傍の行動チャンクをそのまま再生（VINN 型） | 生成モデルを使わない単純な非パラメトリック手法の水準 | 推奨 |
| — | REFACTOR-VLA | 主張 1 の直接の競合。コードが公開されていれば同じ評価で比べる。なければ関連研究で議論する | 確認待ち |

評価の順序：

1. S1 軽量版（オフラインの行動誤差）で B0〜B5 と B8 を比べ、差が出ない比較対象を早く見つける
2. S2 の LIBERO 閉ループで B0〜B6
3. 主評価の RoboCasa365（atomic → Composite-Unseen）と実機で、B0、B2、B3、B4、B6 と本案。ここで本案が B2（raw 記憶）と B5（ラベル由来の単位）を上回らなければ、主張 1 を狭める

## 5. 未確認の事項

- REFACTOR-VLA・TraceFlow・NSL・UniSteer は要旨しか読んでいない。条件付けの経路、追加時に勾配を使うかどうか、コードが公開されているかは、本文で確認する必要がある
- RoboCasa365 の baseline の値は第三者の集計で、公式の表とは照合していない
- Moto と DSRL は採択先を確認していない
- この文書は網羅的な優先権調査ではない。投稿前に再調査する
