# Optimus-Rの公開状況、再現仕様、S2向け比較計画

調査日：2026-10-28 UTC。対象はLi et al., **Inline Memory Meets Reusable Skills: Memory-centric Framework for Vision-Language-Action Model**（Optimus-R）、[arXiv:2609.39794v1](https://arxiv.org/html/2609.39794v1)、2026-09-30公開。今回確認できた版はv1。[既存精読の§3.1](related_work_retrieval_memory.md)を前提に、再現に必要な点を本文・付録で再照合した。コード編集・重みの取得・学習・GPU評価は行っていない。

**判断：構成要素の独自実装は可能だが、公式資産と重要な設定を確認できず、現時点で論文数値の厳密な再現を約束できない。** また、原著のtarget適応と、S2の90-only zero-shotは別実験である。

## 1. コード・重み・データ・ライセンス

| 資産 | 2026-10-28の確認結果 | 確認先・利用上の意味 |
| --- | --- | --- |
| 論文 | HTML・PDF公開。v1、2026-09-30。24ページとの著者記載 | [arXiv abstract](https://arxiv.org/abs/2609.39794v1)、[本文・付録](https://arxiv.org/html/2609.39794v1) |
| Optimus-R公式コード | **公開先を確認できない**。未公開と断定するものではない | 論文に対応repoへのリンクなし。[第一著者の論文一覧](https://lizaijing.github.io/)は論文へのリンクのみ。GitHubの論文ID検索は該当0件、著者の公開repo一覧にもRを確認できず |
| 独立project page | **未確認** | [著者の研究紹介記事](https://lizaijing.github.io/blog/en/memory-centric-vla/)には手法紹介があるが、Rの実装・checkpointへのリンクは確認できず |
| Optimus-R checkpoint | **公開先未確認** | Hugging Faceの`OptimusR`・論文ID・`Optimus-R`検索、論文・著者ページを照合。類似名の音声等のモデルは除外。再現に必要なStage A/B別重みも未確認 |
| Optimus-R処理済みデータ・bank | **公開先未確認** | HF dataset検索で該当を確認できず。初期128 prototype、exemplar、Stage別split/抽出index、前処理・実機データの配布先は未確認 |
| ベンチマーク原データ | LIBERO自体は公開 | [公式LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。これはOptimus-Rが使用したデータrevision・train subset・特徴bankが公開済みという意味ではない |
| Optimus-Rコード／重みのライセンス | **未確認** | 論文のarXiv perpetual non-exclusive licenseは論文の配布許諾であり、未発見のソフトウェアや重みにMIT等を割り当てられない |
| 混同しやすい別手法 | [OptimusVLA](https://github.com/lizaijing/OptimusVLA)は別の公式repo、MIT、default branch `master` | Optimus-Rの公開コードとして流用・表記しない。GPM/LCMとRのquery–skill/token経路は異なる |
| 実装の候補基盤 | [openpi](https://github.com/Physical-Intelligence/openpi)は公開、コードApache-2.0 | R論文はπ0.5初期化を述べるが、**著者実装がJAX/PyTorch/openpiのどれかは未確認**。基盤重み・Gemma等の条件はそれぞれ別に確認する |

検索証跡は今回の出力ディレクトリ（ローカル記録: `.skillnet-state/outputs/memory-pilot/research-baselines-optimusr-20261028T172815Z`）に保存した。GitHub論文ID検索、著者repo API、HFモデル・データセットAPI等の応答を含む。「見つからない」という結論の有効日は調査日に限られる。著者への連絡はしていない。

## 2. 実装できる仕様と、不足する仕様

### 2.1 Forward経路と表現

以下は[§3、Appendix C.1、D.3](https://arxiv.org/html/2609.39794v1#S3)で確認できる。

1. 環境時刻`t`の画像・ロボット状態`O_t`、指示`L`、学習可能memory seed tokensをbackboneへ入れる。更新済みの通常prefix `H_vl`と、**memory token位置の出力`M_t`**を得る。既定`m=4`。action denoising途中のhidden stateを値にする方式ではない。
2. `M_t`をquery用・skill用の**別々のattention pooler**で集約する。各枝は学習可能`u_b∈R^d`と`W_b∈R^(d×d)`を持ち、`softmax(u_b^T W_b M_i^T / sqrt(d))`で4位置を加重和する（式14）。
3. 別々のencoder `f_q/f_s`で`q_t,s_t∈R^256`へ写す。`s_t`をSkill Decoderで未来のaction chunk `A_t∈R^(H×d_a)`へ復元し、行動情報を学習する。**sのencoderに未来の実行行動そのものを入力する記載ではない。**
4. Stage Aでは現在の`Tψ(s_t)`、B/C・推論では検索後の`Tψ(s_tilde)`をflow policyへ条件付けする。projectorの出力は`n_s×d`、既定`n_s=4`。
5. 推論ではquery枝だけを使い、skill用pooler/encoder/decoderは不要。通常＋memory prefixをcacheし、skill tokensをcacheへ延長した後にactionを生成する。**開始ノイズ分布をskillで置き換える経路ではない。**

式9→15は**top-2の値を先に集約して4 tokensへ投影**する。一方§4.1に「4 skill tokens per retrieved prototype」という表現があり、単純連結8 tokensと読める余地もある。主経路は式を優先するが、公式実装による確定が必要。

### 2.2 初期bank、検索、追加

| 処理 | 確認できる内容 | 再現上の注意 |
| --- | --- | --- |
| 初期特徴抽出 | Stage A後のinterfaceを固定し、導入データから`(q_i,s_i,A_i,ρ_i)`を抽出。`ρ_i=t_i/T_episode` | 行動chunkと収録時刻・episode全長を使うoffline構築 |
| 間引き | chunk内で隣接actionの変化量を平均し、低変化のsampleを減らす。normalized progressのbinごとにsampling | 低速度そのものではなく**action値の変化量**。位置・回転・gripperのスケールで結果が変わり得る |
| cluster | query空間でcluster後、skill分散が高い群をskill空間で分割 | prototypeの単位は時刻sample群のcluster。タスク1個や意味的運動区間1個とは限らない |
| 保存 | `p_q=normalize(mean(q))`、`p_s=mean(s)`、support数、最大3個のlatent `(q,s)` exemplar。初期128 entry | raw画像・行動をreplay bankへ保存する方式ではない |
| 検索 | `q'=A_mem q`、残差付きkey `p_q+Δp_q`に対する`sim`でtop-2。選択内softmax温度`τ_mem`で重み付け | 学習可能温度。全bankにsoftmaxしてからtop-kする仕様と混ぜない |
| 条件値 | `s_hat=Σ α_j (p_s,j+Δp_s,j)`。必要ならrank-8残差Skill Adapter、次にToken Projector | Adapterの低rank積をzero-initする。**両因子を共にゼロ初期化して勾配を消さない** |
| reuse | 最大類似度`>0.75`なら既存entryへ割当、active residualの学習、supportとFIFO exemplar更新 | base prototypeの平均を上書きする方式ではない |
| append | `≤0.75`ならcandidate buffer。定期cluster後、支持50 sample以上かつquery/skill双方の分散が閾値以下なら追加。key/valueは平均、残差はゼロ | 50は実演数ではない。queryは整合後の座標で新keyを作る。A_memが後に動くと既存keyとの整合も論点になる |

出典：[Appendix C.2–C.5](https://arxiv.org/html/2609.39794v1#A3.SS2)。queryとskillの二つの空間を分けたことだけで、skillが画像・言語から独立した運動表現になる保証はない。

### 2.3 Stage A/B/Cの更新対象と損失

**Stage Bはbackboneを更新する。Stage Cも全固定ではない。** 構造追記の非勾配処理と、全学習系の固定を区別する。[Table 5／Appendix D](https://arxiv.org/html/2609.39794v1#A4)

| パラメータ・操作 | Stage A | Stage B：cross-domain bridge | Stage C：in-domain continual |
| --- | --- | --- | --- |
| Memory seed tokens、両pooler/encoder、Skill Decoder、Token Projector | 学習 | 固定 | 固定 |
| VLA backbone | last layersを学習 | **再び学習** | 固定 |
| `A_mem`、`τ_mem`、activeなkey/value residuals | 使用前 | 学習 | **学習** |
| 既存base prototypes | A後に構築 | 固定 | 固定 |
| 任意のrank-8 Skill Adapter | 未使用 | 学習 | 固定または不使用 |
| bankの構造・統計 | A後に初期化 | Table 5に構造追加の記載なし | reuse／append、support/FIFO更新 |

- **A**：`L_pre = L_fm + 0.1 L_q_nce + 1.0 L_skill`。skill復元はHuber。queryの正例は「同taskかつ進捗差≤0.1」または「同trajectoryで時刻差≤6」。正例のないanchorを除き、全anchorで正例なしならNCE=0。manual skill境界を要求しない一方、**task ID、系列情報、言語条件は使う**。
- **B/C**：`L_life = L_fm + 0.1 ||s_tilde − stopgrad(s_t)||² + 1e−4 Σ_active(||Δp_q||²+||Δp_s||²)`。Bはdomain mismatchに応じてadapterを有効化する。単にbackbone固定でinterfaceだけ学習する実装はBの再現にならない。
- **C replay**：current batch `B`に対して過去latent対`floor(B/3)`を追加するが、使うのはalignment項。`L_fm`はcurrentの行動実演だけ。raw過去軌道を要求しないことと、**新しい対象の行動教師を使わないことは別**。

| 確認済みhyperparameter | 値 |
| --- | --- |
| 共通の報告ハードウェア・規模 | 8×NVIDIA A800、global batch 256、30,000 steps（§4.1） |
| A：新規module LR / 解凍backbone LR | `2e−4 / 2e−5` |
| A：新規module weight decay | `1e−4` |
| A：query対照温度／進捗差／時刻窓 | `0.07 / 0.1 / 6` |
| B/C：A_mem LR／検索温度LR／active residual LR | `1e−4 / 5e−4 / 1e−4` |
| B：任意adapter rank／報告LR | `8 / 5e−5`。Cで学習するという意味ではない |
| C：reuse閾値／新cluster最小支持数／exemplar上限 | `0.75 / 50 samples / 3 per entry` |

出典：[Table 6](https://arxiv.org/html/2609.39794v1#A4.SS2)。8 A800はGPU数で、GPU時間の報告ではない。一般設定の30kを**A/B/Cそれぞれ30k**と解釈しない。Appendix G.5のSpatial逐次実験には別途**10k steps/task**とあるが、Table 1や実機の各stageへ転記できない。

### 2.4 未記載・再確認が必要な仕様

| 不足項目 | 論文から確定できない点 | 影響／補完する場合の扱い |
| --- | --- | --- |
| Stage Aデータと各適応split | 初期interfaceの正確な学習コーパス、task数、各benchmarkとの重複、30/50/70%のsample選択とseed | target漏洩、data efficiencyの再現に直結。最優先で必要 |
| 学習量 | A/B/C別のstep数、in-domain Table 1がどのstageの何runか、Stage Bを省略する条件 | 総学習量・更新予算が一致しない可能性 |
| backbone | 使用π0.5重みのrevision、解凍last layersの数・対象、Bで解凍する範囲、flow/action expertのparameter mask | dense π0.5とSkillNet MoEは異なる。stage別trainable treeが必要 |
| memory stateの層 | 具体的なTransformer層番号、最終normの前後、memory-token挿入位置とattention mask、stateの表現 | queryの条件と分布が変わる。action側hiddenを代用しない |
| encoder/decoder/projector | 深さ・hidden幅・活性化、norm、Skill DecoderのH、action padding/mask、Huber delta、Tψの具体的層 | **2層MLP/GELU等を原著仕様と断定できない**。独自選択なら明記 |
| similarity・温度 | `sim`の具体的定義、query/residual keyのnormalization、τ_mem初期値・正値化・下限 | 本文の確認範囲で`cosine`の明記は見つからず。可視化用L2正規化と学習実装を混同しない |
| bank構築 | 初期clusteringアルゴリズム、K=128と再分割後Kの整合、停止規則、分割閾値、progress bin数、間引き確率 | 勝手なk-means/均等bin設定は再現仕様の補完である |
| Cの追加規則 | candidate clusterの実行周期、ε_q/ε_s、buffer上限、sample重複処理、初期candidate cluster数 | 追加数・重複・忘却を左右する |
| Bのadapter起動 | novelty/alignmentの閾値、連続batch数、初期A_memのidentityへの近さ | oracleでtarget成績を見て有効化しない |
| optimization | optimizer、β/ε、LR schedule、warmup、clip、EMA、backbone BのLR、lossのreduce方法 | AdamW等は候補にできるが未確認の原著値として扱わない |
| multi-GPU NCE | 256全体でnegativeを共有するか、GPU内batchのみか、近傍正例を含めるsampler | **gradient accumulationだけでは対照損失の候補集合を再現できない** |
| 評価 | LIBERO commit・image処理・normalization・chunk実行数・NFE・horizon上限・seed・init states | 500/suite以外の細部不足。1 pp差の解釈に影響 |
| token数の記載 | 式では集約後4tokens、§4.1はper-prototypeと表現 | 4/8tokensを未確定のまま混ぜない |

これらが未解決なら、実装を**「論文に基づく独自実装（補完設定一覧付き）」**と呼ぶ。原著のcheckpoint再評価・完全再現とは呼ばない。未知値をtarget test成功率で選ぶことはS2では禁止し、source90だけのvalidationで固定する。

## 3. 二つに分ける再現・比較

### P：原論文のLIBERO設定を再現する段

目的は原著のTable 1の数値と機構を再現できるか確認すること。**こちらのS2評価とは別のデータ領域・出力を使う。**

1. 公式code/weightsの公開を再確認する。見つかれば論文・著者ページとの対応、ライセンス、commit、checkpoint hash、Stage Aの学習元を記録する。別のOptimus名のrepoを代用しない。既存checkoutを上書きせず、専用ディレクトリで公式手順を使う。
2. 公式の**full-data checkpoint**があれば、まず4suite各500 rolloutでTable 1の`99.2 / 99.4 / 98.4 / 94.2、平均97.8%`を再評価する。同じ設定のπ0.5は`98.8 / 98.2 / 98.0 / 92.4、平均96.9%`。checkpoint再評価と、学習からの再現を区別する。
3. 公式codeのみなら、公開Stage A checkpointがあればそこから適応し、なければπ0.5からA→bank構築→論文のin-domain適応を実行する。**AのコーパスとTable 1へのstage割当が不明な現状では厳密な設定一致は未達**。Bはcross-domain用であり、Table 1を再現するために無条件でBを足さない。
4. codeがなければ§4のopenpi実装で進めるが、不足仕様を固定した独自再実装と明示する。最小では100%条件を先に確認し、次に30%条件でLong 86.0%、平均88.6%対π0.5の62.0%、72.1%というdata-efficiency差が残るかを見る。全割合×全benchmarkを最初から回さない。
5. 初回は50/taskの1 run、差が小さい最終報告は複数seed。原著はseed別分散未確認なので、単一の集計値との一致だけで再現成功と断定しない。task別・suite別、paired初期状態、区間推定、bank K、失敗分類、実際の更新数も保存する。追加の学習seedは1 run分の学習時間を要する。
6. ablationは既定`K0=128, Kr=2, m=4`の97.8%に対してtop-1の97.0%をまず確認する（Table 10）。検索完全off、uniform和、全固定append-onlyは原著未報告なので、**自分たちの追加ablation**として示す。これらを原著の数値再現判定に使わない。

**現時点の到達状況：Pは未実施。公開checkpointも未確認で、97.8%を再現できたという証拠はない。** 仕様が揃わない場合もS2向けの機構比較は進められるが、P成功の代わりにはならない。

### S：SkillNet π0.5・90-onlyのS2に合わせた比較段

目的はtoken条件付け経路と、提案の開始分布経路を**同じsource-only条件**で比較すること。こちらの移植版は`Optimus-R-style / SkillNet90 / no-target-adaptation`等と表記する。

1. 共通の`pi05_libero_moe_skill_4_90`、source90のデータrevision、action/state正規化、画像・subtask生成、既存skill入力、rollout上限を固定する。**`pi05_libero`やSkillNet40、Pで対象40taskに適応した重みを初期値に使わない**。後から90だけで学習しても既存のtarget学習履歴は消えない。
2. source90をepisode/scene単位で導入学習・validationに分け、memory interfaceをStage Aの損失で学習し、sourceのみからbankを構築する。原著NCEを再現する比較側はsource task ID・進捗を使うことを明記し、提案のラベルなし性と混同しない。eval task IDは入力・bank構築の教師として与えない。
3. Aではencoded skillを与えるため、推論時のretrieved skillとの分布差が残る。**source90だけで**backbone/interface固定のStage C型alignment・residual学習を追加し、retrievalを使って行動できることをsource validationで確かめる。これは原著のtarget Stage Cの代替ではなく、S2条件に合わせた導入学習である。sourceの導入中にbankを増やすかは予め固定し、学習終了時Kとbytesを報告する。
4. targetではA_mem、温度、residual、adapter、bankを**全て固定**し、Long・Skill02〜09を評価する。対象デモも対象成功軌道も保存しない。S2ではStage B/Cによる**target**適応は実行しない。テストを見てεやtop-kを調整しない。
5. 同じsource-only初期値に対して、元SkillNet、source微調整後のメモリなし、R-style token条件付け、R-style shuffle、提案の開始分布条件付けを比較する。追加のsource更新量・解凍範囲・学習時間を揃えるか、完全固定と更新ありを分けて表示する。原著Aのlast-layer更新と提案のVLM固定が一致しなければ、**両方VLM固定の追加対照**を置き、原著変更版と明記する。
6. Longは10×50=500、Skillは8×50=400、計900 rollout/model/seedを最終評価の最小単位とする。pilotは20/task。source90 task0〜9の保持、SkillとLongの重複を除いた集計も確認する。文献のSkillNet全9タスク47.8%をこの8タスクのbaselineにしない（再集計44.5%、π0.5 28.5%）。[成功率比較表](baselines_success_rates.md)

原著の新タスク追加性能はこのS2では評価しない。append-only対Stage Cを調べるなら、対象経験を許す独立した継続追加トラックが必要で、source-only zero-shot成績と混ぜない。

## 4. code非公開時のopenpi上の実装範囲

以下は**今後の実装計画**であり、この調査ではファイルを変更していない。ローカルはJAX/Flax NNXのSkillNet MoEで、原著のdense π0.5との構造差がある。

| 作業 | 必要な変更の範囲 | 規模 |
| --- | --- | --- |
| memory prefix | `embed_prefix`周辺でseed token・mask/positionを追加し、Gemma呼出しから該当位置の出力を取得 | 中。既存prefixのattention・state表現を理解して実装 |
| 二枝interface | 式14のpooler、q/s encoder、Skill Decoder、Token Projector、stage別parameter mask、補助損失 | 中。encoder幅等は独自補完が必要 |
| token条件付け | `sample_actions`と学習forwardを同じ経路へ統一。memory prefix→q→bank→4 skill tokens→prefix cache延長→flow | **大**。mask・position・KV cache・学習時の勾配を扱う |
| bank・replay | 抽出、二段cluster、固定base/可変residual、active index、candidate/FIFO、再開用serialization | 中〜大。既存checkpointとbankを同時version管理 |
| training | A/B/C別optimizerとtrainable tree、NCE正例sampler、Huber、alignment/replay、温度正値化 | 大。新moduleのcheckpoint補完と再開互換を含む |
| 評価 | 既存LIBERO clientを使い、query/index/weight、K、latency、成功率、更新有無を記録 | 小〜中。評価中のbank書込みを禁止できること |

重要な実装上の切り分け：

- SkillNetの`gemma_moe_skill.py`は、skill embeddingを**トークンごと・層ごとのMoE routing**へcross-attentionする。Rのskill tokensを既存`skill_embeds`に差し込むだけでは、**flow policyのprefix条件付けを再現したことにならない**。初回は既存router入力を共通のまま保持し、Rの経路を別に足す。
- 現在の`pi0_moe_skill.py`はprefixの出力を捨ててKV cacheだけを保持する推論経路があるため、qを得る出力の受け渡しが必要。skill tokensは新しいprefix入力として各層で処理し、cacheを延長する。任意のhiddenをKV末尾へ貼るだけでは同等でない。
- prefix/memoryを作る際に未来のtarget actionやflowのnoisy-action suffixを読ませない。robot stateが実際にqueryから見えるかはπ0.5のtokenizer/configとmaskまで確認する。原著の記号`O=(image,state)`だけを根拠に可視性を仮定しない。
- A→bank→Cの導入を省き、未学習のToken Projectorを既存の凍結action expertへ足すだけの失敗を「Optimus-Rは効かない」と解釈しない。
- Top-kのindex選択は離散で、選ばれたkey/value・温度・A_memへの勾配を残す。flow lossからskill tokensへの勾配が必要なので、**backbone重みを固定しても経路全体をstop-gradientしない**。Cがpureな軽量MLP学習になるわけではない。

最小の実装順はA＋固定128bank＋top-2＋token経路→source内C→S2。cross-domainのB・adapter動的起動・targetでのappendはS2の必須部分ではなく、原著全機構の再現段で追加する。Bを省いた版を原著full pipelineとは呼ばない。

## 5. H200 1枚でのGPU時間：未実測の計画見積もり

論文は8×A800・batch256・30k stepsを記すが、wall time・GPU-hours・stage別step数は報告を確認できない。以下は**著者の実測でも、ローカルH200で測った値でもない**。架空の精密値を避けるため、global batch256のoptimizer 1 updateをA/B **8〜24秒**、C **3〜10秒**と仮置きした感度計算である。共有GPUの空き、MoE、NCEの実装、画像token数、解凍範囲、再計算で範囲を外れ得る。8 A800からH200への単純な枚数比例は使っていない。

`GPU時間[h] = optimizer updates × 秒/update ÷ 3600`。一枚を割当てた実行時間として計算し、競合時の待ち時間・データ準備・実装工数は含めない。

| 計画上の作業 | 仮の更新数 | 仮の秒/update | H200×1の時間 | 位置付け |
| --- | ---: | ---: | ---: | --- |
| A動作pilot | 2,000 | 8〜24 | 4.4〜13.3 h | 数値再現の主張には使わない |
| C動作pilot | 1,000 | 3〜10 | 0.8〜2.8 h | sourceだけで検索経路の勾配を確認 |
| Aの本学習シナリオ | 30,000 | 8〜24 | **66.7〜200 h** | 30kをAへ割当てる仮定。原著stage配分は未確認 |
| Bを30k行うシナリオ | 30,000 | 8〜24 | 66.7〜200 h | cross-domain段を行う場合のみ。S2には不要 |
| source Cの比較シナリオ | 10,000 | 3〜10 | **8.3〜27.8 h** | S2用の独自予算。原著Table 1のstep数ではない |
| Cを30k行うシナリオ | 30,000 | 3〜10 | 25〜83.3 h | 原著Cに30k必要という意味ではない |
| source特徴抽出・bank構築 | フレーム数と抽出strideで変化 | 未実測 | **1〜10 hを仮予約** | 全574,571フレームか間引きかで大きく変わる。cluster CPU時間は別 |

この仮定では、**S2移植版1 seedはA30k＋source C10k＋特徴抽出で約76〜238 GPU h（約3〜10日）**。pilotは約6〜27 h＋評価。仮にA/B/Cを各30k行う別シナリオなら合計約160〜493 h＋評価だが、これは**原著の必要学習量の推定ではない**。まず計測して予算を更新する。手法比較の複数seed・公平な対照学習は別に加算する。

実行時に見積もりを確定する手順：

1. H200の実際の空きVRAMを確認し、小microbatchで100〜200 effective updatesを計測する。compile/warmupを除く中央値・p90、peak VRAM、images/secをAとCで別々に記録する。**この調査では実行しない**。
2. global256は、例えばmicro8×accum32で合わせる案があるが、収まるかは未確認。NCEをmicro8内だけで計算すると256 batchのNCEと異なる。元実装のnegative集合が不明なうちは、effective batch全体で正負例を扱う実装・再計算コストを明記し、単純accumだけを同等としない。
3. S2評価は900 rollout/model/seed。仮に1 episodeの実行が5〜120秒なら**1.25〜30 h/model/seed**の実行時間になる。多くのSkill失敗が1,600 control stepsに達すると長い側へ寄る。これはCPU simulatorを含む仮のwall timeで、GPUの純演算時間とは異なる。実測のepisode長・action query数・推論時間で置き換える。
4. 4suiteの論文再評価は2,000 rollout/model/run、同仮定なら2.8〜66.7 h。raw前処理やfull LIBEROのdownloadを評価GPU時間へ混ぜない。公式checkpointが得られればPのcheckpoint再評価はA/B/C学習を省けるが、その重みをSへ流用できるかは学習履歴次第。

実施時は[共通gate文書](memory_pilot_gates.md)に従う。今回の依頼は調査だけであり、gate段階の凍結実験を越える学習は開始していない。将来の実行も`XLA_PYTHON_CLIENT_PREALLOCATE=false`を維持し、Pythonには`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=8`（複数processなら合計32以下）、`taskset -c 8-47`、長時間処理は`nice -n 10`を用いる。PyTorchを使うなら`torch.set_num_threads`も同値。各runを新しいoutputディレクトリへ保存し、既存checkpoint・log・他担当の変更は上書きしない。

## 6. 完了判定と優先順位

| 優先度 | 得るべき証拠 | 満たさない場合の表記 |
| --- | --- | --- |
| 1 | 公式資産、Stage Aデータ、stage別steps・解凍範囲、bank閾値、token仕様 | 「論文数値の厳密再現は未達」 |
| 2 | Aで行動復元が成立し、source bank検索で性能を保ち、Cで許可したparameterだけが変わる | 学習が成立していない移植版を競合手法の失敗として扱わない |
| 3 | P：同じdense π0.5・対象4suite設定でTable 1の値とablationの傾向を検証 | SkillNet MoE上だけの実験は「原著再現」ではない |
| 4 | S：同じSkillNet90・source予算・targetアクセスなしでtoken対開始分布を比較 | 「S2向け機構比較」と明記し、Pの学習済み参考値と分離 |
| 5 | no-memory/shuffle、task別・seed別、計算量・bank容量、source保持と重複除外集計 | 成功率だけでは検索内容や運動再利用の寄与を証明できない |

最初に必要なのは大規模なB/C再学習ではなく、**原著不足仕様の解消、source-onlyなtoken経路の導入検証、同一条件での無改変SkillNet90評価**である。
