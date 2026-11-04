# DMP型の運動原始・VLAルーティング：関連研究と新規性

調査日：2026-11-02 UTC。対象は [dmp_routing_plan.md](dmp_routing_plan.md)。**分割・事前基準は変更しない。** 本文・付録・表を一次資料で確認した。未追試であり、異なるデータ・観測・試行数の成功率を横断ランキングには使わない。

**結論：DMPの形／ゴール／時間の分離、ネットワークによるパラメータ予測、運動原始と拡散の結合、無注釈分節化、原始の追加は、それぞれ既存である。差の候補は「意味ラベルを使わず形成した、個数を固定しない形の辞書を1本選び、場面から幾何・時間を指定してVLAの生成を実際に支配する」という組合せと、その因果的な検証にある。現在のS4は埋め込みと出力行を学習するため、全パラメータ固定／勾配なし追加とは呼べない。**

可変長単位のPRISE／QueST／LOTUS／BUDS／LOVE等の詳細は [related_action_units.md](related_action_units.md)、言語ベンチマーク・Hi Robot／π0.5／SayCan等は [abstract-prompt調査](/work/.skillnet-state/outputs/memory-pilot/abstract-prompt-20261101/related_work.md) を参照。本稿はDMPとの接続と、その後に確認した近接研究を補う。

## 1. 今回の表現のうち、古典的DMPですでにあるもの

標準的な離散DMPは、概略 `τ² ÿ = α(β(g−y)−τ ẏ) + f(x;w)`、`τ ẋ = −αx x` と書ける。`w`が形、`g`が収束先、`τ`が時間尺度を担い、1本の実演からforcing termを回帰できる。**形を保持して開始点・ゴール・速度を変えること、勾配降下を使わずに1本から原始を作ること自体は新規性ではない。** 位置と姿勢の扱い、空間正規化の退化、連結、接触時の適応はDMP研究の既存論点である。[Saveriano et al., Dynamic Movement Primitives in Robotics: A Tutorial Survey, arXiv:2102.03861v1](https://arxiv.org/html/2102.03861v1)（2021-02-07、§2–4。IJRRの最終版とは版を区別）。

設計書の`phase_s: 0→1`は使えるが、上記のcanonical phase `x: 1→0`とは別の規約である。**物理的な進み具合、DMPのcanonical phase、flowの補間時刻を区別する。** また`g−y0≈0`の戻り運動・回転だけの動作では、開始→ゴール方向・距離だけの正規化が退化しうる。単一ベクトルは3D座標系の残る回転自由度も決めない。「正規化したwが場面非依存」とは仮定であって、正規化だけで証明された性質ではない。

## 2. ネットワークが運動原始のパラメータを出す系

| 研究・確認版 | 何を予測・生成するか／監督 | 表から確認した結果・範囲 | 今回との差 |
| --- | --- | --- | --- |
| **Neural Dynamic Policies（NDP）**, Bahl et al., 2020 preprint／NeurIPS 2020：[2012.02788v1](https://arxiv.org/html/2012.02788v1)（2020-12-04）、§2–4、Tables 1–2 | 画像または状態→**DMP weights `w`とgoal `g`**→微分可能ODE rollout。模倣・RLの両方でend-to-end学習。離散スキル名・境界辞書は作らない | MuJoCo模倣のFaucetはNN **.532→NDP .790**、Push **.002→.208**。一方Pickは**.672→.408**と悪化（Table 1）。数字描画のheld-out lossはCNN-DMP8.46、NDP .74（Table 2）。LIBERO・言語頑健性の評価なし | 「prefixからDMPパラメータを出す」は既存。今回の差は、wを連続回帰するのでなく**共有された辞書から1個選び、g・τ等を分離**すること |
| **ProDMPs**, Li et al., RA-L 2023：[2210.01531v1](https://arxiv.org/html/2210.01531v1)（2022-10-04）、§III–V、Tables I・III–IV | DMPのODEをbasisに変換し、ProMP型の分布表現と統合。画像等からweight分布の平均・共分散を予測し、初期位置・速度を条件に滑らかな軌跡を得る | Table IIIの100 picksで実機 **mean 87%、sample 75%**、simulation99%／82%。画像条件の描画、Bayesian aggregation等も評価。言語・自動辞書化なし | MPパラメータをニューラル生成すること、初期条件付きdecoder、滑らかな再計画は既存。確率分布を学ぶProDMPと、1本の形を選ぶ本案は分ける |
| **Deep Segmented DMP Networks（DSDNet）**, Anarossi et al., CASE 2023：[2309.00320v1](https://arxiv.org/html/2309.00320v1)（2023-09-01）、§III–V、Tables I–III | 画像→区間数`n`と複数区間の`y0,g,τ,w`。**上限M個**を出し、必要区間数を予測。共有decoderで区間間のwの構造を学ぶ。切断等の区間化された教師を用い、全taskに共通の個数未固定辞書を無注釈発見する方式ではない | real cuttingデータのRMSEは**.016m**（比較CNN-DMP .014m）、DTW **.418 vs .762**、著者のtask-achievement基準で100% vs13.636%（Table III）。後者をLIBERO型の閉ループSRと混同しない | 可変長のDMP列とτを予測する構成は既存。**辞書の増分形成、未知原始追加、VLA／言語、1本の選択**は本案と異なる |

## 3. 拡散・flowと運動原始の結合：直接の近接研究

| 研究・確認版 | 原始と生成モデルの接続 | 表の数値／評価条件 | 差・注意 |
| --- | --- | --- | --- |
| **Movement Primitive Diffusion（MPD）**, Scheikl et al., RA-L 2024：[2312.10008v2 PDF](https://arxiv.org/pdf/2312.10008v2)（2024-06-10）、§III-D、Table I | **ノイズを加えたaction列→ネットワークがProDMP weightsを出す→ProDMP decoderでaction列へ戻す**。denoiserの内部に構造化decoderを入れる。元論文はweight空間への直接拡散ではなく、**action列空間の拡散**を採ると明記 | 4つの手術・変形物simulation平均SRはstate **97.90%**、image **94.40%**（DP-T97.00%／93.05%）。5seed、100rollout、最高SRのepochを使用。実機評価もあるが本稿では図の値を転記しない | **MP＋拡散は既存**。本案は辞書の1形を先に選んで展開し、それをbridge開始点にする。MPDは各denoise段でwを予測・decodeし、離散辞書選択なし。DMPを開始点にするだけでは、MPDのような出力の構造制約は得られない |
| **FODMP**, Shi et al.：[2603.24806v1](https://arxiv.org/html/2603.24806v1)（2026-03-25）、§III–IV、Table I | ProDMPパラメータ`[w,g]`への生成を**consistency distillationで1ステップ化**。1回選んだパラメータから時間構造を持つ動作を生成する。離散primitive library・意味ラベルなし分節化ではない | MetaWorld／ManiSkillの難度群平均SR **78.2%**、MPD64.1%、DP50.1%。推論 **17.2ms**、MPD168.6ms、DP119.7ms。3seed、各task50rollout、最後10checkpoint平均。DP等はposition control、MPD等はvelocity controlを用いるため表の差を純粋な表現だけの効果としない。実機Push-T／ball catchあり | 「短いchunkより時間構造を持つ原始を生成する」は既存。今回に近い動機だが、原始の**発見・共有・追加・言語による選択**は対象外 |
| **DROM**, Pomponi et al.：[2609.37348v1](https://arxiv.org/html/2609.37348v1)（2026-09-29）、§3–5、Tables 1–10 | **名前付き17skillの実演をDMPで増幅**し、別途language-conditioned diffusionで軌跡を直接生成。LLMが上位指示をskill列へ分解する。DMPは主に**訓練データ生成**に使い、推論時に選んだwをbridgeへ入れる方式ではない | Table 4の実機Stack Cube **MPD44→DROM94%**、Open Drawer50→86%。Table 6の長期taskはHide70.4%、Clean Up60.9%。Table 7/10のOverallには構成認識率×実行率の積で示す値もあり、全てを同一の一括閉ループ試行数からの比率とみなさない | DMPで形を保ちg等を変える＋skill条件diffusion＋言語階層は既存。既知skillの実演と名称を使う。**無注釈辞書、MoE routing、固定追加は異なる**。同名のMPDはここでは主に *Motion Planning Diffusion* の比較であり、上段の *Movement Primitive Diffusion* と区別 |
| **Sparse Compositional Flow Matching（SCFM）**, Tang et al.：[2605.23341v1](https://arxiv.org/html/2605.23341v1)（2026-05-22、HTML題名 *No Magic, Just Building Blocks*）、§3、Tables 2・5 | learnable length mask付きのmotion atom辞書と**二値の開始位置行列**を学び、その配置行列をflowで生成。物理空間で原始を配置・連結し、連続性と時間的な隙間を幾何lossで制約。辞書サイズMは設定・ablation対象 | OXEのhard群ADE/FDE **.006333/.008919**、比較LaM-SLidE .007841/.011282（Table 2）。主要評価はtrajectory prediction、**閉ループtask成功率・抽象指示・実機方策効果は未確認** | **運動辞書＋flowによる構成**は既存。DMPのw/g/τ分離ではなくatomの配置生成。個数固定なし・実行中の単一選択・共有VLAによる適応・後からの追加が差の候補 |
| **Smooth Action Chunking via Movement-Primitive Representations**, Nomura et al.：[2610.39451v1](https://arxiv.org/html/2610.39451v1)（2026-10-29）、§IV–V、Tables I–III | ACT／DiTがsmooth basisの係数を出す。ProMPのclosed-form conditioningで、前chunkの最終commandをvia-pointにして次chunkを接続。**接続補正には再学習不要**。時間窓自体は固定 | 実機ACT Pick-and-Passのwithin-chunk jerk **280.1→18.9**、SR80.0→73.3%（+MP）、+CondのSR83.3%（Table II）。成功率の一貫した改善を示す論文ではない。simulation DiT Transfer Cubeは85.7→77.3%と低下（Table I） | **滑らかさの改善とtask成功は別**という直近の対照。今回はsmooth basis採用の効果だけをスキル発見の効果にしない。原始の辞書化・routing・意味理解は扱わない |

さらに、**情報を持つ下書きから生成するbridge自体は既存**である。[BRIDGeR, arXiv:2402.16075v2](https://arxiv.org/html/2402.16075v2)（2024-05-02、§III–IV）は任意のsource policyからstochastic interpolantsで目標方策へ移す。本案の差は、sourceが単なる検索action chunkではなく**選択した形を予測g・τで展開した動作**であること。このsourceの質と因果的寄与を示す必要がある。

## 4. 無注釈の分節・個数未固定の辞書化・後からの追加

| 一次資料 | 何がすでにあるか | 本案との境界 |
| --- | --- | --- |
| **Niekum et al., Learning and Generalization of Complex Tasks from Unstructured Demonstrations**, IROS 2012：[著者本文PDF](https://people.cs.umass.edu/~sniekum/pubs/NiekumIROS2012.pdf)、§II–IV。**arXiv ID未確認** | **BP-AR-HMMで区間・繰り返すmodeを推定し、DMPを作る**。単位数やtask構造を先に与えず、物体座標系を使って新しい配置へ動作を移す。PR2実機で多段taskを実行する | **無注釈・個数未固定・DMP・再利用という大枠は既存**。現代VLAのprefixによる選択とflowへの接続が違う。新規形追加だけの機能を新規としない |
| **Lioutikov et al., Learning Movement Primitive Libraries through Probabilistic Segmentation**, IJRR 2017：[著者本文PDF](https://gjmaeda.github.io/pdf/lioutikovLearningMovementPrimitive2017.pdf)、[出版元](https://doi.org/10.1177/0278364917713116)。**arXiv ID未確認** | 分節と**確率的MPのライブラリ学習を相互に改善**するProbS。異なる時間尺度の同じ形を同一原始として扱い、実演のbit encodingによるコンパクトさも評価。実機で文字列・椅子組立、未実演の組立順序で再利用 | S1の「再構成可能な区間を作り、共有形の辞書へ」という発想は既存。貪欲誤差切断＋DP-meansは別の具体化だが、その置換だけでは強い新規性になりにくい |
| **Daab et al., Incremental Learning of Full-Pose Via-Point Movement Primitives on Riemannian Manifolds**, 2023 preprint：[2312.08030v1](https://arxiv.org/html/2312.08030v1)（2023-12-13）、§IV–VI、Tables I–II | **追加・更新・削除・mode統合・分割**等のライブラリ操作を定式化し、位置・姿勢を逐次更新。Table Iでadd/improveとmergeの推定がbatch推定に数値精度で一致。評価は逐次mocapデータで、VLA成功率ではない | 勾配を使わない／統計量による原始更新・追加も既存。原始パラメータ自体は更新するため、あらゆる意味で「重みが不変」ではない。ニューラル重みを固定した選択・生成と区別する |
| **DynaMimicGen**, Pomponi et al.：[2511.16223v1](https://arxiv.org/html/2511.16223v1)（2025-11-20）、§3–4、Tables 1–4 | 区間ごとDMPを1実演等からfitし、物体poseの変化でgを更新してデータを増やす。**既知の物体中心subtask列、各時刻の物体pose、taskに応じた区切り情報**を仮定 | 自動処理と「subtask注釈や知識を使わない」は別。Table 3のimage DP・MugCleanup D1はD-MG60.67%、MimicGen38.00%。これは増幅データで学んだ方策の結果で、凍結モデルへ原始を追加した結果ではない |

DP-meansも既存である。[Kulis & Jordan, Revisiting k-means: New Algorithms via Bayesian Nonparametrics, arXiv:1111.0352v2](https://arxiv.org/html/1111.0352v2)（2012-06-14）は距離和にcluster数のpenaltyを加えてKを決める。**固定Kを避けることは妥当だが、それ自体が新規性ではない。** λとDMP分節のεをTrainで決める現在の方針を守り、MDLの符号化・残差モデル・位置と回転の尺度を明記する。λを変えてDevの通過率へ合わせない。

PRISE／BUDS／LOTUSとの細部の比較は既存の [行動単位調査](related_action_units.md) §2–3に譲る。特にLOTUSは辞書の追加時に方策を学習する。**「生の軌跡を記憶」「primitiveのパラメータを記憶」「expertネットワークを増やす」**は、追加コストと表現能力が違う。

## 5. 原始・スキルでMoEをルーティングする研究

| 一次資料 | expert／routingの単位と監督 | 設計書との差と証拠の強さ |
| --- | --- | --- |
| **SkillNet**, ICML 2026：[著者PDF](https://vipl-vsu.github.io/skillnet-website/static/pdfs/skillnet.pdf)、§3.2–3.3、Tables 1–3。**arXiv ID・PDF版番号は未確認** | motion code／VerbNet／skill realizationの階層を作り、**スキル埋め込みをMoE routingのsoft constraint**にする。VLMによる意味・運動属性の注釈を使う。DMP wの自己組織化ではない | Table 1のLIBERO-Skillはπ0.5 **31.8%**、vanilla MoE33.1%、SkillNet47.8%。これは論文値で、設計書の公開checkpointの実測53.8%とは別。主張は未見**動作合成**であり、抽象指示耐性ではない。今回の差は注釈カテゴリ一覧から、現在の状態に応じて選ぶ**自前の運動単位**へ置き換える点 |
| **ECS: Emergent Compositional Skills in Mixture-of-Experts VLAs**, Shah et al.：[2607.20771v2](https://arxiv.org/html/2607.20771v2)（2026-07-24）、§2–3 | π0／SmolVLAのaction FFNへLoRA expert（rank16）を付け、視覚・言語・状態からrouting。**chunkの選択を全action層で共有**する。境界・skill IDの人手教師なし。16expertの固定容量、DMPではない | approach／transport／release等の再利用と手動routing介入を示す。ただし定量成功は主に図であり、本稿は図から値を推測しない。expertが運動段階へ分化するという主張は既存。**物理的な形・g・τ・phaseと結び付いた辞書**、個数未固定、追加を別に示す必要 |
| **AtomicVLA**, Zhang et al.：[2603.07648v2](https://arxiv.org/html/2603.07648v2)（2026-09-06）、§3.2–3.3、Tables 3・5 | high-levelのtask chain／進捗／atomic skill abstractionを生成し、skillのscalar embeddingでrouterを条件付ける。意味的なatomic annotationを使い、DMP形の無注釈発見ではない | Table 3の対応するπ0.5ベース比較は45.0→AtomicVLA*63.3%。長期taskの結果で、同義・抽象度を統制した言語耐性ではない。**skill・phaseをrouterへ渡すだけでは新規性にならない** |
| **Compositional Motion Generation from Demonstration with Object-Centric Neural Fields**, Tekden & Bekiroglu：[2607.07129v1](https://arxiv.org/html/2607.07129v1)（2026-07-08）、Method、Table I | object-centric neural fieldを共有し、**temporal MoEでobject-conditioned movement primitiveを混ぜて軌跡生成**。DMPではなく、実演の時刻がtask phaseに対応するよう概ね整列されていると仮定 | Table Iの低データCube Stacking accuracyは**.97±.02**、gatingなし.85±.06。これは静的観測からのtask-specific trajectory生成で、VLAや無注釈の可変長辞書ではない。**primitive＋MoEも既存**だが、候補を混ぜる点は本案と明確に違う |

**辞書のprimitive数Kとニューラルexpert数は別である。** 現設計はK個のDMPを4個等のFFN expertで実現しうるため、`prim_id = expert_id`とは限らない。公開実装の [gemma_moe_skill.py](../../../skill_moe/skillnet/src/openpi/models/gemma_moe_skill.py#L376) ではskillのcross-attentionからrouter logitsを作る（423行付近）。S3でadaRMSとbridgeにも注入すると、primitiveは複数経路で作用する。**primシャッフルで落ちるだけでは、MoEが原始ごとに分化したとまでは言えない。** 設計書の「スキルを替えても行動がほぼ不変」は現在の診断結果であり、SkillNet論文全体の否定として一般化しない。

## 6. 抽象・間接指示への頑健性を運動原始で改善したか

### 6.1 直接近い2026年の研究

| 一次資料 | 手法・追加の扱い | 表で確認した結果と解釈 |
| --- | --- | --- |
| **Language Movement Primitives（LMP）**, Dai et al.：[2602.02839v3](https://arxiv.org/html/2602.02839v3)（2026-05-22）、§IV–V、Table I | VLMが画像＋RGB-Dから得る物体pose記述＋指示を読み、taskを分解し、**DMP wとgoal offset等を直接生成**する。実演なし・VLM fine-tuningなし。形辞書の自己教師発見・検索ではない | **31実機task×5試行**。Table IはLMP **65%**、TrajGen35%、π0.5 28%。ただし65%は**人間judgeの最大3回の自然言語訂正と、失敗前状態へのreset**を含む。judge-free初回は**54%**。decomposer-free41%、両方なし24%。旧版・検索抄録の「20task80%」をv3の結果に混ぜない |
| **CLASP**, Knauer et al.：[2606.08169v1](https://arxiv.org/html/2606.08169v1)（2026-06-06）、§3.2–3.5、Tables 1–2 | **TP-KMPのskillをVLMで選択・parameter binding**。2–5のkinesthetic実演から名前・preconditionを持つschemaを作る。能力不足なら実演を要求し、MP＋schemaを追加。**VLMのfine-tuningなしで新skillを利用**する。確率的な軌跡融合も使うため本案の単一候補条件と異なる | Table 1：object generalization **40/44**、pose generalization（vision）**23/29**、manual pose29/29、composition16/16、新規ring insertion **11/15**。Table 2のπ0.5との比較は4実演/skill対5・50実演等で、同一訓練量・同一観測の比較ではない。名前付きskill・個別実演を使うので、無注釈分節化の先例とはしない |
| **PrimitiveVLA**, Li et al.：[2605.28634v2](https://arxiv.org/html/2605.28634v2)（2026-07-18）、§3–5、Tables 1・3 | **固定11種**（Grasp、Place、Pull等）の意味的primitiveを定義し、VLM/LLMで実演を分解、canonical instructionと物体maskでVLAを学習。推論はplannerとLLM生成switch code。**人手作業なしでも、意味ラベル／既定語彙なしではない** | Table 3：π0.5のLIBERO-Long **30.50→80.25%**、90-Novel56.00→75.50%、90は96.80→96.20%。Longは90からの0-shot generalizationとして報告。simulation各task50試行、実機・RLBench各20試行。抽象／言い換えを統制したpaired評価ではなく、**未見task合成への効果** |

LMPは「VLMが原始パラメータを通して抽象的目標を実行する」という構成の直接の先行例である。しかし、DMP表現、強い上位VLM、物体pose処理、分解、人の訂正をすべて合わせた結果であり、**DMPだけが抽象指示の理解を改善したという因果分離はない**。本文の限界でも抽象的な相対指示のgroundingに失敗すると述べる。

DROMの言語評価も参考になる。Table 9の600指示でprimitive認識accuracy86.2%、Table 8では誤ったskill列が42/250と報告され、**言語側の選択誤りと実行誤りを分けている**。ただし、既定skillに対するannotated instruction分類を、本案のラベルなし運動型への意味理解と同一視しない。

Hi Robotの同じ合成データを使ったflat比較、SkillDiffuserのskill条件diffusionと言い換え評価は既存調査を参照。したがって**階層化で抽象指示に強くなるという一般論は新規でない**。一方、「同じprefix・同じ学習予算で、無注釈DMP辞書を使う条件だけがL3に強く、primitiveを入れ替えると対応する型が変わる」という比較は、今回確認した資料では未確認である。

## 7. 固定重みでの追加：何が同じで何が違うか

| 操作 | 本案／既存研究での位置づけ | 言ってよいこと |
| --- | --- | --- |
| 実演から新しいDMP wを回帰して保存 | 古典DMP、Niekum、VMP library、CLASPに先例 | 「ニューラルモデルを再学習せず原始パラメータを作る」。**追加そのものは新規でない** |
| VLMを固定して新MP＋schemaを読ませる | CLASPが新skill追加と実機実行を示す | 「foundation model固定の原始追加」も広くは既存。意味schema・人の実演と、無注釈形辞書との差が必要 |
| backbone固定で新embedding・router出力行を学習 | **現S4がこれ**。少量の新しいパラメータは最適化する | 「backbone固定の増分適応／局所パラメータ追加」。**全パラメータ固定、追加時勾配なしとは呼ばない** |
| selector・shape encoder・flow・embeddingをすべて固定し、未知の形を追加して使う | 現在のID embedding／出力行方式では自動的には実現しない。内容を読むselector等が必要 | 無注釈形成・個数未固定・単一候補・共有生成まで限定すると、**今回の確認範囲では差の候補**。性能と被覆の証明はまだない |

補足：[Harness VLA, arXiv:2607.08448v5](https://arxiv.org/html/2607.08448v5)（2026-09-24）は凍結VLA＋analytic primitive＋成功・失敗のmemoryによる階層化であるが、本文は**primitive vocabularyを固定し、追加しない**と明記する。通常経路との使い分けや名前付きanalytic primitiveが中心で、全行動を無注釈DMPから生む本案と同じではない。凍結VLAにメモリを付けるだけで新規性とするのは避ける。

## 8. 現設計を評価するときに限定すべき解釈

これらは**事前基準を変更する提案ではない**。既定S1–S3の結果を誇張しないための注意と、Train／Devで可能な補助解析である。

- **S1の再構成は表現の上限**：Dev区間の真のg・τを与えて1cm以内でも、それを画像から選び・予測できる証拠ではない。`w=0`に近い直進型＋自由なg・τだけで説明できる程度を併記すると、辞書の必要性を判別しやすい。S1(c)は開閉だけへのcollapseを防ぐ有用な基準だが、意味的なskill discoveryの十分条件ではない。
- **位相の終端とtaskの完了は別**：τだけで進むphaseは、把持・接触が失敗しても終端へ行く。現在のs予測が時間の時計か、観測に基づく進行かを分けて見る。offlineの未来全体を使った分節は可能だが、S2/S3の選択入力に未来の正解g・境界・長さを使わない。
- **flowの開始点がDMPでも最終出力はDMPとは限らない**：自由なaction-space bridgeには、古典DMPの収束・滑らかさの保証は引き継がれない。DMP形をどこまで変えたか、wrong primitiveでどの運動が変わるかを確認する。
- **単一候補と確率性は両立する**：形は1本でも、その周囲の生成ノイズは使える。同じ観測に対して完全に同じ開始点を決定論的ODEへ入れるだけなら多峰的な条件分布にはならない。開始点のノイズ、訓練時sourceと推論時sourceの一致を仕様に残す。候補の平均は不要。
- **データの型・単位を合わせる**：DMPの絶対pose6次元と、SkillNetの相対action＋gripperは別物。位置mと回転radの誤差・正規化、SO(3)での向きの差、初期速度の引継ぎを記録する。これは新しい評価分割や閾値を足す要求ではなく、S1誤差とS3入力を解釈するために必要な仕様である。
- **複数注入経路の効果を分ける**：MoE、adaRMS、bridgeを一度に変えたS3の成功は構成全体の効果。既定のshuffle低下は「primitive情報を使う」証拠であり、各FFN expertとDMP型の対応までは同定しない。追加の経路ablationはDevのみ、Diagの再利用なしで行う。
- **L3の点差と一般化の射程**：既定の9task×10初期状態・L0/L3・通過閾値はそのまま使い、同じ初期状態でのpaired差と不確実性も併記する。L3の一部が複数の行動で満たせる要求なら、特定のLIBERO成功述語との一致と「指示理解一般」は別。promptsや事前基準を結果に合わせて直さない。

## 9. 設計書の構成別：既存と同じ部分／差になり得る部分

| 設計書の構成 | 既存と同じ部分・近い先行研究 | 差になり得る部分／成立条件 |
| --- | --- | --- |
| S1：形w・g・τ・phaseへの分解 | 古典DMP、NDP、ProDMP。空間・時間の分離、1実演fit | **この分解単独は新規でない**。形の場面非依存性・再利用・選択可能性を実測する必要 |
| 再構成誤差／gripper変化で自動分節 | DSDNetの分節DMP、ProbS、Niekum等の無注釈分節 | **単なる自動分節では弱い**。意味注釈を一切使わず、可変長で、下流の選択・生成まで有用という組合せを示す |
| DP-means＋Train MDL、個数未固定 | DP-means、Bayesian nonparametric skill discovery、MP libraryの圧縮 | 既定語彙・固定KのPrimitiveVLA／SCFM／ECSとの差。ただし推定KやMDLの採用だけで十分ではない |
| S2：観測・指示→prim, g, τ, s | NDP／LMPのparameter generation、階層skill選択、進捗条件付け | 自己教師の形を**内容・適用条件で選択し、長さや幾何と分離**できるか。正解パラメータを与えるoracleとは分ける |
| 1本選び、混ぜない | PRISEの単一macro選択、通常のoption実行 | 混合しないこと自体は新規でない。曖昧な候補の平均より段階誤りを減らす実測が必要 |
| S3：primをMoE router／adaRMSへ | SkillNet、AtomicVLA、ECS、skill条件拡散 | **手作り意味ラベルのskillでなく物理的な形に対応**する点。dictionary primitiveとFFN expertの違いを保ち、shuffleの因果効果を示す |
| DMPを展開してflow bridgeのsourceへ | 情報を持つsourceのBRIDGeR、MP＋生成のMPD／FODMP | **場面のタイミングを持ち込まず、形だけを持ち込むsource**が効果を持つか。g・τ回帰や単なる平滑化の効果を除く |
| L3に強く、L0を維持 | LMP／Hi Robot／DROMの階層言語実行、PrimitiveVLAの未見合成 | 同じデータ・prefixで**無注釈運動型が必要条件になる比較**は確認範囲で空白。S3のG0／shuffleがここを検証する |
| S4：新形追加、embedding／出力行だけ学習 | MP library拡張、局所的な増分適応 | 現状は**backbone固定の少数パラメータ適応**。厳密な全重み固定の追加を主張するなら、追加行を学習しない設計が必要 |

## 10. 差を強める設計変更の提案（最大3つ）

**以下は将来の派生条件であり、今回の分割・事前基準・S1–S3の実行順を変更しない。** 比較のため、現設計を先にそのまま評価する。

1. **S4を内容によるpointerへ拡張する。** 新規IDの出力行／embeddingを追加学習する代わりに、初期訓練で学んで固定したshape encoder `E(w)`とquery `q(prefix, history)`で可変個数の候補をscoreし、hard top-1で選ぶ。生成器にも`E(w)`を渡す。追加はfitしたwと適用条件の事例の保存だけにする。**全重み固定の追加を実現できれば最も主張が強まる**。ただし新しい形が生成器のsupport外なら実行できず、保証はない。CLASPの名前付きschemaによる追加とは、無注釈内容と共有生成器で区別する。
2. **形を消せる自由なbridgeと、形を保つ補正を分ける。** 派生条件として、選択DMPを主軌道とし、始終点を変えない位相依存の補正basis内でflowを学ぶ構成を比較する。g・τ・scene adaptationを許しつつ、選択形を全面的に描き直せる経路を制約する。MPDの構造化decoderの知見を使えるが、**shapeの選択が主役だと分かる介入**を重視する。補正を縛りすぎると接触・修正に弱くなるので、現S3を置き換える前提にはしない。
3. **新規性の比較軸を「shape依存」と「言語階層の利得」に分ける。** 既定G0／shuffleに加える将来の補助条件として、`g,τ,s`は同じでwを直進型へ置換する条件と、同じprefix／同じ上位解釈を使ってwだけを替える条件を用意する。全体SRに加え、g・τを変えたときの型の保持、wrong shape時の段階誤り、flowの補正量を測る。これにより**DMP一般の平滑化・goal回帰・強い言語plannerだけでは説明できない差**を示せる。追加のmodel selectionにDiagは使わず、既定の通過判定とは分けて報告する。

一次資料・表抽出・版履歴は [research/sources](/work/.skillnet-state/outputs/memory-pilot/dmp-route-20261102/research/sources/) に保存した。HTML未提供のMovement Primitive DiffusionはarXiv PDF本文を使用。コード・設計書・既存調査文書・評価プロセスは変更していない。
