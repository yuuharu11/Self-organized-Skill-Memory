# 運動メモリに何を保存するか：生の行動列・潜在コード・原始動作

確認日：2026-11-04 UTC。承認済み `plan_memory_gap.md` と[段階2の結果](../07_pi05_memory40/memory_gap/results.md)を先に読み、下記の版指定の一次資料の本文・付録を確認した。数値はローカルの実測と論文著者の報告を分ける。方式の採否は決めず、保存形式を選ぶための選択肢と必要な証拠を整理する。事前基準・分割・実験計画は変更しない。

## 1. 今回の観測が示す範囲

段階2は、通常π0.5が難しいとしたEval **141行・69episode**でのオフライン診断。t0=.2、同じnoise・10積分で、最終行動MSEは次のとおり。値は正規化したactive7の10ステップ平均で、閉ループ成功率ではない。

| 条件 | 最終MSE | この比較で分かること |
|---|---:|---|
| A：通常方策 | .05125 | 基準 |
| B：ゼロ下書き | .03453 | 検索した動作がなくても得られる効果を含む対照 |
| C：現行VLM検索 | .06702 | このnoise量ではBより悪い |
| D：memory内最良候補の12次元base＋予測residualによる復元 | .01691 | 正しく選べると仮定した場合の診断 |
| D-raw：同じ候補の生の行動列 | .00461 | 現在の圧縮・復元経路に大きな損失がある |
| D-raw：同じtaskのmemoryを除外 | .00514 | 別taskのmemoryにも行動距離上のよい候補がある |
| E：正解行動を圧縮したbase＋予測residual | .01342 | 圧縮・復元を通る参照値であり、到達可能な誤差の絶対下限ではない |

出典：[段階2の結果、固定設定・主表・追加診断](../07_pi05_memory40/memory_gap/results.md)。D-raw−Dは **−.012305、episode bootstrap 95% CI [−.015876, −.008922]**。ただしD/D-rawは**評価にも使う正解行動MSEで候補を選んだ診断用の上限**で、実行時に選べる性能ではない。同task除外はmemory側の除外であり、基盤π0.5がそのtaskを未学習であるという意味ではない。

事前判定は **G(D)≥.5により段階4-B「検索が障害」**。これを維持する。同時に、今回の12次元baseと予測residualによる復元経路には損失がある。ここから「どの圧縮表現も生行動より不利」「生行動に替えれば成功率が上がる」と一般化はできない。圧縮器・分解・residual予測・生成器との相互作用を含む結果である。

## 2. 区別すべき三つの保存

| 対象 | 例 | 「追加」の意味 |
|---|---|---|
| **経験の値** | 生のaction chunk、圧縮コード列、DMPのw・goal・duration | 新しい経験のkey/valueを追記する |
| **共有された型・辞書** | k-means中心、VQ codebook、可変個数の原始動作、typed program | 新しい経験を既存の型へ登録することと、新しい型そのものを増やすことは別 |
| **検索キー・補助context** | 画像埋め込み、VLM特徴、自己状態、履歴、事前計算したcontext | 小さなaction valueでも、key/contextが全容量を支配し得る |

BeT・VQ-BeT・QueST・LAPAは主として**学習した行動表現・出力空間**の研究であり、そのcodebookをそのまま「追記可能な経験memory」と呼ばない。VQ内部の最近傍量子化は、現在画像をkeyに過去経験を検索する操作とも異なる。

「ラベルなし」は、①手動のsubtask/skill注釈なし、②task IDなし、③言語・成功ラベルもなし、を分ける。本研究の芯に照らすと、task IDを使わずに運動の型を形成したかと、新しい型を**全学習パラメータ固定**で追加できたかを別々に確認する必要がある。

## 3. 指定された手法：保存する値・検索キー・追加の評価

| 手法・版／一次資料 | 保存・学習するものと使い方 | 検索キー、または検索しない場合の選択入力 | 未学習taskへの実演追加／固定されるもの |
|---|---|---|---|
| **VINN**、Pariほか、RSS 2022。**2112.01511v2、2021-12-06**。[本文III・付録A–B](https://arxiv.org/html/2112.01511v2)、[会議](https://roboticsproceedings.org/rss18/p010.html) | 各実演frameの画像特徴と**その時点の生のaction**。原著の主設定は複数stepのskill chunkではない。近傍actionをsoftmin距離で加重し、直接制御。 | ImageNet初期化後、実演画像でBYOL学習したencoderの現在画像特徴。**L2距離**のkNN。本文式はexp(−L2)、付録擬似コードも同じ。 | encoder固定後に新しいkey/actionを入れる操作は非パラメトリック。ただし原著の評価は既習の扉と遮蔽・初期位置変化が中心。**未知taskの実演を固定encoderへ追加する前後比較は未確認**。扉のgripper抽出には別の4クラス推定器があり、全系を運動注釈不要とはしない。 |
| **BeT**、Shafiullahほか、NeurIPS 2022。**2206.11251v2、2022-10-11**。[§2・付録B–C](https://arxiv.org/html/2206.11251v2) | 生actionをk-meansで分けた**action中心＋連続offset**。中心を固定して、Transformerがbin確率とoffsetを予測。過去経験のraw chunkを検索するmemoryではない。 | 検索keyは**該当なし**。観測履歴→bin/offsetの学習済み予測。訓練時のbin割当は正解actionから最近傍中心を求める。 | 固定bin分類head。新taskデモを追加するだけで新しいbinや行動型を使う評価は**未確認**。中心＋正解offsetなら数学的には復元可能だが、実行時offsetは方策が予測する。 |
| **VQ-BeT**、Leeほか、ICML 2024。**2403.03181v2、2024-06-28**。[§3・付録B.1/C](https://arxiv.org/html/2403.03181v2)、[会議](https://proceedings.mlr.press/v235/lee24y.html) | **actionまたは固定長chunkを2段のresidual VQ**へ。primary/secondary codeの埋め込み和をdecoderへ渡し、観測依存offsetを加える。latentは多くの環境で512次元だが、保存indexの情報量とは別。 | 経験検索keyは**該当なし**。観測履歴・goalからcodeを予測。量子化時はaction encoder latentとcodebookのL2最近傍。 | tokenizer学習後にcode予測方策を学習。固定codebookへの新経験の符号化はできるが、**全重み固定で未知task実演を追記し、新しい型を獲得する評価は未確認**。大codebook実験は再学習した別モデル。 |
| **QueST**、Meteほか、NeurIPS 2024。**2407.15840v3、2024-09-11**。[§4–5・付録B–D](https://arxiv.org/html/2407.15840v3)、[会議](https://papers.nips.cc/paper_files/paper/2024/hash/076c3e48fa502c660902105965fdd9f6-Abstract-Conference.html) | **action-onlyの32step窓→8個のFSQ token**。因果conv/attention encoderと、code列を読むdecoder。単独codeが意味あるskillとは限らず、本文は**code列がmotionを表す**と説明。 | 経験検索keyは**該当なし**。画像・自己状態・task埋め込み→skill priorがcode列をAR予測。stage Iはactionだけで表現学習。 | LIBERO-90→保留Longの5実演転移は**評価あり**。ただし**skill priorをfine-tune**し、decoder更新あり／固定の両方を比較。decoder固定を「全重み固定でmemory追加」と呼ばない。 |
| **LAPA**、Yeほか、ICLR 2025。**2410.11758v2、2025-05-15**。[§3・5・付録A/F](https://arxiv.org/html/2410.11758v2) | 現在・未来の**画像対の変化**をVQでlatent action列にする。標準は**8語彙×4token＝4096通り**。画像decoderは現在画像とlatentから未来画像を復元。**生の制御action chunkの圧縮器ではない**。 | 経験検索keyは**該当なし**。量子化時は画像対のlatentをL2でcodeへ。事前学習VLMは現在画像・指示→latentを予測。 | 実ロボット行動化では**latent headを捨て、実action headへ交換し、言語モデルをfine-tune**（vision encoder固定）。未知物体・指示等への汎化は評価するが、**固定モデルへ経験を追記して適応する評価は未確認**。推論時に必ずlatent skillを介するモデルと解釈しない。 |
| **REFACTOR-VLA**、Shaik・Venkataraman。**2609.01215v1、2026-09-01**、採択先未確認。[§3–5・付録G–I](https://arxiv.org/html/2609.01215v1) | 提案上は世界モデルでfragmentの振る舞いを比較し、**typed λ programのlibrary**をMDL等で拡張。typed emitterとlibrary-conditioned rectified-flow decoderをwake/sleepで学習。**実証されたlibraryは言語側のtemplateで、運動側は0件**（後述）。DMP数値パラメータの記憶ではない。 | 実行時に過去のraw経験をcosineで検索する方式とは異なる。世界モデルのfragment特徴をSiamese encoderへ蒸留、cosine由来のBEKでクラスタ化。主実装は**Kをsuiteのtask数に合わせた固定KMeans**。typed emitterがprogramを予測。 | wakeでpolicy/decoder、sleepでBEK headを更新。**全重み固定で新task実演だけを追加する評価は未確認**。主結果の世界モデル補助InfoNCEはtask_indexを使用。label-free代替も報告するが、本文・付録間で評価範囲に不整合がある。 |
| **TraceFlow**、Zhangほか。**2609.20646v2、2026-09-28**、採択先未確認。[III–IV](https://arxiv.org/html/2609.20646v2) | **時系列キー＋実行した生action列＋episode/task metadata＋成功/失敗bit**のtrace。anchor以後のHstep窓から成功/失敗のkernel densityを作り、固定flowの速度場へ吸引/反発を加える。**開始点交換ではなくvelocity guidance**。 | VLM状態→オフラインcontrastive学習した3層headの単位vector、cosine。π0.5では現在1時点または[-20,−10,0]の3時点、PrediMemでは上位/下位特徴等。forward trackerと局所照合で進捗整列。tokenは検索keyであり、action値は生のまま。 | **trace追記・round更新は全重み固定で評価済み**。ただしtask-specific setup後の同じtaskの自己成功/失敗経験。target-taskの訓練traceを初期bankに使用する設定が中心。**未学習taskへの外部実演追加の評価は未確認**。成功bitも不要にした運動型発見ではない。 |
| **RA-VLA**、Jangほか。**2608.25585v2、2026-10-07**、arXiv commentsはICML 2026。[§4–5・付録B–F](https://arxiv.org/html/2608.25585v2) | 固定窓segmentごとに **事前計算VLM特徴H′・自己状態s′・生action chunk A′**を保存。検索contextをaction headのcross-attentionへ、さらに**検索actionの平均をflow初期値**に使う。h=16、stride4、4denoise。 | VLMのLLMに入る前の画像・text token→**2層Transformer→mean pool、512次元、cosine**。同task内の実演対のaction列をDTWで合わせた正例により学習。検索時は観測・言語だけを使う。 | **保留taskの実演を追加し、追加時の勾配更新なしで評価済み**。LIBEROは3suiteでICILを学習し1suite保留、保留taskごとに3実演。実機はleave-one-task-out、4実演。事前にretrieverとgrounded action headを学習。運動型codebookの形成・新型追加はしない。 |

**比較の要点：RA-VLAは本研究の「重み固定で、今回の導入学習に含めていないtaskの実演を追加する」主張に直接近い。** 「全重み固定の実演追加」だけを新規性にはできない。一方、QueST/LAPAの転移は重み更新を含むため、この軸では別の比較になる。

## 4. 圧縮と性能について実際に報告されたこと

| 研究・出典 | 確認できた数値・比較 | 圧縮の判断に使える範囲 |
|---|---|---|
| **VINN Tables I–III** | 扉開け実機30試行：VINN **80%**、同じ表現上のBC **53.3%**。遮蔽別評価もある。 | 良い検索表現＋生actionが機能する先例。**raw action対圧縮valueの比較ではない**。オフラインMSEと実機成績は一致しないことも議論。 |
| **BeT Table 3、付録C.2/Fig.6** | offsetなしのKitchen性能は、通常BeTで正規化して **.78**、binなし **.68**。k増加はbin分類を難しくし、offsetのMSEは減らすと説明。 | **中心だけでは細部が足りず、補正が必要**という証拠。Table 3は絶対成功率ではない。kの容量対性能はFig.6であり、bit数と成功率のPareto比較は未確認。 |
| **VQ-BeT Table 12、付録B.1** | Kitchen条件付き、codebook各層 **16→256**（組合せ **256→65536**）で達成goal数 **3.78→3.61 /4**。Kitchen非条件付きは各層 **16→64**で **3.66→3.75**、p4-entropy **4.07→3.57**。PushTも16→64でfinal coverage **.78→.77**、deadcode maskingあり **.79**。 | **大辞書なら制御が単調に良くなるわけではない**。表現力・code選択・mode多様性のtrade-off。decoder/offsetも学習したモデル全体の比較であり、生chunkと同じ検索keyでvalueだけ置き換えた比較ではない。 |
| **VQ-BeT §4.6、Table 13、Fig.5** | 標準は2段RVQ、各codebook8–16程度、latentは多くの設定で512次元。offsetの除去、1段VQ、chunking等を比較。 | offsetとresidual VQの重要性を報告。**512次元latentを毎経験保存する場合と、indexだけ保存する場合を混ぜない**。1step制御が主の設定もあり、本研究の10step圧縮との単純比較は不可。 |
| **QueST Tables 1・3–4・7、Fig.4** | 32step→8token、FSQ levels **[8,5,5,5]＝1000通り**。LIBERO-90 **88.6±.4%**、少数実演転移 **68.8±1.7%**、decoder固定 **66.0±3.6%**。VQ置換の90は **81.2±.6%**。Fig.4は時間間引きF・語彙数の感度。 | 小さな1codeではなく**code列の表現容量**で制御する先例。大きすぎる間引きは少数実演性能に影響。本文には「1024 vectors」ともあるが、付録の積・vocab sizeは1000で不整合。68.8の主比較は§5.3の**8task**で、付録Table 8の10task全体と同一視しない。 |
| **LAPA §5.1/Fig.5、付録F/Fig.16** | token長・語彙の増加と下流性能を比較。標準は8⁴。SIMPLERでは表Table 11でLAPA(Bridge) **57.3%**、action教師ありActionVLA **63.5%**。 | latent容量が下流に効く例。ただし容量の曲線は図であり、正確な各点の表値は未確認。**画像変化を圧縮した事前学習の効果**で、固定生成器の下書きとしてraw actionに勝つ証拠ではない。§6は把持等の微細動作の弱さを認める。 |
| **REFACTOR-VLA §4.7、付録G.3** | 言語slotの **3抽象・1211 nats**、256/256実演の書換えに2抽象を使用。運動slotはMDL閾値を変えても **0抽象**、DTW事前整列でも改善せず。 | **1211 natsは言語programのMDL gain**であり、action valueの圧縮率でも、同じ制御性能を維持したmemory bytes削減でもない。主要NMIはtask-indexに対するcluster評価で、成功率ではない。生actionより良い運動圧縮を示した根拠には使わない。 |
| **TraceFlow Tables I・III–IV** | LIBERO-Plus Long **79.83→81.10%**、対応2519試行、p=.0733。26task aggregate TSR **34.92→34.99%**。実機ordered packingは **21/50→39/50→追記後47/50**、順序誤り **20/50→2/50→0/50**（Table II）。 | 追記型raw traceと順序を扱う先例。シミュレーションの全体的な改善は確認されていない。**action圧縮率のablationは未確認**。D5/D10はkeyの間引きと出力次元も変わるので、pureなaction圧縮比較ではない。 |
| **RA-VLA Tables 1・3–4、付録B** | 保留suite平均は原表 **.3845＝38.45%**、同じaction-aware retrieverのRICL_R **.2085＝20.85%**。Goalのbuffer実演数1/2/3/4で **48.2/51.8/53.2/55.2%**。1segment当たりkey **1KB**、cached VLM特徴 **2MB**、action chunk **1KB**と著者記載。 | 追加実演数・contextへの依存は評価。**raw対latent valueの圧縮比較は未確認**。記憶の主要部分はactionではなくcached特徴になり得る。Table 1等のcaptionは%だがセルは0–1形式なので、割合として換算したことを明記する。 |

数値の一次資料：[VINN](https://arxiv.org/html/2112.01511v2#S4)、[BeT](https://arxiv.org/html/2206.11251v2#S3)、[VQ-BeT](https://arxiv.org/html/2403.03181v2#A2.SS1)、[QueST](https://arxiv.org/html/2407.15840v3#A3)、[LAPA](https://arxiv.org/html/2410.11758v2#A7.SS2)、[REFACTOR-VLA](https://arxiv.org/html/2609.01215v1#S4.SS7)、[TraceFlow](https://arxiv.org/html/2609.20646v2#S4)、[RA-VLA](https://arxiv.org/html/2608.25585v2#S5)。QueST Table 1の±は4seedの標準誤差と明記され、付録のfew-shot protocolには9seedの別表もある。異なる表の±・task数をまとめて同一の評価条件にしない。

**今回確認した指定論文には、同一の実演・検索key・候補・生成器を固定し、生chunk／複数圧縮率のcode／原始動作パラメータをvalueだけ交換して、memory全体のbytes対閉ループ成功率を比較した一揃いの実験は見当たらなかった。** この比較がないことと、各潜在表現が無効であることは別である。

## 5. 生chunkの検索・再生・初期化に近い追加の比較対象

| 研究・確認版 | 保存value・key・使い方 | 追加時固定／未学習taskの評価 | 容量・性能の報告と限界 |
|---|---|---|---|
| **REGENT**、Sridharほか、ICLR 2025。**2412.04759v2、2025-02-24**。[§3–4・付録A](https://arxiv.org/html/2412.04759v2) | 生の **(state, previous reward, action)** tuple。state vectorまたはImageNet ResNet18画像特徴のL2、ProcGenはSSIM。検索contextをTransformerへ渡し、最近傍actionとの距離依存補間。R&P baselineはtop-1 actionを直接再生。 | **あり**。訓練から保留した環境の少数実演を参照し、policyのfine-tuneなしで適応。ロボット側のMetaWorldはproprioceptive stateであり、LIBEROのVLAと同じ入力ではない。 | model/data規模の比較はあるが、raw actionと圧縮valueのrate–performance比較は未確認。**新task実演の無勾配追加そのものは既存**とするための比較。 |
| **RT-Cache**、Kwonほか、Humanoids 2025とarXiv記載。**2505.09040v3、2025-08-25**、現題 *Training-Free Retrieval for Real-Time Manipulation*。[III–V](https://arxiv.org/html/2505.09040v3) | 時刻ID付きの**生7D action列・画像**をDBへ。凍結DINOv2 1024D＋SigLIP 1152Dをconcat/L2正規化、cosine検索。近傍時刻からNstep再生、top-1または平均。周波数10Hzに統一。 | **実演追加あり、勾配なし**。新object/camera/pose組合せへ最大27短episodeを追加。単一実演を追加したpushingは定性的に報告。厳格な保留task列＋旧task保持という評価は未確認。 | Table Iはdisk約100GB、そのうちembedding約95GB、metadata約3.1GB。**action圧縮は将来課題**。zero-shot到達0%、few-shotで視点別96/93/52%は§IV-C/Fig.5の値であり、表値ではない。主taskの成功は「手先が把持可能領域へ到達」で、把持・完了までの成功と取り違えない。 |
| **RAGDP**、Odonchimedほか。**2507.21452v1、2025-07-29**、採択先未確認。[III–IV](https://arxiv.org/html/2507.21452v1) | 正規化した**生action chunk**をvalue。状態履歴の正規化vector、またはDP encoder特徴をkeyにL2 top-1。検索actionへ加ノイズし、中間diffusionから仕上げる。主評価はstate入力。 | 追加学習なしで既存DPへ適用。DBへの追記は構造上可能だが、**保留task実演追加・旧task保持は未確認**。 | Tables III–IVはsampling速度対task completion：EDMの×4設定で平均completion **80.54%**、元性能に対するrecovery **103.36%**。これはmemory圧縮のtrade-offではない。LIBEROの結果は未確認。 |
| **OptimusVLA**、Liほか、CVPR 2026。**2602.20200v2、2026-05-18**。[§3・付録B–C](https://arxiv.org/html/2602.20200v2) | **全軌道**とtask embeddingを保存。現在VLM prefixのmean pool→学習済みhead→cosine。進捗ρtでwindowを切り、検索chunk群の平均・分散を持つ単一Gaussianからflow開始。 | head・LCMを導入学習。推論は固定。**新task外部実演の追記評価は未確認**。ρtの具体的な推定法も未確認。 | LongはGPM/LCMなし **92.4%**、GPMのみ **94.8%**、両方 **96.4%**（Table 4）。bank数とkを比較（Table 5）するが、action圧縮は比較しない。詳細は[直前の精読](retrieval_init_methods_check.md)。 |
| **RTS**、Zhaoほか。**2605.10094v2、2026-05-12**。[§4・6・付録A–I](https://arxiv.org/html/2605.10094v2) | **成功prefixの画像key＋生chunk**。visual patch特徴を2×2平均pool・flatten・多視点concat、cosine＋候補間DTW。elite actionへnoiseを加え、confidence依存t0から固定flowを実行。 | **成功経験の追記は無勾配で評価済み**。同taskへfine-tune済み方策の反復配備。未学習taskの外部実演追加は未確認。成功判定にはtaskごとの参照動画1本とVLAC critic。 | Table 4：通常 **92.4%**、top-1 **93.6%**、full **94.4%**、直接再生 **87.8%**。Table 6は容量0/3k/5k entryで **61.0/68.5/70.8%**。生value対codeの圧縮比較はない。[精読](retrieval_init_methods_check.md)参照。 |
| **Reflex**、Linほか。**2611.00524v1、2026-10-31**、採択先未確認。[§3・付録D/G/M](https://arxiv.org/html/2611.00524v1) | **生action chunk**とcompact state key。policy vision特徴をpool→cameraごとにPCA64D＋自己状態。actionが近い別episode対からMahalanobis metricを推定、chunkを局所加重＋小さなresidual補正。停滞時・周期的に元policyへ戻る。 | policyは固定。ただし新sceneの3episode追加時に**ridge map・検索metricを再推定**（CPU、閉形式の推定を含む）。全学習パラメータ固定とは異なる。評価は未見sceneで、未学習taskの運動型追加とは別。 | Table 7：LIBERO-Plusの2suiteで通常π0.5 **79.5%**、未適応cache **62.6%**、適応 **74.9%**、停滞check込み **77.1%**。actionを圧縮せず**keyを圧縮**。actionの圧縮率比較は未確認。元policyへ戻る設計は本研究の芯と異なるが、検索key・value比較の対照にはなる。 |

RT-Cacheの題名は古いv1の *Efficient Robot Trajectory Retrieval System* とv3で異なるため、上は確認したv3に合わせた。Reflexは2026-10-31公開の新しい近接例であり、検索が改善したとの主張を出す場合に見落とせない。ただし元方策へのfallbackを含む全体成績を、スキルが常に生成を担う方式の成績と同じ条件では扱わない。

## 6. REFACTOR-VLA・TraceFlowから読み違えてはいけないこと

**REFACTOR-VLA：** 題名はunsupervised motor-program libraryだが、主結果のPhase Aはtask_indexによる補助教師を使い、クラスタ数もtask数に合わせる。label-free代替について、§5はquarter-episode正例でLongのNMI **.761**、no-InfoNCE **.736**を上回ると説明する一方、付録G.2はLongのtemporal-window samplerをcamera-ready課題とする。運動slotのMDL sweepの閾値集合も§4.7とG.3で一致しない。ただし**運動抽象0件という結果は一致**する。したがって「運動だけから型が生まれ、その辞書を追加すると制御が良くなる」と実証済みの論文として引用しない。return-preservation gateもLIBEROではreturn headの重み0で不活性と§3.1に明記される。[§3.1・4.7・5、付録G.2–G.3](https://arxiv.org/html/2609.01215v1#S3.SS1)

**TraceFlow：** 成功・失敗の生traceを全重み固定で追記し、順序誤りを減らす先例である。一方、初期bankは対象taskの訓練traceであり、学習に含めていないtaskへ外部実演を追加する評価ではない。simのheadlineはsuiteごとに選んだ設定を含み、stacking各系列は単一historyで、ピークroundは事後の観測。全26taskの改善を確認できず、LIBERO-Plusのp=.0733を「同等」や「改善確定」としない。確認したv2には独立した付録はなく、本文のTables I–IV・limitationsが評価条件と限界を明記している。[IV–V](https://arxiv.org/html/2609.20646v2#S4)

## 7. 二つの選択肢で残る主張と必要な比較

ここからは**本研究への解釈・比較案**。方式の採否を決めるものではない。承認済み段階2–5の条件・閾値を変更する提案でもない。

| 論点 | 生の行動列の記憶へ寄せる場合 | 圧縮した運動の型を保つ場合 |
|---|---|---|
| そのままでは新規性にできない部分 | raw action検索・直接再生はVINN/RT-Cache、検索actionから生成開始はRAGDP/OV/RTS/RA-VLA、固定重みの経験追記はREGENT/RA-VLA/TraceFlow/RTSと重なる。 | action中心＋補正はBeT、無注釈action codeはVQ-BeT/QueST、動画latentはLAPA、構造library学習はREFACTOR-VLA等と重なる。低次元化だけでは新規性にならない。 |
| **差になり得る主張** | 生valueでも、**無注釈で繰返し再利用できる運動の型を形成し、その型を単一選択して、追加時は全重み固定で新taskを改善する**なら組合せの差候補。型を形成せず全経験を蓄積するだけなら、芯は「創発型」から「経験検索・適応」へ狭まる。 | **可変の新型を追記できるcontent表現**と、固定selector/decoderでの利用、必要な細部を保ったままraw memoryより良い容量・転用・選択性能を示すことが差候補。既存codeへの符号化だけを新型の追加と呼ばない。 |
| 型と値の関係 | 型の会員経験として**生の代表chunk・複数実例を保存**する形もある。ラベルなしの型とraw fidelityは両立し得る。ただしクラスタIDを付けるだけでは型の有用性は未証明。 | 型の中心に加え、形パラメータ・目標・時間・gripper・小さな残差を保持する選択肢。必要な残差がrawと同程度なら、実効圧縮率は小さい。DMP/ProMPの形・goal・duration分離は既存であり、[DMP調査](../04_segment_skills_dmp/related_dmp_routing.md)を参照。 |
| 直接近い対照 | **RA-VLA型の保留task実演context＋平均action初期化**、同じ生成器のtop-1 raw＋noise（RAGDP/RTS型）、直接再生、型を使わない全経験kNN。原著とは別に、同じbackboneへ移した比較だと明記。 | 同じkey・候補IDを用いた**raw、現b12、容量を変えた連続codec、RVQ/FSQ token列、shapeパラメータ＋残差**。QueST/VQ-BeTは表現学習の比較で、タスク追加時固定の比較と区別。 |
| 型が主役である証拠 | 型shuffle・値shuffle・型を介さない同容量のflat memoryとの比較。型で候補を束ねることで誤段階・検索後誤差・成功率が変わるか。型形成にtask/skill注釈を使わない。 | 圧縮codeを替えると生成運動が意図どおり変わるか。decoderが観測だけでactionを決めてcodeを無視していないか。**code置換による出力差が大きいだけでは正しさの証拠ではない**ので、成功・型内調整も測る。 |
| 追加可能性の証拠 | 90のみで全導入学習・key・thresholdを固定し、Longの実演をmemoryへ追加。追加task実演を使う条件をゼロショットと呼ばない。無関係実演を同数追加し、旧task保持・追加順序も測る。 | 左と同じ。encoder/decoderだけでなく、**新ID embedding・classifier出力行・codebook更新・normalization更新**も止める。新しいcodeを追加するたびにheadを学ぶなら「全重み固定」とは言わない。 |
| 費用とリスク | valueの細部は残るが、検索負荷・容量・phaseのaliasing・absolute座標への依存が残る。wrong raw priorは強い誘導で悪化し得る。 | 容量・冗長性を抑える余地がある一方、gripper/timing/接触の細部を失う、decoder support外の新型を使えない、選択が細分化し過ぎる危険がある。 |

「rawかlatentか」を一つの比較だけで決めないため、次を分けると解釈が明確になる。

1. **候補・検索を固定した復元比較**：同じ候補IDのrawと各codecを、同じkey・noise・生成器で比べる。正解自身のencode/decodeで、圧縮そのものとquery側residual予測を分離する。rawで選んだoracle候補に加え、圧縮後の距離で選ぶoracleも補助的に出すと、**raw距離で選んだ候補が有利になる選択バイアス**を把握できる。oracleはいずれも診断専用。
2. **実行可能な選択の比較**：実行時情報だけで検索し、Recall@kに加えて、**選んだ下書きがBより良い割合・難しい行の生成誤差・段階誤り**を測る。学習するkeyに追加情報を使うなら、自己状態・履歴の有無を列別に示す。未来actionをqueryに入れない。
3. **容量を揃える比較と経験を揃える比較**：同じ経験数では圧縮損失、同じ総bytesでは被覆の違いを見る。両者を併記する。元policyへの追加学習の効果はB、値の意味は無関係・shuffleで分ける。t0/NFE変更は別要因として揃える。
4. **最終的な主張の確認**：段階3の失敗状態からの分岐と、条件が満たされた後の段階5で、シミュレーションでの閉ループ成功・救済/悪化・旧task保持を測る。オフラインMSEのoracle上限をこの結果の代替にはしない。

## 8. 容量の数え方と未確認事項

総容量は概念的に **key＋action value＋context/metadata＋index＋共有辞書/decoder**。共有モデルを既に保持している場合も、その有無を方式間で揃える。rawの10×7 float32は**280 bytes**、padding込み10×32なら**1280 bytes**、12 float32は**48 bytes**という計算になるが、これは**payloadだけの算術**であり実測保存量ではない。r・goal/時間・gripper等を別保存すれば加算し、code indexと展開後latentのサイズも分ける。RA-VLAやRT-Cacheの報告は、key/context側が大きい場合にactionを圧縮しても総容量の削減が限られることを示す。

以下は今回**未確認**：指定された潜在手法で、全パラメータ固定のまま新しい運動型を可変個数追記した結果／同じVLA・同じ検索と学習予算でraw・codec・DMP valueのbytes対成功率を網羅した結果／今回のb12損失が潜在次元だけで決まるという根拠。これらがないことから、どちらの保存方式を採用すべきかまでは決めない。

原資料のHTML・表JSON・版履歴は、新規出力 `.skillnet-state/outputs/memory-pilot/research-what-to-store-20261104T092616Z/` に保存した。OptimusVLA/RTSは[直前の本文確認](retrieval_init_methods_check.md)とその保存資料も参照した。日付なしのファイル名は[文書規則](../README.md)、旧gatesパスの移動先は[gates.md](../01_memory_gates/gates.md)に従った。この作業の編集範囲は本ファイルとsurveys/READMEの案内1行のみ。コード編集・学習・rollout・GPU利用・commitは実施していない。
