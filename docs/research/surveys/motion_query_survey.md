調査: 完了

# スキルメモリのクエリ：状況から必要な運動を読む

## 1. 結論と調査の範囲

**最終層の平均特徴だけで、凍結VLAに含まれる運動情報の有無は判断できない。** π0.5の中間層・層ごとのaction expertへの接続に、最終層の平均とは異なる情報がある。一方、学習可能なquery tokenを足すだけで、凍結π0.5から未知タスクの型とθを取り出せるという直接の実証は、今回確認した本文にはない。近い成功例では、行動を教師にした表現学習、幾何、履歴、または基盤モデルの更新を使っている。[因果介入分析 §2–4](https://arxiv.org/html/2610.20750v1)、[Action QFormer Tables II–III](https://arxiv.org/html/2607.14635v1)

本研究への有力な組合せは、**因果的な履歴＋平均前の凍結特徴のreadout＋運動内容との学習済み対応付け**である。型を選ぶqueryと、選んだ型を場面へ配置するθのreadoutを分ける。これは以下の文献から導く**提案**であり、本設定での改善は未確認。

依頼に記載された実測は、正解による型選択でMSE .035、θも正解なら .003、通常π0.5 .364、現行特徴によるθ回帰はギャップ回収−4%。いずれも**依頼の診断値**であり、今回再集計した結果でも閉ループ成功率でもない。旧LongのTop-1 .105対oracle .029も、評価条件を揃えず今回と差し引かない。運動単位の経緯は[09 の実験一覧](../09_unseen_libero/experiments.md#2-メモリに未学習の動きはあるか正解で選ぶ上限)を参照する。

本文・付録を確認した版をURLの`vN`で指定する。以下の「数値なし／未確認」は、該当設定の数値を確認できなかったという意味で、手法が存在しないという意味ではない。文献のSR、OOD検出AUROC、特徴のcosine、正解参照の軌道転送率は相互に代替できない。また、probeや追加moduleの学習から保留したtaskが、基盤VLA自体に未学習とは限らない。今回の未学習LIBEROへの転移と同一視しない。

## 2. RL Tokenは何を読んでいるか

Physical Intelligenceの *RL Token: Bootstrapping Online RL with Vision-Language-Action Models*、**arXiv:2604.23073v2**。[§IV-A、Fig. 2、付録B](https://arxiv.org/html/2604.23073v2)

| 確認項目 | 本文で確認できた仕様 |
|---|---|
| 読み出す場所 | §IV-AはVLAの**最終層の入力トークンに対応する埋め込み** (z_{1:M})。Fig. 2はSigLIP＋Gemma側のimage embeddings (N\times2048) をencoderへ渡す。**action expertのaction-query隠れ状態を読む構成ではない**。一般式は言語も含められるが、実験では指示がタスク内で固定なので言語埋め込みを省く、と脚注1に明記。 |
| ボトルネック | 埋め込み列に学習可能な`<rl>`を末尾追加し、小さなencoder Transformerのその位置の出力を使う。Fig. 2は**1本の2048次元**。readout Transformerの深さ・head数は未確認。 |
| 再構成対象 | decoder＋線形投影が、**元の最終層埋め込み列**を自己回帰で復元する特徴MSE。画像pixel、未来行動、θの再構成ではない。decoderには正解の先行埋め込みをteacher forcingし、対象埋め込みはstop-gradient。 |
| 凍結の順序 | readout損失はVLAへ逆伝播しない。別のVLA教師あり微調整は併用可能。**実験はタスク実演でVLAとreadoutを2k–10k step学習してから、両者を固定してonline RL**。最初から最後まで公開VLAを固定した実験とは異なる。 |
| 行動との接続 | RL actor/criticはRL tokenと自己状態を使い、actorは別途VLAの参照action chunkも受ける。TD3、参照行動への正則化等で学習。readoutの再構成だけで行動を学ぶわけではない。 |
| 評価 | 実機4タスク。LIBEROは未確認。ねじ挿入の20%→65%、最大約3倍の高速化は**要旨・実験本文／図の報告**で、比較表の値ではない。平均poolingとの同条件比較は未確認。 |

**移植案と限界：** 凍結prefixの未集約tokenに外付けencoder＋readoutを載せる部分は使える。ただし、本論文は「特徴再構成で欲しい運動が読める」ことも「学習済みattention poolingより優れる」ことも示していない。言語を省く設定も本研究には移さない。再構成は補助対照とし、型の運動内容・復元行動・θに対する教師を与えたreadoutと比べる。自己回帰decoderが先行tokenから復元できるだけになっていないか、readoutをshuffleした復元も確認する、というのが本研究への提案である。原論文のphase切替やRL制御への引継ぎは、今回の「メモリが主役・ゲートなし」の設計には持ち込まない。

## 3. VLA内部のどこに次の運動の情報があるか

| 一次資料・確認場所 | どこを読む／何で学習・分析するか | 確認できた結果 | 本設定への移し方と難点 |
|---|---|---|---|
| **Causal Analysis of Robot Foundation Models Enables Efficiency Gains**、**2610.20750v1**、§2・4、Fig. 2、Table 2。[本文](https://arxiv.org/html/2610.20750v1) | 固定方策で反実仮想場面間のactivationを差し替える。VLM内部のresidualと、**層ごとのVLM→action expertのconditioning interface**を区別。早い画像・名詞表現、後の幾何的な位置表現を分析。 | π0.5で**14–16層**が有効な帯域。単層で35%超、3層で70%超の軌道転送はFig. 2の正規化指標。Table 2は実機7タスク合計**各方策100試行**、全18層64/100対14–16層67/100。task time 34.9→26.5秒は成功試行のtask別中央値の平均で、推論時間ではない。 | 14–16層を候補に、複数層のtoken／KVを同容量headでprobeする。**最終層平均と各層KVは別物**。他checkpointの層番号を最適値として固定しない。パッチで情報を移せることは、queryから線形に予測できる証拠ではない。π0.5では画像tokenも指示の影響を受けるため「画像＝外観だけ」としない。 |
| **Decoding Task Progress from VLA Representations**、**2608.13474v1**、§4.2・付録B.3、Table 1。[本文](https://arxiv.org/html/2608.13474v1) | 凍結π0.5のSigLIP・Gemma・action expertの層別特徴を集約し、残り時間 (1-t/T) を線形L1回帰。VLA更新なし。 | 指示に依存する進捗の層別分析はGemma **L10付近**が強く、L17で低下。図のR²は概ね.33–.39の帯域。Table 1の保留task・episode集約OOD AUROC **.851±.056**は検出性能で、運動選択のSRではない。 | 履歴付きqueryへ進捗の補助表現を持たせる先例。訓練実演の時間を教師にできるが、残り時間は速度・軌道長のproxyであり、同じ進捗から正しい型やθが一意に決まるとは限らない。実行時には未来のTを入力しない。 |
| **VLA-Trace**、**2605.30117v3**、§3.2、Tables 2・6。[本文](https://arxiv.org/html/2605.30117v3) | 固定π0.5/OpenVLAで表現類似性とattention経路を分析。prefix内の視覚–言語相互作用と、生成側から画像／textへ読む経路を別々に遮断。 | π0.5 Longは93.5%→**text参照なし39.0%**、image参照なし0%、各suite200episode。Table 6はBOS＋newline除去57.5%、instruction除去86.0%。Goalのtext除去は96.0→96.5%。 | 指示語だけでなく特殊tokenを保存して読む対照が必要。全text平均は機能の違うtokenを潰し得る。ただしattention遮断は分布も変えるため、数値から最適poolingや単一の意味tokenを断定しない。 |
| **Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models**、**2603.19183v2**、付録A.3、Tables 2・7。[本文](https://arxiv.org/html/2603.19183v2) | 固定π0.5/OpenVLAのblock出力residualをSAEで再構成。主結果は**全tokenを時点ごとに平均**、PGは0/5/11/17層等。SAEを学習し、feature介入で因果性を検査。 | Table 2のDROID実機は無介入39/40、最もgeneralとされた4featureを除去0/40、random memorized除去37/40。Table 7のgeneral割合はLIBERO2.62%、DROID10.81%で、著者の分類基準による。 | 凍結特徴の中に制御へ因果的に効く要素がある証拠。ただし「general」が将来のDCT型・θと一致するとは示していない。SAE再構成を増やす前に直接probeを置く方が軽い。 |

**action expertの隠れ状態について：** そこには観測だけでなく、入力したnoisy action、flow時刻、denoising履歴が入る。従って「prefixより運動に近い」ことと「未知運動を正しく選べる」ことは別である。今回確認した資料では、凍結π0.5の特定expert層・特定flow時刻が、未学習LIBEROタスクの型／θ検索に最良という比較は**未確認**。試すなら正解actionで汚した状態ではなく、固定noiseからの通常推論で得た状態・速度予測を読む。

ローカル実装でも、[pi0.pyのprefix生成](../../../skill_moe/skillnet/src/openpi/models/pi0.py#L422)、[suffix生成](../../../skill_moe/skillnet/src/openpi/models/pi0.py#L471)、[生成](../../../skill_moe/skillnet/src/openpi/models/pi0.py#L651)は別経路で、expert出力の最後のH tokenをaction投影へ渡す。[tokenizer.py](../../../skill_moe/skillnet/src/openpi/models/tokenizer.py#L24)ではπ0.5のstateは離散化したtextへ含まれる。従って現在の「言語平均」が純粋な指示だけかstate・`Action:`等も含むかをmask単位で確認すべきで、**prefixにstateがないと決めつけない**。これは通常π0.5側のコード確認であり、実際に使うcheckpoint・呼出経路のmask確認は別途必要。

## 4. 学習可能なquery／readoutを追加する先例

ここでのattentionは情報を読む機構であり、複数の運動valueを混ぜて実行することとは分けられる。

| 手法・一次資料 | queryと読む場所／学習信号・凍結 | 数値と評価範囲 | 移植案・難点 |
|---|---|---|---|
| **BLIP-2**、**2301.12597v3**、§3。[本文](https://arxiv.org/html/2301.12597v3) | Q-Formerの**32学習query**が凍結image encoderをcross-attend。画像–text対照・matching・生成、その後凍結LLMへの接続を学ぶ。Q-Former等を更新し、基盤を固定。 | VLA／LIBEROの数値は該当なし。 | 本研究では言語＋stateでqueryを条件付け、平均前の視覚／VLM tokenを読む。原著のimage–text目的をそのまま使っても運動やθの識別は保証されない。32queryという数自体に運動上の根拠はない。 |
| **Octo**、**2405.12213v2**、§III-A、Tables I–II。[本文](https://arxiv.org/html/2405.12213v2) | 各時間blockのreadout tokenが現在までの観測・task tokenを読む。**観測／task側はreadoutを参照しない**。readoutをaction diffusion headへ。行動lossで全体を学ぶ。 | Table IIのWidowX40試行でdiffusion83%、MSE35%、離散18%。**readout有無のablationではない**。Table Iの適応平均72%は微調整込み。 | 凍結prefixの外側に受動的readoutを置く設計根拠。π0.5のprefixへ単にtokenを追加すると元tokenやKVが変化し得るので、外付けcross-attentionか明示的な片方向maskが必要。 |
| **RoboFlamingo**、**2311.01378v3**、§4.2–4.4、Table 1。[本文](https://arxiv.org/html/2311.01378v3) | learned Perceiver resampler→言語側cross-attention→token max pool→LSTM＋行動head。vision・LLM本体を固定し、resampler・接続・headをBCの姿勢MSE＋grip BCEで学習。 | CALVIN ABC→D平均連鎖長**2.48/5**、ABCD→D **4.09/5**。best checkpointの値。LIBEROは未確認。 | 「基盤固定でも学習readout＋履歴で行動化」は実例あり。ただし画像encoder直後から接続し直す構成で、既存π0.5の最終平均へpoolだけ足す実験ではない。原著のgated接続は採用せず、履歴・readout部分だけ参考にする。 |
| **Action QFormer**、**2607.14635v1**、§III・IV-C、Tables II–III。[本文](https://arxiv.org/html/2607.14635v1) | learned queryとinstruction表現を自己attentionし、**既定はLM融合前のvision encoder特徴**をcross-attend。行動教師でqueryと上流の表現を形成。post-LM特徴を読む条件も比較。 | Table IIIの左右行動cosineはdirect/full更新 **.891**、QFormer/full **.470**、QFormer/action更新遮断 **.898**。Table IIのchair誘導なし成功25→62.5%、指示提供25→87.5%、各8試行。**移動navigationであり腕操作ではない**。 | instruction-conditioned queryが「意味」から「行動の違い」を読む先例。ただし上流を更新した結果が中心で、**凍結＋QFormerだけの効果を保証しない**。本研究では外付けheadだけの成否を先に測る。 |
| **VLA-R**、**2511.12405v1**、§3、Tables 1–2。[本文](https://arxiv.org/html/2511.12405v1) | 凍結YOLOEの視覚・text整合・box分布特徴をOW-QFormerで集約。別Transformerが**運動軌道そのもの**をencodeし、対照学習でqueryと整合、cosine最大候補を検索。 | Table 1は屋外車輪移動の障害通過event成功、rough terrain .96（117event）、dense trees .93（70event）。LIBEROなし。試行数が手法間で異なる。 | **観測queryと運動内容keyの二塔**は追加可能な型辞書に直接参考になる。原著の辞書は運動学で作るx/y/yaw軌道で、無注釈操作スキルではない。型keyを内容から計算すれば固定ID分類器より追記に向く。 |
| **REKI: Registering Knowledge in VLAs**、**2610.22178v1**、§4.1、Tables 2・4。[本文](https://arxiv.org/html/2610.22178v1) | **action expert側に32 register/frame**。現在のVLM最終特徴と過去registerを読み、**action tokenを読まない**。そのKVを3frame・1秒間隔の窓で保持。flow lossでexpert・backbone・vision末尾も更新。 | Table 4はLIBERO97.5%、LIBERO-Plus90.4%。**全モデルの結果で、registerだけの寄与ではない**。Table 2が更新範囲を明記。 | noisy actionに依存しない観測履歴tokenを作る設計が有用。ただしπ0.5へtokenを足すだけで原著を再現できず、凍結移植の性能は未確認。経験型メモリではなくepisode内の文脈記憶。 |

Perceiverを3D制御へ使う **PerAct（2209.05451v2）** も、言語＋RGB-D voxel＋自己状態をlatent cross-attentionで読み、位置voxel／回転／gripを教師にする先例である。凍結VLMのreadoutではなく方策を学習する。今回の型／θ予測と同じ条件の数値は未確認。[§3](https://arxiv.org/html/2209.05451v2)

## 5. 検索型方策のキー：行動に対応する距離をどう作るか

**重要な区別：** 記録済みのtarget実演の未来まで読んで訓練データを選ぶ検索と、実行中の現在・過去だけから次の運動を選ぶ検索は異なる。前者の検索器をそのままonline queryにすると、未来画像・行動の入力が欠ける。

| 手法・版／本文 | query・key、距離、学習信号 | 凍結・評価値 | 本研究への具体案／難点 |
|---|---|---|---|
| **STRAP**、**2412.15182v2**、§4、Table 1。[本文](https://arxiv.org/html/2412.15182v2) | target実演を停止点から分節し、DINOv2等のframe特徴のL2と**subsequence DTW**でprior実演のサブ軌道を検索。凍結視覚特徴。 | 選んだデータでBCを学習。LIBERO-10平均 **45.6%**（DINOv2、K=100）対BC37.9%、best-K58.1%はK選択条件が異なる。 | **サブ軌道の時系列整列**を運動key教師の作成に利用可能。online観測→型の学習ではなく、targetデモを必要とする。未来まで見た検索成績を実行可能な成績にしない。 |
| **COLLAGE**、**2508.01131v2**、§3、Table 1。[本文](https://arxiv.org/html/2508.01131v2) | DINO視覚、GMflow由来motion、PointNetの3D object shape、言語の複数cueでサブ軌道検索。cue別に仮方策を学び、target保留行動の尤度でデータ群の融合重みを学ぶ。 | 最終方策を再学習。LIBERO-10 **50.5%対STRAP45.6%**、target5実演、3seed×50試行。 | 学習行動で「どの見方の検索が役立つか」を選ぶ発想は有用。ただし**online queryごとのrerankerではない**。shapeは物体形状で、本研究の回転・scale不変な運動形状とは異なる。 |
| **Behavior Retrieval**、**2304.08742v2**、§III、Table I。[本文](https://arxiv.org/html/2304.08742v2) | **状態と行動を合わせたβ-VAE latent**でtargetの少数実演とpriorをL2照合。再構成＋KLが表現教師。 | retrieval後にBC＋target微調整。simulation平均 **72±2%**対pretrain＋FT36±2%、3seed・100試行の著者報告。原著LIBERO値は未確認。 | 行動を含むkeyが外観だけより意味を持つ先例。ただしquery側にも実演actionが必要。運動側encoderをteacherにし、現在観測側へ蒸留する一段が本研究では必要。 |
| **FlowRetrieval**、**2408.16944v2**、§III、付録B Table 1。[本文](https://arxiv.org/html/2408.16944v2) | 記録画像の**現在→先のframeのGMflow**をVAEで圧縮し、target実演とのL2でデータ検索。行動学習へ予測flowの補助lossも追加。 | codec・方策を学習。原表LIBERO-Can **90%**対BC64%、flow補助のみ71%。LIBERO全suite平均ではない。 | 観測を未来の運動へ結び付ける教師としてflowが有望。実行時には**過去→現在flowか未来flow予測器**が必要で、原著のqueryをそのまま使えない。camera motionと物体運動の混同、2D→3Dの不定性が残る。 |
| **RA-VLA**、**2608.25585v2**、§4・5、Tables 1・3、付録B。[本文](https://arxiv.org/html/2608.25585v2) | **LLMに入る前の画像＋text token**→2層Transformer→平均、512D cosine。**同taskの別実演をaction-DTWで対応付けた正例**で対照学習。自己状態はkey入力ではなく検索context側。保存はVLM特徴・自己状態・生action。生成側にも関連contextを使わせるadherence loss。 | VLM固定、retriever・生成headを事前学習。3suite訓練→保留suiteへ3実演追加、追加時更新なし。Table 1平均 **.3845=38.45%**対同retrieverのRICL_R20.85%。GoalはTable 3でadherenceなし9.8%対あり53.2%。原表のセルを割合から%へ換算。 | **最も直接的なaction-aware queryの先例**。本研究ではtask IDによる正例限定を外し、別taskでも近い正準運動を正例にする。逆に同じ型でもθは違うので、θをkey距離へ混ぜ過ぎない。平均action初期化は使わず1型を選ぶ。原著の改善はretrieverだけの効果ではない。 |
| **ReCAP**、*Retrieve, Don't Retrain*、**2606.15631v1**、§4.2、付録D–E、Table 1。[本文](https://arxiv.org/html/2606.15631v1) | 最初に**指示埋め込み＋初期物体位置＋初期自己状態**のL2でtrajectoryを絞る。次に現在の物体pose・自己状態の二乗L2とDINOv3のcosineの加重距離でsubframe検索。**訓練時だけ未来action距離も使い、推論では除く**。SAM3由来の物体特徴。 | retrieval規則は幾何中心で、action対照encoderではない。Cosmos系の生成器を学んだ後固定し、未知taskのpoolデモを追記。RoboTwin保留5task **31.5%**対retrieval再生26.0%、元方策4.0%。 | θ向けに**場面幾何＋手先状態**をqueryへ加える根拠。訓練／推論で検索条件が変わる点は本研究では避け、教師だけに未来actionを使う。物体検出・座標取得、指示に合う対象の選択が別問題。層別probeのL10/L15はCosmos系でありπ0.5へ直用しない。 |
| **MemoryVLA**、**2508.19236v2**、§3.2–3.3、Tables 6–7、付録C.3。[本文](https://arxiv.org/html/2508.19236v2) | **PrismaticのLM前視覚token256本＋LLaMAのEOS最終特徴1本**を別streamにする。現在queryが時間位置符号付きの過去特徴をcross-attend。 | diffusion行動方策を学習、凍結π0.5への後付けではない。Bridgeでcognitiveのみ63.5%、perceptualのみ64.6%、両方71.9%。時間符号なし69.8%対あり71.9%。 | 空間詳細と意味要約を分け、**履歴付きquery**を作る先例。経験間の運動型検索ではなくepisode内記憶。原著のgateは採用せず、同じ履歴特徴を直接readoutへ渡す対照にする。 |
| **SAILOR**、*Learning and Retrieval from Prior Data for Skill-based Imitation Learning*、**2210.11435v2**、§4、Tables 2・7。[本文](https://arxiv.org/html/2210.11435v2) | 状態・行動10stepをVAEで64D skillへ。再構成＋KLに、latent間の時間差を予測する**temporal predictability**を追加。target実演latentとのL2でpriorデータ選択。実行policyは**10frame履歴からskillを予測**。 | skill・policyの学習／target適応あり。CALVIN Setting Up **77.3±3.1%**、時間予測なし68.0%、retrievalなし65.0%。retrieval自体は主に学習データの選別。 | 前後関係を持つ運動codeと、履歴→欲しいskillの教師を分ける点が近い。固定DCT型へ移す場合は時間予測を補助lossにできるが、同じ時間差でも進み方が違い、長さの暗記にもなり得る。 |

**名称の注意：** 上のReCAPは検索型の2606.15631。Physical Intelligenceの **RECAP／π*0.6（2511.14759v2）** は経験からのRL・advantage conditioningであり、同じ検索器ではない。[π*0.6本文](https://arxiv.org/html/2511.14759v2)

追加で近いのは **TraceFlow（2609.20646v2、§III–IV）** の、VLM特徴を行動対応のcontrastive headへ通すkeyと、3時点履歴・進行に沿った照合である。生の成功／失敗action traceをvelocity guidanceへ使う。実機ordered packingの順序誤り **20/50→2/50→追記後0/50** はTable II。一方、成功ラベルを使い、無注釈の型形成ではなく、今回のtop-1型選択の成績でもない。[本文](https://arxiv.org/html/2609.20646v2)、[既存の精読](what_to_store.md#6-refactor-vlatraceflowから読み違えてはいけないこと)

## 6. 状況から潜在行動・スキル符号を予測する

| 手法・版／本文 | 「状況→欲しい運動」の学習 | 凍結・結果 | 型メモリ＋別θへの示唆 |
|---|---|---|---|
| **LAPA**、**2410.11758v2**、§3・5、付録A/F。[本文](https://arxiv.org/html/2410.11758v2) | 現在・未来画像対をVQにし、現在画像＋codeから未来画像を復元。次にVLMが現在画像・指示からlatent actionを予測。制御actionなしの動画も教師になる。 | vision固定、言語モデルを更新。実actionへの適応では**latent headを交換してfine-tune**。Table 11 SIMPLER Bridge57.3%対action教師あり63.5%。 | 観測→運動内容の予測は直接学習すべき、という先例。ただし画像変化codeは外観・cameraも拾い、DCTの運動型と同一ではない。全重み固定の新型追加を示していない。 |
| **CLAP**、**2601.04061v2**、§III-B–E・IV-C、Tables IV・VI。[本文](https://arxiv.org/html/2601.04061v2) | action VQ-VAEとvideo変化VAEを**対照学習で整合**。VLMは観測・指示→subtask text＋action codeをNTP学習。RF headはVLMのKVを読む。 | token policyも更新し、固定referenceへのKLで保持。LIBERO平均97.2%、Long93.0%。Table VIはhigh-level特徴93.4%、low-level96.1%、**low＋mid＋KM97.2%**。最後は複数変更を含む。 | 行動内容をteacherにすることと、多層readoutの参考。**正確な抽出層番号は未確認**。subtask tokenを教師にする経路は本研究に移さない。凍結VLMで同じ効果が出る証拠ではない。 |
| **ConLA**、**2602.00557v1**、§3、Tables 1・3。[本文](https://arxiv.org/html/2602.00557v1) | 動画変化をmotion／visualに分け、未来画像再構成、**action-classのsupervised contrastive**、visual対照でmotionを分離。VLMが現在画像・指示からcodeを予測し、action適応。 | VLM側を学習。human-video条件の実機平均52.1%（LAPA）→64.6%、action featureのみ58.4%。 | 外観に依らない運動queryの学習に近いが、**action-classラベルを使う構成は本研究の無注釈条件と違う**。正準化した実運動距離から正例を作る置換は提案で、原著の結果ではない。 |
| **QueST**、**2407.15840v3**、§4–5、Tables 1・7。[本文](https://arxiv.org/html/2407.15840v3) | action-onlyで32step→8FSQ tokenのcodecを学び、画像・自己状態・task表現から**skill priorがcode列をAR予測**。 | LIBERO-90 88.6±.4%、保留task5実演適応68.8±1.7%、decoder固定66.0±3.6%。**priorは微調整**。後2値は主比較8taskで、全Long10task平均とは異なる。 | 観測→型の写像を行動codecの教師で学ぶ直接例。単一codeが単一skillとは限らず、code列が運動を表す。固定語彙分類器をそのまま使うと、未知の型の無勾配追記ができない。 |
| **BeT／VQ-BeT**、**2206.11251v2／2403.03181v2**、§2／§3・Table 12。[BeT](https://arxiv.org/html/2206.11251v2)、[VQ-BeT](https://arxiv.org/html/2403.03181v2) | 観測履歴からaction bin／RVQ codeと連続offsetを教師あり予測。再構成で運動codeを作り、code分類・offset行動lossでpolicyを学ぶ。検索経験は持たない。 | VQ-BeT Kitchen条件付きはcodebook各層16→256でgoal数3.78→3.61/4。容量増が選択性能を上げるとは限らない。今回と同じLIBERO未知型追加の数値は未確認。 | 「粗い運動の選択＋型内調整」は既存。型だけの距離で整合し、θは候補条件付きheadにする設計が自然。ただしoffsetが全行動を再生成すると型が主役でなくなるので、shuffle・型置換の介入が必要。 |

VQやlatent actionの容量・追加可能性の詳細は[保存表現の既存調査](what_to_store.md)を参照。**固定codebook内のcodeを予測できることと、新しい型を追記して固定queryで選べることは異なる。** 本研究では`classify(K IDs)`ではなく、`score(query, encode(shape_k))`の形なら、Kが増えても出力層を増設せず使える。ただし追加した型が学習済みmetricの分布外に出る問題は残る。

## 7. 物体・幾何・affordanceからθを読む

型は外観・向き・大きさを捨てる一方、θには対象までの相対方向・距離が必要である。**型用keyを幾何に不変にすることと、θ用readoutが幾何を保持することは両立させる必要がある**。以下はθの直接回帰を保証する文献ではなく、幾何を行動へ接続する先例。

| 手法・版／出典 | query／幾何表現、学習・凍結 | 確認値 | 移植案と難点 |
|---|---|---|---|
| **ATM: Any-point Trajectory Modeling**、**2401.00025v3**、§IV、付録Table VI。[本文](https://arxiv.org/html/2401.00025v3) | 現在RGB・指示・query点から未来の2D点軌跡を予測。CoTracker由来のtrackを教師に事前学習。**track transformerを固定し**、予測track＋現在画像からaction policyを学ぶ。実行時は32点grid。 | DP→ATM-DPはLIBERO-90 **33.85→62.89%**、Long **37.33→44.00%**。suiteで方策を学習した結果で、凍結π0.5の転移ではない。 | 未来の欲しい動きをqueryの中間教師にする有力例。型readout／θ headに予測trackを渡せるが、別のtrack事前学習が必要。2D方向・pixel距離をそのまま3D θに変換できない。 |
| **RoboTAP**、**2308.15975v2**、§III。[本文](https://arxiv.org/html/2308.15975v2) | 固定TAPIRでdemo間の点を追跡。grip・force閾値でphaseを切り、共通に動く点と目標位置を抽出、visual servoで順に実行。学習したVLM queryではない。 | few-shot実機。今回と比較できるLIBERO値は未確認。 | **段階ごとの「何を、どこへ動かすか」**を分ける参考。原著はデモ間で同数phaseを仮定し、誤差閾値で段階移行。本研究の自由な型検索を解決済みとはしない。 |
| **3D Diffuser Actor**、**2402.10885v3**、§3、Table 3。[本文](https://arxiv.org/html/2402.10885v3) | RGB-Dの3D視覚特徴、言語、手先pose／履歴、noisy軌道tokenを**3D相対位置attention**で結合。位置・回転のdenoisingを行動教師で学ぶ。 | RLBench平均は2D **47.0%**、3Dで相対attentionなし71.3%、full81.3%。 | gripper基準の幾何特徴→θ readoutの根拠。深度・外部calibrationが追加で要る。3D相対attentionは主に平行移動への構造で、回転・scale不変な型の保証ではない。基盤固定で小headだけの結果は未確認。 |
| **RAM**、*Retrieval-Based Affordance Transfer*、**2407.04689v1**、§3、Tables 1–2。[本文](https://arxiv.org/html/2407.04689v1) | object中心画像＋言語task＋contact点・以後の2D waypointを保存。意味／3D形状による検索、dense correspondenceでcontact・運動方向を移し、depthで3D化。既存特徴による転送が中心。 | sim52.62%対Robo-ABC41.54%、実機20/30=66.7%対15/30=50.0%。LIBEROではない。 | **型を選ぶことと、接触点・方向を新物体へ配置することを分離**する直接の参考。task category、物体抽出、動画注釈・手動custom waypointを含み、本研究の無注釈形成とは異なる。 |
| **HAMSTER／3D HAMSTER**、**2502.05485v4／2606.31329v2**、手法節。[HAMSTER](https://arxiv.org/html/2502.05485v4)、[3D版](https://arxiv.org/html/2606.31329v2) | 上位VLMを学習し、画像・指示から粗い手先path／gripを予測して下位policyへ。3D版はdepth encoder・3D waypointでmetric planningを接続。現在の状況→欲しい軌道を明示化する。 | 本設定に対応する凍結π0.5・θのみreadoutの数値は未確認。 | θを一度に回帰する代わりに**接触点・終点・方向を予測してθへ写す**案。上位を訓練する原著と、凍結特徴の小headは別。pathを自由に生成させるとメモリ型を迂回するため、予測幾何は選択した型の配置に限定する。 |

**本研究への幾何案：** 現在の手先poseと、画像から得た対象候補の相対位置・方向、過去の手先変位、grip状態をreadoutへ渡す。候補物体は指示でattentionして選ぶが、物体名／subtask正解を教師にしない。教師は訓練実演の運動とする。シミュレータの真の物体poseを使う条件は**特権情報の診断**として分離し、公式RGB入力の性能と合算しない。実depthが使えない場合、2視点対応・既知cameraと手先の投影点で補助できる可能性はあるが、本設定で十分な3D精度が得られるかは未確認。

## 8. 本設定に落とす場合のクエリの形と切り分け

ここからは研究への提案。既存の分割・事前判定を変更したり、実験を実施したりするものではない。

**query側**を (q_t=R(F^{\ell}_{t-h:t},s_{t-h:t},a^{exec}_{<t},\text{instruction}))、**型側key**を (k_j=E(\text{shape}_j,\text{grip}_j,\text{duration}_j)) とする。まず共通のreadoutで情報源だけを比較する。その後に内容依存score (S(q_t,k_j)) で**1型**を選び、

\[
j^*=\arg\max_j S(q_t,k_j),\qquad
\hat\theta_t=P(q_t,k_{j^*},s_t),\qquad
\hat a=\mathrm{Place}(\mathrm{shape}_{j^*},\hat\theta_t)
\]

とする。必要な生成器はこの型の中を場面へ合わせる。context tokenのattentionは使っても、複数の運動valueを和にしない。duration／gripの入力は現行メモリに存在する項目だけを使い、保存していなければ新しい入力を使った比較として分ける。

訓練時には別episodeの運動同士を正準空間で照合し、近い複数候補を正例とする。RA-VLAのaction-DTWを参考にできるが、**task IDやsubtask注釈では正例を作らない**。型keyは向き・scaleに不変、θ headはそれらに感度を持つようにする。θは全候補共通の1本ではなく**選択型の内容を条件に**予測する。同じ場面でも型の正準軸の定義でθが変わるからである。対照lossだけでなく、配置後の運動誤差でも学習できる。ただし学習時だけ完全なoracle型を与えると実行時とのずれがあるので、訓練側でも実際のselectorが返す候補に対してθを学ぶ対照が必要。

| 比較・指標 | 何を区別するか |
|---|---|
| 同じ小headで、現行3平均／token保持／層別／expert特徴を比較。入力ごとに訓練側で正規化、head容量と履歴を揃える。 | 読む場所の問題と、attentionの学習容量の問題を分ける。平均済み3本しか保存していなければ、後から空間tokenは復元できず、凍結forwardの再抽出が必要。 |
| stateのみ、履歴のみ、現行特徴＋state、＋履歴、＋新readout。 | 「新しいattentionが効いた」のか「追加した過去・幾何が効いた」のかを分ける。すでに試したpoolingの単なる大型化を避ける。 |
| oracle型＋予測θ／予測型＋oracleθ／両方予測／両方oracle。 | 型の選択と配置の律速を分ける。oracleは診断専用。現在の .035/.003の差を、実行可能な性能として報告しない。 |
| θの回転角・log-scale誤差に加え、配置後の方向cosine、ADE/FDE、grip誤り、段階の停滞／飛越。 | 軸角を普通のMSEで平均する問題、短い動作／停止が全行動MSEに有利な問題を避ける。θ_src、平均・ゼロ運動、通常方策も対照に残す。 |
| 単一oracle-IDのaccuracyに加え、複数妥当候補に対するRecall@k、選択後の運動誤差、型置換／shuffle介入。 | 同じ運動を表す冗長候補のID違いを誤答扱いしない。メモリへの感度が大きいだけで正しいとはしない。 |
| 訓練taskだけでmetric・readout・正規化・閾値を固定し、未学習taskで追加前後を測る。既存のTrain/Dev/Evalを維持する。 | 固定ID分類ではない内容keyが、新型を**全重み固定で追加した後**にも働くかを確かめる。新型ごとのembedding学習・出力層追加・正規化再推定を無勾配追記と呼ばない。 |

**θの同定可能性にも注意する。** ほぼ直線の型では、その軸周りの回転が変わっても実現運動が変わらない場合がある。正準化の符号・軸の規則やゼロに近いscaleにも曖昧さがある。従って「oracle θと軸角が違う」ことだけで失敗とせず、**復元した運動の距離**を主に評価する。これは型＋姿勢の分解から生じる本研究側の論点で、前記論文がこのDCT分解について検証した結果ではない。

expert特徴の比較では、例えばnoise固定で通常推論の初回・中盤・終盤のaction-token特徴／予測速度をcacheする。flow時刻はこの実装の**1=noise、0=action**とともに記録する。正解actionで作った中間状態をqueryへ入れない。また、同一時点でメモリ選択を先に必要とする生成経路から特徴を読むと循環になるため、読むのはメモリ選択前の固定された基盤経路に限る。これは特徴を読むための計算であり、実行actionを通常経路へ逃がすgateではない。

履歴は実行時に得られる過去画像・自己状態・**実際に実行したaction**だけに限定する。オフライン実演の直前actionを使ったprobeは教師軌道上の結果なので、閉ループのpolicy履歴にもそのまま通用するとはしない。

## 9. 軽く試せる順の推奨

| 順 | 案 | 必要な計算／失敗したときに分かること |
|---|---|---|
| **1** | **現行特徴＋因果的なstate・履歴で、型を条件にしたθ head**。直前の手先変位・姿勢・gripを明示し、履歴なし／θ_srcと比較する。 | 既存cacheと小MLP／GRUで済む範囲から開始、**LoRA不要**。失敗しても「履歴は無用」と一般化せず、少なくとも現行特徴＋小headでは配置のgapを回収できないと分かる。 |
| **2** | **凍結prefixの層・token別readout診断**。中間層、14–16層、最終層、特殊／state tokenを同じheadで比較し、平均化の影響を調べる。 | **凍結forwardの再cache＋小head、LoRA不要**。失敗は選んだ層・集約でのreadabilityが低いことを示すだけで、VLA全体に運動情報がない証拠にはならない。 |
| **3** | **action-awareな内容keyと、少数queryの外付けreadout**。型のDCT形状をencodeし、履歴・指示で条件付けたqueryを運動対応の教師で学ぶ。θは別の候補条件付きhead。 | 平均前の凍結token cacheから**readout／二塔だけ学習、LoRA不要**。RLT型の特徴再構成のみを対照に置く。失敗すれば、単なるpoolingから運動教師へ変えても保留taskへの対応付けが不足する。 |
| **4** | **凍結action expertの状態・予測速度をqueryにする**。固定noiseの通常推論で取り、prefix案と同じselector／θ headへ渡す。 | 新たにexpert forwardが必要、通常推論を全て読むならそのNFE分。cache後の学習は小head、**LoRA不要**。改善すればprefixからの読出し位置が有力、失敗でもnoise・時刻・基盤のOOD誤りを切り分ける必要がある。 |
| **5** | **点／物体とgripperの相対幾何をθ用queryに足す**。まず実行時画像・手先から取れる量、真の物体poseは別の特権診断。 | frozen tracker／視覚encoder＋小head。metric3Dはdepth・calibrationが別途必要、**LoRAは必須でない**。特権幾何だけ効くなら知覚・対象同定が律速、幾何でも効かなければ型配置・θの同定を再検討する。 |

最初の推奨は**1→2**。履歴と読む位置を安く切り分けてから3へ進む。Q-Formerの追加や基盤LoRAを先に大きくしても、情報源・運動教師・配置の曖昧さのどれを直したか分からなくなる。今回確認した一次資料には、この凍結π0.5＋無注釈DCT型＋別θという条件で改善を保証するものはなく、以上は検証可能な候補の順位である。
