# 検索した行動を生成の開始点にする手法の本文確認

確認日：2026-11-04 UTC。対象は承認済み `plan_memory_gap.md` の「先行研究の本文確認」。OptimusVLA と RTS の指定版の本文・全付録を HTML と PDF で照合した。以下の数値は著者報告であり、この調査での再現結果ではない。独立した比較実験を実施したとは扱わない。「未確認」は、その版で仕様・評価を確認できなかったことを表す。

**結論：検索した行動を flow / diffusion の開始点にすることも、重み固定で成功経験を追加することも既存である。ただし、今回の段階5に相当する「今回の追加学習に含めていないタスクへ外部実演だけを追加する」評価は、両論文で確認できなかった。** RTS は同じ凍結方策の通常生成との比較なので、報告された改善を方策への追加学習の効果とは解釈しない。OptimusVLA は部品除去の比較を持つが、今回の修正 Expert を同じ学習条件で訓練したゼロ下書き対照とは異なる。

## 1. 書誌と確認範囲

| 略称 | 確認した版 | 一次資料 |
|---|---|---|
| OV | Zaijing Li ほか、*Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation*（OptimusVLA）。arXiv:2602.20200v2、2026-05-18。arXiv comments に CVPR 2026 採択と記載。 | [版・著者・日付](https://arxiv.org/abs/2602.20200v2)、[本文・付録A–E](https://arxiv.org/html/2602.20200v2)、[PDF、17ページ](https://arxiv.org/pdf/2602.20200v2) |
| RTS | Jianchao Zhao ほか、*Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs*。arXiv:2605.10094v2、2026-05-12。採択先は今回未確認。 | [版・著者・日付](https://arxiv.org/abs/2605.10094v2)、[本文・付録A–I](https://arxiv.org/html/2605.10094v2)、[PDF、19ページ](https://arxiv.org/pdf/2605.10094v2) |

OptimusVLA は Optimus-R（arXiv:2609.39794）と別論文。今回、コード実行・公開実装との照合は行っていない。本文にない実装の詳細は推測で補わない。

## 2. 検索・保存・生成の比較

| 確認軸 | OptimusVLA | RTS |
|---|---|---|
| 検索キーの特徴 | VLM の multimodal **prefix tokens 全体をトークン方向に mean pooling**し、2層 MLP の Prior Head、L2正規化を通す。形式上の観測は現在の多視点画像・自己状態と指示。学習は **task ID で同じタスクの軌道を正例**にする InfoNCE。VLMの何層目か、padding等のpooling mask、各入力の厳密なtoken構成は**未確認**。§3.1–3.2、付録B.2、C.2、式22–24。 | **VLA の visual encoder の画像patch特徴**。各視点を空間グリッドに戻し、**2×2 average pooling → flatten → 視点間concat → 正規化**。言語条件付きVLM prefixのmean / MaxSimではない。キーへ指示・自己状態・履歴を連結する記述は確認できない。encoderの層番号は**未確認**。§4.1。 |
| キーを作る時点 | オフラインは「各episodeを1回forwardしてキーを作る」と記載するが、**episodeのどの観測時刻を使うか未確認**。オンラインqueryは現在観測。付録B.5では **episode開始時または大きなtask drift時だけ検索し、候補ID・重みをcache**。毎chunkで検索し直すと断定しない。 | 過去の各意思決定時点の画像キーと、その時点の生成action chunkを一時bufferへ。検索queryは現在画像。episode終了後に採用されたprefixを長期memoryへ追記。§4.1–4.2。 |
| 距離・候補数 | L2正規化キーの内積＝cosine、FAISS **IndexFlatIP**。top-k、softmax温度 τs。主なTable 5設定はk=8。検索スコアはタスクの意味の近さを学習しており、行動距離を直接学習したものではない。付録B.2–B.3。 | cosineのtop-Kに閾値 γsim を適用。さらに候補action chunks **相互の多変量DTW**を計算し、他候補との距離の中央値 ri が大きい候補を除く。正解未来とのDTWではない。デフォルト **K=10、γsim=.9992、softmax温度=.05**。§4.2、付録G Tables 8–9。 |
| 保存する値・粒度 | キー1件に**全行動軌道 J と長さ・chunk情報のmetadata**。prototypeや圧縮運動コードではない。付録B.2式25。LCMは別の短期記憶で、直前chunkとMambaの内部状態を扱う。 | 長期memoryのentryは **(画像キー, Hステップaction chunk)**。成功と判定されたepisodeの、進捗最大時刻までのentry集合を保存。全軌道を一つの値として検索する構造ではない。§4.1。予測Hステップ中の実行部分だけを保存するか、全chunkかの実装上の切り分けは**未確認**（数式上はa_tというchunk）。 |
| 軌道の時刻合わせ | 進捗scalar ρt∈[0,1] により各検索軌道のwindow indexを **ui=floor[ρt(Nchunks(i)−1)]** とし、長さH0、stride Δで切り出す。H0≠モデルHなら線形補間。**ρtを実行時にどう推定・更新するか未確認**。LCMの「進捗認識」という説明だけから、ρtの具体式を補わない。付録B.3式28。 | query画像に対応するentryのchunkを使う。DTWは**候補除外**に使う記述で、候補をDTW warpしてから平均する仕様は**未確認**。集約はchunk内の同じindex h同士。参照動画の進捗は書込み採否・prefix切断に使い、検索windowのρtとしては使わない。§4.1–4.2、付録C。 |
| 複数候補の利用 | αi=softmax(si/τs)、μ=ΣαiCi、Var=Σαi(Ci−μ)²。**moment matchingした単一の対角Gaussian**に分散floorを置き、μ+λ(s̄)ε⊙sqrt(Var)から開始。LCMの学習済みbias Btを加える。個別modeを一つ選ぶGMM samplingではない。§3.1–3.3、付録B.3–B.4、式29–32。 | 類似度softmaxでelite priorを集約。Euclidean成分は加重平均、回転はSO(3)のgeodesic mean、離散gripperは投票、強い開閉競合時は最近傍gripperへ。単純な全次元の算術平均とは限らない。§4.2式4–5、付録C式22–31。 |
| flow / diffusionへの注入 | 事前学習時はGaussian source、**推論時に検索priorでsourceを交換**。検索の加重cosine s̄でnoise scale λとNFEを調整。開始時刻t0を変更する方式とは記載されない。flow本体は後段学習で固定。付録B.1、B.4、C.2–C.3。 | 固定flowの **xt0=(1−t0)aelite+t0ε** からt=0へ積分。平均cosineの正規化とDTW不整合の標準偏差からconfidenceを作り、sigmoidでt0へ写す。空memory・検索失敗時は通常Gaussian開始へ戻る。diffusion版も元のforward noisingで中間状態を作る。§4.3式6–10、付録B。 |

出典：[OV §3・付録B–C](https://arxiv.org/html/2602.20200v2#S3)、[RTS §4・付録A–C](https://arxiv.org/html/2605.10094v2#S4)。両手法とも、検索の類似度そのものが「動きの正しさ」を保証するという結果ではない。

## 3. 学習・追加・未学習タスク・対照条件

| 確認軸 | OptimusVLA | RTS |
|---|---|---|
| 導入時の学習 | Stage I：π0.5型VLAのCFM学習。Stage II：基盤VLA固定、Prior Headだけをtask-contrastiveに学習。Stage III：VLM・flow・GPM固定、LCMだけを **正解action−prior mean** の残差MSEで学習。Table 6はStage I **30k steps / batch512**、II・IIIは各 **1k / batch64**。ただし付録C.2はStage IIを **20 epochs** とも記載し、換算関係は未確認。 | 下流実演でfine-tune済みVLAと事前学習済みVLAC criticを使用。推論・成功記憶の追記時にVLAもcriticも更新しない。実機設定では各task **100実演**で基盤を学習済み、batch64、LR2.5e−5、8k–10k steps。その後同じ重みへオンライン記憶を追加。§3.1、付録A・D.3。 |
| memory追加時に更新が必要か | 記載されたキー計算・全軌道・FAISSへの追記自体には、構造上は勾配を要しないと読める。ただしこれは**構造からの推論**。原著はオフライン構築・固定bankの評価で、新task追加時の再学習要否・実測効果は**未確認**。付録Aはonline bank更新・継続学習を将来課題とする。 | **不要、評価済み。** 成功prefixのentryを非パラメトリックに追記。空memoryから300試行で蓄積する継続配備実験がある。容量超過時はFIFO。§4.1、§6、付録E。 |
| 未学習taskへの外部実演追加 | **未確認。** CALVIN ABC→Dは未見環境への転移、実機は照明・配置・物体instanceの変動。taskを学習から保留し、そのtaskの外部実演だけをbankへ追加する前後比較とは異なる。LIBERO-90のみで学習→Longへ実演追加という設定も未確認。 | **未確認。** 主目的は既に下流fine-tuneした方策の同じtask・似た環境での反復配備。付録Aの「unseen tasksにも参照動画が使える」というcriticの説明は、方策が未学習taskの動作を外部実演追加だけで獲得した実験ではない。 |
| 必要な追加情報 | 指示とtask IDがPrior Headの学習に必要。進捗ρtとその推定法は再現上の未解決仕様。 | taskごとに**学習集合の成功動画1本**をVLACの参照に使用。進捗criticの事前学習には成功動画の時間順序からの教師信号を使う。手動skill IDは不要だが、成功情報・参照動画・言語を不要にした無注釈運動発見ではない。付録A。 |
| 通常生成／memoryなし対照 | **あり。** Table 4でGPM・LCM両方なし、GPMのみ、LCMのみ、両方を比較。 | **あり。** 同じ凍結π0.5のoriginal samplerに対し、memoryなし、top-1、soft aggregation、直接再生、出力補間等を比較。Tables 4–5。追加学習のない方式なので、両条件の方策重みの差による説明は当てはまらない。 |
| 同じ追加学習をしたゼロ下書き対照 | **未確認。** GPMなしLCMありの行はあるが、同じLCMをゼロ／無関係なpriorで同じ教師・更新数により学習し直したのか、各行の再学習・固定条件を含めて明示を確認できない。部品除去を今回のB条件と同一視しない。 | **該当する追加学習がない。** ただし、ゼロanchorを同じt0・NFEで使う対照は**未確認**。通常samplerとの比較だけでは、中間初期化・ノイズ量変更から独立した検索内容の効果を全て分けたとは言えない。 |
| 無関係・shuffle memory対照 | **未確認。** memory数・kの比較はあるが、同数の無関係な軌道や値shuffleは確認できない。 | **未確認。** 「Unverified Memory」は失敗を含む全経験であり、同数の無関係task memoryではない。 |
| 主目的 | **成功率＋推論・学習効率＋時間的一貫性**。少数実演を新taskへ追記する適応が主評価ではない。 | **継続配備での成功率・閉ループ安定性の改善**。少数外部実演による未学習task適応ではない。検索込みの推論全体はTable 7で通常方策より遅く、主たる速度改善法とは扱わない。 |

出典：[OV §3.4・付録A–C](https://arxiv.org/html/2602.20200v2#S3.SS4)、[RTS §3–6・付録A・D–G](https://arxiv.org/html/2605.10094v2#S3)。ここでの未学習taskは、承認済み計画どおり「今回の追加学習に含めていないtask」。非公開の基盤事前学習全体に含まれないという保証までは置かない。

## 4. 数値と、その数値が答える問い

### 4.1 OptimusVLA

| 出典 | 設定と著者報告値 | 読めること／限界 |
|---|---|---|
| Table 1、§4.1 | LIBERO Spatial / Object / Goal / Long：π0.5 **98.8 / 98.2 / 98.0 / 92.4%**、OV **99.6 / 99.8 / 98.4 / 96.4%**、4suite平均 **96.9→98.6%**。各suite500 rollouts。seed数は未確認。 | 対象suiteでの学習・メモリ利用を含む結果。90-only→Long未学習taskへの実演追加の数値には使えない。 |
| Table 4、§4.3 | LIBERO-Long：GPMなし・LCMなし **92.4%**、LCMのみ **93.2%**、GPMのみ **94.8%**、両方 **96.4%**。 | 同論文内でのGPMの差はLCMありで **+3.2 pp**、LCMなしで **+2.4 pp**。+4.0 pp全体をmemory内容単独の効果にしない。各行のLCM再学習条件は上記のとおり未確認。 |
| Table 5、§4.3 | Long、bank **6500軌道・k8：96.4% / k16：94.8%**、**1300・k1：92.6% / k8：95.2%**、**130・k1：92.4% / k8：93.6%**。 | 保存量・候補数の影響。**6500軌道・k1の値はこの表にはない**。容量を増やすことと、新task実演の追記効果は別。 |
| Table 2、§4.1 | CALVIN ABC→D、平均達成長 π0.5 **4.26→4.45**。 | 未見環境転移の結果。新taskデモだけを追加する評価ではない。 |

出典：[OV Tables 1–2・4–5](https://arxiv.org/html/2602.20200v2#S4)。速度は§4.4とFig.5にLIBERO **6.5倍の推論速度、NFE 10.0→3.2**、実機2.9倍と記載されるが、表から抽出した数値ではないため上の表と分ける。公式実装・同一環境での再計測は未実施。

### 4.2 RTS

| 出典 | 設定と著者報告値 | 読めること／限界 |
|---|---|---|
| Table 1、§5.1 | LIBERO-10（Long）、π0.5 **92.4±.2→94.4±.3%**、π0 **81.6±.8→84.4±.4%**。各task50試行、再現系平均は3seedのmean±std。 | 同じ凍結方策で成功経験を利用するシミュレーションでの閉ループ評価。task別にはBook in Caddy **100→92%**等の悪化もある。反復配備での平均改善は、全taskで改善するという意味ではない。 |
| Table 4、§6 | π0.5通常 **92.4%**、top-1中間初期化 **93.6%**、top-K加重prior **94.0%**、top-K+DTWの直接再生 **87.8%**、出力補間 **93.0%**、全体の中間初期化 **94.4%**。中間初期化条件は全てdynamic t0。 | 検索actionをそのまま実行するより、生成器で修正する方がよい。同じt0・NFEのゼロanchor、無関係anchorの対照はない。top-1対全体の+0.8 ppを、適応t0だけの効果にしない。 |
| Table 5、§6 | memoryなし **92.4%**、未検証の全経験 **87.6%**、予測成功のfull trajectory **91.8%**、予測成功prefix **94.4%**、oracle success **94.8%**。discriminator precision **.970**、recall **.678**。 | 保存量だけではなく品質・切断が効く。oracleは診断用の上限。なおoracle行の名称はFull Trajectoryなのにtruncation欄は✓であり、細部は未確認。 |
| Table 6、付録E | Moka Pots on Stoveを300 test trajectoriesで継続配備。最終累積SR：容量0 **61.0%**、1k **64.0%**、3k **68.5%**、5k **70.8%**、無制限 **71.2%**。有限容量はFIFO。 | 固定重みでmemory蓄積に伴う改善を測った直接例。Table 1の50/task条件とは別実験。同じtaskの経験追加であり、外部の未学習task実演の追加効果ではない。 |
| Table 7、付録F | 通常方策を1.00倍とした推論時間：elite prior取得後のsampler **.95倍**、検索・filter・集約込み **1.10倍**。 | full pipelineは10%長い。数値は現在の意思決定処理の時間であり、episode後のVLAC判定コストまで含むかは未確認。 |
| Table 8、付録G | top-K：**K1 93.6 / K3 93.8 / K5 94.1 / K10 94.4 / K20 94.2%**。 | 多候補化は単調には効かない。画像cosine上位候補を増やせばよい、という保証ではない。 |

出典：[RTS Tables 1・4–5](https://arxiv.org/html/2605.10094v2#S5)、[Tables 6–9](https://arxiv.org/html/2605.10094v2#A5)。t0の比較は**表でなくFig.4(b)と§6本文**で、Moka Pots上の最良固定t0=.6が **68%**、dynamic t0が **70%**。この+2 ppを全LIBEROへの一般的な適応効果とは扱わない。実機では付録D.3が各task50試行・同じ初期状態と明記し、Table 3の4本試験管の全完了率は **18→24%**。この値とLIBEROのシミュレーションでの閉ループ評価を混同しない。

## 5. 本文だけでは埋まらない仕様・論文内の不整合

| 論文 | 未確認・不整合 | 実装・解釈への影響 |
|---|---|---|
| OV | episodeキーの観測時刻、VLMの層番号・mask、Prior Headの次元・活性化、ρtの取得・更新、H/H0/stride、τs、分散floor・λ/NFE上下限の具体値。 | 「task embeddingで検索する」以上の完全再現仕様は揃わない。特に実行時ρtを正解進捗で置くと特権条件になる。 |
| OV | Table 6はStage IIを1k steps、付録C.2は20 epochs。 | データ量とepoch/step換算の記載がなく、同じ設定と断定できない。 |
| OV | §3.2の初期化は原則その時点のqueryで説明、付録B.5はsession cache。再検索条件の「significant task drift」の定量定義なし。 | 何回検索したか・検索時間・段階誤認への挙動の再現に影響。cache条件を省略しない。 |
| OV | Table 3のRoboTwin Hard **38%は掲載8taskの平均**。付録Table 7の16task平均は **30%**。§4.1は16taskを評価と説明。 | 38%を16task全体の成績として引用しない。LIBEROの表値との矛盾ではない。 |
| RTS | encoder層、chunk Hと実行周期、DTW局所距離・正規化・除去閾値、tmin、進捗判定間隔Δ、各taskのmemory分離方法・試行順序等の詳細は未確認。 | 高いcosine閾値.9992やDTWは表現・単位に依存。数値だけをこちらのVLM特徴へ移して同じ条件とはしない。 |
| RTS | §4.1式2は0–100の累積進捗を表す一方、付録Gのη=.95との正規化位置が明示されない。 | 実装時に「.95」と「95」の変換を推測で確定しない。 |
| RTS | oracle full trajectory行とtruncation✓の表記、及び検索候補の時間warp実施の有無。 | oracleの厳密な保存範囲、DTWが整列まで行うかを原著コード確認なしに断定しない。 |

出典：[OV 付録B–D](https://arxiv.org/html/2602.20200v2#S2a)、[RTS §4・Table 5・付録G](https://arxiv.org/html/2605.10094v2#S4)。未確認項目を埋めるための実装推定や追加実験は今回行っていない。

## 6. 既存ノートとの重複・補足・訂正すべき読み方

既存ノートそのものは依頼の範囲外なので書き換えない。以下を今回の照合結果として参照する。

| 既存記録 | 今回の判定 |
|---|---|
| [novelty.md §4.3](../03_anchor_flow/novelty.md#43-optimusvla-複数検索の平均分散とadaptive-noiseまで重なる)：OVの加重平均・分散・adaptive noise、GMMではなく単一Gaussian | **整合、重複。** 付録B.3はmoment-matched single Gaussianを明記。新たにprefix mean pooling・task-ID教師・ρtの未記載・session cacheを補足。 |
| [novelty.md §6](../03_anchor_flow/novelty.md#6-確信度と開始ノイズ開始時刻の適応はどこまで既知か)：OVはλ/NFE、RTSはt0を変える | **整合、重複。** RTSのconfidenceはsoftmax entropyではない。固定.6の68%対dynamic70%は§6・Fig.4の単一taskの値。旧ノートはここを「§5のt0分析」とリンク説明しており、**正しくは§6**。 |
| [novelty.md §3・§9](../03_anchor_flow/novelty.md#3-比較表-開始点ラベル追加時更新確信度)：RTSの勾配なし継続追加 | **整合。ただし射程を限定。** 同一taskでの自己成功経験の追加。未学習taskへの外部実演追加を立証したという読み方は不可。OVの「推論時固定」も追加task適応の実証とは分ける。 |
| [baselines_success_rates.md §4–5](baselines_success_rates.md)：OV Long96.4%、RTS92.4→94.4%、90-onlyではない | **表値・比較不可の理由は整合。** RTSの参照動画1本/taskと対象taskへのfine-tune済みという限定も整合。対象を追加学習から保留した段階5の期待改善量に転用しない。 |
| 同ノートのRTS試行数50/task×3seed | Table 1のcaptionに沿う要約。ただしtrial順・空memoryからの蓄積方法は完全には特定できず、**全150試行を独立zero-shotと扱わない**。 |
| 旧ノート§8のOV「GPMのみ94.8、LCMのみ93.2」、RTS top-1 93.6等 | **整合。** ただしGPM/LCM除去表を「同じ修正Expertをゼロ下書きで学習した対照」や、RTSを「追加学習の効果しか示していない」と読み替えない。 |

**数値の主要な矛盾は確認しなかった。** 追加すべき限定は、検索キーの内容・検索時点・進捗の未記載と、継続配備／未学習task適応／追加学習を分けることである。

## 7. 承認済みmemory-gap計画への含意

以下は論文の観測結果ではなく、今回の計画への解釈。事前基準・分割・実行許可は変更しない。

1. **段階2のB（ゼロ下書き）は必要。** OVのLCM追加学習を含む表も、RTSのtraining-free結果も、こちらの修正Expertの追加学習で得られる効果を差し引く代わりにはならない。こちらでは同じモデル・ノイズ・積分回数でC/DをBと比べる計画を維持する。
2. **候補の存在と、実行時に選べることを別に測る。** RTS自身も付録Iで視覚的aliasingとphaseの違いによる誤検索を認める。RTSのDTWは候補間の多数派の整合であり、少数の正しい候補を保証する機構ではない。D/Eは正解を使う診断で、実行可能な性能・汎化の証拠にしない。
3. **段階5の差の候補は狭く明記する。** 「固定VLAに成功memoryを追加して改善」はRTSと重なる。一方、全導入学習・選択規則をLIBERO-90で固定し、今回の追加学習に含めていないLongへ外部実演5/20/50本だけを追加し、ゼロ・90のみ・無関係実演と比較する評価は、今回確認した両版に見当たらない。ただし、この2論文だけで世界初とは主張しない。
4. こちらの観測の表現は **「今回の既知タスク・設定では、メモリ固有の改善を確認できていない」** を維持する。差を確認できない結果を同等とせず、仮説を観測と混ぜない。シミュレーションでの閉ループ評価と実機の著者報告を分ける。

## 8. 取得・作業記録

原資料は `.skillnet-state/outputs/memory-pilot/research-retrieval-init-20261104T082054Z/` に新規保存した（HTML、表JSON、PDF、抽出テキスト、manifest）。Git管理外のローカル証跡で、このcheckoutだけで再取得不要になるとは扱わない。PDF抽出の初回は不正Unicodeで失敗したため、再抽出した `.pdf.safe.txt` を照合に使用した。著者HTMLとPDFで上の主要表値を確認した。

指定された旧 `docs/memory_pilot_gates.md` は存在しないため、移動先の [gates.md](../01_memory_gates/gates.md) と [研究文書の規則](../README.md)を確認した。変更範囲は本ファイルとsurveys/READMEの案内1行のみ。コード・モデル・実験計画・判定条件を変更せず、学習・ロールアウト・GPUジョブ・commitは実施していない。
