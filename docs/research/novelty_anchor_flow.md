# アンカー開始分布型スキルメモリ: 新規性と実現性

確認日: **2026-10-28 UTC**。対象は[構想§1–2](anchor_flow_skill_memory.md)。[既存調査§3.1](related_work_retrieval_memory.md)のOptimus-R精読を前提に、下記の版指定本文を追加確認した。数値は著者報告であり、再現実験ではない。コード・既存文書・モデルは変更せず、この文書と出力ディレクトリの同内容の報告だけを新規作成した。指定の旧gatesパスは移動しているため、現行の[memory_pilot_gates.md](memory_pilot_gates.md)を参照した。

## 1. 結論

**現在の広い主張のままでは、新規性はかなり狭い。** 特に見落とせないのは次の四つである。

- **Retrieve-then-Steer**は、成功経験を勾配なしで蓄積し、検索結果から行動priorを作り、検索の確信度で**開始時刻t0**を変えて、固定VLAのflowを途中から実行する。今回の「継続追加＋検索＋確信度による開始時刻」の広い組合せは既存である。
- **LAFM**は、自己教師ありのlatent actionで運動を分け、それに対応する**平均・分散を持つ開始分布**を観測から選ぶ。運動モードの情報を開始分布へ入れること自体も既存である。
- **OptimusVLA**は、複数の検索軌道からsoftmax加重した平均・分散を作り、**検索類似度で初期ノイズ量とNFEを適応**させる。Optimus-Rとは別論文であり、本案の開始分布に関してはこちらがさらに近い。
- **SL-FM／MOSS**は、modeをvelocity fieldへ別入力せず、開始分布を通じて行動を制御する。「skill tokenを入れず、初期条件だけにする」という経路の違いも、それ単独では新規性にしにくい。

**残る候補は、可変個数の運動単位を実際の行動経験から形成し、学習済みの表現・方策を固定したまま新単位を追加して、未見運動への適応を改善すること。** raw chunk記憶より少ない容量で、運動型を保ち、外観・配置を越えて転用できるなら意味がある。単に検索scoreをentropyへ替えることや、中心に分散を足すことだけでは弱い。全ての要素を満たす同一方式の先例は今回未確認だが、世界初とは断定できない。

以下、文献の確認事項と本案への数学的・実験的な指摘を分ける。

## 2. 書誌・確認した版

arXivの年月と会議年を区別した。「採択先未確認」は会議が存在しないという意味ではない。版の日付はarXivのsubmission historyに基づく。

| 略称 | 確認した書誌・版と日付 | 一次資料 |
|---|---|---|
| Retrieve-then-Steer（以下RTS） | Jianchao Zhaoら, **Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs**。arXiv:2605.10094 **v2、2026-05-12**（初稿05-11）。採択先未確認。 | [v2書誌](https://arxiv.org/abs/2605.10094v2)、[v2本文§4–5・付録](https://arxiv.org/html/2605.10094v2) |
| LAFM | Bruno Machado, Alexandre Chapin, Emmanuel Dellandréa, Liming Chen, **Flowing With Purpose: Latent Action Guided Flow Matching Policies For Robotic Manipulation**。arXiv:2606.23420 **v1、2026-06-22**。採択先未確認。 | [v1](https://arxiv.org/abs/2606.23420v1)、[本文§2–3](https://arxiv.org/html/2606.23420v1) |
| OptimusVLA | Zaijing Liら, **Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation**。CVPR 2026、arXiv:2602.20200 **v2、2026-05-18**（初稿02-22）。 | [v2書誌・採択記載](https://arxiv.org/abs/2602.20200v2)、[本文§3.2・付録B](https://arxiv.org/html/2602.20200v2) |
| SL-FM | He Zhangら, **Source-Lifted Flow Matching for Intervenable Multimodal Imitation**。arXiv:2607.10206 **v2、2026-09-28**（初稿07-11、著者一覧更新あり）。採択先未確認。 | [v2](https://arxiv.org/abs/2607.10206v2)、[本文](https://arxiv.org/html/2607.10206v2) |
| MOSS | Zirun Ma, Jindou Jia, Tuo An, Jiaqi Bai, Jiahua Dong, Jianfei Yang, **MOSS: Mode-Oriented Source Shaping for Robot Flow Matching Policies**。arXiv:2610.22193 **v1、2026-10-15**。採択先未確認。 | [v1](https://arxiv.org/abs/2610.22193v1)、[本文§4・付録C–D](https://arxiv.org/html/2610.22193v1) |
| RAGDP | Sodtavilan Odonchimedら, **Retrieve-Augmented Generation for Speeding up Diffusion Policy without Additional Training**。arXiv:2507.21452 **v1、2025-07-29**。採択先未確認。 | [v1](https://arxiv.org/abs/2507.21452v1)、[本文](https://arxiv.org/html/2507.21452v1) |
| BRIDGeR | Kaiqi Chen, Eugene Lim, Kelvin Lin, Yiyang Chen, Harold Soh, **Don't Start from Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion**。RSS 2024、arXiv:2402.16075 **v4、2024-07-11**。 | [v4](https://arxiv.org/abs/2402.16075v4)、[本文](https://arxiv.org/html/2402.16075v4)、[会議原稿](https://www.roboticsproceedings.org/rss20/p122.pdf) |
| R2-Diff | Takeru Oba, Norimichi Ukita, **R2-Diff: Denoising by diffusion as a refinement of retrieved motion for image-based motion prediction**。arXiv:2306.09483 **v1、2023-06-15**。本比較はこの版を対象とし、出版版との差分は未比較。 | [v1](https://arxiv.org/abs/2306.09483v1)、[本文](https://arxiv.org/html/2306.09483v1) |
| PriorGrad | Sang-gil Leeら, **PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior**。ICLR 2022、arXiv:2106.06406 **v2、2022-02-20**。 | [v2](https://arxiv.org/abs/2106.06406v2)、[本文](https://arxiv.org/html/2106.06406v2) |
| Data-dependent couplings | Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, Eric Vanden-Eijnden, **Stochastic interpolants with data-dependent couplings**。ICML 2024、arXiv:2310.03725 **v3、2024-09-23**。 | [v3](https://arxiv.org/abs/2310.03725v3)、[本文§3](https://arxiv.org/html/2310.03725v3) |
| MM-FM | Gaoxiang Luo, Frank Cole, Sihang Zhang, Yuxiang Wan, Yulong Lu, Ju Sun, **Flow Matching for Multimodal Distributions**。CVPR 2026と著者公式ページに記載。arXiv版・公開日の正確な日付は未確認。 | [著者公式・手法と結果](https://mm-flow.github.io/)。会議[HTML](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Flow_Matching_for_Multimodal_Distributions_CVPR_2026_paper.html)／PDFは今回403で本文取得不可。下記は著者公式で確認した範囲。 |
| RA-VLA | Sanghwan Jangら, **RA-VLA: Retrieval-Augmented VLA for Test-Time Adaptation**。arXiv:2608.25585 **v2、2026-10-07**（初稿08-26）。著者書誌にICML 2026と記載、会議録との照合は未確認。 | [v2](https://arxiv.org/abs/2608.25585v2)、[本文](https://arxiv.org/html/2608.25585v2) |
| STEP | Jinhao Liら, **STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction**。arXiv:2602.08245 **v2、2026-05-02**。ICML 2026採択と著者記載。 | [v2](https://arxiv.org/abs/2602.08245v2)、[本文](https://arxiv.org/html/2602.08245v2) |
| ADPro | Zezeng Li, Rui Yang, Ruochen Chen, ZhongXuan Luo, Liming Chen, **ADPro: a Test-time Adaptive Diffusion Policy via Manifold-constrained Denoising and Task-aware Initialization for Robotic Manipulation**。arXiv:2508.06266 **v2、2025-09-30**。採択先未確認。 | [v2](https://arxiv.org/abs/2508.06266v2)、[本文](https://arxiv.org/html/2508.06266v2) |
| FRS | Andy Tang, William Chen, Andrew Wagenmaker, Chelsea Finn, Sergey Levine, **Improving Robotic Generalist Policies via Flow Reversal Steering**。arXiv:2606.13675 **v2、2026-06-12**。採択先未確認。本件では開始分布への誘導との関係を参照。 | [v2](https://arxiv.org/abs/2606.13675v2)、[本文](https://arxiv.org/html/2606.13675v2) |
| ResiP | Lars Ankile, Anthony Simeonov, Idan Shenfeld, Marcel Torne, Pulkit Agrawal, **From Imitation to Refinement — Residual RL for Precise Assembly**。arXiv:2407.16677 **v4、2024-12-12**。本件では採択先を確定しない。 | [v4](https://arxiv.org/abs/2407.16677v4)、[本文](https://arxiv.org/html/2407.16677v4) |
| SDEdit | Chenlin Mengら, **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations**。ICLR 2022、arXiv:2108.01073 **v2、2022-01-05**。 | [v2](https://arxiv.org/abs/2108.01073v2)、[本文](https://arxiv.org/html/2108.01073v2) |
| Optimus-R | Zaijing Liら, **Inline Memory Meets Reusable Skills: Memory-centric Framework for Vision-Language-Action Model**。arXiv:2609.39794 **v1、2026-09-30**。採択先未確認。 | [v1](https://arxiv.org/abs/2609.39794v1)、[本文](https://arxiv.org/html/2609.39794v1)。既存調査§3.1を使用し、token経路を再照合。 |

## 3. 比較表: 開始点・ラベル・追加時更新・確信度

「ラベルなし」は**手動のskill／mode IDを不要にすること**と、言語・タスク・成功情報も使わないことを区別する。「追加未評価」は、通常の推論時に凍結されることだけを根拠に、継続適応を実証したとしないための表示。出典は各行から第2節の版指定本文へ対応する。

| 研究 | (a) 開始点をどこから得るか | (b) 単位・ラベルの由来 | (c) 経験・単位追加時の勾配 | (d) 確信度で開始ノイズ／時刻を変えるか |
|---|---|---|---|---|
| **RTS** | 現観測から成功chunkを検索し、DTWで不整合を除外、softmax加重したelite actionを作る。`x_t0=(1−t0)a_elite+t0ε`。 | 手動skill IDは不要。学習済みVLAC critic、指示、**taskごとに1本の成功参照動画**で進捗・成功prefixを判定。自己組織化した運動prototypeではない。 | **不要**。固定VLAへのonline成功記憶の追記を評価。criticも事前学習済み。 | **する**。平均類似度とDTW不整合の分散からconfidenceを作り、sigmoidでt0へ。**entropyではない**。 |
| **LAFM** | 観測からlatent action indexを予測し、対応する学習済み`N(μ_k,Σ_k)`を開始分布にする。 | 自己教師あり動画LAMのcodeで疑似ラベル化。手動skill IDではない。標準codebookは**512**。言語指示も入力。 | priorのμ/logσ・index予測器・FMを学習。固定後に新codeを無勾配で増やす実験は**未確認**。 | 観測でpriorを選ぶ。検索confidence／entropyによるt0適応は**未確認**。 |
| **OptimusVLA** | 類似taskの全軌道から進捗対応chunkを検索。加重平均・分散を持つ**一つのGaussianにmoment matching**して開始。 | task意味埋め込み、同taskを正例にするInfoNCE、進捗整合。運動だけから発見する単位ではない。 | Prior Head等を段階学習。推論時固定。新運動を追記だけで獲得する実験は**未確認**。 | **する**。softmax重み付き類似度でノイズscale λとNFEを変える。**t0ではない**。 |
| **SL-FM** | 観測依存のsource mixtureからhandleを選ぶ。sourceを補助直交座標へliftし、共有fieldで運ぶ。 | 明示的な教師modeへのtarget分割ではなく、end-to-endにsource handlesを学習。評価のroute labelと区別。 | source mixtureとfieldを学習。勾配なしのhandle追加は**未確認**。 | source構造と選択は適応。検索確信度によるt0は**未確認**。 |
| **MOSS** | modeごとのfuture action平均を中心化し、分岐強度に応じてsource中心と分散を変える。 | **mode labelを使用**。推論時もuser／plannerがmodeを指定。labelをfieldへ直接入力しない。 | 統計をoffline計算しstrength estimator・fieldを学習。新modeの追記実験は**未確認**。 | **局所multimodalityでsourceの隔たり・分散を適応**。検索entropyや開始時刻ではない。 |
| **RAGDP** | expert観測–action DBから近傍完成chunkを検索し、中間noise levelへ加ノイズして生成。 | skill ID不要。expert demonstrationsのchunkで、明示的な運動prototypeではない。 | DB構築・推論に追加学習不要。逐次追加後の旧新タスク保持の評価は**未確認**。 | leap ratioは手動hyperparameter。confidence依存の自動化は当該版で**未確認**。 |
| **BRIDGeR** | Gaussianに限らずheuristic／CVAE等のsource policy。source→expertのstochastic interpolantを学ぶ。 | source設計に依存。手動skill IDを必須としないが、無ラベルの成長memoryを提案したものではない。 | sourceに対応したbridgeの学習が必要。固定bridgeへの新単位追記のみは**未確認**。 | confidenceからt0を選ぶ方式は**未確認**。 |
| **R2-Diff** | 画像から訓練motionを検索し、noiseを加えて修正。 | skill IDではなく訓練motion。 | noise scheduleを含む学習あり。継続追記時の全固定評価は**未確認**。 | 検索motionの距離をnoise設計に用いるが、各queryのconfidenceによるt0適応は**未確認**。 |
| **PriorGrad** | 条件から平均・分散を計算する非標準Gaussian prior。 | vocoderはmel情報、acoustic modelはphoneme統計。運動単位ではない。 | adaptive priorを前提とした生成器訓練。robot memory追加という設定はない。 | **条件依存のprior統計**。検索confidenceによるt0ではない。 |
| **Data-dependent couplings** | base–targetを独立にせず、データに依存するjoint couplingを構成。 | class labelは理論上必須でなく、必要なら追加条件として扱える。 | couplingに対応したvelocity／scoreの回帰学習。継続memory追加は評価対象外。 | confidenceによるrobot t0適応は評価対象外。 |
| **MM-FM** | DINOv2等のlatentにGMMをfitし、対応mode内でsource–targetをcouple。 | 画像クラス注釈を使わないGMM群。mode conditioningなしのfieldもある。 | GMM推定とFM事前学習。成長skill memoryの追記は著者公式の確認範囲では未評価。 | そのようなrobot t0適応は公式説明にない。会議本文全体は取得不可。 |
| **RA-VLA** | 検索expert segmentsの**平均chunk**を初期noisy actionに使う。 | 言語・観測と、行動整合の検索を使う実演segment。prototypeとして運動型を形成しない。 | 検索器・context adherenceを事前学習。held-out taskでは実演buffer追加だけ。 | confidence／entropyでt0を変える記載は確認できない。 |
| **STEP** | 前chunkと現観測から、学習したpredictorがwarm-start actionを作る。 | 連続するchunk。ラベル付きskill集合ではない。 | predictorの訓練が必要。episode間memoryの追記ではない。 | 状態・速度に応じた摂動あり。検索confidenceのt0ではない。 |
| **ADPro** | gripper／target point cloudの粗いregistrationから初期行動を作る。 | 幾何対応。運動skill IDや経験群の中心ではない。 | 既存DPへ推論時適用、再訓練不要と記載。skill memory追加の評価ではない。 | 幾何・manifoldを使うadaptive denoising。検索entropyのt0ではない。 |
| **FRS** | 粗い参照行動を逆flowでnoiseへ戻し、再生成する。 | human／VLM等の参照行動。skill IDを必須としない。 | zero-shot steeringは基盤重み固定。noise actorを学ぶ別設定もある。memory追記は主題でない。 | 検索confidenceによるt0は確認できない。 |
| **ResiP** | frozen BCの完成行動にRL residualを加える。 | task rewardとBC経験。prototype形成ではない。 | **residual actorはRLで更新**する。 | flowの開始分布・t0を変える方法ではない。**residual補正とsource初期化は別**。 |

SDEditはガイド画像を指定したnoise levelへ崩して再生成する画像側の先例で、robot memoryやquery confidenceからのt0を実証したものではない。逆に、GMFlowのような**velocity／denoising分布をGaussian mixtureにする**研究を、混合**開始分布**の先例と混同してはいけない。本件の混合開始分布にはLAFM・SL-FM・MOSS・MM-FMが直接関連する。

## 4. 最も近い先例との具体的な境界

### 4.1 RTS: 継続記憶＋確信度t0は既存

RTS本文式(8)–(9)は、平均検索類似度を正規化した値とDTW不整合の標準偏差からconfidence cを作り、`t0=t_min+(1−t_min)·sigmoid(−γc)`へ写す。初期状態は式(6)の`(1−t0)a_elite+t0ε`で、低confidenceならGaussianの通常samplerへ近づく。空memory／検索不成立時は通常生成へ戻る。[v2§4.3](https://arxiv.org/html/2605.10094v2)

本案との差候補は、成功raw chunksを蓄積する代わりに、**運動encoderが作った可変個数の単位・中心・分散を保持すること**、成功critic・task参照動画への依存を減らすこと、未学習運動への転用である。entropyに替えるだけでは、本質的に違う適応原理を示したことにはならない。RTSも人手のtest-time成功ラベルを不要にしているので、「手作業で成功を付けない」を差としない。

### 4.2 LAFM／SL-FM／MOSS: source-onlyも、無ラベルlatentも既存

LAFMのpriorは、**実際の運動コードの経験平均そのものではなく、学習するμ/logσ embedding**。codebook数Cに合わせた分類headを持つ。本案が可変集合への類似度検索を使えば、固定Cの分類headを拡張せずに新単位へ到達できる。ただしこれは表現能力・継続適応を実験で示すべき差である。[LAFM v1§2.2–2.4](https://arxiv.org/html/2606.23420v1)

SL-FMはmode handleを別条件としてfieldへ渡さず、無ラベルのsource mixtureを学ぶ。MOSSはmode labelを使うが、fieldには渡さずsourceのみで選択する。本案の「外部skill条件なし」という表現は、**「skill ID／skill tokenをvelocity fieldへ別入力しない」**まで限定する。開始分布も観測とmemoryに依存するため、広義には条件付けである。[SL-FM v2](https://arxiv.org/html/2607.10206v2)、[MOSS v1§4](https://arxiv.org/html/2610.22193v1)

### 4.3 OptimusVLA: 複数検索の平均・分散とadaptive noiseまで重なる

OptimusVLA式(10)–(14)は、`α_i=softmax(s_i/τ)`、`μ=Σα_i C_i`、`Var=Σα_i(C_i−μ)^2`から、`x_src=μ+λ(s̄)ε⊙sqrt(Var)`を作る。類似度s̄が高いほどλとNFEを小さくする。ここは本案の「中心＋ばらつき」に極めて近い。[v2§3.2・付録B](https://arxiv.org/html/2602.20200v2)

ただし実装式は**複数modeの一つをサンプルするGMMではなく、moment-matchedの単一Gaussian**。離れた運動modeを保った混合samplingが、平均化より有効な条件を示す余地はある。一方、それだけではLAFM／SL-FM／MOSS等を越える継続memoryの新規性にはならない。

## 5. Optimus-Rとの差: 経路は違うが、それだけでは足りない

| 論点 | Optimus-R v1（既存§3.1＋本文再照合） | 本案 |
|---|---|---|
| skill valueの生成元 | 現画像・状態・言語のmemory token状態から符号化。future action予測で学習。 | 実際の区間Aと状態sをEncしてbを得る。 |
| query表現の学習 | task ID・正規化進捗等をquery対照学習に使用。 | 運動単位の発見にtask／skill IDを使わない構想。ただしサブタスク・画像・状態のquery入力は使う。 |
| 注入経路 | 検索skillをToken Projectorでskill tokensへ写し、multimodal tokensとともにFlow Policyを条件付ける。 | 検索値を復号し、source分布に入れる。fieldへ別skill tokenを渡さない。 |
| 追加時 | entry初期化は平均だが、Stage Cでquery整合・温度・active key/value残差を勾配更新。 | 表現・方策・検索器を固定し、統計・memory項目を追加する構想。 |
| 未検証の点 | 全学習パラメータ固定で追記するablationは既存精読で未報告。 | 新単位を足すだけで新運動を実行できるか、旧能力を保持するかは未実証。 |

**経路の差はOptimus-Rとの機構上の差分としては成立する。しかしsource-onlyの先例があるので、それ自体を一般的な新規性にしない。** 比較は、同じ運動単位・同じquery・同じ経験を用い、token経路／source経路を同一予算で学習する統制が必要。原著Optimus-R全体との比較だけでは、task監督・valueの生成元・学習予算まで同時に変わる。[Optimus-R v1§3](https://arxiv.org/html/2609.39794v1)

また、画像・サブタスク・状態から検索する以上、完全にannotation-freeな体系とは呼べない。正確には**運動単位の形成に手動skill IDや意味境界を使わない**。queryのサブタスク表現が運動分類を代行していないかも別に調べる。

## 6. 確信度と開始ノイズ／開始時刻の適応は、どこまで既知か

| 比較先 | 適応に使うもの | 変えるもの | 本案との差 |
|---|---|---|---|
| RTS | 絶対的な検索類似度＋DTW不整合 | **t0と初期状態のanchor/noise比** | 本案のH(w)は別scoreだが、確信度に応じてmemory priorを弱める原理は同じ。 |
| OptimusVLA | softmax加重したcosine類似度s̄ | **初期noise scaleとNFE** | 生成開始時刻を変えるとは記載しない。探索幅の適応は既存。 |
| MOSS | 近傍future actionのmode間／mode内のばらつきから学習したmultimodality | source中心の分離と方向別分散 | 曖昧さを全て「低confidence」と扱わず、分岐を保持する設計。 |
| RAGDP／SDEdit | 指定した開始noise level | 中間初期状態と残り生成域 | query confidenceで自動選択することは確認できない。 |

RTSはMoka Pots on Stoveにおいて、**最良の固定t0=0.6で成功率0.68、動的t0で0.70**と報告している。したがって「方策での確信度適応は未確認」とは書けない。一方、同論文の全intermediate-init ablationは動的t0を用い、適応だけの大きな一般的利得を示すものではない。[RTS v2§5のt0分析](https://arxiv.org/html/2605.10094v2)

**H(w)単独の問題（以下は本案への分析）:**

- 全keyから遠いqueryでも、相対softmaxは一つへ尖り得る。低entropyを「正しい検索」と同一視できない。
- 同じ運動を表す近い項目を複製するとentropyが上がり得る。memoryが成長しただけでt0が変わり、継続追加の効果と混ざる。
- 二つの同じ運動がw=(0.5,0.5)の場合と、左右の逆運動が同じ重みの場合はHが同じ。前者は冗長、後者は意味のある多峰性であり、必要なnoise量が同じとは限らない。
- 温度・top-k・総単位数がHを変える。H/logKの正規化だけでは重複・絶対距離・誤確信を解決できない。

最低限、entropy単独を、固定t0、絶対類似度、RTS型score、単位間の復号運動のばらつきを併用するscoreと比較する。成功予測・anchor誤差のrisk–coverage曲線を、学習内validationで較正し、未見外観／運動で保留評価する。

## 7. 数式上の実現性: 現在の§2で確定していない点

以下は先行論文の結果ではなく、**構想文書の式からの分析**である。コードの修正は行わない。

### 7.1 source分布と「途中時刻の分布」を混同しない

構想の学習式は`t=1`がsource、`t=0`がdataの向きで、`X_t=t·X_src+(1−t)·A`。この経路の**t0の周辺分布**は、sourceとtargetのjoint couplingが定める`Law[t0·X_src+(1−t0)·A | o]`である。

ところが推論式は、t0にかかわらず`Σw_k N(Dec(c_k,s),σ_k²I)`から直接サンプルし、それを時刻t0の状態とする。この二つの分布は一般には一致しない。これは単にNFEを調整する問題ではない。data-dependent couplingの理論は、**選んだjointと経路の周辺分布**を運ぶ結果であり、任意の時刻へsourceを差し込んだ性能保証ではない。[Albergoら v3§3](https://arxiv.org/html/2310.03725v3)

設計としては、(A)新しいsource分布を学習し、そのsourceから**全区間**を運ぶ、(B)Gaussian系の学習に合わせ、`X_t0=(1−t0)A_anchor+t0ε`等で**中間noise state**を作る、(C)t0ごとのsource endpointと時間幅に合う訓練経路を定義する、を区別する。RTSは(B)、LAFMは(A)に近い。現在の3種類のsourceを混ぜる訓練だけで、この区別が消えるとは言えない。

### 7.2 t0→1だけでは純粋noiseに戻らない

現在の(★)は、t0→1でも平均が`Dec(c_k,s)`の混合Gaussianのままであり、`N(0,I)`にはならない。**平均の寄与と分散をt0でどう変えるか、または通常Gaussianへの分岐をどう定義するか**が不足している。RTSの式(6)は、t0=1でanchor係数が明示的にゼロになる点が違う。

### 7.3 b空間のσと行動空間のσは別

memoryのばらつきはb空間で記録する一方、(★)の平均はDecを通したH×D_a行動空間にある。bのscalar分散をそのままactionの等方分散へ使うには、単位・正規化・decoderの変換を定義する必要がある。非線形Decでは、`Dec(E[b],s)`は`E[Dec(b,s)]`とも一般には違う。

比較候補は、bの分布をサンプルしてDecした経験分布、復号actionから推定する対角共分散、局所Jacobianで伝播する共分散。これらは本案への提案であり、既存結果としない。少数supportで分散を小さく見積もる問題や、低rankなsource・mode崩壊を避けるnoise floorも較正が必要である。

### 7.4 混合分布と平均値のblendは異なる

`k~Categorical(w), X~N(μ_k,Σ_k)`はmodeを選ぶsampling。`X=Σw_k μ_k+noise`は平均化で、同じ分布ではない。混合の全分散は`Σw_k Σ_k + Σw_k(μ_k−μ̄)(μ_k−μ̄)^T`で、単位内の調整幅と単位間の運動の違いを分ける。左右回避の中心平均が衝突する場合、共分散を足すだけで意図したmodeが必ず保たれるとは限らない。

### 7.5 初期のhead学習と、追加時の固定を分ける

構想はaction headをアンカー開始用に学習し、その後の単位追加では固定する。この前段の学習は明記すべきで、全方式がtraining-freeなわけではない。また、同じo・tに対してsource分布を後から変えると、固定velocity fieldにとっては訓練分布の変更となる。正則な決定論的ODEの同一flow mapが、任意に異なる全source分布を同じtarget分布へ正確に運ぶことは一般にはできない。これは追加が無効だという意味ではなく、**新しい運動priorを渡したときに望ましい一般化が起きるかは実験事項**という意味である。

さらに、正解区間bを使うsourceが訓練で多すぎると、実際の検索誤りへの耐性を過大評価する。同じepisodeを除くだけでなく、重複trajectory・近い収録系列を含む漏洩も除き、actual retrievalを使う学習・評価条件を主にする。

## 8. LIBERO閉ループで成功率が上がった実績

| 研究・版 | 報告値 | その結果が支えること／支えないこと |
|---|---|---|
| **RTS v2、Table 1** | LIBERO-10、各task50 trials、再現系平均は3 seeds。π0.5 **92.4±0.2→94.4±0.3%**、π0 **81.6±0.8→84.4±0.4%**。 | 固定VLA＋成功記憶＋検索prior＋adaptive t0が閉ループで改善した直接例。ただし既にtaskへfine-tuneした基盤のpersistent deploymentであり、未知運動の新規学習の証拠ではない。 |
| **RTS v2、Table 4** | 通常92.4%、直接再生87.8%、top-1初期化93.6%、top-k加重prior94.0%、DTW付き全体94.4%。 | 記憶を使えば常に改善するわけではなく、直接再生は悪化。Table 1にはtaskごとの悪化もある。運動prototypeの利点は別途必要。 |
| **LAFM v1、Table 3** | LIBERO-90、FM **82.6±0.2%**、FM＋latent action分類 **85.4±0.1%**、LAFM **93.0±0.2%**。全LIBERO評価は各task50 trials×3 seedsと記載。 | 運動latentに対応するsource priorの学習が有効な強い例。ただしlearned priors・全体学習の結果で、無勾配の可変単位追加ではない。 |
| **OptimusVLA v2、Table 4** | LIBERO-Long: GPM/LCMなし92.4%、**GPMのみ94.8%**、LCMのみ93.2%、両方96.4%。 | source prior memoryの部品を切り分けた実績。両方あり対GPMなしは+3.2ポイント。全体平均98.6%という値だけを開始分布の効果としない。 |
| **RA-VLA v2、Table 1** | 未学習suiteを保留し、taskごとに3 expert demosをbufferへ。action-aware検索のRICL_R **20.85%→RA-VLA 38.45%**。各保留task50 trials。 | 未見taskで重みを更新せず実演bufferから適応した例。ただしcontext cross-attention・検索器・学習損失等も変わり、初期平均chunkだけの効果ではない。 |
| **STEP v2、Table 18** | π0.5のLIBERO平均、10-step **97.2%→1-step＋STEP 96.3%**。 | warm startをVLAで実行できる例。これは成功率向上の根拠ではない。 |

したがって**LIBERO閉ループでの実現性には肯定的な先例がある**。ただし本案の「ラベルなし運動単位・可変個数・追加時固定・未見運動への改善」を一体として検証した先例は今回未確認。RAGDP／BRIDGeRの結果をLIBEROでの実績と読み替えない。

## 9. 残る新規性と、査読で突かれる点

| 主張 | 判定 | 主要な反論・必要な証拠 |
|---|---|---|
| 観測依存の非Gaussian sourceでflowを始める | **既存** | PriorGrad、BRIDGeR、couplings、LAFM等。 |
| ラベルなしの運動latentをsource選択へ使う | **既存に近い** | LAFM・SL-FM。実行運動からの事後符号と、動画latent／learned handleの違いを示す。 |
| skill tokenを入れずsourceだけで行動を誘導する | **既存** | SL-FM・MOSS。Optimus-Rとの経路差だけでは不十分。 |
| 勾配なしの経験追加＋検索prior＋confidence t0 | **既存** | RTS。entropyへの変更だけは小さな設計差。 |
| 複数検索から平均・ばらつきを推定しnoiseを適応 | **既存** | OptimusVLA。真のmixtureとmoment matchingの差は検証余地あり。 |
| 可変個数の、運動上非冗長な単位を追記して新運動へ転用 | **貢献候補・未検証** | raw success memoryと同容量・同経験・同学習予算で比較。旧task保持、追加順序、圧縮率だけでなく単位の固有寄与を示す。 |
| 外観・配置に依存しない運動型が自己組織化する | **貢献候補・未検証** | decoderが使える相対運動を保ちつつ、外観・対象の交換で転用できるか。現在のbがchunk全体の圧縮だけなら「運動コード」と呼ぶ。 |
| entropy t0が未知状況に頑健なfallbackを与える | **そのままでは成立しない** | Hは絶対的な信頼度ではない。低confidence時の実際のGaussian復帰と、OODでの誤確信を検証する。 |

最も守りやすい課題設定は、**自己組織化した少数の運動単位を、可変のsource-memoryとして追加し続けたとき、raw chunkの成功記憶に比べて、未見運動・外観転用・保存効率をどこまで改善できるか**。単なる既存部品の組合せ以上の効果が出なければ、表現・継続適応・校正のどれかへ主張を絞る必要がある。

## 10. 反論を潰す最小の実験

ここは**提案**であり、今回の調査では実行しない。元の構想・既存gatesの評価許可や計算予算をこの文書だけで拡張しない。

### 実験1: 近い方式との差を、同じ方策・経験で分ける

最初に、同じaction headの学習データ・更新量・観測・NFEを揃える。対照は、(i) Gaussian開始、(ii) raw chunk＋再ノイズ（RAGDP型）、(iii) RTS型のraw成功記憶＋confidence t0、(iv) 検索chunkからmoment-matched Gaussian（OptimusVLA型）、(v) 提案の運動単位mixture。全方式へ同じ経験を与え、成功filterの有無による差を主張と混ぜない。

RTS原著のVLAC criticを使わない対照は「原著完全再現」ではなく、同じmemory内容での機構比較と明記する。単位形成の効果には、固定数のlearned priors（LAFM型）または同じb集合を圧縮せず保持する対照も必要。token／sourceの差を主張するなら、**同じ単位・queryを使うtoken-only条件**を追加し、Optimus-R原著との多要因の違いと区別する。

### 実験2: mixtureの利点は、多峰性を保つことか

同一観測で左右回避など二つの有効運動がある条件を用意し、oracle検索で検索誤りを除く。component sampling、中心の加重平均、moment-matched Gaussian、top-1、単位内分散なしを比較する。接触・衝突、成功率、mode coverage、同一mode内の配置追随を測る。

best-of-8誤差はoracleが良い1本を選ぶ上限であり、実際の1回実行の成功率を代替しない。各サンプル成功率とmode分布も出す。modeが続く複数chunkで、再samplingによる左右切替が起きないかも確認する。

### 実験3: entropyによるt0適応に固有の利点があるか

最良の固定t0、同じt0ヒストグラムをquery間でshuffleした条件、entropy、絶対類似度、RTS型scoreを比較する。生成NFEを揃え、速度の差を目的にせず成功率・fallbackの妥当性を測る。memoryの重複追加、類似外観の違う運動、遠いOOD queryを故意に含める。

必要な結果は、単に平均成功が高いことだけでなく、**誤検索時にGaussian通常方策へ戻って回復すること**。開始時刻を変えるだけで分布を変えない実装と、式に整合した初期化を混ぜて報告しない。スケジュールと閾値は保留評価を見ずに決める。

### 実験4: 真の継続追加と、運動型の転用

encoder・decoder・query encoder・temperature・action headを全固定し、未見運動の収録経験だけを追加する。追加前後の新task成功と旧task保持を評価し、追加された単位の使用率・値shuffleの効果・同じmemory bytesのraw記憶との差を出す。新しい観測keyの追加による被覆改善と、新しい運動valueの追加を別々に無効化する。

単位発見はtask IDなしで行い、評価時のmotion規則は幾何・軌道から定義する。外観だけを変えた同運動と、外観が似た別運動を交差させる。sourceとqueryはepisode・初期状態で分け、同一episodeや近接収録の漏洩を除く。LIBERO-90→保留task列という構想の系列を使う場合も、事前encoderの学習データへ保留運動が入っていないかを記録する。

少なくとも対応する初期状態でのpaired rolloutと複数seedを使う。構想の20試行/taskはpilotにはなるが、2〜3ポイントの差の確定には信頼区間が広い。50試行/task×3 seeds等の文献プロトコルを参考にしつつ、差の不確実性を示す。新単位を増やしてもdecoder／fieldが生成不能な運動は獲得できないので、oracle値を使った表現上限も併記する。

## 11. 調査範囲と未確認事項

主要論文は第2節の版を使用。RTS・LAFM・OptimusVLA・SL-FM・MOSSの方法と関連する表を優先して精読し、Optimus-Rのtoken経路を既存調査と照合した。MM-FMは会議本文が403のため、著者公式のmethod／theory説明まで。新規性の否定を第三者のAI要約だけに依存していない。

本案と同一の可変運動単位・無勾配追加・実測分散mixture・entropy t0を一体化した先例、ならびにそのLIBERO未見運動での有効性は**未確認**。新しい研究ほど採択先・公開コード・後続評価が不明であり、プレプリントの著者報告を再現済みの事実とは扱わない。今回、Optimus-R／OptimusVLA／RTS等の公開実装を用いた同一条件の再現は行っていない。
