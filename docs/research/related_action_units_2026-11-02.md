# 可変長の行動単位を発見・選択・実行する研究：比較と次の実験

調査：2026-11-01 UTC。依頼名の日付に合わせて本ファイルを作成。arXiv本文・付録と一次資料の表を確認した。以下の「確認」は論文が報告した内容であり、こちらでの追試ではない。**「無注釈の可変長スキルを発見し、1つ選び、観測に応じた下位方策で実行する」は既存である。BPEによる構成はPRISEと直接重なる。残る有力な問いは、意味ラベルを使わず作った単位を、全モデル固定で追加できる記憶として扱い、未見の組合せで選択・終了・実行まで改善できるかである。**

今回の0.5秒チャンクの結果は、長い単位を試す根拠にはなる。ただし「短い窓に段階情報が絶対にない」「長くすれば選択できる」は未検証の仮説として扱う。長い単位ほど被覆と終了判定が難しくなり、選択の失敗が長時間持続する。

## 1. 「可変長」と「ラベルなし」の区別

| 区別 | 意味と本研究への含意 |
| --- | --- |
| **実行単位の可変長** | 境界または終了条件を学び、単位ごとに異なる時間だけ実行する。PRISE、LOVE、CompILEが該当。BUDS／LOTUSの教師区間も可変長だが、実行時に必ずその区間を最後まで保持する構成とは限らない |
| **固定窓内の複数トークン** | 固定長の行動列を複数コードで表す。QueSTは可変長の動作をコード列で表現できるが、主要実験は固定窓・固定間引きであり、自然な動作境界の発見器ではない |
| **符号列の可変長** | FAST等で、同じ時間幅を何トークンで表すかが変わる。これは実行する動作単位の可変長と異なる |
| **無注釈にも段階がある** | 人手の境界なし／人手のスキル名なし／タスク言語も単位形成に不使用、は別条件。CompILEの区間数、BUDSの長さ閾値、AtomSkillのVLMラベル等を明記する。「人が各軌跡を切らない」は「設計上の仮定やハイパーパラメータがない」を意味しない |

## 2. 可変長の分節化・単位発見を直接扱う研究

### 2.1 PRISE：今回のBPE案に最も直接重なる

**Zheng et al., PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control, ICML 2024.** [arXiv:2402.10450v3](https://arxiv.org/html/2402.10450v3)（2024-06-06）、§3.1–3.4、§5、App.C。

- **区切り・個数**：観測履歴と行動から、各時刻の離散コードを学ぶ。行動の復元に加え、次の潜在状態を予測する損失を使う。そのコード列にBPEを適用し、頻出する隣接ペアを順次結合する。実験の原始コード数は**10、BPE語彙は200**。各単位は原始コード列とその長さを持つ。境界・スキル名の教師は使わない。ただしLIBERO実装の観測encoderはタスク言語をFiLMで取り込むため、単位学習全体が言語不使用とは言えない。
- **選択・実行**：上位が**1個のスキルトークン**を選択し、その中の原始コードを順に実行する。各時刻の行動decoderは**その時刻の観測＋原始コード**を読む。生の運動列の単純再生ではなく、状況依存の実行がすでにある。一方、原始コードの順序は選択時に決まり、実行途中で自由に並べ替えるわけではない。
- **学習・追加**：LIBERO-90で事前学習→LIBERO-LONGの**最初の8タスク、各5実演**。適応時は上位方策とdecoderを更新する。**全パラメータ固定で未知のスキルを記憶に追加する実験は未確認**。
- **結果**：Table 3の8タスク平均SRは**52.7±0.9%**、BPEなし43.5±0.6%、事前学習ACT 42.0±3.2%。MetaWorldの5未見タスクでは66.8±1.8%（Table 1）。最高checkpointを評価結果から選ぶプロトコルであり、こちらの事前固定した最終評価と同列にしない。実機評価は未確認。
- **失敗と対処**：語彙を増やしすぎると低頻度・長い単位が過適合する。コード数が多すぎると一致列が見つからず短くなる（Fig.9–10）。長いコード列の実行誤差への対策に、下流で実行horizonを制限するハイパーパラメータを入れる。したがってBPEだけで適切な長さが無条件に決まるという結果ではない。

**差分の評価**：BPEをflow decoderへ置き換えるだけでは弱い。PRISEを重要な比較対象にし、「同一の単位形成・同一の観測履歴で、既存の状況依存decoderより何が増えるか」「新しい単位を追加したときにselectorもdecoderも更新しないで使えるか」を分離する。

### 2.2 BUDS・LOTUS：可変長の再利用と増分ライブラリは既存

| 研究・一次資料 | 区切り／単位数・長さの決定 | 選択・実行・更新 | 評価・結果と制限 |
| --- | --- | --- | --- |
| **BUDS**, Zhu et al., RA-L 2022：[2109.13841v2](https://arxiv.org/html/2109.13841v2)（2022-01-22）、§III-B/C、Tables I–IV | 画像＋固有感覚の自己教師特徴を学び、各実演の**時間的に隣接する区間を凝集クラスタリングで結合**。底の要素はW=10ステップ。木を長さ・区間数の閾値で切り、別実演の区間をspectral clustering。クラスタ上限を設定し、平均長が短いクラスを隣接スキルへ統合する。境界／名前の人手注釈なし。Kは設計・調整対象 | 上位が1個のskill indexと潜在subgoalを出し、対応するgoal-conditioned方策が観測から行動を出す。subgoalは区間内の先読み画像から学ぶ。上位4Hz、下位20Hzで、**分節化した全区間の長さをそのまま実行保証する方式ではない**。新task用meta-controller等は勾配学習 | robosuiteとFranka実機。Kitchen SR **72.0±4.0%**、BC24.4±5.3%。K=3/6/11では24.2/72.0/44.6%で粒度依存が大きい。実機**56%／50試行**、CP18%。Multitask-Kitchenでは未見の初期条件・部分目標の組合せへの再利用を調べるが、LIBERO-90→10ではない |
| **LOTUS**, Wan et al., ICRA 2024：[2311.02058v4](https://arxiv.org/html/2311.02058v4)（2024-11-23）、§IV-A/B、§V | DINOv2の区間内global pooling特徴で隣接区間を結合。基礎集合はspectral clusteringの**Kをsilhouette最大化**で選ぶ。新区間は既存partitionとのsilhouetteと閾値で既存へ割当／新規partitionを作る。境界・スキル名の注釈なし。視覚的区間の発見であり、運動だけの単位ではない | 上位が**skill index＋潜在subgoal**を生成、各skillは別の観測依存GMM方策。新しいskillの学習と既存skillの更新、meta-controller学習、experience replayを行う。**ライブラリ拡張は勾配なしではない** | LIBERO-Object/Goal各10task、LIBERO-50で継続学習。基礎task50実演、新task10実演、過去task5実演を保存。GoalのFWT/NBT/AUCは**61/30/56%**、ER53/36/47%（主結果表captionはTable I）。実機MUTEXでも評価。LIBERO-90→10ゼロショットではない |

LOTUSのsilhouetteは**画像特徴のクラスタ分離**を測る。開閉2群のように分けやすい特徴に偏っても高くできるため、運動単位としての有用性・被覆・再利用を保証しない。BUDS／LOTUSとの重要な差は「メモリ項目を増やす」と「新しい方策ネットワークを学ぶ」を切り離せるかである。

### 2.3 LOVE・CompILE：境界を潜在変数として学ぶ

| 研究・一次資料 | 境界・個数・長さ | 実行と評価 | 本研究への教訓 |
| --- | --- | --- | --- |
| **LOVE: Learning Options via Compression**, Jiang et al., 2022 preprint／ICLR 2023：[2212.04590v1](https://arxiv.org/html/2212.04590v1)（2022-12-08）、§4–7、Tables 2–6 | 潜在skillとtermination変数を学ぶ。再構成尤度を確保しつつ、**skill列の符号長＝平均区間数×利用skillのエントロピー**を減らす。初期skill数Kは設定し、利用頻度が小さいskillを除く。実験にK=10の設定があり、K=2～50の感度分析もある。各実演の区間数・最大区間長を正解から与えない。ただし最適化安定化のため訓練時に**最短3ステップ**を課す | 1個選んだoptionを観測依存方策で実行し、学んだterminationで終了。Gridworld・3D視覚navigation。境界**precision .90／recall .94／F1 .92**、VTA F1 .32、最短長付きVTA .36（Table 2）。訓練より長い5物体の新task等をRLで学ぶ。LIBERO／実機は未確認。下流はprimitive actionも選べるため、常にskillから生成するこちらの制約とは違う | **復元できるだけでは分節化は不定**という指摘が核心。最短長だけの対処は不十分。圧縮しすぎると再構成を犠牲にするので制約が必要。境界F1は評価用の正解境界で測り、訓練に正解境界は使わない |
| **CompILE**, Kipf et al., ICML 2019：[1812.01483v2](https://arxiv.org/html/1812.01483v2)（2019-05-14）、§2–4、App.A、Tables 1・3 | 微分可能な境界posterior＋区間ごとの潜在コードで軌跡を復元。Poisson型の長さpriorで極端な短区間・全軌跡1区間を抑える。**訓練時は最大区間数Mを設定**。主要実験では真の区間数3／5を与える弱い監督がある。コード容量も設定する。境界ラベル不使用と、完全に単位数も知らない設定は異なる | 学んだdecoderを低位方策にし、上位がlatent codeを選ぶ。実行時用に**未来を見ないtermination方策**を別に学ぶ。2D taskと連続reacherで3段→5段の長い組合せへ。連続制御の無境界／無ID教師版は境界F1 **74.3%→69.3%**（3段／5段、tol=0、Table 1）。99%台のz-/b-CompILEは追加監督あり | 人手の境界なしでも、区間数や長さpriorが結果を強く規定する。Mを3→5と過大指定すると3段taskのF1が74.3→47.3%（Table 3）。offlineの未来依存境界を、そのままonline終了判定に使わない |

## 3. 関連するが、主要実験は自然な可変長境界の発見ではない手法

| 研究・確認版 | 単位と長さ／個数の決定 | 利用・更新 | 評価・確認した結果 |
| --- | --- | --- | --- |
| **QueST**, Mete et al., NeurIPS 2024：[2407.15840v3](https://arxiv.org/html/2407.15840v3)（2024-09-11）、§4–5、Tables 3・4・7、Fig.2 | action-onlyのcausal conv＋masked attention＋FSQ。**T=32、間引きF=4、8コード**で表す。App.BのFSQ levels [8,5,5,5]は**1000通り**。自然境界を推定して1区間1IDにする方式ではない。論文自身も「単独コードではなくコード列が意味あるmotion」と説明 | 上位は画像・状態・taskからコード列をAR予測し、decoderで行動に戻す。通常8ステップ実行して再計画。stage Iはtaskラベル不使用。5-shotでは上位を学習し、decoder固定／更新の両方を比較 | LIBERO-90 **88.6±0.4%**、90→10の各5実演で**68.8±1.7%**、decoder固定でも**66.0±3.6%**（Table 7）。これは「decoder固定」だが**selector固定・デモ追加なしではない**。実機評価は未確認。t-SNEの共有可視化はあるが、明確な境界の正しさは示していない |
| **OPAL**, Ajay et al., ICLR 2021：[2010.13611v3](https://arxiv.org/html/2010.13611v3)（2021-05-04）、§3–5、Tables 1–4 | **固定c=10**の状態行動列を連続latentへ。標準latent次元8。単位数は離散的に発見しない | 無注釈offline経験からVAE・state-conditioned prior・低位方策を学び、上位のoffline/online RLまたはfew-shot ILに使う。段階名・報酬なしでprimitiveを学ぶが、下流学習あり | D4RL AntMaze・Kitchen、MetaWorld。antmaze-large(diverse)でCQL **14.9→70.3%**（Table 1）、few-shot BC **9.2→63.5%**（Table 2）。LIBERO／実機は未確認。長さを増やすと復元が難しくなるというトレードオフを議論 |
| **SPiRL**, Pertsch et al., CoRL 2020：[2010.11944v1](https://arxiv.org/html/2010.11944v1)（2020-10-22）、Method・Experiments | 固定時間幅のaction列を連続VAE skillへ。離散単位の個数・境界を発見する方法ではない | 状態依存skill priorで、上位RLの探索を正則化する。新taskで勾配学習。primitive名やrewardなしで事前学習可能 | navigation／robot manipulationで下流return・学習効率を評価。LIBEROの数値、今回の可変長・固定追加に対応する数値は**未確認**。ライブラリを大きくすると選択・探索が難しくなるという動機が今回に直結 |
| **SkiMo**, Shi et al., CoRL 2022：[2207.07560v2](https://arxiv.org/html/2207.07560v2)（2022-12-11）、§3–4、App. | **H=10固定、skill latent 10次元**（Table 2）。skill decoderと**実行後の状態を予測するskill dynamics**を共同学習 | 上位はskill空間でCEM／model-based RLを使う。下流でdynamics・上位を適応。境界教師なしだが可変長発見ではない | Maze・FrankaKitchen・CALVINのシミュレーション。成功・return・sample efficiencyと長期状態予測を評価。H=1/5は抽象化不足、15/20はMazeで悪化という感度分析。LIBERO／実機は未確認。**行動復元以外に「何が変わる単位か」を学ぶ**点が参考 |
| **VQ-BeT**, Lee et al., ICML 2024：[2403.03181v2](https://arxiv.org/html/2403.03181v2)（2024-06-28）、§3、Tables 1・2・5 | 1行動または**事前設定長のaction chunk**をresidual VQ。コードブック数・サイズ・窓長は設定する。粗／細コードは時間的な前半／後半ではない | Transformerが離散コードと連続offsetを予測。固定decoderの表現＋観測依存offsetで多峰性を扱う。継続的な新規単位追加は未確認 | 元論文はPushT、Kitchen等7環境と実機。実機単独task **47/50**、2task列 **19/30**（DiffusionPolicy-Tは45/50、11/30）。目標数、IoU、**行動モードのentropy**も評価。LIBERO値はQueST等の再実装と区別する |
| **SkillDiffuser**, Liang et al., CVPR 2024：[2312.11598v3](https://arxiv.org/html/2312.11598v3)（2024-03-28）、§4–5、Tables 1–5 | 画像＋task言語からVQ skillを学ぶ。**固定間隔でskillを予測**し、同じcodeを複数horizonで維持可能。実装のskill setは20、skill予測horizonは8。Table 5の分析は別途horizon10 | 選んだskill embeddingを条件として**拡散で潜在state trajectory**を生成し、inverse dynamicsでactionにする。skill境界や名前の注釈は不要だが、単位形成にtask言語を使う | MetaWorld、LOReL Sawyerのシミュレーション。LOReL平均SR **43%**（LISA40%、Table 1）、未見の多段composition **25.21%**（LISA20.89%、Table 3）。Table 5は17利用skill、1指示平均1.55skillと頻度を報告。実機／LIBEROは未確認。**「離散skill選択→拡散で適応」はすでにある** |
| **SPRINT**, Zhang et al., ICRA 2024：[2306.11886v3](https://arxiv.org/html/2306.11886v3)（2024-01-29）、§III–IV、Tables I–III | **最初に言語注釈付きsubtrajectoryが必要**。LLMで隣接区間のinstructionを要約し、offline RLで別軌跡のskillをchainingする。無注釈の境界・単位発見ではない | 長い／新しい言語taskでのpretraining集合を拡張し、下流RLへ適応。機械的な軌跡連結ではなく価値伝播を使う | ALFRED-RLと実機。実機未見物体組合せの長さ8で**SR40%、達成subgoal6.2**、language BC compositionは20%、5.2（Table I、各5試行）。少数試行に注意。長さ2/4/8で分けた評価は参考 |
| **STRAP**, Memmel et al., ICLR 2025：[2412.15182v2](https://arxiv.org/html/2412.15182v2)（2025-08-18）、§4.2–4.4、Table 1、App.A | 対象の少数実演を**手先速度が閾値より低い遷移点**等のheuristicで分節化。prior側はDINOv2／CLIP＋**subsequence DTW**で可変長区間を照合する。skill IDの離散語彙は作らない。Kは取得件数 | 取った区間を対象実演と合わせて**方策を学習**。実行時に1個検索してその中を生成する構成ではない | LIBERO-90の4500軌跡＋LIBERO-10各5実演。DINOv2 K=100でSR **45.6%**、taskごとbest Kで58.1%、BC37.9%（Table 1）。後者はK選択条件を分ける。実機あり。長すぎて複数動作が入る区間では検索が悪化した例を明記 |

QueSTの「可変長motionを表せる」という主張は否定しない。ただし**1つの可変長運動単位を選んで、その終了まで生成する**本案とは、トークンの意味と実行単位が異なる。QueST decoderを使うなら、その上に境界／macroの発見・選択が必要になる。

## 4. 2025–2026年の特に近い手法・反例

| 研究・一次資料 | 重なる点と重要な差 | 評価・使える証拠 |
| --- | --- | --- |
| **STAR**, Li et al., ICML 2025：[2506.03863v3](https://arxiv.org/html/2506.03863v3)（2026-04-07）、§3–4、Tables 1–2 | residual quantizationの**rotation-based gradient**でcodebook collapseを抑え、causal skill Transformerで合成。主要表現は固定窓のコードで、自然境界／勾配なしメモリ追加ではない。rotationはデータの物理的回転augmentationという意味ではない | LIBERO-90 **90.8%**、Long88.5%（Table 1）。Longを含む各suiteで学習した評価で、90→Longゼロショットではない。code利用率・多様性・ablationと実機あり。コード数増加だけで型が多様になるわけではないという対処例 |
| **AtomSkill: Learning Semantic Atomic Skills for Multi-Task Robotic Manipulation**, Zhu et al.：[2512.18368v2](https://arxiv.org/html/2512.18368v2)（2026-07-02）、§III-A/C、Tables I–V | **グリッパー切替で可変長分節化→VLMがskill名を付与→semantic contrastive学習**。区間は固定サイズへresampleしてVQ。拡散がskill latentを選び、観測依存decoderが即時actionと**終端keypose**を予測。keyposeへの近さで次のskillへ。**無注釈の単位発見ではない**。拡散は主にskill samplerであり、actionそのものを拡散するとの混同を避ける | RLBench 12task平均SR **74.2%**、MT-ACT68.6%（Table I）。実機single-armはATP/SR **.93/90.0%**（Table II）。境界・進捗の扱いが非常に近い。切替のない連続動作での分節化、VLMのlabel noiseを限界として挙げる |
| **BAAT: Behavior-Aligned Action Tokenization**, Dong et al.：[2609.27513v1](https://arxiv.org/html/2609.27513v1)（2026-09-23）、§III、Tables I・III・VI | **action-only Soft-DTW**で異なる実演の似たmotionを対応付け、FSQ座標をalign。予測したコードから**履歴条件付きdiffusion decoder**でactionを復元。固定Hのchunkであり、可変長境界や動的なメモリ追加ではない。具体的H／FSQ容量の全実験共通値は本調査では未確認 | 選択taskによるLIBERO-All／RoboCasa／RoboTwin平均45.2%。**Table VI：alignmentなし→ありでpolicy SR 70.2→79.0%、replay SR 80.5→71.5%、MSEも悪化**。強すぎるalignmentではSRが0まで崩れる。復元精度と閉ループの有用性が一致しない直接の例。実機あり |
| **CATok: Rethinking Causal Action Tokenization with Conditional Annealing in Flow Matching**, Zhang et al.：[2609.35469v1](https://arxiv.org/html/2609.35469v1)（2026-09-28）、§3、Tables 1–3 | **離散action token→flow-matching decoder**がすでにある。tokenは拡散時刻の粗→細の残差情報を表し、物理時間の動作単位ではない。LIBEROは**H=20、16token、codebook4096**（Table 3）。凍結tokenizerを使うが未知単位を追加するメモリではない | LIBERO平均95.9%、Long91.0%（Table 1、各suiteで訓練）。実機あり。今回の「flowで単位内を生成」だけを新規性として主張できない |
| **LexiconVLA**, Wei et al.：[2609.36774v1](https://arxiv.org/html/2609.36774v1)（2026-09-29）、§3–5、App.A.1/A.3/A.4 | **再利用可能なatomic-action codebookを検索し、共有VLAを条件付ける**点が近い。global36／detail192entry、各512次元。AtomActionの区切りには複数物理信号と規則に加え、**専門家のtask別phase数prior・曖昧な境界の手修正・18種類の人手actionラベル**が入る。全重み固定で評価するが、**評価時固定とライブラリへの新規追加は別** | 69task・57,803segmentでcodebook事前学習。policy学習18task→policy未学習8taskでBridgeVLA **16.67→34.17%**、25episode/task×3seed。8taskがcodebook事前学習にも未見と保証されるかは**未確認**。task単位CVのaction probe、perplexity、atomic SR等を測る。atomic SRはVLM判定を含むので評価者依存に注意 |
| **FAST**, Pertsch et al.：[2501.09747v1](https://arxiv.org/html/2501.09747v1)（2025-01-16）、Method | DCT係数を量子化してBPE。**固定chunk内の周波数係数の圧縮**であり、物理時間上の可変長動作境界をBPEで発見するPRISEと異なる | 大規模実機VLAで有効、π0相当の性能をより少ない訓練計算で報告。トークン数の削減や可変符号長を、そのまま意味ある運動単位の発見の証拠にしない |

## 5. 音声の単位発見から持ち込めるもの

| 研究・一次資料 | 何が発見／結合されるか | 評価と、本研究に移すときの注意 |
| --- | --- | --- |
| **HuBERT**, Hsu et al., 2021：[2106.07447v1](https://arxiv.org/html/2106.07447v1)（2021-06-14）、§III・IV | MFCCのk-means **100cluster**から始め、masked predictionで特徴を学び、次に中間層特徴を**500cluster**へ再cluster。中心はframe-wise pseudo labelと予測可能な表現で、BPEの論文ではない | phone purity、cluster purity、phone-normalized MI、下流WER。**pureなクラスタほどよい、とは限らない**：1frame1clusterならpurityは自明に100%。ラベルはこれらの評価やASR fine-tuningには使用し、自己教師事前学習の教師とは分ける |
| **HuBERT-AP: Speech Pre-training with Acoustic Piece**, Ren et al., Interspeech 2022：[2204.03240v1](https://arxiv.org/html/2204.03240v1)（2022-04-07）、§3–4、Tables 3–5 | HuBERTの500code列に**SentencePieceで頻出パターンの結合**を適用。語彙1k/2k/3k等を比較し1kを推奨。結合後のpiece IDを元frameへ再展開し、frame数を維持した学習targetにする。原文の主説明はSentencePieceであり、アルゴリズム設定を確かめず全てをBPEと同一視しない | 境界F1 **.421→.628**（Table 4）。100hラベルfine-tuning＋4gram LMのtest-other WER **8.1→7.1**（Table 5）。可変長の頻出構造を使う先例だが、**pieceが意味ある語・完全な行動に一致する保証はない** |
| **Generative Spoken Language Modeling from Raw Audio**, Lakhotia et al., TACL 2021：[2102.01192v2](https://arxiv.org/html/2102.01192v2)（2021-09-09）、§3–4 | 連続音声を離散pseudo-textにし、LMと音声decoderを学ぶ。**連続して同じIDが出る部分のdeduplication**でLMの文脈効率を上げる | ABX、bitrate、生成の品質・多様性、意味的評価等。単純dedupはロボットでは速度・保持時間を消す危険がある。durationを別に保存／予測する設計が必要 |

関連する[MonoBERT／PolyBERT, arXiv:2306.08920v1](https://arxiv.org/html/2306.08920v1)（2023-06-15）はphoneme-based pseudo unitの改善を扱うが、wav2vec-U型の**unpaired phoneme text**も用いる。「無注釈音声だけ」と読み替えてロボットへ直接対応させない。音声でのよい単位を示す中心的証拠は、見た目のcluster図だけでなく、情報量と下流課題を組み合わせた評価にある。

## 6. 汎化とMSE以外の評価：何を借りるか

| 測りたいこと | 先行例 | 本研究での測定と限界 |
| --- | --- | --- |
| **未見taskの少数実演への転移** | PRISE 90→Long8taskの5-shot、QueST 90→Long10taskの5-shot、STRAP 90＋Long各5実演 | **ゼロショットと5-shotは分ける**。実演を使ってkeyを追加するだけでも「デモなし」ではない。新taskをcodebook構築、selector学習、decoder学習のどこから除いたかを列別に記録する |
| **新しい順序・長い合成** | CompILE 3段→5段、LOVE 3物体→5物体、SkillDiffuserのLOReL多段composition、BUDSの未見初期条件 | 見たことのある単位を未見の順に使う集合を用意し、単位種類未見とは区別する。構成部分の出現有無を訓練集合で監査する。**全task成功率と段階ごとの到達率**を主指標にする |
| **継続追加と忘却** | LOTUSのFWT（新task学習後の転移）、NBT（忘却、低いほどよい）、AUC（学習過程全体） | 追加前後で旧task成功率／新task成功率、追加時間・メモリ量・更新したパラメータ数を出す。保存件数だけ増えたことを能力の継続拡張と呼ばない |
| **境界・終了の正しさ** | LOVE precision/recall/F1、CompILEのtol=0/1、HuBERT-APの音素境界F1 | 訓練で使わない小さな評価用境界集合を用意できればF1と±時間許容を報告。唯一の正解境界がない動作ではF1だけで判定せず、**早すぎる終了／終了しない／別段階へ戻る／放さない**を閉ループで数える |
| **単位の再利用・重複** | SkillDiffuser Table 5のskill利用数・instruction別頻度、VQ-BeTのmode entropy、STARのcode多様性、LexiconVLAのperplexityとtask-grouped probe | 有効語彙数、長さ分布、別episode／別taskで使われる割合、死んだ単位の割合、同一状態で単位を入れ替えた**生成結果の差**。利用率やsilhouette単独は、静止や開閉だけで稼げるので不十分 |
| **選択の正しさと実行可能性** | PRISEのcode間decoder分布の差、LexiconVLAのatomic SRとtask成功の乖離 | GTの1IDとの一致だけでなく、**実行可能な候補集合に対するtop-1精度**、選んだ単位を実行した終状態の適合、適用外選択率を測る。類似単位が複数正解ならset-valued評価にする。正解の未来をselectorの入力にしない |
| **再構成と制御の乖離** | BAAT Table VI、VQ-BeTのIoU／達成goal数／実機成功、SPRINTの達成subgoal数 | MSEは位置・回転・gripper別に補助指標として残す。主は**全体SR、達成predicate数、成功までのaction数、段階誤り率**。時間のずれにはDTWを併記できるが、DTWが失敗動作の順序を許していないか確認する |

訓練にラベルを使わないことと、評価に物体状態・成功述語・人間の誤り分類を使うことは両立する。評価用の意味ラベルを単位形成・閾値選択へ戻さない。公開SkillNetが学習済みのtaskを使う診断は**メモリ機構の診断**であり、基盤方策そのものにとっての未見task汎化とは呼ばない。

## 7. 新規性の率直な位置づけ

| 主張候補 | 判断・必要な差 |
| --- | --- |
| ラベルなしの可変長単位／1個選ぶ階層化 | **既存**。PRISE、BUDS、LOVE、CompILEが直接の先例。単位数や長さpriorの違いを明示する必要がある |
| 別episodeの頻出離散列をBPEでまとめる | **PRISEの中心そのもの**。episodeをまたいで頻度を数えるのは新規性にならない。新しい境界信号／選択可能性の目的が有効なら、その増分を検証する |
| 選択後に状況に応じて実行する | **既存**。PRISEは毎時刻の観測でdecodeし、BUDS／LOTUSはsubgoal-conditioned。生成器を使うだけでは差にならない |
| 選んだ離散単位を拡散／flowで調整する | **広い構成は既存**。SkillDiffuser、BAAT、CATokと重なる。物理的に可変長の単位、その開始・終了、履歴依存の適用条件、凍結追加の組合せまで限定する |
| スキルのライブラリが増える | **既存**。LOTUSが直接の先例。ただしskill方策等を学習・更新する。保存内容を同じ共有生成器へ渡して使えるかは差の候補 |
| **全パラメータ固定で新しい可変長単位を追加し使える** | **今回の確認範囲では新規性の候補**。未知taskで固定評価するLexiconVLAや、既存codebookを固定して上位を学ぶQueSTと区別する。新しいID embedding・selectorのclass追加・decoderの追加学習が必要なら、この主張は成立しない。BPEの新macroが既存primitiveの単なる連結か、訓練時になかった運動を扱えるのかも分ける |
| 選択と生成の分担そのもの | **古くからある**。差にするなら「選択が運動の型を決め、生成が幾何・速度・接触への適応だけを担う」を介入で示す。正しい単位／別単位／定数・maskで、行動の型と幾何調整がどう変わるか測る |

推奨する研究上の主張は、**「自己教師で形成した可変長運動単位を内容で指定できる記憶にし、単一候補を保持して閉ループ生成する。新しい記憶を追加してもselector・生成器の重みを変えず、既存単位の未見合成と新規記憶への適応を両立する」**という検証可能な組合せ。ただし、これは現時点では目標であり、先行研究に対する優位や成立が確認された結論ではない。

## 8. 失敗しやすい点と最小の対処

| 失敗 | 根拠・既存の対処 | 最初に入れる検査／対処【提案】 |
| --- | --- | --- |
| 短い単位へ潰れる／1task丸ごとになる | LOVEの尤度だけでは不定、CompILEの長さprior、BUDSの短クラス統合、PRISEの語彙・horizon制限 | 長さ分布とheld-out圧縮率を出す。最短長の強制だけを解決としない。語彙数・切り方は訓練内開発集合だけで決める |
| 開閉・静止が単位を支配する | 現在の実測。AtomSkillはgripper keyframeがない連続動作を限界とする。HuBERTのpurityの注意も同型 | translation／rotation／gripperを別々に正規化し、**gripperを固定した群内でも未来の運動・状態変化が区別できるか**測る。gripper-only、motion-only、両者のablation。開閉境界を全単位の硬い制約にしない |
| 速度・休止の違いで同じ流れが別IDになる | PRISEの厳密なcode列一致、BAATのSoft-DTW対応付け | まず時間伸縮への安定性を測る。dedupするなら継続時間を別保存。DTW整列を単位学習に足す場合も、閉ループが改善するかで判定する |
| 被覆不足／未知の必要動作が記憶にない | QueSTの限界、OPALのデータsupport、PRISEの大語彙の低頻度問題 | oracle top-1で**終状態に到達可能な記憶の被覆率**を測る。被覆が低ければrerankを続けない。task語彙を増やす前に収集・primitive表現を見直す |
| 境界が揺れる／onlineで終了できない | CompILEのoffline境界とcausal terminationの分離、BUDS上位の低周波化、AtomSkillの終端keypose | offlineの全軌跡から作る境界と、実行中の終了判定を分ける。推論には過去と現在だけを渡す。終了誤りを独立評価し、GT長・GT未来はoracleのみにする |
| 同じ単位で幾何・物体が変わると失敗 | BUDS／LOTUSのsubgoal conditioning、BAATのhistory-conditioned decoder | skill内容とscene条件を分離し、単位選択を固定したまま初期配置・速度を変える。長いraw action列をそのまま再生しない |
| 生成器が記憶を無視する／平均でごまかす | 今回の失敗、BAATのreplayとpolicyの乖離 | 同じ容量の無記憶対照に加え、記憶のswap・mask・誤単位で**型に対応する性能低下**が起きるか確認。全体MSEの小ささを通過条件にしない |
| 新規追加で語彙の意味が変わる | BPE再学習・VQ再clusterによるID変動、LOTUSのreplayの必要性 | primitive表現・既存辞書を固定し、新macroと出典をappendする。selectorは固定class headではなく**内容を読むpointer／retrieval score**として設計する。既存memoryの再parseで旧IDを壊さない。これだけで未知単位を選べる保証はなく、別に測る |

## 9. 最初の決着実験【提案。未実施】

**最初はPRISE型のBPEを選ぶ。** end-to-endの境界latentと生成器を一度に学ぶLOVE型より、発見した単位の実体・長さ・出典を監査でき、勾配なしmacro追加の検証へつなげやすい。ただし、PRISE再現を飛ばして独自のBPE効果としない。

1. **訓練内で作る**：LIBERO-90の実際のtask／episode一覧を固定し、episodeを単位形成用・開発用・最終診断用へ分割する。同一episodeの隣接部分を異なるsplitへ入れない。LIBERO-10／Skillの軌跡はまず使わない。単位形成のencoderにはaction・運動状態のみを与え、task／subtask名は与えない。これは言語も使うPRISE実装との差なので、PRISEそのままの対照も残す。
2. **境界の比較**：A=現在の10step窓、B=単純なgripper変化点、C=motion-code列へのBPE。BPEはepisode終端を越えて結合しない。1episodeの反復だけでできた単位を避けるため、別episodeでのsupportも記録する。語彙候補は訓練内開発集合で比較し、圧縮と運動・終状態の保持を両方見る。別episode数制限や正規化も設計上のpriorであり、完全自動の定数なし発見とは呼ばない。
3. **学習前の被覆診断**：長さ分布、別episode再利用、gripper群内の未来効果の分離、速度伸縮への安定性を測る。候補を1つずつ与えた場合に、進む／運ぶ／放す等の異なる状態変化を再現できるかを小規模に調べる。圧縮率・silhouetteだけが改善し、未来の効果も選択可能性も改善しなければ、生成器の長時間学習へ進まない。
4. **単位内の生成**：選んだ1個の**可変長コード列またはその内容embedding**を共有flow生成器の必須条件にし、scene・現在状態・過去履歴で低位の10step行動を生成する。**スキルの長さと低位の実行周期を分離**し、同じ単位を保持したまま生成を更新する。次単位への移行は擬似境界から学ぶcausal終了予測で行う。未来の正解状態・正解の実行長は使わない。複数候補の平均、別の補助方策への切替は入れない。
5. **選択と実行を分離**：同じ生成器についてoracle単位（上限）→実際のtop-1選択→別単位／maskを比較する。oracleでも閉ループの段階達成が上がらなければ表現・生成・終了を再検討。oracleのみ良ければselector／適用条件が課題。実選択でも改善し、単位swapで対応する型が変わる場合に、スキルが主役という証拠になる。
6. **閉ループを主にする**：同じ初期状態・seedでA/B/Cと無記憶対照をpaired評価。最初は訓練内開発task **10task×20初期状態＝200組**をscreeningに使い、全体SR、達成predicate、段階誤り、早期終了／滞留を報告する。この規模で小差を確証としない。境界・被覆の通過条件と最終のpaired SR判定は実験前に固定し、MSEや最良checkpointへの事後変更をしない。
7. **凍結追加は別の段**：encoder／selector／flow／終了予測を全固定し、訓練時に辞書へ入れなかった単位の経験をappendする。旧task、新task、未見順序で追加前後を比較する。未知macroが既存primitiveの新連結なのか、primitive自体が未知なのかを区別する。LIBERO-10を5実演使った追加は**5-shot・重み更新なし**と記し、ゼロショットとは呼ばない。

本調査ではモデル実行・コード編集・実装コストの実測はしていない。文献の主要本文・表・版履歴の証跡は [research-action-units-20261101T163753Z](/work/.skillnet-state/outputs/memory-pilot/research-action-units-20261101T163753Z/) に保存した。

## 推奨（5行）

1. **最初はPRISE型の運動コード＋BPE**。グリッパー切替は比較条件とし、唯一の硬い境界にはしない。
2. 開閉を固定した群内でも、別episodeで繰り返す**運動と終状態の変化**を表せるか、学習前に検査する。
3. 1個の単位を保持して低位flowを閉ループ更新し、単位長と10stepの実行周期を分離する。
4. 主評価はpaired閉ループSR・段階誤り・終了誤り。oracle／実選択／swapを分け、MSEは補助にする。
5. 新規性はBPEやskill条件flowではなく、**全重み固定の記憶追加と未見合成の成功**で検証する。
