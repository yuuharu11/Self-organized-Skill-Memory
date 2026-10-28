# 凍結方策に追加する運動経験メモリ：関連研究とRSS投稿判断

調査担当D／確認日：2026-10-25 UTC。
`skill_representation_review.md` と `memory_pilot.md` を読んだ上での文献調査。コードの編集、実装、学習、論文結果の再現は行っていない。

Web検索を利用できたため、以下は原則としてarXiv本文、会議論文集、出版社ページで確認した。著者のプロジェクトページは補助資料とした。自動生成レビューや論文紹介サイトを根拠にはしていない。「会議未確認」は未採択を意味しない。本文を確認できた版と、会議採択の確認は区別する。評価については著者の報告を記載しており、独立検証済みという意味ではない。

## 1. 先に結論

**現在の構想を「検索キーと運動表現を分け、複数の値を重み付き和にして凍結VLAへ渡す新手法」と説明するだけでは、新規性は弱い。** 特に **Optimus-R（2026-09）** は、分離したquery/skill表現、行動再構成で接地したskill値、top-kのsoftmax加重和、π0.5のFlow Policyへのskill token条件付け、メモリ拡張を既に組み合わせている。式の形まで近い。[Optimus-R本文 §3](https://arxiv.org/html/2609.39794v1)

**SkillMemo** は、検索キーとMoEゲート系列の値を分け、検索値をゲートへ融合する直接の競合である。**RICL、REGENT、RA-VLA** は、事前学習後に新しい実演を参照して、対象タスクでは追加の勾配更新なしに適応する先行例である。[SkillMemo §3](https://arxiv.org/html/2608.05970v1)、[RICL会議論文](https://proceedings.mlr.press/v305/sridhar25a.html)、[REGENT会議論文](https://openreview.net/pdf?id=NxyfSW6mLK)、[RA-VLA §4](https://arxiv.org/html/2608.25585v1)

残る研究仮説は、**追加時には検索器・投影・アダプタ・生成器も全て固定し、言語的なスキル注釈なしの実行済み運動区間を追記するだけで、別の状況への再利用と未見の組合せを改善できるか**である。ただし、この条件の組合せも「初」と断定できない。運動表現、検索、適用条件、時間的接続のどこに新しい仕組みがあり、それが必要なのかを示す必要がある。以下の新規性・実験判断は本調査からの提案であり、各論文やRSSの公式要件ではない。

## 2. 比較する研究案と用語

想定するメモリを `M = {(k_i, z_i, c_i)}` とする。`k_i` は状況・目的の検索キー、`z_i` は実行済み区間の運動表現、`c_i` は開始条件、到達結果、区間長、表現の版等。現在までに得られる情報からqueryを作り、`z_bar = Σ_i α_i z_i` を凍結生成器へ渡す構想を比較対象とする。

「重み更新なし」は、次の三段階を分けて記録する。

| 段階 | 固定・更新の契約 | 主張への影響 |
| --- | --- | --- |
| 導入学習 | 元のSkillNet/π0.5を固定しても、新しい運動エンコーダ、検索器、投影、アダプタを学習する可能性がある | システム全体がtraining-freeとは呼べない |
| 経験追加 | 凍結済みエンコーダで記録を符号化し、索引・統計・記憶件数だけ変更する | 本案が狙う「追加時の勾配更新なし」 |
| 制御中 | 現観測から検索し行動生成する。オンライン勾配更新はしない | 通常の推論時固定だけでは独自性にならない |

プロトタイプの残差や検索整合行列を最適化する場合も、方策本体が凍結されているだけで「追加学習なし」には当たらない。単なる平均・クラスタ集約等の非勾配の記憶管理とは区別する。

また「ラベルなし」は、少なくとも、運動名・区間境界・タスクID・言語指示・成功判定のどれを使わないかを明示する。状態・行動の記録を使う自己教師あり学習と、全く教師信号を使わない学習は同義ではない。現在のpilotはskill IDと注釈を残すため、まずは**既存モデルへの検索メモリ追加の検証**であり、ラベルなしの運動概念獲得の証拠にはならない。

## 3. 最優先で比較すべきメモリ・VLA研究

### 3.1 Optimus-R — Inline Memory Meets Reusable Skills

**精読対象**：arXiv:2609.39794v1（2026-09-30、24ページ）。2026-10-25に本文§3–4、付録C–G、Fig. 2、Tables 4–6・8–13を照合した。会議採択未確認。[版指定の書誌](https://arxiv.org/abs/2609.39794v1)、[HTML本文](https://arxiv.org/html/2609.39794v1)、[PDF](https://arxiv.org/pdf/2609.39794v1)

**差分判断の要点**：分離したkey/valueと重み付き和は既存。一方、Optimus-Rのskill値は、**現在の画像・ロボット状態・言語から作り、未来行動の予測で学習した表現**であって、**実際に実行された運動区間そのものを入力する事後エンコーダの出力ではない**。またStage Cは全パラメータ固定ではない。この二点は本案が検証できる差分候補であるが、性能優位は未実証。

#### 3.1.1 queryとskill値の生成元：層・時刻・pooling

以下は本文式(1)–(5)、付録C.1の式(14)–(15)、付録D.1・D.3による。[本文§3.1–3.2](https://arxiv.org/html/2609.39794v1#S3.SS1)、[付録C.1](https://arxiv.org/html/2609.39794v1#A3.SS1)、[付録D](https://arxiv.org/html/2609.39794v1#A4)

| 要素 | 本文・付録に記載された内容 | 読み違えてはいけない点 |
| --- | --- | --- |
| 入力 | 環境時刻tの `O_t=(I_t,x_t)`：画像とロボット状態、言語指示L、学習可能なmemory seed tokens `E_mem` | 行動列や未来観測をskill encoderへ直接入力する式ではない |
| 読み出し元 | 視覚・言語backboneの出力 `[H_t^vl; M_t]` のうち、prefixへ挿入したmemory token位置の状態 `M_t ∈ R^(m×d)`。既定m=4 | action expertのsuffix隠れ状態や、flowの特定denoising stepをpoolingする説明ではない |
| pooling | query枝とskill枝で独立したattention pooling。枝bごとに学習するquery `u_b` と行列 `W_b` から `β_i=softmax_i(u_b^T W_b M_(t,i)^T / sqrt(d))`、`m_bar_t^b=Σ_i β_i M_(t,i)^T` | 合成する軸は同一時刻のm個のmemory tokens。運動区間の複数時刻を平均する操作ではない |
| 出力 | 別々のencoder `q_t=f_q(m_bar_t^q)`、`s_t=f_s(m_bar_t^s)`。既定 `d_q=d_s=256` | 分離しているのはreadout枝。両者の入力元は同じ、言語を含む現在文脈 |
| 行動による監督 | 小さいSkill Decoderが `D_s(s_t)` から `A_t=a_(t:t+H−1)` を再構成し、Huber損失を使う | 論文はreconstructionと呼ぶが、計算経路は現在文脈から未来chunkを予測する形。記録した実運動を後から符号化する `q(z ∣ s,a,...,s',a')` とは異なる |
| 方策への接続 | Stage Aは `T_ψ(s_t)`、Stages B/C・推論は検索・補正した `T_ψ(s_tilde)` を使い、flow方策を条件付ける | Stage Aのforwardにはbankを使わない。学習時にsを作る経路と、検索後の実行経路を分ける |

**層について判明する範囲**：Fig. 2と式(1)はbackbone出力のmemory-token statesを示す。取り出すTransformerの具体的な層番号、Stage Aで解凍する「last layers」の数、encoder/pooler周辺の全実装仕様、Hの具体値は、確認したv1本文・付録では特定できない。「第何層・最終拡散時刻」と推測して補わない。環境時刻tとflow内部の生成時刻も区別する。

付録D.3では通常のmultimodal＋memory prefixをcacheし、skill tokensを追加して行動生成する実装を説明する。推論時はskill側pooler・Skill Encoder・Skill Decoderは不要で、query枝とbankを使う。式(15)のprojector出力は `R^(n_s×d)`、既定 `n_s=4`。§4.1の「per retrieved prototype」という表現だけから、top-2を必ず8 tokenへ連結するとまでは断定しない。式(9)と(15)の主経路は**先にskillを加重集約してから投影**する。

#### 3.1.2 プロトタイプの単位と注釈依存

初期bankの1件は**保持した時刻サンプル群のクラスタ**であり、1タスク、1実演、1つの明示的な運動区間と一対一に対応するものではない。本文§3.3、付録C.2の式(16)–(19)は次の構築手順を示す。[付録C.2](https://arxiv.org/html/2609.39794v1#A3.SS2)

1. Stage A後、固定したinterfaceで各サンプルの `(q_i,s_i,A_i,ρ_i)` を抽出する。`ρ_i=t_i/T_traj(i)` はエピソード全長で正規化した時刻。
2. chunk内の行動変化量 `ν_i=mean_h ||a_(i,h)−a_(i,h−1)||_2` に基づいて低変化サンプルを間引き、進捗を固定binへ分けて層化samplingする。
3. query空間でclusterを作り、skill空間の分散が大きいclusterをskill空間でさらに分割する。
4. 最終cluster `C_j` ごとに、`p_j^q=normalize(mean_i q_i)`、`p_j^s=mean_i s_i`、`n_j=|C_j|` を保存する。replay用latent `(q,s)` 対を最大3件保持する。既定初期bankは128件。

| 情報・注釈 | 利用箇所 | 「ラベルなし」との関係 |
| --- | --- | --- |
| タスクID `y_i` | Stage Aのquery対照損失の正例定義：同じtaskかつ `abs(ρ_i−ρ_j)≤δ`。既定δ=0.1 | 初期表現学習はtask labelを使う。教師なしのquery学習ではない |
| 実演ID・時刻 | 同じtrajectoryの `abs(t_i−t_j)≤w` も正例。既定w=6 | 収録の系列構造を使う |
| 正規化進捗ρ | 正例定義とbank構築時のsampling | 手で付けた意味的subtask進捗ではなく、時刻と全長から計算する値。ただし終了長を知るoffline処理が必要 |
| 言語指示L | backboneへの入力。queryとskill両方の生成元に入る | skill値も言語から独立した運動符号とは限らない |
| 行動chunk | skill decoderの監督と間引き指標 | 行動記録を使うが、値のencoder入力には直接入らない |
| 手動のskill ID・意味境界 | 構築式・正例定義には記載なし | 手動skill注釈を不要にしていることと、task/language情報を不要にすることは別 |

正例集合の出典は付録D.1の式(27)–(28)。初期clusterをtask ID別に作る制約は記載されておらず、task labelをprototypeの単位とは扱わない。また、Stage Cの追加判定そのものはtask IDやρを式に含まないが、追加されるq/sは上記の監督で作った表現であり、言語入力も使う。

#### 3.1.3 Stage C：追加手順、最適化、データ量、未報告事項

**新規entryの初期化は平均による非勾配処理だが、Stage C全体は勾配最適化を伴う。** 二つを分けて読む必要がある。[本文§3.4](https://arxiv.org/html/2609.39794v1#S3.SS4)、[付録C.4](https://arxiv.org/html/2609.39794v1#A3.SS4)、[付録D.1–D.2／Tables 5–6](https://arxiv.org/html/2609.39794v1#A4)

- **既存への割当**：整合query `q'=A_mem q` と残差込みkey `p_j^q+Δp_j^q` の最大類似度が0.75を超えると既存prototypeを使い、そのactive residualの学習、support統計、最大3件のexemplarのFIFO更新に使う。
- **新規追加**：最大類似度が0.75以下ならcandidate bufferへ入れる。定期的なclustering後、支持サンプル数が `N_min=50` 以上、query分散とskill分散がそれぞれ閾値以下のclusterだけを追加する。50は**サンプル数**であり50実演ではない。追加keyは整合query平均の正規化、valueはencoded skill平均。新残差はゼロ初期化。
- **更新対象**：query整合行列 `A_mem`、検索温度 `τ_mem`、activeなkey/value残差 `Δp_j^q, Δp_j^s`、bank構造・統計。backbone、memory tokens、両pooler/encoder、Skill Decoder、Token Projector、既存base prototypesは固定。Stage B由来のSkill Adapterも固定または不使用。
- **損失**：`L_life=L_fm+0.1||s_tilde−stopgrad(s_t)||²+10^-4 Σ_active(||Δp^q||²+||Δp^s||²)`。flow損失は現在taskの画像・状態・行動実演に計算する。latent replayはalignment項にだけ使い、current:Bに対し過去対をfloor(B/3)追加する。新:過去はおよそ3:1。旧実演のraw画像・行動を保存しないが、**新taskの行動教師は使う**。
- **学習率**：`A_mem` とactive residualsは `10^-4`、温度は `5×10^-4`。新規entryにだけ学習を限定する方式ではなく、batchで検索された既存entryの残差も更新対象になる。

| 確認項目 | v1で確認できる値・限界 |
| --- | --- |
| 実機の新taskデータ | §4.4は1taskあたり20／50／80実演の設定。Aの4taskからBの5taskへ追加する |
| 収録全体 | 付録FではAが計570実演、Bが計420実演。これを各20／50／80条件の使用件数と同一視しない。samplingの全詳細は不明 |
| 一般的な学習設定 | §4.1は8×A800、global batch 256、30,000 stepsを記載。ただし実機Stage Cのtaskごとのstep数やA/B/Cへの配分は明示されておらず、Cで一律30,000 stepsとは断定できない |
| 別の逐次学習実験 | 付録G.5のLIBERO-Spatial 10task列は**1taskあたり10,000 steps**。実機Stage Cのstep数の根拠にはならない |
| 再現に不足する設定 | candidate clusteringの実行間隔、分散閾値ε_q/ε_s、進捗bin数、初期clusterの具体的アルゴリズム等は、確認した記載から数値を確定できない |
| 全固定のまま追記するablation | **未報告**。zero residualでentryを初期化する式はあるが、以後の `A_mem/τ_mem/Δp` まで固定した性能比較は見当たらない |

`w/o Prototype Residuals` は残差更新の除去、`w/o Stage-B Bridge` はbridgeの除去であり、どちらも「全パラメータ固定・追記のみ」の統制条件とは書かれていない。通常推論時に重みを更新しないことや、学習後のzero-shot実演も、追加学習なしのStage Cを実証するものではない。

#### 3.1.4 ablation結果と、メモリ寄与について言える範囲

数値は著者報告、単位は成功率%。差はpercentage point（pp）。下表には対応するseed間誤差・信頼区間の記載がなく、数値差だけで統計的有意差とは判断しない。[§4.6／Table 4](https://arxiv.org/html/2609.39794v1#S4.SS6)、[付録G.1–G.2／Tables 9–10](https://arxiv.org/html/2609.39794v1#A7)

| Table 4の条件 | RoboTwin Hard 50% / 100%データ | 実機9task 20 / 50 / 80実演 |
| --- | --- | --- |
| Full | 21.6 / 55.0 | 33.3 / 61.7 / 72.2 |
| w/o Inline Memory：detached promptsへ置換 | 16.5 / 45.2 | 20.6 / 48.3 / 60.0 |
| Coupled Query-Skill：共通latentへ置換 | 18.2 / 48.0 | 22.8 / 50.0 / 62.8 |
| w/o Prototype Residuals | 17.0 / 46.5 | 17.8 / 46.7 / 58.3 |
| w/o Stage-B Bridge | 21.4 / 53.8 | 12.8 / 35.6 / 48.3 |

実機20実演ではFullとの差が、inline置換で12.7 pp、key/value結合で10.5 pp、残差なしで15.5 pp、bridgeなしで20.5 pp。とくにdomain shiftではbridgeの影響が大きい。ただし **w/o Inline Memoryは検索なしではなくdetached promptsへの置換**。複数部品と適応学習の比較であり、「保存した経験だけの効果が12.7 pp」とは言えない。

| 問い | 対応するablationと結果 | 解釈上の限界 |
| --- | --- | --- |
| 検索を完全に外すとどうなるか | 同一checkpointでbank経路だけをoffにする結果は**未報告** | ベースπ0.5との比較もinterface学習・適応方法等を同時に変えている |
| top-1で十分か | Table 10の `(K_0,K_r,m)=(128,1,4)`：97.0%、既定 `(128,2,4)`：97.8%。top-2の差は+0.8 pp | 少なくともこのLIBERO設定では複数検索の上積みは小さい。実機top-1比較は未報告 |
| top-4ならどうか | `(128,4,8)`：97.4%、`(128,4,1)`：96.8% | 既定からmemory token数も変えており、top-kだけの効果は分離できない |
| 重み付き和が必要か | softmax和を均等和・sampling・複数token連結・行動空間混合へ替える比較は**未報告** | top-1比較だけでは「加重和が新しい運動を合成した」とは分からない |
| clusteringが必要か | Table 9：Full 97.8%、skill-aware clusteringなし95.8%（−2.0 pp） | query clusterのskill整合を保つ設計の寄与 |
| replay exemplarが必要か | exemplar保存なし96.3%（−1.5 pp） | 過去のquery-skill対応の保持に寄与 |
| 運動に基づく間引きが必要か | 一様間引き97.2%（−0.6 pp） | ここでのuniformは検索重みの一様化ではない |
| 初期容量 | 64件97.2%、128件97.8%、256件97.8%（他は既定） | この範囲では128→256の上積みなし |

補足：Table 1のLIBERO Fullとπ0.5の差は30%データで88.6対72.1（+16.5 pp）、100%で97.8対96.9（+0.9 pp）。これは**方式全体**の差。Table 11の81→109件、検索1.80→1.90 msも逐次適応中の測定で、全固定追記のみの結果ではない。

また「外観が違っても同じ運動を再利用する」をOptimus-Rが全く扱わないとは書けない。Fig. 5は物体・容器のzero-shot変形を例示し、付録G.4／Table 12ではSpatial→Goalの共有bowl-on-plate動作が追加学習前に68%、Goal全体は8%。Goal適応後は当該task98%、全体96%、bank16→27件である。既存の選択的転用を上回る統制実験が必要。

#### 3.1.5 実機9taskと評価試行数

プラットフォームは14-DoF双腕GALAXEA R1 Lite、手首・第三者カメラ、画像224×224。以下は付録F／Table 8のtask familyを整理したもの。細かな動作順や成功条件が未記載の部分は名称から補わない。[§4.3–4.4](https://arxiv.org/html/2609.39794v1#S4.SS3)、[付録F／Table 8](https://arxiv.org/html/2609.39794v1#A6)

| 群 | 名称 | 操作・対象 | 目標物／配置先 | 評価試行数 |
| --- | --- | --- | --- | --- |
| A・単腕 | Reorientation | bottle、mugの向き変更 | 表では指定なし | cross-domain：各task20 rollout |
| A・単腕 | Block Stacking | blockの積み重ね | 表では指定なし | 同上 |
| A・単腕 | Place Object onto Plate | can、bowl、fruitの配置 | plate | 同上 |
| A・単腕 | Place Object onto Tablecloth | bowl、plate、cupの配置 | tablecloth | 同上 |
| B・双腕 | Put Objects into Container | bottle、canを容器へ | basket | 同上 |
| B・双腕 | Bimanual Basket Placement | basketの配置 | tablecloth | 同上 |
| B・双腕 | Multi-object Collection | fruit、blockの回収 | plate、tablecloth | 同上 |
| B・双腕 | Sequential Pick and Place | fruit、cupの連続配置 | bowl、tablecloth、plate | 同上 |
| B・双腕 | Bimanual Coordination Sequential Task | basket、fruitsを扱う連続協調 | tablecloth、basket | 同上 |

- **cross-domain（§4.3／Table 2）**：20／50／80**学習実演**/taskと、20**評価rollout**/taskは別。1方式・1データ条件につき9×20=180試行。Optimus-Rの平均成功率は33.3／61.7／72.2%。この180は記載からの算術であり、論文全体の総実行数ではない。
- **lifelong（§4.4／Fig. 4）**：Aの4taskを学習した後、Bの5taskへ適応し、新task成功と旧task保持・忘却を評価する。§4.4には独立した評価rollout数やseed数の再掲がないため、cross-domainの20試行を自動的に全チェックポイントへ当てはめない。80実演設定で旧task77.5%、平均忘却10.0 pp、新20実演設定で新task21.0%という報告はあるが、追加時全固定の結果ではない。

#### 3.1.6 本案が差を検証できる三つの実験

以下は**未実施の研究提案**であり、Optimus-Rの既知の失敗結果ではない。今回は文献と文書の更新のみで、学習やGPU評価は行わない。導入学習を要する比較はG0–G3後の別段階とする。`memory_pilot_gates.md` のG1で重み付き和が破綻した場合は、その失敗を隠さず、複数token等へ進む。

共通条件は、同一backbone・同じ導入データと予算、同じ実行済み運動、同じ初期状態、総保存bytesの一致。原著Stage Cの勾配あり条件と、こちらで実装する「Optimus-Rの全学習パラメータ固定・bankのみ更新」条件を分け、後者を**原著で報告されたablationとは呼ばない**。導入時と追加時の注釈要件も別々に数える。

**実験1：注釈を追加しない少量運動経験の即時登録と逐次追加**

- **設定**：同じ物体操作の別の移動方向・回転・gripperタイミングを、実行可能な運動変種として保留する。追加streamは画像・状態・実行行動・収録時刻のみとし、新task ID、意味的区間境界、進捗ラベルは渡さない。共通の汎用指示が必要なbackboneには同じ固定文を与える。同一の有限データから作る候補sample数を10／25／50／100と変え、実演数・実行step数も併記する。重複sampleを水増しして支持数を満たさない。
- **比較**：本案の区間別z追記、全固定Optimus-R＋原著の再利用／追加規則、支持数制約だけ外した全固定Optimus-R、原著Stage Cの学習あり条件。固定条件では `A_mem/τ_mem/Δp` も更新せず、学習あり条件の追加GPU時間を別計上する。
- **測定**：追加直後の新変種成功、旧task保持、最初に新entryが使われるまでの実データ量、追加時間・bytes、同じkey周辺への検索競合。適用可能な変種に限定し、固定decoderで表現不能な新物理能力を要求しない。
- **Optimus-Rが不利になるという仮説**：原著の追加には支持50sampleと二空間の低分散が必要で、類似queryは新しい運動でも既存entryへ割り当てられ得る。原著はさらに残差等の学習で適応する。本案が実運動zを1区間ずつ登録できれば、低データ・ゼロ勾配予算で先に改善する可能性がある。**支持数は50実演ではなく、通常の長い1実演でも満たし得る**ため、小データ効果を誇張しない。
- **反証**：支持数を緩めた全固定Optimus-Rも同程度なら、優位は運動表現ではなく登録規則に由来する。本案が旧taskを壊す、または登録しても改善しなければ仮説は支持されない。

**実験2：同一開始文脈から異なる運動を行った経験を、値が区別して保存できるか**

- **設定**：同一の開始画像・ロボット状態・指示から、同じ到達点へ向かう左迂回／右迂回など二つの実行済み軌道を収録する。意味的運動ラベルは学習に使わず、評価時だけ軌道形状で群分けする。まず検索を共通のoracle選択に固定し、**valueの情報量だけ**を比較する。
- **比較・測定**：Optimus-Rの開始時刻 `s_t=f_s(M_t(O_t,L))` と、本案の区間全体の状態・行動から得るzを保存する。前者のcluster前の個別sとcluster後の値を両方評価し、prototype平均の影響も分ける。固定したデコーダ／方策に左右の値を入れ替え、相対手先軌跡、回転、gripper系列の再現誤差と、出力軌道の二様性を測る。初期乱数を固定する。
- **Optimus-Rが不利になるという仮説**：式(1)–(5)の入力が同じなら、決定論的なs_tは実際の二つの未来運動を区別できない。未来actionによる監督は、追加後の実運動情報をencoder入力に与えることとは違う。本案の事後的な区間encoderには、その区別を保持できる余地がある。
- **公正性・反証**：これは値の表現能力を測る機構実験で、oracle選択の成功を検索性能として報告しない。後続のclosed-loopでは、どちらが適切か判断できる障害物・目標の観測情報をqueryへ与える。同じqueryから何の追加情報もなく正しい様式を選べるとは要求しない。Optimus-Rの区間内の別時刻sを使う条件も加え、開始時刻限定の不利かを確かめる。本案もzの加重和で左右を潰すなら、表現を区別できても最終方式の優位にはならない。

**実験3：同じ物理的運動を、外観が違う対象・背景へ全固定で転用できるか**

- **設定**：物理形状・質量・接触条件を揃えた対象で、同一の物体相対手先運動を背景・色・textureの異なるsceneへ移す。初段はcameraと幾何を固定し、視点変更や新形状は別条件にする。source外観の運動経験だけを保存し、target外観のaction実演は追加せず評価する。全ての条件を導入学習・validationから保留する。
- **比較・測定**：まず共通の幾何／自己状態keyと同じ検索候補を使ってvalueの転用性を測り、次に各方式本来のqueryでend-to-end評価する。共通keyがsimulatorの特権情報を使う場合は両方式へ同じ情報を渡し、機構評価と明記する。source/target成功、運動の類似度、検索Recall@k、外観差に伴うq/s距離、value差し替えの行動効果を測る。原著Stage Bで20実演を使う条件は追加学習ありの参照値として別表示する。
- **Optimus-Rが不利になるという仮説**：q/sの両方が画像・言語文脈由来で、task・進捗の対照損失と行動予測だけでは外観情報の除去は保証されない。本案が物体相対の実運動からzを作れば、外観に左右されにくい値になる可能性がある。原著Table 4でbridge除去の影響が実機に大きいことは着想の根拠だが、この実験の勝敗を保証しない。
- **反証**：Optimus-RにもFig. 5とTable 12の転用実績があるため、「初の外観を越えた再利用」とは主張しない。共通keyで差がなく、本来のqueryだけで差が出るなら検索器の差であり、運動valueの利点という仮説は支持されない。物理条件を変えた失敗も外観不変性の失敗と混同しない。

**この精読で絞れた主張**：主張候補は「記録された運動を値として保持すること」と「追加時の全学習パラメータ固定」が、少量追加・多様式保持・外観転用のどこで有効かという実証である。query/skill分離、softmax加重和、π0.5へのtoken条件付けを改めて新規性の中心には置かない。

### 3.2 SkillMemo — Expert-guided Skill Memory Framework

- **会議・年／一次資料**：arXiv、2026-08-06、v1。会議採択未確認。[書誌](https://arxiv.org/abs/2608.05970)、[本文 §3–4](https://arxiv.org/html/2608.05970v1)
- **保存・検索**：区間内のlatent feature重心をkey、区間全体のMoEゲート重み系列をvalueとする。現在特徴とのcosine類似度でtop-N検索。
- **利用**：検索ゲート系列を時刻整合させて平均し、現在ゲートと係数λで融合する。信頼度閾値と有限容量時の削除規則もある。
- **更新**：導入時にMoE等の学習あり。推論の検索・融合は勾配不要。全重み固定のまま追記を繰り返す性能評価は、確認した本文からは確定できない。
- **評価・実機**：LIBERO、Push-T、BlockPush、Franka Kitchen、UR5e実機。DP、複数VLA、π0.5で評価。
- **本案との差分**：valueがエキスパート数に依存するゲート系列か、共有生成器で実行する運動潜在か。潜在値に替えるだけでなく、転用性・圧縮・追加効率の利点を示す必要がある。言語的運動ラベルなしの暗黙分解も既存。

### 3.3 Stellar VLA — Continually Evolving Skill Knowledge in VLA

- **会議・年／一次資料**：初稿2025、確認版は2026-10-18のv5。会議採択未確認。[版履歴](https://arxiv.org/abs/2511.18085)、[v5本文 §3–4](https://arxiv.org/html/2511.18085v5)
- **保存・検索**：視覚・言語からVAE潜在を学び、DPMMのタスク知識空間、またはHDPのタスク・スキル階層を保持。潜在のクラスタ所属確率・中心との関係を推定する。
- **利用**：関係特徴とtop-Kクラスタ埋め込みの加重集約等を知識tokenにして、MoE action headのルータを条件付ける。
- **更新**：VAE・知識分布・方策を継続学習し、少量の経験リプレイも使う。固定方策への記憶追記のみではない。
- **評価・実機**：LIBERO-goal/long/30タスク設定、双腕実機7タスク。
- **本案との差分**：エピソードの運動潜在をそのまま追加する方式と、知識分布を更新してパラメータ共有を制御する方式の違い。**v5はスキル階層も持つため「Stellarはタスク単位だけ」とは書けない。**

### 3.4 RICL — Adding In-Context Adaptability to Pre-Trained VLA Models

- **会議・年／一次資料**：CoRL 2025。[会議ページ](https://proceedings.mlr.press/v305/sridhar25a.html)、[本文 §4–6](https://arxiv.org/html/2508.02062v1)
- **保存・検索**：実演の画像・状態・action chunk。現在の上方画像をDINOv2で埋め込み、L2距離で近傍を選ぶ。
- **利用**：複数近傍をVLAのcontextへ並べ、最寄り実演actionのtoken分布とモデル予測も距離で補間する。
- **更新**：π0-FAST-DROIDのLLMを事前にpost-trainし、画像エンコーダは固定。対象タスクでは10–20実演をbufferへ入れて勾配なし適応。対象タスクで追加fine-tuneする別設定もある。
- **評価・実機**：Franka DROID実機。未知物体、動き、別シーン等の8タスク。
- **本案との差分**：観測・行動例をcontextへ展開するか、圧縮した運動潜在を集約するか。新実演の追加だけで固定モデルを改善する発想は先行する。

### 3.5 RA-VLA — Retrieval-Augmented VLA for Test-Time Adaptation

- **会議・年／一次資料**：arXiv、2026-08、v1。会議採択未確認。[書誌](https://arxiv.org/abs/2608.25585)、[本文 §4–5](https://arxiv.org/html/2608.25585v1)
- **保存・検索**：固定窓の実演segmentを、検索keyと、キャッシュしたVLM特徴・状態・行動のvalueに分ける。queryは現在の画像・言語。行動DTWで得た対応を使って検索器を学習する。
- **利用**：複数segmentをaction headのcross-attention層に配分。検索action chunkの平均でflowの初期値も与える。
- **更新**：検索器とcontextを使うaction headを事前学習。VLMは固定。対象タスクでは少数実演を追加し、全パラメータ固定で適応する。
- **評価・実機**：訓練タスクから除外したLIBEROタスクとUR5e実機。
- **本案との差分**：コンパクトな運動潜在、明示的な運動単位、注釈依存の削減が候補。キー／値分離、追加時固定、複数経験による生成誘導は既存。

### 3.6 Retrieval-VLA — Training-Free In-Context Adaptation for VLA Models

- **会議・年／一次資料**：**CVPR 2026 Findings track**。main trackと混同しない。[IEEE出版社ページ](https://ieeexplore.ieee.org/abstract/document/11698238)
- **保存・検索**：実演の疎な物体中心keypoint表現とaction plan。現シーンをqueryとして過去実演を参照する。
- **利用**：cross-attentionによる検索で得たaction planをベースVLA出力と融合する。
- **更新**：著者はtraining-free／plug-and-play、基盤モデル更新不要と記載。ただし本調査では出版社の抄録までの確認であり、全補助モジュールの学習履歴・融合の式は**未確認**。
- **評価・実機**：抄録ではLIBERO、CALVIN、π0/OpenVLA。実機評価は**未確認**。
- **本案との差分**：幾何的なaction plan参照か、運動潜在による内部条件付けか。凍結VLAへの後付け検索という広い主張の先行例。

### 3.7 OptimusVLA — Global Prior Meets Local Consistency

- **会議・年／一次資料**：CVPR 2026。[会議ページ](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Global_Prior_Meets_Local_Consistency_Dual-Memory_Augmented_Vision-Language-Action_Model_for_CVPR_2026_paper.html)、[本文 §3](https://arxiv.org/html/2602.20200)
- **保存・検索**：Global Prior Memoryはタスク埋め込みと全軌道を保存し、現在の視覚・言語特徴からcosine top-k検索。Local Consistency Memoryは実行行動履歴を処理する。
- **利用**：検索軌道のaction blockからsoftmax加重平均・分散を作り、flow生成の初期分布にする。履歴から生成したbiasも加える。
- **更新**：VLA、検索用Prior Head、履歴モジュールを段階的に学習。推論時は固定。固定後の経験追記だけによる継続適応は未確認。
- **評価・実機**：LIBERO、CALVIN、RoboTwin 2.0、実機の汎化・長期タスク。
- **本案との差分**：行動空間の事前分布を作るか、運動潜在で条件付けるか。**複数検索結果の加重和でflowを誘導する点も既存。** Optimus-Rとは別論文。

## 4. 検索方策・in-context方策の基礎と追加研究

### 4.1 VINN — The Surprising Effectiveness of Representation Learning for Visual Imitation

- **会議・年／一次資料**：RSS 2022。[会議ページ・論文](https://www.roboticsproceedings.org/rss18/p010.html)
- **保存・検索**：実演画像の埋め込みと対応行動。現在画像の埋め込みから近傍を検索。
- **利用**：近傍行動の局所重み付き回帰で制御する。生成器内部へのtoken注入ではない。
- **更新**：視覚表現は事前学習するが、その上の制御は非パラメトリック。固定表現に実演を追加することに行動方策の勾配学習は不要。
- **評価・実機**：pushing、stacking等のデータ評価と、実機door opening。
- **本案との差分**：生の行動の加重平均か、実行可能な潜在表現を生成器に渡すか。「検索＋重み付き和＋追加学習なし」の基本要素は既存。

### 4.2 REGENT — A Retrieval-Augmented Generalist Agent That Can Act In-Context

- **会議・年／一次資料**：ICLR 2025、初稿2024。[会議論文](https://openreview.net/pdf?id=NxyfSW6mLK)、[本文 §4](https://arxiv.org/html/2412.04759)
- **保存・検索**：対象環境の実演から状態・直前報酬・行動を保持。状態空間または画像特徴空間の距離で近傍を選ぶ。
- **利用**：queryと近傍をTransformerへ入力し、その予測と最寄り行動を距離で補間する。
- **更新**：検索contextを使う一般方策を事前学習。未知環境では実演bufferのみで適応可能。fine-tune設定も別に評価。
- **評価・実機**：MetaWorld、MuJoCo、Atari、BabyAI、ProcGen。報告の中心はシミュレーション／ゲームで、実機評価なし。
- **本案との差分**：運動区間の潜在表現やVLA-MoEに限定しない。固定重みで実演を追加して適応することの強い先行例。

### 4.3 ICRT — In-Context Imitation Learning via Next-Token Prediction

- **会議・年／一次資料**：arXiv 2024。ICRA 2025の書誌あり。ただし出版社本文はアクセス制限のため会議版の差分は未確認。[arXiv](https://arxiv.org/abs/2408.15980)、[著者の方法説明](https://icrt.dev/)
- **保存・検索**：画像・自己状態・行動の実演系列をpromptとして与える。自動検索メモリは主構成ではない。
- **利用**：実演と実行履歴を因果Transformerのcontextに入れ、連続action chunkを予測する。
- **更新**：ICL能力を事前学習する。対象タスクではpromptを渡し、方策を更新しない。言語・報酬を入力しない構成。
- **評価・実機**：Franka実機、未知タスク・物体・配置。
- **本案との差分**：検索・潜在集約・容量管理は追加の論点。しかし「言語ラベルなしの実演で固定方策を条件付ける」だけでは新しくない。

### 4.4 StellaVLA — In-Context Structured Demonstration for Generalizable VLA Models

- **会議・年／一次資料**：arXiv、2026-08、v1。会議採択未確認。[書誌](https://arxiv.org/abs/2608.11671)、[本文 §3–4](https://arxiv.org/html/2608.11671v1)
- **保存・検索**：VLMで実演を計画・subgoal・keyframe・言語化した2D/3D運動へ変換。タスク指示の言語埋め込みでtop-1実演を選ぶ。
- **利用**：構造化実演をprefixへ入力。学習時はaction expertとspatial-language expertを並列に監督し、推論はaction expertのみ使う。
- **更新**：事前にモデルをfine-tune。対象タスクのcontext利用では更新しない。
- **評価・実機**：VLA-Arena、LIBERO、LIBERO-Plus、AgileX Piper実機4タスク。
- **本案との差分**：自動生成した意味・運動言語を使う方式と、注釈なしの運動潜在を使う方式。**Stellar VLAとは別の研究**。人手注釈不要でも、意味ラベルを全く使わないわけではない。

### 4.5 ICI-VLA — In-Context Imitation with Spatiotemporally Aligned Demonstrations

- **会議・年／一次資料**：arXiv、2026-09、v1。会議採択未確認。[書誌](https://arxiv.org/abs/2609.07581)、[本文 §3](https://arxiv.org/html/2609.07581v1)
- **保存・検索**：VLMがsubtaskを付けたmicro-demonstration。画像・状態・数値テキストのaction chunkを保存。意味フィルタと行動DTWで検索器の学習対を構成する。
- **利用**：現在観測から検索した例をcontextに入れ、text-action VLMが行動を生成。Target Action Maskingで単純な数値コピーへの依存を抑える。
- **更新**：検索器と方策をoffline学習し、推論時は両方固定。
- **評価・実機**：LIBERO、RoboTwin 2.0、双腕Aloha実機4タスク。
- **本案との差分**：意味ラベル付きの短い実演と、学習された運動潜在の違い。短区間化・位相整合・推論時固定も既存要素。

### 4.6 Robota-Embed — Unified VLA Embeddings for Robot ICL and Evolution

- **会議・年／一次資料**：arXiv、2026-10-14、v1。会議採択未確認。[書誌](https://arxiv.org/abs/2610.19495)、[本文 §3・5](https://arxiv.org/html/2610.19495v1)
- **保存・検索**：言語・現在画像・現在のnative actionを含むqueryと、その後4秒の動画・行動のkeyを別adapterで表現し、cosine検索する。
- **利用**：検索動画で凍結DreamZeroのvisual cacheを初期化する経路と、検索実演のContext TokensでFast-WAMを条件付ける経路。潜在加重和ではない。
- **更新**：検索埋め込みは事前学習。DreamZero経路は追加方策更新なし。Fast-WAM経路はcontextを使う学習あり。context運用自体を探索する別設定もある。
- **評価・実機**：DROID-Sim、LIBERO、ManiSkillと実機。実機は成功率だけでなくtask progressも報告。
- **本案との差分**：検索ベクトルは経験を選び、動画／tokenが実行を条件付ける。「検索表現と実行用情報の分離」も広く既存。queryに必要な現在actionの定義も公平比較で合わせる。

### 4.7 Behavior Prompting Policy（BPP）

- **会議・年／一次資料**：arXiv、2026-06。会議採択未確認。[論文](https://arxiv.org/abs/2606.30457)、[著者ページ](https://behavior-prompting.github.io/)
- **保存・検索**：ユーザーが与える単一の運動実演をbehavior promptとする。大規模bufferからの自動検索は中心ではない。
- **利用**：promptと現在観測から行動を生成するin-context visuomotor方策。
- **更新**：多様なタスクでprompt利用を事前学習。新しい実演による指定時にはfine-tune不要。
- **評価・実機**：DrawAnything、LIBERO-Gen、iPhUMIを利用した実機での実演・操作。
- **本案との差分**：経験の自動選択・圧縮・逐次保存が本案の論点。新運動を実演で指定して固定方策を使うこと自体は既存。

## 5. 検索したデータで再学習する系列

この系列の検索は主として**訓練集合の選別**である。現在観測から推論時に検索する本案との違いを明記する。検索に対象実演の行動や未来frameを使えても、オンラインqueryに同じ情報を与えてよいわけではない。

### 5.1 Behavior Retrieval — Few-Shot IL by Querying Unlabeled Datasets

- **会議・年／一次資料**：RSS 2023。[会議ページ](https://roboticsproceedings.org/rss19/p011.html)、[本文 §III](https://arxiv.org/html/2304.08742)
- **保存・検索**：ラベルなしのoffline状態・行動対。状態・行動VAEと行動を含む埋め込みを使い、少数の対象実演とのL2距離で関連transitionを選ぶ。
- **利用**：検索データと対象実演を混ぜ、behavior cloningで方策を学ぶ。推論時の記憶条件付けではない。
- **更新**：表現学習と、対象タスクの方策学習の両方あり。
- **評価・実機**：RoboSuite、PyBullet WidowX、WidowX実機等。
- **本案との差分**：データ選別か、固定方策のオンライン条件付けか。「ラベルなし経験の検索再利用」自体は先行する。

### 5.2 FlowRetrieval — Flow-Guided Data Retrieval for Few-Shot IL

- **会議・年／一次資料**：CoRL 2024（PMLR刊行年は2025）。[会議論文](https://proceedings.mlr.press/v270/lin25a.html)、[著者の方法説明](https://flow-retrieval.github.io/)
- **保存・検索**：過去データの現在／未来frame間のoptical flowをVAE潜在へ圧縮。対象実演と運動が似たデータを検索する。
- **利用**：対象・検索データで方策をco-trainし、flow予測の補助損失も用いる。
- **更新**：flow表現と対象方策を学習する。記憶追記だけではない。
- **評価・実機**：Square Assembly、LIBEROのpick-and-place、WidowXおよびFranka実機。Bridge、DROID等を利用。
- **本案との差分**：視覚的運動類似性によるデータ選択か、実行可能な運動潜在の制御利用か。外観を越えた運動検索という狙いは既存。

### 5.3 STRAP — Robot Sub-Trajectory Retrieval for Augmented Policy Learning

- **会議・年／一次資料**：ICLR 2025、初稿2024。[会議ページ](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a06e129e01e0d2ef853e9ff67b911360-Abstract-Conference.html)、[本文 §4–5](https://arxiv.org/html/2412.15182)
- **保存・検索**：過去実演と視覚基盤モデルのframe特徴。対象実演を手先速度等で分割し、特徴距離とsubsequence DTWで過去の部分軌道を選ぶ。
- **利用**：選んだ部分軌道と少数の対象実演で方策を学習する。
- **更新**：検索用基盤特徴は既成のものを使えるが、対象方策は更新する。
- **評価・実機**：LIBERO-10、DROID-Kitchen、Franka/DROID実機。大きな過去datasetからの検索も評価。
- **本案との差分**：潜在を推論条件にするか、時間整合した訓練データを選ぶか。「運動単位・部分軌道で再利用する」だけでは新しくない。

### 5.4 HSR — Hierarchical Skill Retrieval for Data-Efficient Adaptation of VLA Models

- **会議・年／一次資料**：arXiv、2026-08。会議採択未確認。[論文・本文 §IV–V](https://arxiv.org/html/2608.24042)
- **保存・検索**：過去実演、言語埋め込み、VAE行動特徴。計画をsubtaskへ分解し、言語検索後に行動特徴でrerankする。
- **利用**：検索episodeで事前学習し、対象実演と選別frameでfine-tuneする二段階のデータ利用。
- **更新**：SmolVLAのaction expert等を更新。新経験の追記のみではない。
- **評価・実機**：LIBERO-long、xArm実機の複合操作。過去データにLIBERO-90、DROID等。
- **本案との差分**：言語で指定されたスキルを対象とするデータ選別。注釈なしの運動潜在と固定生成器の研究とは問題設定が異なる。

### 5.5 SAILOR — Learning and Retrieval from Prior Data for Skill-based IL

- **会議・年／一次資料**：CoRL 2022（PMLR刊行2023）。[会議論文](https://proceedings.mlr.press/v205/nasiriany23a/nasiriany23a.pdf)、[本文](https://arxiv.org/html/2210.11435)
- **保存・検索**：過去の状態・行動区間。学習した連続skill空間で対象実演との距離を測り、関連区間を選ぶ。
- **利用**：選別区間で高位skill方策の教師を増やし、低位decoderで行動化する。
- **更新**：skillモデルと対象タスクの高位方策の学習あり。
- **評価・実機**：Franka Kitchen、CALVIN、実機長期操作。
- **本案との差分**：ラベルなしの時間拡張skillと検索は既存。検索値の直接集約と、固定後の追記による適応を切り分ける必要がある。

## 6. 潜在スキル・検索ルーティング・継続メモリの隣接研究

### 6.1 FIST — Hierarchical Few-Shot Imitation with Skill Transition Models

- **会議・年／一次資料**：ICLR 2022、初稿2021。[論文](https://arxiv.org/abs/2107.08981)、[方法 §3](https://arxiv.org/html/2107.08981)
- **保存・検索**：少数の対象実演の状態列を、学習距離で現在状態から検索する。別途、ラベルなし状態・行動区間の連続skillを学ぶ。
- **利用**：最寄り実演状態からlook-aheadした状態を目標に、逆skillモデルでzを推定してdecoderを動かす。保存zの加重和ではない。
- **更新**：初期skill学習に加え、対象実演でencoder/decoder/逆モデルをfine-tuneする標準設定。更新なしのablationもある。
- **評価・実機**：maze navigation、シミュレーションの7-DoF Kitchen。実機評価なし。
- **本案との差分**：運動潜在と検索距離の分離に関する重要な先行例。検索先の実演内未来状態は利用可能だが、実行中の真の未来状態を入力する方式ではない。

### 6.2 IsCiL — Incremental Learning of Retrievable Skills for Efficient Continual Task Adaptation

- **会議・年／一次資料**：NeurIPS 2024。[会議論文](https://papers.nips.cc/paper_files/paper/2024/file/1f0832859514e53a0e4f229fc9b3a4a2-Paper-Conference.pdf)、[方法 §3](https://arxiv.org/html/2410.22658)
- **保存・検索**：観測・言語subgoalの固定埋め込みに対する複数prototypeと、対応するskill adapterのpool。
- **利用**：prototype類似度でadapterを選び、低位方策を動かす。機能的には検索ルーティングだが、MoEゲート系列を融合する方式ではない。
- **更新**：baseを保護しつつ、skill prototypeと専用LoRA adapterを追加・学習する。
- **評価・実機**：Franka Kitchen、MetaWorld。報告範囲では実機評価なし。
- **本案との差分**：増えるのが入力経験ベクトルか、学習するadapterパラメータか。基盤の凍結とシステム全体の学習不要を混同しないための比較対象。

### 6.3 LEGION — Preserving and Combining Knowledge in Robotic Lifelong RL

- **会議・年／一次資料**：Nature Machine Intelligence、2025。[出版社論文](https://doi.org/10.1038/s42256-025-00983-2)、[著者の設計・実機説明](https://ghiara.github.io/LEGION/)
- **保存・検索**：言語意味と観測から得たタスク知識を、非パラメトリックBayesのクラスタ空間に保持し、現在タスクの潜在知識を推論する。
- **利用**：推論した知識で方策を条件付ける。生の経験近傍を直接再生するメモリでも、SkillMemo型のMoEゲート記憶でもない。
- **更新**：逐次RLで表現・知識・方策を学ぶ。凍結は学習後の実機deploymentの説明であり、新task習得全体が更新不要という意味ではない。
- **評価・実機**：MetaWorld系の逐次操作と実機・長期タスクのデモ。
- **本案との差分**：タスク分布の知識獲得と、固定生成器への運動経験追加の違い。Stellarの知識空間設計の前段として重要。

### 6.4 LOTUS — Continual IL Through Unsupervised Skill Discovery

- **会議・年／一次資料**：ICRA 2024、初稿2023。[論文](https://arxiv.org/abs/2311.02058)、[著者ページ](https://ut-austin-rpl.github.io/Lotus/)
- **保存・検索**：未分割実演から発見したskill library。open-vocabulary視覚表現で反復区間・既存skillとの関係を扱う。
- **利用**：meta-controllerがskillを組み合わせ、各skill方策が実行する。
- **更新**：既存skillの更新と新skillの追加、meta-controllerの学習を伴う。
- **評価・実機**：LIBEROを含む継続模倣と実機操作。
- **本案との差分**：ライブラリの増加や教師なしskill発見は先行する。新しいパラメトリックskillを学ぶ方式と、固定decoderへの経験追記を区別する。

### 6.5 MemoryVLA — Perceptual-Cognitive Memory in VLA Models

- **会議・年／一次資料**：ICLR 2026、初稿2025。確認版v2。[会議論文](https://openreview.net/pdf?id=54U3XHf7qq)、[本文](https://arxiv.org/html/2508.19236v2)
- **保存・検索**：エピソード中の低位視覚・高位認知tokenと時刻。現在のworking memoryをqueryとして履歴bankへcross-attentionする。
- **利用**：検索履歴と現在tokenをgateで融合し、diffusion action expertを条件付ける。類似履歴の統合で容量を管理する。
- **更新**：メモリ対応モデルを学習。推論中はtoken bankが更新されるが、別taskの新運動を追加する学習不要skill libraryが中心ではない。
- **評価・実機**：SimplerEnv、LIBERO、Mikasa-Robo、Franka/WidowX実機の一般・時間依存タスク。
- **本案との差分**：現在エピソードの履歴を覚えることと、過去エピソードの運動を再利用すること。両者は別の能力であり、必要なら併用・独立ablationする。

## 7. 本案に最も近い構成の重なり

| 主張したくなる要素 | 一次資料で確認した先行例 | 判断 |
| --- | --- | --- |
| 検索keyと実行に使うvalueを分ける | SkillMemo、Optimus-R、RA-VLA等 | 新規性の主張不可 |
| 行動再構成で接地した潜在skill値 | FIST、SAILOR、Optimus-R等 | 新規性の主張不可 |
| 複数検索値の重み付き和 | VINN、Optimus-R、OptimusVLA等 | 新規性の主張不可 |
| メモリに基づくMoEルーティング | SkillMemo、Stellar v5 | 新規性の主張不可 |
| 新実演追加後、対象taskで方策を更新しない | REGENT、RICL、RA-VLA等 | 新規性の主張不可 |
| 凍結VLAへの後付け検索 | Retrieval-VLA、RICL等 | 広い主張は不可。導入時の更新範囲も比較 |
| 注釈なし経験から運動skillを抽出 | FIST、SAILOR、LOTUS、SkillMemo等 | 人手ラベル不要だけでは不可 |
| 記憶数をexpert数や潜在次元と独立に増やす | 非パラメトリックメモリ一般、Optimus-R等 | 基本設計として既存 |
| 分離表現＋潜在加重和＋π0.5生成条件付け | Optimus-R | 本案の中心構成と直接重複 |

Optimus-Rとの比較では、少なくとも式(3)–(9)のquery/skill分離と加重検索、§3.4の更新対象を対応付ける。SkillMemoとの比較では、式(5)のゲート系列value、式(7)の平均融合に対して、運動潜在を使う必要性を示す。単に融合係数を均等平均からsoftmaxへ変えることは強い貢献にはならない。

本調査は2026-10-25時点の対象中心の調査であり、網羅的な優先権調査ではない。未確認の会議採択、実装の公開範囲、追記学習の詳細は投稿前に再確認する。特にRetrieval-VLAは抄録を越える詳細を本調査の確定事項に含めない。

## 8. 新規性を主張できる点・できない点

### 現状では主張できない点

1. 「初の検索拡張VLA」「初のskill memory」「keyとskillの初の分離」「初の複数記憶合成」「追加実演で重み更新なしに適応」。いずれも先行例がある。
2. 元SkillNetのskill IDを使った内部特徴を保存しただけで「ラベルなし運動スキルを発見した」。pilotの実装目標と最終表現研究を混同する。
3. 潜在ベクトルを増やしたことから「方策が未知の運動能力を獲得した」「無限にskillを増やせる」。固定decoderの表現可能範囲は増えない。表現外の動きには失敗し得る。
4. `Σ α_i z_i` を計算しただけで「スキル合成」。代替動作の補間、ノイズ低減、系列的な再構成は異なる。生成器がzを無視しても同じ式は書ける。
5. LIBERO-40学習済みモデルに、同40taskの別実演を保存して「モデルに未知のtaskへzero-shot転移」。示せるのはまず検索による補助効果である。

### 条件付きで主張できる候補

- **追加時の完全固定を満たす、運動中心の非パラメトリック適応**：導入学習後は全学習パラメータを固定し、注釈なし区間の追記だけで状況を越えた再利用が起きる。RICL/RA-VLAとも同じ追加データ・計算量で比較し、圧縮した運動表現の利点を示した場合に限る。
- **検索可能性と実行可能性を両立する表現学習の具体的新機構**：単にencoderを二つ置くのではなく、外観が変わっても同じ運動を選び、似た外観でも不適切な動作を避ける目的・適用条件・整合方法を提案し、Optimus-Rの分離表現を越える。
- **固定生成器のもとで成立する合成の条件と限界**：位相、姿勢、接触、gripper状態等の不整合を扱う仕組みにより、単一記憶ではできない転移や接続を実証する。無条件の線形性は仮定しない。
- **容量制約下の継続追加に対する新しい管理法と実証**：同容量で新旧skillを保ち、既存の使用頻度・古さ・信頼度に基づく削除を越える。容量管理単独はSkillMemoとの差分が必要。

これらは未検証の候補である。現在の設計記述だけでは、特にOptimus-Rとの差が「更新しない運用と別のbackboneに置き換えたもの」に留まる危険がある。運動表現の機構と反証可能な能力差を中心に据える必要がある。

## 9. RSS採択に最低限必要な比較・実験

以下は、この構想でRSSを狙うための研究上の最低線という本調査の判断。採択保証や会議公式の必須試行数ではない。

### 9.1 比較対象と公平な予算

| 優先度 | 必要な比較 | 解決する疑問 |
| --- | --- | --- |
| 必須 | 同一の凍結SkillNet、メモリ無効、同パラメータ数のadapterのみ | 検索情報による改善か、追加学習部分の改善か |
| 必須 | random／誤対応／top-1／複数均等平均／softmax加重和／候補ごとのsampling | 検索と集約が必要か、平均でmultimodalityを壊さないか |
| 必須 | SkillMemo型のゲート記憶と、本案の運動潜在記憶 | なぜゲート系列を保存するだけでは足りないか |
| 必須 | Optimus-R型のquery-skill分離・token条件付け | 中心構成の直接競合を越えるか |
| 必須 | RICLまたはRA-VLA型の実演context、VINN／retrieve-and-play | 圧縮潜在が、実演参照や単純再生を越えるか |
| 必須 | 全固定の追記と、同じ新データによるLoRA／adapter更新 | 低コストの代わりに性能を失っていないか |
| 主張に応じ必須 | Stellar型の継続学習、BR/FlowRetrieval/STRAP型の検索データ学習 | 継続学習やデータ効率の優位を主張できるか |
| 主張に応じ必須 | 同一表現の単一空間key/value、DMP、生action chunk、FIST/SAILOR型latent | 学習した運動表現と分離設計の必要性 |

異なるbackboneの論文掲載値を並べるだけでは足りない。最低でも近い機構を同じSkillNet/π0.5系へ載せた統制比較を行い、原著実装を使う比較は別に記載する。取得実演数、成功の選別、人手・VLM注釈、導入学習量、追加時GPU時間、総保存bytes、context長を合わせる。公開実装がなく再構成した場合は、原著の再現と自分の実装した近似baselineを明確に区別する。

### 9.2 データ分割と追加時固定の証明

- 導入学習、メモリ初期構築、追加stream、validation、最終評価を分離する。エピソード・初期状態・同一収録sessionの漏洩を防ぎ、隣接窓を別splitへ置かない。
- 公開重みが既に学習したtaskと、こちらの導入学習で保留したtaskを区別する。pretraining履歴が不明なら「基盤モデルに完全未知」とは断言しない。
- 追加前後で方策、encoder、query、投影、adapterの全学習tensorの一致を確認し、変えたメモリ・索引・統計を記録する。追加ごとの最適化、温度調整、task別閾値調整を隠さない。
- queryは現時点までの観測・指示・実行履歴に限定する。完了後にvalueを符号化するための過去区間全体の利用は可能だが、評価episodeの未来行動・到達状態をqueryや記憶へ混ぜない。
- 自己収集を主張するなら全試行からの採用規則を固定する。成功実演だけ、失敗区間混在、無選別の各設定を分け、成功判定の情報・人手負担も数える。

### 9.3 「運動を覚えた」を検証する主実験

1. **逐次追加曲線**：同一checkpointのまま0→少数→多数の新しい運動経験を追加し、新task成功率と旧task成功率を毎段階評価する。負の転移、検索先の変化、容量と順序への依存も測る。重み固定でも検索競合による忘却は起き得る。
2. **外観と運動の交差実験**：同じ運動／異なる物体・背景、似た外観／異なる接触・方向・gripper操作を交差させる。検索精度だけでなく、検索値を実際に使ったclosed-loop成功率を測る。
3. **未見の組合せ**：既知の部分運動を新しい順序・対象・開始状態で接続する。メモリに完成taskの実演を入れず、単一区間の再生・時間的stitching・潜在補間を区別する。単一記憶では達成できないケースを含める。
4. **記憶への因果介入**：key固定でvalueを交換、value固定でkeyを交換、zのゼロ化・シャッフル、関連／無関連記憶の差し替えを行う。改善が運動内容に由来するかを確かめる。
5. **表現能力の限界**：未知区間の再構成、別開始状態での実行、接触・障害物下での成否を分けて測る。追加だけでは改善しない動きと、decoder更新が必要な領域を示す。

### 9.4 集約とメモリのablation

- top-k、温度、区間長、固定窓／学習境界、時間・位相整合、z次元、件数、圧縮率を比較する。pilotのaction側隠れ状態と、運動区間の学習latentを別条件にする。
- `π(a|o, Σ αz)` と、各 `π(a|o,z_i)` の行動混合、候補選択、順次実行を比較する。これらは非線形decoderでは同値ではない。
- 互換性のない動作、左右の回避、gripper開閉等を混ぜ、平均が無効な中間動作になる条件を含める。重みのentropy・検索安定性・実際の成功を結び付ける。
- 生軌道・キャッシュ画像特徴・latent・索引・metadata・replay分を含む総bytesで容量を統一する。FIFO、reservoir、使用頻度／信頼度による保持、prototype集約と比較する。
- 初期構築、1件追加、検索、行動生成の時間を分離し、end-to-end遅延の中央値・p95、メモリ件数による増加、学習する場合のGPU時間も示す。

### 9.5 評価環境と統計

- LIBERO Spatial 10task各1試行の疎通確認はpilotに留める。LIBEROの複数suite、特にlongと保留した組合せに広げ、CALVINまたはRoboTwin等の性質の異なる環境を少なくとも一つ加える。
- 導入学習を伴う主条件は少なくとも3 seedを目安にし、同じ初期状態を各方式へ割り当てる。シミュレーションは主要比較でtaskごと50以上のrolloutを出発点とし、小さい改善なら試行数を増やす。seed・task単位のばらつきと95%区間を報告し、窓数を独立試行数として数えない。
- 本構想は実機でも意味があることを示したい。少なくとも複数の運動family・複合task・外観変化を含む6–10 task程度、主要比較はtaskあたり20–30試行程度を出発点とする。全条件を広く薄く回すより、**追加前後、top-1、SkillMemo型、Optimus-R型または実演context型、本案**の差を判別できる試行数を確保する。
- 部分進捗と完全成功を分け、追加経験の収集コスト、物体摂動からの復帰、失敗・不適切記憶の影響も報告する。実機で主張する能力はシミュレーションと対応させる。

### 9.6 投稿を判断する条件

**現時点のpilotだけでRSSへ進むのは厳しい。** 凍結モデルに検索経路を加えてLIBEROの既知taskが少し改善する結果は、Optimus-R／SkillMemoとの差分を支えにくい。

進める条件は、(a) 追加時に全学習パラメータ固定、(b) 意味的skill注釈を使わない運動表現、(c) 同予算の直接競合を越える別状況への転移または未見の接続、(d) 逐次追加による新旧性能と容量・遅延の実測、を揃えること。中でも **「なぜこの運動表現なら、既存のゲート記憶・実演context・query-skill prototypeでは難しい再利用ができるのか」**に答えられるかが判断点になる。

それが示せなければ、「新しいskill memory方式」という主張を広げず、固定VLAの非パラメトリック適応における表現・容量・合成の比較研究として問いを立て直す方が妥当である。

## 10. エキスパート・プリミティブの創発

追加調査日：2026-10-25。以下は一次資料で確認した先行研究と、それを踏まえた未実施の実験提案。コード・重み・実験出力は変更していない。既存のG0–G3は導入前の判定として維持し、本節の表現・エキスパート学習はその後の研究計画とする。

### 10.1 今回の構想で、何が創発するのか

想定する経路は、`PaliGemma系VLM → 状況query → 経験メモリの検索 → 連続運動表現z → 少数の共有action側エキスパートの選択・調整 → 行動`。メモリ件数Mと、共有エキスパート数Kは別の量であり、経験の追加ごとにKを増やす構想ではない。

**MoEを置いたこと、エキスパート使用率が分かれたこと、運動を人間が命名できたことだけでは、運動プリミティブの創発を示したことにならない。** 本節で狙う証拠は、意味的な運動ラベルを与えずに学んだ内部構造が、(a) 外観やタスクを越えて同じ運動的役割を担い、(b) その構造への介入で対応する行動が変わり、(c) 新しい組合せや開始条件で再利用できることである。必要なら「特徴の分化」「運動局面の分化」「独立して実行可能なskill」を段階的に主張する。

| 呼び方 | 単位 | そのまま同一視できないもの |
| --- | --- | --- |
| 層内expert | Transformerの一層のFFN、あるいはUNetの一部 | 単独で実行できる制御器ではない。expert番号も層ごとに別物 |
| action expert一式 | 複数層を一貫して使うdecoder／adapter経路 | それでも状態・ノイズ・共有経路に依存し、意味的プリミティブとは限らない |
| 時間拡張primitive／option | 複数環境stepにわたる方策と、必要なら終了条件 | 一回のdenoising計算や一つのaction tokenとは時間尺度が違う |
| 運動表現z | 実行済み区間の内容を表す連続値 | expert ID、routing重み、タスク埋め込みとは別に検証すべき対象 |
| メモリentry | 状況keyと経験value、適用条件等 | 一つのentryが一つの新expertや新しい運動概念になるわけではない |

「少数の原子的expert」を最初から一対一の意味単位に固定しすぎない。複数expertの組で一つの運動を表す分散表現もあり得る。層内MoEしか調べていないなら、まず「運動に関連した共有計算経路の分化」と呼ぶ方が正確である。

### 10.2 直接競合するMoE・複数デコーダ研究

以下の表の「弱点」は原著の否定ではなく、本案の創発主張を判断する上で証拠が届いていない範囲である。arXivのみ記載したものは、本調査で会議採択を確認できていない。

| 論文・年／一次資料 | expert・primitiveの単位と分化の方法 | 原著の根拠 | 本案との関係・評価上の限界 |
| --- | --- | --- | --- |
| **Emergent Compositional Skills in Mixture-of-Experts VLAs**：arXiv 2026、確認版v2。[本文§2–3](https://arxiv.org/html/2607.20771v2) | π0／SmolVLAのaction FFNにrank-16 LoRA expertを設け、全action層で同じrouting選択を共有。flow模倣＋load balancingで学習し、事前のsubtask分解を与えない | LIBEROで異なるタスク間の運搬・把持・release/retract、使用頻度、系列を可視化。**同じchunkでrouter割当を手動変更**し、把持等の行動変化や回復を例示。denseと同程度のtask性能 | 最も直接的な先行例。「MoEから再利用可能な運動役割が創発」「手動routing介入」は既存。ただし言語指示を入力し、主証拠は選んだrolloutの定性評価。原著自身、task固有・解釈困難なexpertと意味対応の不完全さを認める。実機検証は記載なし |
| **Di-BM / Learning Diverse Skills for Behavior Models with MoE**：arXiv 2026。[§4・6.4–6.6](https://arxiv.org/html/2601.12397v1) | Diffusion PolicyのCNN／FFNのexpertと、expert別観測分布を表すEBMを共同学習。観測領域の重複・被覆を目的で制御し、primitiveラベルを直接与えない | RoboTwinとNova5実機。task横断の並進・alignment局面のrouting曲線、t-SNE、単一expert固定と全expert切替の比較、vanilla/task-wise MoEとの比較、新taskへのfine-tune | 既に「task別でなく局面別の分化」を主張。視覚・言語特徴を使う。単一expertで全taskを実行する性能低下は切替の必要性を示すが、対応する局面だけを壊した因果実験ではない。新task適応は追記のみではない |
| **Sparse Diffusion Policy（SDP）**：CoRL 2024、刊行2025。[会議](https://proceedings.mlr.press/v270/wang25c.html)、[§3–4](https://arxiv.org/html/2407.01531) | 層内FFNをMoE化。**task別router**を使い、層ごとのtask Tとexpert Eの相互情報量 `I(T;E)` を最大化。skillを複数層のtop-k経路の組合せとして説明 | MimicGen・DexArt・Adroit、FANUC実機。task×expert頻度図、Coffee系の局面とexpert score、転移・MI損失ablation | taskラベルを使うため本案の「taskラベルなし」の証拠ではない。共有expert使用と転移は参考になるが、同じexpertが同じ運動効果を生むことまでは頻度図だけで証明できない。転移実験はrouterに加えvision encoderもfine-tuneする記述がある |
| **MoDE / Mixture-of-Denoising Experts**：ICLR 2025。[会議](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2bfc4acb2e5c4af1576d14b0df9a664c-Abstract-Conference.html)、[§3・付録](https://arxiv.org/html/2412.12953) | 拡散TransformerのMLP expert。routerは主に**noise level σだけ**で選び、denoising局面ごとに分業。load balancingを併用 | CALVIN／LIBERO。σ×層×expert使用図、routing・top-k・負荷分散のablation、cacheによる計算効率 | これはdenoising専門化であり、環境中の把持・運搬などの分化と同じではない。expert入力自体は文脈依存でも、routingの変化をそのまま運動skillの創発と解釈できない |
| **KinRT / Route by Kinematics, Act by Observation**：arXiv 2026。[§3・ablation](https://arxiv.org/html/2607.26807v1) | 未来H=50のactionと時間差分を連結・標準化・PCAし、frameごとにK-means。得た運動cluster IDでrouterを監督し、推論は視覚・言語だけを使う | RoboTwinとDIYRobot実機。VLM／task ID／action／velocityによるclusteringの比較、運動類似度分布、希少運動への性能 | **運動から作る擬似ラベルは人手のtask・言語ラベルではない**。本案が自己教師ありclusteringを使うなら強い比較対象。「kinematics-supervised」と呼ぶだけで本案から除外できない。cluster内の運動整合性は、その特徴でclusterを作ったことからも生じるため、別の実行指標が必要 |
| **SMoDP / Semantically Structured MoE for Compositional Robotic Manipulation**：arXiv 2026。[方法・Fig. 9](https://arxiv.org/html/2605.23477v1) | VLMが実演をverb–noun skillと境界で注釈し、skill predictorと意味的対照学習でchunkのroutingを構造化 | LIBEROとALOHA実機。意味対phase ID、注釈noise・境界ずれ、異なる物体の同じ局面でのrouting heatmap、転移 | **人手注釈不要でも意味ラベルは使う**。学習に使った言語skillとの整合性を示す図は、意味監督の効果であってラベルなし創発の証拠ではない。外観を越える共有の比較には重要 |
| **MoS-VLA**：arXiv 2025。[§4–5](https://arxiv.org/html/2510.16617) | OpenVLAの共有body＋複数action出力headをbasis関数とし、連続係数の線形結合で方策を表す。新実演からL1誤差の線形計画で係数を校正 | OXEの保留dataset、sim・実機、新contextの予測改善。係数の低次元可視化はlab条件ごとのclusterを示す | 「共有basisを連続係数で調整」は先行する。ただしbasisごとの運動意味を確立した研究ではなく、係数は主にcontext適応。言語とdataset/contextのまとまりを使い、追加時には係数の最適化を行う |

Emergent Compositional Skills v2はroutingについて「一forwardで全層共有」「推論は一chunk」と書く一方、図captionではnoised action・flow時刻も挙げる。全denoising stepで一つのexpertを固定するかは記述が十分統一されていないため、再現比較時には選択の時間尺度を実装と照合する。本案も、**環境時刻・chunk内位置・flow時刻・Transformer層**を別々に記録する必要がある。

### 10.3 制御・スキル発見の先行例：MoE以外も外せない

#### 10.3.1 制御器・option・DMPを単位にする研究

| 論文・会議／一次資料 | 単位・発見条件 | 分化・再利用の根拠 | 本案への示唆と限界 |
| --- | --- | --- | --- |
| **MCP**：NeurIPS 2019。[会議](https://papers.nips.cc/paper_files/paper/2019/hash/95192c98732387165bf8e396c0f2dad2-Abstract.html)、[本文§3・5](https://arxiv.org/html/1905.09808) | 状態依存Gaussian primitive方策を重み付き**分布の積**で同時合成。primitiveには状態だけ、gateには状態＋goalを与える非対称設計。motion imitationで共同学習 | simulated humanoid等。歩行位相とgateの対応、primitive出力のPCA、複数runでの分化、primitiveを凍結して新gateを学ぶ転移、単一primitive選択との比較 | 共有primitiveの連続調整・合成と非対称な情報制限は既存。primitiveの意味IDは手で与えないが、参照motionと報酬は使う。位相相関・出力clusterだけでtask横断の因果的役割が一意に決まるわけではない。積による制御とz平均・FFN平均は別演算 |
| **Option-Critic**：AAAI 2017。[会議](https://ojs.aaai.org/index.php/AAAI/article/view/10916)、[本文](https://arxiv.org/html/1609.05140) | intra-option方策と終了関数をtask rewardから同時学習。手動のsubgoalを指定しない | four-roomsのdoorway付近の終了確率、Pinballの軌跡、Seaquestの上昇／下降option、goal変更への適応 | 複数環境stepのまとまりと終了条件を評価する手本。**task rewardあり**で、ラベルなしの運動経験モデルとは異なる。終了が短縮する退化に正則化を用い、自然な時間単位は自動保証されない |
| **DIAYN**：ICLR 2019。[論文](https://arxiv.org/abs/1802.06070)、[本文§3–4](https://arxiv.org/html/1802.06070) | sampled skill変数zで条件付けた方策。task rewardなしに状態からzを識別する相互情報量と行動entropyを使う | simulated locomotionの多様な動作、固定zでのrollout、下流RL・階層的組合せへの利用 | zは外部の意味ラベルではないが、「識別できる」だけなら静的姿勢・居場所でも達成できる。**分かれた到達状態と、転用可能な運動プリミティブは同義ではない**。別々のexpert networkを必要としない |
| **DADS**：ICLR 2020。[会議論文](https://openreview.net/pdf?id=HJgLZR4KvH)、[本文](https://arxiv.org/html/1907.01657) | `I(S';Z ∣ S)` に基づき、skillごとの状態遷移を多様かつ予測可能にする。skill dynamicsも学ぶ | simulated robotで固定zの運動、連続zの補間、学んだdynamicsを用いたgoalへの計画・組合せ | 状態そのものより「何を変化させるか」を評価する重要な参照。ただし選ぶ状態成分・探索範囲に左右され、接触を含む意味的skill発見が自動保証されるわけではない |
| **Di-SkilL**：ICML 2024。[会議](https://proceedings.mlr.press/v235/celik24a.html)、[本文](https://arxiv.org/html/2403.06966v2) | expertごとにcontext依存のmotion primitiveパラメータ分布とEBMのcontext分布を学ぶ。実装はProDMP。最大entropyと自動curriculumで同じ問題の複数解を狙う | simulated table tennisや障害物を含む操作。context被覆・多様な経路と成功率、curriculumやexpert構成の比較 | expertが機能するtrajectory生成器なのでFFNより単位が明確。ただしcontext・task rewardを与えるRLであり、受動的なラベルなし運動区間の分解とは異なる |
| **MoMP / Learning to Select and Generalize Striking Movements in Robot Table Tennis**：IJRR 2013。[著者版本文](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf) | 人が教示した打撃運動をDMPとしてlibrary化。ball・robot状態と打撃条件に基づくgateで選択・一般化 | 実機で未教示のincoming ballや人間相手の卓球への一般化 | 「少数運動の選択と調整」は長い先行研究を持つ。教示された打撃templateを単位にしており、未分割データからprimitiveが創発した証拠とはしない。DMP内部のGaussian基底と、複数DMPを混ぜるexpertも別物 |

#### 10.3.2 離散コード・動画区間・複数skill方策を単位にする研究

| 論文・会議／一次資料 | 単位 | 分化・再利用の根拠 | 限界・比較する意味 |
| --- | --- | --- | --- |
| **VQ-BeT / Behavior Generation with Latent Actions**：ICML 2024。[会議](https://proceedings.mlr.press/v235/lee24y.html)、[本文§3–4・付録](https://arxiv.org/html/2403.03181) | actionまたはchunkのResidual VQ code。primary/secondary codeを共有decoderで行動化 | modeの被覆・実行taskのentropy、decoded centroid、codebook構成の比較。simとStretch実機 | codeを量子化したこと自体が意味的primitive発見ではない。task達成の多様性は個々のcodeの因果的役割とは別。**codebook entryはexpert networkではない**が、連続zの必要性を測る比較対象 |
| **PRISE**：ICML 2024。[会議](https://proceedings.mlr.press/v235/zheng24b.html)、[本文§3・5](https://arxiv.org/html/2402.10450) | 状態依存のaction量子化列からBPEで学ぶ可変長skill token | LIBERO／MetaWorldでの共有・few-shot改善、BPE除去、code数、token長分布の比較 | 高頻度列の圧縮は明確な構成原理だが、頻出する停止や収録癖も選ばれ得る。**可変長だから意味的境界を発見した**とは限らない。固定窓zへの強い比較 |
| **XSkill**：CoRL 2023。[会議](https://proceedings.mlr.press/v229/xu23a.html)、[方法・評価](https://arxiv.org/html/2307.09955) | 未分割・未整合の人間／robot動画clipから、共有prototypeと時間的表現を学ぶ。共有の条件付きdiffusion方策へ接続 | human/robotのt-SNE、時間を揃えたprototype列、未見skill順序とcross-embodiment実行、prototype数・目的関数の比較。sim・実機 | 既に言語skillラベルなしの発見とdiffusion条件付けを扱う。原著は同じlab camera setupに評価が限られると明記。物体やsceneで分かれたprototypeを運動概念と同一視しない。prototype数とexpert数は別 |
| **LOTUS**：ICRA 2024。[論文](https://arxiv.org/abs/2311.02058)、[方法§IV・評価§V](https://arxiv.org/html/2311.02058) | DINOv2の視覚的な時間区間を階層clusteringし、反復区間をskill partitionへまとめる。各partitionにgoal条件付き方策を学ぶ | Silhouetteによる分割・追加、task横断の共有局面図、新skill追加なしablation、FWT/NBT/AUC、LIBEROとMUTEX実機 | 意味的skill境界は人手で指定しないが、高位meta-controllerはtask言語を使う。視覚clusterのまとまりと運動の同一性は別。skill方策も追加・更新するため、K固定の本案とは異なる |

FIST／SAILORの連続skill表現は§5.5・6.1も参照。これらも含め、**人手の運動ラベルなしで区間を潜在化すること、共有制御器を条件付けること、それを再利用することは既存**である。新規性の検討では「MoE」という名称の有無だけで比較対象を狭めない。

### 10.4 「創発」を調べる評価と、その弱点

| 評価 | 何が分かるか／先行例 | 単独では不足する理由 | 本案での使い方 |
| --- | --- | --- | --- |
| task×expert使用率、entropy、負荷分散 | SDP、MoDE等。collapseや共有計算の有無 | taskラベルを識別しただけ、乱数で均等分割しただけでも良い値になる。使用頻度とexpertの機能は別 | collapseの診断に留め、運動効果との関係を別に測る。真の運動頻度が偏る場合に完全一様を要求しない |
| 相互情報量・識別probe | SDPの `I(Task;Expert)`、DIAYNのskill–state、DADSのskill–transition | MIの相手で意味が変わる。`I(z;gate)` は設計上高くても当然。学習に使った目的の高さだけでは独立した証拠にならない | 未使用episodeでexpertから**相対変位・回転・接触・gripper変化**を予測し、同容量の外観／task probeと比較 |
| routingと意味的phaseの整合 | SkillMemo、SMoDP、MCP、Emergent系の曲線 | 人が後から都合よくphaseを付ける選択、task・進捗・ノイズ時刻による交絡。意味監督を使った学習なら循環的な評価にもなる | 評価用のphase定義と集計を先に固定し、全episodeを集計。学習に評価ラベルを戻さない |
| t-SNE・PCA・cluster指標 | MCP、XSkill、MoS-VLA、LOTUS等 | 射影・縮尺・近傍設定で見え方が変わり、外観・場所・labで分かれてもよい。Silhouetteの最大化だけでは運動単位と断定できない | 補助図に留め、元空間の保留データでの予測・転用を主指標にする |
| expert固定・除去・router差し替え | Di-BMの単一expert、Emergent系のmanual routing | 除去による性能低下は単なる容量低下・分布外入力でも起きる。数例の手動成功には選択biasがある | 状態・生成noiseを固定し、active/inactive expert、同normの摂動、同計算量の代替routeを対照にする。局面別の予測された変化を集計 |
| task横断の転移、primitiveを固定した再利用 | MCP、SDP、XSkill、LOTUS等 | 下流のrouter・視覚・decoderの学習が補っている可能性。既知taskの別物体だけでは強い組合せ汎化とは限らない | 何を固定したかを列挙し、同じ運動／別taskと別運動／同じ外観を交差。完成taskの軌道を保留する |
| 未見の接続・同時合成・潜在補間 | MCP、DADS、PRISE等 | 線形に動くだけでは意味的合成ではない。単純な軌道平均・動作再生でも達成できる設定がある | 「順序の接続」「同時の制約充足」「同じprimitive内の振幅・向きの調整」を別々に評価 |

本案では運動をB、task・外観・進捗・生成noise・層などの交絡要因をN、expert選択をEとすると、`I(E;B ∣ N)` やそれに相当する保留データprobeは有用な候補。ただし、**taskと運動が完全に対応するdatasetでは両者を識別できない**。同じ運動を異なるtaskに、異なる運動を同じtask・外観に含めた実験が先に必要である。状態を条件に入れすぎて運動の情報まで除く場合にも注意し、単に条件付きMIを一つ報告して済ませない。

MI・probe・境界精度は評価用の自動kinematicsや少量の独立annotationを使ってよいが、「学習ラベルなし」と「評価ラベルなし」を混同しない。モデル選択に使わない最終保留集合で、episode単位のbootstrap／seed間変動、対応を崩すpermutation対照を出す。隣接窓を独立試行として数えない。

### 10.5 SkillMemoとの差を保つためのzとexpertの設計

SkillMemoはkeyに区間特徴の重心、valueに**MoEゲート系列**を保存し、検索したgateを現在gateへ融合する。原著はexpert activationの時間変化、意味的局面との対応、PID由来の情報量目的、性能ablationを根拠に暗黙のatomic skillを主張する。ただし目的関数やrouting曲線だけで、本案が欲しい運動的意味・独立再利用・因果性が保証されるわけではない。[SkillMemo §3–4](https://arxiv.org/html/2608.05970v1)

**保存値を「z」と呼び替えるだけでは差分は残らない。** 次を設計・評価の契約とする。

1. **zが何を記憶するかを、gateから独立して決める。** 実行済み区間の相対姿勢・行動・gripper・接触等から符号化し、保留した運動内容や結果を読み出せるかを検証する。zをgate系列の圧縮・蒸留だけで学ぶと、機能的にはSkillMemoの低次元化に留まる。action再構成だけも既存であり、それだけで差分を主張しない。
2. **「選択」と「調整」の実効自由度を分ける。** gateだけを変えるのか、選んだexpertへ連続的な振幅・方向・位相・接触条件等も与えるのかを明示する。例えば `g(z,h,σ)` と、expert内部の条件 `m(z)` は役割が違う。FiLM等の条件付けを加えること自体は新規ではなく、同じexpertを別の状況で使える効果が必要。
3. **top-1でも調整できると仮定しない。** 選択後に重みを正規化して1にする実装なら、gate-only経路には選択以外の連続自由度が残らないことがある。shared expertとの混合重み等は実装に依存する。現在のSkillNetについては共通gate文書どおりskill経路がrouter logitsだけに作用するため、現経路の感度確認と、運動生成の連続調整ができるという主張を分ける。
4. **情報の抜け道を検査する。** 上位VLM・共有expert・通常の観測だけで動作を解き、z経路を無視する退化がある。逆にexpertがtask言語を直接読んで全動作を担当すると、zは意味的task selectorになる可能性がある。MCPの非対称入力は参考になるが、入力を制限した効果と新規性は別に測る。
5. **二つの時刻を混ぜない。** 環境時間の把持→運搬と、生成中の高noise→低noiseを分ける。同じ環境状態・zでσだけを変える対照と、同じσで運動だけ変える対照を置き、MoDE型の分業だけではないと示す。expert番号は層をまたいで同一意味とは仮定しない。
6. **メモリとprimitiveの容量を別に評価する。** Kを固定してMだけ増やす。Mの追加で既存primitiveの選び方・調整・組合せは変わり得るが、固定したexpert自体が新しく学習・誕生したとは呼ばない。未表現の力学・接触を記憶追加だけで獲得できる保証もない。

この区別を直接検査する最低限の対照は、(a) SkillMemo型のgate系列保存、(b) gate系列を同容量の連続zへ圧縮する方式、(c) 運動から作ったzだがgate-only、(d) zから選択と内部調整を行う方式。同じ記憶bytes・導入学習量・expert数で比較する。(d)が(c)を越えず、zの情報もgateだけで説明できるなら、「運動表現による選択と調整」という独自の主張は弱い。

### 10.6 タスク・言語ラベルなしという条件の置き方

意味的primitiveを人手で与えないことと、**task/languageを一切使わないこと**は別条件である。上位PaliGemmaに指示を与え、その出力や損失勾配がmotor表現・expertへ入るなら、運動名の注釈がなくても言語を使った学習である。事前学習済みVLMも意味知識を持つため、システム全体を「意味的事前知識なし」とは呼べない。

| 段階 | 本案で主張を明確にする条件 |
| --- | --- |
| 運動表現・primitiveの発見 | 区間の状態・行動・時系列を用い、task ID、指示、VLM生成のverb–nounラベル・境界を監督に使わない。物理的変換・圧縮・未来予測等の自己教師信号は使用可能。どのbranchがVLM特徴を読むかも開示 |
| 状況検索・高位の目的指定 | 目的を指定する言語を使う場合は「検索／実行時の目的指定」として別に記載。運動encoderへ指示を流す条件、流さない条件、同じ運動に指示だけ変更する条件を比較 |
| 経験追記 | 導入済みのencoder・検索器・expert・調整器を全固定し、実行済み区間のkey/valueと管理情報だけ追加。新しいsemanticラベルの生成を要求しない |
| 評価 | task名・kinematic class・接触イベントは評価のためだけに使い、発見用の訓練・閾値選択・最終モデル選択へ逆流させない |

KinRTのような行動由来のcluster IDやVQ codeは**自己教師ラベル**であり、それを使っただけで「タスク・言語ラベルなし」という条件から外れるわけではない。差分を「自分はラベルなし、相手は教師あり」と言葉だけで処理せず、何が実データから自動的に作られるかを揃える。

また、現在の公開SkillNetをそのまま凍結して調べる場合、既存skill ID・注釈を使って獲得したexpertを後から発見・解釈している。**その実験だけでは、今回ラベルなしの学習によってexpertが創発したとは主張できない。** 導入段階でprimitiveをどう学んだかと、固定後の検索・追記による再利用を別の結果として出す必要がある。

### 10.7 創発を説得力をもって示すための最低限の実験

以下は採択の公式要件ではなく、本案の中心主張を支えるための実験案。G0–G2の感度・補間・kNNは入口であり、以下の証拠を代替しない。

**実験A：運動とtask・外観・進捗を交差させた発見評価。**

同じ相対移動・回転・gripper操作・接触遷移を、異なるtask・物体・背景・開始位置で収録する。同じscene・taskにも複数の運動を含める。少なくとも3種類以上の物理的運動familyと、各familyが複数taskに現れる構成を作り、収録episode、物体／外観、運動の組合せを保留する。学習には意味的運動ラベルを渡さず、評価用のkinematic基準は先に決める。

評価するのは、元空間でのzによる運動の近傍再現、expert使用と運動・接触の関係、task・外観・σを揃えた場合の保持。同じ開始状態に対する複数運動を扱い、expertが単なる位置binやphase番号でないかを見る。失敗例も含む全rolloutの集計と、最低3 seedを目安とした再現性を示す。random/frozen routing、σ-only routing、task/言語監督ありroutingを対照にする。

**実験B：z・gate・expertの役割を分離した因果介入。**

現在観測・初期noise・状態・通常のVLM条件を固定し、zだけ交換、gateだけ固定／交換、expert出力の除去／代替を別々に行う。例えば、同じ到達系expertを選び続ける範囲でzを変え、移動方向・距離・速度・手首回転が予想どおり変わるかを測る。接触やgripperのように連続補間が不適切な変数は別扱いにする。

「調整」の最も明確な対照は**gateの値まで固定したままzを変える**条件。行動が変わるのは内部条件付け経路に因果効果がある証拠になる。gate-only実装では変化しないのが正しく、その場合は調整の意味をgate混合へ限定する。expert介入は同じ運動局面を複数taskで再現し、active/inactive除去、normを揃えた摂動、同計算量の代替を対照とする。単なる性能低下ではなく、運搬だけ・閉gripperだけ等の**事前に予想した局面の変化**を成功率とkinematicsで測る。Emergent系の定性介入を、選別しない多試行へ拡張する。

**実験C：固定primitiveのtask横断再利用と未見の接続。**

K固定のまま、Aで使った運動を異なる物体・配置のBに利用し、さらに訓練にない順序で組み合わせる。完成taskの実演をメモリへ入れず、部品の経験だけを登録する。新taskでexpert／decoderを学習する比較と、全固定で検索・追記だけの比較を分ける。境界での姿勢・速度・接触整合、部分成功と完全成功、単純な軌道再生との違いを測る。

同一expertが複数taskで使われるだけでなく、**同じ介入が同じ運動的役割を変える**ことまで確認する。成功が単一の巨大な共有branchに由来するなら、primitive再利用の証拠にはならない。skill維持時間・終了条件、固定窓と可変境界の違いも記録する。固定窓だけで十分なら「action-chunk表現」と呼び、意味的境界の自動発見まで広げない。

**実験D：必要な分解かどうかを示す統制比較。**

| 比較条件 | 問い |
| --- | --- |
| dense decoder＋同じz、active FLOPsを揃えたdenseと総容量を揃えたdense | MoE自体が必要か、単なる容量・計算量の違いか。両予算の一致を同時に仮定しない |
| random/fixed routing、σ-only、task/language監督、kinematics cluster監督 | 分化の源が運動なのか、noise・意味ラベル・擬似ラベルなのか |
| 同じzの単一decoder、複数完全decoder、層内MoE、全層で一貫したexpert経路 | どの粒度の分解が役立つか。FFN番号と実行可能なprimitiveの違い |
| K=1／2／4／8等と各条件のseed、未使用expert率、実行性能 | Kを増やす効果、collapse、過分割、最小の共有容量 |
| 運動encoderの代わりにraw chunk・DMP・VQ/PRISE型表現、zなし・シャッフルz | 学習した連続運動表現が必要か |
| SkillMemo、gate圧縮z、gate-only z、選択＋調整z | gate系列の記憶を越えた効果があるか |
| メモリなし／random／top-1／複数検索、同じ記憶bytes | expertの学習効果と、経験メモリの効果を分離できるか |

全てを最初に実装する必要はないが、中心主張に対応する対照を欠いたまま、routing可視化だけで創発を結論しない。最重要の近接比較はEmergent Compositional Skills、Di-BM、KinRT、SkillMemoであり、意味監督ありの参照としてSDP／SMoDP、表現の参照としてXSkill／PRISE／VQ-BeTを置く。原著実装がない場合、同じbackboneへ移した自作baselineと原著の再現を区別する。

**実験E：Mだけを増やす全固定の逐次追加。**

運動encoder・query・expert・router・調整器を固定して、経験だけを段階的に追加する。Kは一定とし、新しい組合せの成功、旧task保持、適用外運動の拒否・失敗、bytes・追加時間・検索遅延を同時に示す。評価episodeの未来をqueryへ使わず、実行済み区間だけを登録する。単なる窓の重複追加ではなく、新しい開始条件・運動変種を追加する。

ここで改善したものは「固定primitiveの新しい利用・調整・組合せ」である。**primitiveが創発した証拠は導入学習の実験A–D、メモリ追記の価値は実験E**として二つを結ぶ。全固定でも、検索競合により旧能力が失われる可能性を測る。実機でも、別task・別外観で同じ運動的役割が再現する少数の主要比較を、多試行で確認する。

### 10.8 この構想の新規性についての判断

「MoEにするとprimitiveが自然に分かれる」「少数のbasisを連続係数で調整する」「人手のskill注釈なしでスキルを見つける」は、それぞれ明確な先行例がある。とくに2026年のEmergent Compositional SkillsとDi-BMを確認した以上、**MoEからの運動役割の創発そのものを初の貢献にはできない**。

残る主張候補は、task・言語を監督にしない運動経験から、外観やtaskを越えて再利用できる役割と連続的な調整を獲得し、その運動表現を少数の共有expertへ接続したまま、固定後は経験追加だけで利用範囲を広げられること。そのためには、(1) ラベル・情報経路の開示、(2) 運動に対する定量的な分化、(3) 交絡を抑えた介入、(4) 保留した状況・接続での実行、(5) メモリとexpert容量を分けた逐次追加、が必要である。

構成の組合せだけで新規性が成立したとは判断しない。「zにどの情報が必要で、どの役割のexpertが、どの条件で再利用・調整できるか」を反証可能な形で示せれば、この構想の研究の芯になる。
