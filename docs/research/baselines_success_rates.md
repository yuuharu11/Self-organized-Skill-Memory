# S2と比較する既存手法の成功率

調査日：2026-10-28 UTC。一次資料の著者報告を整理した文献調査であり、この作業で学習・ロールアウトは実施していない。「未確認」は0%を意味しない。

## 1. 比較の基準

[anchor_flow_skill_memory.md のS2](anchor_flow_skill_memory.md)の**2026-10-28の変更**を採用する。`pi05_libero_moe_skill_4_90`を土台に、導入学習・メモリ・正規化統計・閾値選択にはLIBERO-90だけを使い、LIBERO-10（Long）とLIBERO-Skill `skill_obj_02`〜`09`を評価する。対象タスクのデモ、成功したテスト軌道の蓄積、対象データによる更新は使わない。S2本文の古いコードブロックにある逐次追加は、この評価には適用しない。

以下では「ゼロショット」を**対象タスクの行動データを、重み学習にもメモリにも与えない**意味で使う。事前学習済み基盤モデルまで含めた完全な非接触は保証しない。`pi05_base`の全事前学習データは非公開で、LIBEROとの重複は未確認。

| 区分 | 意味 |
| --- | --- |
| 条件整合 | LIBERO-90→LIBERO-Skill、対象デモなし。S2の対応部分と比較可能。ただしチェックポイント・評価実装を揃えた再評価が必要 |
| 部分的な参考 | source-onlyの転移だが、学習元・対象suiteが異なる |
| 対象学習済み | Long等の評価対象デモを学習に使用。ゼロショットS2の対照ではない |
| 対象メモリあり | 重み固定でも、対象デモ・対象テスト経験を使う。今回のS2とは異なる |

「LIBERO-40」はSpatial/Object/Goal/Long各10タスクの合計。**40学習済みモデルのLong成功率は、90のみで学習したモデルのLongゼロショット成功率ではない。**

## 2. 最も近い比較：SkillNetのLIBERO-90→LIBERO-Skill

出典はSkillNet著者公開PDFの§4.1、Table 1、Appendix D.2/Table 9、Table 10。全方式がLIBERO-90の各タスク50実演で学習され、LIBERO-Skillは未学習のskill compositionとして評価される。各タスク50試行、9タスクなら計450試行。seed数は当該記載から未確認。[SkillNet著者PDF](https://vipl-vsu.github.io/skillnet-website/static/pdfs/skillnet.pdf)

| 手法（SkillNet論文中の実験） | 学習データ／対象デモ | LIBERO-Skill全9タスク、公表平均 % | S2の02〜09、再集計平均 % | 試行数 | ゼロショット／比較可否 |
| --- | --- | ---: | ---: | --- | --- |
| OpenVLA-OFT | LIBERO-90、Skillデモなし | 0.0 | 0.0 | 50/task、8タスク部分は400 | はい／条件整合 |
| OpenVLA | 同上 | 2.0 | 2.25 | 同上 | はい／条件整合 |
| π0 | 同上 | 3.3 | 3.75 | 同上 | はい／条件整合 |
| GR00T-N1.6 | 同上 | 12.2 | 13.5 | 同上 | はい／条件整合 |
| π0.5 | 同上 | 31.8 | **28.5** | 同上 | はい／条件整合 |
| Vanilla MoE | 同上、π0.5上の比較用MoE | 33.1 | 28.0 | 同上 | はい／条件整合 |
| SkillNet | 同上 | **47.8** | **44.5** | 同上 | はい／条件整合。ただし公開重みでの再現値ではない |

**8タスク平均は今回の算術集計であり、論文がそのまま報告した値でも新規ロールアウトでもない。** 丸め済み9タスク平均から逆算せず、Table 1の個別セルから`skill_obj_01`に対応する`0101`列を除いた。残る8列を等重みで平均した。paperの列順と公開task ID順が異なるため、指示文をTable 9と[公開タスク一覧](../training_and_evaluation.md#libero-skill-evaluation)で照合した。

| 公開task ID | 論文のskill composition列 | π0.5 % | Vanilla MoE % | SkillNet % |
| --- | --- | ---: | ---: | ---: |
| skill_obj_02 | 301：drawerを開け、bowlをplateへ | 16 | 26 | 56 |
| skill_obj_03 | 014：bowlをbottom drawerへ置いて閉める | 22 | 38 | 60 |
| skill_obj_04 | 401：drawerを閉め、bowlをplateへ | 88 | 90 | 92 |
| skill_obj_05 | 46：drawerとmicrowaveを閉める | 20 | 2 | 28 |
| skill_obj_06 | 32：bowlをstackし、drawerを開ける | 14 | 10 | 24 |
| skill_obj_07 | 601：bowlをplateへ置き、microwaveを閉める | 60 | 44 | 56 |
| skill_obj_08 | 74：drawerを閉め、stoveを消す | 0 | 0 | 20 |
| skill_obj_09 | 501：bowlをplateへ置き、microwaveを開ける | 8 | 14 | 20 |
| 02〜09平均 | 各50試行、計400 | 28.5 | 28.0 | 44.5 |

skill compositionの数字の並びだけから実行順を推測しない。上表の対応は指示文による。`skill_obj_03`はS2文書でLIBERO-10 task 3とほぼ同じ場面・タスクとされるため、**Long＋Skillの18タスクを独立した18種類と見なした総合平均だけで結論を出さない**。suite別・重複除外の感度分析も必要。

### 公開実装との対応と、未報告の値

| 設定 | 確認できたもの | 成功率に関する限界 |
| --- | --- | --- |
| SkillNet LIBERO-90 | 公開config `pi05_libero_moe_skill_4_90`、20,000 steps、batch 32、horizon 10。著者PDF Table 10とも一致 | **LIBERO-90自身の成功率と、90→Longのゼロショット成功率は、確認した著者PDF・公開評価文書に数値を見つけられない** |
| SkillNet LIBERO-40 | 公開config `pi05_libero_moe_skill_4_40`、30,000 steps、batch 128。4suite全て学習済み | 以下のTable 3の値。90モデルの結果として流用不可 |
| 公開重み | [SkillNet-LIBERO-90](https://huggingface.co/jsw19/SkillNet-LIBERO-90)、[SkillNet-LIBERO-40](https://huggingface.co/jsw19/SkillNet-LIBERO-40) | この調査では重みの再評価をしておらず、PDFの各数値と同一runかは未確認 |

ローカル[学習・評価文書](../training_and_evaluation.md)では90変換後が73 taskキー、7,874 skill-slice episodes、574,571 frames。[データ処理文書](../libero_data_processing.md)ではslice前4,006 RLDS episodesである。73は言語指示の重複除去に対応するという既存調査があるが、**73というメタデータ値だけから「90のうち17タスク欠落」と断定しない**。source task/sceneとsliceの対応をmanifestで照合する。50実演×90と変換後slice数も同一単位ではない。

公開Skill評価スクリプトはzero-shot時800→1,600 control steps＋初期安定化10 stepsを使うと[ローカル文書](../training_and_evaluation.md)に記載される。S2が別の上限を使う場合、公表値との一致は保証されない。論文Table 11とローカル40学習文書には学習率・EMAの差もあるため、論文値の再現時はconfigと版を保存する。

## 3. Long/4suiteの高成功率：対象学習済みの参考値

単位%。`平均`はSpatial/Object/Goal/Longの平均で、LIBERO-90平均ではない。試行数は当該結果の一次資料で確認した範囲のみを記す。

| 手法・出典 | 学習データ（Longデモの使用） | Spatial / Object / Goal / Long | 平均 | 評価試行数 | S2でのゼロショットか／直接比較できない理由 |
| --- | --- | --- | ---: | --- | --- |
| π0：OFT Table I掲載の既報値 [OFT] | 対象suiteへfine-tune、Long使用。90-onlyではない | 96.8 / 98.8 / 95.8 / **85.2** | 94.2 | この転載行の元実験の試行数は未確認 | いいえ。OFT自身の500/suiteを転載baselineへ自動適用しない |
| π0.5：公式openpi [PI] | `physical-intelligence/libero`の40 taskにfine-tune、Long使用、30k steps | 98.8 / 98.2 / 98.0 / **92.4** | **96.85** | 結果表の試行数・seed数は未記載 | いいえ。公式LIBERO適応済み重み。`pi05_base`や90-onlyモデルの値ではない |
| OpenVLA-OFT：suite別policy [OFT] Table I | 各10 taskで別々にfine-tune、Long使用 | 97.6 / 98.4 / 97.9 / **94.5** | 97.1 | 500/suite＝50/task。seed数未確認 | いいえ。複数policyで対象を学習。2画像＋proprio、modified dataset |
| OpenVLA-OFT：40共同policy [OFT] Table XIV | 40 task共同、Long使用 | 97.7 / 98.0 / 96.1 / **95.3** | 96.8 | 論文評価規約500/suite | いいえ。Table Iの97.1と別条件 |
| π0：SkillNet著者表 [SK] Table 3 | 4suiteの同じ実演、各task50、Long使用 | **96.4** / 98.8 / 95.8 / 85.2 | **94.2（原表）** | 50/task、計2,000。seed数未確認 | いいえ。原表Spatialは96.4で、上の既報96.8と異なる |
| π0.5：SkillNet著者表 [SK] Table 3 | 同上、Long使用 | 99.4 / 97.8 / 98.2 / **96.6** | 98.0 | 同上 | いいえ。公式openpiの92.4と別実験 |
| SkillNet：40学習 [SK] Table 3 | 同上、Long使用 | 99.6 / 99.2 / 99.0 / **98.4** | 99.1 | 同上 | いいえ。90→未見Longに対する上限の保証にもならない |
| Optimus-R：100% [OR] Table 1 | 対象4suiteの100%でadapt。Stage Aの正確なコーパス内訳は未確認。Long使用 | 99.2 / 99.4 / 98.4 / **94.2** | 97.8 | 500/suite、計2,000。seed数未確認 | いいえ。targetデモによる適応。90→Long zero-shotではない |
| Optimus-R：30% [OR] Table 1 | 対象4suiteの30%でadapt、Long使用 | 89.6 / 91.8 / 87.0 / **86.0** | 88.6 | 同上 | いいえ。少量学習とゼロショットは別 |
| OptimusVLA [OV] Table 1 | LIBERO対象4suiteの学習設定。GPMもデータ内の軌道を用いる。90-only条件ではない | 99.6 / 99.8 / 98.4 / **96.4** | 98.6 | 500/suite。seed数未確認 | いいえ。Longデータを除いた実験値としては報告されていない |
| LAFM [LA] Table 2 | 4suite共同、前処理後1,693 episodes、40 task、Long使用 | 97.2±0.7 / 99.2±0.0 / 97.2±0.5 / **92.3±0.3** | 96.5±0.3 | 50/task×3 seeds | いいえ。対象を学習した0.11B方策で、SkillNetとの規模・学習量も異なる |
| FM：LAFM内対照 [LA] Table 2 | LAFMと同じ40 task | 96.1±1.0 / 97.9±0.3 / 96.5±0.5 / **91.2±0.2** | 95.4±0.3 | 同上 | いいえ。同論文内の開始分布の寄与を見る対照 |

SkillNet Table 3のπ0行はセル平均が94.05%となり、印刷された94.2%と整合しない。**原表を黙って修正せず記録する。** また、OFTのsingle-image版はLong 90.7%、平均95.3%であり、2画像＋proprio版のLong 94.5%、平均97.1%と混ぜない。

Optimus-Rの中間データ量もLong **91.4%（50%学習）／92.2%（70%学習）**、平均95.6／96.6%。同表の著者再現π0.5はLong **62.0／88.0／91.4／92.4%**、4suite平均72.1／92.1／95.0／96.9%（30／50／70／100%）。Fullの+0.9 ppや少量データの+16.5 ppは**同論文設定内**の差であり、S2で期待できる改善幅ではない。[OR]

## 4. LIBERO-90で学習・評価した値

| 手法・出典 | 学習・評価対象 | LIBERO-90成功率 % | 試行数 | ゼロショット／S2との関係 |
| --- | --- | ---: | --- | --- |
| π0：LAFM著者による学習 [LA] Table 3/7 | 90 task、前処理後3,959 episodes。fine-tune 200k steps、batch 8 | **92.1±0.1** | 50/task×3 seeds＝4,500/seed | いいえ。90のin-domain参考。90→Longは未測定 |
| LAFM [LA] Table 3/5 | 同じ90 task、200k steps、batch 448、0.11B、scratch | **93.0±0.2** | 同上 | いいえ。同論文内のFMとの差は比較可能だが、S2のモデル・データ処理・更新量と異なる |
| FM [LA] Table 3 | LAFMと同じ90 task | 82.6±0.2 | 同上 | いいえ。LAFM内対照 |
| FM＋latent action classification [LA] Table 3 | 同上 | 85.4±0.1 | 同上 | いいえ。補助損失だけの対照 |
| SkillNet／π0.5／OpenVLA-OFT | 今回確認した一次資料 | **90のin-domain値は未確認** | 未確認 | 40suite平均やSkillゼロショット値で代用しない |
| Optimus-R／OptimusVLA／RTS／RA-VLA | 今回確認した本文・付録 | **90の値は未確認** | 未確認 | 4suite平均は90ではない |

LAFMの90実験はLIBERO-90自身の成績であり、**Long・Skillへのzero-shot成功の証拠ではない**。同論文の90とSkillNetの90は前処理・保存単位の実演件数も異なる。

## 5. 検索・メモリを使うが、対象経験へのアクセスが違う実験

| 手法・条件 | 公表結果 | 学習データ／対象経験 | 評価試行数 | ゼロショット／直接比較できない理由 |
| --- | --- | --- | --- | --- |
| RTS、π0.5 [RT] Table 1 | Long **92.4±0.2→94.4±0.3%** | downstreamでfine-tuneした方策。成功テスト軌道を逐次保存。さらにcriticに**学習集合の対象成功動画1本/task**を参照として使用 | 50/task×3 seeds | いいえ。重み固定でもsource-onlyメモリではない。改善は繰返し配備での経験蓄積 |
| RTS、π0 [RT] Table 1 | Long **81.6±0.8→84.4±0.4%** | 同上。LIBEROの正確なfine-tuneデモ件数は当該記載から未確認 | 同上 | いいえ。同一初期状態・メモリ更新順を揃えた別トラックなら比較候補 |
| RA-VLA [RA] Table 1 | 原表：Spatial **0.320**／Object **0.556**／Goal **0.532**／Long **0.130**、平均**0.3845** | 3suiteで学習、1suite保留。保留taskごとに**対象expertデモ3本**をbufferへ入れる | 50/task、seed数未確認 | 対象への勾配更新なしだが**3-shot in-context**。90-onlyでも対象デモなしでもない |
| Optimus-R Spatial→Goal、適応前 [OR] Table 12 | Goal全体 **8%**、共有bowl-on-plate task **68%**、bank 16 | Spatialを元にGoalへ転移、Goalへの追加適応前。初期Stage Aの全データ内訳・独立性は未確認 | この付録表の試行数・seed数は明示を確認できず | 論文上zero-shot／部分的参考。90→Long/Skillとは学習元・対象が異なる |
| 同、Goal適応後 [OR] Table 12 | Goal **96%**、bowl-on-plate **98%**、bank 27 | Goalのデモで追加適応 | 同上 | いいえ。bank増加だけの効果ではなく、残差等の最適化を含む |

RA-VLAのTable 1はcaptionが「%」である一方、セルが0–1形式である。**割合として100倍するとLong 13.0%、平均38.45%**になるが、これは表記の解釈を伴う換算であるため原値も併記した。上の96〜99%の対象学習済み結果と並べて優劣をつけない。

RAGDP・BRIDGeRの今回確認済みの主要結果はRobomimic／Push-T等であり、S2に対応するLIBERO-90→Long/Skill成功率は未確認。異なるベンチマークの数値をLIBEROの空欄へ移さない。[関連する開始分布の調査](novelty_anchor_flow.md)

## 6. S2で最低限そろえる比較

1. **同じ90重みをそのまま使ったSkillNet**、提案のsource-only微調整後でアンカーなし、提案アンカーあり、メモリshuffleを同じ初期状態・乱数で実行する。学習の効果とメモリの効果を分離する。
2. 公表SkillNet 47.8%を目標値として8タスクへ直接置かず、上の**44.5%再集計値とtask別結果**を参考にする。公開checkpointとの一致は自分たちの50/task再評価で確かめる。
3. Longゼロショットの既存一致値は未確認なので、**90重みのLong無改変評価を自前の主要baseline**とする。公式`pi05_libero`・SkillNet40・Optimus-R 94.2%は対象学習済み参考として別表に置く。
4. Optimus-Rはsource90だけでinterfaceとbankを学習・構築し、target更新を止めた移植版を比較する。これは原著Table 1の再現ではない。[再現計画](optimus_r_reproduction.md)
5. pilotは20/taskでもよいが、文献比較をする最終評価は少なくとも50/task。suite別macro成功率、task別成功数/試行数、seed差、同一初期状態での対応差と区間推定を示す。95%付近の1 pp差を1 runだけで断定しない。

## 7. 書誌・版・取得記録

すべて2026-10-28取得。URLは一次資料。PDF・HTML・API応答と再集計記録は今回の証跡ディレクトリ（ローカル記録: `.skillnet-state/outputs/memory-pilot/research-baselines-optimusr-20261028T172815Z`）に保存した。

| 記号 | 論文・資料／確認版 |
| --- | --- |
| SK | Xie et al., **SkillNet: Hierarchical Skill Modeling for Compositional Generalization in Vision-Language Action Models**, ICML 2026。[著者プロジェクト](https://vipl-vsu.github.io/skillnet-website/)、[会議掲載一覧](https://icml.cc/Downloads/2026)、[著者PDF](https://vipl-vsu.github.io/skillnet-website/static/pdfs/skillnet.pdf)。PDFは30ページ、websiteの当該ファイル最終変更commit [`534f2936`](https://github.com/VIPL-VSU/skillnet-website/commit/534f293615dbcaad77eb757b8d65f11c164e09d3)、2026-06-04。本文にarXiv版番号なし。SHA-256 `64992fe98dfda4468d587f47754b76dc89d01cae4f2195cfa154060acfe4314c`。OpenReview本文/APIはアクセス制限で取得不可なので、これを最新camera-readyと断定しない |
| SK-code | [公式SkillNet](https://github.com/VIPL-VSU/SkillNet/tree/52e3caf52bb183413bbdda83c359f76bb41250eb)、commit `52e3caf5`、2026-07-03。ローカルREADMEは研究用に変更されており、元のrelease概要は[docs/skillnet.md](../skillnet.md)。90のHF revisionは `b12cd4af73070cf1c2fc39402546f769a63c19a2`、2026-06-28。重みのダウンロードは実施せず |
| PI | [Physical Intelligence公式openpi LIBERO結果](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md#results)、commit `215abfb2`、2026-08-24。試験実施日は記載なし。対応configとHFデータ`meta/info.json`の40 taskを照合。π0.5の原論文の数字として転載しない |
| OFT | Kim, Finn, Liang, **Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success**, [arXiv:2502.19645v2](https://arxiv.org/html/2502.19645v2)、2025-04-28。Tables I, IV, XIV。[公式LIBERO説明](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LIBERO.md)でもsuite別97.1と40共同96.8の違いを確認 |
| OR | Li et al., **Inline Memory Meets Reusable Skills: Memory-centric Framework for Vision-Language-Action Model**, Optimus-R、[arXiv:2609.39794v1](https://arxiv.org/html/2609.39794v1)、2026-09-30。§4.1–4.2、Tables 1, 12 |
| OV | Li et al., **Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation**, OptimusVLA、[arXiv:2602.20200v2](https://arxiv.org/html/2602.20200v2)、2026-05-18。CVPR 2026との著者記載。Table 1、§4.1 |
| RT | Zhao et al., **Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs**, [arXiv:2605.10094v2](https://arxiv.org/html/2605.10094v2)、2026-05-12。Tables 1, 4–5、§3、Appendix A |
| LA | Machado et al., **Flowing With Purpose: Latent Action Guided Flow Matching Policies For Robotic Manipulation**, [arXiv:2606.23420v1](https://arxiv.org/html/2606.23420v1)、2026-06-22。§3.1、Tables 2–3, 5, 7。±は3 seedの標準偏差（Bessel補正） |
| RA | Jang et al., **RA-VLA: Retrieval-Augmented VLA for Test-Time Adaptation**, [arXiv:2608.25585v2](https://arxiv.org/html/2608.25585v2)、2026-10-07。Table 1、評価設定。ICML 2026とのarXiv著者記載 |
