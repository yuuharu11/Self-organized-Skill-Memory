# スキル条件の診断と改善：外部から参照するための手順

このリポジトリの外（自分の環境の SkillNet）で作業する人向けの入口です。
検討の本文は [docs/research/skill_conditioning_fixes.md](../../../../docs/research/skill_conditioning_fixes.md) にあります。
このページでは、それを読んで内容を確かめ、最初の実験（T0）を始めるまでの手順を説明します。

- このリポジトリ：<https://github.com/yuuharu11/Self-organized-Skill-Memory>（公開）
- 作業の土台：upstream の SkillNet <https://github.com/VIPL-VSU/SkillNet>
  - 下のリンクは、すべてコミット `52e3caf` に固定しています。
  - このリポジトリの SkillNet 部分には研究用の変更が入っています。外部で作業するときは upstream を使ってください。

## 1. 読む順番

1. [skill_conditioning_fixes.md](../../../../docs/research/skill_conditioning_fixes.md)：検討の本文
   - §1：確認した事実
   - §2：外部の案の検証
   - §3：順位
   - §4：タスク設定
   - §5：位置づけの評価
2. 比較対象の論文
   - SkillNet（ICML 2026）：[ICML](https://icml.cc/virtual/2026/poster/65559)、[著者PDF](https://vipl-vsu.github.io/skillnet-website/static/pdfs/skillnet.pdf)
   - AtomicVLA（CVPR 2026）：[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AtomicVLA_Unlocking_the_Potential_of_Atomic_Skill_Learning_in_Robots_CVPR_2026_paper.html)、[arXiv 2603.07648](https://arxiv.org/abs/2603.07648)
   - SkillMemo：[arXiv 2608.05970](https://arxiv.org/abs/2608.05970)
   - Optimus-R：[arXiv 2609.39794](https://arxiv.org/abs/2609.39794)
3. 公表されている成功率の比較条件：[baselines_success_rates.md](../../../../docs/research/baselines_success_rates.md) の §2

本文の中には、`.skillnet-state/` や日付つきの run 名など、このリポジトリを作業した環境にしかない記録への言及があります。これらは外からは見られません。その代わり、必要な結果は下の §2 の手順で確かめ直せます。

## 2. 本文の事実を upstream で確かめる

| 本文の主張 | 確かめる場所（upstream、`52e3caf`） |
| --- | --- |
| action expert の FFN が MoE で、expert は top-1、共有 expert が 1 つ | [gemma_moe_skill.py L327–L404](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/src/openpi/models/gemma_moe_skill.py#L327-L404)（`top_k`、`shared_expert`） |
| スキルはルーターの logit にだけ入る | 同じファイルの [L379](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/src/openpi/models/gemma_moe_skill.py#L379)。cross-attention の出力が `router_logits` にだけ使われている |
| 損失は load balancing だけ | 同じファイルの [L404](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/src/openpi/models/gemma_moe_skill.py#L404) |
| expert 数が 4 | 設定名 `pi05_libero_moe_skill_4_90`（[training_and_evaluation.md](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/docs/training_and_evaluation.md)） |
| 評価では、タスク全体のスキル列を毎 step そのまま渡す | [main_skill_obj_test_moe_skill.py L217](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/examples/libero/main_skill_obj_test_moe_skill.py#L217)（`"skills": all_classes`） |
| 学習データはスキル単位で切られ、各フレームに現在のスキル `class` がある | [libero_data_processing.md](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/docs/libero_data_processing.md) の「Skill-Slice Inputs」 |
| 論文のスキルは 6 桁の運動コードの階層だが、公開設定は平らな ID | [skill_hierarchy.md](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/docs/skill_hierarchy.md) |
| 引き出しと電子レンジの開閉がすべてクラス 3 | [instruct2plan_obj_90.json](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/examples/libero/annotations/instruct2plan_obj_90.json) の `all_classes`。fallback の [transforms_skill.py L334](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/src/openpi/transforms_skill.py#L334) とはクラス番号がずれている（open が 2 と 3） |

### クラス 3 の中の運動の違いを測り直す

[measure_articulation_motion.py](measure_articulation_motion.py) は単独で動きます（numpy、h5py、scipy が必要）。

1. LIBERO-90 のデモ（hdf5）を取得します。LIBERO 公式リポジトリの `benchmark_scripts/download_libero_datasets.py` で `libero_90` を指定してください。
2. このスクリプトだけを取得して実行します。

```bash
curl -LO https://raw.githubusercontent.com/yuuharu11/Self-organized-Skill-Memory/main/skill_moe/skillnet/examples/skill_conditioning/measure_articulation_motion.py
python measure_articulation_motion.py /path/to/LIBERO/libero_90
```

期待される出力（各 50 デモの平均）は次のとおりです。

| ファイル | クラス | 関節変化 | 移動 cm | 曲がり | 手首回転 ° | グリッパ cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SCENE1 上段の引き出しを開ける | 3 | 0.26 | 19.7 | 1.93 | 8.4 | 7.8 |
| SCENE1 下段の引き出しを開ける | 3 | 0.41 | 20.9 | 1.87 | 64.2 | 7.7 |
| SCENE2 上段の引き出しを開ける | 3 | 0.23 | 19.0 | 1.73 | 6.2 | 7.9 |
| SCENE5 上段の引き出しを閉める | 3 | 0.22 | 20.7 | 1.34 | 11.5 | 7.9 |
| SCENE4 下段の引き出しを閉める | 3 | 0.61 | 22.8 | 1.36 | 76.9 | 7.8 |
| SCENE10 上段の引き出しを閉める | 3 | 0.19 | 15.5 | 1.10 | 8.3 | 8.0 |
| SCENE7 電子レンジを開ける | 3 | 1.70 | 27.2 | 2.56 | 22.3 | 8.0 |
| SCENE6 電子レンジを閉める | 3 | 1.64 | 22.3 | 1.21 | 12.2 | 7.9 |
| SCENE3 コンロをつける | 4 | 0.87 | 4.1 | 1.63 | 53.7 | 3.4 |
| SCENE8 コンロを消す | 4 | 1.41 | 2.4 | 2.33 | 87.5 | 4.3 |

計測のしかたは次のとおりです。
- 対象の関節は、ロボット以外の状態のうち、最初と最後で最も大きく変わった次元とします。
- その関節が最大速度の20%を超えて動いている区間だけを集計します。
- 「曲がり」は、手先が動いた道のりを始点と終点の直線距離で割った値です。1なら直線です。
- 動いた方向は世界座標で出しています。シーンごとに家具の向きが違うので、シーン間では比べないでください。

## 3. 最初の実験 T0：LIBERO-Skill でスキル条件が効いているか

本文 §3 の「0. 前提確認」です。upstream の SkillNet を、upstream の手順（[quick_start.md](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/docs/quick_start.md)、[training_and_evaluation.md](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/docs/training_and_evaluation.md)）で用意してください。重みは [jsw19/SkillNet-LIBERO-90](https://huggingface.co/jsw19/SkillNet-LIBERO-90) を使います。

1. **再現**：変更なしで LIBERO-Skill の 9 タスクを各 50 試行し、論文の 47.8% と比べます。
2. **入れ替え**：[main_skill_obj_test_moe_skill.py L217](https://github.com/VIPL-VSU/SkillNet/blob/52e3caf52bb183413bbdda83c359f76bb41250eb/skill_moe/skillnet/examples/libero/main_skill_obj_test_moe_skill.py#L217) で渡す `all_classes` を、各クラスを別のクラスに置き換えた列にします。
   - 置き換えは固定の対応にします：0→3、1→4、3→0、4→1。
   - 例：`[3, 0, 1]` は `[0, 3, 4]` になります。
   - LIBERO-Skill で使われているクラスは 0、1、3、4 だけです。
3. 1 と 2 を同じ初期状態、同じ seed で比べます。
   - 成功率が π0.5 の水準（31.8%）近くまで落ちるなら、スキル条件は組み合わせの汎化に効いています。
   - ほとんど変わらないなら、SkillNet の改善はスキル条件以外から来ています。
4. 余力があれば、同じ観測と同じノイズで、元のスキルと入れ替えたスキルの行動 chunk の差（L2）を測ります。それを、ノイズだけを変えたときの差と比べます。
   - このリポジトリで学習したタスク（libero_10）について測ったときは、スキル入れ替えによる差が、ノイズを変えたときの約 1/4〜1/6 でした。

結果は、成功率（タスク別と全体）、試行数、seed、使った重みの revision を表にして共有してください。
