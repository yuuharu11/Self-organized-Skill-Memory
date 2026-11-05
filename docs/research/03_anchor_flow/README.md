# 03 運動の型から始めるデノイズ（開始点デノイズ）

## 問い

メモリから取り出した運動単位を、フローマッチングの開始点（下書き）にしてデノイズすると、新しいタスクへの汎化が上がるか。S1 は軽量版、S2 は公開 SkillNet π0.5 を使った LIBERO 閉ループ。

## 経緯

| 順 | やったこと | 結果 |
|---|---|---|
| 1 | 設計（S1 / S2）、新規性の調査 | [design.md](design.md)、[novelty.md](novelty.md) |
| 2 | 外部・内部の査読。v2 の設計 | 学習目標の不具合を発見。スキルを補助扱いする初版は撤回（[architecture_v2.md](architecture_v2.md)） |
| 3 | 学習なしで、凍結ヘッドにスキルを入れる。場面に合わせる変換器 F | 既知タスクでは誤差 16〜21% 減、新しいタスクではほぼ消える。F は −.009 で基準（−.02）未達（[offline_results.md](offline_results.md)） |
| 4 | 1,500 行で再評価、ヘッドの微調整 | 「グリッパーはスキルが効く」を撤回。効いたのはスキルと無関係な2点（微調整と少ステップのデノイズ）（[report.md](report.md)） |
| 5 | 閉ループ 240 試行、スキル専用 expert | ① 公開 53.8%、③ FT・記憶なし 43.8%、④ FT・下書きあり 43.8%。④ − ③ ＝ 0（[review_answer.md](review_answer.md)、[progress.md](progress.md)） |
| 6 | 69 タスクで選択（並べ直し）を学習する決着実験 | 事前基準で不通過（G1 − G0 ＝ −.0015）。オラクル下書きなら公開比 35% 減（[proposal_rerank69.md](proposal_rerank69.md) §7） |

## 結論

- **新しいタスクでは、どの使い方でも、スキルなしの方策を上回らなかった。** 閉ループでも、推論時にスキルを使う優位は確認できない。
- ボトルネックは「型の中で場面に合う運動片を選ぶこと」。正しい単位は上位 20 に 90〜98% 入るが、最良の運動片は上位 5 に 1.7〜14% しか入らない。
- 伸びしろはある。検索が完璧なら誤差は大きく下がる（オラクル）。ただし未来の正解で選んだ上限で、実行可能な改善ではない。
- 新規性：RA-VLA（2026-10）が「学習した検索＋取り出した行動から flow」をすでに行っている。
- 次の方向として、未来の運動で型を作り直す案（[direction_future_motion_types.md](direction_future_motion_types.md)）を検討したが保留。行動の単位で区切る方向（[04](../04_segment_skills_dmp/README.md)）へ移った。

## ファイル

**まず読む**

| ファイル | 中身 |
|---|---|
| [report.md](report.md) | 現状報告。前提知識なしで読める |
| [progress.md](progress.md) | 進捗報告（閉ループ確定、失敗分析、決着実験の不通過） |

**設計と提案**

| ファイル | 中身 |
|---|---|
| [design.md](design.md) | 主張と検証計画（S1 軽量版、S2 LIBERO 閉ループ） |
| [architecture_v2.md](architecture_v2.md) | 結果を踏まえた改善案と、各部品の設計の根拠 |
| [novelty.md](novelty.md) | 開始点デノイズの新規性と実現性 |
| [proposal_relational_diagnosis.md](proposal_relational_diagnosis.md) | R0 / R1 / R2：関係・履歴を揃えた運動記憶の診断（提案） |
| [proposal_rerank69.md](proposal_rerank69.md) | 運動片の選択を学習する 69 タスクの決着実験（提案と §7 の結果） |
| [direction_future_motion_types.md](direction_future_motion_types.md) | 方向 B：未来の運動で型を作る案（保留） |

**結果の詳細**

| ファイル | 中身 |
|---|---|
| [offline_results.md](offline_results.md) | 手法の定義と全結果の数値（変換器 F を含む） |
| [report_working_notes.md](report_working_notes.md) | report.md の先行版。§7 外部レビューへの対応と、§8 1,500 行での訂正の全表 |
| [review_answer.md](review_answer.md) | レビューへの回答。評価の前提の監査、閉ループ、スキル専用 expert |

**査読・相談**（[reviews/](reviews/)）

| ファイル | 中身 |
|---|---|
| [reviews/representation.md](reviews/representation.md) | スキル選択が当たらない原因と、表現学習からの改善案 |
| [reviews/vla_closed_loop.md](reviews/vla_closed_loop.md) | S2 閉ループの査読（学習目標の不具合） |
| [reviews/external_s2_pipeline.md](reviews/external_s2_pipeline.md) | 外部レビュー：S2 パイプライン（キーの平均化、温度、中心への丸め） |
| [reviews/consultation_skill_retrieval.md](reviews/consultation_skill_retrieval.md) | 外部モデルへの相談：新しいタスクで正しいスキルを取り出す方法 |

**作業ログ**（[logs/](logs/)、経緯の記録）

| ファイル | 中身 |
|---|---|
| [logs/morning_report_s2.md](logs/morning_report_s2.md) | S2 の自律実験の朝のレポート |
| [logs/handoff_stopped_s2.md](logs/handoff_stopped_s2.md) | 測定停止時点の引き継ぎ（英語） |
| [logs/night_log_s2_closed_loop.md](logs/night_log_s2_closed_loop.md) | S2 閉ループの作業ログと全表 |
| [logs/night_log_skill_expert.md](logs/night_log_skill_expert.md) | スキル専用 expert までの作業ログ（閉ループ完走、スキル専用 expert） |

スクリプトは `skill_moe/skillnet/examples/memory_pilot/anchor_flow/README.md`（S2）と `skill_moe/skillnet/examples/emergent/light/intermediate/README.md`（S1）。
