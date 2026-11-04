# 04 行動の単位で区切るスキルと DMP ルーティング（2026-11-01〜11-02）

## 問い

時間窓（0.5 秒）ではなく、行動の単位でスキルをまとめれば、新しいタスクで選べるようになるか。さらに、DMP 型の原始動作（形・ゴール・時間・位相）に分解して方策がルーティングすれば、抽象的な指示にも応じられるか。

## 結論

- **記憶から運動を取り出す方式は、学習した単純な予測器を上回らなかった**（Dev 未見タスク 3,109 行）。段階の取り違えは、直接回帰 .152 に対して生検索 .195、区間スキルは .23〜.40。
- 重み固定の記憶追加による改善は実在した（DTW .119 → .084）。ただしパラメトリックな予測との差は埋まらない。
- ゼロショットの区間スキル（segskill-v2）は不通過。
- DMP の S3 閉ループは、ユーザー指示で 69/135 試行の時点で停止。途中停止なので、通過・不通過は判定していない。
- 関連研究：無注釈の可変長スキルの発見と選択（PRISE、LOTUS など）、DMP の形・ゴール・時間の分離は、どれも既存。
- 11-02 夜に、新しい方式の実装を保留し、公開 SkillNet の失敗原因の診断（[05](../05_skillnet_diagnosis/README.md)）へ移った。

## ファイル

| ファイル | 中身 |
|---|---|
| [offline_results.md](offline_results.md) | 11-01 夜〜11-02 朝のオフライン結果（検索、区間スキル v1〜v3、5-shot 対照） |
| [dmp_routing_plan.md](dmp_routing_plan.md) | DMP ルーティングの設計と事前基準（停止済み。再開の指示ではない） |
| [related_action_units.md](related_action_units.md) | 可変長の行動単位の発見・選択・実行の研究（調査） |
| [related_dmp_routing.md](related_dmp_routing.md) | DMP 型の運動原始と VLA ルーティングの関連研究（調査） |

スクリプトは [examples/memory_pilot/primitives/](../../../skill_moe/skillnet/examples/memory_pilot/primitives/README.md) と `examples/memory_pilot/dmp_route/`。
