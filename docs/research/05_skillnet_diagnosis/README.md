# 05 公開 SkillNet の失敗診断と介入（2026-11-02〜11-03）

## 問い

新しい方式を足す前に、公開 SkillNet が未見の組み合わせ（LIBERO-Skill）でどこで失敗するかを測る。スキル入力は実際に使われているか。指示の更新や追加学習で、元の成功を保ったまま失敗を救えるか。

## 結論

全体のまとめは [conclusions.md](conclusions.md)。

- **改善した手法はまだない。基準方策は元の SkillNet（S0）のまま。**
- スキル入力：スキル一覧の順序だけを変えても行動は変わらない。ID を変えると内部の expert 経路は変わるが、行動の差は小さい。expert 番号を「把持スキル」などと呼ばない。
- 指示更新（同じ 18 初期状態）：元指示 10、達成確認で切替 7、効果予測 9、正解状態で切替 8。正しいタイミングで切り替えても悪化する。
- 全体指示＋現在の操作（C）：救済 4・悪化 3。無条件の切替は採用しない。接触停滞への物理的な修正は 0/4。
- C の成功行動を元の指示へ蒸留（M1）：S0 12/18、M1 12/18。救済 3・悪化 3 で正味 0。
- **救える例はあるが、同時に元の成功を壊す。** 次の問いは「救済に必要な変更を学びながら、既存の成功をどう保つか」。

## 実験

| 実験 | 事前設定 | 結果 |
|---|---|---|
| DMP 停止と失敗診断 | — | [dmp_stop_and_diagnosis.md](dmp_stop_and_diagnosis.md) |
| スキル利用と指示介入（9 初期観測、4 開発例の同一状態分岐） | [collection_protocol.md](collection_protocol.md) | [skill_and_instruction_interventions.md](skill_and_instruction_interventions.md) |
| 介入効果に基づく指示更新（5 条件 90 走行） | [instruction_update/plan.md](instruction_update/plan.md) | [instruction_update/results.md](instruction_update/results.md) |
| 元の失敗を救う半日診断（60 走行） | [rescue/plan.md](rescue/plan.md) | [rescue/results.md](rescue/results.md) |
| C の成功行動の蒸留（54＋18 走行） | [distillation/plan.md](distillation/plan.md) | [distillation/results.md](distillation/results.md) |

## その他のファイル

| ファイル | 中身 |
|---|---|
| [conditioning_fixes.md](conditioning_fixes.md) | 10-30 の別軸候補：既存スキル系 VLA の構造的な弱点と小さな改善案（診断の出発点） |
| [instruction_update/slide_outline.md](instruction_update/slide_outline.md) | 指示更新の実験をスライドにするときの主張と結果 |
| [instruction_update/related_work.md](instruction_update/related_work.md) | 固定 VLA への指示介入の先行研究との重なり |
| [evidence/](evidence/) | スキル介入の生データ（`skillnet_prompt.jsonl`）と集計 |

スクリプトは [examples/skill_conditioning/](../../../skill_moe/skillnet/examples/skill_conditioning/README.md)。
