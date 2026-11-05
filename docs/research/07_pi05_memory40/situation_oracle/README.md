# Situation oracle

**問い:** シミュレータから得る正確な状況があれば、次の運動を検索できるか。物体名に依存しない状態・履歴キーの上限を測る。

**結論:** 最良は実行済み行動履歴 H だが、Top-1 MSE は Long で 0.191、hard で 0.348。既存の wrist＋prev キーと hold-last の両方を上回らず、S1/S2 は不通過。近い状況でも次の行動には大きなばらつきが残る。

- [事前計画](plan.md): 固定分割、除外規則、指標、判定基準
- [結果](results.md): 状況キー別の検索性能、判定、多峰性
- 実装: `skill_moe/skillnet/examples/memory_pilot/pi05_memory/situation_oracle/`
- 実行出力: `.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/situation-oracle-01/`
