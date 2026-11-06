# 未学習LIBEROタスクへの転移

問い：LIBERO-40で作った運動を、公開LIBERO微調整データに含まれないタスクへ持ち込む余地があるか。

通常pi05_liberoの基準は32/100成功（Wilson95%：23.7–41.7%）。Longの既知4組は成否・終了ステップ数とも一致し、本番の技術エラーは0件だった。失敗68件が残るが、メモリで改善できるかはまだ検証していない。基盤モデルの全事前学習履歴は未確定なので、公開微調整スプリット外という限定付きで扱う。

- [実行前計画](plan.md)：学習データ根拠、機械的選択規則、固定プロトコル。
- [通常方策の結果](baseline_results.md)：Long照合、成功率・区間・失敗像。
- [デモの切り出しの確認](segmentation_demo/README.md)：グリッパーの切替と DCT の復元誤差で切った区間を動画で確認。
- [可変長に切ったときの復元誤差](variable_length_error.md)：切り出し基準と DCT の往復誤差を固定窓と等分に比較。デノイズなし。
- [表現上限のオフライン比較](oracle_results.md)：正解行動で候補を選ぶ検索なしの診断。閉ループ結果ではない。

コードは skill_moe/skillnet/examples/memory_pilot/unseen_libero/、出力は .skillnet-state/outputs/memory-pilot/unseen-libero/baseline-01/。
