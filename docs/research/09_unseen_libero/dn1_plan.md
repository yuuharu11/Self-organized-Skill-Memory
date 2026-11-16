# DN1: 未学習 LIBERO のスキル Expert デノイズ

## 目的

画像向きを修正した LIBERO-90 Eval で、正解に最も近い型を選んだメモリ下書きが、公開 pi05 Expert または圧縮下書きで学習した Expert の bridge により改善するかを確認する。これは正解の型選択を許すオフライン上限診断で、閉ループ検索性能の主張ではない。

## 今回の学習なし評価

- 行：`unseen-rows-02` の固定 Eval 619 窓。難しい行は `oracle-01` の Dev q80 閾値から row ID で結合する。
- メモリ：`cluster-pools-01/pool_unit_08.npz` の95単位。正解の S-shared 型誤差だけで単位を選ぶ。
- 姿勢：選択した型を通常 pi05 の action chunk に回転＋共通倍率で Procrustes 合わせし、グリッパーも通常 pi05 から取る（`memory_closed_loop.py` の規則）。正解は姿勢決定に使わない。
- Expert：公開 `pi05_libero` と `compressed-denoise-01/production/experts/A-vm-08` の Dev 選択 checkpoint。両方に同じ行ID固定ノイズ、t0=.2/.4/.6、10積分を使う。
- 対照：下書き、正解行動下書き、通常 pi05、直前保持。正解下書きはデノイザー上限の診断のみ。
- MSE6 は6軸の二乗誤差和を7で割る（従来の6軸平均×6/7）。active7 は7軸平均。全体・Dev q80 hard、episode bootstrap 5,000回の95%区間を記録する。出力と下書き／通常 pi05 の距離も記録する。

## 後続 Expert 学習の計画

学習下書きはクエリ時の自タスク除外または episode cross-fit で作り、同じエピソードの単位を参照しない。学習時 t0 は毎回 U[.2,.5) から抽選する。更新中の live weights で検証し、凍結・jit 済みの古い重みを評価に流用しない。学習・検証に正解下書きと oracle 下書きを診断対照として含め、主クエリ（pool_unit_08 用の新クエリ）は差し替え可能な関数境界に置く。未学習 Eval 正解は採点と上限診断に限り、t0・checkpoint・早期終了の選択には使わない。
