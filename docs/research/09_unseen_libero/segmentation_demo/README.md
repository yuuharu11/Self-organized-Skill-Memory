# デモの切り出しの確認

同じ基準で LIBERO のデモを切り、切れ目を動画に重ねた。Expert も検索も使わず、基準を目で確かめるための確認である。ε は目安の値で、調整していない。各タスクとも `demo_0` の1本だけである。

## 基準

1. グリッパーが切り替わったら必ず切る。
2. それ以外は区間を1フレームずつ延ばす。
3. 時間方向の DCT 3個で戻した誤差（移動と回転の正規化 MSE）が ε を超えたら切る。最短は5フレーム。

デモは20 Hzなので2フレームごとに使い、他の実験の10 Hzにそろえた。

## 区間の長さ（フレーム）

| デモ | ε=.005 | ε=.01 | ε=.02 | ε=.04 |
|---|---|---|---|---|
| Long：コンロとモカポット（136） | 26, 5, 17, 7, 16, 16, 15, 17, 12, 5 | 31, 19, 5, 24, 23, 20, 9, 5 | 31, 24, 32, 15, 29, 5 | 31, 24, 47, 29, 5 |
| 未学習：黒いボウルを棚へ（81） | 12, 17, 18, 18, 9, 7 | 25, 4, 23, 16, 6, 7 | 29, 34, 11, 7 | 29, 40, 5, 7 |
| 未学習：赤いマグ（60） | 20, 7, 14, 16, 2, 1 | 24, 3, 23, 9, 1 | 27, 32, 1 | 27, 32, 1 |

ε を大きくすると、切れ目はほぼグリッパーの切替だけになる。グリッパーの切替の前後で、最短より短い区間が残ることがある。

## ファイル

- 動画（ε=.02）：[long_moka_pot.mp4](long_moka_pot.mp4)、[unseen_bowl_on_cabinet.mp4](unseen_bowl_on_cabinet.mp4)、[unseen_red_mug.mp4](unseen_red_mug.mp4)
- ε 別の図：[long_moka_pot.png](long_moka_pot.png)、[unseen_bowl_on_cabinet.png](unseen_bowl_on_cabinet.png)、[unseen_red_mug.png](unseen_red_mug.png)
- 区間の一覧：[segments.json](segments.json)
- 再現：`python skill_moe/skillnet/examples/memory_pilot/unseen_libero/segment_demo.py`
