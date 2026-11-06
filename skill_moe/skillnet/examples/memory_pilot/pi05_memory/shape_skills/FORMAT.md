# Shape Skill 表現交換形式

この形式はデータセット外の評価器・Stage B readerとの境界です。値はファイル単位で読み取り可能なNumPy配列、コードとposeは区間単位です。

各 `<name>/` に次を置く。

- `segments.npz`: `task`, `episode`, `start`, `length`, `row_ids`。各配列は `[N]`。`start` は元episode内の10 Hz frame index、`length` はframe数。`row_ids[i]` は区間内で最初に始まるrun-01の10-step窓の**dataset row ID**。該当窓がなければ`-1`。
- `codes.npy`: `[N,D]` float32。`D`は方式ごとの固定潜在次元。
- `pose.npz`: scale、回転、元長さ、ゴール、decoder固有の非潜在情報など。方式が外部poseを必要としない場合は省略する。可変長追加データは`offsets`付きの連結配列で保存する。
- `reconstruct.py`: `decode(codes, pose, length)`。`length`はスカラーまたは[N]。同じ長さなら`float32[N,length,7]`、異なる長さなら長さNのfloat32配列リストを返す。返値は公式q01/q99正規化行動空間（クリップなし）、先頭7値がactive action。readerはパッケージをimportできるよう `skill_moe/skillnet/examples/memory_pilot/pi05_memory` をPYTHONPATHに加える。
- `manifest.json`: representation名・version、fitの対象task/episode条件、正規化統計識別、軌跡/phase定義、poseの単位と回転方向、コード次元、decode仕様、seedを記録する。

Segment row IDは元datasetのglobal row IDであり、memory subset内の位置ではありません。Train/Dev/Evalの除外はreader側で`task`とepisode IDを照合してassertしてください。配列をpickle objectにしないでください。

M1対照ではM1が学習された固定10×32のrun行動窓を使う。可変区間の表現コードはその区間内最初のrowのbであり、復元に必要なposterior residual/state/featureは`pose.npz`に付随する。M1の保存量はコード次元だけでは評価しない。

`canonical_raw` 対照は、圧縮器を通さず位相32点の物理正準化Δ位置・Δ回転（192 float）を`codes.npy`に保存し、scale・proper rotation・元長さ・グリッパー系列を`pose.npz`に保存します。復元時に6次元運動へposeを戻し、グリッパーを連結してからq01/q99で正規化します。
