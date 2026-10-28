# Self-organized Skill Memory

ラベルなしに運動経験から自己組織化する「スキル」をメモリとして持ち、
Vision-Language-Action モデル（π0.5）の未学習の場面・タスクへの汎化を高める研究のコードと研究ノートです。
[SkillNet](https://github.com/VIPL-VSU/SkillNet)（ICML 2026）の公開コードをベースにしています。

> 研究は進行中です。現時点の結果は、凍結した π0.5 の特徴の上で小さなモデルを学習する軽量版での、
> オフラインの行動誤差が中心です。環境での成功率はまだ確定していません。
> 最新の状況は [docs/research/skill_memory_findings.md](docs/research/skill_memory_findings.md) を参照してください。

## 研究の主張

1. **スキル = 場面に依存しない運動の型。** タスク・言語のラベルなしに運動経験から自己組織化し、個数は事前に決めない。
   場面との対応はメモリ（鍵 = 観測の埋め込み、値 = 運動の型）が受け持つ。
2. **継続学習に対応する。** 新しい運動は、勾配による更新なしにメモリへ単位として追加される。
3. **スキルを外部の条件として与えない。** 条件は画像・サブタスク・ロボット状態の埋め込みだけ。
4. **デノイズは、根本的な運動を状況に合わせて最適化する過程である。**
   メモリから取り出した運動の型の混合を、フローマッチングのデノイズの開始点にする。

```
b = Enc(A, s)                                  行動チャンク A と状態 s から運動のコード b
w_k(o) = softmax_k( cos(E(o), κ_k) / τ )       観測 o の埋め込みでメモリの単位 k を重み付け
x_{t0} = (1 − t0)·Dec(b') + t0·ε               取り出した運動の型 b' から t0 でデノイズを開始
```

詳細は [docs/research/anchor_flow_skill_memory.md](docs/research/anchor_flow_skill_memory.md)。

## リポジトリ構成

SkillNet 由来の構成（`skill_moe/`, `data_process/` など）に、次を追加しています。

| パス | 内容 |
| --- | --- |
| `docs/research/` | 研究ノート（方針・判定基準・結果・関連研究）。入口は [docs/research/README.md](docs/research/README.md) |
| `skill_moe/skillnet/examples/emergent/` | 創発スキルの軽量検証（スキル表現の比較 M0〜M6、運動の型から始めるデノイズの S1） |
| `skill_moe/skillnet/examples/memory_pilot/` | 検索メモリの事前判定（G0〜G3）と、LIBERO 閉ループでのデノイズ開始点の検証（S2） |
| `skill_moe/skillnet/src/openpi/models/pi0_anchor_flow.py` ほか | デノイズ開始点の注入（anchor flow）とスキル潜在を持つ π0 の変種 |
| `env/` | 現在のコンテナ用の環境設定・スモークテスト・依存の固定リスト |
| `tests/` | 追加コードのテスト |

## 開発環境

```bash
git clone https://github.com/yuuharu11/Self-organized-Skill-Memory.git
cd Self-organized-Skill-Memory
source env/activate.sh
python env/smoke_test.py --render
python -m pytest -q tests/test_websocket_policy_server.py
```

環境の詳細は [docs/development_environment.md](docs/development_environment.md)、
検証結果は [docs/environment_validation.md](docs/environment_validation.md) を参照してください。

## ベースの SkillNet

SkillNet の学習・評価（LIBERO-40/90、LIBERO-Skill、RoboTwin few-shot）の手順は、元の README を
[docs/skillnet.md](docs/skillnet.md) に移して残しています。データ処理・チェックポイント・評価の詳細は
[docs/quick_start.md](docs/quick_start.md) と [docs/training_and_evaluation.md](docs/training_and_evaluation.md) にあります。

SkillNet を使う場合は、次を引用してください。

```bibtex
@inproceedings{xie2026skillnet,
  title={SkillNet: Hierarchical Skill Modeling for Compositional Generalization in Vision-Language Action Models},
  author={Xie, Senwei and Zhang, Yuntian and Tan, Zhenzhou and Wang, Ruiping and Wang, Pengwei and Zhang, Shanghang and Chen, Xilin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## License

特に記載のない限り Apache-2.0 です。SkillNet および第三者のコンポーネントは元のライセンスに従います。
`NOTICE.md` と `skill_moe/skillnet/third_party/` 以下のライセンスファイルを参照してください。
