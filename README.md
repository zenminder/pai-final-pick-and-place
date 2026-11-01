# 言語指示で動くピック＆プレース — LLMプランナー × MuJoCo（Franka Panda）× VLM検証

Physical AI応用1講座 最終課題（omnicampus: zenminder）

日本語の指示文をLLMが作業計画（JSON）に変換し、MuJoCo上のFranka Pandaアームが逆運動学と物理接触による把持でピック＆プレースを実行、完了後のカメラ画像をVLMが検査して「指示を満たしたか」を判定します。AIの利用は**推論のみ（学習なし）**。ROS/Gazeboは使わず、pipで入るライブラリだけで macOS（Apple Silicon）・GPUなし・ヘッドレスで再現できます。

- デモ動画（等速・倍速なし）: （提出後に追記）
- 使用ツール: MuJoCo 3.x / MuJoCo Menagerie (Franka Panda) / OpenAI Python SDK / imageio-ffmpeg / Pillow

## パイプライン

```
日本語の指示文 + シーン情報（物体名・色・xy座標）
   │ (a) LLMプランナー  … JSON Schema制約付き構造化出力で pick/place 手順を生成、名前を検証
   ▼
作業計画 [{pick: cube_red, place: pad_A}, …]
   │ (b) MuJoCo + Franka Panda … DLS逆運動学で TCP目標 → 7関節の位置制御、物理接触で把持
   ▼
完了後の正面カメラ画像
   │ (c) VLM検証 … 画像 + 指示 + 実行した計画 → 合否と所見
   ▼
(d) オフスクリーン描画 → ffmpeg で MP4（指示・計画・進捗・状態を重畳）
```

## セットアップ

```bash
uv venv -p 3.10 .venv && uv pip install -r requirements.txt   # または: python -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=...   # LLM/VLM を使う場合のみ。リポジトリには絶対にコミットしない
```

`ffmpeg` バイナリは `imageio-ffmpeg` に同梱されるため別途インストール不要です。

## 実行

```bash
.venv/bin/python run_demo.py                                  # 既定の2シナリオ（OpenAI GPTでプラン生成＋画像検証）→ out/demo.mp4, out/demo_log.json
.venv/bin/python run_demo.py --offline                        # APIなし: キーワード規則プランナーで同じパイプラインを実行
.venv/bin/python run_demo.py --instruction "青いキューブを白いパッドへ"   # 任意の指示（複数指定可）
.venv/bin/python pick_place.py --plan "red:pad_A,green:pad_B,blue:pad_A" --video out/test.mp4   # 計画を直接与えて実行
.venv/bin/python eval_grasp.py --trials 30 --seed 0                                          # 把持ロバスト性評価（初期位置ランダム化、動画なし）→ out/eval_grasp.json
.venv/bin/python make_report.py --video-url <YouTube URL> --repo-url <GitHub URL>              # レポートPDFを生成 → out/説明資料_PAI最終課題_zenminder.pdf
```

使用モデルは環境変数 `PAI_MODEL` で変更できます（既定 `gpt-5.5`）。

## ファイル構成

| パス | 内容 |
|---|---|
| `scene.xml` | テーブル・3キューブ・2パッド・カメラ。Menagerie の `panda.xml` を無改変で include |
| `pick_place.py` | `PandaSim`: 物理ステップ、DLS逆運動学、動作プリミティブ、フレーム録画 |
| `ai_planner.py` | LLMプランナー `plan()`、VLM検証 `verify()`、オフライン規則プランナー |
| `run_demo.py` | 指示 → 計画 → 実行 → 検証 → 字幕付きMP4 を一気通貫で生成 |
| `eval_grasp.py` | 初期位置・割当をランダム化したN試行で把持成功率と位置誤差を計測 |
| `make_report.py` | `out/demo_log.json` と `out/eval_grasp.json` からレポートPDFを生成（headless Chrome） |
| `assets/franka_emika_panda/` | MuJoCo Menagerie の Franka Panda モデル（ライセンス同梱） |
| `out/` | 生成物（動画・画像・ログ・PDF）。git 管理外 |

## AIの利用について

- LLMプランナーとVLM検証は OpenAI API（GPT）の推論のみを利用し、学習は行っていません。
- `--offline` ではAIを使わずキーワード照合で計画するため、APIキーなしでもパイプライン全体を再実行できます（「暖色・寒色」のような語彙に現れない指示は空計画になります。これがLLMプランナーの価値です）。
- 物体位置はシミュレータの真値を使用しています（画像からの位置推定は未実装）。

## ライセンス

- Franka Panda モデルは [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) のもので、`assets/franka_emika_panda/LICENSE` と `assets/MENAGERIE_LICENSE` を同梱しています。モデルファイルは無改変です。

本リポジトリのコード（上記モデルファイルを除く）は MIT License（`LICENSE`）です。
