# 言語指示ピック＆プレースの再現性評価 — MuJoCo / Franka Panda

Physical AI応用1講座 最終課題（omnicampus: zenminder）

自然言語（日本語）の指示でロボットにピック＆プレースをさせるシステムを MuJoCo 上の Franka Panda で構築し、**「デモが1回成功した」で終わらせず、動作がどれだけ再現するか・結果の合否をどう信頼できる形で判定できるか**を評価したリポジトリです。把持は補助（ウェルド等）を使わない物理接触のみ。AIの利用は**推論のみ（学習なし）**で、計画生成（LLM）と画像による結果検査（VLM）に用いています。ROS/Gazebo は使わず、pip で入るライブラリだけで macOS（Apple Silicon）・GPUなし・ヘッドレスで再現できます。

- デモ動画（等速・倍速なし）: （提出後に追記）
- 使用ツール: MuJoCo 3.x / MuJoCo Menagerie (Franka Panda) / OpenAI Python SDK / imageio-ffmpeg / Pillow

## 評価の問いと主な結果

| # | 評価の問い | 方法 | 結果 |
|---|---|---|---|
| 1 | **把持の再現性**：初期位置・置き先・実行順を変えても把持→搬送→解放は成功するか | 3キューブの初期位置を一様乱数で配置（最小間隔8cm）、パッド割当と実行順もランダム化、30試行・90把持（`eval_grasp.py`） | 成功 **90/90（100%）**、xy位置誤差 平均 **0.60 cm**（標準偏差 0.06、最大 0.78 cm） |
| 2 | **結果判定の二重化**：シミュレータ座標による定量判定と、画像のみを見る視覚検査（VLM）は一致するか | 2シナリオで両判定を独立に実施（VLMには座標を渡さない） | 2/2 シナリオで一致（両者とも合格） |
| 3 | **曖昧な指示への挙動**：「暖色・寒色」のように色名が現れない指示で計画生成はどう振る舞うか | 同じ指示を LLM と規則ベース（`--offline`）に与えて比較 | 規則ベースは空計画。LLMは赤→黄・青→白を生成し、緑を「どちらでもない」として対象外にした（人の意図とずれうる） |

限界：評価2で観察できたのは「両者とも合格」のケースのみで、意図的に失敗させた場面での VLM の検出力は未検証です。評価1もキューブが正立・疎に配置された条件に限られます。詳細はレポート PDF（`make_report.py` で生成）を参照してください。

## 評価対象システム

```
日本語の指示文 + シーン情報（物体名・色・xy座標）
   │ (a) 計画生成 … LLM の JSON Schema 制約付き構造化出力で pick/place 手順を生成、物体名・パッド名を検証
   ▼
作業計画 [{pick: cube_red, place: pad_A}, …]
   │ (b) 実行 … MuJoCo + Franka Panda、DLS逆運動学で TCP目標 → 7関節の位置制御、物理接触で把持
   ▼
置き終わりのキューブ座標 ＋ 完了後の正面カメラ画像
   │ (c) 判定① 座標で成否・xy誤差を計測 ／ 判定② 画像 + 指示 + 計画 → VLM が合否と所見
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
.venv/bin/python eval_grasp.py --trials 30 --seed 0                                          # 評価1: 把持再現性（APIキー不要、動画なし）→ out/eval_grasp.json
.venv/bin/python run_demo.py                                  # 評価2・3: 既定の2シナリオ（OpenAI GPTで計画生成＋画像検査）→ out/demo.mp4, out/demo_log.json
.venv/bin/python run_demo.py --offline                        # APIなし: キーワード規則プランナーで同じパイプラインを実行
.venv/bin/python run_demo.py --instruction "青いキューブを白いパッドへ"   # 任意の指示（複数指定可）
.venv/bin/python pick_place.py --plan "red:pad_A,green:pad_B,blue:pad_A" --video out/test.mp4   # 計画を直接与えて実行
.venv/bin/python make_report.py --video-url <YouTube URL> --repo-url <GitHub URL>              # レポートPDFを生成 → out/説明資料_PAI最終課題_zenminder.pdf
```

使用モデルは環境変数 `PAI_MODEL` で変更できます（既定 `gpt-5.5`）。

## ファイル構成

| パス | 内容 |
|---|---|
| `scene.xml` | テーブル・3キューブ・2パッド・カメラ。Menagerie の `panda.xml` を無改変で include |
| `pick_place.py` | `PandaSim`: 物理ステップ、DLS逆運動学、動作プリミティブ、成否・誤差の計測、フレーム録画 |
| `eval_grasp.py` | 評価1: 初期位置・割当・実行順をランダム化したN試行で把持成功率と位置誤差を計測 |
| `ai_planner.py` | 計画生成 `plan()`（LLM）、視覚検査 `verify()`（VLM）、オフライン規則プランナー |
| `run_demo.py` | 評価2・3: 指示 → 計画 → 実行 → 二重判定 → 字幕付きMP4 を一気通貫で生成 |
| `make_report.py` | `out/eval_grasp.json` と `out/demo_log.json` からレポートPDFを生成（headless Chrome） |
| `assets/franka_emika_panda/` | MuJoCo Menagerie の Franka Panda モデル（ライセンス同梱） |
| `out/` | 生成物（動画・画像・ログ・PDF）。git 管理外 |

## AIの利用について

- 計画生成と視覚検査は OpenAI API（GPT）の推論のみを利用し、学習は行っていません。
- 評価1（`eval_grasp.py`）は AI を使わず、APIキーなしで再実行できます。`run_demo.py --offline` はキーワード照合で計画するため、こちらもキーなしでパイプライン全体を通せます。
- 物体位置はシミュレータの真値を使用しています（画像からの位置推定は未実装）。

## ライセンス

- Franka Panda モデルは [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) のもので、`assets/franka_emika_panda/LICENSE` と `assets/MENAGERIE_LICENSE` を同梱しています。モデルファイルは無改変です。

本リポジトリのコード（上記モデルファイルを除く）は MIT License（`LICENSE`）です。
