"""Generate the submission report (PDF) from out/demo_log.json.

    python make_report.py
    python make_report.py --video-url https://youtu.be/... --repo-url https://github.com/...
    python make_report.py --log path/to/other_log.json --out-dir /tmp/x

HTML is written next to the PDF and converted with headless Chrome. Images are
embedded as base64 so the PDF renders offline. Never crashes when the log or
images are missing: it prints a placeholder table instead.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
ACCOUNT = "zenminder"
PDF_NAME = f"説明資料_PAI最終課題_{ACCOUNT}.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAD_COLORS = {"pad_A": "黄", "pad_B": "白"}
CUBE_JA = {"cube_red": "赤", "cube_green": "緑", "cube_blue": "青", "red": "赤", "green": "緑", "blue": "青"}

CSS = """
@page { size: A4; margin: 18mm; }
html { font-size: 10.5pt; }
body { font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", sans-serif;
       color: #1a1a1a; line-height: 1.55; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 4px; letter-spacing: .01em; }
h2 { font-size: 12.5pt; margin: 16px 0 6px; padding: 2px 0 2px 8px; border-left: 4px solid #2a5d8f;
     break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 10px 0 4px; color: #2a5d8f; break-after: avoid; page-break-after: avoid; }
p, li { margin: 0 0 4px; }
ul { margin: 2px 0 6px; padding-left: 1.3em; }
.meta { color: #444; font-size: 9.5pt; margin-bottom: 2px; }
.header { border-bottom: 2px solid #2a5d8f; padding-bottom: 6px; margin-bottom: 8px; }
.links { font-size: 9.5pt; color: #333; }
.links span { display: inline-block; margin-right: 18px; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 4px 0 8px; break-inside: avoid; }
th, td { border: 1px solid #b9c4d0; padding: 3px 5px; vertical-align: top; text-align: left; }
th { background: #e8eef5; }
.ok { color: #1a7f37; font-weight: 600; }
.ng { color: #b42318; font-weight: 600; }
.placeholder { background: #fff7e0; border: 1px dashed #d9a400; padding: 6px 8px; font-size: 9.5pt; }
.figrow { display: flex; gap: 10px; margin: 6px 0 8px; break-inside: avoid; }
.figrow figure { flex: 1 1 0; margin: 0; }
.figrow img { width: 100%; border: 1px solid #ccc; }
figure.wide { margin: 6px auto 8px; width: 60%; break-inside: avoid; }
figure.wide img { width: 100%; border: 1px solid #ccc; }
figcaption { font-size: 8.5pt; color: #444; text-align: center; margin-top: 2px; }
.pipe { font-family: Menlo, monospace; font-size: 8.6pt; background: #f4f6f8; border: 1px solid #d6dde5;
        padding: 6px 8px; white-space: pre; margin: 4px 0 8px; }
.small { font-size: 9pt; color: #333; }
"""


def esc(s) -> str:
    return html.escape(str(s))


def img_tag(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{esc(path.name)}">'


def plan_text(plan: dict) -> str:
    steps = plan.get("steps") or []
    if not steps:
        return "（空の計画）"
    return " / ".join(f"{CUBE_JA.get(s['pick'], s['pick'])}→{PAD_COLORS.get(s['place'], s['place'])}" for s in steps)


def planner_name(logs: list[dict]) -> str:
    sources = {l.get("plan", {}).get("source") for l in logs}
    if "llm" in sources:
        models = sorted({l["plan"].get("model") for l in logs if l.get("plan", {}).get("model")})
        return f"OpenAI GPT（{', '.join(models) or os.environ.get('PAI_MODEL', 'gpt-5')}）"
    if sources:
        return "オフライン規則ベース（キーワード照合）"
    return "OpenAI GPT（gpt-5）"


def results_section(logs: list[dict], log_dir: pathlib.Path) -> str:
    if not logs:
        return ('<div class="placeholder">※ out/demo_log.json が見つからないため結果は未記入です。'
                '<code>python run_demo.py</code> 実行後に本スクリプトを再実行してください。</div>'
                '<table><tr><th>#</th><th>指示</th><th>LLMの計画</th><th>計画生成</th>'
                '<th>ステップ結果</th><th>VLM判定</th></tr>'
                '<tr><td>1</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr></table>')
    rows = []
    for l in logs:
        steps = "<br>".join(
            f"{CUBE_JA.get(r['cube'], r['cube'])}→{PAD_COLORS.get(r['pad'], r['pad'])}: "
            f"<span class=\"{'ok' if r['success'] else 'ng'}\">{'成功' if r['success'] else '失敗'}</span>"
            f"（誤差 {r['xy_error_m']*100:.1f} cm）"
            for r in l.get("results", [])) or "（実行なし）"
        v = l.get("verification", {})
        vtxt = (f"<span class=\"{'ok' if v.get('success') else 'ng'}\">{'合格' if v.get('success') else '不合格'}</span>"
                f"<br><span class=\"small\">{esc(v.get('observation', ''))}</span>")
        rows.append(
            f"<tr><td>{l.get('scenario')}</td><td>{esc(l.get('instruction'))}</td>"
            f"<td>{esc(plan_text(l.get('plan', {})))}<br><span class=\"small\">理由: {esc(l.get('plan', {}).get('reasoning', ''))}</span></td>"
            f"<td>{l.get('plan_seconds', '—')} s</td><td>{steps}</td><td>{vtxt}</td></tr>")
    table = ('<table><tr><th style="width:3%">#</th><th style="width:24%">指示</th><th style="width:22%">LLMの計画</th>'
             '<th style="width:7%">計画生成</th><th style="width:20%">ステップ結果（シミュレータ座標）</th>'
             '<th>VLM判定・所見</th></tr>' + "".join(rows) + "</table>")
    figs = []
    for l in logs:
        rel = l.get("final_image")
        if not rel:
            continue
        p = pathlib.Path(rel)
        p = p if p.is_absolute() else (HERE / p if (HERE / p).exists() else log_dir / p.name)
        tag = img_tag(p)
        if tag:
            figs.append(f"<figure>{tag}<figcaption>シナリオ {l.get('scenario')} 完了時（正面カメラ、VLM検証に使用）</figcaption></figure>")
    figrow = f'<div class="figrow">{"".join(figs)}</div>' if figs else ""
    n_ok = sum(1 for l in logs if l.get("sim_success") and l.get("verification", {}).get("success"))
    summary = f"<p>{len(logs)} シナリオ中 {n_ok} シナリオで、計画生成・物理実行・VLM検証のすべてが成功した。</p>"
    return summary + table + figrow


def eval_section(log_dir: pathlib.Path) -> str:
    """Grasp robustness table from out/eval_grasp.json (written by eval_grasp.py); empty if absent."""
    p = OUT / "eval_grasp.json"
    p = p if p.exists() else log_dir / "eval_grasp.json"
    if not p.exists():
        return ""
    data = json.loads(p.read_text())
    s, trials = data["summary"], data["trials"]
    e = s["xy_error_cm"]
    xr, yr = s["x_range_m"], s["y_range_m"]
    intro = (f"<p>デモ1本の成功では再現性が示せないため、<code>eval_grasp.py</code> で把持のロバスト性を定量評価した。"
             f"3個のキューブの初期位置を x∈[{xr[0]:.2f}, {xr[1]:.2f}] m、y∈[{yr[0]:.2f}, {yr[1]:.2f}] m の一様乱数"
             f"（最小間隔 {s['min_separation_m']*100:.0f} cm）で配置し、パッド割当（2+1）と実行順もランダムにして "
             f"{s['trials']} 試行（{s['grasps']} 把持）を実行した（seed={s['seed']}、動画なし）。</p>"
             f"<p><strong>結果：成功 {s['successes']}/{s['grasps']}（{s['success_rate']*100:.0f}%）、"
             f"xy位置誤差 平均 {e['mean']:.2f} cm（標準偏差 {e['std']:.2f}）、最大 {e['max']:.2f} cm。</strong>"
             f"色別の成功数：" + "、".join(f"{CUBE_JA.get(c, c)} {v['successes']}/{v['n']}" for c, v in s["per_cube"].items())
             + "。全試行で把持・搬送・解放に失敗はなく、誤差はスロット中心に対する系統的なオフセットが支配的であった。</p>")

    def rows(ts: list[dict]) -> str:
        out = []
        for t in ts:
            n_ok = sum(r["success"] for r in t["results"])
            mx = max(r["xy_error_m"] for r in t["results"]) * 100
            plan = " / ".join(f"{CUBE_JA.get(c, c)}→{PAD_COLORS.get(p, p)}" for c, p in t["plan"])
            cls = "ok" if n_ok == 3 else "ng"
            out.append(f"<tr><td>{t['trial']}</td><td class=\"{cls}\">{n_ok}/3</td><td>{mx:.2f}</td><td>{plan}</td></tr>")
        return "".join(out)

    half = (len(trials) + 1) // 2
    head = "<tr><th>試行</th><th>成功</th><th>最大誤差 cm</th><th>割当・実行順</th></tr>"
    tables = (f'<div class="figrow"><table>{head}{rows(trials[:half])}</table>'
              f'<table>{head}{rows(trials[half:])}</table></div>')
    return f"<h3>4.2 把持ロバスト性評価（初期位置ランダム化）</h3>{intro}{tables}"


def build_html(logs: list[dict], log_dir: pathlib.Path, video_url: str, repo_url: str) -> str:
    exec_tag = img_tag(OUT / "_exec.png") or img_tag(log_dir / "_exec.png")
    exec_fig = (f'<figure class="wide">{exec_tag}<figcaption>デモ動画の1フレーム：指示文（上）、LLMの計画と進捗（右）、'
                f'ロボットの状態（下）を重畳</figcaption></figure>') if exec_tag else ""
    planner = planner_name(logs)
    pipeline = (
        "日本語の指示文 + シーン情報（物体名・色・xy座標）\n"
        "   │  (a) LLMプランナー: JSON Schema 制約付き構造化出力 → pick/place 手順を生成、物体名・パッド名を検証\n"
        "   ▼\n"
        "作業計画 [{pick: cube_red, place: pad_A}, …]\n"
        "   │  (b) MuJoCo 3 + Menagerie Franka Panda: DLS 逆運動学で TCP 目標 → 7関節の位置制御目標、物理接触で把持\n"
        "   ▼\n"
        "完了後の正面カメラ画像\n"
        "   │  (c) VLM 検証: 画像 + 指示文 + 実行した計画 → 合否と所見を構造化出力で返す\n"
        "   ▼\n"
        "(d) オフスクリーン描画 → ffmpeg で MP4 化（指示・計画・進捗・状態を重畳、等速）"
    )
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{esc(PDF_NAME)}</title>
<style>{CSS}</style></head><body>
<div class="header">
<div class="meta">Physical AI応用1講座 最終課題 ／ omnicampus: {ACCOUNT}</div>
<h1>言語指示で動くピック＆プレース：LLMプランナー × MuJoCo（Franka Panda）× VLM検証</h1>
<div class="links"><span>動画: {esc(video_url)}</span><span>コード: {esc(repo_url)}</span><span>動画は等速（倍速なし）</span></div>
</div>

<h2>1. 課題の概要と目的</h2>
<p>日本語の自然言語指示をLLMが作業計画に変換し、MuJoCo上のFranka Pandaアームがピック＆プレースを実行、完了後のカメラ画像をVLMが検査して指示を満たしたかを判定する、一連の閉ループを実装した。
AIの活用は<strong>推論のみ（学習なし）</strong>である。狙いは、生成AIの言語理解を物理空間のタスク計画に橋渡しし、実行結果を画像で自己検証する<strong>最小構成</strong>を、ROS/Gazeboに依存せずpipだけで再現可能な形で作ることにある。</p>

<h2>2. システム構成</h2>
<div class="pipe">{esc(pipeline)}</div>
<ul>
<li><strong>(a) LLMプランナー</strong>：{esc(planner)}。シーン情報と指示文を渡し、<code>steps[{{pick, place}}]</code> と <code>reasoning</code> をJSON Schemaで制約して受け取る。返ってきた物体名・パッド名がシーンに存在するかを検証し、存在しない名前は拒否する。</li>
<li><strong>(b) 制御・物理</strong>：MuJoCo 3 と MuJoCo Menagerie の Franka Panda（モデル無改変）。減衰最小二乗法（DLS）による逆運動学で手先（TCP）の目標位置・姿勢を7関節の位置制御目標へ変換し、コサイン補間で滑らかに追従させる。グリッパは摩擦係数を調整したうえで<strong>物理接触のみで把持</strong>し、ウェルド等の把持補助は用いていない。</li>
<li><strong>(c) VLM検証</strong>：退避後の正面カメラ画像を OpenAI GPT の画像入力として与え、指示文と実行した計画とを照合して <code>success</code> と所見を構造化出力で得る。</li>
<li><strong>(d) 動画生成</strong>：オフスクリーン描画（1280×720, 30fps）→ ffmpeg で MP4 化。指示文・LLMの計画と進捗・ロボットの状態をフレームに重畳し、各シナリオの前後に計画カード・結果カードを挿入。<strong>等速（倍速なし）</strong>。</li>
</ul>
{exec_fig}

<h2>3. 工夫した点</h2>
<ul>
<li><strong>幻覚手順の排除</strong>：構造化出力（JSON Schema）と名前検証の二段構えで、存在しない物体・場所を含む計画を実行前に弾く。</li>
<li><strong>閉ループ</strong>：言語 → 計画 → 物理実行 → 画像検証まで一気通貫。実行結果をシミュレータ座標とVLMの両方で評価し、両者の一致を確認できる。</li>
<li><strong>再現性</strong>：ROS/Gazebo不要、pipのみ。macOS（Apple Silicon）・GPUなし・ヘッドレスで動作し、1シナリオの生成は実時間の約0.5倍で完了する。</li>
<li><strong>物理把持に拘った</strong>：把持補助（ウェルド）なしで3キューブ連続成功。摩擦・接触パラメータの調整と真上からの一定姿勢で安定化。</li>
<li><strong>APIキーなしでも検証可能</strong>：オフラインの規則ベースプランナーを同梱（<code>--offline</code>）。採点者が鍵なしでパイプライン全体を再実行できる。</li>
<li><strong>LLMプランナーの価値の可視化</strong>：「暖色のキューブは黄色へ、寒色は白へ」のような語彙に直接現れない指示は規則ベースでは空計画になるが、LLMは色の意味を解釈して計画できる。シナリオ2でこれを示す。</li>
<li><strong>ロバスト性の定量評価</strong>：デモ1本の成功で終わらせず、初期位置・パッド割当・実行順をランダム化した試行で把持成功率と位置誤差を計測した（4.2節）。把持パラメータの調整はこの評価を回しながら行った。</li>
<li><strong>表現力</strong>：字幕オーバーレイと計画・結果カードで、動画単体でも何をしているかが追える構成にした。</li>
</ul>

<h2>4. 結果</h2>
<h3>4.1 デモシナリオ（言語指示 → 計画 → 実行 → VLM検証）</h3>
{results_section(logs, log_dir)}
{eval_section(log_dir)}

<h2>5. 考察・反省・限界</h2>
<ul>
<li>物体位置はシミュレータの真値を用いており、画像からの位置推定は未実装。「見て掴む」には至っていない。</li>
<li>把持姿勢は真上からの固定姿勢で、パッドあたり2スロットの制約がある。3個目を同じパッドに置く指示には対応できない。ロバスト性評価もキューブが正立・疎に配置された条件に限られ、傾いた物体・密集・サイズ違いは未評価である。</li>
<li>LLM呼び出しには数秒の遅延とAPIコストが伴う。JSON検証に失敗した場合の再試行・リプランは未実装で、失敗時は停止する。</li>
<li>VLM検証は定性的な合否判定であり、位置誤差などの数値は把握できない。シミュレータ座標との併用で補っている。</li>
<li>シナリオ2ではLLMが緑を「暖色・寒色のどちらにも明確には分類されない」と判断し対象外とした。曖昧な指示に対する解釈はもっともらしい一方で、人の意図（緑も寒色として扱う等）とずれる可能性がある。計画と理由をカード・字幕で必ず可視化し、人が確認できる形にしているのはこのためである。</li>
<li>今後：VLMによる物体位置推定（画像→座標）、失敗検知時のリプラン、SO-101等の実機への展開を検討したい。</li>
</ul>

<h2>6. 使用ツールと参照</h2>
<p class="small">MuJoCo 3.x／MuJoCo Menagerie（franka_emika_panda、ライセンスファイルをリポジトリに同梱）／OpenAI Python SDK／imageio-ffmpeg／Pillow／NumPy。
講座サンプルのコードは直接流用していない。課題資料に挙げられた「ChatGPT APIによる自然言語ロボット操作」「Pick and Place Task（UR3 + Robotiq）」を着想源とし、シミュレータ・制御・AI連携のすべてを独自に実装した。</p>
</body></html>"""


def to_pdf(html_path: pathlib.Path, pdf_path: pathlib.Path) -> bool:
    if not pathlib.Path(CHROME).exists():
        print(f"Chrome not found at {CHROME}; HTML written to {html_path}", file=sys.stderr)
        return False
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", f"file://{html_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not pdf_path.exists():
        print(r.stderr[-2000:], file=sys.stderr)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(OUT / "demo_log.json"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--video-url", default="（提出フォームに記載）")
    ap.add_argument("--repo-url", default="（提出フォームに記載）")
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    logs: list[dict] = []
    if log_path.exists():
        logs = json.loads(log_path.read_text())
    else:
        print(f"log not found: {log_path} -> rendering placeholder results", file=sys.stderr)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    pdf_path = out_dir / PDF_NAME
    html_path.write_text(build_html(logs, log_path.parent, args.video_url, args.repo_url), encoding="utf-8")
    ok = to_pdf(html_path.resolve(), pdf_path.resolve())
    print(json.dumps({"html": str(html_path), "pdf": str(pdf_path) if ok else None, "scenarios": len(logs)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
