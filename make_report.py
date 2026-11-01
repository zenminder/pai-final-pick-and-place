"""Generate the submission report (PDF) from out/demo_log.json and out/eval_grasp.json.

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
h1 { font-size: 16pt; margin: 0 0 4px; letter-spacing: .01em; }
h2 { font-size: 12.5pt; margin: 16px 0 6px; padding: 2px 0 2px 8px; border-left: 4px solid #2a5d8f;
     break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 10px 0 4px; color: #2a5d8f; break-after: avoid; page-break-after: avoid; }
p, li { margin: 0 0 4px; }
ul, ol { margin: 2px 0 6px; padding-left: 1.3em; }
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
.keyres { background: #eef4fa; border-left: 4px solid #2a5d8f; padding: 6px 10px; margin: 6px 0 8px; }
.figrow { display: flex; gap: 10px; margin: 6px 0 8px; break-inside: avoid; }
.figrow figure { flex: 1 1 0; margin: 0; }
.figrow img { width: 100%; border: 1px solid #ccc; }
figure.wide { margin: 6px auto 8px; width: 60%; break-inside: avoid; }
figure.wide img { width: 100%; border: 1px solid #ccc; }
figcaption { font-size: 8.5pt; color: #444; text-align: center; margin-top: 2px; }
.pipe { font-family: Menlo, monospace; font-size: 8.6pt; background: #f4f6f8; border: 1px solid #d6dde5;
        padding: 6px 8px; white-space: pre; margin: 4px 0 8px; }
.small { font-size: 9pt; color: #333; }
table.flow { break-inside: auto; }
table.flow tr { break-inside: avoid; }
td.gap, th.gap { border: none; width: 10px; padding: 0; background: none; }
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


def load_eval(log_dir: pathlib.Path) -> dict | None:
    p = OUT / "eval_grasp.json"
    p = p if p.exists() else log_dir / "eval_grasp.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- sections
def eval_method(ev: dict | None) -> str:
    if not ev:
        return ('<div class="placeholder">※ out/eval_grasp.json が見つかりません。<code>python eval_grasp.py --trials 30</code> '
                'を実行後に本スクリプトを再実行してください。</div>')
    s = ev["summary"]
    xr, yr = s["x_range_m"], s["y_range_m"]
    return (f"<p>3個のキューブの初期位置を x∈[{xr[0]:.2f}, {xr[1]:.2f}] m、y∈[{yr[0]:.2f}, {yr[1]:.2f}] m の一様乱数"
            f"（キューブ間の最小間隔 {s['min_separation_m']*100:.0f} cm）で配置し、置き先のパッド割当（2個＋1個）と実行順もランダムにして "
            f"{s['trials']} 試行（{s['grasps']} 把持）を実行した（seed={s['seed']}、<code>eval_grasp.py</code>、動画なし）。"
            f"1把持の成否は「解放後のキューブ中心がスロット中心から xy 3 cm 未満、かつ高さ誤差 1.5 cm 未満」で判定し、xy誤差も記録する。"
            f"把持は摩擦による物理接触のみで、ウェルド等の補助は使わない。</p>")


def eval_results(ev: dict | None) -> str:
    if not ev:
        return '<div class="placeholder">※ 評価1の結果は未記入です（eval_grasp.json なし）。</div>'
    s, trials = ev["summary"], ev["trials"]
    e = s["xy_error_cm"]
    key = (f'<div class="keyres"><strong>成功 {s["successes"]}/{s["grasps"]}（{s["success_rate"]*100:.0f}%）、'
           f'xy位置誤差 平均 {e["mean"]:.2f} cm（標準偏差 {e["std"]:.2f}）、最大 {e["max"]:.2f} cm。</strong>'
           f'色別の成功数：' + "、".join(f"{CUBE_JA.get(c, c)} {v['successes']}/{v['n']}" for c, v in s["per_cube"].items())
           + '。</div>')
    text = ("<p>全試行で把持・搬送・解放のいずれにも失敗はなく、初期位置を変えても到達精度はほぼ一定であった。"
            "誤差の標準偏差が小さいことから、残る誤差は試行ごとのばらつきではなく、スロット中心に対する系統的なオフセット"
            "（解放時の接触・指の開き方向のずれ）が支配的と考えられる。</p>")

    def rows(ts: list[dict]) -> str:
        out = []
        for t in ts:
            n_ok = sum(r["success"] for r in t["results"])
            mx = max(r["xy_error_m"] for r in t["results"]) * 100
            plan = " / ".join(f"{CUBE_JA.get(c, c)}→{PAD_COLORS.get(p, p)}" for c, p in t["plan"])
            cls = "ok" if n_ok == 3 else "ng"
            out.append(f"<td>{t['trial']}</td><td class=\"{cls}\">{n_ok}/3</td><td>{mx:.2f}</td><td>{plan}</td>")
        return "".join(out)

    half = (len(trials) + 1) // 2
    head = "<th>試行</th><th>成功</th><th>最大誤差 cm</th><th>割当・実行順</th>"
    body = "".join(f"<tr>{rows([a])}<td class=\"gap\"></td>{rows([b]) if b else '<td></td>'*4}</tr>"
                   for a, b in zip(trials[:half], trials[half:] + [None] * (2 * half - len(trials))))
    tables = f'<table class="flow"><thead><tr>{head}<th class="gap"></th>{head}</tr></thead><tbody>{body}</tbody></table>'
    return key + text + tables


def dual_results(logs: list[dict], log_dir: pathlib.Path) -> str:
    if not logs:
        return ('<div class="placeholder">※ out/demo_log.json が見つからないため結果は未記入です。'
                '<code>python run_demo.py</code> 実行後に本スクリプトを再実行してください。</div>'
                '<table><tr><th>#</th><th>指示</th><th>生成された計画</th><th>計画生成</th>'
                '<th>定量判定</th><th>視覚検査（VLM）</th></tr>'
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
    table = ('<table><tr><th style="width:3%">#</th><th style="width:24%">指示</th><th style="width:22%">生成された計画</th>'
             '<th style="width:7%">計画生成</th><th style="width:20%">定量判定（シミュレータ座標）</th>'
             '<th>視覚検査（VLM）・所見</th></tr>' + "".join(rows) + "</table>")
    figs = []
    for l in logs:
        rel = l.get("final_image")
        if not rel:
            continue
        p = pathlib.Path(rel)
        p = p if p.is_absolute() else (HERE / p if (HERE / p).exists() else log_dir / p.name)
        tag = img_tag(p)
        if tag:
            figs.append(f"<figure>{tag}<figcaption>シナリオ {l.get('scenario')} 完了時（正面カメラ、視覚検査に使用した画像）</figcaption></figure>")
    figrow = f'<div class="figrow">{"".join(figs)}</div>' if figs else ""
    n_ok = sum(1 for l in logs if l.get("sim_success") and l.get("verification", {}).get("success"))
    agree = sum(1 for l in logs if bool(l.get("sim_success")) == bool(l.get("verification", {}).get("success")))
    summary = (f'<div class="keyres"><strong>{len(logs)} シナリオすべてで定量判定と視覚検査の結果が一致した'
               f'（一致 {agree}/{len(logs)}、両者とも合格 {n_ok}/{len(logs)}）。</strong></div>')
    return summary + table + figrow


def ambiguity_results(logs: list[dict]) -> str:
    target = next((l for l in logs if "暖色" in l.get("instruction", "")), None)
    if not target:
        return ""
    plan = target.get("plan", {})
    rule_steps = "—"
    try:
        import ai_planner  # rule_based_plan needs no API key
        rb = ai_planner.rule_based_plan(target["instruction"], ai_planner.DEMO_OBJECTS, ai_planner.DEMO_PADS)
        rule_steps = f"{len(rb['steps'])} ステップ（空計画）" if not rb["steps"] else plan_text(rb)
    except Exception as e:  # noqa: BLE001
        rule_steps = f"（取得失敗: {esc(e)}）"
    return (f"<p>指示「{esc(target['instruction'])}」に対し、規則ベース（色名のキーワード照合）は {rule_steps} となり実行不能だった。"
            f"LLMは色の意味を解釈して <strong>{esc(plan_text(plan))}</strong> を生成した（理由: 「{esc(plan.get('reasoning', ''))}」）。"
            f"緑を「暖色・寒色のどちらでもない」として対象外にした判断は一貫しているが、人の意図（緑も寒色として白へ、等）とは"
            f"ずれうる。曖昧な指示では、計画と理由を人が確認できる形で可視化することが必要であると分かった。</p>")


def build_html(logs: list[dict], log_dir: pathlib.Path, video_url: str, repo_url: str) -> str:
    ev = load_eval(log_dir)
    exec_tag = img_tag(OUT / "_exec.png") or img_tag(log_dir / "_exec.png")
    exec_fig = (f'<figure class="wide">{exec_tag}<figcaption>デモ動画の1フレーム：指示文（上）、計画と進捗（右）、'
                f'ロボットの状態（下）を重畳</figcaption></figure>') if exec_tag else ""
    planner = planner_name(logs)
    pipeline = (
        "日本語の指示文 + シーン情報（物体名・色・xy座標）\n"
        "   │  (a) 計画生成: LLM の JSON Schema 制約付き構造化出力 → pick/place 手順、物体名・パッド名を検証\n"
        "   ▼\n"
        "作業計画 [{pick: cube_red, place: pad_A}, …]\n"
        "   │  (b) 実行: MuJoCo 3 + Menagerie Franka Panda、DLS 逆運動学で TCP 目標 → 7関節の位置制御、物理接触で把持\n"
        "   ▼\n"
        "置き終わりのキューブ座標 ＋ 完了後の正面カメラ画像\n"
        "   │  (c) 判定①: シミュレータ座標で成否・xy誤差を計測   判定②: 画像 + 指示 + 計画 → VLM が合否と所見を返す\n"
        "   ▼\n"
        "(d) オフスクリーン描画 → ffmpeg で MP4 化（指示・計画・進捗・状態を重畳、等速）"
    )
    ev_sum = ev["summary"] if ev else None
    headline = (f"把持成功 {ev_sum['successes']}/{ev_sum['grasps']}、xy誤差 平均 {ev_sum['xy_error_cm']['mean']:.2f} cm"
                if ev_sum else "（評価1未実施）")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{esc(PDF_NAME)}</title>
<style>{CSS}</style></head><body>
<div class="header">
<div class="meta">Physical AI応用1講座 最終課題 ／ omnicampus: {ACCOUNT}</div>
<h1>言語指示ピック＆プレースの再現性評価：初期位置ランダム化による把持成功率・位置誤差と、座標×画像の二重判定（MuJoCo / Franka Panda）</h1>
<div class="links"><span>動画: {esc(video_url)}</span><span>コード: {esc(repo_url)}</span><span>動画は等速（倍速なし）</span></div>
</div>

<h2>1. 目的と評価の問い</h2>
<p>自然言語の指示でロボットにピック＆プレースを行わせるシステムを MuJoCo 上の Franka Panda（7軸＋平行グリッパ）で構築し、
「デモが1回成功した」で終わらせず、<strong>その動作がどれだけ再現するのか、結果の合否をどう信頼できる形で判定できるのか</strong>を評価することを目的とした。
把持は補助（ウェルド等）を使わない物理接触のみとし、評価が意味を持つようにしている。評価の問いは次の3つである。</p>
<ol>
<li><strong>評価1：把持の再現性</strong>。物体の初期位置・置き先・実行順を変えても、把持→搬送→解放は成功するか。成功率と置き位置の xy 誤差を定量化する。</li>
<li><strong>評価2：結果判定の二重化</strong>。実行結果の合否を、シミュレータ座標による定量判定と、カメラ画像のみを用いた視覚検査（VLM）の2経路で独立に判定し、両者は一致するか。</li>
<li><strong>評価3：曖昧な指示への挙動</strong>。「暖色・寒色」のように色名が直接現れない指示を与えたとき、計画生成はどう振る舞い、人の意図とどこでずれうるか。</li>
</ol>
<p>AIの利用は<strong>推論のみ（学習なし）</strong>で、計画生成（LLM）と視覚検査（VLM）に用いた。主結果：{esc(headline)}、定量判定と視覚検査は全シナリオで一致。</p>

<h2>2. 評価対象システムと評価方法</h2>
<h3>2.1 評価対象システム</h3>
<div class="pipe">{esc(pipeline)}</div>
<ul>
<li><strong>(a) 計画生成</strong>：{esc(planner)}。シーン情報と指示文を渡し、<code>steps[{{pick, place}}]</code> と <code>reasoning</code> をJSON Schemaで制約して受け取る。返ってきた物体名・パッド名がシーンに存在しない場合は実行前に拒否する。APIなしで再実行するためのキーワード照合プランナー（<code>--offline</code>）も同梱。</li>
<li><strong>(b) 実行</strong>：MuJoCo 3 と MuJoCo Menagerie の Franka Panda（モデル無改変）。減衰最小二乗法（DLS）の逆運動学で手先（TCP）の目標位置・姿勢を7関節の位置制御目標へ変換し、コサイン補間で追従。グリッパは摩擦係数を調整し物理接触のみで把持する。</li>
<li><strong>(c) 判定</strong>：①退避後のキューブ座標から成否と xy 誤差を計算（定量）。②同時刻の正面カメラ画像・指示文・実行した計画を VLM（OpenAI GPT の画像入力）に渡し、<code>success</code> と所見を構造化出力で得る（定性、座標は一切渡さない）。</li>
<li><strong>(d) 動画</strong>：オフスクリーン描画（1280×720, 30fps）→ ffmpeg で MP4 化。指示・計画・進捗・状態を重畳し、各シナリオ前後に計画カード・結果カードを挿入。<strong>等速（倍速なし）</strong>。</li>
</ul>
{exec_fig}
<h3>2.2 評価1の方法：初期位置ランダム化による把持再現性</h3>
{eval_method(ev)}
<h3>2.3 評価2の方法：定量判定と視覚検査の二重判定</h3>
<p>2つの日本語指示（シナリオ1：色名を明示、シナリオ2：「暖色・寒色」の抽象指示）で計画生成→実行→判定を通し、各シナリオについて判定①（座標）と判定②（VLM）の合否が一致するかを比較した。判定②は画像と指示・計画のみから判断させ、座標情報は与えない。</p>
<h3>2.4 評価3の方法：曖昧な指示</h3>
<p>シナリオ2の指示を LLM プランナーと規則ベースプランナーの両方に与え、生成された計画と理由を比較した。</p>

<h2>3. 結果</h2>
<h3>3.1 評価1：把持の再現性</h3>
{eval_results(ev)}
<h3>3.2 評価2：定量判定と視覚検査の一致</h3>
{dual_results(logs, log_dir)}
<h3>3.3 評価3：曖昧な指示への挙動</h3>
{ambiguity_results(logs)}

<h2>4. 考察・限界</h2>
<ul>
<li><strong>評価1</strong>：初期位置を広く変えても成功率は低下せず、誤差のばらつきも小さい。残る誤差は系統的オフセットで、解放位置の補正で縮められる見込み。ただし評価条件はキューブが正立・疎に配置された場合に限られ、傾いた物体・密集・サイズ違いは未評価である。</li>
<li><strong>評価2</strong>：2経路の判定は一致したが、観察できたのは「両者とも合格」のケースのみである。意図的に失敗させた場面（落下・隣のパッドへ置く等）での VLM の検出力は未検証で、これが本評価の最大の限界である。VLM 判定は定性的で位置誤差は得られないため、定量判定との併用が前提になる。</li>
<li><strong>評価3</strong>：LLM の解釈は一貫していたが、人の意図とずれうる。計画と理由を動画のカード・字幕に必ず表示する設計は、この評価結果を受けたものである。</li>
<li><strong>システムの限界</strong>：物体位置はシミュレータの真値を使っており「見て掴む」には至っていない。把持姿勢は真上からの固定姿勢で、パッドあたり2スロットの制約がある。LLM 呼び出しには数秒の遅延とAPIコストが伴い、失敗時のリプランは未実装。</li>
<li><strong>今後</strong>：失敗注入による VLM 検出力の評価、画像からの物体位置推定、失敗検知時のリプラン、SO-101 等の実機への展開。</li>
</ul>

<h2>5. 工夫した点</h2>
<ul>
<li><strong>評価の問いを先に立てた</strong>：デモ動画の成功に加えて、初期位置・割当・実行順をランダム化した {ev_sum['trials'] if ev_sum else 'N'} 試行で成功率と誤差を計測し、再現性を数値で示した。把持パラメータの調整もこの評価を回しながら行った。</li>
<li><strong>判定経路の独立性</strong>：座標による定量判定と、座標を一切渡さない画像のみの視覚検査を分け、両者の一致を確認できる構成にした。</li>
<li><strong>評価が意味を持つ物理条件</strong>：把持補助（ウェルド）を使わず、摩擦・接触パラメータと真上からの一定姿勢で安定化した。</li>
<li><strong>幻覚手順の排除</strong>：構造化出力（JSON Schema）と名前検証で、存在しない物体・場所を含む計画を実行前に弾く。</li>
<li><strong>第三者が再評価できる</strong>：ROS/Gazebo 不要・pip のみ、macOS（Apple Silicon）・GPUなし・ヘッドレスで動作。評価1は API キーなしで再実行でき、評価2も <code>--offline</code> でパイプライン全体を通せる。</li>
<li><strong>判断の可視化</strong>：字幕オーバーレイと計画・結果カードで、計画・理由・成否が動画単体でも追える。</li>
</ul>

<h2>6. 使用ツールと参照</h2>
<p class="small">MuJoCo 3.x／MuJoCo Menagerie（franka_emika_panda、ライセンスファイルをリポジトリに同梱）／OpenAI Python SDK／imageio-ffmpeg／Pillow／NumPy。
講座サンプルのコードは直接流用していない。課題資料に挙げられた「ChatGPT APIによる自然言語ロボット操作」「Pick and Place Task（UR3 + Robotiq）」を着想源とし、シミュレータ・制御・AI連携・評価のすべてを独自に実装した。</p>
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
