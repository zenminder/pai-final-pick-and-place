"""End-to-end demo: Japanese instruction -> LLM plan -> Panda executes -> VLM verifies -> annotated MP4.

    python run_demo.py                      # 2 default scenarios, OpenAI GPT planner + verifier
    python run_demo.py --offline            # keyword planner, no API access needed
    python run_demo.py --instruction "..."  # your own instruction(s), repeatable

The video is rendered at real time (1x); no speed-up is applied.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ai_planner
from pick_place import CUBES, PADS, PandaSim

HERE = pathlib.Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 30
PAD_COLORS = {"pad_A": "yellow", "pad_B": "white"}
CUBE_BY_BODY = {body: color for color, body in CUBES.items()}  # cube_red -> red
JA_STATUS = {"approach": "接近", "descend": "下降", "grasp": "把持", "lift": "持ち上げ", "carry": "搬送",
             "release": "リリース", "retreat": "退避", "pick": "ピック", "place": "プレース"}
FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
DEFAULT_INSTRUCTIONS = [
    "赤と青のキューブを黄色のパッドに、緑のキューブを白いパッドに置いてください",
    "暖色のキューブは黄色のパッドへ、寒色のキューブは白いパッドへ移してください",
]


# ----------------------------------------------------------------------------- text helpers
def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=fnt) > max_w or ch == "\n":
            lines.append(cur)
            cur = "" if ch == "\n" else ch
        else:
            cur += ch
    return lines + [cur] if cur else lines


def panel(draw: ImageDraw.ImageDraw, box, alpha=170) -> None:
    draw.rectangle(box, fill=(10, 12, 20, alpha))


def ja_status(text: str) -> str:
    for en, ja in JA_STATUS.items():
        text = text.replace(en, ja)
    return text


class Overlay:
    """Draws instruction, plan progress and robot status on every recorded frame."""

    def __init__(self, scenario: int, instruction: str, steps: list[tuple[str, str]], planner_label: str):
        self.scenario, self.instruction, self.steps, self.planner_label = scenario, instruction, steps, planner_label
        self.current, self.done = -1, set()
        self.f_big, self.f_mid, self.f_small = font(30), font(24), font(19)

    def __call__(self, img: np.ndarray, sim: PandaSim) -> np.ndarray:
        im = Image.fromarray(img).convert("RGBA")
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # top: instruction
        lines = wrap(d, f"指示: {self.instruction}", self.f_big, W - 60)
        panel(d, (0, 0, W, 24 + 38 * len(lines)))
        for i, ln in enumerate(lines):
            d.text((30, 14 + 38 * i), ln, font=self.f_big, fill=(255, 255, 255, 255))
        # right: plan
        px, py = W - 420, 24 + 38 * len(lines) + 16
        panel(d, (px - 16, py - 10, W - 16, py + 40 + 34 * len(self.steps)))
        d.text((px, py), f"LLMの計画 ({self.planner_label})", font=self.f_small, fill=(180, 220, 255, 255))
        for i, (cube, pad) in enumerate(self.steps):
            mark = "完了" if i in self.done else ("実行中" if i == self.current else "待機")
            color = (120, 255, 140, 255) if i in self.done else (
                (255, 230, 120, 255) if i == self.current else (255, 255, 255, 255))
            d.text((px, py + 30 + 34 * i), f"[{mark}] {i+1}. {cube} → {pad} ({PAD_COLORS[pad]})",
                   font=self.f_mid, fill=color)
        # bottom: status
        panel(d, (0, H - 54, W, H))
        d.text((30, H - 42), f"状態: {ja_status(sim.status)}", font=self.f_mid, fill=(255, 255, 255, 255))
        d.text((W - 330, H - 40), f"シナリオ {self.scenario}   t = {sim.t:5.1f} s (等速)", font=self.f_small,
               fill=(200, 200, 200, 255))
        return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def card(writer, title: str, body: list[str], seconds: float, bg=(18, 22, 34)) -> None:
    """Append a static text slide to the video."""
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f_title, f_body = font(44), font(27)
    y = 90
    for ln in wrap(d, title, f_title, W - 160):
        d.text((80, y), ln, font=f_title, fill=(255, 255, 255))
        y += 56
    y += 20
    for item in body:
        for ln in wrap(d, item, f_body, W - 160):
            d.text((80, y), ln, font=f_body, fill=(220, 225, 235))
            y += 40
        y += 8
    frame = np.asarray(im)
    for _ in range(int(seconds * FPS)):
        writer.append_data(frame)


# ----------------------------------------------------------------------------- scenario
def scene_description(sim: PandaSim):
    objects = {body: {"color": color, "pos": sim.body_pos(body)[:2].round(3).tolist()} for color, body in CUBES.items()}
    pads = {pad: {"color": PAD_COLORS[pad], "pos": sim.body_pos(pad)[:2].round(3).tolist()} for pad in PADS}
    return objects, pads


def run_scenario(idx: int, instruction: str, writer, offline: bool, speed: float, out_dir: pathlib.Path) -> dict:
    sim = PandaSim(camera="overview", speed=speed)
    objects, pads = scene_description(sim)

    t0 = time.time()
    plan = (ai_planner.rule_based_plan if offline else ai_planner.plan)(instruction, objects, pads)
    plan_sec = time.time() - t0
    steps = [(CUBE_BY_BODY[s["pick"]], s["place"]) for s in plan["steps"]]
    label = "offline" if offline else ai_planner.MODEL
    print(f"[scenario {idx}] plan ({label}, {plan_sec:.1f}s): {steps}\n  reasoning: {plan['reasoning']}")

    card(writer, f"シナリオ {idx}: 「{instruction}」",
         [f"プランナー: {label}（推論のみ）",
          "シーン情報（物体名・色・座標）と指示文をLLMに渡し、ピック＆プレース手順をJSONで生成。",
          "計画: " + " / ".join(f"{c} → {p}({PAD_COLORS[p]})" for c, p in steps),
          f"LLMの理由: {plan['reasoning']}"], 7)

    overlay = Overlay(idx, instruction, steps, label)
    sim.overlay = overlay
    sim.attach_writer(writer)
    sim.hold(0.6)
    results = []
    for i, (cube, pad) in enumerate(steps):
        overlay.current = i
        results.append(sim.pick_and_place(cube, pad))
        overlay.done.add(i)
    sim.log("plan finished: park")
    sim.park()
    sim.hold(0.8)
    sim.writer = None

    final_png = out_dir / f"scenario{idx}_final.png"
    sim.save_png(final_png, "front")
    sim_ok = bool(results) and all(r["success"] for r in results)
    if not results:
        verification = {"success": False, "observation": "計画が空のため実行なし（プランナーが指示を解釈できず）"}
    elif offline:
        verification = {"success": sim_ok, "observation": "（オフライン）シミュレータ座標で判定"}
    else:
        verification = ai_planner.verify(str(final_png), instruction, plan["steps"])
    print(f"[scenario {idx}] sim success={sim_ok}  VLM: {verification}")

    card(writer, f"シナリオ {idx} 結果",
         [f"{i+1}. {r['cube']} → {r['pad']}: {'成功' if r['success'] else '失敗'}（位置誤差 {r['xy_error_m']*100:.1f} cm）"
          for i, r in enumerate(results)]
         + [f"VLM検証（最終画像を{label}が判定）: {'合格' if verification['success'] else '不合格'}",
            f"VLMの所見: {verification['observation']}"], 7)

    return {"scenario": idx, "instruction": instruction, "plan": plan, "plan_seconds": round(plan_sec, 2),
            "results": results, "sim_success": sim_ok, "verification": verification,
            "final_image": str(final_png.relative_to(HERE))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", action="append", help="repeatable; defaults to 2 built-in scenarios")
    ap.add_argument("--offline", action="store_true", help="keyword planner instead of the OpenAI API")
    ap.add_argument("--video", default="out/demo.mp4")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 slows robot motions")
    args = ap.parse_args()

    out_dir = (HERE / args.video).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    instructions = args.instruction or DEFAULT_INSTRUCTIONS
    wall = time.time()
    writer = imageio.get_writer(str(HERE / args.video), fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    card(writer, "言語指示で動くピック＆プレース",
         ["LLMプランナー × MuJoCo（Franka Panda）× VLM検証",
          "1. 日本語の指示文 → LLMが作業計画（JSON）を生成",
          "2. 逆運動学（DLS-IK）で7軸アームを制御し、物理シミュレーションで把持・搬送",
          "3. 完了後のカメラ画像をVLMが検査し、指示を満たしたか判定",
          "動画は等速（1x）です。"], 6)
    logs = [run_scenario(i + 1, text, writer, args.offline, args.speed, out_dir) for i, text in enumerate(instructions)]
    n_ok = sum(l["sim_success"] and l["verification"]["success"] for l in logs)
    card(writer, "まとめ", [f"{len(logs)} シナリオ中 {n_ok} シナリオで計画・実行・検証がすべて成功",
                          "学習なし・推論のみ。シーン情報は物体名と座標のみをテキストで与え、把握は物理接触で実現。",
                          "コードとレポートはGitHubリポジトリを参照。"], 5)
    writer.close()
    (out_dir / "demo_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2))
    print(json.dumps({"video": args.video, "wall_seconds": round(time.time() - wall, 1),
                      "scenarios_ok": n_ok, "total": len(logs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
