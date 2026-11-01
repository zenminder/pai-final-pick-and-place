"""AI layer for the pick-and-place demo: LLM task planner + VLM scene verifier (OpenAI API).

Two inference-only uses of an LLM/VLM:
  1. plan()   : natural-language instruction (Japanese/English) -> JSON pick/place steps
  2. verify() : rendered camera image after execution -> did the scene satisfy the instruction?

Set OPENAI_API_KEY in the environment. Nothing here reads or stores the key itself.
`--offline` uses a keyword-based fallback planner so the demo runs without any API access.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from typing import Any

MODEL = os.environ.get("PAI_MODEL", "gpt-5.5")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"pick": {"type": "string"}, "place": {"type": "string"}},
                "required": ["pick", "place"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["steps", "reasoning"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "observation": {"type": "string"},
    },
    "required": ["success", "observation"],
    "additionalProperties": False,
}

PLANNER_SYSTEM = (
    "You are the task planner for a Franka Panda arm in a MuJoCo tabletop simulation. "
    "You receive a scene description (movable objects and target pads, each with a name, color and xy position in meters) "
    "and a human instruction, possibly in Japanese. Produce an ordered list of pick-and-place steps that satisfies the "
    "instruction. Use ONLY the object and pad names given in the scene. If an instruction mentions a relative order "
    "(e.g. nearest first), use the positions. Keep `reasoning` to one or two sentences in the same language as the instruction."
)

VERIFIER_SYSTEM = (
    "You are a vision-based inspector for a robot pick-and-place task. You see a camera image of the final tabletop scene, "
    "the original instruction, and the executed plan. Judge from the image whether each cube ended up on the pad required "
    "by the instruction. Answer `success` true only if all required placements look correct. Write `observation` in the "
    "same language as the instruction, in one or two sentences."
)


def scene_text(objects: dict[str, dict[str, Any]], pads: dict[str, dict[str, Any]]) -> str:
    def fmt(d: dict[str, dict[str, Any]]) -> str:
        return "\n".join(
            f"  - {name}: color={v['color']}, xy=({v['pos'][0]:.2f}, {v['pos'][1]:.2f})" for name, v in d.items()
        )

    return f"Movable objects:\n{fmt(objects)}\nTarget pads:\n{fmt(pads)}"


def _client():
    from openai import OpenAI  # imported lazily so --offline works without the SDK

    return OpenAI()


def _structured(client, system: str, content: list[dict], name: str, schema: dict) -> dict[str, Any]:
    """One Responses API call constrained to `schema`; returns the parsed JSON."""
    kwargs: dict[str, Any] = {}
    if MODEL.startswith(("gpt-5", "o")):  # reasoning models: keep latency low
        kwargs["reasoning"] = {"effort": "low"}
    response = client.responses.create(
        model=MODEL,
        instructions=system,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
        **kwargs,
    )
    if not response.output_text:
        raise RuntimeError(f"empty/refused response: {response.output}")
    return json.loads(response.output_text)


def plan(instruction: str, objects: dict, pads: dict, client=None) -> dict[str, Any]:
    """Ask the LLM for a pick/place plan. Returns {"steps": [...], "reasoning": str, "source": "llm"}."""
    client = client or _client()
    data = _structured(
        client, PLANNER_SYSTEM,
        [{"type": "input_text", "text": f"{scene_text(objects, pads)}\n\nInstruction: {instruction}"}],
        "pick_place_plan", PLAN_SCHEMA,
    )
    for step in data["steps"]:
        if step["pick"] not in objects or step["place"] not in pads:
            raise ValueError(f"plan references unknown name: {step}")
    data["source"] = "llm"
    data["model"] = MODEL
    return data


def verify(image_path: str, instruction: str, steps: list[dict], client=None) -> dict[str, Any]:
    """Ask the VLM to inspect the final frame. Returns {"success": bool, "observation": str}."""
    client = client or _client()
    with open(image_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return _structured(
        client, VERIFIER_SYSTEM,
        [
            {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}", "detail": "high"},
            {"type": "input_text", "text": f"Instruction: {instruction}\nExecuted plan: {json.dumps(steps, ensure_ascii=False)}"},
        ],
        "scene_verification", VERIFY_SCHEMA,
    )


# --- offline fallback (keyword matching) ----------------------------------------------------------
COLOR_WORDS = {
    "red": ["red", "赤"], "green": ["green", "緑"], "blue": ["blue", "青"],
    "yellow": ["yellow", "黄"], "white": ["white", "白"],
}


def rule_based_plan(instruction: str, objects: dict, pads: dict) -> dict[str, Any]:
    """Very small keyword planner: '<color cube> を <color pad> に' pairs, in order of appearance."""
    def find(color: str) -> list[tuple[int, str]]:
        return [(m.start(), color) for w in COLOR_WORDS[color] for m in re.finditer(w, instruction, re.I)]

    obj_by_color = {v["color"]: k for k, v in objects.items()}
    pad_by_color = {v["color"]: k for k, v in pads.items()}
    obj_hits = sorted(h for c in obj_by_color for h in find(c))
    pad_hits = sorted(h for c in pad_by_color for h in find(c))
    steps = []
    for pos, oc in obj_hits:
        later = [p for p in pad_hits if p[0] > pos]
        if later:
            steps.append({"pick": obj_by_color[oc], "place": pad_by_color[later[0][1]]})
    return {"steps": steps, "reasoning": "offline keyword planner", "source": "rule"}


DEMO_OBJECTS = {
    "cube_red": {"color": "red", "pos": (0.42, 0.22)},
    "cube_green": {"color": "green", "pos": (0.46, 0.00)},
    "cube_blue": {"color": "blue", "pos": (0.42, -0.22)},
}
DEMO_PADS = {
    "pad_A": {"color": "yellow", "pos": (0.62, 0.17)},
    "pad_B": {"color": "white", "pos": (0.62, -0.17)},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", default="暖色のキューブは黄色のパッドへ、寒色のキューブは白いパッドへ移してください")
    ap.add_argument("--offline", action="store_true", help="use the keyword planner instead of the LLM")
    ap.add_argument("--verify", metavar="PNG", help="also run the VLM verifier on this image")
    args = ap.parse_args()

    fn = rule_based_plan if args.offline else plan
    result = fn(args.instruction, DEMO_OBJECTS, DEMO_PADS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.verify:
        print(json.dumps(verify(args.verify, args.instruction, result["steps"]), ensure_ascii=False, indent=2))
