"""Grasp robustness evaluation: randomised cube positions, N trials, no video.

    python eval_grasp.py --trials 10 --seed 0

For each trial the three cubes are dropped at random reachable positions (min 8 cm apart),
a random 2+1 pad assignment is executed, and per-grasp success / xy error are recorded.
Writes out/eval_grasp.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from pick_place import CUBES, PandaSim

HERE = pathlib.Path(__file__).resolve().parent
X_RANGE, Y_RANGE, MIN_SEP = (0.36, 0.54), (-0.28, 0.28), 0.08


def random_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    while True:
        pts = np.column_stack([rng.uniform(*X_RANGE, 3), rng.uniform(*Y_RANGE, 3)])
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(3)
        if d.min() >= MIN_SEP:
            return {cube: p for cube, p in zip(CUBES, pts)}


def place_cubes(sim: PandaSim, pos: dict[str, np.ndarray]) -> None:
    for cube, xy in pos.items():
        adr = sim.m.joint(f"{CUBES[cube]}_j").qposadr[0]
        sim.d.qpos[adr:adr + 7] = [xy[0], xy[1], 0.02, 1, 0, 0, 0]
        sim.d.qvel[sim.m.joint(f"{CUBES[cube]}_j").dofadr[0]:][:6] = 0
    mujoco.mj_forward(sim.m, sim.d)


def run_trial(trial: int, rng: np.random.Generator) -> dict:
    sim = PandaSim(width=320, height=180)  # small renderer: never used, keeps init cheap
    pos = random_positions(rng)
    place_cubes(sim, pos)
    sim.hold(0.3)  # let cubes settle
    cubes = list(CUBES)
    rng.shuffle(cubes)
    pads = ["pad_A", "pad_A", "pad_B"] if rng.random() < 0.5 else ["pad_B", "pad_B", "pad_A"]
    plan = list(zip(cubes, pads))
    t0 = time.time()
    results = sim.run_plan(plan)
    return {"trial": trial, "initial_xy": {c: [round(float(v), 3) for v in p] for c, p in pos.items()},
            "plan": plan, "results": results, "wall_seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/eval_grasp.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    trials = [run_trial(i + 1, rng) for i in range(args.trials)]
    grasps = [r for t in trials for r in t["results"]]
    errs = np.array([r["xy_error_m"] for r in grasps]) * 100
    ok = [r["success"] for r in grasps]
    summary = {
        "trials": args.trials, "seed": args.seed, "grasps": len(grasps), "successes": int(sum(ok)),
        "success_rate": round(sum(ok) / len(grasps), 3),
        "xy_error_cm": {"mean": round(float(errs.mean()), 2), "std": round(float(errs.std()), 2),
                        "max": round(float(errs.max()), 2)},
        "per_cube": {c: {"n": sum(r["cube"] == c for r in grasps),
                         "successes": sum(r["success"] for r in grasps if r["cube"] == c)} for c in CUBES},
        "x_range_m": X_RANGE, "y_range_m": Y_RANGE, "min_separation_m": MIN_SEP,
    }
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "trials": trials}, ensure_ascii=False, indent=2))

    print(f"\n{'trial':>5} {'successes':>9} {'max err cm':>10}  plan")
    for t in trials:
        e = max(r["xy_error_m"] for r in t["results"]) * 100
        print(f"{t['trial']:>5} {sum(r['success'] for r in t['results']):>7}/3 {e:>10.1f}  "
              + ", ".join(f"{c}->{p[-1]}" for c, p in t["plan"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
