"""Franka Panda pick-and-place in MuJoCo (headless, no viewer required).

Scene: scene.xml (stock Menagerie Panda + table + 3 cubes + 2 target pads).
Control: damped-least-squares IK on the hand TCP -> joint position actuators.

CLI example:
    python pick_place.py --plan "red:pad_A,green:pad_B,blue:pad_A" --video out/test.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import imageio
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCENE = HERE / "scene.xml"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # hand frame -> fingertip centre (m)
GRIP_OPEN, GRIP_CLOSE = 255.0, 0.0  # actuator8 ctrl values
CUBE_HALF, PAD_HALF_Z = 0.02, 0.005
PAD_SLOTS = [np.array([0.0, 0.03, 0.0]), np.array([0.0, -0.03, 0.0])]  # 2 cubes per pad
CUBES = {"red": "cube_red", "green": "cube_green", "blue": "cube_blue"}
PADS = {"pad_A": "pad_A", "pad_B": "pad_B"}


class PandaSim:
    """Thin wrapper: physics stepping, IK, motion primitives, frame recording."""

    def __init__(self, scene: pathlib.Path = SCENE, width: int = 1280, height: int = 720,
                 fps: int = 30, camera: str = "overview", speed: float = 1.0):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self._dk = mujoco.MjData(self.m)  # scratch data for IK (kinematics only)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        mujoco.mj_forward(self.m, self.d)

        self.hand = self.m.body("hand").id
        self.arm_q = np.array([self.m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_v = np.array([self.m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_u = np.array([self.m.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.grip_u = self.m.actuator("actuator8").id
        self.q_lo = self.m.jnt_range[self.arm_v, 0]
        self.q_hi = self.m.jnt_range[self.arm_v, 1]
        self.R_down = self.d.xmat[self.hand].reshape(3, 3).copy()  # home orientation = gripper pointing down
        self.q_home = self.d.qpos[self.arm_q].copy()

        self.renderer = mujoco.Renderer(self.m, height, width)
        self.camera, self.fps, self.speed = camera, fps, speed
        self.writer = None
        self._next_frame_t = 0.0
        self.overlay = None  # optional callable(img: np.ndarray, sim: PandaSim) -> np.ndarray
        self.status = ""      # free text shown by overlay (set by primitives)
        self.events: list[tuple[float, str]] = []
        self.placed: dict[str, int] = {p: 0 for p in PADS}

    # ---------------------------------------------------------------- helpers
    @property
    def t(self) -> float:
        return self.d.time

    def body_pos(self, name: str) -> np.ndarray:
        return self.d.xpos[self.m.body(name).id].copy()

    def tcp(self, d: mujoco.MjData | None = None) -> np.ndarray:
        d = self.d if d is None else d
        R = d.xmat[self.hand].reshape(3, 3)
        return d.xpos[self.hand] + R @ TCP_OFFSET

    def log(self, text: str) -> None:
        self.status = text
        self.events.append((round(self.t, 2), text))
        print(f"[{self.t:6.2f}s] {text}")

    # ----------------------------------------------------------------- render
    def render(self, camera: str | None = None) -> np.ndarray:
        self.renderer.update_scene(self.d, camera or self.camera)
        return self.renderer.render()

    def save_png(self, path: str | pathlib.Path, camera: str | None = None) -> np.ndarray:
        img = self.render(camera)
        imageio.imwrite(str(path), img)
        return img

    def start_video(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264",
                                         quality=8, macro_block_size=1)
        self._next_frame_t = self.t

    def attach_writer(self, writer) -> None:
        """Record into an externally managed imageio writer (shared across scenarios)."""
        self.writer = writer
        self._next_frame_t = self.t

    def stop_video(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _record(self) -> None:
        if self.writer is None or self.t + 1e-9 < self._next_frame_t:
            return
        img = self.render()
        if self.overlay is not None:
            img = self.overlay(img, self)
        self.writer.append_data(img)
        self._next_frame_t += 1.0 / self.fps

    # ---------------------------------------------------------------- physics
    def step(self) -> None:
        mujoco.mj_step(self.m, self.d)
        self._record()

    def hold(self, duration: float) -> None:
        for _ in range(int(duration * self.speed / self.m.opt.timestep)):
            self.step()

    # --------------------------------------------------------------------- IK
    def ik(self, target_pos: np.ndarray, R_target: np.ndarray | None = None,
           q_init: np.ndarray | None = None, iters: int = 300, damping: float = 0.05,
           gain: float = 0.5) -> tuple[np.ndarray, float]:
        """Damped least squares IK for the TCP (position + orientation). Returns (q, pos_err)."""
        R_target = self.R_down if R_target is None else R_target
        dk = self._dk
        dk.qpos[:] = self.d.qpos
        if q_init is not None:
            dk.qpos[self.arm_q] = q_init
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        quat_err, rotvec = np.zeros(4), np.zeros(3)
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, dk)
            mujoco.mj_comPos(self.m, dk)
            R = dk.xmat[self.hand].reshape(3, 3)
            tcp = dk.xpos[self.hand] + R @ TCP_OFFSET
            e_pos = target_pos - tcp
            mujoco.mju_mat2Quat(quat_err, (R_target @ R.T).ravel())
            mujoco.mju_quat2Vel(rotvec, quat_err, 1.0)
            err = np.concatenate([e_pos, rotvec])
            if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(rotvec) < 1e-3:
                break
            mujoco.mj_jac(self.m, dk, jacp, jacr, tcp, self.hand)
            J = np.vstack([jacp, jacr])[:, self.arm_v]
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            dk.qpos[self.arm_q] = np.clip(dk.qpos[self.arm_q] + gain * dq, self.q_lo, self.q_hi)
        return dk.qpos[self.arm_q].copy(), float(np.linalg.norm(e_pos))

    # ------------------------------------------------------------- primitives
    def move_joints(self, q_target: np.ndarray, duration: float) -> None:
        """Smoothly interpolate the arm position setpoints to q_target."""
        q0 = self.d.ctrl[self.arm_u].copy()
        n = max(1, int(duration * self.speed / self.m.opt.timestep))
        for i in range(n):
            a = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
            self.d.ctrl[self.arm_u] = q0 + a * (q_target - q0)
            self.step()

    def move_tcp(self, pos, duration: float = 1.2) -> float:
        q, err = self.ik(np.asarray(pos, float), q_init=self.d.ctrl[self.arm_u].copy())
        if err > 0.01:
            print(f"  ! IK residual {err*1000:.1f} mm at {pos}")
        self.move_joints(q, duration)
        return err

    def gripper(self, open_: bool, duration: float = 0.6) -> None:
        self.d.ctrl[self.grip_u] = GRIP_OPEN if open_ else GRIP_CLOSE
        self.hold(duration)

    def go_home(self, duration: float = 1.5) -> None:
        self.move_joints(self.q_home, duration)

    def park(self, duration: float = 1.5) -> None:
        """Move the arm up and back so cameras see the table clearly."""
        self.move_tcp([0.3, 0.0, 0.45], duration)

    # ------------------------------------------------------------------- task
    def place_target(self, pad: str) -> np.ndarray:
        slot = PAD_SLOTS[min(self.placed[pad], len(PAD_SLOTS) - 1)]
        return self.body_pos(PADS[pad]) + slot + np.array([0, 0, PAD_HALF_Z + CUBE_HALF])

    def pick_and_place(self, cube: str, pad: str, hover: float = 0.12) -> dict:
        """Pick `cube` ('red'|'green'|'blue') and place it on `pad` ('pad_A'|'pad_B')."""
        t0 = self.t
        obj = self.body_pos(CUBES[cube])
        target = self.place_target(pad)
        up = np.array([0, 0, hover])

        self.log(f"pick {cube}: approach")
        self.gripper(True, 0.1)
        self.move_tcp(obj + up, 1.4)
        self.log(f"pick {cube}: descend")
        self.move_tcp(obj + np.array([0, 0, 0.002]), 1.0)
        self.hold(0.2)
        self.log(f"pick {cube}: grasp")
        self.gripper(False, 0.6)
        self.log(f"pick {cube}: lift")
        self.move_tcp(obj + up, 1.0)
        self.log(f"place {cube} -> {pad}: carry")
        self.move_tcp(target + up, 1.6)
        self.log(f"place {cube} -> {pad}: descend")
        self.move_tcp(target + np.array([0, 0, 0.004]), 1.0)
        self.hold(0.2)
        self.log(f"place {cube} -> {pad}: release")
        self.gripper(True, 0.5)
        self.log(f"place {cube} -> {pad}: retreat")
        self.move_tcp(target + up, 0.9)
        self.hold(0.3)

        final = self.body_pos(CUBES[cube])
        xy_err = float(np.linalg.norm(final[:2] - target[:2]))
        on_pad = xy_err < 0.03 and abs(final[2] - target[2]) < 0.015
        self.placed[pad] += 1
        res = {"cube": cube, "pad": pad, "xy_error_m": round(xy_err, 4), "success": bool(on_pad),
               "sim_seconds": round(self.t - t0, 2)}
        self.log(f"{cube} -> {pad}: {'OK' if on_pad else 'FAILED'} (xy err {xy_err*100:.1f} cm)")
        return res

    def run_plan(self, plan: list[tuple[str, str]]) -> list[dict]:
        results = [self.pick_and_place(c, p) for c, p in plan]
        self.log("plan finished: park")
        self.park()
        return results


def parse_plan(text: str) -> list[tuple[str, str]]:
    plan = []
    for item in text.split(","):
        cube, pad = item.strip().split(":")
        plan.append((cube.strip(), pad.strip()))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="red:pad_A,green:pad_B,blue:pad_A")
    ap.add_argument("--video", default="out/test.mp4")
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 slows motions (more sim seconds)")
    ap.add_argument("--snapshot", default="out/final.png", help="PNG of the final scene (top camera)")
    args = ap.parse_args()

    sim = PandaSim(camera=args.camera, speed=args.speed)
    sim.save_png(str(pathlib.Path(args.snapshot).with_name("initial.png")), "top")
    wall = time.time()
    sim.start_video(args.video)
    sim.hold(0.5)
    results = sim.run_plan(parse_plan(args.plan))
    sim.hold(0.5)
    sim.stop_video()
    sim.save_png(args.snapshot, "top")
    print(json.dumps({"results": results, "sim_seconds": round(sim.t, 1),
                      "wall_seconds": round(time.time() - wall, 1)}, indent=2))


if __name__ == "__main__":
    main()
