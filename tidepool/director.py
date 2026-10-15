"""Run a `.map` headless with the stock app, driven by a script, and film it.

    python -m tidepool.director shoot.json

Imports jumper's own play entry point and swaps its loop for one that feeds the
controller a scripted gamepad, renders frames offscreen and logs contacts. It
runs from the jumper repo (`tidepool.shoot` does that), since play reads its
`.env` there.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STEP_HZ_FALLBACK = 200.0


@dataclass
class Pad:
    """A gamepad the controller believes in. Sticks are set by the script."""
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0
    buttons: set = field(default_factory=set)
    connected: bool = True

    def state(self):
        return self

    def close(self):
        pass


def key_table(bundle: Path) -> dict[str, tuple[str, str | None]]:
    """Mode name -> (key, modifier) from the app's controller.toml."""
    import tomllib

    fsm = tomllib.loads((bundle / "controller.toml").read_text("utf-8"))["fsm"]
    out = {}
    for b in fsm.get("button", []):
        if b.get("key") and b["name"] not in out and not b["key"].startswith("keypad"):
            out[b["name"]] = (b["key"], b.get("with"))
    return out


class Camera:
    """Free camera presets. `chase` and `side` move with the robot, the rest sit still."""

    def __init__(self, cfg: dict, target: list[float] | None):
        import mujoco

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.preset = cfg.get("preset", "chase")
        self.distance = float(cfg.get("distance", {"top": 2.6, "low": 1.6}.get(self.preset, 1.3)))
        self.elevation = float(cfg.get("elevation", {"top": -89, "low": -8, "chase": -18, "side": -12}.get(self.preset, -20)))
        self.azimuth = cfg.get("azimuth")
        self.height = float(cfg.get("height", 0.08))
        self.target = target
        self.lookat = None
        self.smooth = float(cfg.get("smooth", 0.08))

    def update(self, pos, yaw_deg: float):
        import mujoco

        # MuJoCo's azimuth is the direction the camera looks along, so a chase
        # camera shares the robot's heading and sits behind it.
        if self.preset in ("chase", "side"):
            want = np.array([pos[0], pos[1], self.height])
            az = yaw_deg + (0.0 if self.preset == "chase" else 90.0)
        else:
            t = self.target if self.target is not None else pos
            want = np.array([(pos[0] + t[0]) / 2, (pos[1] + t[1]) / 2, self.height])
            az = float(self.azimuth) if self.azimuth is not None else yaw_deg - 35.0
        if self.azimuth is not None and self.preset in ("chase", "side"):
            az = yaw_deg + float(self.azimuth)
        self.lookat = want if self.lookat is None else self.lookat + (want - self.lookat) * self.smooth
        self.cam.lookat[:] = self.lookat
        self.cam.distance = self.distance
        self.cam.elevation = self.elevation
        self.cam.azimuth = az
        return self.cam


def yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wear(skin: Path) -> None:
    """Make every jumper.xml load with the skin's upper shell.

    play never loads the robot packed in a .map, so the shell is swapped where
    the model is read instead.
    """
    import tempfile
    import xml.etree.ElementTree as ET
    import zipfile

    import mujoco

    root = Path(tempfile.mkdtemp(prefix="tidepool-skin-"))
    with zipfile.ZipFile(skin) as z:
        z.extractall(root)
    robot = root / "assembled" / "jumper"
    xml = ET.parse(robot / "robot.xml").getroot()
    meshdir = robot / (xml.find("compiler").get("meshdir", "") if xml.find("compiler") is not None else "")
    files = {m.get("name"): m.get("file") for m in xml.iter("mesh")}
    parts = [(str((meshdir / files[g.get("mesh")]).resolve()), [float(v) for v in g.get("rgba", "1 1 1 1").split()])
             for g in xml.iter("geom") if (g.get("name") or "").startswith("shellflow_upper_")]
    if not parts:
        raise SystemExit(f"[shoot] {skin}: no upper shell in this skin")

    load = mujoco.MjSpec.from_file

    def from_file(path, *a, **kw):
        spec = load(path, *a, **kw)
        if Path(path).name != "jumper.xml":
            return spec
        old = spec.geom("upper_shell_link_visual")
        body = old.parent
        cls = old.classname
        spec.delete(old)
        for i, (file, rgba) in enumerate(parts):
            spec.add_mesh(name=f"skin_upper_{i}", file=file)
            body.add_geom(name=f"skin_upper_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=f"skin_upper_{i}", rgba=rgba, density=0,
                          contype=0, conaffinity=0, group=2)
        return spec

    mujoco.MjSpec.from_file = staticmethod(from_file)


def run(cfg: dict) -> None:
    jumper = Path(cfg["jumper"])
    sys.path.insert(0, str(jumper / "scripts"))
    if cfg.get("skin"):
        wear(Path(cfg["skin"]))
    import play  # jumper's entry point

    sys.argv = ["play.py", "--app", cfg["app"], "--scene", cfg["map"], "--backend", "native",
                "--device", "cpu", "--headless"]

    def loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
        import cv2
        import imageio
        import mujoco
        import torch

        unwrapped = env.unwrapped
        sim = unwrapped.sim
        step_dt = float(unwrapped.step_dt)
        hz = 1.0 / step_dt

        # Play scatters a random base pose; pin it so every take is the same.
        start = cfg.get("start") or {}
        at = start.get("at", [0.0, 0.0])
        half = math.radians(float(start.get("yaw", 0.0))) / 2
        sim.data.qpos[:, 0:2] = torch.tensor([float(at[0]), float(at[1])])
        sim.data.qpos[:, 3:7] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
        sim.data.qvel[:, :6] = 0.0
        sim.forward()
        model, data = sim.env_mjdata(0)

        pad = Pad()
        policy._pad = pad
        keys = key_table(Path(cfg["bundle"]))
        from controller import vocabulary
        codes = {name: c["glfw"] for name, c in vocabulary.key_codes().items()}
        codes["ctrl"] = codes.get("key_left_ctrl", 341)

        W, H = cfg.get("size", [1280, 720])
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
        model.vis.global_.offheight = max(model.vis.global_.offheight, H)
        renderer = mujoco.Renderer(model, H, W)
        if cfg.get("shadows", True):
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        watch = [i for i, n in enumerate(names) if any(w in n for w in cfg.get("watch", []))]
        target = [float(data.xpos[watch[0]][0]), float(data.xpos[watch[0]][1])] if watch else None
        cam = Camera(cfg.get("camera") or {}, target)
        gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"

        fps = float(cfg.get("fps", 50))
        every = max(1, round(hz / fps))
        out_fps = hz / every * float(cfg.get("speed", 1.0))
        out = Path(cfg["out"])
        writer = imageio.get_writer(str(out), fps=out_fps, codec="libx264", quality=8,
                                    pixelformat="yuv420p", macro_block_size=1)
        events = sorted(cfg.get("script", []), key=lambda e: float(e.get("at", 0)))
        duration = float(cfg.get("duration", (float(events[-1]["at"]) if events else 0) + 3.0))
        total = int(round(duration * hz))
        held: list[tuple[int, int]] = []     # (release step, keycode)
        log = []
        f6 = np.zeros(6)
        obs = env.get_observations()
        print(f"[shoot] {duration:.1f}s at {hz:g} Hz, {W}x{H}, {out_fps:g} fps -> {out}")

        def press(name: str, mod: str | None = None, *, step: int):
            code = codes[name]
            if mod:
                policy._viewer_keys.report(codes[mod], True)
                held.append((step + 3, codes[mod]))
            policy._viewer_keys.report(code, True)
            held.append((step + 2, code))

        def act(e: dict, step: int):
            if "walk" in e:
                d = e["walk"]
                mag = float(e.get("speed", 1.0))
                pad.ly = {"forward": -mag, "back": mag}.get(d, 0.0)
                pad.lx = {"left": -mag, "right": mag}.get(d, 0.0)
            if "turn" in e:
                mag = float(e.get("speed", 1.0))
                pad.rx = {"left": -mag, "right": mag}.get(e["turn"], 0.0)
            if "stop" in e:
                pad.lx = pad.ly = pad.rx = 0.0
            if e.get("jump"):
                press(*keys["jump"], step=step) if "jump" in keys else press("key_space", step=step)
            for kind in ("dance", "gesture"):
                if kind in e:
                    mode = f"{kind}_{e[kind]}"
                    if mode not in keys:
                        raise SystemExit(f"[shoot] no {mode} in this app; have {sorted(keys)}")
                    press(*keys[mode], step=step)
            if "claw" in e:
                press(*keys[f"claw_{e['claw']}"], step=step)

        for step in range(total):
            t = step * step_dt
            while events and float(events[0].get("at", 0)) <= t:
                act(events.pop(0), step)
            for s, code in [h for h in held if h[0] <= step]:
                policy._viewer_keys.report(code, False)
                held.remove((s, code))
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            model, data = sim.env_mjdata(0)
            hits = []
            for k in range(data.ncon):
                c = data.contact[k]
                b1, b2 = names[model.geom_bodyid[c.geom1]], names[model.geom_bodyid[c.geom2]]
                if b1.startswith("robot") != b2.startswith("robot") or any(names[w] in (b1, b2) for w in watch):
                    mujoco.mj_contactForce(model, data, k, f6)
                    hits.append([gname(c.geom1), gname(c.geom2), round(float(f6[0]), 2)])
            log.append({"t": round(t, 4), "root": [round(float(v), 4) for v in data.qpos[:7]],
                        "watch": [[round(float(v), 4) for v in data.xpos[w]] for w in watch], "hits": hits})
            if step % every == 0:
                renderer.update_scene(data, cam.update(data.qpos[:3], yaw_of(data.qpos[3:7])))
                writer.append_data(renderer.render())
            if step % int(hz) == 0:
                print(f"\r[shoot] {t:5.1f}s / {duration:.1f}s", end="", flush=True)
        writer.close()
        print(f"\n[shoot] wrote {out}")
        out.with_suffix(".contacts.json").write_text(json.dumps(log))

    play._loop = loop
    play.main()


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
