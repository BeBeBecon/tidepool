"""Motions without training: a recording plus an ONNX that just follows it.

The policy is one MatMul. The controller already hands it `ref_joint_pos`
relative to the home pose, so `action = ref / action_scale` puts the target on
the recording. Packed motions replace the shipped gestures and dances in a copy
of the app, which keeps its keys, its controllers and its reference.json.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

from . import paths

HZ = 50.0
SLOTS = ["gesture_hello", "gesture_bow", "gesture_paw", "gesture_salute",
         "dance_crab", "dance_brazilian", "dance_maze", "dance_dream_wings"]
KEYS = ["1", "2", "3", "4", "Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4"]
TIPS = [("LF_palm_pad_b", 0, 4), ("RF_palm_pad_b", 5, 4), ("LM_foot_tip", 10, 3),
        ("RM_foot_tip", 13, 3), ("LR_foot_tip", 16, 3), ("RR_foot_tip", 19, 3)]
BRACE = 0.7     # s to shift the weight back before both palms lift
MARGIN = 0.03   # m, how far behind the front legs the centre of mass has to sit
FAST = 8.0      # rad/s, a guess: nothing here has been timed on a servo


def source() -> Path:
    return paths.default_app().with_suffix("")


def contract() -> dict:
    return json.loads((source() / f"{SLOTS[0]}.json").read_text("utf-8"))


class Rig:
    """The crab as forward and inverse kinematics. `q` is always in wire order."""

    def __init__(self, c: dict | None = None):
        import mujoco

        self.mj = mujoco
        self.c = c or contract()
        self.names = self.c["wire_joint_order"]
        self.home = np.array([self.c["default_joint_pos"][n] for n in self.names])
        self.lo, self.hi = np.array([self.c["joint_limits"][n] for n in self.names]).T
        self.z = float(self.c["base_height"])
        self.m = mujoco.MjModel.from_xml_path(str(paths.profile().with_name("baseline.xml")))
        self.d = mujoco.MjData(self.m)
        self.body = {t: self.m.body(f"{t}_link").id for t, _, _ in TIPS}
        self._set(self.home)
        self.goal = {t: self.d.xpos[b].copy() for t, b in self.body.items()}

    def _set(self, q, pitch: float = 0.0, roll: float = 0.0, back: float = 0.0) -> None:
        cy, sy, cx, sx = np.cos(pitch / 2), np.sin(pitch / 2), np.cos(roll / 2), np.sin(roll / 2)
        self.d.qpos[:7] = -back, 0, self.z, cy * cx, cy * sx, sy * cx, -sy * sx
        self.d.qpos[7:] = q
        self.mj.mj_kinematics(self.m, self.d)

    def com(self, q, pitch: float = 0.0, roll: float = 0.0, back: float = 0.0) -> float:
        self._set(q, pitch, roll, back)
        self.mj.mj_comPos(self.m, self.d)
        return float(self.d.subtree_com[0, 0])

    def support(self) -> float:
        """Front edge of the four legs, with a margin."""
        return float(max(self.goal[t][0] for t in self.goal if t.endswith("foot_tip"))) - MARGIN

    def balance(self, q, pitch=0.0, roll=0.0, lifted=(), reach=None) -> tuple[np.ndarray, float]:
        """`solve`, and with both palms up slide the body back until it stands on its legs."""
        q = self.solve(q, pitch, roll, lifted, reach)
        back = 0.0
        if {"LF", "RF"} <= set(lifted):
            for _ in range(5):
                back = max(0.0, back + self.com(q, pitch, roll, back) - self.support())
                q = self.solve(q, pitch, roll, lifted, reach, back)
        return q, back

    def solve(self, q, pitch: float = 0.0, roll: float = 0.0, lifted=(), reach: dict | None = None,
              back: float = 0.0) -> np.ndarray:
        """Move `q` so each limb hits its goal. Limbs in `lifted` ride along, `reach` gives them a target."""
        q = np.array(q, float)
        goal = {t: g for t, g in self.goal.items() if t[:2] not in lifted}
        goal.update({f"{k}_palm_pad_b": np.asarray(v, float) for k, v in (reach or {}).items()})
        jac = np.zeros((3, self.m.nv))
        for _ in range(24):
            self._set(q, pitch, roll, back)
            self.mj.mj_comPos(self.m, self.d)
            for tip, first, n in TIPS:
                if tip in goal:
                    self.mj.mj_jacBody(self.m, self.d, jac, None, self.body[tip])
                    J = jac[:, 6 + first:6 + first + n]
                    err = goal[tip] - self.d.xpos[self.body[tip]]
                    q[first:first + n] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            q = np.clip(q, self.lo, self.hi)
        return q

    def bodies(self, q, pitch: float = 0.0, roll: float = 0.0, back: float = 0.0) -> list[list[float]]:
        self._set(q, pitch, roll, back)
        return np.hstack([self.d.xpos, self.d.xquat]).round(4).tolist()

    def tips(self, q, pitch: float = 0.0, roll: float = 0.0, back: float = 0.0) -> dict[str, list[float]]:
        self._set(q, pitch, roll, back)
        return {k: self.d.xpos[self.body[f"{k}_palm_pad_b"]].round(4).tolist() for k in ("LF", "RF")}


def timeline(keys: list[dict], rig: Rig) -> tuple[np.ndarray, np.ndarray]:
    """Eased 50 Hz joints and lean from keyframes, from home back to home."""
    rest = {"t": 0.0, "q": rig.home, "pitch": 0.0, "roll": 0.0, "back": 0.0, "lifted": []}
    both = lambda k: {"LF", "RF"} <= set(k.get("lifted", ()))
    keys = sorted(keys, key=lambda k: k["t"])
    plan = [rest, *keys, rest]
    stops, now = [rest], 0.0
    # Both palms leave the floor only once the weight is over the legs, and land first.
    for prev, k in zip(plan, plan[1:]):
        now += k["t"] - prev["t"] if prev is not rest and k is not rest else 0.6
        if both(prev) != both(k):
            back = (k if both(k) else prev)["back"]
            brace = {**rest, "q": rig.solve(rig.home, back=back), "back": back}
            stops.append({**brace, "t": now})
            now += BRACE
        stops.append({**k, "t": now})
    t = np.arange(int(np.ceil(stops[-1]["t"] * HZ)) + 1) / HZ
    at = np.array([s["t"] for s in stops])
    i = np.clip(np.searchsorted(at, t, side="right") - 1, 0, len(stops) - 2)
    gap = np.maximum(at[i + 1] - at[i], 1e-6)
    u = np.clip((t - at[i]) / gap, 0, 1)
    u = (u * u * (3 - 2 * u))[:, None]
    q = np.array([s["q"] for s in stops], float)
    lean = np.array([[s["pitch"], s["roll"], s.get("back", 0.0)] for s in stops], float)
    return np.clip(q[i] * (1 - u) + q[i + 1] * u, rig.lo, rig.hi), lean[i] * (1 - u) + lean[i + 1] * u


def speed(q: np.ndarray) -> float:
    return float(np.abs(np.gradient(q, axis=0)).max() * HZ)


def _arm(rig: Rig, j1=0.0, j3=0.0, j0=0.0, claw=0.0, side="LR", q=None) -> np.ndarray:
    """Offsets from home, mirrored: J1 turns the same way on both arms, the rest opposite."""
    q = rig.home.copy() if q is None else q.copy()
    for s, first, sign in (("L", 0, 1), ("R", 5, -1)):
        if s in side:
            q[first] += sign * j0
            q[first + 1] += j1
            q[first + 3] += sign * j3
            q[first + 4] = sign * -claw
    return q


def _key(rig: Rig, t, q, pitch=0.0, roll=0.0, lifted=()) -> dict:
    q, back = rig.balance(q, pitch, roll, lifted)
    return {"t": round(t, 3), "q": q.round(5).tolist(), "pitch": pitch, "roll": roll,
            "back": round(back, 4), "lifted": list(lifted)}


def preset(name: str, rig: Rig) -> list[dict]:
    both, left = ("LF", "RF"), ("LF",)
    key = functools.partial(_key, rig)
    if name == "wave":
        return [key(.0, _arm(rig, -1.4, side="L"), lifted=left)] + [
            key(.35 * (i + 1), _arm(rig, -1.4, .45 * (-1) ** i, side="L"), lifted=left) for i in range(6)]
    if name == "clap":
        up, shut = _arm(rig, -1.1, .3), _arm(rig, -1.1, .3, .75)
        return [key(.0, up, lifted=both)] + [
            key(.8 + .3 * i, shut if i % 2 == 0 else up, lifted=both) for i in range(6)]
    if name == "snip":
        return [key(.0, _arm(rig, -.9), lifted=both)] + [
            key(.8 + .22 * i, _arm(rig, -.9, claw=1.1 if i % 2 == 0 else 0), lifted=both) for i in range(8)]
    if name == "bow":
        p = np.radians(14)
        return [key(.0, rig.home, p), key(.9, rig.home, p)]
    if name == "sway":
        r = np.radians(10)
        return [key(.5 * (i + 1), rig.home, 0, s * r) for i, s in enumerate((1, -1, 1, -1))]
    if name == "flex":
        flex = _arm(rig, -1.7, .55, -.3)
        p = np.radians(-6)
        return [key(.0, flex, p, lifted=both), key(1.2, flex, p, lifted=both)]
    raise KeyError(name)


PRESETS = ["wave", "clap", "snip", "bow", "sway", "flex"]


def trajectory(names: list[str], q: np.ndarray) -> dict:
    return {"_source": "tidepool motion", "joint_order": names, "q": q.round(5).tolist(),
            "qd": (np.gradient(q, axis=0) * HZ).round(5).tolist(), "root_quat": [[1.0, 0.0, 0.0, 0.0]] * len(q)}


def follower(c: dict) -> bytes:
    """[1, obs] -> [1, action]: read `ref_joint_pos`, undo the action scale."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    obs, act = c["observation"]["dim"], c["action"]["dim"]
    term = next(t for t in c["observation"]["terms"] if t["name"] == "ref_joint_pos")
    columns = c["obs_joint_order"]
    w = np.zeros((obs, act), np.float32)
    for row, name in enumerate(c["action_joint_order"]):
        w[term["offset"] + columns.index(name), row] = 1 / c["action_scale"]
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "w"], ["actions"])], "follower",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, act])],
        [numpy_helper.from_array(w, "w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _deploy():
    spec = importlib.util.spec_from_file_location("deploy", paths.jumper() / "scripts" / "deploy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _board(onnx_bytes: bytes) -> Path | None:
    """A converted follower, if one was made from exactly this ONNX."""
    rknn = paths.out_dir() / "studio-board" / "actor.rknn"
    try:
        made = json.loads(rknn.with_suffix(".rknn.json").read_text("utf-8"))["onnx_sha256"]
    except (OSError, KeyError, ValueError):
        return None
    return rknn if made == hashlib.sha256(onnx_bytes).hexdigest() else None


def pack(out: Path, takes: dict[str, np.ndarray]) -> Path:
    """Copy the shipped bundle to `out`/jumper with each mode in `takes` replaced, and zip it."""
    root = out / "jumper"
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(source(), root)
    bundle = json.loads((root / "bundle.json").read_text("utf-8"))

    for mode, q in takes.items():
        entry = bundle["modes"][mode]
        path = root / entry["contract"]
        c = json.loads(path.read_text("utf-8"))
        (root / entry["trajectory"]).unlink()
        file = f"{mode}.trajectory.json"
        (root / file).write_text(json.dumps(trajectory(c["wire_joint_order"], q)), "utf-8")
        c["reference"].update(file=file, rec_hz=HZ, n_frames=len(q), duration=len(q) / HZ)
        path.write_text(json.dumps(c), "utf-8")
        model = follower(c)
        (root / entry["models"]["onnx"]).write_bytes(model)
        board, converted = root / entry["models"]["rknn"], _board(model)
        if converted:
            shutil.copy(converted, board)
        else:
            board.unlink()
            del entry["models"]["rknn"]
            bundle["notes"].append(f"{mode}: no .rknn, the board runs a stub for it.")
        entry["trajectory"] = file
        bundle["built_from"][mode]["from"] = "(no checkpoint: follows its recording)"

    bundle["files"] = {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
                                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "bundle.json"}
    (root / "bundle.json").write_text(json.dumps(bundle, indent=2), "utf-8")

    deploy = _deploy()
    deploy.validate_app(root)
    app = deploy.zip_bundle(root)
    if deploy.check_reference(root):
        raise SystemExit("reference check failed")
    return app
