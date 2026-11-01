"""Motions without training: a trajectory plus an ONNX that just follows it.

The policy is one MatMul. The controller already hands it `ref_joint_pos`
relative to the home pose, so `action = ref / action_scale` puts the target on
the recording. The result replaces `gesture_hello` in a copy of the shipped app,
which keeps its keys, its controllers and its reference.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

from . import paths

MODE = "gesture_hello"
HZ = 50.0


def wave(contract: dict, seconds: float = 5.0) -> np.ndarray:
    """Left claw up and waving, right claw swinging low. Starts and ends at home."""
    names = contract["wire_joint_order"]
    t = np.arange(int(seconds * HZ)) / HZ
    ramp = np.clip(np.minimum(t, seconds - t - 1 / HZ) / 0.8, 0, 1)
    env = ramp * ramp * (3 - 2 * ramp)
    swing = np.sin(2 * np.pi * 1.5 * t)
    q = np.tile([contract["default_joint_pos"][n] for n in names], (len(t), 1))
    q[:, names.index("LF_J1_joint")] += -1.4 * env
    q[:, names.index("LF_J3_joint")] += 0.4 * env * swing
    q[:, names.index("RF_J1_joint")] += 0.25 * env * swing
    limits = contract["joint_limits"]
    lo, hi = np.array([limits[n] for n in names]).T
    return np.clip(q, lo, hi)


def trajectory(contract: dict, q: np.ndarray) -> dict:
    return {"_source": "tidepool motion", "joint_order": contract["wire_joint_order"],
            "q": q.round(5).tolist(), "qd": (np.gradient(q, axis=0) * HZ).round(5).tolist(),
            "root_quat": [[1.0, 0.0, 0.0, 0.0]] * len(q)}


def follower(contract: dict) -> bytes:
    """[1, obs] -> [1, action]: read `ref_joint_pos`, undo the action scale."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    obs, act = contract["observation"]["dim"], contract["action"]["dim"]
    term = next(t for t in contract["observation"]["terms"] if t["name"] == "ref_joint_pos")
    columns = contract["obs_joint_order"]
    w = np.zeros((obs, act), np.float32)
    for row, name in enumerate(contract["action_joint_order"]):
        w[term["offset"] + columns.index(name), row] = 1 / contract["action_scale"]
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


def pack(out: Path, name: str = "wave", rknn: Path | None = None) -> Path:
    """Copy the shipped bundle to `out`/jumper with MODE replaced, and zip it."""
    src = paths.default_app().with_suffix("")
    root = out / "jumper"
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(src, root)

    bundle = json.loads((root / "bundle.json").read_text("utf-8"))
    entry = bundle["modes"][MODE]
    contract_path = root / entry["contract"]
    contract = json.loads(contract_path.read_text("utf-8"))

    q = wave(contract)
    file = f"{name}.motion.trajectory.json"
    (root / entry["trajectory"]).unlink()
    (root / file).write_text(json.dumps(trajectory(contract, q)), "utf-8")
    contract["reference"].update(file=file, rec_hz=HZ, n_frames=len(q), duration=len(q) / HZ)
    contract_path.write_text(json.dumps(contract), "utf-8")
    (root / entry["models"]["onnx"]).write_bytes(follower(contract))

    board = root / entry["models"]["rknn"]
    if rknn:
        shutil.copy(rknn, board)
    else:
        board.unlink()
        del entry["models"]["rknn"]
        bundle["notes"].append(f"{MODE}: no .rknn yet, the board runs a stub for it.")
    entry["trajectory"] = file
    bundle["built_from"][MODE].update(**{"from": "(no checkpoint: follows its recording)"})
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


if __name__ == "__main__":
    import sys

    print(pack(Path(sys.argv[1]), rknn=Path(sys.argv[2]) if len(sys.argv) > 2 else None))
