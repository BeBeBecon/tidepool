import json

import numpy as np
import pytest

from tidepool import motion, paths

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def rig():
    try:
        return motion.Rig()
    except (paths.MissingRepo, FileNotFoundError):
        pytest.skip("needs jumper's example bundle")


def test_follower_lands_on_reference(rig):
    c = rig.c
    term = next(t for t in c["observation"]["terms"] if t["name"] == "ref_joint_pos")
    obs = np.random.default_rng(0).normal(size=(1, c["observation"]["dim"])).astype(np.float32)
    act = ort.InferenceSession(motion.follower(c)).run(None, {"obs": obs})[0][0]
    assert np.allclose(act * c["action_scale"], obs[0, term["offset"]:term["offset"] + term["dim"]])


@pytest.mark.parametrize("name", motion.PRESETS)
def test_presets_go_home_to_home_inside_limits(rig, name):
    q, lean = motion.timeline(motion.preset(name, rig), rig)
    assert np.allclose(q[0], rig.home) and np.allclose(q[-1], rig.home)
    assert (q >= rig.lo).all() and (q <= rig.hi).all()
    assert np.abs(lean[-1]).max() == 0


def test_lean_keeps_supports_planted(rig):
    q = rig.solve(rig.home, 0.2, -0.1)
    rig._set(q, 0.2, -0.1)
    for tip, body in rig.body.items():
        assert np.linalg.norm(rig.d.xpos[body] - rig.goal[tip]) < 2e-3


def test_reach_moves_a_lifted_palm(rig):
    target = [0.22, 0.12, 0.2]
    q = rig.solve(rig.home, lifted=("LF",), reach={"LF": target})
    assert np.linalg.norm(np.array(rig.tips(q)["LF"]) - target) < 5e-3
