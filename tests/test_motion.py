import json

import numpy as np
import pytest

from tidepool import motion, paths

ort = pytest.importorskip("onnxruntime")


@pytest.fixture
def contract():
    try:
        return json.loads((paths.default_app().with_suffix("") / f"{motion.MODE}.json").read_text("utf-8"))
    except (paths.MissingRepo, FileNotFoundError):
        pytest.skip("needs jumper's example bundle")


def test_follower_lands_on_reference(contract):
    term = next(t for t in contract["observation"]["terms"] if t["name"] == "ref_joint_pos")
    obs = np.random.default_rng(0).normal(size=(1, contract["observation"]["dim"])).astype(np.float32)
    act = ort.InferenceSession(motion.follower(contract)).run(None, {"obs": obs})[0][0]
    ref = obs[0, term["offset"]:term["offset"] + term["dim"]]
    assert np.allclose(act * contract["action_scale"], ref)


def test_wave_is_home_to_home_inside_limits(contract):
    q = motion.wave(contract)
    home = [contract["default_joint_pos"][n] for n in contract["wire_joint_order"]]
    limits = np.array([contract["joint_limits"][n] for n in contract["wire_joint_order"]]).T
    assert np.allclose(q[0], home) and np.allclose(q[-1], home)
    assert (q >= limits[0]).all() and (q <= limits[1]).all()
