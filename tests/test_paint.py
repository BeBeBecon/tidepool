import numpy as np
import pytest

from tidepool import paint


def test_band_steps_front_to_back():
    t = np.linspace(0, 1, 4000)
    s = paint.band(t, 4, np.random.default_rng(0))
    assert s.min() == 0 and s.max() == 3
    assert (s[:200] == 0).all() and (s[-200:] == 3).all()
    assert np.all(np.diff(np.convolve(s, np.ones(200) / 200, "valid")) > -0.05)


def test_parts_cover_limbs_only():
    links = ["base_link", "upper_shell_link", "LF_palm_link", "LF_finger_tip_link", "RM_hip_link", "RR_calf_link"]
    for kind in paint.PAINTS:
        p = paint.parts(kind, links)
        assert set(p) == set(links[2:])
        assert all(len(c) == 7 and c.startswith("#") for c in p.values())


def test_unknown_paint():
    with pytest.raises(paint.shell.ShellError):
        paint.make("plaid")
