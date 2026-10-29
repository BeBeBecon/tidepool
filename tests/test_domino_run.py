import math

import pytest

from tidepool import scene, spec, toys


def test_line_spacing_and_heading():
    objs = toys.expand("domino_run", 0, {"path": "line", "count": 5, "start": [0, 0], "height": 0.1})
    assert len(objs) == 5
    xs = [o["pos"][0] for o in objs]
    assert xs == pytest.approx([0, 0.06, 0.12, 0.18, 0.24])
    assert all(o["euler"][2] == 0 for o in objs)
    assert all(o["kind"] == "dynamic" for o in objs)
    assert len({o["name"] for o in objs}) == 5


def test_spiral_keeps_spacing():
    objs = toys.expand("domino_run", 0, {"path": "spiral", "count": 30, "height": 0.1})
    gaps = [math.dist(a["pos"][:2], b["pos"][:2]) for a, b in zip(objs, objs[1:])]
    assert min(gaps) > 0.055 and max(gaps) < 0.065


def test_unknown_parameter_is_reported():
    with pytest.raises(ValueError, match="cuont"):
        toys.expand("domino_run", 0, {"cuont": 3})


def test_spec_to_scene_json():
    s = spec.parse({
        "name": "t",
        "spawn": {"at": [-0.5, 0]},
        "toys": [{"domino_run": {"count": 3}}],
    })
    doc = scene.build(s)
    assert doc["spawn"]["position"] == [-0.5, 0.0, 0.0]
    assert len(doc["objects"]) == 3
    assert doc["ground"]["material"] == "checker_light"


def test_bad_look():
    s = spec.parse({"name": "t", "look": "mars", "toys": []})
    with pytest.raises(spec.SpecError, match="look"):
        scene.build(s)
