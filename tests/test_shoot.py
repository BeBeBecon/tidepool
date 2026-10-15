import pytest

from tidepool import shoot, spec


def test_shoot_block_defaults_and_errors():
    s = shoot.check({"camera": "low", "script": [{"at": 1, "walk": "forward"}]})
    assert s["camera"] == {"preset": "low"} and s["size"] == [1280, 720]
    assert shoot.check(None) == {}
    with pytest.raises(shoot.ShootError, match="preset"):
        shoot.check({"camera": {"preset": "drone"}})
    with pytest.raises(shoot.ShootError, match="unknown keys"):
        shoot.check({"script": [{"at": 0, "fly": True}]})
    with pytest.raises(shoot.ShootError, match="size"):
        shoot.check({"size": [0, 10]})


def test_spec_carries_shoot():
    s = spec.parse({"name": "s", "shoot": {"size": [1080, 1920], "script": []}})
    assert s.shoot["size"] == [1080, 1920]
    with pytest.raises(spec.SpecError, match="shoot"):
        spec.parse({"name": "s", "shoot": {"nope": 1}})
    assert spec.parse({"name": "s"}).shoot == {}
