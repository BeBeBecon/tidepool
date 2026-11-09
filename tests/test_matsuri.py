import pytest

from tidepool import scene, spec, toys


@pytest.mark.parametrize("kind", ["chochin", "maku"])
def test_props_stay_out_of_the_way(kind):
    objs = toys.expand(kind, 0, {})
    assert objs and all(o["kind"] == "static" and o["collision"] is False for o in objs)
    assert len({o["name"] for o in objs}) == len(objs)


def test_every_prop_material_is_defined():
    s = spec.Spec(name="m", look="matsuri", toys=[spec.ToyCall("chochin", {}), spec.ToyCall("maku", {})])
    doc = scene.build(s)
    defined = {m["name"] for m in doc["materials"]}
    used = {o["material"] for o in doc["objects"]} | {doc["ground"]["material"]}
    builtin = {"wood_dark"}
    assert used <= defined | builtin
