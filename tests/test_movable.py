import numpy as np
import pytest
from PIL import Image

from tidepool import room, spec


def test_hull_is_convex_and_thinned():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 2))
    h = room.hull(pts, limit=16)
    assert 3 <= len(h) <= 16
    # every input point is inside or on the hull (counter-clockwise ring)
    for a, b in zip(h, np.roll(h, -1, axis=0)):
        cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        assert (cross >= -1e-9).all()


def test_extrude_stands_on_the_floor_with_photo_uvs():
    mask = np.zeros((60, 80), bool)
    mask[20:35, 50:60] = True
    hm = np.zeros((60, 80), np.float32)
    hm[mask] = 0.09
    f = room._Frame(12, 16, 3.2)
    text, pos, height = room.extrude(mask, hm, room.Movable(mask=None, name="dog"), f)
    assert height == pytest.approx(0.144)     # 0.4 x the 0.36 m narrow side beats the shallow depth
    hm[mask] = 0.25
    assert room.extrude(mask, hm, room.Movable(mask=None, name="dog"), f)[2] == pytest.approx(0.25)
    assert room.extrude(mask, hm, room.Movable(mask=None, name="dog", height=0.09), f)[2] == 0.09
    hm[mask] = 0.09
    lines = text.splitlines()
    v = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.startswith("v ")])
    vt = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.startswith("vt ")])
    assert len(v) == len(vt) and len(v) % 2 == 0
    assert v[:, 2].min() == 0.0 and v[:, 2].max() == pytest.approx(height)
    assert abs(v[:, 0].mean()) < 1e-3 and abs(v[:, 1].mean()) < 1e-3     # centred
    assert 0.6 < vt[:, 0].mean() < 0.72 and 0.5 < vt[:, 1].mean() < 0.7  # right of centre, upper half
    assert pos[0] > 0 and pos[1] > 0
    assert sum(l.startswith("f ") for l in lines) >= 2 * (len(v) // 2) + 2 * (len(v) // 2 - 2)


def test_read_mask_alpha_and_grey(tmp_path):
    rgba = Image.new("RGBA", (8, 6), (255, 0, 0, 0))
    rgba.putpixel((2, 2), (255, 0, 0, 255))
    rgba.save(tmp_path / "a.png")
    grey = Image.new("L", (8, 6), 0)
    grey.putpixel((5, 1), 255)
    grey.save(tmp_path / "g.png")
    a = room.read_mask(tmp_path / "a.png", (6, 8))
    g = room.read_mask(tmp_path / "g.png", (6, 8))
    assert a.sum() == 1 and a[2, 2]
    assert g.sum() == 1 and g[1, 5]


def test_spec_movable_block(tmp_path):
    Image.new("RGB", (8, 6)).save(tmp_path / "desk.png")
    Image.new("L", (8, 6)).save(tmp_path / "shiba dog.png")
    s = spec.parse({"name": "d", "photo": {"file": "desk.png", "movable": "shiba dog.png"}}, base=tmp_path)
    assert [m.name for m in s.photo.movable] == ["shiba_dog"]
    assert s.photo.movable[0].mass == 0.3 and s.photo.movable[0].height is None
    s = spec.parse({"name": "d", "photo": {"file": "desk.png",
                    "movable": [{"mask": "shiba dog.png", "name": "pup", "height": 0.2, "mass": 1}]}}, base=tmp_path)
    assert s.photo.movable[0].name == "pup" and s.photo.movable[0].height == 0.2
    with pytest.raises(spec.SpecError, match="not found"):
        spec.parse({"name": "d", "photo": {"file": "desk.png", "movable": "nope.png"}}, base=tmp_path)
    with pytest.raises(spec.SpecError, match="unknown keys"):
        spec.parse({"name": "d", "photo": {"file": "desk.png", "movable": [{"mask": "shiba dog.png", "wait": 1}]}},
                   base=tmp_path)
