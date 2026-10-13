import numpy as np
import pytest

from tidepool import room, spec


def bumpy(h=60, w=80):
    y, x = np.mgrid[0:h, 0:w]
    d = np.sin(x / 9.0) * np.cos(y / 7.0)
    d[10:20, 10:20] = d.min()  # a flat low patch
    rgb = np.stack([x / w, y / h, np.full_like(d, 0.5)], -1)
    return d, rgb


def test_detrend_removes_the_slope_and_keeps_the_bump():
    y, x = np.mgrid[0:60, 0:80]
    slope = 2.0 + 0.01 * x + 0.02 * y
    bump = np.zeros_like(slope)
    bump[20:30, 30:45] = 0.3
    r = room.detrend(slope + bump)
    assert r[20:30, 30:45].min() > 0.8
    outside = np.ones_like(r, bool)
    outside[18:32, 28:47] = False
    assert r[outside].mean() < 0.05


def test_grid_shape_and_range():
    d, rgb = bumpy()
    H, C = room.grid(d, rgb, cells=16)
    assert H.shape == (12, 16) and C.shape == (12, 16, 3)
    assert 0 <= H.min() and H.max() <= 1
    assert 0 <= C.min() and C.max() <= 1


def test_columns_are_static_boxes_inside_the_photo():
    d, rgb = bumpy()
    H, C = room.grid(d, rgb, cells=16)
    p = room.Photo(file=None, width=3.2, relief=0.08, cells=16)
    t = room.columns(H, C, p)
    assert t.size == (1.6, pytest.approx(1.2))
    assert all(o["kind"] == "static" and o["shape"] == "box" for o in t.objects)
    for o in t.objects:
        assert abs(o["pos"][0]) + o["size"][0] <= 1.6 + 1e-6
        assert abs(o["pos"][1]) + o["size"][1] <= 1.2 + 1e-6
        assert 0 < 2 * o["size"][2] <= 0.08 + 1e-6
        assert o["pos"][2] == pytest.approx(o["size"][2])
    assert len({o["name"] for o in t.objects}) == len(t.objects)


def test_spawn_sits_on_the_flat_patch_and_is_cleared():
    d, rgb = bumpy()
    H, C = room.grid(d, rgb, cells=16)
    p = room.Photo(file=None, width=3.2, cells=16, clear=0.3)
    t = room.columns(H, C, p)
    sx, sy = t.spawn
    assert -1.6 < sx < 1.6 and -1.2 < sy < 1.2
    assert all(np.hypot(o["pos"][0] - sx, o["pos"][1] - sy) >= 0.3 for o in t.objects)


def test_spec_photo_block(tmp_path):
    from PIL import Image
    img = tmp_path / "desk.png"
    Image.new("RGB", (8, 6)).save(img)
    s = spec.parse({"name": "d", "photo": {"file": "desk.png", "width": 2.5}}, base=tmp_path)
    assert s.photo.file == img.resolve() and s.photo.width == 2.5
    assert s.spawn_at is None
    with pytest.raises(spec.SpecError, match="unknown keys"):
        spec.parse({"name": "d", "photo": {"file": "desk.png", "hight": 1}}, base=tmp_path)
    with pytest.raises(spec.SpecError, match="needs a photo"):
        spec.parse({"name": "d", "spawn": {"at": "auto"}})
