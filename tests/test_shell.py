import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("manifold3d")

from tidepool import paths, shell


def mug(tilt=(0.4, 0.0, 0.7)):
    body = trimesh.creation.cylinder(radius=0.3, height=0.6, sections=48)
    hole = trimesh.creation.cylinder(radius=0.26, height=0.6, sections=48)
    hole.apply_translation([0, 0, 0.06])
    handle = trimesh.creation.torus(major_radius=0.14, minor_radius=0.035)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([0.42, 0, 0])
    m = trimesh.boolean.union([trimesh.boolean.difference([body, hole], engine="manifold"), handle],
                              engine="manifold")
    m.apply_transform(trimesh.transformations.rotation_matrix(np.linalg.norm(tilt), tilt))
    return m


def test_lay_down_puts_mouth_front_and_handle_up():
    m = shell.lay_down(mug(), length=150, smooth=0)
    v = m.vertices
    top = v[np.argmax(v[:, 2])]
    assert abs(top[0]) < 15                      # handle on the centre line
    assert np.abs(v[:, 0]).max() <= shell.HALF_WIDTH + 1e-6
    assert v[:, 2].min() > shell.SHELL_TOP - 20
    # an open mouth: a ray along the axis from the front gets past the middle before it hits the base
    axis_z = v[:, 2].min() + np.abs(v[:, 0]).max()       # body radius above its lowest point
    hit, _, _ = m.ray.intersects_location([[0, -400, axis_z]], [[0, 1, 0]], multiple_hits=False)
    assert hit[0, 1] > v[:, 1].mean()


def test_palette_finds_ground_and_white():
    rng = np.random.default_rng(0)
    pink = rng.normal([200, 140, 130], 4, (6000, 3))
    brown = rng.normal([120, 80, 50], 4, (2500, 3))       # a print on the mug
    white = rng.normal([210, 210, 200], 3, (4000, 3))
    body, light = shell.palette(np.vstack([pink, brown, white]).astype(np.float32))
    r, g, b = (int(body[i:i + 2], 16) for i in (1, 3, 5))
    assert r > g > 0 and r - b > 40
    assert light.upper() > "#E0E0E0"


def test_ams_matches_shellflow_reader(tmp_path):
    design = pytest.importorskip("shellflow.simulation") if _has_design() else pytest.skip("no jumper-design")
    m = trimesh.creation.icosphere(subdivisions=2, radius=50)
    m.export(tmp_path / "s.stl")
    m = trimesh.load_mesh(tmp_path / "s.stl", process=True)
    slots = np.arange(len(m.faces)) % 3
    shell.write_ams(tmp_path / "ams.3mf", m, slots, ["#D9140D", "#A9856F", "#F0F0F0"])
    colors, palette = design.colors_from_ams(tmp_path / "ams.3mf", m)
    assert list(colors) == list(slots)
    assert palette == ["#D9140D", "#A9856F", "#F0F0F0"]


def _has_design():
    try:
        paths.jumper_design()
        return True
    except paths.MissingRepo:
        return False
