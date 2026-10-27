"""Built-in shells for when there is no photo: simple shapes, in the CAD frame.

Millimetres, front is -y, up is +z, and the stock shell's top is at z = 78.7.
Each model returns its parts as (mesh, color slot) and the colors of slots 1-3;
slot 0 is the stock shell, red unless the model gives a third value.
"""

from __future__ import annotations

import numpy as np
import trimesh

BASE = 68.0      # sink a little into the stock shell so the union is one piece


def blob(c, r, sub=3):
    m = trimesh.creation.icosphere(subdivisions=sub)
    m.apply_scale(r)
    m.apply_translation(c)
    return m


def box(c, ext, tilt=None):
    m = trimesh.creation.box(extents=ext)
    if tilt:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(tilt[0]), tilt[1]))
    m.apply_translation(c)
    return m


def cyl(c, r, h, axis=(0, 0, 1)):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=48)
    z = np.array([0.0, 0.0, 1.0])
    a = np.asarray(axis, float)
    if not np.allclose(a, z):
        m.apply_transform(trimesh.geometry.align_vectors(z, a))
    m.apply_translation(c)
    return m


def cut(a, b):
    return trimesh.boolean.difference([a, b], engine="manifold")


def join(*ms):
    return trimesh.boolean.union(list(ms), engine="manifold")


def sneaker():
    sole = join(box((0, -5, BASE + 8), (70, 120, 16)),
                cyl((0, -65, BASE + 8), 35, 16), cyl((0, 55, BASE + 8), 35, 16))
    upper = cut(blob((0, -5, BASE + 18), (36, 68, 40)), cyl((0, 32, BASE + 58), 24, 40))
    upper = join(upper, box((0, 64, BASE + 44), (18, 8, 24)))
    toe = blob((0, -56, BASE + 18), (32, 26, 22))
    laces = [box((0, y, BASE + 18 + 40 * np.sqrt(max(0.0, 1 - ((y + 5) / 68) ** 2)) - 1), (36, 5, 5))
             for y in (-34, -22, -10, 2)]
    parts = [(sole, 2), (upper, 1), (toe, 2)] + [(l, 2) for l in laces]
    return parts, ["#2D4A8A", "#F4F2EE"]


def shiba():
    brown, white, black = 1, 2, 3
    body = blob((0, 12, BASE + 24), (42, 60, 34))
    chest = blob((0, -34, BASE + 26), (26, 18, 24))
    head = blob((0, -56, BASE + 54), (38, 32, 33))
    muzzle = blob((0, -84, BASE + 46), (19, 14, 13))
    nose = blob((0, -98, BASE + 50), (6, 4, 5), sub=2)
    eyes = [blob((x, -82, BASE + 62), (4.5, 3, 5), sub=2) for x in (-15, 15)]
    ears = []
    for x in (-21, 21):
        e = trimesh.creation.cone(radius=12, height=24, sections=24)
        e.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-18 if x < 0 else 18), (0, 1, 0)))
        e.apply_translation((x, -52, BASE + 76))
        ears.append(e)
    paws = [blob((x, -46, BASE + 10), (12, 16, 9)) for x in (-24, 24)]
    tail = trimesh.creation.torus(major_radius=13, minor_radius=7)
    tail.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    tail.apply_translation((0, 70, BASE + 48))
    parts = ([(body, brown), (chest, white), (head, brown), (muzzle, white), (nose, black), (tail, brown)]
             + [(e, black) for e in eyes] + [(e, brown) for e in ears] + [(p, white) for p in paws])
    return parts, ["#D9893F", "#F6F0E6", "#2B2420"]


def crate():
    """The box the robot came in, lying open to the front."""
    top = BASE + 110
    outer = box((0, -5, BASE + 55), (150, 140, 110))
    inner = box((0, -17, BASE + 55), (138, 140, 98))
    shell = cut(outer, inner)
    flap = box((0, -75 - 22, top - 10), (148, 48, 4), tilt=(55, (1, 0, 0)))
    tape = box((0, -5, top + 0.6), (40, 141, 1.5))
    label = box((75.6, 15, BASE + 62), (1.5, 62, 40))
    return [(shell, 1), (flap, 1), (tape, 2), (label, 3)], ["#C8A26A", "#9C7440", "#F4F2EE"]


def blade(spine, width, thick):
    """A flat strip of varying width along a 2D spine (x, z), `thick` mm deep in y. Watertight."""
    spine = np.asarray(spine, float)
    d = np.gradient(spine, axis=0)
    n = np.c_[-d[:, 1], d[:, 0]] / np.linalg.norm(d, axis=1)[:, None]
    left, right = spine + n * width[:, None] / 2, spine - n * width[:, None] / 2
    ring = np.vstack([left, right[::-1]])                       # outline, once round
    k = len(spine)
    v = np.vstack([np.c_[ring[:, 0], np.full(len(ring), -thick / 2), ring[:, 1]],
                   np.c_[ring[:, 0], np.full(len(ring), thick / 2), ring[:, 1]]])
    m = len(ring)
    f = []
    for i in range(k - 1):                                       # front and back faces as a strip
        a, b, c, e = i, i + 1, 2 * k - 2 - i, 2 * k - 1 - i      # left i, left i+1, right i+1, right i
        f += [[a, c, b], [a, e, c], [a + m, b + m, c + m], [a + m, c + m, e + m]]
    for i in range(m):                                           # walls
        j = (i + 1) % m
        f += [[i, j, j + m], [i, j + m, i + m]]
    mesh = trimesh.Trimesh(v, f, process=True)
    mesh.fix_normals()
    return mesh


def kabuto():
    """Not a thing on the back: the shell itself, blue, with a samurai helmet's horns."""
    gold, navy = 1, 2
    t = np.linspace(0, 1, 40)
    horns = []
    for side in (1, -1):
        spine = np.c_[side * (6 + 54 * t ** 1.6), 112 * t]
        h = blade(spine, 15 * (1 - t) ** 0.6 + 2.0, 5)
        h.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-14), (1, 0, 0)))
        h.apply_translation((0, -70, 74))
        horns.append(h)
    crest = cyl((0, -76, 82), 15, 7, axis=(0, 1, 0))
    boss = cyl((0, -80, 82), 8, 4, axis=(0, 1, 0))
    # the ridge down the middle of a helmet bowl, following the shell's top
    ys = np.linspace(-62, 62, 14)
    top = np.interp(ys, [-75, -60, -40, 0, 40, 80], [78.2, 78.7, 78.1, 74.4, 67.1, 55.3])
    ridge = join(*[blob((0, y, z), (5, 9, 4), sub=2) for y, z in zip(ys, top)])
    parts = [(h, gold) for h in horns] + [(crest, gold), (boss, navy), (ridge, navy)]
    return parts, ["#D4A93A", "#14264F"], "#1F4FA8"


MODELS = {"sneaker": sneaker, "shiba": shiba, "box": crate, "kabuto": kabuto}
