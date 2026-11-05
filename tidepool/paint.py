"""Paint jobs for the stock shell: up to four colors face by face, one color per limb part."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import shell

LIMB = re.compile(r"[LR][FMR]_\w+_link\Z")
PAD = re.compile(r"pad|insert|foot_tip|finger_tip")


def reach(link: str) -> float:
    """0 at the body, 1 at the tip."""
    for k, t in (("shoulder", 0), ("hip", 0), ("upper_arm", .25), ("thigh", .25),
                 ("forearm", .55), ("calf", .55), ("palm", .75)):
        if k in link:
            return t
    return 1.0


def band(t, n, rng):
    """Steps of a gradient, the seams dithered so four filaments pass for a fade."""
    v = np.clip(t, 0, 0.999) * n
    i = np.floor(v)
    seam = np.clip((v - i - 0.25) / 0.5, 0, 1)
    return np.minimum(i + (rng.random(len(t)) < seam), n - 1).astype(int)


def sunset(c, u, rng):
    return band(u[:, 1], 4, rng)


def tidepool(c, u, rng):
    return band(u[:, 2], 4, rng)


def sprinkles(c, u, rng, gap=20.0, r=6.5):
    from scipy.spatial import cKDTree

    dots = []
    for p in c[rng.permutation(len(c))]:        # dart throwing on the surface, so the flat top gets some too
        if all(np.linalg.norm(p - d) > gap for d in dots[-400:]):
            dots.append(p)
    dist, i = cKDTree(dots).query(c)
    return np.where(dist < r, 1 + i % 3, 0)


LEGS = {"LF": "#FF5D73", "RF": "#3EC1D3", "LM": "#FFC93C", "RM": "#7BD389", "LR": "#B388EB", "RR": "#FF9F43"}

# name: (shell colors, shell pattern, limb part -> color)
PAINTS = {
    "sunset": (["#FFB347", "#F0603C", "#C2307A", "#4B2A7A"], sunset,
               lambda l, p: "#FFF2D6" if PAD.search(l) else p[3 - round(reach(l) * 3)]),
    "tidepool": (["#0F2F4A", "#1B6F8A", "#3FB3B0", "#BFE8D8"], tidepool,
                 lambda l, p: "#E9D8A6" if PAD.search(l) else p[round(reach(l) * 2)]),
    "sprinkles": (["#FFF1D0", "#FF5D73", "#3EC1D3", "#FFC93C"], sprinkles,
                  lambda l, p: p[0] if PAD.search(l) else LEGS[l[:2]]),
}


def parts(kind: str, links: list[str]) -> dict[str, str]:
    pal, _, part = PAINTS[kind]
    return {l: part(l, pal) for l in links if LIMB.match(l)}


def make(kind: str, name: str | None = None, out: Path | None = None) -> Path:
    import json
    import xml.etree.ElementTree as ET

    import trimesh

    from . import paths

    if kind not in PAINTS:
        raise shell.ShellError(f"no paint {kind!r}; have {', '.join(PAINTS)}")
    name = name or kind
    out = shell._fresh(name, out, None)
    pal, pattern, _ = PAINTS[kind]
    cad = out / f"{name}-shell-mm.stl"
    shell.stock_shell().subdivide().export(cad)        # finer faces, finer dither
    m = trimesh.load_mesh(cad, process=True)
    c = m.triangles_center
    u = (c - c.min(0)) / np.ptp(c, 0)                   # front is -y in the CAD frame
    slots = pattern(c, u, np.random.default_rng(0))
    profile = paths.profile()
    base = json.loads(profile.read_text(encoding="utf-8"))["baseline_mjcf"]
    links = [b.get("name") for b in ET.parse(profile.parent / base).iter("body")]
    return shell.package(out, name, cad, slots, pal, parts=parts(kind, links))
