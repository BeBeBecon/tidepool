from __future__ import annotations

import math
from typing import Any

from . import Ctx, toy

RED, WHITE = [0.86, 0.2, 0.14, 1.0], [0.98, 0.94, 0.84, 1.0]
MATERIALS = [
    {"name": "chochin", "rgba": [1, 1, 1, 1], "emission": 0.55, "shininess": 0.1, "specular": 0.1},
    {"name": "lacquer", "rgba": [0.08, 0.06, 0.05, 1], "shininess": 0.6, "specular": 0.4},
    {"name": "cloth", "rgba": [1, 1, 1, 1], "shininess": 0.05, "specular": 0.05},
]


def _deco(ctx: Ctx, part: str, shape: str, size, pos, material: str, rgba=None, **kw) -> dict[str, Any]:
    o = {"name": ctx.name(part), "shape": shape, "size": size, "pos": pos, "kind": "static",
         "material": material, "collision": False, **kw}
    if rgba:
        o["rgba"] = rgba
    return o


@toy
def chochin(ctx: Ctx, start: tuple[float, float] = (1.0, -1.0), end: tuple[float, float] = (1.0, 1.0),
            count: int = 7, height: float = 0.55, size: float = 0.06, colors: str = "red",
            sag: float = 0.06) -> list[dict[str, Any]]:
    """Paper lanterns hung on a line between two poles. They glow; nothing touches them."""
    (x0, y0), (x1, y1) = start, end
    pole = height + size * 2
    out = [_deco(ctx, "pole", "cylinder", [0.012, pole / 2], [x, y, pole / 2], "wood_dark")
           for x, y in (start, end)]
    for i in range(count):
        u = (i + 1) / (count + 1)
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        z = height - sag * math.sin(math.pi * u)
        c = RED if colors == "red" or (colors == "mix" and i % 2 == 0) else WHITE
        out.append(_deco(ctx, "body", "ellipsoid", [size * .8, size * .8, size], [x, y, z], "chochin", c))
        for s in (1, -1):
            out.append(_deco(ctx, "cap", "cylinder", [size * .45, size * .1], [x, y, z + s * size * .95], "lacquer"))
        out.append(_deco(ctx, "cord", "cylinder", [0.002, size * .4], [x, y, z + size * 1.4], "lacquer"))
    # the line itself, a few straight pieces along the sag
    pts = [(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, height + size * 1.8 - sag * math.sin(math.pi * u))
           for u in (k / 8 for k in range(9))]
    for a, b in zip(pts, pts[1:]):
        o = _deco(ctx, "line", "capsule", [0.002], None, "lacquer", fromto=[*a, *b])
        del o["pos"]
        out.append(o)
    return out


@toy
def maku(ctx: Ctx, start: tuple[float, float] = (1.2, -1.5), end: tuple[float, float] = (1.2, 1.5),
         height: float = 0.7, stripe: float = 0.18) -> list[dict[str, Any]]:
    """A red and white festival curtain, standing on the floor."""
    (x0, y0), (x1, y1) = start, end
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(2, round(length / stripe))
    yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
    w = length / n
    out = []
    for i in range(n):
        u = (i + .5) / n
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        out.append(_deco(ctx, "stripe", "box", [w / 2, 0.006, height / 2], [x, y, height / 2], "cloth",
                         RED if i % 2 == 0 else WHITE, euler=[0, 0, yaw]))
    top = [(x0 + x1) / 2, (y0 + y1) / 2, height + 0.02]
    out.append(_deco(ctx, "rail", "box", [length / 2, 0.012, 0.02], top, "lacquer", euler=[0, 0, yaw]))
    return out
