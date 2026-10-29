from __future__ import annotations

from typing import Any

from . import Ctx, toy
from . import paths

COLORS = [
    [0.90, 0.25, 0.20, 1.0],
    [0.98, 0.70, 0.15, 1.0],
    [0.25, 0.65, 0.35, 1.0],
    [0.20, 0.50, 0.90, 1.0],
    [0.70, 0.35, 0.80, 1.0],
]


@toy
def domino_run(ctx: Ctx, path: str = "line", count: int = 20, start: tuple[float, float] = (0.3, 0.0),
               height: float = 0.08, spacing: float | None = None, heading: float = 0.0,
               colors: str = "rainbow", mass: float = 0.03, **shape: Any) -> list[dict[str, Any]]:
    """A row of dominoes along a path: line, arc, spiral or zigzag.

    `height` is the domino height in metres; width and thickness follow the usual
    1 : 2 : 4 proportions. Spacing defaults to 0.6 x height, which topples reliably.
    Extra keyword arguments go to the path: `radius`/`left` for arc, `inner`/`growth`/
    `left` for spiral, `leg`/`angle` for zigzag.
    """
    if count < 1:
        return []
    if path not in paths.SHAPES:
        raise ValueError(f"domino_run: path must be one of {', '.join(paths.SHAPES)}")
    if spacing is None:
        spacing = 0.6 * height
    hz = height / 2
    hy = height / 4
    hx = height / 16

    kwargs: dict[str, Any] = dict(start=tuple(start), count=count, spacing=spacing, **shape)
    if path in ("line", "arc", "zigzag"):
        kwargs["heading"] = heading
    try:
        samples = paths.SHAPES[path](**kwargs)
    except TypeError as e:
        raise ValueError(f"domino_run: {e}") from None

    out = []
    for i, (x, y, h) in enumerate(samples):
        obj = {
            "name": ctx.name(),
            "shape": "box",
            "size": [hx, hy, hz],
            "pos": [x, y, hz + 0.001],
            "euler": [0.0, 0.0, h],
            "kind": "dynamic",
            "mass": mass,
            "material": "plastic_white",
            "friction": "wood",
        }
        if colors == "rainbow":
            obj["rgba"] = COLORS[i % len(COLORS)]
        elif colors != "white":
            raise ValueError("domino_run: colors must be 'rainbow' or 'white'")
        out.append(obj)
    return out
