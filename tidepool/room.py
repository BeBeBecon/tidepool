"""A photo as terrain: depth becomes a grid of box columns, the photo itself is the floor.

Columns are static boxes, not a heightfield or a mesh. A mesh collides as its convex
hull (a desk would become one bulge) and heightfields are an optional capability for
map consumers; boxes work everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import depth as depth_mod

TEXTURE = "photo.png"
TEXTURE_WIDTH = 1024


@dataclass
class Photo:
    file: Path
    width: float = 3.0      # metres across the photo
    relief: float = 0.06    # tallest column
    cells: int = 20         # columns across
    quant: float = 0.01     # height step; gives the voxel look and merges near-equal cells
    clear: float = 0.36     # flat circle around the spawn
    flatten: bool = True    # subtract the perspective slope so only objects stand up


@dataclass
class Terrain:
    objects: list[dict[str, Any]]
    spawn: tuple[float, float]
    size: tuple[float, float]   # half-extents of the floor


def detrend(depth: np.ndarray) -> np.ndarray:
    """What sticks up from the dominant surface, 0..1.

    A photo taken at an angle is mostly perspective: near is close, far is far, and
    the pillow on the bed is a ripple on that slope. Fit a plane to the surface
    (twice, the second time ignoring whatever stood out the first time) and keep
    only what is closer to the camera than the plane.
    """
    h, w = depth.shape
    step = max(1, min(h, w) // 64)
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].ravel().astype(np.float64)
    A = np.c_[xs.ravel() / w, ys.ravel() / h, np.ones(z.size)]
    keep = np.ones(z.size, bool)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        keep = res < np.percentile(res, 60)
    yy, xx = np.mgrid[0:h, 0:w]
    plane = coef[0] * xx / w + coef[1] * yy / h + coef[2]
    r = np.maximum(depth - plane, 0)
    top = np.percentile(r, 99.5)
    return np.clip(r / (top + 1e-9), 0, 1)


def grid(depth: np.ndarray, rgb: np.ndarray, cells: int, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell height (0..1) and mean colour. rgb is HxWx3 in 0..1."""
    h, w = depth.shape
    rows = max(2, round(cells * h / w))
    if flatten:
        d = detrend(depth)
    else:
        d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    H = np.zeros((rows, cells))
    C = np.zeros((rows, cells, 3))
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cells + 1).astype(int)
    for r in range(rows):
        for c in range(cells):
            block = d[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            H[r, c] = np.percentile(block, 75)   # small things survive the cell
            C[r, c] = rgb[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].reshape(-1, 3).mean(0)
    return H, C


def _mean3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9


def flattest(H: np.ndarray) -> tuple[int, int]:
    """Row and column of the low, flat 3x3 patch where the crab should start."""
    m = _mean3(H)
    sd = np.sqrt(np.maximum(_mean3(H ** 2) - m ** 2, 0))
    r, c = np.unravel_index(np.argmin(m + 2 * sd), H.shape)
    return int(r), int(c)


def columns(H: np.ndarray, C: np.ndarray, photo: Photo, prefix: str = "col") -> Terrain:
    rows, cols = H.shape
    cell = photo.width / cols
    depth_m = cell * rows
    cx = lambda c: -photo.width / 2 + (c + 0.5) * cell
    cy = lambda r: depth_m / 2 - (r + 0.5) * cell   # image row 0 is the far edge
    sr, sc = flattest(H)
    spawn = (round(cx(sc), 3), round(cy(sr), 3))
    q = photo.quant
    objs = []
    for r in range(rows):
        for c in range(cols):
            hgt = round(H[r, c] * photo.relief / q) * q if q > 0 else H[r, c] * photo.relief
            if hgt < q / 2 or np.hypot(cx(c) - spawn[0], cy(r) - spawn[1]) < photo.clear:
                continue
            objs.append({
                "name": f"{prefix}_{r}_{c}",
                "shape": "box",
                "size": [round(cell / 2, 4), round(cell / 2, 4), round(hgt / 2, 4)],
                "pos": [round(cx(c), 4), round(cy(r), 4), round(hgt / 2, 4)],
                "kind": "static",
                "material": "plastic_white",
                "friction": "wood",
                "rgba": [*(round(float(v), 3) for v in C[r, c]), 1.0],
            })
    return Terrain(objs, spawn, (photo.width / 2, depth_m / 2))


def terrain(photo: Photo, cache_dir: Path) -> Terrain:
    img = Image.open(photo.file).convert("RGB")
    d = depth_mod.cached(photo.file, cache_dir)
    rgb = np.asarray(img, dtype=np.float32) / 255
    H, C = grid(d, rgb, photo.cells, flatten=photo.flatten)
    return columns(H, C, photo)


def write_texture(photo: Photo, out: Path) -> Path:
    img = Image.open(photo.file).convert("RGB")
    w = min(TEXTURE_WIDTH, img.width)
    img = img.resize((w, max(1, round(img.height * w / img.width))))
    dest = out / TEXTURE
    out.mkdir(parents=True, exist_ok=True)
    img.save(dest, optimize=True)
    return dest


def apply(doc: dict[str, Any], photo: Photo, t: Terrain) -> None:
    """Put the terrain into an mjscene document in place."""
    doc.setdefault("materials", []).append({
        "name": "photo", "rgba": [1, 1, 1, 1],
        "texture": {"file": TEXTURE}, "texrepeat": [1, 1], "texuniform": False,
    })
    doc["ground"] = {"material": "photo", "friction": "wood", "size": list(t.size)}
    doc["objects"] = t.objects + doc.get("objects", [])
    doc["tags"] = sorted({*doc.get("tags", []), "photo"})
