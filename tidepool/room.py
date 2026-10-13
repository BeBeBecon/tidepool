"""A photo as terrain.

Depth from the photo becomes a relief. Two ways to turn it into geometry:

  smooth   one textured surface mesh for the eye (no collision) and one invisible
           convex prism per grid cell for the feet. Slopes are planar, so the crab
           walks up them instead of climbing steps. The default.
  blocks   static box columns coloured from the photo. Chunky, cheap, no mesh files.

Neither uses a MuJoCo heightfield (an optional capability for map consumers) or a
single collision mesh (MuJoCo collides meshes by convex hull, so a bed would become
one bulge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from . import depth as depth_mod

TEXTURE = "photo.png"
TEXTURE_WIDTH = 2048
FINE = 6            # visual vertices per collision cell
EPS = 0.004         # below this a cell is floor
LIFT = 0.002        # keeps the surface mesh off the ground plane


@dataclass
class Photo:
    file: Path
    width: float = 3.0      # metres across the photo
    relief: float = 0.12    # tallest point
    cells: int = 20         # collision cells across
    style: str = "smooth"   # or "blocks"
    gamma: float = 0.7      # <1 lifts low things (pillows) relative to the tallest
    quant: float = 0.01     # blocks only: height step
    clear: float = 0.36     # flat circle around the spawn
    flatten: bool = True    # subtract the perspective slope so only objects stand up


@dataclass
class Terrain:
    objects: list[dict[str, Any]]
    spawn: tuple[float, float]
    size: tuple[float, float]               # half-extents of the floor
    meshes: list[dict[str, str]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)   # relative path -> OBJ text


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


def heights(depth: np.ndarray, photo: Photo) -> np.ndarray:
    """Height in metres per pixel."""
    if photo.flatten:
        r = detrend(depth)
    else:
        r = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    return r ** photo.gamma * photo.relief


def grid(depth: np.ndarray, rgb: np.ndarray, cells: int, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell height (0..1) and mean colour, for the blocks style. rgb is HxWx3 in 0..1."""
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


class _Frame:
    """Photo pixels to world metres: x across, y up the photo (row 0 is +y)."""

    def __init__(self, rows: int, cols: int, width: float):
        self.rows, self.cols = rows, cols
        self.cell = width / cols
        self.width = width
        self.depth = self.cell * rows

    def x(self, c: float) -> float:
        return -self.width / 2 + c * self.cell

    def y(self, r: float) -> float:
        return self.depth / 2 - r * self.cell

    @property
    def size(self) -> tuple[float, float]:
        return (self.width / 2, self.depth / 2)


def columns(H: np.ndarray, C: np.ndarray, photo: Photo, prefix: str = "col") -> Terrain:
    rows, cols = H.shape
    f = _Frame(rows, cols, photo.width)
    sr, sc = flattest(H)
    spawn = (round(f.x(sc + 0.5), 3), round(f.y(sr + 0.5), 3))
    q = photo.quant
    objs = []
    for r in range(rows):
        for c in range(cols):
            hgt = round(H[r, c] * photo.relief / q) * q if q > 0 else H[r, c] * photo.relief
            x, y = f.x(c + 0.5), f.y(r + 0.5)
            if hgt < q / 2 or np.hypot(x - spawn[0], y - spawn[1]) < photo.clear:
                continue
            objs.append({
                "name": f"{prefix}_{r}_{c}",
                "shape": "box",
                "size": [round(f.cell / 2, 4), round(f.cell / 2, 4), round(hgt / 2, 4)],
                "pos": [round(x, 4), round(y, 4), round(hgt / 2, 4)],
                "kind": "static",
                "material": "plastic_white",
                "friction": "wood",
                "rgba": [*(round(float(v), 3) for v in C[r, c]), 1.0],
            })
    return Terrain(objs, spawn, f.size)


def corners(hm: np.ndarray, cells: int) -> np.ndarray:
    """Heights at the (rows+1) x (cells+1) corners of the collision grid, blurred to the cell."""
    h, w = hm.shape
    rows = max(2, round(cells * h / w))
    peak = float(hm.max()) or 1.0
    img = Image.fromarray((hm / peak * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(w / cells / 3))
    b = np.asarray(img, dtype=np.float32) / 255 * peak
    ys = np.round(np.linspace(0, h - 1, rows + 1)).astype(int)
    xs = np.round(np.linspace(0, w - 1, cells + 1)).astype(int)
    Hc = b[ys[:, None], xs[None, :]]
    Hc[Hc < EPS] = 0.0
    return Hc


def surface_obj(Hc: np.ndarray, f: _Frame, fine: int = FINE) -> str:
    """One textured surface over the whole photo, upsampled from the corner heights."""
    rows, cols = Hc.shape[0] - 1, Hc.shape[1] - 1
    big = Image.fromarray(Hc.astype(np.float32)).resize((cols * fine + 1, rows * fine + 1), Image.BICUBIC)
    Hv = np.maximum(np.asarray(big, dtype=np.float32), 0)
    nr, nc = Hv.shape
    out = []
    for i in range(nr):
        for j in range(nc):
            out.append(f"v {f.x(j / fine):.4f} {f.y(i / fine):.4f} {Hv[i, j] + LIFT:.4f}")
            out.append(f"vt {j / (nc - 1):.4f} {1 - i / (nr - 1):.4f}")
    vid = lambda i, j: i * nc + j + 1
    for i in range(nr - 1):
        for j in range(nc - 1):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            out.append(f"f {a}/{a} {d}/{d} {c}/{c}")
            out.append(f"f {a}/{a} {c}/{c} {b}/{b}")
    return "\n".join(out) + "\n"


def prism_obj(xy: list[tuple[float, float]], top: list[float]) -> str:
    v = [f"v {x:.4f} {y:.4f} -0.02" for x, y in xy]
    v += [f"v {x:.4f} {y:.4f} {z + LIFT:.4f}" for (x, y), z in zip(xy, top)]
    faces = ["f 1 4 3 2", "f 5 6 7 8", "f 1 2 6 5", "f 2 3 7 6", "f 3 4 8 7", "f 4 1 5 8"]
    return "\n".join(v + faces) + "\n"


def smooth(hm: np.ndarray, photo: Photo) -> Terrain:
    Hc = corners(hm, photo.cells)
    rows, cols = Hc.shape[0] - 1, Hc.shape[1] - 1
    f = _Frame(rows, cols, photo.width)
    cell_h = (Hc[:-1, :-1] + Hc[:-1, 1:] + Hc[1:, 1:] + Hc[1:, :-1]) / 4
    sr, sc = flattest(cell_h)
    spawn = (round(f.x(sc + 0.5), 3), round(f.y(sr + 0.5), 3))

    files = {"assets/surface.obj": surface_obj(Hc, f)}
    meshes = [{"name": "surface", "file": "assets/surface.obj"}]
    objs: list[dict[str, Any]] = [{
        "name": "surface", "shape": "mesh", "mesh": "surface", "kind": "static",
        "collision": False, "material": "photo",
    }]
    n = 0
    for r in range(rows):
        for c in range(cols):
            top = [Hc[r, c], Hc[r, c + 1], Hc[r + 1, c + 1], Hc[r + 1, c]]
            if max(top) < EPS:
                continue
            x0, x1, y0, y1 = f.x(c), f.x(c + 1), f.y(r), f.y(r + 1)
            name = f"p{n}"
            files[f"assets/{name}.obj"] = prism_obj([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], top)
            meshes.append({"name": name, "file": f"assets/{name}.obj"})
            objs.append({
                "name": name, "shape": "mesh", "mesh": name, "kind": "static",
                "material": "plastic_white", "friction": "wood", "rgba": [1, 1, 1, 0],
            })
            n += 1
    return Terrain(objs, spawn, f.size, meshes, files)


def terrain(photo: Photo, cache_dir: Path) -> Terrain:
    img = Image.open(photo.file).convert("RGB")
    d = depth_mod.cached(photo.file, cache_dir)
    if photo.style == "blocks":
        rgb = np.asarray(img, dtype=np.float32) / 255
        H, C = grid(d, rgb, photo.cells, flatten=photo.flatten)
        return columns(H, C, photo)
    if photo.style != "smooth":
        raise ValueError(f"photo.style must be smooth or blocks, not {photo.style!r}")
    return smooth(heights(d, photo), photo)


def write_assets(photo: Photo, t: Terrain, out: Path) -> None:
    """The texture and the mesh files, next to the scene JSON."""
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for stale in assets.glob("*.obj"):
        stale.unlink()
    for rel, text in t.files.items():
        (out / rel).write_text(text)
    img = Image.open(photo.file).convert("RGB")
    w = min(TEXTURE_WIDTH, img.width)
    img = img.resize((w, max(1, round(img.height * w / img.width))))
    img.save(out / TEXTURE, optimize=True)


def apply(doc: dict[str, Any], photo: Photo, t: Terrain) -> None:
    """Put the terrain into an mjscene document in place."""
    doc.setdefault("materials", []).append({
        "name": "photo", "rgba": [1, 1, 1, 1],
        "texture": {"file": TEXTURE}, "texrepeat": [1, 1], "texuniform": False,
    })
    if t.meshes:
        doc.setdefault("meshes", []).extend(t.meshes)
    doc["ground"] = {"material": "photo", "friction": "wood", "size": list(t.size)}
    doc["objects"] = t.objects + doc.get("objects", [])
    doc["tags"] = sorted({*doc.get("tags", []), "photo"})
