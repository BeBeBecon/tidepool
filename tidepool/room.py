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
class Movable:
    """Something in the photo the crab can push: a mask, extruded into a convex body."""
    mask: Path
    name: str = "thing"
    height: float | None = None   # metres; None takes it from the depth map
    mass: float = 0.3


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
    movable: list[Movable] = field(default_factory=list)


@dataclass
class Terrain:
    objects: list[dict[str, Any]]
    spawn: tuple[float, float]
    size: tuple[float, float]               # half-extents of the floor
    meshes: list[dict[str, str]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)   # relative path -> OBJ text
    face: tuple[float, float] | None = None               # what an auto spawn should look at


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


def smooth(hm: np.ndarray, photo: Photo, avoid: np.ndarray | None = None) -> Terrain:
    """`avoid` is a pixel mask of places the spawn must not pick (under a movable)."""
    Hc = corners(hm, photo.cells)
    rows, cols = Hc.shape[0] - 1, Hc.shape[1] - 1
    f = _Frame(rows, cols, photo.width)
    cell_h = (Hc[:-1, :-1] + Hc[:-1, 1:] + Hc[1:, 1:] + Hc[1:, :-1]) / 4
    if avoid is not None:
        taken = Image.fromarray(avoid.astype(np.uint8) * 255).resize((cols, rows), Image.BOX)
        cell_h = cell_h + (np.asarray(taken) > 0) * 10.0
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


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """A hand-painted mask as a bool array the size of the depth map.

    Paint on a transparent layer, or white on black: the alpha channel counts
    when there is one, brightness otherwise.
    """
    img = Image.open(path)
    img = img.convert("LA") if img.mode in ("RGBA", "LA", "P") else img.convert("L")
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    a = np.asarray(img)
    if a.ndim == 3:
        return a[..., 1] > 127
    return a > 127


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * px + 1))
    return np.asarray(img) > 0


def hull(pts: np.ndarray, limit: int = 32) -> np.ndarray:
    """Convex hull of 2-D points (monotone chain), thinned to at most `limit` corners."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = np.array(lower[:-1] + upper[:-1])
    if len(h) > limit:
        h = h[np.linspace(0, len(h), limit, endpoint=False).astype(int)]
    return h


def extrude(mask: np.ndarray, hm: np.ndarray, m: Movable, f: _Frame) -> tuple[str, list[float], float]:
    """The masked region as a prism standing on the floor, textured with the photo.

    Returns the OBJ text, the body position and the height. Vertices are centred
    on the footprint so MuJoCo's free joint sits in the middle of the object.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        raise ValueError(f"movable {m.name!r}: the mask is empty")
    px = f.width / w
    world = np.c_[-f.width / 2 + (xs + 0.5) * px, f.depth / 2 - (ys + 0.5) * px]
    ring = hull(world)
    if m.height is not None:
        height = m.height
    else:
        # Monocular depth flattens small things, so a plush reads as a few cm.
        # Nothing on a bed is flatter than 0.4 x its narrow side.
        narrow = float(min(np.ptp(ring[:, 0]), np.ptp(ring[:, 1])))
        height = max(float(np.percentile(hm[mask], 90)), 0.4 * narrow, 0.03)
    cx, cy = ring.mean(0)
    v, vt = [], []
    for z in (0.0, height):
        for x, y in ring:
            v.append(f"v {x - cx:.4f} {y - cy:.4f} {z:.4f}")
            vt.append(f"vt {(x + f.width / 2) / f.width:.4f} {(y + f.depth / 2) / f.depth:.4f}")
    n = len(ring)
    bot = lambda k: k + 1
    top = lambda k: n + k + 1
    faces = []
    for k in range(1, n - 1):
        faces.append(f"f {top(0)}/{top(0)} {top(k)}/{top(k)} {top(k + 1)}/{top(k + 1)}")
        faces.append(f"f {bot(0)}/{bot(0)} {bot(k + 1)}/{bot(k + 1)} {bot(k)}/{bot(k)}")
    for k in range(n):
        a, b, c, d = bot(k), bot((k + 1) % n), top((k + 1) % n), top(k)
        faces.append(f"f {a}/{a} {b}/{b} {c}/{c}")
        faces.append(f"f {a}/{a} {c}/{c} {d}/{d}")
    text = "\n".join(v + vt + faces) + "\n"
    return text, [round(float(cx), 4), round(float(cy), 4), LIFT], height


def terrain(photo: Photo, cache_dir: Path) -> Terrain:
    img = Image.open(photo.file).convert("RGB")
    d = depth_mod.cached(photo.file, cache_dir)
    if photo.style == "blocks":
        if photo.movable:
            raise ValueError("photo.movable needs style: smooth")
        rgb = np.asarray(img, dtype=np.float32) / 255
        H, C = grid(d, rgb, photo.cells, flatten=photo.flatten)
        return columns(H, C, photo)
    if photo.style != "smooth":
        raise ValueError(f"photo.style must be smooth or blocks, not {photo.style!r}")
    hm = heights(d, photo)
    if not photo.movable:
        return smooth(hm, photo)

    # Flatten the ground under each movable (wider than the mask, or the blurred
    # collision prisms poke out under it), then stand the object there.
    h, w = hm.shape
    masks = [read_mask(m.mask, hm.shape) for m in photo.movable]
    avoid = np.zeros(hm.shape, bool)
    for mask in masks:
        avoid |= dilate(mask, int(w / photo.cells))
    t = smooth(np.where(avoid, 0.0, hm), photo, avoid)
    rows = max(2, round(photo.cells * h / w))
    f = _Frame(rows, photo.cells, photo.width)
    for m, mask in zip(photo.movable, masks):
        text, pos, _ = extrude(mask, hm, m, f)
        name = m.name
        t.files[f"assets/{name}.obj"] = text
        t.meshes.append({"name": name, "file": f"assets/{name}.obj"})
        t.objects.append({
            "name": name, "shape": "mesh", "mesh": name, "kind": "dynamic",
            "mass": m.mass, "material": "photo", "friction": "wood", "pos": pos,
        })
        if t.face is None:
            t.face = (pos[0], pos[1])
    return t


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
