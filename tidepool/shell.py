"""Photo of a thing -> a Jumper shell (.skin) wearing it, hermit crab style.

TripoSR makes the mesh, the mesh is laid on its side with its opening to the
front and fused onto the stock shell, and jumper-design's shellflow builds and
verifies the package. Colors go in through a Bambu-style ams.3mf, the same way
a 4-color print would.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from . import paths

STOCK_RED = "#D9140D"
HEX = re.compile(r"#[0-9A-Fa-f]{6}\Z")
CODES = ["4", "8", "0C", "1C"]       # Bambu paint codes for filament slots 1-4
SHELL_TOP = 78.7                     # mm, top of the stock shell in CAD frame
HALF_WIDTH = 88.0                    # mm, past this the legs swing through it


class ShellError(RuntimeError):
    pass


def triposr() -> Path:
    return paths._find("TIDEPOOL_TRIPOSR", "TripoSR", "tsr/system.py")


def reconstruct(photo: Path, work: Path, res: int = 256):
    """Photo -> (trimesh with vertex colors, RGB pixels of the cut-out thing)."""
    import types

    import rembg
    import torch
    import trimesh
    from PIL import Image
    from skimage.measure import marching_cubes

    def mcubes(level, iso):
        v, f, _, _ = marching_cubes(level.cpu().numpy(), iso)
        return torch.from_numpy(v[:, [2, 1, 0]].copy()).float(), torch.from_numpy(f[:, [0, 2, 1]].copy()).long()

    sys.modules.setdefault("torchmcubes", types.SimpleNamespace(marching_cubes=mcubes))
    sys.path.insert(0, str(triposr()))
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    # transformers 5 renamed the ViT layers the checkpoint was saved with
    rename = [(r"\.attention\.attention\.query", ".attention.q_proj"),
              (r"\.attention\.attention\.key", ".attention.k_proj"),
              (r"\.attention\.attention\.value", ".attention.v_proj"),
              (r"\.attention\.output\.dense", ".attention.o_proj"),
              (r"\.intermediate\.dense", ".mlp.fc1"),
              (r"\.output\.dense", ".mlp.fc2")]
    load = TSR.load_state_dict

    def remap(self, sd, *a, **kw):
        out = {}
        for k, v in sd.items():
            if ".encoder.layer." in k:
                k = k.replace(".encoder.layer.", ".layers.")
                for pat, rep in rename:
                    k = re.sub(pat, rep, k)
            out[k] = v
        return load(self, out, *a, **kw)

    TSR.load_state_dict = remap
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)

    img = Image.open(photo)
    img.thumbnail((1024, 1024))
    cut = remove_background(img, rembg.new_session("u2net"))    # bria's default is non-commercial
    a = np.asarray(cut).astype(np.float32) / 255.0
    pixels = (a[..., :3][a[..., 3] > 0.95] * 255).astype(np.float32)
    cut = resize_foreground(cut, 0.85)
    a = np.asarray(cut).astype(np.float32) / 255.0
    a = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4]) * 0.5
    framed = Image.fromarray((a * 255).astype(np.uint8))
    framed.save(work / "input.png")

    with torch.no_grad():
        codes = model([framed], device="cpu")
    mesh = model.extract_mesh(codes, True, resolution=res)[0]
    mesh.export(work / "mesh.ply")
    return trimesh.load(work / "mesh.ply", process=True), pixels


def lay_down(m, length: float = 150.0, smooth: int = 80):
    """Orient a cup-like mesh: opening to the front (-y), handle up, sitting on the shell. In place."""
    import trimesh

    m = max(m.split(only_watertight=False), key=lambda p: p.area)
    m.fill_holes()
    m.fix_normals()

    # the two biggest flat faces of the hull are the rim and the base
    hull = m.convex_hull
    areas = [hull.area_faces[f].sum() for f in hull.facets]
    axis = hull.facets_normal[int(np.argmax(areas))]
    c = m.bounds.mean(0)
    ext = np.ptp(m.vertices @ axis)

    # the bounding box leans toward the handle, so find the body's own centre with a circle fit
    u = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    s, _ = trimesh.sample.sample_surface_even(m, 6000, seed=0)       # by area, not vertex density
    p = np.c_[(s - c) @ u, (s - c) @ v]
    keep = np.ones(len(p), bool)
    for _ in range(3):
        A = np.c_[2 * p[keep], np.ones(keep.sum())]
        (x0, y0, k), *_ = np.linalg.lstsq(A, (p[keep] ** 2).sum(1), rcond=None)
        radius = np.sqrt(k + x0 ** 2 + y0 ** 2)
        r = np.hypot(p[:, 0] - x0, p[:, 1] - y0)
        keep = r < radius * 1.1
    c = c + x0 * u + y0 * v
    i = np.argmax(r)                                                # the handle, or whatever sticks out most
    up = (p[i, 0] - x0) * u + (p[i, 1] - y0) * v
    up /= np.linalg.norm(up)

    def depth(sign):
        # rays fired inward from the open end travel further, into the mouth
        rng = np.random.default_rng(0)
        rr, t = rng.uniform(0, radius * 0.5, 64), rng.uniform(0, 2 * np.pi, 64)
        start = c + sign * axis * ext
        pts = start + rr[:, None] * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
        hit, _, _ = m.ray.intersects_location(pts, np.tile(-sign * axis, (64, 1)), multiple_hits=False)
        return np.mean((start - hit) @ (sign * axis)) if len(hit) else 0.0

    mouth = axis if depth(1) > depth(-1) else -axis

    src = np.stack([np.cross(up, mouth), mouth, up])
    rot = np.diag([1.0, -1.0, 1.0]) @ src
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, -rot @ c
    m.apply_transform(T)
    if np.linalg.det(rot) < 0:
        m.invert()
    m.apply_scale(length / np.ptp(m.vertices[:, 1]))
    half = np.abs(m.bounds[:, 0]).max()
    if half > HALF_WIDTH:
        m.apply_scale(HALF_WIDTH / half)
    lo, hi = m.bounds
    m.apply_translation([0, -5 - (lo[1] + hi[1]) / 2, SHELL_TOP - lo[2] - 12])
    if smooth:
        trimesh.smoothing.filter_taubin(m, iterations=smooth)
    return m


def palette(pixels: np.ndarray) -> list[str]:
    """Body and accent color from the photo: split on redness, white-balance on the lighter half."""
    from scipy.cluster.vq import kmeans2
    from skimage.filters import threshold_otsu

    red = pixels[:, 0] - pixels[:, 2]
    warm = red > threshold_otsu(red)
    if warm.all() or not warm.any():
        warm = pixels.mean(1) < np.median(pixels.mean(1))
    cent, k = kmeans2(pixels[warm], min(3, int(warm.sum())), seed=0, minit="++")
    # the ground color is a big cluster, and lighter than any print on it
    share = np.bincount(k, minlength=len(cent)) / len(k)
    big = np.flatnonzero(share >= 0.25)
    body = cent[big[np.argmax(cent[big].mean(1))]]
    light = np.percentile(pixels[~warm], 90, axis=0)
    gain = 240 / np.maximum(light, 1)

    def fix(c, sat):
        c = c * gain
        return np.clip(c.mean() + (c - c.mean()) * sat, 0, 255)

    return [_hex(fix(body, 1.6)), _hex(fix(light, 1.0))]


def _hex(c) -> str:
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in c)


def paint(shell, stock, colored, smooth: int = 12) -> np.ndarray:
    """Slot per face of `shell`: 0 stock, 1 body (the warmer half of the colors), 2 accent."""
    import trimesh
    from scipy.spatial import cKDTree
    from skimage.filters import threshold_otsu

    pts, _ = trimesh.sample.sample_surface_even(stock, 400000, seed=0)
    centers = shell.triangles_center
    d, _ = cKDTree(pts).query(centers)
    vc = colored.visual.vertex_colors[:, :3].astype(float)
    red = vc[:, 0] - vc[:, 2]
    lab = (red > threshold_otsu(red)).astype(float)
    # neighbours vote, so a printed pattern doesn't come out as speckle
    for _ in range(smooth):
        lab = np.array([lab[n].mean() if len(n) else x for n, x in zip(colored.vertex_neighbors, lab)])
    _, near = cKDTree(colored.vertices).query(centers)
    return np.where(d < 0.8, 0, np.where(lab[near] > 0.5, 1, 2))


def write_ams(path: Path, mesh, slots: np.ndarray, colors: list[str]) -> None:
    """A minimal Bambu 3MF: one object, per-triangle paint codes, vertices exactly as in the STL."""
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
    f32 = lambda x: repr(float(np.float32(x)))
    body = [f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" {ns}><resources>'
            '<object id="1" type="model"><mesh><vertices>']
    body += [f'<vertex x="{f32(a)}" y="{f32(b)}" z="{f32(c)}"/>' for a, b, c in mesh.vertices]
    body.append("</vertices><triangles>")
    body += [f'<triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{CODES[s]}"/>'
             for (a, b, c), s in zip(mesh.faces, slots)]
    body.append("</triangles></mesh></object></resources></model>")
    main = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" {ns} '
            'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"><resources>'
            '<object id="2" type="model"><components><component p:path="/3D/Objects/object_1.model" '
            'objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object></resources>'
            '<build><item objectid="2"/></build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("3D/3dmodel.model", main)
        z.writestr("3D/Objects/object_1.model", "".join(body))
        z.writestr("Metadata/project_settings.config", json.dumps({"filament_colour": colors}))


def stock_shell():
    """The stock upper shell in the CAD millimetre frame shellflow's seam check wants."""
    import trimesh

    design = paths.jumper_design()
    profile = json.loads(paths.profile().read_text(encoding="utf-8"))
    t = profile["cad_mm_to_parent_m"]
    m = trimesh.load_mesh(design / "robots/jumper/meshes/visual/upper_shell_link.stl")
    m.vertices = ((m.vertices - np.asarray(t["translation_m"])) @ np.asarray(t["rotation_matrix"])) / t["scale"]
    return m


def make(photo: Path, name: str, *, colors: list[str] | None = None, length: float = 150.0,
         out: Path | None = None) -> Path:
    import trimesh

    for c in colors or []:
        if not HEX.match(c):
            raise ShellError(f"color {c!r}: use #RRGGBB")
    out = out or paths.out_dir() / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sys.path.insert(0, str(paths.jumper_design() / "src"))
    from shellflow.packages import verify_package
    from shellflow.simulation import assemble
    from shellflow.skin_package import export_skin

    mesh, pixels = reconstruct(photo, out)
    thing = lay_down(mesh, length)
    colored = thing.copy()
    thing = thing.simplify_quadric_decimation(face_count=30000)
    stock = stock_shell()
    fused = trimesh.boolean.union([stock, trimesh.Trimesh(thing.vertices, thing.faces)], engine="manifold")
    cad = out / f"{name}-shell-mm.stl"
    fused.export(cad)

    fused = trimesh.load_mesh(cad, process=True)      # what assemble will see, face for face
    picked = palette(pixels)
    for i, c in enumerate(colors or []):
        picked[i] = c
    write_ams(out / "ams.3mf", fused, paint(fused, stock, colored), [STOCK_RED] + picked[:2])

    model = out / "assembly"
    assemble(cad, model, paths.profile(), faces=60000)
    report = json.loads((model / "simulation-report.json").read_text(encoding="utf-8"))
    if report["colors"]["mode"] == "monochrome":
        print(f"[shell] colors were not applied: {report['colors'].get('reason')}", file=sys.stderr)
    skin = out / f"{name}.skin"
    title = name.replace("-", " ").replace("_", " ").title()
    export_skin(model, paths.profile(), cad, skin, package_id=name, title=title)
    result = verify_package(skin, platform_profile=paths.profile(), mujoco=True)
    if not result.get("ok"):
        raise ShellError(f"verify-package: {json.dumps(result)}")
    print(f"[shell] colors {' '.join([STOCK_RED] + picked[:2])}")
    return skin


def preview(skin: Path, png: Path, view: str = "three-quarter") -> None:
    import subprocess

    design = paths.jumper_design()
    subprocess.run([paths.python(), str(design / "scripts/preview_simulation.py"),
                    str(skin.parent / "assembly"), "--render", str(png), "--view", view],
                   cwd=design, capture_output=True, timeout=300, check=False)
