"""The closet: every shell you have made, on one crab, in the browser.

`export` writes a folder the page can load: the robot without its shell once,
each shell on its own, the eye frames, and a manifest. `serve` puts it on
localhost, since browsers will not fetch .glb files from file://.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import face, paths

PAGE = Path(__file__).with_name("closet.html")
STOCK_RGBA = [0.85, 0.08, 0.05, 1.0]


def _posed(scene_xml: Path):
    import mujoco
    import sys

    sys.path.insert(0, str(paths.jumper_design() / "src"))
    from shellflow.simulation import load_mujoco

    m = load_mujoco(scene_xml)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _geom_mesh(m, d, g):
    import trimesh

    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    v = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return trimesh.Trimesh(v.copy(), f.copy(), process=False)


def _thin(mesh, keep: float):
    if keep >= 1 or len(mesh.faces) < 400:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=max(200, int(len(mesh.faces) * keep)))


def _linear(c: float) -> float:
    # MJCF colors are sRGB (jumper-design's be-web/1); glTF wants linear
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _glb(groups: dict[tuple, list], path: Path, keep: float) -> int:
    """One mesh per color, so the page draws a handful of objects instead of forty."""
    import trimesh
    from trimesh.visual.material import PBRMaterial

    scene = trimesh.Scene()
    faces = 0
    for rgba, meshes in groups.items():
        mesh = _thin(trimesh.util.concatenate(meshes), keep)
        mesh.merge_vertices()
        mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
            baseColorFactor=[_linear(c) for c in rgba[:3]] + [rgba[3]], metallicFactor=0.0, roughnessFactor=0.9))
        scene.add_geometry(mesh)
        faces += len(mesh.faces)
    scene.export(path)          # normals are creased on the page, so edges stay sharp
    return faces


def _is_visual(m, g) -> bool:
    import mujoco

    return (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0
            and m.geom_conaffinity[g] == 0 and m.geom_group[g] in (0, 2))


def export(skins: list[Path], out: Path, keep: float = 0.25) -> dict:
    import mujoco
    import trimesh

    if out.exists():
        shutil.rmtree(out)
    (out / "shells").mkdir(parents=True)
    (out / "eyes").mkdir()
    work = Path(tempfile.mkdtemp(prefix="tidepool-closet-"))
    name = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    entries = []
    body_done = False
    for skin in skins:
        dest = work / skin.stem
        with zipfile.ZipFile(skin) as z:
            z.extractall(dest)
        m, d = _posed(dest / "assembled" / "jumper" / "scene.xml")
        shell, body = {}, {}
        for g in range(m.ngeom):
            if not _is_visual(m, g):
                continue
            rgba = tuple(np.round(m.geom_rgba[g], 3).tolist())
            (shell if name(m, g).startswith("shellflow_upper_") else body).setdefault(rgba, []).append(_geom_mesh(m, d, g))
        if not body_done:
            n = _glb(body, out / "body.glb", keep)
            stock = trimesh.load_mesh(paths.profile().parent / "meshes/visual/upper_shell_link.stl")
            b = m.body("upper_shell_link").id
            stock.apply_transform(_pose(d.xmat[b], d.xpos[b]))
            _glb({tuple(STOCK_RGBA): [stock]}, out / "shells" / "stock.glb", 0.25)
            entries.append({"id": "stock", "title": "Stock"})
            base = m.body("base_link").id
            eyes = _eye_quads(d.xmat[base].reshape(3, 3), d.xpos[base])
            print(f"[closet] body {n} faces")
            body_done = True
        _glb(shell, out / "shells" / f"{skin.stem}.glb", 0.3)
        entries.append({"id": skin.stem, "title": skin.stem.replace("-", " ").replace("_", " ")})

    clip = face.frames(face.source(paths.jumper()))
    from PIL import Image

    for f, k in face.FACES.items():
        Image.fromarray(clip[k]).save(out / "eyes" / f"{f}.png")
    manifest = {"shells": entries, "faces": list(face.FACES), "eyes": eyes}
    (out / "closet.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copy(PAGE, out / "index.html")
    shutil.rmtree(work, ignore_errors=True)
    return manifest


def _pose(xmat, xpos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(xmat).reshape(3, 3)
    T[:3, 3] = xpos
    return T


def _eye_quads(R, p) -> list[list[list[float]]]:
    """World corners of the two eye screens: bottom-left, bottom-right, top-right, top-left as seen from the front."""
    y = np.array([0.0, 1.0, 0.0])
    quads = []
    for sy in (face.EYE_Y, -face.EYE_Y):
        c = face.CENTER + face.NORMAL * 0.0012 + y * sy
        h = face.HALF
        local = [c - y * h - face.UP * h, c + y * h - face.UP * h, c + y * h + face.UP * h, c - y * h + face.UP * h]
        quads.append([(R @ q + p).round(5).tolist() for q in local])
    return quads


def serve(root: Path, port: int = 8765, open_browser: bool = True) -> None:
    import functools
    import http.server
    import webbrowser

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"[closet] {url}  (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
