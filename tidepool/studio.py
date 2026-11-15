"""The studio: pose the crab in the browser, play the takes on pads, pack an .app.

`export` writes one .glb per link, since the page has to move them. `serve` is
closet's localhost plus a few POSTs, because MuJoCo does the kinematics here and
the page only draws.
"""

from __future__ import annotations

import functools
import http.server
import json
import shutil
import threading
import webbrowser
import zipfile
from pathlib import Path

import numpy as np

from . import closet, face, motion, paths

PAGE = Path(__file__).with_name("studio.html")


def _local_mesh(m, g):
    import mujoco
    import trimesh

    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return trimesh.Trimesh(v @ R.reshape(3, 3).T + m.geom_pos[g], f.copy(), process=False)


def export(skins: list[Path], out: Path, keep: float = 0.5) -> dict:
    import tempfile

    import mujoco

    rig = motion.Rig()
    m = rig.m
    if out.exists():
        shutil.rmtree(out)
    (out / "links").mkdir(parents=True)
    (out / "shells").mkdir()
    (out / "eyes").mkdir()
    links = []
    for b in range(m.nbody):
        groups: dict[tuple, list] = {}
        for g in np.flatnonzero(m.geom_bodyid == b):
            if closet._is_visual(m, g):
                groups.setdefault(tuple(np.round(m.geom_rgba[g], 3).tolist()), []).append(_local_mesh(m, g))
        if groups:
            closet._glb(groups, out / "links" / f"{b}.glb", keep)
            links.append(b)

    shells = []
    work = Path(tempfile.mkdtemp(prefix="tidepool-studio-"))
    for skin in skins:
        dest = work / skin.stem
        with zipfile.ZipFile(skin) as z:
            z.extractall(dest)
        sm, sd = closet._posed(dest / "assembled" / "jumper" / "scene.xml")
        base = sm.body("base_link").id
        inv = np.linalg.inv(closet._pose(sd.xmat[base], sd.xpos[base]))
        groups = {}
        for g in range(sm.ngeom):
            name = mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if closet._is_visual(sm, g) and name.startswith("shellflow_upper_"):
                mesh = closet._geom_mesh(sm, sd, g)
                mesh.apply_transform(inv)
                groups.setdefault(tuple(np.round(sm.geom_rgba[g], 3).tolist()), []).append(mesh)
        closet._glb(groups, out / "shells" / f"{skin.stem}.glb", 0.6)
        shells.append({"id": skin.stem, "title": skin.stem.replace("-", " ").replace("_", " ")})
    shutil.rmtree(work, ignore_errors=True)

    from PIL import Image

    Image.fromarray(face.frames(face.source(paths.jumper()))[face.FACES["normal"]]).save(out / "eyes" / "normal.png")
    manifest = {"links": links, "base": m.body("base_link").id, "stock": m.body("upper_shell_link").id,
                "shells": shells, "eyes": closet._eye_quads(np.eye(3), np.zeros(3)),
                "skins": sorted(str(s) for s in skins)}
    (out / "studio.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


class Studio:
    def __init__(self, root: Path):
        self.root = root
        self.rig = motion.Rig()
        self.lock = threading.Lock()
        self.saved = root / "takes.json"

    def _pose(self, q, pitch, roll, back=0.0) -> dict:
        return {"q": np.round(q, 5).tolist(), "bodies": self.rig.bodies(q, pitch, roll, back),
                "tips": self.rig.tips(q, pitch, roll, back), "pitch": pitch, "roll": roll, "back": round(back, 4)}

    def info(self, _) -> dict:
        keys = {"presets": motion.PRESETS, "keys": motion.KEYS, "home": self._pose(self.rig.home, 0.0, 0.0),
                "names": self.rig.names}
        keys["pads"] = json.loads(self.saved.read_text("utf-8")) if self.saved.exists() else None
        return keys

    def solve(self, p: dict) -> dict:
        q, back = self.rig.balance(p["q"], p["pitch"], p["roll"], p.get("lifted", ()), p.get("reach"))
        return self._pose(q, p["pitch"], p["roll"], back)

    def preset(self, p: dict) -> dict:
        return {"keys": motion.preset(p["name"], self.rig)}

    def play(self, p: dict) -> dict:
        q, lean = motion.timeline(p["keys"], self.rig)
        frames = [self.rig.bodies(q[i], *lean[i]) for i in range(0, len(q), 2)]
        return {"frames": frames, "fps": motion.HZ / 2, "speed": round(motion.speed(q), 1), "fast": motion.FAST}

    def save(self, p: dict) -> dict:
        self.saved.write_text(json.dumps(p["pads"]), "utf-8")
        return {}

    def export(self, p: dict) -> dict:
        takes = {motion.SLOTS[i]: motion.timeline(pad["keys"], self.rig)[0]
                 for i, pad in enumerate(p["pads"]) if pad and len(pad["keys"]) > 1}
        if not takes:
            return {"error": "nothing on the pads yet"}
        app = motion.pack(self.root / "build", takes)
        board = json.loads((app.with_suffix("") / "bundle.json").read_text("utf-8"))
        stub = [m for m in takes if "rknn" not in board["modes"][m]["models"]]
        return {"app": str(app), "modes": list(takes), "stub": stub}


def serve(root: Path, port: int = 8766, open_browser: bool = True) -> None:
    studio = Studio(root)
    routes = {"/api/" + n: getattr(studio, n) for n in ("info", "solve", "preset", "play", "save", "export")}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            size = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            with studio.lock:
                try:
                    reply = routes[self.path](body)
                except SystemExit as e:
                    reply = {"error": str(e)}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    handler = functools.partial(Handler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"[studio] {url}  (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
