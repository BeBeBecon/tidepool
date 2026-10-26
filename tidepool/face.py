"""The crab's eyes, which the simulator leaves as a black plate.

Two screens are glued onto the display window and fed frames from the face
animation jumper ships with its dance task (Apache-2.0, KingKong Robotics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FPS = 24
FACES = {"normal": 1200, "wide": 192, "happy": 528, "surprised": 792, "sparkle": 1104, "blink": 0}
SIZE = 132
CROP = 30          # the shapes sit in the middle of the frame; zoom in on them

# display window in base_link: front face leans back about 33 degrees
CENTER = np.array([0.0836, 0.0, 0.048])
NORMAL = np.array([0.836, 0.0, 0.549])
UP = np.array([-0.549, 0.0, 0.836])
EYE_Y = 0.026
HALF = 0.018


def source(jumper: Path) -> Path:
    return jumper / "tasks" / "jumper" / "dance" / "media" / "demo.mp4"


def frames(video: Path) -> np.ndarray:
    import imageio.v3 as iio

    return np.stack([np.ascontiguousarray(f[CROP:SIZE - CROP, CROP:SIZE - CROP, :3]) for f in iio.imiter(video)])


def add(spec, first: np.ndarray, work: Path) -> None:
    """Add the two eye screens to a jumper MjSpec, showing `first` until told otherwise."""
    import mujoco
    from PIL import Image

    png = work / "tp_eye.png"
    Image.fromarray(first).save(png)
    spec.add_texture(name="tp_eye", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
    mat = spec.add_material(name="tp_eye", emission=1.0, specular=0.0, rgba=[1, 1, 1, 1])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tp_eye"
    body = spec.body("base_link")
    y = np.array([0.0, 1.0, 0.0])
    for side, sy in (("l", EYE_Y), ("r", -EYE_Y)):
        c = CENTER + NORMAL * 0.0012 + y * sy
        corners = [c - y * HALF - UP * HALF, c + y * HALF - UP * HALF, c + y * HALF + UP * HALF, c - y * HALF + UP * HALF]
        spec.add_mesh(name=f"tp_eye_{side}", uservert=np.ravel(corners).tolist(),
                      userface=[0, 1, 2, 0, 2, 3],
                      usertexcoord=[1, 1, 0, 1, 0, 0, 1, 0],
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        body.add_geom(name=f"tp_eye_{side}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"tp_eye_{side}",
                      material="tp_eye", density=0, contype=0, conaffinity=0, group=2)


class Player:
    """Which frame the eyes show at time t: a held face with blinks, or the demo loop."""

    def __init__(self, clip: np.ndarray, face: str = "normal", blink_every: float = 3.2):
        self.clip = clip
        self.face, self.since = face, 0.0
        self.blink_every = blink_every

    def set(self, face: str, t: float) -> None:
        if face != "demo" and face not in FACES:
            raise ValueError(f"face {face!r}: choose from demo, {', '.join(FACES)}")
        self.face, self.since = face, t

    def frame(self, t: float) -> int:
        if self.face == "demo":
            return int((t - self.since) * FPS) % len(self.clip)
        if self.face in ("normal", "wide", "surprised") and (t % self.blink_every) < 0.12:
            return FACES["blink"]
        return FACES[self.face]
