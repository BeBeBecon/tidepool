"""Film a scene: the `shoot` block of a YAML, run through `tidepool.director`.

    shoot:
      camera: {preset: chase, distance: 1.2}     # chase / side / low / top
      size: [1080, 1920]                         # portrait for shorts
      speed: 1.2                                 # playback speed of the file
      script:
        - {at: 0.0, walk: forward}
        - {at: 2.5, jump: true}
        - {at: 4.0, turn: left}
        - {at: 5.0, stop: true}
      duration: 8
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from . import paths

PRESETS = ("chase", "side", "low", "top")
ACTIONS = {"at", "walk", "turn", "speed", "stop", "jump", "dance", "gesture", "claw"}


class ShootError(ValueError):
    pass


def check(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ShootError("shoot must be a mapping")
    extra = set(raw) - {"camera", "size", "speed", "fps", "script", "duration", "start", "shadows", "watch"}
    if extra:
        raise ShootError(f"shoot: unknown keys {', '.join(sorted(extra))}")
    cam = raw.get("camera") or {}
    if isinstance(cam, str):
        cam = {"preset": cam}
    if cam.get("preset", "chase") not in PRESETS:
        raise ShootError(f"shoot.camera.preset: choose from {', '.join(PRESETS)}")
    size = raw.get("size", [1280, 720])
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(int(v) > 0 for v in size)):
        raise ShootError("shoot.size must be [width, height]")
    script = raw.get("script") or []
    if not isinstance(script, list):
        raise ShootError("shoot.script must be a list")
    for i, e in enumerate(script):
        if not isinstance(e, dict):
            raise ShootError(f"shoot.script[{i}]: write a mapping like {{at: 1.0, walk: forward}}")
        bad = set(e) - ACTIONS
        if bad:
            raise ShootError(f"shoot.script[{i}]: unknown keys {', '.join(sorted(bad))}")
    out = dict(raw)
    out["camera"] = cam
    out["size"] = [int(size[0]), int(size[1])]
    out["script"] = script
    return out


def film(map_path: Path, shoot: dict[str, Any], out: Path, *, watch: list[str] | None = None,
         app: Path | None = None, skin: Path | None = None) -> int:
    app = app or paths.default_app()
    bundle = _bundle_dir(app)
    cfg = {**shoot, "jumper": str(paths.jumper()), "app": str(app), "bundle": str(bundle),
           "map": str(map_path), "out": str(out), "watch": shoot.get("watch") or watch or [],
           "skin": str(skin) if skin else None}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    env = dict(os.environ)
    env.setdefault("MJRL_NUM_ENVS", "1")
    env["PYTHONPATH"] = str(paths.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [paths.python(), "-m", "tidepool.director", cfg_path]
    try:
        return subprocess.run(cmd, cwd=paths.jumper(), env=env).returncode
    finally:
        os.unlink(cfg_path)


def _bundle_dir(app: Path) -> Path:
    """The unzipped app, kept in the cache so controller.toml can be read."""
    if app.is_dir():
        return app
    dest = Path(os.environ.get("TIDEPOOL_CACHE", Path.home() / ".cache" / "tidepool")) / "app" / app.stem
    if not (dest / "controller.toml").exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(app) as z:
            z.extractall(dest)
    return dest
