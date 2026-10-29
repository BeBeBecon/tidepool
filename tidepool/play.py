"""Open a `.map` in jumper's viewer with the shipped app driving the crab."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import paths


def command(map_path: Path, *, app: Path | None = None, headless: bool = False,
            steps: int | None = None, ui: bool = False, speed: float | None = None) -> list[str]:
    app = app or paths.default_app()
    exe = paths.python() if headless else paths.mjpython()
    cmd = [exe, "scripts/play.py", "--app", str(app), "--scene", str(map_path),
           "--backend", "native", "--device", "cpu"]
    if headless:
        cmd.append("--headless")
    elif not ui:
        cmd.append("--no-viewer-ui")
    if steps is not None:
        cmd += ["--steps", str(steps)]
    if speed is not None:
        cmd += ["--speed", str(speed)]
    return cmd


def run(map_path: Path, *, timeout: float | None = None, **kw) -> int:
    cmd = command(map_path, **kw)
    env = dict(os.environ)
    env.setdefault("MJRL_NUM_ENVS", "1")
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=paths.jumper(), env=env, timeout=timeout)
    return proc.returncode
