"""Where the official repositories live.

tidepool does not vendor jumper or jumper-design. It expects both to be cloned
next to this repository, or pointed at with TIDEPOOL_JUMPER and
TIDEPOOL_JUMPER_DESIGN.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MissingRepo(RuntimeError):
    pass


def _find(env: str, folder: str, marker: str) -> Path:
    hint = os.environ.get(env)
    candidates = [Path(hint).expanduser()] if hint else []
    candidates.append(ROOT.parent / folder)
    for c in candidates:
        if (c / marker).exists():
            return c.resolve()
    raise MissingRepo(
        f"cannot find {folder}: clone https://github.com/KingKongRobotics/{folder} "
        f"next to tidepool or set {env}"
    )


def jumper() -> Path:
    return _find("TIDEPOOL_JUMPER", "jumper", "scripts/play.py")


def jumper_design() -> Path:
    return _find("TIDEPOOL_JUMPER_DESIGN", "jumper-design", "scripts/shellflow.py")


def default_app() -> Path:
    hint = os.environ.get("TIDEPOOL_APP")
    if hint:
        return Path(hint).expanduser().resolve()
    return jumper() / "out" / "bundle_example" / "jumper.app"


def profile() -> Path:
    return jumper_design() / "robots" / "jumper" / "profile.json"


def python() -> str:
    return sys.executable


def mjpython() -> str:
    exe = Path(sys.executable).with_name("mjpython")
    if exe.exists():
        return str(exe)
    return "mjpython"


def out_dir() -> Path:
    return Path(os.environ.get("TIDEPOOL_OUT", ROOT / "out")).resolve()
