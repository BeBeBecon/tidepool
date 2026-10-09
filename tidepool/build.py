"""Compile a Spec to a verified `.map` through jumper-design's shellflow."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import paths, scene
from .spec import Spec


class BuildError(RuntimeError):
    pass


@dataclass
class Built:
    spec_json: Path
    map: Path
    report: dict


def _shellflow(*args: str) -> dict:
    design = paths.jumper_design()
    cmd = [paths.python(), str(design / "scripts" / "shellflow.py"), *args]
    proc = subprocess.run(cmd, cwd=design, capture_output=True, text=True, timeout=600)
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        raise BuildError(
            f"shellflow {args[0]} failed (exit {proc.returncode})\n{proc.stdout}\n{proc.stderr}"
        ) from None
    if not result.get("ok"):
        detail = result.get("error") or json.dumps(result, indent=2)
        raise BuildError(f"shellflow {args[0]}: {detail}")
    return result


def build(spec: Spec, out: Path | None = None, *, verify: bool = True, preview: Path | None = None) -> Built:
    out = out or paths.out_dir() / spec.name
    out.mkdir(parents=True, exist_ok=True)
    doc = scene.build(spec)
    spec_json = out / f"{spec.name}.scene.json"
    spec_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    target = out / f"{spec.name}.map"
    if target.exists():
        target.unlink()
    args = ["export-map", str(spec_json), "--output", str(target),
            "--profile", str(paths.profile())]
    if preview is not None:
        args += ["--preview", str(preview)]
    report = _shellflow(*args)
    if verify:
        report["verify"] = _shellflow("verify-package", str(target),
                                      "--profile", str(paths.profile()), "--mujoco")
    return Built(spec_json=spec_json, map=target, report=report)


def clean(spec: Spec) -> None:
    shutil.rmtree(paths.out_dir() / spec.name, ignore_errors=True)
