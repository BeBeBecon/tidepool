from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from . import build as build_mod
from . import paths, play, scene, spec as spec_mod, toys

TEMPLATES = Path(__file__).resolve().parents[1] / "toys"


def cmd_toys(args) -> int:
    for name in toys.names():
        print(f"{name:14} {toys.describe(name)}")
    return 0


def cmd_new(args) -> int:
    src = TEMPLATES / f"{args.template}.yaml"
    if not src.exists():
        have = ", ".join(p.stem for p in sorted(TEMPLATES.glob("*.yaml")))
        print(f"no template {args.template!r}; have: {have}", file=sys.stderr)
        return 1
    dest = Path(args.out or f"{args.template}.yaml")
    if dest.exists() and not args.force:
        print(f"{dest} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    shutil.copy(src, dest)
    print(f"wrote {dest}")
    return 0


def cmd_room(args) -> int:
    photo = Path(args.photo)
    if not photo.exists():
        print(f"no such photo: {photo}", file=sys.stderr)
        return 1
    name = args.name or re.sub(r"[^A-Za-z0-9_.-]+", "-", photo.stem).strip("-").lower() or "room"
    dest = Path(args.out or photo.with_name(f"{name}.yaml"))
    if dest.exists() and not args.force:
        print(f"{dest} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    rel = os.path.relpath(photo.resolve(), dest.resolve().parent)
    opts = {k: getattr(args, k) for k in ("width", "relief", "cells") if getattr(args, k) is not None}
    photo_block = f"photo: {rel}" if not opts else "photo:\n  file: " + rel + "".join(
        f"\n  {k}: {v}" for k, v in opts.items())
    dest.write_text(
        f"name: {name}\n"
        f"description: A photo turned into ground the crab can walk on.\n"
        f"look: wood\n"
        f"{photo_block}\n"
        f"toys: []\n",
        encoding="utf-8",
    )
    print(f"wrote {dest}")
    if args.no_build:
        return 0
    args.spec = str(dest)
    args.no_verify = False
    args.json_only = False
    return cmd_build(args)


def _load(path: str) -> spec_mod.Spec:
    try:
        return spec_mod.load(path)
    except (spec_mod.SpecError, OSError) as e:
        raise SystemExit(f"error: {e}") from None


def cmd_build(args) -> int:
    s = _load(args.spec)
    if args.json_only:
        doc = scene.build(s)
        print(json.dumps(doc, indent=2))
        return 0
    try:
        built = build_mod.build(s, verify=not args.no_verify)
    except (spec_mod.SpecError, build_mod.BuildError, paths.MissingRepo) as e:
        raise SystemExit(f"error: {e}") from None
    size = built.map.stat().st_size / 1e6
    n = scene.stats(scene.build(s))["objects"]
    print(f"{built.map}  ({size:.1f} MB, {n} objects)")
    v = built.report.get("verify", {})
    for w in v.get("warnings", []) or []:
        print(f"warning: {w}")
    return 0


def cmd_play(args) -> int:
    s = _load(args.spec)
    target = paths.out_dir() / s.name / f"{s.name}.map"
    if args.rebuild or not target.exists():
        try:
            built = build_mod.build(s, verify=False)
        except (spec_mod.SpecError, build_mod.BuildError, paths.MissingRepo) as e:
            raise SystemExit(f"error: {e}") from None
        target = built.map
    try:
        return play.run(target, headless=args.headless, steps=args.steps, ui=args.ui,
                        speed=args.speed, timeout=args.timeout)
    except paths.MissingRepo as e:
        raise SystemExit(f"error: {e}") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tidepool", description="A toy box for the Jumper crab robot.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("toys", help="list the toys you can put in a scene").set_defaults(fn=cmd_toys)

    n = sub.add_parser("new", help="write a starter YAML from a template")
    n.add_argument("template", nargs="?", default="spiral")
    n.add_argument("-o", "--out")
    n.add_argument("--force", action="store_true")
    n.set_defaults(fn=cmd_new)

    r = sub.add_parser("room", help="turn a photo into terrain: writes a YAML and builds it")
    r.add_argument("photo")
    r.add_argument("-o", "--out", help="where to write the YAML (default: next to the photo)")
    r.add_argument("--name")
    r.add_argument("--width", type=float, help="metres across the photo (default 3.0)")
    r.add_argument("--relief", type=float, help="tallest column in metres (default 0.06)")
    r.add_argument("--cells", type=int, help="columns across (default 20)")
    r.add_argument("--force", action="store_true")
    r.add_argument("--no-build", action="store_true")
    r.set_defaults(fn=cmd_room)

    b = sub.add_parser("build", help="compile a YAML into a verified .map")
    b.add_argument("spec")
    b.add_argument("--no-verify", action="store_true", help="skip verify-package --mujoco")
    b.add_argument("--json-only", action="store_true", help="print the mjscene JSON and stop")
    b.set_defaults(fn=cmd_build)

    pl = sub.add_parser("play", help="build (if needed) and open the scene with the crab")
    pl.add_argument("spec")
    pl.add_argument("--rebuild", action="store_true")
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--steps", type=int)
    pl.add_argument("--speed", type=float)
    pl.add_argument("--ui", action="store_true", help="keep the viewer's side panels")
    pl.add_argument("--timeout", type=float)
    pl.set_defaults(fn=cmd_play)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
