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
    here = dest.resolve().parent
    rel = os.path.relpath(photo.resolve(), here)
    opts = {k: getattr(args, k) for k in ("width", "relief", "cells", "style", "inpaint")
            if getattr(args, k) is not None}
    for m in args.movable or []:
        if not Path(m).exists():
            print(f"no such mask: {m}", file=sys.stderr)
            return 1
    masks = [os.path.relpath(Path(m).resolve(), here) for m in args.movable or []]
    if not opts and not masks:
        photo_block = f"photo: {rel}"
    else:
        photo_block = "photo:\n  file: " + rel + "".join(f"\n  {k}: {v}" for k, v in opts.items())
        if masks:
            photo_block += "\n  movable:" + "".join(f"\n    - mask: {m}" for m in masks)
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


def cmd_shoot(args) -> int:
    from . import shoot as shoot_mod

    s = _load(args.spec)
    target = paths.out_dir() / s.name / f"{s.name}.map"
    if args.rebuild or not target.exists():
        try:
            target = build_mod.build(s, verify=False).map
        except (spec_mod.SpecError, build_mod.BuildError, paths.MissingRepo) as e:
            raise SystemExit(f"error: {e}") from None
    out = Path(args.out or paths.out_dir() / s.name / f"{s.name}.mp4")
    watch = [m.name for m in s.photo.movable] if s.photo else []
    try:
        return shoot_mod.film(target, s.shoot, out, watch=watch, skin=s.skin)
    except paths.MissingRepo as e:
        raise SystemExit(f"error: {e}") from None


def cmd_shellify(args) -> int:
    from . import shell

    photo = Path(args.photo)
    if not photo.exists():
        print(f"no such photo: {photo}", file=sys.stderr)
        return 1
    name = args.name or re.sub(r"[^A-Za-z0-9_.-]+", "-", photo.stem).strip("-").lower() or "shell"
    colors = [c.strip() for c in args.colors.split(",")] if args.colors else None
    try:
        skin = shell.make(photo, name, colors=colors, length=args.length)
    except (shell.ShellError, paths.MissingRepo) as e:
        raise SystemExit(f"error: {e}") from None
    png = skin.with_suffix(".png")
    shell.preview(skin, png)
    print(f"wrote {skin}" + (f" and {png}" if png.exists() else ""))
    print(f"wear it in a scene with: skin: {os.path.relpath(skin)}")
    return 0


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
    r.add_argument("--relief", type=float, help="tallest point in metres (default 0.12)")
    r.add_argument("--cells", type=int, help="collision cells across (default 20)")
    r.add_argument("--style", choices=["smooth", "blocks"], help="smooth surface (default) or box columns")
    r.add_argument("--movable", action="append", metavar="MASK.png",
                   help="a painted mask of something in the photo the crab can push (repeatable)")
    r.add_argument("--inpaint", choices=["lama", "fast", "none"],
                   help="how to fill the photo where a movable was: lama (default), fast, none")
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

    sh = sub.add_parser("shoot", help="film the scene headless, driven by the YAML's shoot script")
    sh.add_argument("spec")
    sh.add_argument("-o", "--out", help="mp4 to write (default: out/<name>/<name>.mp4)")
    sh.add_argument("--rebuild", action="store_true")
    sh.set_defaults(fn=cmd_shoot)

    sf = sub.add_parser("shellify", help="turn a photo of a cup-like thing into a shell the crab wears")
    sf.add_argument("photo")
    sf.add_argument("--name")
    sf.add_argument("--colors", metavar="BODY[,ACCENT]",
                    help="override the colors picked from the photo, e.g. '#E8A9A0,#F5F2EC'")
    sf.add_argument("--length", type=float, default=150.0, help="front-to-back size in mm (default 150)")
    sf.set_defaults(fn=cmd_shellify)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
