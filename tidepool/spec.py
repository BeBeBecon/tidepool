"""The YAML a scene is written in.

    name: spiral-dominoes
    description: Forty dominoes in a spiral. Walk into the first one.
    look: studio
    spawn: {at: [-0.6, 0], yaw: 0}
    toys:
      - domino_run: {path: spiral, turns: 2.5, count: 40, start: [0.3, 0]}

Each entry under `toys` is a single-key mapping: the toy's name, then its
parameters. Toys are expanded by `tidepool.toys`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .room import Photo

NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")


class SpecError(ValueError):
    pass


@dataclass
class ToyCall:
    kind: str
    params: dict[str, Any]


@dataclass
class Spec:
    name: str
    description: str = ""
    look: str = "studio"
    spawn_at: tuple[float, float] | None = (0.0, 0.0)   # None: let the photo pick a flat spot
    spawn_yaw: float = 0.0
    toys: list[ToyCall] = field(default_factory=list)
    photo: Photo | None = None
    source: Path | None = None


def load(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SpecError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: top level must be a mapping")
    spec = parse(raw, default_name=path.stem, base=path.parent)
    spec.source = path
    return spec


def parse(raw: dict[str, Any], default_name: str = "scene", base: Path | None = None) -> Spec:
    name = str(raw.get("name") or default_name)
    if not NAME.match(name):
        raise SpecError(f"name {name!r}: use letters, digits, '_', '.', '-' and start with a letter")

    photo = _photo(raw.get("photo"), base or Path.cwd())

    spawn = raw.get("spawn")
    if spawn is None:
        spawn = {} if photo is None else {"at": "auto"}
    if not isinstance(spawn, dict):
        raise SpecError("spawn must be a mapping like {at: [x, y], yaw: 0}")
    at = spawn.get("at", [0.0, 0.0])
    if at == "auto":
        if photo is None:
            raise SpecError("spawn.at: 'auto' needs a photo")
        at = None
    elif not (isinstance(at, (list, tuple)) and len(at) == 2):
        raise SpecError("spawn.at must be [x, y] (or 'auto' with a photo)")

    toys_raw = raw.get("toys") or []
    if not isinstance(toys_raw, list):
        raise SpecError("toys must be a list")
    toys = []
    for i, entry in enumerate(toys_raw):
        if isinstance(entry, str):
            toys.append(ToyCall(entry, {}))
            continue
        if not (isinstance(entry, dict) and len(entry) == 1):
            raise SpecError(f"toys[{i}]: write one toy per entry, e.g. '- domino_run: {{count: 20}}'")
        kind, params = next(iter(entry.items()))
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise SpecError(f"toys[{i}] ({kind}): parameters must be a mapping")
        toys.append(ToyCall(str(kind), dict(params)))

    known = {"name", "description", "look", "spawn", "toys", "photo"}
    extra = set(raw) - known
    if extra:
        raise SpecError(f"unknown top-level keys: {', '.join(sorted(extra))}")

    return Spec(
        name=name,
        description=str(raw.get("description") or ""),
        look=str(raw.get("look") or "studio"),
        spawn_at=None if at is None else (float(at[0]), float(at[1])),
        spawn_yaw=float(spawn.get("yaw", 0.0)),
        toys=toys,
        photo=photo,
    )


def _photo(raw: Any, base: Path) -> Photo | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"file": raw}
    if not isinstance(raw, dict) or "file" not in raw:
        raise SpecError("photo must be a path or a mapping with 'file'")
    known = {"file", "width", "relief", "cells", "style", "gamma", "quant", "clear", "flatten"}
    extra = set(raw) - known
    if extra:
        raise SpecError(f"photo: unknown keys {', '.join(sorted(extra))}; takes {', '.join(sorted(known))}")
    file = (base / str(raw["file"])).resolve()
    if not file.exists():
        raise SpecError(f"photo: {file} not found")
    try:
        return Photo(
            file=file,
            width=float(raw.get("width", Photo.width)),
            relief=float(raw.get("relief", Photo.relief)),
            cells=int(raw.get("cells", Photo.cells)),
            style=str(raw.get("style", Photo.style)),
            gamma=float(raw.get("gamma", Photo.gamma)),
            quant=float(raw.get("quant", Photo.quant)),
            clear=float(raw.get("clear", Photo.clear)),
            flatten=bool(raw.get("flatten", Photo.flatten)),
        )
    except (TypeError, ValueError) as e:
        raise SpecError(f"photo: {e}") from None
