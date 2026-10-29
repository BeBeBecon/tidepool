"""Turn a Spec into an mjscene JSON document (the input of `shellflow export-map`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from . import room, toys
from .spec import Spec, SpecError

LOOKS: dict[str, dict[str, Any]] = {
    "studio": {
        "sky": {"preset": "studio"},
        "lighting": {"preset": "studio_3point"},
        "ground": {"material": "checker_light", "friction": "wood", "size": [8, 8]},
    },
    "wood": {
        "sky": {"preset": "indoor"},
        "lighting": {"preset": "indoor_office"},
        "ground": {"material": "wood", "friction": "wood", "size": [8, 8]},
    },
    "night": {
        "sky": {"preset": "night"},
        "lighting": {"preset": "night_moon"},
        "ground": {"material": "steel_dark", "friction": "concrete", "size": [8, 8]},
    },
    "beach": {
        "sky": {"preset": "clear_noon"},
        "lighting": {"preset": "noon"},
        "ground": {"material": "sand", "friction": "sand", "size": [12, 12]},
    },
}


def build(spec: Spec, cache_dir: Path | None = None, with_terrain: bool = False):
    """The mjscene document. With a photo, `cache_dir` is where its depth map is kept.

    `with_terrain` also returns the `room.Terrain` (or None) so the caller can write
    its mesh files next to the JSON.
    """
    if spec.look not in LOOKS:
        raise SpecError(f"look {spec.look!r}: choose from {', '.join(LOOKS)}")
    objects: list[dict[str, Any]] = []
    for i, call in enumerate(spec.toys):
        try:
            objects.extend(toys.expand(call.kind, i, call.params))
        except (KeyError, TypeError, ValueError) as e:
            raise SpecError(f"toys[{i}]: {e}") from None

    terrain = None
    spawn_at = spec.spawn_at
    yaw = spec.spawn_yaw
    if spec.photo is not None:
        terrain = room.terrain(spec.photo, cache_dir or spec.photo.file.parent / ".tidepool")
        if spawn_at is None:
            spawn_at = terrain.spawn
            if terrain.face is not None:   # an auto spawn looks at the first movable
                yaw = math.degrees(math.atan2(terrain.face[1] - spawn_at[1], terrain.face[0] - spawn_at[0]))
    if spawn_at is None:
        spawn_at = (0.0, 0.0)

    doc: dict[str, Any] = {
        "name": spec.name,
        "title": spec.name.replace("-", " ").replace("_", " ").title(),
        "description": spec.description or f"{spec.name}, made with tidepool.",
        "tags": ["tidepool", *sorted({c.kind for c in spec.toys})],
        **LOOKS[spec.look],
        "defaults": {"friction": "wood", "contact": "default"},
        "spawn": {
            "position": [spawn_at[0], spawn_at[1], 0.0],
            "yaw": round(yaw, 2),
            "clearance": 0.3 if terrain else 0.4,
        },
        "objects": objects,
    }
    if terrain is not None:
        room.apply(doc, spec.photo, terrain)
    return (doc, terrain) if with_terrain else doc


def stats(doc: dict[str, Any]) -> dict[str, int]:
    def count(objs: list[dict[str, Any]]) -> int:
        n = 0
        for o in objs:
            n += 1 + count(o.get("children") or [])
        return n

    return {"objects": count(doc.get("objects", []))}
