"""Turn a Spec into an mjscene JSON document (the input of `shellflow export-map`)."""

from __future__ import annotations

from typing import Any

from . import toys
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


def build(spec: Spec) -> dict[str, Any]:
    if spec.look not in LOOKS:
        raise SpecError(f"look {spec.look!r}: choose from {', '.join(LOOKS)}")
    objects: list[dict[str, Any]] = []
    for i, call in enumerate(spec.toys):
        try:
            objects.extend(toys.expand(call.kind, i, call.params))
        except (KeyError, TypeError, ValueError) as e:
            raise SpecError(f"toys[{i}]: {e}") from None

    doc: dict[str, Any] = {
        "name": spec.name,
        "title": spec.name.replace("-", " ").replace("_", " ").title(),
        "description": spec.description or f"{spec.name}, made with tidepool.",
        "tags": ["tidepool", *sorted({c.kind for c in spec.toys})],
        **LOOKS[spec.look],
        "defaults": {"friction": "wood", "contact": "default"},
        "spawn": {
            "position": [spec.spawn_at[0], spec.spawn_at[1], 0.0],
            "yaw": spec.spawn_yaw,
            "clearance": 0.4,
        },
        "objects": objects,
    }
    return doc


def stats(doc: dict[str, Any]) -> dict[str, int]:
    def count(objs: list[dict[str, Any]]) -> int:
        n = 0
        for o in objs:
            n += 1 + count(o.get("children") or [])
        return n

    return {"objects": count(doc.get("objects", []))}
