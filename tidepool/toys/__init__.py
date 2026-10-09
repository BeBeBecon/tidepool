"""Toys: small functions that turn a few parameters into mjscene objects.

A toy takes a `Ctx` (for unique names) and keyword parameters, and returns a
list of mjscene object dicts. Register new ones with `@toy`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

Toy = Callable[..., list[dict[str, Any]]]

_REGISTRY: dict[str, Toy] = {}


@dataclass
class Ctx:
    prefix: str
    counters: dict[str, int] = field(default_factory=dict)

    def name(self, part: str = "") -> str:
        key = f"{self.prefix}_{part}" if part else self.prefix
        n = self.counters.get(key, 0)
        self.counters[key] = n + 1
        return f"{key}_{n}"


def toy(fn: Toy) -> Toy:
    _REGISTRY[fn.__name__] = fn
    return fn


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(kind: str) -> Toy:
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(f"unknown toy {kind!r}; available: {', '.join(names())}") from None


def describe(kind: str) -> str:
    fn = get(kind)
    doc = inspect.getdoc(fn) or ""
    return doc.split("\n")[0]


def expand(kind: str, index: int, params: dict[str, Any]) -> list[dict[str, Any]]:
    fn = get(kind)
    sig = inspect.signature(fn)
    allowed = {p for p in sig.parameters if p != "ctx"}
    open_ended = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    unknown = set(params) - allowed
    if unknown and not open_ended:
        raise TypeError(
            f"{kind}: unknown parameter(s) {', '.join(sorted(unknown))}; "
            f"takes {', '.join(sorted(allowed))}"
        )
    return fn(Ctx(f"{kind}_{index}"), **{k: v for k, v in params.items() if k != "shape"})


from . import domino_run  # noqa: E402,F401
