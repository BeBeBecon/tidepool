"""2D paths that toys lay things along. Each returns (x, y, heading_deg) samples.

Every path takes `start`, `count` and `spacing`: the first sample sits at
`start`, and consecutive samples are `spacing` apart along the curve.
"""

from __future__ import annotations

import math

Sample = tuple[float, float, float]
Point = tuple[float, float]


def _walk(points: list[Point], count: int, spacing: float) -> list[Sample]:
    """Place `count` samples `spacing` apart in arc length along a polyline."""
    if count <= 0:
        return []
    out: list[Sample] = []
    i = 0
    acc = 0.0
    seg = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(seg)
    for k in range(count):
        s = min(k * spacing, total)
        while i < len(seg) - 1 and acc + seg[i] < s:
            acc += seg[i]
            i += 1
        a, b = points[i], points[i + 1]
        t = (s - acc) / seg[i] if seg[i] else 0.0
        out.append((
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
        ))
    return out


def line(start: Point, count: int, spacing: float, heading: float = 0.0) -> list[Sample]:
    h = math.radians(heading)
    return [
        (start[0] + i * spacing * math.cos(h), start[1] + i * spacing * math.sin(h), heading)
        for i in range(count)
    ]


def arc(start: Point, count: int, spacing: float, radius: float = 0.5,
        heading: float = 0.0, left: bool = True) -> list[Sample]:
    """Leave `start` at `heading` and bend around a circle of `radius`."""
    if radius <= 0:
        raise ValueError("arc: radius must be positive")
    need = spacing * max(count - 1, 0)
    sign = 1.0 if left else -1.0
    h0 = math.radians(heading)
    cx = start[0] - sign * radius * math.sin(h0)
    cy = start[1] + sign * radius * math.cos(h0)
    n = max(count * 4, 2)
    sweep = need / radius
    pts = []
    for k in range(n):
        h = h0 + sign * sweep * k / (n - 1)
        pts.append((cx + sign * radius * math.sin(h), cy - sign * radius * math.cos(h)))
    return _walk(pts, count, spacing)


def spiral(start: Point, count: int, spacing: float, inner: float = 0.2,
           growth: float = 0.1, left: bool = True) -> list[Sample]:
    """Wind outward from `start`. `inner` is the first radius, `growth` the gain per turn.

    The spiral is as long as the dominoes need, so more of them means more turns.
    """
    if inner <= 0 or growth <= 0:
        raise ValueError("spiral: inner and growth must be positive")
    need = spacing * max(count - 1, 0)
    sign = 1.0 if left else -1.0
    pts: list[Point] = []
    length = 0.0
    a = 0.0
    da = math.radians(3.0)
    prev = None
    while True:
        r = inner + growth * a / (2 * math.pi)
        p = (r * math.cos(sign * a), r * math.sin(sign * a))
        if prev is not None:
            length += math.dist(prev, p)
        pts.append(p)
        prev = p
        if length >= need and len(pts) > 1:
            break
        a += da
    ox, oy = start[0] - pts[0][0], start[1] - pts[0][1]
    pts = [(x + ox, y + oy) for x, y in pts]
    return _walk(pts, count, spacing)


def zigzag(start: Point, count: int, spacing: float, leg: float = 0.5,
           angle: float = 60.0, heading: float = 0.0) -> list[Sample]:
    """Straight legs of length `leg`, alternating +-angle/2 around `heading`."""
    if leg <= 0:
        raise ValueError("zigzag: leg must be positive")
    need = spacing * max(count - 1, 0)
    legs = max(math.ceil(need / leg), 1)
    pts = [start]
    x, y = start
    for i in range(legs):
        h = math.radians(heading + (angle / 2 if i % 2 == 0 else -angle / 2))
        x += leg * math.cos(h)
        y += leg * math.sin(h)
        pts.append((x, y))
    return _walk(pts, count, spacing)


SHAPES = {"line": line, "arc": arc, "spiral": spiral, "zigzag": zigzag}
