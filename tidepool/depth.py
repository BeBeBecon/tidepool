"""Monocular depth for a photo, via Depth Anything V2 (small). CPU is fine: under a second."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

_pipe = None


class DepthUnavailable(RuntimeError):
    pass


def _pipeline():
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline
        except ImportError:
            raise DepthUnavailable(
                "depth estimation needs torch and transformers: pip install -e '.[room]'"
            ) from None
        _pipe = pipeline("depth-estimation", model=MODEL, device="cpu")
    return _pipe


def estimate(img: Image.Image) -> np.ndarray:
    """Relative depth, same size as `img`, larger = closer to the camera."""
    out = _pipeline()(img.convert("RGB"))
    d = np.asarray(out["predicted_depth"], dtype=np.float32)
    if d.shape != (img.height, img.width):
        d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR), dtype=np.float32)
    return d


def cached(photo: Path, cache_dir: Path) -> np.ndarray:
    """`estimate` with the result kept next to the build, keyed on the photo bytes."""
    data = photo.read_bytes()
    key = hashlib.sha1(data).hexdigest()[:12]
    cache = cache_dir / f"depth-{key}.npy"
    if cache.exists():
        return np.load(cache)
    d = estimate(Image.open(photo))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, d)
    return d
