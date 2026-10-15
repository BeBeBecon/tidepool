"""Fill the hole a movable leaves in the photo.

`lama` runs the LaMa model (Apache-2.0, https://github.com/advimman/lama) from a
TorchScript export, on the CPU. The weights are fetched once into
~/.cache/tidepool. `fast` is OpenCV's Telea method: no download, blurrier.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
METHODS = ("lama", "fast", "none")


def cache_dir() -> Path:
    return Path(os.environ.get("TIDEPOOL_CACHE", Path.home() / ".cache" / "tidepool"))


def weights() -> Path:
    path = cache_dir() / "big-lama.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching LaMa weights (206 MB) to {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(LAMA_URL, tmp)
        tmp.rename(path)
    return path


def fill(img: Image.Image, mask: np.ndarray, method: str = "lama", grow: int = 12) -> Image.Image:
    """`mask` is a bool array the size of the image; True is what to remove.

    The mask is grown by `grow` pixels so the object's soft edge and shadow go too.
    """
    if method not in METHODS:
        raise ValueError(f"inpaint must be one of {', '.join(METHODS)}, not {method!r}")
    if method == "none" or not mask.any():
        return img
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    m = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    if method == "fast":
        return Image.fromarray(cv2.inpaint(rgb, m, 7, cv2.INPAINT_TELEA))
    return Image.fromarray(_lama(rgb, m))


def _lama(rgb: np.ndarray, m: np.ndarray, margin: int = 160) -> np.ndarray:
    """Runs on a window around the hole: the whole photo took a minute on the CPU."""
    import torch

    model = torch.jit.load(str(weights()), map_location="cpu").eval()
    ys, xs = np.nonzero(m)
    H, W = m.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin + 1)
    crop, hole = np.array(rgb[y0:y1, x0:x1]), m[y0:y1, x0:x1]
    h, w = hole.shape
    ph, pw = (-h) % 8, (-w) % 8
    x = torch.from_numpy(crop).permute(2, 0, 1).float()[None] / 255
    k = torch.from_numpy((hole > 0).astype(np.float32))[None, None]
    x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    k = torch.nn.functional.pad(k, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        y = model(x, k)[0]
    filled = (y.permute(1, 2, 0).clamp(0, 1).numpy()[:h, :w] * 255).round().astype(np.uint8)
    out = np.array(rgb)
    out[y0:y1, x0:x1] = np.where(hole[..., None] > 0, filled, crop)
    return out
