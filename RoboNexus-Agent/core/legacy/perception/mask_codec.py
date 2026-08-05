"""Legacy portable PNG mask/image codec used by perception transports."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def encode_rgb(rgb: np.ndarray) -> str:
    return _encode_png(rgb.astype(np.uint8), "RGB")


def decode_rgb(payload: str) -> np.ndarray:
    return np.asarray(_decode_png(payload).convert("RGB"), dtype=np.uint8)


def encode_mask(mask: np.ndarray) -> str:
    return _encode_png((mask > 0).astype(np.uint8) * 255, "L")


def decode_mask(payload: str) -> np.ndarray:
    return (np.asarray(_decode_png(payload).convert("L")) > 0).astype(np.uint8)


def _encode_png(array: np.ndarray, mode: str) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_png(payload: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload, validate=True)))
